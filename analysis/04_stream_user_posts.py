"""One streaming pass over user_posts.tar.gz (235M posts, one jsonl per user) without unpacking it.
  pigz -dc user_posts.tar.gz | python3 analysis/04_stream_user_posts.py
Outputs (data/derived):
  user_day.parquet     user, day, n_post, n_reply, n_repost, n_quote, n_eng   (all 4M users)
  feed_interactions.parquet  replies / reposts / quotes whose target post is a feed-surfaced post
  texts/shard_XX.jsonl.gz    authored text (no reposts) for endorsers + a fixed random sample of feed-post likers
"""
import sys, tarfile, json, gzip, os, random
import numpy as np, pandas as pd
from multiprocessing import Pool
O='data/derived'
B=pd.read_parquet(f'{O}/feed_likes.parquet'); L=pd.read_parquet(f'{O}/post_likes.parquet'); P=pd.read_parquet(f'{O}/feed_posts.parquet',columns=['post'])
FEEDPOSTS=frozenset(P.post.tolist())
rng=np.random.default_rng(0)
likers=np.array(sorted(set(L[L.feed.isin(['Science','Blacksky','News','#UkrainianView'])].liker)-set(B.user)))
KEEP=frozenset(B.user.tolist())|frozenset(rng.choice(likers,80000,replace=False).tolist())
pd.Series(sorted(KEEP)).to_frame('user').to_parquet(f'{O}/text_users.parquet')
EPOCH=pd.Timestamp('2023-01-01').toordinal()
import datetime
def day(d):
    s=str(d); 
    try: return datetime.date(int(s[:4]),int(s[4:6]),int(s[6:8])).toordinal()-EPOCH
    except Exception: return -1
def work(batch):
    days={}; inter=[]; texts=[]
    for raw in batch:
        for line in raw.split(b'\n'):
            if not line: continue
            try: d=json.loads(line)
            except Exception: continue
            u=d['user_id']; dy=day(d['date'])
            k=(u,dy); c=days.get(k)
            if c is None: c=days[k]=[0,0,0,0,0]
            rp=d.get('repost_from'); rt=d.get('reply_to'); q=d.get('quotes')
            if rp is not None: c[2]+=1; kind='repost'; tgt=rp
            elif rt is not None: c[1]+=1; kind='reply'; tgt=rt
            elif q is not None: c[3]+=1; kind='quote'; tgt=q
            else: c[0]+=1; kind='post'; tgt=None
            lg=d.get('langs') or []
            if 'eng' in lg: c[4]+=1
            if tgt is not None and tgt in FEEDPOSTS: inter.append((u,d['date'],kind,tgt))
            if kind!='repost' and q is not None and q in FEEDPOSTS and kind!='quote': inter.append((u,d['date'],'quote',q))
            if u in KEEP and kind!='repost' and d.get('text'):
                texts.append(json.dumps({'post_id':d['post_id'],'user_id':u,'date':d['date'],'kind':kind,'lang':lg[:1],'text':d['text']},ensure_ascii=False))
    a=np.array([(u,dy,*c) for (u,dy),c in days.items()],dtype=np.int32).reshape(-1,7)
    return a,inter,texts
def batches(t,nbytes=24_000_000):
    cur=[]; sz=0
    for m in t:
        if not m.isfile(): continue
        cur.append(t.extractfile(m).read()); sz+=m.size
        if sz>=nbytes: yield cur; cur=[]; sz=0
    if cur: yield cur
if __name__=='__main__':
    os.makedirs(f'{O}/texts',exist_ok=True)
    t=tarfile.open(fileobj=sys.stdin.buffer,mode='r|')
    A=[]; I=[]; nt=0; nb=0
    outs=[gzip.open(f'{O}/texts/shard_{i:02d}.jsonl.gz','wt',compresslevel=3) for i in range(16)]
    with Pool(40) as pool:
        for a,inter,texts in pool.imap_unordered(work,batches(t),chunksize=1):
            A.append(a); I.extend(inter); nb+=1
            for s in texts: outs[nb%16].write(s+'\n')
            nt+=len(texts)
            if nb%200==0: print(nb,'batches',sum(len(x) for x in A),'user-days',len(I),'feed interactions',nt,'texts',flush=True)
    for o in outs: o.close()
    A=np.concatenate(A); D=pd.DataFrame(A,columns=['user','day','n_post','n_reply','n_repost','n_quote','n_eng'])
    D=D.groupby(['user','day'],as_index=False).sum()
    D.to_parquet(f'{O}/user_day.parquet'); print('user_day',len(D),D.user.nunique(),D[['n_post','n_reply','n_repost','n_quote']].sum().to_dict())
    pd.DataFrame(I,columns=['user','date','kind','post']).to_parquet(f'{O}/feed_interactions.parquet'); print('feed interactions',len(I),'texts',nt,'DONE')
