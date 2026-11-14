"""RQ5: what do endorsers do in the 8 weeks after liking a feed generator? Mutually exclusive types, per feed.
Also: share of endorsers with no trace of prior existence (endorse on arrival), and feed authorship (contributors)."""
import numpy as np, pandas as pd, scipy.sparse as sp
O='data/derived'; FEEDS=['Science','News','#UkrainianView']; H=8; LASTW=62
M=np.load(f'{O}/panel_meta.npz'); users=M['users']; first_post=M['first'].astype(int); N=len(users); uix=pd.Series(np.arange(N),index=users)
dense=lambda n: np.asarray(sp.load_npz(f'{O}/{n}.npz').todense(),dtype=np.int32)
act=dense('act'); own=dense('own'); la=dense('likes_ALL')
B=pd.read_parquet(f'{O}/feed_likes.parquet'); EP=pd.Timestamp('2023-01-01'); B['w']=((B.ts-EP).dt.days//7).astype(int)
P=pd.read_parquet(f'{O}/feed_posts.parquet',columns=['feed','author','ts']); P['w']=((P.ts-EP).dt.days//7).astype(int)
rows=[]
for f in FEEDS:
    lk=dense('likes_'+f.strip('#')); it=dense('inter_'+f.strip('#')); startw=int(np.argmax(lk.sum(0)>0))
    b=B[(B.feed==f)&(B.w>=startw+1)&(B.w<=LASTW-H)]; u=uix.reindex(b.user).values.astype(int); c=b.w.values
    sl=lambda A: np.array([A[i,w+1:w+H+1].sum() for i,w in zip(u,c)]); pre=lambda A: np.array([A[i,:w].sum() for i,w in zip(u,c)])
    L8,I8,A8,O8=sl(lk),sl(it),sl(act),sl(la-lk)
    typ=np.select([L8>0,I8>0,(A8>0)|(O8>0)],['likes feed posts','replies/reposts feed posts only','active elsewhere, no feed engagement'],'no observable activity')
    auth=set(P[P.feed==f].author); newc=(pre(act)==0)&(pre(la)==0)
    r=dict(feed=f,n=len(u)); r.update({k:100*v for k,v in pd.Series(typ).value_counts(normalize=True).items()})
    r['endorsed on arrival (no prior trace) %']=100*newc.mean(); r['ever authored a post in the feed %']=100*np.mean([x in auth for x in b.user])
    r['median posts in 8w (active-elsewhere type)']=np.median(A8[typ=='active elsewhere, no feed engagement']) if (typ=='active elsewhere, no feed engagement').any() else np.nan
    r['likes-feed type: median target likes 8w']=np.median(L8[L8>0])
    # weeks with any target like out of 8 among likers
    wk=np.array([(lk[i,w+1:w+H+1]>0).sum() for i,w in zip(u,c)]); r['likers: mean weeks (of 8) with a like']=wk[wk>0].mean(); r['liked in >=4 of 8 weeks %']=100*(wk>=4).mean()
    rows.append(r)
T=pd.DataFrame(rows).set_index('feed').T; pd.set_option('display.width',200); print(T.round(1).to_string()); T.to_csv('results/rq5_endorser_types.csv')
