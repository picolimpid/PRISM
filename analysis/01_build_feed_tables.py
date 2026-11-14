"""Build tidy parquet tables from the Zenodo Bluesky feed files.
Inputs (data/bluesky_zenodo/x): feed_posts/*.jsonl.gz, feed_posts_likes/*.csv.gz, ../feed_bookmarks.csv
Outputs (data/derived): feed_likes.parquet (feed-generator likes = the explicit signal),
 post_likes.parquet (likes on feed-surfaced posts), feed_posts.parquet.
Timestamps in the release are coarsened to the minute (YYYYMMDDHHMM)."""
import gzip, json, glob, os
import pandas as pd
R='data/bluesky_zenodo'; O='data/derived'
def ts(s): return pd.to_datetime(s.astype(str), format='%Y%m%d%H%M', errors='coerce')
b=pd.read_csv(f'{R}/feed_bookmarks.csv',names=['feed','user','ts'])
b['ts']=ts(b.ts); b=b.dropna().sort_values('ts').drop_duplicates(['feed','user'])
b.to_parquet(f'{O}/feed_likes.parquet'); print('feed likes',len(b),b.user.nunique())
L=[]
for f in sorted(glob.glob(f'{R}/x/feed_posts_likes/*.csv.gz')):
    d=pd.read_csv(f,names=['liker','author','post','ts']); d['feed']=os.path.basename(f)[:-7]; L.append(d)
L=pd.concat(L); L['ts']=ts(L.ts); n0=len(L); L=L.dropna().drop_duplicates(['feed','liker','post'])
L.to_parquet(f'{O}/post_likes.parquet'); print('post likes',n0,len(L),L.liker.nunique())
P=[]
for f in sorted(glob.glob(f'{R}/x/feed_posts/*.jsonl.gz')):
    feed=os.path.basename(f)[:-9]; rows=[]
    with gzip.open(f,'rt') as fh:
        for l in fh:
            d=json.loads(l)
            rows.append((feed,d['post_id'],d['user_id'],d['date'],d.get('like_count'),d.get('reply_count'),d.get('repost_count'),d.get('reply_to') is not None,(d.get('langs') or [None])[0],d.get('text') or ''))
    P.append(pd.DataFrame(rows,columns=['feed','post','author','date','like_count','reply_count','repost_count','is_reply','lang','text']))
P=pd.concat(P); P['ts']=ts(P.date); n0=len(P); P=P.drop_duplicates(['feed','post'])
P.to_parquet(f'{O}/feed_posts.parquet'); print('feed posts',n0,len(P))
print(P.groupby('feed').agg(posts=('post','size'),authors=('author','nunique'),first=('ts','min'),last=('ts','max')))
print(L.groupby('feed').agg(likes=('post','size'),likers=('liker','nunique'),posts=('post','nunique'),first=('ts','min'),last=('ts','max')))
print(b.groupby('feed').agg(n=('user','size'),first=('ts','min'),last=('ts','max')))
