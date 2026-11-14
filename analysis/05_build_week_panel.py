"""User x calendar-week panel (weeks counted from 2023-01-01) for all users in user_posts.
Saves data/derived/panel.npz: users, act[u,w] (posts+replies+reposts+quotes), first_week, eng_major,
and per-feed like / interaction week matrices for the users that have any (sparse rows -> dense small arrays)."""
import numpy as np, pandas as pd, scipy.sparse as sp
O='data/derived'; FEEDS=['Science','Blacksky','News','#UkrainianView']
D=pd.read_parquet(f'{O}/user_day.parquet'); D=D[(D.day>=0)&(D.day<7*64)]
EP=pd.Timestamp('2023-01-01'); NW=64
L=pd.read_parquet(f'{O}/post_likes.parquet'); L=L[(L.ts>=EP)&(L.ts<EP+pd.Timedelta(weeks=NW))]
BK=pd.read_parquet(f'{O}/feed_likes.parquet')
posters=np.sort(D.user.unique())
# universe = everyone who posted, liked an archived feed post, or endorsed a feed (non-posters get all-zero activity rows)
users=np.union1d(np.union1d(posters,L.liker.unique()),BK.user.unique()); uix=pd.Series(np.arange(len(users)),index=users)
print('posters',len(posters),'universe',len(users))
D['w']=D.day//7; D['n']=D.n_post+D.n_reply+D.n_repost+D.n_quote
r=uix.reindex(D.user).values
act=sp.coo_matrix((D.n.values,(r,D.w.values)),shape=(len(users),NW)).tocsr()
own=sp.coo_matrix(((D.n_post+D.n_reply+D.n_quote).values,(r,D.w.values)),shape=(len(users),NW)).tocsr()
g=D.groupby('user').agg(first=('w','min'),last=('w','max'),n=('n','sum'),eng=('n_eng','sum'))
g=g.reindex(users)
sp.save_npz(f'{O}/act.npz',act); sp.save_npz(f'{O}/own.npz',own)
L['w']=((L.ts-EP).dt.days//7).astype(int)
FI=pd.read_parquet(f'{O}/feed_interactions.parquet'); FP=pd.read_parquet(f'{O}/feed_posts.parquet',columns=['feed','post'])
FI=FI.merge(FP,on='post'); FI['ts']=pd.to_datetime(FI.date.astype(str),format='%Y%m%d%H%M',errors='coerce'); FI=FI.dropna()
FI=FI[(FI.ts>=EP)&(FI.ts<EP+pd.Timedelta(weeks=NW))]; FI['w']=((FI.ts-EP).dt.days//7).astype(int)
m=sp.coo_matrix((np.ones(len(L),dtype=np.int32),(uix.reindex(L.liker).values,L.w.values)),shape=(len(users),NW)).tocsr()
sp.save_npz(f'{O}/likes_ALL.npz',m)   # likes on archived posts of all 11 feeds
BK=BK[(BK.ts>=EP)&(BK.ts<EP+pd.Timedelta(weeks=NW))]; BK['w']=((BK.ts-EP).dt.days//7).astype(int)
for f in FEEDS:
    l=L[(L.feed==f)&L.liker.isin(uix.index)]
    m=sp.coo_matrix((np.ones(len(l),dtype=np.int32),(uix.reindex(l.liker).values,l.w.values)),shape=(len(users),NW)).tocsr()
    sp.save_npz(f'{O}/likes_{f.strip("#")}.npz',m)
    i=FI[(FI.feed==f)&FI.user.isin(uix.index)]
    m=sp.coo_matrix((np.ones(len(i),dtype=np.int32),(uix.reindex(i.user).values,i.w.values)),shape=(len(users),NW)).tocsr()
    sp.save_npz(f'{O}/inter_{f.strip("#")}.npz',m); print(f,'likes',len(l),'interactions',len(i))
np.savez(f'{O}/panel_meta.npz',users=users,first=g['first'].fillna(999).values,last=g['last'].fillna(-1).values,n=g.n.fillna(0).values,eng=(g.eng/g.n.clip(lower=1)).fillna(-1).values)
print('weeks with activity',np.asarray(act.sum(0)).ravel().tolist())
