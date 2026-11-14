"""A2 step 1: flag each like on a feed-archived post by whether the liker follows the post's author.
followers.csv.gz rows are "u,v" = u follows v (Failla & Rossetti 2024); it is a snapshot without timestamps, so a follow
created AFTER the like still counts as "follows". Writes post_likes_follow.parquet and per-feed user x week matrices
likesF_<feed>.npz (author followed) and likesN_<feed>.npz (author not followed)."""
import numpy as np, pandas as pd, scipy.sparse as sp
O='data/derived'; R='data/bluesky_zenodo'; EP=pd.Timestamp('2023-01-01'); NW=64; BASE=10_000_000
L=pd.read_parquet(f'{O}/post_likes.parquet'); assert L.liker.max()<BASE and L.author.max()<BASE
key=L.liker.values.astype(np.int64)*BASE+L.author.values.astype(np.int64); uk=np.unique(key); hit=np.zeros(len(uk),bool); n=0
for ch in pd.read_csv(f'{R}/followers.csv.gz',names=['u','v'],dtype=np.int64,chunksize=20_000_000):
    k=ch.u.values*BASE+ch.v.values; pos=np.searchsorted(uk,k); pos[pos>=len(uk)]=0; m=uk[pos]==k; hit[pos[m]]=True; n+=len(ch); print(n,int(hit.sum()),flush=True)
L['follows']=hit[np.searchsorted(uk,key)]; L['self']=L.liker.values==L.author.values
L.to_parquet(f'{O}/post_likes_follow.parquet'); print('edges',n,'likes',len(L),'share on followed authors',L.follows.mean().round(4),'self-likes',L.self.mean().round(4))
print(L.groupby('feed').follows.mean().round(3).to_string())
users=np.load(f'{O}/panel_meta.npz')['users']; uix=pd.Series(np.arange(len(users)),index=users)
L=L[(L.ts>=EP)&(L.ts<EP+pd.Timedelta(weeks=NW))]; L['w']=((L.ts-EP).dt.days//7).astype(int)
for f in ['Science','News','#UkrainianView']:
    for nm,m in [('F',L.follows.values),('N',~L.follows.values)]:
        l=L[(L.feed==f).values&m&L.liker.isin(uix.index).values]
        M=sp.coo_matrix((np.ones(len(l),dtype=np.int32),(uix.reindex(l.liker).values,l.w.values)),shape=(len(users),NW)).tocsr(); sp.save_npz(f'{O}/likes{nm}_{f.strip("#")}.npz',M); print(f,nm,len(l))
