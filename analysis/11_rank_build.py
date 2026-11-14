"""Line B (original PRISM idea, with feed-grounded candidates): build a ranking task on the Science feed.
User = anyone with >=40 likes on English Science-feed posts. Chronological 80/20 split of their likes.
Each evaluation like is one impression: the liked post + 19 Science-feed posts published within +-6h of it (before the like) that the user never liked
and did not write. Outputs data/derived/rank_task.pkl"""
import numpy as np, pandas as pd, pickle
rng=np.random.default_rng(3); O='data/derived'
P=pd.read_parquet(f'{O}/rank_posts.parquet'); P['row']=np.arange(len(P)); P=P[(P.feed=='Science')&(P.lang=='eng')&(P.text.str.len()>=20)]
L=pd.read_parquet(f'{O}/post_likes.parquet'); L=L[(L.feed=='Science')&(L.ts<='2024-04-25')].drop(columns='author').merge(P[['post','row','author','ts']].rename(columns={'ts':'pts'}),on='post')
L=L[L.liker!=L.author].sort_values(['liker','ts'])
B=pd.read_parquet(f'{O}/feed_likes.parquet'); end=B[B.feed=='Science'].set_index('user').ts
n=L.groupby('liker').size(); U=n[n>=40].index.values; print('eligible users',len(U),'endorsers among them',np.isin(U,end.index).sum())
U=rng.choice(U,min(len(U),1500),replace=False)
Ps=P.sort_values('ts'); pts=Ps.ts.values; prow=Ps.row.values; pauth=Ps.author.values
task={}
for u,g in L[L.liker.isin(U)].groupby('liker'):
    k=int(len(g)*0.8); hist=g.iloc[:k]; ev=g.iloc[k:]; liked=set(g.row); imps=[]
    for r in ev.itertuples():
        cand=[]
        for hw in (6,12):   # negatives posted within +-6h (else +-12h) of the liked post and before the like: removes the recency shortcut
            lo=np.searchsorted(pts,np.datetime64(r.pts-pd.Timedelta(hours=hw))); hi=np.searchsorted(pts,np.datetime64(min(r.ts,r.pts+pd.Timedelta(hours=hw))))
            cand=[prow[i] for i in range(lo,hi) if prow[i] not in liked and pauth[i]!=u]
            if len(cand)>=19: break
        if len(cand)<19: continue
        imps.append((r.row,list(rng.choice(cand,19,replace=False))))
    if len(imps)>=5: task[u]=dict(hist=hist.row.tolist(),hist_ts=hist.ts.tolist(),imps=imps[:60],endorser=u in end.index,end_ts=end.get(u,None),split_ts=ev.ts.iloc[0])
print('users',len(task),'impressions',sum(len(t['imps']) for t in task.values()),'endorsers',sum(t['endorser'] for t in task.values()))
pickle.dump(task,open(f'{O}/rank_task.pkl','wb'))
