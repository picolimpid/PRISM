"""RQ7: does a user's own writing move toward the Science feed after endorsing it, or was it already moving?
Outcome: weekly mean cosine between the user's English posts (bge-small) and the centroid of English Science-feed posts.
Treated: Science endorsers. Controls (active comparator): users who endorsed a different feed in the same calendar week and never
endorsed Science. Controls are reweighted within cohort week; no further matching (stated limitation).
Event-time profile of the treated-control difference with user-bootstrap CIs; DiD vs weeks -8..-2."""
import numpy as np, pandas as pd
O='data/derived'; rng=np.random.default_rng(4); H=8
T=pd.read_parquet(f'{O}/rq7_texts.parquet'); E=np.load(f'{O}/rq7_emb.npy').astype(np.float32)
P=pd.read_parquet(f'{O}/rank_posts.parquet'); PE=np.load(f'{O}/rank_posts_emb.npy').astype(np.float32)
cen=PE[((P.feed=='Science')&(P.lang=='eng')).values].mean(0); cen/=np.linalg.norm(cen)
T['sim']=E@cen; UW=T.groupby(['user','w']).sim.mean()
B=pd.read_parquet(f'{O}/feed_likes.parquet'); EP=pd.Timestamp('2023-01-01'); B['w']=((B.ts-EP).dt.days//7).astype(int)
sci=B[B.feed=='Science'].set_index('user').w; oth=B[(B.feed!='Science')&~B.user.isin(sci.index)].groupby('user').w.min()
def prof(u,c): return np.array([UW.get((u,c+k),np.nan) for k in range(-H,H+1)])
rows=[]
for c in range(22+H,62-H+1):
    t=sci.index[sci.values==c]; k=oth.index[oth.values==c]
    if len(t)==0 or len(k)<5: continue
    tp=np.array([prof(u,c) for u in t]); kp=np.nanmean(np.array([prof(u,c) for u in k]),0)
    for r in tp: rows.append(np.concatenate([r,kp]))
A=np.array(rows); tr=A[:,:2*H+1]; ct=A[:,2*H+1:]; d=tr-ct; n=len(d); print('treated users',n)
bi=rng.integers(0,n,(1000,n)); out=[]
for j,k in enumerate(range(-H,H+1)):
    b=np.nanmean(d[bi][:,:,j],1); out.append(dict(k=k,treated=np.nanmean(tr[:,j]),control=np.nanmean(ct[:,j]),diff=np.nanmean(d[:,j]),lo=np.percentile(b,2.5),hi=np.percentile(b,97.5),n_posting=int(np.isfinite(tr[:,j]).sum())))
R=pd.DataFrame(out); R.to_csv('results/rq7_writing_event.csv',index=False); print(R.round(4).to_string(index=False))
m=lambda a,b: np.nanmean(d[:,a+H:b+H+1],1)
for nm,v in [('pre gap (-8..-2)',m(-8,-2)),('pre drift (-4..-1) - (-8..-5)',m(-4,-1)-m(-8,-5)),('DiD post(2..8) - pre(-8..-2)',m(2,8)-m(-8,-2)),('DiD late(5..8) - pre(-8..-2)',m(5,8)-m(-8,-2))]:
    v=v[np.isfinite(v)]; b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(2000)]
    print(f'{nm}: {v.mean():+.4f} [{np.percentile(b,2.5):+.4f},{np.percentile(b,97.5):+.4f}] n={len(v)}  (SD of user-week sim = {T.sim.std():.3f})')
