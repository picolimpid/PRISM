"""Advisor-panel follow-ups on the event study (PLAN.md 6.7):
 A2  likes split by whether the liker follows the post's author (snapshot graph), not-yet-treated and never-endorser controls;
 A5  heterogeneity by prior engagement (weeks with a like in -8..-1) and by time since first visible trace;
 A7  covariate balance (standardized mean differences) before and after matching for the full and nyt designs.
usage: python3 analysis/23_a2_a5_a7.py <feed> [n_boot=1000]"""
import sys, time, numpy as np, pandas as pd
from rq3lib import *
f=sys.argv[1]; NB=int(sys.argv[2]) if len(sys.argv)>2 else 1000; F=Feed(f); tag=f.strip('#'); t0=time.time(); stats=[]; weekly=[]
def report(E,name,**meta):
    e=E[name]
    for s,v in e['point'].items(): stats.append(summarize(v,e['bs'][s],stat=s,n=e['n'],n_ctrl=e['n_ctrl'],**meta))
    for j,k in enumerate(range(-H,H+1)):
        lo,hi=np.percentile(e['bwk'][:,j],[2.5,97.5]); weekly.append(dict(**meta,k=k,n=e['n'],treated=e['tmean'][j],control=e['cmean'][j],diff=e['wk'][j],lo=lo,hi=hi))
R={d:match(F,d,0) for d in ['full','nyt','nytlag5']}
# ---- A2
for d in ['full','nyt','nytlag5']:
    for o in ['like_followed','like_notfollowed']:
        E=estimate({'r':build(F,R[d],o)},NB,seed=1); report(E,'r',feed=f,analysis='A2',design=d,stratum='all',outcome=o)
    print('A2',d,f'[{time.time()-t0:.0f}s]',flush=True)
# ---- A5
def strata(Rd):
    u,a=Rd['u'],Rd['a']; lw=np.array([F.nz[i,max(x-8,0):x].sum() for i,x in zip(u,a)]); ten=a-first_seen[u]
    return {'prior like-weeks: 0':lw==0,'prior like-weeks: 1-3':(lw>=1)&(lw<=3),'prior like-weeks: 4-8':lw>=4,
            'first trace <=4 wk ago':ten<=4,'first trace 5-12 wk ago':(ten>=5)&(ten<=12),'first trace >12 wk ago':ten>12}
for d in ['full','nyt']:
    for nm,m in strata(R[d]).items():
        if m.sum()<30: continue
        E=estimate({'r':build(F,subset(R[d],R[d]['u'][m]),'like_any')},NB,seed=1); report(E,'r',feed=f,analysis='A5',design=d,stratum=nm,outcome='like_any')
    print('A5',d,f'[{time.time()-t0:.0f}s]',flush=True)
pd.DataFrame(stats).to_csv(f'results/a2a5_stats_{tag}.csv',index=False); pd.DataFrame(weekly).to_csv(f'results/a2a5_weekly_{tag}.csv',index=False)
# ---- A7
def cov(idx,a):
    idx=np.asarray(idx); g=lambda C,x,y: (C[idx,np.maximum(a-y,0)]-C[idx,np.maximum(a-x,0)]).astype(float)   # weeks a-x .. a-y-1
    fp=first_post[idx]; return np.column_stack([np.log1p(g(C_act,4,0)),np.log1p(act[idx,a-1]),np.where(fp<a,a-fp,0),(fp>=a).astype(float),np.clip(a-first_seen[idx],0,80),
        np.log1p(F.LK[idx,a-1]),np.log1p(g(F.C_lk,4,1)),np.log1p(g(F.C_lk,8,4)),np.log1p(g(F.C_lk,16,8)),g(F.C_nz,8,0),g(F.C_nz,16,0),np.log1p(g(F.oth,4,0)),(C_all[idx,a]>0).astype(float),(engbin[idx]==1).astype(float)])
NAMES=['log activity, weeks -4..-1','log activity, week -1','weeks since first post','no post yet','weeks since first trace','log target likes, week -1','log target likes, weeks -4..-2',
       'log target likes, weeks -8..-5','log target likes, weeks -16..-9','weeks with a target like, last 8','weeks with a target like, last 16','log other-feed likes, weeks -4..-1','any archived like before','mostly English']
rng=np.random.default_rng(7); rows=[]
for d in ['full','nyt']:
    Rd=R[d]; u,a,pk=Rd['u'],Rd['a'],Rd['picks']; Xt=cov(u,a); v=pk>=0
    Xc=np.zeros_like(Xt)
    for j in range(pk.shape[1]):
        m=v[:,j]; Xc[m]+=cov(pk[m,j],a[m])
    Xc/=v.sum(1)[:,None]                                   # mean of each treated user's matched controls
    Xall=cov(pk[v],np.repeat(a,v.sum(1)))                  # all matched controls (for the variance)
    pool=[]                                                # unmatched comparison: random sample of the eligible control pool in each cohort week
    for c in np.unique(a):
        cm=((~F.isend) if d=='full' else (F.isend&(F.endw>=c+NYT_GAP)))&(first_seen<=c-1); ii=np.flatnonzero(cm); n_t=int((a==c).sum())
        if len(ii)==0: continue
        ii=rng.choice(ii,min(len(ii),max(200,20*n_t)),replace=False); pool.append((cov(ii,np.full(len(ii),c)),np.full(len(ii),n_t/len(ii))))
    Xp=np.vstack([p[0] for p in pool]); wp=np.concatenate([p[1] for p in pool])     # weights reproduce the treated cohort mix
    mp=np.average(Xp,0,weights=wp); vp=np.average((Xp-mp)**2,0,weights=wp)
    for j,nm in enumerate(NAMES):
        sd_b=np.sqrt((Xt[:,j].var()+vp[j])/2); sd_a=np.sqrt((Xt[:,j].var()+Xall[:,j].var())/2)
        rows.append(dict(feed=f,design=d,covariate=nm,treated_mean=Xt[:,j].mean(),pool_mean=mp[j],matched_mean=Xc[:,j].mean(),
                         smd_before=(Xt[:,j].mean()-mp[j])/sd_b if sd_b>0 else 0.0,smd_after=(Xt[:,j].mean()-Xc[:,j].mean())/sd_a if sd_a>0 else 0.0))
A7=pd.DataFrame(rows); A7.to_csv(f'results/a7_balance_{tag}.csv',index=False)
pd.set_option('display.width',220); print(A7.round(3).to_string(index=False)); print('done',f'{time.time()-t0:.0f}s')
