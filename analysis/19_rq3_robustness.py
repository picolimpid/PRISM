"""Round-2 RQ3 robustness, pre-specified in PLAN.md section 6 (committed before this script was written):
 6.1 real vs placebo dates on the SAME treated users (S=13 primary; 16, 20 as sensitivity), with the paired difference;
 6.2 trajectory-based control designs T1 seq8, T2 persist16, T3 nyt (T4, the outcome model, is in 20_rq3_regadj.py);
 6.3 matching granularity: coarse / fine bins, K=1/5/10 controls, matching horizon L=1/3/5/9/13 weeks before the endorsement.
usage: python3 analysis/19_rq3_robustness.py <feed> [n_boot=1000]   -> results/rq3r_{stats,weekly,match}_<feed>.csv"""
import sys, time, numpy as np, pandas as pd
from rq3lib import *
f=sys.argv[1]; NB=int(sys.argv[2]) if len(sys.argv)>2 else 1000
F=Feed(f); tag=f.strip('#'); t0=time.time()
print(f,'startw',F.startw,'eff',F.eff,'endorsers',int(F.isend.sum()),flush=True)
stats=[]; weekly=[]; mlog=[]
def report(E,name,**meta):
    e=E[name]
    for s,v in e['point'].items(): stats.append(summarize(v,e['bs'][s],stat=s,n=e['n'],n_ctrl=e['n_ctrl'],**meta))
    if meta.get('outcome')=='like_any':
        for j,k in enumerate(range(-H,H+1)):
            lo,hi=np.percentile(e['bwk'][:,j],[2.5,97.5])
            weekly.append(dict(**meta,k=k,n=e['n'],treated=e['tmean'][j],control=e['cmean'][j],diff=e['wk'][j],lo=lo,hi=hi))
def flush():
    pd.DataFrame(stats).to_csv(f'results/rq3r_stats_{tag}.csv',index=False); pd.DataFrame(weekly).to_csv(f'results/rq3r_weekly_{tag}.csv',index=False)
    pd.DataFrame(mlog).to_csv(f'results/rq3r_match_{tag}.csv',index=False)
def run(design,K=10,shifts=(13,16,20),outcomes=('like_any','other_any','act_any')):
    R0=match(F,design,0,K); base=dict(feed=f,design=design,K=K)
    mlog.append(dict(**base,S=0,eligible=R0['n_elig'],matched=len(R0['u']),no_trace_before=R0['n_noexist'],median_pool=float(np.median(R0['npool'])) if len(R0['u']) else np.nan))
    print(f'{design} K={K} real: eligible {R0["n_elig"]} matched {len(R0["u"])}  [{time.time()-t0:.0f}s]',flush=True)
    if len(R0['u'])<20: return
    for o in outcomes:
        E=estimate({'r':build(F,R0,o)},NB,seed=1); report(E,'r',**base,sample='all',S=0,date='real',outcome=o)
    for S in shifts:
        RP=match(F,design,S,K)
        common=np.intersect1d(R0['u'],RP['u'])
        mlog.append(dict(**base,S=S,eligible=RP['n_elig'],matched=len(RP['u']),no_trace_before=RP['n_noexist'],median_pool=float(np.median(RP['npool'])) if len(RP['u']) else np.nan,common=len(common)))
        print(f'   placebo S={S}: eligible {RP["n_elig"]} matched {len(RP["u"])} common with real {len(common)}',flush=True)
        if len(common)<20: continue
        a=subset(R0,common); b=subset(RP,common); assert (a['u']==b['u']).all()
        E=estimate({'real':build(F,a,'like_any'),'plac':build(F,b,'like_any')},NB,seed=2)
        m=dict(**base,sample='same',S=S,outcome='like_any')
        report(E,'real',**m,date='real'); report(E,'plac',**m,date='placebo')
        for s in E['real']['point']:
            stats.append(summarize(E['real']['point'][s]-E['plac']['point'][s],E['real']['bs'][s]-E['plac']['bs'][s],stat=s,n=E['real']['n'],n_ctrl=np.nan,**m,date='real-placebo'))
    flush()
for d in ['full','active','seq8','persist16','nyt']: run(d)                 # 6.1 + 6.2
for d in ['coarse','fine']: run(d,shifts=(13,),outcomes=('like_any',))      # 6.3 bins
for K in [1,5]: run('full',K=K,shifts=(13,),outcomes=('like_any',))         # 6.3 number of controls
for L in [1,3,5,9,13]: run(f'lag{L}',shifts=(),outcomes=('like_any',))      # 6.3 matching horizon
flush(); print('done',f'{time.time()-t0:.0f}s')
