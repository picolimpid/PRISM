"""Post hoc checks recorded in PLAN.md 6.6 (written after seeing 19/20, before running this):
 A. survivorship in the placebo: never-endorser controls restricted to accounts with any trace at week >= a+S;
 B. not-yet-treated controls with a longer gap (13/17/21 weeks) and with treated restricted to accounts with a trace at week >= c+gap;
 C. pre-endorsement ramp relative to not-yet-treated controls matched only up to week -5 / -9.
usage: python3 analysis/21_rq3_posthoc.py <feed> [n_boot=1000]"""
import sys, time, numpy as np, pandas as pd
from rq3lib import *
f=sys.argv[1]; NB=int(sys.argv[2]) if len(sys.argv)>2 else 1000; F=Feed(f); tag=f.strip('#'); t0=time.time(); stats=[]; weekly=[]
def report(E,name,**meta):
    e=E[name]
    for s,v in e['point'].items(): stats.append(summarize(v,e['bs'][s],stat=s,n=e['n'],n_ctrl=e['n_ctrl'],**meta))
    for j,k in enumerate(range(-H,H+1)):
        lo,hi=np.percentile(e['bwk'][:,j],[2.5,97.5]); weekly.append(dict(**meta,k=k,n=e['n'],treated=e['tmean'][j],control=e['cmean'][j],diff=e['wk'][j],lo=lo,hi=hi))
def flush():
    pd.DataFrame(stats).to_csv(f'results/rq3r_posthoc_stats_{tag}.csv',index=False); pd.DataFrame(weekly).to_csv(f'results/rq3r_posthoc_weekly_{tag}.csv',index=False)
# A
for design in ['full','seq8']:
    R0=match(F,design,0)
    for S in (13,16):
        RP=match(F,design,S,ctrl_alive=S); common=np.intersect1d(R0['u'],RP['u'])
        print(f'A {design} S={S} placebo with surviving controls: matched {len(RP["u"])} common {len(common)} [{time.time()-t0:.0f}s]',flush=True)
        if len(common)<20: continue
        E=estimate({'real':build(F,subset(R0,common),'like_any'),'plac':build(F,subset(RP,common),'like_any')},NB,seed=2)
        m=dict(feed=f,design=design+'+ctrl_alive',sample='same',S=S)
        report(E,'real',**m,date='real'); report(E,'plac',**m,date='placebo')
        for s in E['real']['point']: stats.append(summarize(E['real']['point'][s]-E['plac']['point'][s],E['real']['bs'][s]-E['plac']['bs'][s],stat=s,n=E['real']['n'],n_ctrl=np.nan,**m,date='real-placebo'))
    flush()
# B
for gap in (13,17,21):
    for ta in (None,gap):
        R=match(F,'nyt',0,gap=gap,treat_alive=ta); name=f'nyt gap{gap}'+('+treated_alive' if ta else '')
        print(f'B {name}: matched {len(R["u"])} [{time.time()-t0:.0f}s]',flush=True)
        if len(R['u'])<20: continue
        E=estimate({'r':build(F,R,'like_any')},NB,seed=1); report(E,'r',feed=f,design=name,sample='all',S=0,date='real')
    flush()
# C
for L in (5,9):
    for ta in (None,13):
        R=match(F,f'nytlag{L}',0,treat_alive=ta); name=f'nytlag{L}'+('+treated_alive' if ta else '')
        print(f'C {name}: matched {len(R["u"])} [{time.time()-t0:.0f}s]',flush=True)
        if len(R['u'])<20: continue
        E=estimate({'r':build(F,R,'like_any')},NB,seed=1); report(E,'r',feed=f,design=name,sample='all',S=0,date='real')
    flush()
print('done',f'{time.time()-t0:.0f}s')
