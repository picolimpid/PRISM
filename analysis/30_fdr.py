"""Multiple-comparison pass over every test reported in REPORT.md (PLAN.md 6.4).
p-values are two-sided normal approximations from the bootstrap standard error (or from the CI width when only the CI was
stored; ratios are tested on the log scale against 1). Benjamini-Hochberg at q=0.05 within each family and over everything.
Outputs: results/fdr_all.csv (one row per test) and results/fdr_summary.csv (per family)."""
import os, re, glob, numpy as np, pandas as pd
from scipy.stats import norm
rows=[]
def add(family,label,est,se=None,lo=None,hi=None,log=False,p=None):
    if p is None:
        if log:
            if not (est>0 and lo>0 and hi>0): return
            est_,lo_,hi_=np.log(est),np.log(lo),np.log(hi)
        else: est_,lo_,hi_=est,lo,hi
        if se is None or not np.isfinite(se):
            if lo is None or not np.isfinite(lo_) or not np.isfinite(hi_): return
            se=(hi_-lo_)/3.92
        if not np.isfinite(se) or se<=0 or not np.isfinite(est_): return
        p=2*norm.sf(abs(est_)/se)
    if np.isfinite(p): rows.append(dict(family=family,label=label,est=est,p=float(p)))
T=['Science','News','UkrainianView']
# first round: matched event study and its placebo (all outcomes, three key statistics)
for fn,fam in [('results/rq3_summary.csv','RQ3 round 1'),('results/rq3_summary_placebo16.csv','RQ3 round 1 placebo(16)')]:
    S=pd.read_csv(fn); S=S[S.spec.isin(['early','full','active'])&S.stat.isin(['pre_drift(-4..-1 minus -8..-5)','DiD post(2..8)-pre(-8..-2)','DiD late(5..8)-pre(-8..-2)'])]
    for r in S.itertuples(): add(fam,f'{r.feed}|{r.spec}|{r.outcome}|{r.stat}',r.est,lo=r.lo,hi=r.hi)
# round 2: designs, same-sample placebo, granularity
for t in T:
    for fn,fam in [(f'results/rq3r_stats_{t}.csv','RQ3 round 2 designs'),(f'results/rq3r_posthoc_stats_{t}.csv','RQ3 round 2 post hoc')]:
        S=pd.read_csv(fn); S=S[~S.stat.isin(['pre(-8..-2)','wk-1'])]
        for r in S.itertuples(): add(fam,f"{r.feed}|{r.design}|K{getattr(r,'K',10)}|{r.sample}|S{r.S}|{r.date}|{getattr(r,'outcome','like_any')}|{r.stat}",r.est,se=r.se)
    S=pd.read_csv(f'results/rq3r_regadj_{t}.csv')
    for r in S.itertuples(): add('RQ3 round 2 outcome model',f'{r.feed}|regadj|{r.sample}|S{r.S}|{r.date}|{r.stat}',r.est,se=r.se)
    S=pd.read_csv(f'results/a2a5_stats_{t}.csv'); S=S[~S.stat.isin(['pre(-8..-2)','wk-1'])]
    for r in S.itertuples(): add('A2/A5 follow split and heterogeneity',f'{r.feed}|{r.analysis}|{r.design}|{r.stratum}|{r.outcome}|{r.stat}',r.est,se=r.se)
S=pd.read_csv('results/a2_ratio.csv')
for r in S.itertuples():
    if r.ratio_of_RR>0 and r.se_log>0: add('A2/A5 follow split and heterogeneity',f'{r.feed}|{r.design}|{r.window}|ratio of RR (not followed / followed)',np.log(r.ratio_of_RR),se=r.se_log)
# RQ2
for fn in ['results/rq2_prediction.csv','results/rq2_prediction_cold.csv']:
    S=pd.read_csv(fn)
    for r in S.itertuples():
        for k in ['dAUC_hybrid','dAUC_shuffled','dAP_hybrid','dAP_shuffled']: add('RQ2 prediction',f'{r.feed}|{r.pop}|{k}',getattr(r,k),lo=getattr(r,k+'_lo'),hi=getattr(r,k+'_hi'))
        add('RQ2 prediction',f'{r.feed}|{r.pop}|O/E',r.OE,lo=r.OE_lo,hi=r.OE_hi,log=True)
# writing (A8 supersedes the first-round RQ7 numbers)
if os.path.exists('results/a8_writing_summary.csv'):
    S=pd.read_csv('results/a8_writing_summary.csv'); S=S[S.se.notna()]
    for r in S.itertuples(): add('RQ7/A8 writing',f'{r.feed}|{r.anchor}|shift{r.shift}|{r.stat}',r.est,se=r.se)
# Line B paired comparisons
for l in open('results/rankB_paired.txt'):
    m=re.match(r'(.*): ([+-][0-9.]+) \[([+-][0-9.]+),([+-][0-9.]+)\]',l.strip())
    if m: add('Line B ranking',m.group(1),float(m.group(2)),lo=float(m.group(3)),hi=float(m.group(4)))
# A1 timing ratios, A6 regression
S=pd.read_csv('results/a1_timing_summary.csv',index_col=0)
for f in S.columns:
    for k in ['rate_ratio_10min','rate_ratio_1h','rate_ratio_24h','other_feeds_rate_ratio_1h']: add('A1 timing',f'{f}|{k}',S.loc[k,f],lo=S.loc[k+'_lo',f],hi=S.loc[k+'_hi',f],log=True)
S=pd.read_csv('results/a6_who_engages.csv')
for r in S.itertuples(): add('A6 regression',f'{r.feed}|{r.model}|{r.term}',r.OR,p=r.p)
D=pd.DataFrame(rows)
def bh(p):
    p=np.asarray(p); n=len(p); o=np.argsort(p); q=np.empty(n); q[o]=np.minimum.accumulate((p[o]*n/np.arange(1,n+1))[::-1])[::-1]; return np.minimum(q,1)
D['q_family']=D.groupby('family').p.transform(bh); D['q_global']=bh(D.p.values)
D.to_csv('results/fdr_all.csv',index=False)
G=D.groupby('family').agg(tests=('p','size'),p_lt_05=('p',lambda x:int((x<.05).sum())),q_family_lt_05=('q_family',lambda x:int((x<.05).sum())),q_global_lt_05=('q_global',lambda x:int((x<.05).sum())))
G.loc['ALL']=[len(D),int((D.p<.05).sum()),int((D.q_family<.05).sum()),int((D.q_global<.05).sum())]; G.to_csv('results/fdr_summary.csv')
pd.set_option('display.width',250); pd.set_option('display.max_rows',400); pd.set_option('display.max_colwidth',140); print(G.to_string())
L=D[(D.p<.05)&((D.q_family>=.05)|(D.q_global>=.05))].sort_values('p'); print('\nsignificant before, not after correction:',len(L)); print(L.round(4).to_string(index=False))
