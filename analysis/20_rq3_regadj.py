"""Round-2 design T4 (PLAN.md 6.2): outcome-model adjustment instead of matching.
A gradient-boosting model is trained on NEVER-endorsers to predict, from the previous 16 weeks of week-by-week history,
whether a user likes a post of the feed in week a+k (k=0..8). For endorsers we report observed - predicted at the real
endorsement week and, on the same users, at placebo dates S weeks earlier. 5-fold cross-fitting by user: out-of-fold
predictions give the calibration check on held-out never-endorsers; endorsers get the mean of the 5 fold models.
The bootstrap is over endorsers only (the outcome model is treated as fixed; stated in the report).
usage: python3 analysis/20_rq3_regadj.py <feed> [n_boot=1000]"""
import sys, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as GB
from sklearn.model_selection import GroupKFold
from rq3lib import *
f=sys.argv[1]; NB=int(sys.argv[2]) if len(sys.argv)>2 else 1000; tag=f.strip('#'); t0=time.time()
F=Feed(f); rng=np.random.default_rng(20); SHIFTS=[13,16,20]; KS=list(range(0,9)); NREST=15000
C_oth=F.oth
def feats(idx,a):
    col=lambda Mx,j,cap: np.minimum(Mx[idx,a-j],cap) if a-j>=0 else np.zeros(len(idx),dtype=np.int32)
    lk=[col(F.LK,j,20) for j in range(1,17)]
    ot=[np.minimum(likes_all[idx,a-j]-F.LK[idx,a-j],20) for j in range(1,9)]
    ac=[np.log1p(act[idx,a-j]) for j in range(1,9)]
    extra=[W(F.C_nz,a-16,a)[idx],np.log1p(W(F.C_lk,a-16,a)[idx]),np.log1p(F.C_lk[idx,a]),np.log1p(C_oth[idx,a]),np.log1p(C_act[idx,a]),
           np.clip(a-first_post[idx],-1,80),np.clip(a-first_seen[idx],0,80),engbin[idx],np.full(len(idx),a)]
    return np.column_stack(lk+ot+ac+extra).astype(np.float32)
def targets(idx,a): return np.column_stack([F.nz[idx,a+k] for k in KS]).astype(np.int8)
# ---- training rows: never-endorsers, every anchor week
Xs=[];Ys=[];Ws=[];Us=[]
for a in range(F.eff+4,LASTW-H+1):
    base=(~F.isend)&(first_seen<=a-1); hist=W(F.C_nz,a-16,a)>0
    i1=np.flatnonzero(base&hist); rest=np.flatnonzero(base&~hist); i2=rng.choice(rest,min(NREST,len(rest)),replace=False)
    idx=np.r_[i1,i2]; w=np.r_[np.ones(len(i1)),np.full(len(i2),len(rest)/len(i2))]
    Xs.append(feats(idx,a)); Ys.append(targets(idx,a)); Ws.append(w); Us.append(idx)
X=np.vstack(Xs); Yk=np.vstack(Ys); wgt=np.concatenate(Ws); grp=np.concatenate(Us); del Xs,Ys
print(f,'training rows',len(X),'users',len(np.unique(grp)),'features',X.shape[1],f'[{time.time()-t0:.0f}s]',flush=True)
# ---- endorsers: real anchors (first round's cohort range) and placebo anchors on the same users
tr=np.flatnonzero(F.isend&(F.endw>=F.startw+H)&(F.endw<=LASTW-H)&(first_seen<=F.endw-1)); c=F.endw[tr]
def rows_at(sel,S):
    Xp=np.empty((len(sel),X.shape[1]),np.float32); Yp=np.empty((len(sel),len(KS)),np.int8)
    for a in np.unique(c[sel]-S):
        m=(c[sel]-S)==a; Xp[m]=feats(tr[sel][m],a); Yp[m]=targets(tr[sel][m],a)
    return Xp,Yp
sets={('real',0):np.arange(len(tr))}
for S in SHIFTS: sets[('placebo',S)]=np.flatnonzero((first_seen[tr]<=c-S-1)&(c-S>=F.eff+4))
TX={k:rows_at(sel,k[1]) for k,sel in sets.items()}
print('endorsers',{f'{k[0]}{k[1]}':len(v) for k,v in sets.items()},flush=True)
# ---- cross-fitted models, one per event week
oof=np.zeros(Yk.shape,np.float32); TP={k:np.zeros(TX[k][1].shape,np.float32) for k in sets}
for fold,(a_,b_) in enumerate(GroupKFold(5).split(X,Yk[:,0],grp)):
    for j,k in enumerate(KS):
        m=GB(max_iter=150,learning_rate=0.1,max_leaf_nodes=31,l2_regularization=1.0,random_state=0).fit(X[a_],Yk[a_,j],sample_weight=wgt[a_])
        oof[b_,j]=m.predict_proba(X[b_])[:,1]
        for key_ in sets: TP[key_][:,j]+=m.predict_proba(TX[key_][0])[:,1]/5
    print('fold',fold,f'[{time.time()-t0:.0f}s]',flush=True)
# ---- calibration on held-out never-endorsers (weighted), overall and by decile of predicted risk for weeks 2..8
post=[KS.index(k) for k in range(2,9)]; pm=oof[:,post].mean(1); ym=Yk[:,post].mean(1)
cal=[dict(feed=f,group='all never-endorsers',n=len(pm),obs=np.average(ym,weights=wgt),pred=np.average(pm,weights=wgt))]
nzrows=X[:,32]>0      # feature 32 = weeks with a like in the previous 16
q=np.quantile(pm[nzrows],np.linspace(0,1,11)); bins=np.clip(np.digitize(pm[nzrows],q[1:-1]),0,9)
for b in range(10):
    m=bins==b; cal.append(dict(feed=f,group=f'history>0, risk decile {b+1}',n=int(m.sum()),obs=ym[nzrows][m].mean(),pred=pm[nzrows][m].mean()))
pd.DataFrame(cal).assign(diff=lambda d:d.obs-d.pred).to_csv(f'results/rq3r_regadj_calib_{tag}.csv',index=False)
print(pd.DataFrame(cal).assign(diff=lambda d:d.obs-d.pred).round(4).to_string(index=False),flush=True)
# ---- endorsers: observed - predicted, Bayesian bootstrap over endorsers, paired real vs placebo on the same users
def resid_stats(Yp,Pp):
    r=Yp.astype(np.float32)-Pp; out={f'wk{k}':r[:,j] for j,k in enumerate(KS)}
    out['gap(2..8)']=r[:,post].mean(1); out['gap(5..8)']=r[:,[KS.index(k) for k in range(5,9)]].mean(1); return out
rows=[]
def add(v,w,**meta):
    est=float(v.mean()); bs=(w*v[None,:]).sum(1)/w.sum(1); rows.append(summarize(est,bs,**meta))
Wall=rng.exponential(size=(NB,len(tr))).astype(np.float32)
R0=resid_stats(*[TX[('real',0)][1],TP[('real',0)]])
for s,v in R0.items(): add(v,Wall,feed=f,design='regadj',sample='all',S=0,date='real',stat=s,n=len(tr),obs=float(TX[('real',0)][1][:,post].mean()),pred=float(TP[('real',0)][:,post].mean()))
for S in SHIFTS:
    sel=sets[('placebo',S)]
    if len(sel)<20: continue
    RP=resid_stats(TX[('placebo',S)][1],TP[('placebo',S)]); w=Wall[:,sel]
    for s in R0:
        m=dict(feed=f,design='regadj',sample='same',S=S,stat=s,n=len(sel))
        add(R0[s][sel],w,**m,date='real',obs=float(TX[('real',0)][1][sel][:,post].mean()),pred=float(TP[('real',0)][sel][:,post].mean()))
        add(RP[s],w,**m,date='placebo',obs=float(TX[('placebo',S)][1][:,post].mean()),pred=float(TP[('placebo',S)][:,post].mean()))
        add(R0[s][sel]-RP[s],w,**m,date='real-placebo',obs=np.nan,pred=np.nan)
D=pd.DataFrame(rows); D.to_csv(f'results/rq3r_regadj_{tag}.csv',index=False)
pd.set_option('display.width',220)
print(D[D.stat.isin(['wk0','gap(2..8)','gap(5..8)'])].round(4).to_string(index=False)); print('done',f'{time.time()-t0:.0f}s')
