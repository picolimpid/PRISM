"""A2, paired comparison: is the endorsement-week jump (and the later excess) relatively larger for likes on posts whose
author the user does NOT follow than for posts whose author the user follows? Same matched sets, same Bayesian-bootstrap
weights for both outcomes; reports rate ratios (endorsers / controls) and the ratio of the two rate ratios."""
import sys, numpy as np, pandas as pd
from rq3lib import *
rows=[]
for f in FEEDS:
    F=Feed(f)
    for d in ['nyt','full']:
        R=match(F,d,0); objs={o:build(F,R,o) for o in ['like_followed','like_notfollowed']}
        rng=np.random.default_rng(3); ids=np.unique(np.r_[R['u'],R['picks'][R['picks']>=0]]); NB=1000; Wt=rng.exponential(size=(NB,len(ids))).astype(np.float32)
        it=np.searchsorted(ids,R['u']); valid=R['picks']>=0; ic=np.searchsorted(ids,np.where(valid,R['picks'],ids[0]))
        res={}
        for o,ob in objs.items():
            Yt,Yc=ob['Yt'],ob['Yc']; cm=np.einsum('nk,nkj->nj',valid.astype(np.float32),Yc)/valid.sum(1)[:,None]
            pt={'wk0':(Yt[:,H].mean(),cm[:,H].mean()),'wk2..8':(Yt[:,IX(2,8)].mean(),cm[:,IX(2,8)].mean())}; bs={k:np.empty((NB,2)) for k in pt}
            for b in range(NB):
                wt=Wt[b,it]; wc=Wt[b,ic]*valid; cmb=np.einsum('nk,nkj->nj',wc,Yc)/wc.sum(1)[:,None]; sw=wt.sum()
                bs['wk0'][b]=((wt*Yt[:,H]).sum()/sw,(wt*cmb[:,H]).sum()/sw); bs['wk2..8'][b]=((wt*Yt[:,IX(2,8)].mean(1)).sum()/sw,(wt*cmb[:,IX(2,8)].mean(1)).sum()/sw)
            res[o]=(pt,bs)
        for k in ['wk0','wk2..8']:
            r=dict(feed=f,design=d,window=k,n=len(R['u']))
            lr={}
            for o in objs:
                (t,c),b=res[o][0][k],res[o][1][k]; e=np.log(max(t,1e-9)/max(c,1e-9)); bb=np.log(np.maximum(b[:,0],1e-9)/np.maximum(b[:,1],1e-9)); lr[o]=(e,bb)
                r[o+'_treated']=t; r[o+'_control']=c; r[o+'_RR']=np.exp(e); r[o+'_RR_lo'],r[o+'_RR_hi']=np.exp(np.percentile(bb,[2.5,97.5]))
            e=lr['like_notfollowed'][0]-lr['like_followed'][0]; bb=lr['like_notfollowed'][1]-lr['like_followed'][1]
            r['ratio_of_RR']=np.exp(e); r['ratio_lo'],r['ratio_hi']=np.exp(np.percentile(bb,[2.5,97.5])); r['se_log']=bb.std(); rows.append(r)
            print({a:(round(b,3) if isinstance(b,float) else b) for a,b in r.items()},flush=True)
pd.DataFrame(rows).to_csv('results/a2_ratio.csv',index=False)
