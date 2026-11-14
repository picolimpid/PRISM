"""RQ2 (the original proposal's hybrid question, observationally): does the explicit signal (having liked the feed
generator before week c) predict engagement with the feed's posts in weeks c..c+7 beyond the implicit history?
Models: implicit-only, explicit-only, hybrid, hybrid with the explicit flag shuffled across users.
Populations: A = endorsers-before-c + 10 random existing non-endorsers each (case-control); B = everyone with >=1 like on the
feed's posts in the 8 weeks before c ("already engaging"). 5-fold CV grouped by user, bootstrap over users for CIs.
Pre-specified criterion (PLAN.md): hybrid - implicit dAUC CI excludes 0 and shuffled ~ 0."""
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier as GB
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
O='data/derived'; FEEDS=['News','#UkrainianView']; LASTW=62
rng=np.random.default_rng(11)
M=np.load(f'{O}/panel_meta.npz'); users=M['users']; first_post=M['first'].astype(int); eng=M['eng']; N=len(users)
uix=pd.Series(np.arange(N),index=users)
dense=lambda n: np.asarray(sp.load_npz(f'{O}/{n}.npz').todense(),dtype=np.int32)
act=dense('act'); likes_all=dense('likes_ALL')
B=pd.read_parquet(f'{O}/feed_likes.parquet'); EP=pd.Timestamp('2023-01-01'); B['w']=((B.ts-EP).dt.days//7).astype(int)
cs=lambda a: np.concatenate([np.zeros((a.shape[0],1),dtype=np.int64),np.cumsum(a,1,dtype=np.int64)],1)
C_act=cs(act); C_all=cs(likes_all)
nz=likes_all>0; first_like=np.where(nz.any(1),nz.argmax(1),999)
fe=B.groupby('user').w.min(); first_end=np.full(N,999); first_end[uix.reindex(fe.index).values.astype(int)]=fe.values
first_seen=np.minimum(np.minimum(first_post,first_like),first_end)
res=[]
for f in FEEDS:
    lk=dense('likes_'+f.strip('#')); it=dense('inter_'+f.strip('#')); C_l=cs(lk); C_i=cs(it); C_o=C_all-C_l
    startw=int(np.argmax(lk.sum(0)>0)); bf=B[B.feed==f]
    endw=np.full(N,999); endw[uix.reindex(bf.user).values.astype(int)]=bf.w.values
    rows=[]
    for c in range(startw+8,LASTW-8+1,4):
        w=lambda C,a,b: (C[:,c-b]-C[:,c-a]) if False else (C[:,c-a+1]-C[:,max(c-b,0)])   # weeks c-b .. c-a inclusive (a<=b, lags)
        E=(endw<c).astype(int)
        X=np.column_stack([w(C_l,1,1),w(C_l,2,2),w(C_l,3,4),w(C_l,5,8),w(C_l,9,16),C_l[:,c],
                           w(C_o,1,1),w(C_o,2,4),w(C_o,5,8),C_o[:,c],
                           w(C_act,1,1),w(C_act,2,4),w(C_act,5,8),C_act[:,c],
                           w(C_i,1,4),C_i[:,c],np.clip(c-first_post,-1,80),eng])
        y=((C_l[:,c+8]-C_l[:,c])>0).astype(int)
        exist=first_seen<c
        e_idx=np.flatnonzero((E==1)&exist); pool=np.flatnonzero((E==0)&exist&(endw>=999))   # never-endorsers as non-endorsed
        a_idx=np.concatenate([e_idx,rng.choice(pool,min(len(pool),10*len(e_idx)),replace=False)]) if len(e_idx) else np.array([],int)
        b_idx=np.flatnonzero(exist&(w(C_l,1,8)>0)&((E==1)|(endw>=999)))
        for pop,idx in [('A_case_control',a_idx),('B_already_engaging',b_idx)]:
            if len(idx)==0: continue
            d=pd.DataFrame(X[idx]); d['E']=E[idx]; d['since']=np.where(E[idx]==1,c-endw[idx],-1); d['y']=y[idx]; d['u']=idx; d['c']=c; d['pop']=pop; rows.append(d)
    D=pd.concat(rows)
    for pop,d in D.groupby('pop'):
        d=d.reset_index(drop=True); Xi=d[list(range(18))].values; y=d.y.values; g=d.u.values
        Es=d.E.values.copy(); rng.shuffle(Es)
        sets={'implicit':Xi,'explicit':d[['E','since']].values,'hybrid':np.column_stack([Xi,d.E.values,d.since.values]),'shuffled':np.column_stack([Xi,Es,np.where(Es==1,5,-1)])}
        P={k:np.zeros(len(d)) for k in sets}
        for tr,te in GroupKFold(5).split(Xi,y,g):
            for k,Xk in sets.items():
                m=GB(max_iter=200,learning_rate=0.08,max_leaf_nodes=31,l2_regularization=1.0,random_state=0).fit(Xk[tr],y[tr]); P[k][te]=m.predict_proba(Xk[te])[:,1]
        ug=np.unique(g); gi=None  # row bootstrap (most users contribute 1-2 rows)
        def boot(fn,nb=200):
            v=[]
            for _ in range(nb):
                if gi is not None: ii=np.concatenate([gi[u] for u in rng.choice(ug,len(ug))])
                else: ii=rng.integers(0,len(y),len(y))
                if y[ii].min()==y[ii].max(): continue
                v.append(fn(ii))
            return np.percentile(v,[2.5,97.5])
        r=dict(feed=f,pop=pop,n=len(d),users=len(ug),pos_rate=y.mean(),E_rate=d.E.mean())
        for k in sets: r['auc_'+k]=roc_auc_score(y,P[k]); r['ap_'+k]=average_precision_score(y,P[k])
        for k in ['hybrid','shuffled']:
            lo,hi=boot(lambda ii: roc_auc_score(y[ii],P[k][ii])-roc_auc_score(y[ii],P['implicit'][ii])); r[f'dAUC_{k}']=r['auc_'+k]-r['auc_implicit']; r[f'dAUC_{k}_lo']=lo; r[f'dAUC_{k}_hi']=hi
            lo,hi=boot(lambda ii: average_precision_score(y[ii],P[k][ii])-average_precision_score(y[ii],P['implicit'][ii])); r[f'dAP_{k}']=r['ap_'+k]-r['ap_implicit']; r[f'dAP_{k}_lo']=lo; r[f'dAP_{k}_hi']=hi
        e=d.E.values==1
        # post hoc calibration view: implicit-only model's expected vs observed engagement among endorsers
        r['obs_E']=y[e].mean(); r['exp_implicit_E']=P['implicit'][e].mean(); lo,hi=boot(lambda ii: y[ii][e[ii]].mean()/max(P['implicit'][ii][e[ii]].mean(),1e-9)) if e.sum()>20 else (np.nan,np.nan)
        r['OE']=r['obs_E']/r['exp_implicit_E']; r['OE_lo']=lo; r['OE_hi']=hi
        r['obs_nonE']=y[~e].mean(); r['exp_implicit_nonE']=P['implicit'][~e].mean()
        res.append(r); print(r,flush=True); pd.DataFrame(res).to_csv('results/rq2_prediction_news_ukr.csv',index=False)
R=pd.concat([pd.read_csv('results/rq2_prediction_partial.csv').query("feed=='Science'"),pd.DataFrame(res)]); R.to_csv('results/rq2_prediction.csv',index=False)
pd.set_option('display.width',250); print(R.set_index(['feed','pop']).T.round(4).to_string())
