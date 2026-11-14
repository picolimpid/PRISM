"""RQ3/RQ4/RQ8: matched event study around the explicit endorsement (liking a feed generator).
Unit = user; cohort = calendar week c that contains the endorsement. Controls never endorsed the feed and are
exact-matched within (feed, cohort week) on coarsened pre-period history. Three specifications:
  naive : existed before c, language only
  early : history up to week c-5 only (activity, tenure, target-feed likes, other-feed likes)  -> tests pre-endorsement drift
  full  : history up to week c-1
Week 0 (the week containing the endorsement) mixes pre and post days and is excluded from the DiD.
usage: python3 analysis/06_match_event_study.py [K controls=10] [n_boot=1000] [placebo shift in weeks=0]"""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
O='data/derived'; FEEDS=['Science','News','#UkrainianView']; NW=64; LASTW=62; H=8   # Blacksky archive only covers 2024-02/03: no pre-period
K=int(sys.argv[1]) if len(sys.argv)>1 else 10; NB=int(sys.argv[2]) if len(sys.argv)>2 else 1000
SHIFT=int(sys.argv[3]) if len(sys.argv)>3 else 0   # placebo: pretend the endorsement happened SHIFT weeks earlier
TAG=f'_placebo{SHIFT}' if SHIFT else ''
rng=np.random.default_rng(7)
M=np.load(f'{O}/panel_meta.npz'); users=M['users']; first_post=M['first'].astype(int); eng=M['eng']; N=len(users)
uix=pd.Series(np.arange(N),index=users)
dense=lambda n: np.asarray(sp.load_npz(f'{O}/{n}.npz').todense(),dtype=np.int32)
act=dense('act'); likes_all=dense('likes_ALL')
LK={f:dense('likes_'+f.strip('#')) for f in FEEDS}; IN={f:dense('inter_'+f.strip('#')) for f in FEEDS}
B=pd.read_parquet(f'{O}/feed_likes.parquet'); EP=pd.Timestamp('2023-01-01'); B['w']=((B.ts-EP).dt.days//7).astype(int)
def first_nz(a):
    nz=a>0; f=np.where(nz.any(1),nz.argmax(1),999); return f
first_like=first_nz(likes_all)
first_end=pd.Series(999,index=users); fe=B.groupby('user').w.min(); first_end.loc[fe.index]=fe.values
first_seen=np.minimum(np.minimum(first_post,first_like),first_end.values)
engbin=np.where(eng<0,2,(eng>0.5).astype(int))
cs=lambda a: np.concatenate([np.zeros((a.shape[0],1),dtype=np.int64),np.cumsum(a,1,dtype=np.int64)],1)   # cs[:,j]=sum of weeks <j
C_act=cs(act); C_all=cs(likes_all); C_lk={f:cs(LK[f]) for f in FEEDS}
win=lambda C,a,b: C[:,b]-C[:,a]      # sum over weeks a..b-1
dg=np.digitize
def keys(f,c,spec):
    oth=C_all-C_lk[f]
    if spec=='naive': parts=[(engbin,3)]
    elif spec=='active':   # active comparator: controls endorsed a different feed in the same week
        parts=[(dg(win(C_act,c-4,c),[1,16]),3),(dg(win(C_lk[f],c-4,c),[1,3]),3),(dg(c-first_post,[1,12]),3),(engbin,3)]
    elif spec=='early':
        e=c-4   # information up to week c-5 inclusive
        parts=[(dg(win(C_act,e-4,e),[1,4,16,64]),5),(dg(e-first_post,[1,4,12,26]),5),(dg(win(C_lk[f],e-4,e),[1,3]),3),
               (dg(win(oth,e-4,e),[1,5]),3),((C_all[:,e]>0).astype(int),2),(engbin,3)]
    else:
        parts=[(dg(win(C_act,c-4,c),[1,4,16,64]),5),(dg(act[:,c-1],[1,4]),3),(dg(c-first_post,[1,4,12,26]),5),
               (dg(LK[f][:,c-1],[1,2]),3),(dg(win(C_lk[f],c-4,c-1),[1,3]),3),(dg(win(C_lk[f],c-8,c-4),[1,3]),3),
               (dg(win(oth,c-4,c),[1,5]),3),((C_all[:,c]>0).astype(int),2),(engbin,3)]
    k=np.zeros(N,dtype=np.int64)
    for b,base in parts: k=k*base+b
    return k
OUT={'like_any':lambda f:(LK[f]>0),'like_cnt':lambda f:np.minimum(LK[f],20),'other_any':lambda f:((likes_all-LK[f])>0),
     'other_cnt':lambda f:np.minimum(likes_all-LK[f],20),'act_any':lambda f:(act>0),'act_log':lambda f:np.log1p(act),'inter_any':lambda f:(IN[f]>0)}
rows=[]; summ=[]; pairs=[]
for f in FEEDS:
    startw=int(np.argmax(LK[f].sum(0)>0)); bf=B[B.feed==f]; endorsers=uix.reindex(bf.user).values.astype(int)
    isend=np.zeros(N,bool); isend[endorsers]=True
    cw=pd.Series(bf.w.values-SHIFT,index=endorsers)
    Y={o:fn(f) for o,fn in OUT.items()}
    oth_end=np.zeros((N,NW),bool); bo=B[(B.feed!=f)&(B.w<NW)]; oth_end[uix.reindex(bo.user).values.astype(int),bo.w.values]=True
    for spec in ['naive','early','full','active']:
        T=[];Cm={o:[] for o in OUT};Tm={o:[] for o in OUT}; n_elig=0; n_noexist=0
        for c in range(startw+H,LASTW-H+1):
            t=cw.index[cw.values==c].values
            if len(t)==0: continue
            lim=c-1 if spec!='early' else c-5
            ex=first_seen[t]<=lim; n_noexist+=(~ex).sum(); t=t[ex]; n_elig+=len(t)
            if len(t)==0: continue
            k=keys(f,c,spec); cmask=(~isend)&(first_seen<=lim)
            if spec=='active': cmask=(~isend)&oth_end[:,c]&(first_seen<=lim)
            tk=k[t]; cand=np.flatnonzero(cmask&np.isin(k,np.unique(tk)))
            order=np.argsort(k[cand],kind='stable'); cand=cand[order]; ck=k[cand]
            lo=np.searchsorted(ck,tk,'left'); hi=np.searchsorted(ck,tk,'right')
            for i,u in enumerate(t):
                if hi[i]==lo[i]: continue
                pool=cand[lo[i]:hi[i]]; pick=pool if len(pool)<=K else rng.choice(pool,K,replace=False)
                T.append((u,c,len(pool)))
                if spec=='full': pairs.extend((f,c,users[u],users[v]) for v in pick)
                for o in OUT:
                    Tm[o].append(Y[o][u,c-H:c+H+1]); Cm[o].append(Y[o][pick,c-H:c+H+1].mean(0))
        n=len(T)
        print(f,spec,'eligible',n_elig,'matched',n,'no evidence of existence before c',n_noexist,flush=True)
        if n==0: continue
        bi=rng.integers(0,n,(NB,n))
        for o in OUT:
            tm=np.array(Tm[o],float); cm=np.array(Cm[o],float); d=tm-cm
            bd=d[bi].mean(1)        # NB x 17
            for j,kk in enumerate(range(-H,H+1)):
                rows.append(dict(feed=f,spec=spec,outcome=o,k=kk,n=n,treated=tm[:,j].mean(),control=cm[:,j].mean(),diff=d[:,j].mean(),
                                 lo=np.percentile(bd[:,j],2.5),hi=np.percentile(bd[:,j],97.5)))
            ix=lambda a,b: slice(a+H,b+H+1)
            stats={'pre(-8..-2)':d[:,ix(-8,-2)].mean(1),'pre_drift(-4..-1 minus -8..-5)':d[:,ix(-4,-1)].mean(1)-d[:,ix(-8,-5)].mean(1),
                   'wk-1':d[:,H-1],'wk0':d[:,H],'post(1..8)':d[:,ix(1,8)].mean(1),
                   'DiD post(2..8)-pre(-8..-2)':d[:,ix(2,8)].mean(1)-d[:,ix(-8,-2)].mean(1),
                   'DiD late(5..8)-pre(-8..-2)':d[:,ix(5,8)].mean(1)-d[:,ix(-8,-2)].mean(1),
                   'gap late(5..8)':d[:,ix(5,8)].mean(1)}
            for s,v in stats.items():
                b=v[bi].mean(1); summ.append(dict(feed=f,spec=spec,outcome=o,stat=s,n=n,est=v.mean(),lo=np.percentile(b,2.5),hi=np.percentile(b,97.5),
                                                 treated_post=tm[:,ix(2,8)].mean(),control_post=cm[:,ix(2,8)].mean()))
pd.DataFrame(rows).to_csv(f'results/rq3_event_study{TAG}.csv',index=False)
S=pd.DataFrame(summ); S.to_csv(f'results/rq3_summary{TAG}.csv',index=False)
if not SHIFT: pd.DataFrame(pairs,columns=['feed','c','treated','control']).to_parquet(f'{O}/matched_pairs_full.parquet')
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
S['ci']=S.apply(lambda r: f"{r.est:+.4f} [{r.lo:+.4f},{r.hi:+.4f}]",axis=1)
for o in ['like_any','act_any']:
    print('\n==',o); print(S[(S.outcome==o)&(S.spec=='active')].pivot_table(index=['feed','spec','n'],columns='stat',values='ci',aggfunc='first').to_string())
