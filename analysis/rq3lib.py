"""Shared code for the round-2 RQ3 robustness analyses (PLAN.md section 6): panel loading, coarsened-exact-matching
designs, and a Bayesian bootstrap over users in which treated AND control users carry weights (so re-used controls
do not make the intervals too narrow). Weeks are counted from 2023-01-01; `a` is the anchor week of a cohort
(the real endorsement week c, or c-S for a placebo date)."""
import warnings, zlib, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import norm
warnings.filterwarnings('ignore', category=RuntimeWarning)
O='data/derived'; NW=64; LASTW=62; H=8; EP=pd.Timestamp('2023-01-01'); FEEDS=['Science','News','#UkrainianView']
M=np.load(f'{O}/panel_meta.npz'); users=M['users']; first_post=M['first'].astype(int); eng=M['eng']; N=len(users)
uix=pd.Series(np.arange(N),index=users)
dense=lambda n: np.asarray(sp.load_npz(f'{O}/{n}.npz').todense(),dtype=np.int32)
cs=lambda a: np.concatenate([np.zeros((a.shape[0],1),dtype=np.int64),np.cumsum(a,1,dtype=np.int64)],1)   # cs[:,j] = sum of weeks < j
act=dense('act'); likes_all=dense('likes_ALL'); C_act=cs(act); C_all=cs(likes_all)
B=pd.read_parquet(f'{O}/feed_likes.parquet'); B['w']=((B.ts-EP).dt.days//7).astype(int)
_nz=likes_all>0; first_like=np.where(_nz.any(1),_nz.argmax(1),999); del _nz
_fe=B.groupby('user').w.min(); first_end=np.full(N,999); first_end[uix.reindex(_fe.index).values.astype(int)]=_fe.values
first_seen=np.minimum(np.minimum(first_post,first_like),first_end)      # first week with any trace of the account
engbin=np.where(eng<0,2,(eng>0.5).astype(int))
def _last_nz(a):
    nz=a>0; return np.where(nz.any(1),a.shape[1]-1-nz[:,::-1].argmax(1),-1)
_le=B.groupby('user').w.max(); last_end=np.full(N,-1); last_end[uix.reindex(_le.index).values.astype(int)]=_le.values
last_seen=np.maximum(np.maximum(_last_nz(act),_last_nz(likes_all)),last_end)   # last week with any trace (post, like on an archived post, feed endorsement)
dg=np.digitize
def W(C,a,b): return C[:,max(b,0)]-C[:,max(a,0)]                         # sum over weeks a..b-1

class Feed:
    def __init__(s,f,inter=False):
        s.f=f; s.LK=dense('likes_'+f.strip('#')); s.C_lk=cs(s.LK); s.nz=s.LK>0; s.C_nz=cs(s.nz)
        tot=s.LK.sum(0); s.startw=int(np.argmax(tot>0)); s.eff=int(np.argmax(tot>=500))    # eff: first week with a real volume of archived likes
        bf=B[B.feed==f]; s.endw=np.full(N,999); s.endw[uix.reindex(bf.user).values.astype(int)]=bf.w.values; s.isend=s.endw<999
        s.oth=C_all-s.C_lk
        bo=B[(B.feed!=f)&(B.w<NW)]; s.oth_end=np.zeros((N,NW),bool); s.oth_end[uix.reindex(bo.user).values.astype(int),bo.w.values]=True
        s.IN=dense('inter_'+f.strip('#')) if inter else None
    def Y(s,o):
        if o=='like_any': return s.nz
        if o=='like_cnt': return np.minimum(s.LK,20)
        if o=='like2': return s.LK>=2
        if o=='other_any': return (likes_all-s.LK)>0
        if o=='act_any': return act>0
        if o=='inter_any': return s.IN>0
        if o=='like_followed': return dense('likesF_'+s.f.strip('#'))>0       # A2: liker follows the post's author (snapshot)
        if o=='like_notfollowed': return dense('likesN_'+s.f.strip('#'))>0
        raise KeyError(o)

def keyparts(F,a,design):
    """List of (bin index per user, number of bins). Controls must fall in the same cell as the treated user."""
    if design=='full':        # first-round specification: history up to week a-1
        return [(dg(W(C_act,a-4,a),[1,4,16,64]),5),(dg(act[:,a-1],[1,4]),3),(dg(a-first_post,[1,4,12,26]),5),
                (dg(F.LK[:,a-1],[1,2]),3),(dg(W(F.C_lk,a-4,a-1),[1,3]),3),(dg(W(F.C_lk,a-8,a-4),[1,3]),3),
                (dg(W(F.oth,a-4,a),[1,5]),3),((C_all[:,a]>0).astype(int),2),(engbin,3)]
    if design=='coarse':      # granularity check: fewer, wider bins
        return [(dg(W(C_act,a-4,a),[1,16]),3),(dg(a-first_post,[1,12]),3),(dg(F.LK[:,a-1],[1]),2),(dg(W(F.C_lk,a-8,a-1),[1,3]),3),
                ((W(F.oth,a-4,a)>0).astype(int),2),(engbin,3)]
    if design=='fine':        # granularity check: more, narrower bins
        return [(dg(W(C_act,a-4,a),[1,2,4,8,16,32,64,128]),9),(dg(act[:,a-1],[1,2,4,8,16]),6),(dg(a-first_post,[1,2,4,8,12,18,26]),8),
                (dg(F.LK[:,a-1],[1,2,3,5]),5),(dg(W(F.C_lk,a-4,a-1),[1,2,3,5,9]),6),(dg(W(F.C_lk,a-8,a-4),[1,2,3,5,9]),6),
                (dg(W(F.oth,a-4,a),[1,2,5,10]),5),((C_all[:,a]>0).astype(int),2),(engbin,3)]
    if design=='seq8':        # T1: exact 0/1 sequence of "liked a post of this feed" over weeks a-8..a-1
        bits=np.zeros(N,dtype=np.int64)
        for j in range(1,9): bits=bits*2+(F.nz[:,a-j] if a-j>=0 else 0)
        return [(bits,256),(dg(W(F.C_lk,a-8,a),[1,4,11]),4),(dg(W(C_act,a-4,a),[1,16]),3),(dg(a-first_post,[1,4,12]),4),
                ((W(F.oth,a-4,a)>0).astype(int),2),(engbin,3)]
    if design=='persist16':   # T2: persistence (weeks with a like) and volume over the previous 16 weeks, plus recent level
        return [(dg(W(F.C_nz,a-16,a),[1,3,6,10]),5),(dg(W(F.C_lk,a-16,a),[1,4,11,31]),5),(dg(F.LK[:,a-1],[1,2]),3),(dg(W(F.C_lk,a-4,a-1),[1,3]),3),
                (dg(W(C_act,a-4,a),[1,4,16]),4),(dg(a-first_post,[1,4,12]),4),((W(F.oth,a-4,a)>0).astype(int),2),(engbin,3)]
    if design=='nyt':         # T3: controls are not-yet-treated endorsers of the same feed
        return [(dg(W(C_act,a-4,a),[1,16]),3),(dg(W(F.C_lk,a-4,a),[1,3]),3),(dg(W(F.C_nz,a-8,a),[1,3,6]),4),(dg(a-first_post,[1,12]),3),(engbin,3)]
    if design=='active':      # first-round active comparator: endorsed a different feed in the same week
        return [(dg(W(C_act,a-4,a),[1,16]),3),(dg(W(F.C_lk,a-4,a),[1,3]),3),(dg(a-first_post,[1,12]),3),(engbin,3)]
    if design.startswith('lag') or design.startswith('nytlag'):   # history up to week a-L inclusive (lag5 = the first round's "early"); nytlagL uses not-yet-treated controls
        L=int(design.replace('nyt','')[3:]); e=a-L+1
        return [(dg(W(C_act,e-4,e),[1,4,16,64]),5),(dg(e-first_post,[1,4,12,26]),5),(dg(W(F.C_lk,e-4,e),[1,3]),3),
                (dg(W(F.oth,e-4,e),[1,5]),3),((C_all[:,max(e,0)]>0).astype(int),2),(engbin,3)]
    raise KeyError(design)

def key(F,a,design):
    k=np.zeros(N,dtype=np.int64)
    for b,base in keyparts(F,a,design): k=k*base+b
    return k

NYT_GAP=13
def match(F,design,S=0,K=10,seed=0,gap=NYT_GAP,ctrl_alive=None,treat_alive=None):
    """Match each endorser of feed F (real cohort week c, anchor a=c-S) to up to K controls in the same cell.
    S=0: the first round's cohort range. S>0: same real cohorts, kept only if the placebo anchor has >=4 archived weeks before it."""
    rng=np.random.default_rng([seed,S,K,zlib.crc32(design.encode()),FEEDS.index(F.f),gap])   # crc32: stable across processes
    L=int(design.replace('nyt','')[3:]) if 'lag' in design else 1; nyt=design.startswith('nyt')
    c0=F.startw+H if S==0 else max(F.startw+H,F.eff+4+S)
    U=[];A=[];C=[];P=[];NP=[]; n_elig=0; n_noexist=0
    for c in range(c0,LASTW-H+1):
        a=c-S; t=np.flatnonzero(F.endw==c)
        if len(t)==0: continue
        lim=a-L; ex=first_seen[t]<=lim; n_noexist+=int((~ex).sum()); t=t[ex]; n_elig+=len(t)
        if treat_alive is not None: t=t[last_seen[t]>=c+treat_alive]      # post hoc: treated known to be around later, as nyt controls are
        if len(t)==0: continue
        k=key(F,a,design)
        if design=='active': cmask=(~F.isend)&F.oth_end[:,a]
        elif nyt: cmask=F.isend&(F.endw>=a+gap)&(F.endw!=c)
        else: cmask=~F.isend
        cmask&=first_seen<=lim
        if ctrl_alive is not None: cmask&=last_seen>=a+ctrl_alive               # post hoc: controls known to be around later, as placebo pseudo-endorsers are
        tk=k[t]; cand=np.flatnonzero(cmask&np.isin(k,np.unique(tk)))
        order=np.argsort(k[cand],kind='stable'); cand=cand[order]; ck=k[cand]
        lo=np.searchsorted(ck,tk,'left'); hi=np.searchsorted(ck,tk,'right')
        for i,u in enumerate(t):
            if hi[i]==lo[i]: continue
            pool=cand[lo[i]:hi[i]]; pick=pool if len(pool)<=K else rng.choice(pool,K,replace=False)
            row=np.full(K,-1,dtype=np.int64); row[:len(pick)]=pick
            U.append(u);A.append(a);C.append(c);P.append(row);NP.append(len(pool))
    return dict(u=np.array(U,dtype=np.int64),a=np.array(A,dtype=np.int64),c=np.array(C,dtype=np.int64),
                picks=np.array(P,dtype=np.int64).reshape(-1,K),npool=np.array(NP),n_elig=n_elig,n_noexist=n_noexist)

def subset(R,keep_users):
    m=np.isin(R['u'],keep_users); o=np.argsort(R['u'][m])
    return {k:(v[m][o] if isinstance(v,np.ndarray) else v) for k,v in R.items()}

def build(F,R,outcome):
    """Event-time outcome profiles (weeks a-8..a+8) for treated users and their matched controls."""
    Y=F.Y(outcome); u=R['u']; picks=R['picks']; cols=R['a'][:,None]+np.arange(-H,H+1)[None,:]
    Yt=Y[u[:,None],cols].astype(np.float32); pk=np.where(picks>=0,picks,0)
    Yc=Y[pk[:,:,None],cols[:,None,:]].astype(np.float32)
    bad=cols<F.startw; Yt[bad]=np.nan; Yc[np.broadcast_to(bad[:,None,:],Yc.shape)]=np.nan    # weeks before the feed archive starts
    return dict(u=u,picks=picks,valid=picks>=0,Yt=Yt,Yc=Yc)

IX=lambda a,b: slice(a+H,b+H+1)
def user_stats(d):
    nm=lambda x: np.nanmean(x,1); pre=nm(d[:,IX(-8,-2)])
    return {'pre(-8..-2)':pre,'drift(-4..-1 minus -8..-5)':nm(d[:,IX(-4,-1)])-nm(d[:,IX(-8,-5)]),'wk-1':d[:,H-1],'wk0':d[:,H],
            'gap(2..8)':nm(d[:,IX(2,8)]),'gap(5..8)':nm(d[:,IX(5,8)]),'DiD(2..8)':nm(d[:,IX(2,8)])-pre,'DiD(5..8)':nm(d[:,IX(5,8)])-pre}

def estimate(objs,NB=1000,seed=0):
    """Point estimates with equal weights; Bayesian bootstrap (Exp(1) weight per user, shared across the objects passed in,
    so differences between objects on the same users are paired)."""
    rng=np.random.default_rng(seed)
    ids=np.unique(np.concatenate([np.r_[o['u'],o['picks'][o['valid']]] for o in objs.values()]))
    Wt=rng.exponential(size=(NB,len(ids))).astype(np.float32); out={}
    for name,o in objs.items():
        it=np.searchsorted(ids,o['u']); ic=np.searchsorted(ids,np.where(o['valid'],o['picks'],ids[0])); Yt,Yc,valid=o['Yt'],o['Yc'],o['valid']
        cm=np.einsum('nk,nkj->nj',valid.astype(np.float32),Yc)/valid.sum(1)[:,None]; d=Yt-cm; st=user_stats(d)
        bs={k:np.empty(NB) for k in st}; bwk=np.empty((NB,2*H+1)); bs['logRR(2..8)']=np.empty(NB)
        tp=Yt[:,IX(2,8)].mean(1); st_rr=float(np.log(tp.mean()/max(cm[:,IX(2,8)].mean(),1e-9)))
        for b in range(NB):
            wt=Wt[b,it]; wc=Wt[b,ic]*valid
            db=Yt-np.einsum('nk,nkj->nj',wc,Yc)/wc.sum(1)[:,None]
            for k,v in user_stats(db).items():
                m=np.isfinite(v); bs[k][b]=(wt[m]*v[m]).sum()/wt[m].sum()
            bs['logRR(2..8)'][b]=np.log(max((wt*tp).sum(),1e-9)/max((wt*(Yt-db)[:,IX(2,8)].mean(1)).sum(),1e-9))
            mm=np.isfinite(db); bwk[b]=np.where(mm,db*wt[:,None],0).sum(0)/np.maximum((mm*wt[:,None]).sum(0),1e-9)
        out[name]=dict(point={**{k:float(np.nanmean(v)) for k,v in st.items()},'logRR(2..8)':st_rr},bs=bs,wk=np.nanmean(d,0),bwk=bwk,
                       tmean=np.nanmean(Yt,0),cmean=np.nanmean(cm,0),n=len(o['u']),n_ctrl=int(len(np.unique(o['picks'][valid]))))
    return out

def summarize(est,bsamp,**meta):
    se=float(np.std(bsamp)); lo,hi=np.percentile(bsamp,[2.5,97.5])
    return dict(**meta,est=est,se=se,lo=float(lo),hi=float(hi),p=float(2*norm.sf(abs(est)/se)) if se>0 else np.nan)
