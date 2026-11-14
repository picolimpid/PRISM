"""A8 (PLAN.md 6.7): writing similarity to a feed around (i) the FIRST engagement of any kind with that feed -- the treatment
definition of Pal et al. 2026 -- and (ii) the endorsement of the feed (the RQ7 anchor), now with matching on writing history
and a balanced-panel requirement, plus a placebo anchor 13 weeks earlier on the same users.
Outcome: weekly mean cosine (bge-small) between a user's English posts and the centroid of English posts carried by the feed.
Sample: users in the text sample (all feed endorsers + 80,000 random feed-post likers) with English posts.
Treated: anchor week c in the feed's cohort range. Controls: sample users who never had the anchoring event for this feed,
exact-matched within cohort week on quintile of pre-period similarity, number of pre-period weeks with posts, posting volume,
tenure and prior weeks with a like on the feed; up to 5 controls. Both groups need >=2 weeks with posts on each side of the anchor.
NOT a replication of Pal et al.: different sample, encoder (bge-small vs MiniLM), matching (CEM vs stratified PSM) and outcome."""
import warnings, zlib, numpy as np, pandas as pd, scipy.sparse as sp
warnings.filterwarnings('ignore',category=RuntimeWarning)
O='data/derived'; EP=pd.Timestamp('2023-01-01'); H=8; NW=64; LASTW=62; K=5; NB=1000; rng=np.random.default_rng(29)
T=pd.read_parquet(f'{O}/a8_texts.parquet',columns=['user','w']); S=np.empty((len(T),2),np.float32)
for i in range(3): S[i::3]=np.load(f'{O}/a8_sim_part{i}.npy')
M=np.load(f'{O}/panel_meta.npz'); users=M['users']; first_post=M['first'].astype(int); N=len(users); uix=pd.Series(np.arange(N),index=users)
dense=lambda n: np.asarray(sp.load_npz(f'{O}/{n}.npz').todense(),dtype=np.int32); act=dense('act')
cs=lambda a: np.concatenate([np.zeros((a.shape[0],1),dtype=np.int64),np.cumsum(a,1,dtype=np.int64)],1); C_act=cs(act)
su=np.sort(T.user.unique()); spos=pd.Series(np.arange(len(su)),index=su); pidx=uix.reindex(su).values.astype(int)     # sample users -> panel rows
B=pd.read_parquet(f'{O}/feed_likes.parquet'); B['w']=((B.ts-EP).dt.days//7).astype(int)
L=pd.read_parquet(f'{O}/post_likes.parquet'); L['w']=((L.ts-EP).dt.days//7).astype(int)
FP=pd.read_parquet(f'{O}/feed_posts.parquet',columns=['feed','post','author','ts']); FP['w']=((FP.ts-EP).dt.days//7).astype(int)
FI=pd.read_parquet(f'{O}/feed_interactions.parquet'); FI['ts']=pd.to_datetime(FI.date.astype(str),format='%Y%m%d%H%M',errors='coerce'); FI=FI.dropna(); FI['w']=((FI.ts-EP).dt.days//7).astype(int)
summ=[]; weekly=[]
def run(f,col,anchor_name,anchor,never,c0,shift=0,nyt=False):
    """anchor: Series user -> anchor week (treated); never: boolean array over sample users (eligible controls)."""
    global rng; rng=np.random.default_rng([29,zlib.crc32((f+anchor_name).encode()),shift])     # one stream per run, independent of run order
    d=T.assign(s=S[:,col]).groupby(['user','w']).s.mean().reset_index(); SIM=np.full((len(su),NW+H+1),np.nan,np.float32); SIM[spos.reindex(d.user).values,d.w.values]=d.s.values
    lk=dense('likes_'+f.strip('#'))[pidx]; C_nz=cs(lk>0)
    q=np.nanquantile(SIM,[.2,.4,.6,.8]); a_all=anchor.reindex(su).values; a_t=a_all-shift          # NaN for non-treated
    Tm=[];Cm=[];n_elig=0; pre_gap_raw=[]; cov_t=[]; cov_c=[]
    for c in range(c0,LASTW-H+1):
        a=c-shift
        if a-H<12: continue
        cols=np.arange(a-H,a+H+1); Y=SIM[:,cols]; pre=Y[:,:H]; post=Y[:,H+1:]
        npre=np.isfinite(pre).sum(1); npost=np.isfinite(post).sum(1); ok=(npre>=2)&(npost>=2)
        mpre=np.nanmean(np.where(np.isfinite(pre),pre,np.nan),1)
        a4=C_act[pidx,a]-C_act[pidx,max(a-4,0)]; ten=a-first_post[pidx]; lw=C_nz[:,a]-C_nz[:,max(a-8,0)]
        key=((np.digitize(mpre,q)*3+np.digitize(npre,[3,6]))*5+np.digitize(a4,[1,4,16,64]))*3+np.digitize(ten,[4,12]); key=key*3+np.digitize(lw,[1,4])
        if nyt: key=(np.digitize(mpre,q)*3+np.digitize(npre,[3,6]))*3+np.digitize(a4,[4,32])      # post hoc: controls have the same event >=13 weeks later; coarser cells for the small pool
        t=np.flatnonzero((a_t==a)&ok); cand=np.flatnonzero(((a_all>=c+13) if nyt else never)&ok); n_elig+=len(t)
        if len(t)==0 or len(cand)==0: continue
        pre_gap_raw.append((len(t),np.nanmean(mpre[t])-np.nanmean(mpre[cand])))
        o=np.argsort(key[cand],kind='stable'); cand=cand[o]; ck=key[cand]; lo=np.searchsorted(ck,key[t],'left'); hi=np.searchsorted(ck,key[t],'right')
        for i,u in enumerate(t):
            if hi[i]==lo[i]: continue
            pool=cand[lo[i]:hi[i]]; pick=pool if len(pool)<=K else rng.choice(pool,K,replace=False)
            Tm.append(Y[u]); Cm.append(np.nanmean(Y[pick],0)); cov_t.append((mpre[u],npre[u],np.log1p(a4[u]))); cov_c.append((mpre[pick].mean(),npre[pick].mean(),np.log1p(a4[pick]).mean()))
    if len(Tm)<50: print(f,anchor_name,'shift',shift,'too few matched:',len(Tm)); return
    Tm=np.array(Tm); Cm=np.array(Cm); D=Tm-Cm; n=len(D); ks=np.arange(-H,H+1); bi=rng.integers(0,n,(NB,n))
    um=lambda X,a,b: np.nanmean(X[:,a+H:b+H+1],1)
    did=(um(Tm,2,8)-um(Tm,-8,-2))-(um(Cm,2,8)-um(Cm,-8,-2)); late=(um(Tm,5,8)-um(Tm,-8,-2))-(um(Cm,5,8)-um(Cm,-8,-2)); pg=um(Tm,-8,-2)-um(Cm,-8,-2)
    wk=np.nanmean(D,0); bw=np.array([np.nanmean(D[b],0) for b in bi])
    def sl(v):                                                       # slope of the weekly gap over weeks -8..-2
        m=np.isfinite(v[:7]); return np.polyfit(ks[:7][m],v[:7][m],1)[0] if m.sum()>=3 else np.nan
    def add(stat,est,bs): summ.append(dict(feed=f,anchor=anchor_name,shift=shift,stat=stat,n=n,eligible=n_elig,est=est,se=np.nanstd(bs),lo=np.nanpercentile(bs,2.5),hi=np.nanpercentile(bs,97.5)))
    add('pre-trend slope of gap per week (-8..-2)',sl(wk),np.array([sl(b) for b in bw]))
    for nm,v in [('matched pre gap (-8..-2)',pg),('DiD post(2..8) - pre(-8..-2)',did),('DiD late(5..8) - pre(-8..-2)',late),('gap in anchor week',D[:,H])]:
        m=np.isfinite(v); vb=v[m]; b2=rng.integers(0,len(vb),(NB,len(vb))); add(nm,vb.mean(),vb[b2].mean(1))
    w_=np.array([x[0] for x in pre_gap_raw],float); g_=np.array([x[1] for x in pre_gap_raw]); summ.append(dict(feed=f,anchor=anchor_name,shift=shift,stat='UNMATCHED pre-period level gap',n=n,eligible=n_elig,est=np.average(g_,weights=w_),se=np.nan,lo=np.nan,hi=np.nan))
    ct=np.array(cov_t); cc=np.array(cov_c)
    for j,nm in enumerate(['pre similarity','pre weeks with posts','log activity']):
        sd=np.sqrt((ct[:,j].var()+cc[:,j].var())/2); summ.append(dict(feed=f,anchor=anchor_name,shift=shift,stat='SMD after matching: '+nm,n=n,eligible=n_elig,est=(ct[:,j].mean()-cc[:,j].mean())/sd if sd>0 else 0,se=np.nan,lo=np.nan,hi=np.nan))
    for j,k in enumerate(ks): weekly.append(dict(feed=f,anchor=anchor_name,shift=shift,k=k,n=n,treated=np.nanmean(Tm[:,j]),control=np.nanmean(Cm[:,j]),diff=wk[j],lo=np.percentile(bw[:,j],2.5),hi=np.percentile(bw[:,j],97.5),n_posting=int(np.isfinite(Tm[:,j]).sum())))
    print(f,anchor_name,'shift',shift,'eligible',n_elig,'matched',n,'sd of user-week similarity',float(np.nanstd(SIM)).__round__(4),flush=True)
for f,col in [('Science',0),('News',1)]:
    lkall=dense('likes_'+f.strip('#')); startw=int(np.argmax(lkall.sum(0)>0)); c0=startw+H; del lkall
    posts=set(FP[FP.feed==f].post)
    ev=pd.concat([L[L.feed==f][['liker','w']].rename(columns={'liker':'user'}),FI[FI.post.isin(posts)][['user','w']],B[B.feed==f][['user','w']],FP[FP.feed==f][['author','w']].rename(columns={'author':'user'})])
    fe=ev.groupby('user').w.min(); en=B[B.feed==f].set_index('user').w
    never_fe=~np.isin(su,fe.index.values); never_en=~np.isin(su,en.index.values)
    print(f,'sample users',len(su),'ever engaged',int((~never_fe).sum()),'never engaged',int(never_fe.sum()),'endorsers',int((~never_en).sum()),flush=True)
    for shift in (0,13):
        run(f,col,'first engagement of any kind',fe,never_fe,c0,shift)
        run(f,col,'endorsement',en,never_en,c0,shift)
    run(f,col,'first engagement of any kind; controls engage >=13 wk later',fe,never_fe,c0,0,nyt=True)
    run(f,col,'endorsement; controls endorse >=13 wk later',en,never_en,c0,0,nyt=True)
    pd.DataFrame(summ).to_csv('results/a8_writing_summary.csv',index=False); pd.DataFrame(weekly).to_csv('results/a8_writing_weekly.csv',index=False)
R=pd.DataFrame(summ); pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
R['v']=R.apply(lambda r: f"{r.est:+.4f}"+(f" [{r.lo:+.4f},{r.hi:+.4f}]" if np.isfinite(r.lo) else ''),axis=1)
print(R.pivot_table(index=['feed','anchor','shift','n'],columns='stat',values='v',aggfunc='first').T.to_string())
