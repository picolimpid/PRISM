"""Construct-validity descriptives requested by the advisor-panel pass (PLAN.md 6.7): A1, A3, A4.
A1  hour-level timing of likes on the feed's posts around the moment a user likes the feed generator.
    Baseline = same user, same clock time, +-7/14/21 days. Negative control = likes on other feeds' archived posts.
A3  are endorsers contributors (authors of posts the feed carries) or audience?
A4  endorsement behaviour across all 11 feeds: time since first visible trace, multi-feed bursts; what endorsers with no prior
    trace do afterwards; posts inside the feeds that ask people to like the feed.
Outputs: results/a1_timing_{hourly,summary}.csv, results/a3_contributors.csv, results/a4_{latency,bursts,notrace_types,solicit}.csv"""
import numpy as np, pandas as pd, scipy.sparse as sp
O='data/derived'; FEEDS=['Science','News','#UkrainianView']; EP=pd.Timestamp('2023-01-01'); rng=np.random.default_rng(22)
B=pd.read_parquet(f'{O}/feed_likes.parquet'); L=pd.read_parquet(f'{O}/post_likes.parquet'); P=pd.read_parquet(f'{O}/feed_posts.parquet',columns=['feed','post','author','ts','text'])
mins=lambda s: ((s-EP).dt.total_seconds()//60).astype(np.int64)
B['m']=mins(B.ts); L['m']=mins(L.ts); DAY=1440
# ---------------- A1
hourly=[]; summ=[]
def per_user(d):
    d=d.sort_values('m'); return {u:g.m.values for u,g in d.groupby('liker')}
def count(arr,lo,hi):     # number of events with lo <= m < hi
    return 0 if arr is None else int(np.searchsorted(arr,hi,'left')-np.searchsorted(arr,lo,'left'))
for f in FEEDS:
    Lf=L[L.feed==f]; wk=(Lf.m//(7*DAY)).value_counts().sort_index(); eff=int(wk.index[np.argmax(wk.values>=500)])*7*DAY; end=int(Lf.m.max())
    b=B[(B.feed==f)&(B.m>=eff+22*DAY)&(B.m<=end-22*DAY)]; T=per_user(Lf); Oth=per_user(L[(L.feed!=f)&L.liker.isin(b.user)])
    n=len(b); hrs=np.arange(-72,72); HT=np.zeros((n,len(hrs))); HO=np.zeros((n,len(hrs))); shifts=[-21,-14,-7,7,14,21]
    win={'10min':10,'1h':60,'24h':1440}; real={k:np.zeros(n) for k in win}; plac={k:np.zeros((n,len(shifts))) for k in win}; realO=np.zeros(n); placO=np.zeros((n,len(shifts)))
    before=same=after=0
    for i,(u,t0) in enumerate(zip(b.user.values,b.m.values)):
        a=T.get(u); o=Oth.get(u)
        for j,h in enumerate(hrs): HT[i,j]=count(a,t0+h*60,t0+(h+1)*60); HO[i,j]=count(o,t0+h*60,t0+(h+1)*60)
        for k,wd in win.items():
            real[k][i]=count(a,t0-wd,t0+wd+1)
            for j,s in enumerate(shifts): plac[k][i,j]=count(a,t0+s*DAY-wd,t0+s*DAY+wd+1)
        realO[i]=count(o,t0-60,t0+61)
        for j,s in enumerate(shifts): placO[i,j]=count(o,t0+s*DAY-60,t0+s*DAY+61)
        if a is not None:
            w=a[(a>=t0-60)&(a<=t0+60)]; before+=int((w<t0).sum()); same+=int((w==t0).sum()); after+=int((w>t0).sum())
    for j,h in enumerate(hrs): hourly.append(dict(feed=f,hour=h,n=n,target_likes_per_user=HT[:,j].mean(),target_share_users=(HT[:,j]>0).mean(),other_likes_per_user=HO[:,j].mean()))
    bi=rng.integers(0,n,(1000,n))
    def ratio(x,y):
        est=x.mean()/max(y.mean(),1e-12); bs=x[bi].mean(1)/np.maximum(y[bi].mean(1),1e-12); return est,np.percentile(bs,2.5),np.percentile(bs,97.5)
    r=dict(feed=f,n=n)
    for k in win:
        e,lo,hi=ratio(real[k],plac[k].mean(1)); r[f'rate_ratio_{k}']=e; r[f'rate_ratio_{k}_lo']=lo; r[f'rate_ratio_{k}_hi']=hi
        r[f'share_users_like_within_{k}']=(real[k]>0).mean(); r[f'share_users_like_within_{k}_placebo']=(plac[k]>0).mean()
    e,lo,hi=ratio(realO,placO.mean(1)); r['other_feeds_rate_ratio_1h']=e; r['other_feeds_rate_ratio_1h_lo']=lo; r['other_feeds_rate_ratio_1h_hi']=hi
    r['other_feeds_likes_per_user_1h']=realO.mean(); r['target_likes_per_user_1h']=real['1h'].mean(); r['target_likes_per_user_1h_placebo']=plac['1h'].mean()
    tot=before+same+after; r.update({'likes_in_pm60min':tot,'share_before_endorsement':before/max(tot,1),'share_same_minute':same/max(tot,1),'share_after_endorsement':after/max(tot,1)})
    summ.append(r); print(f,{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()},flush=True)
pd.DataFrame(hourly).to_csv('results/a1_timing_hourly.csv',index=False); pd.DataFrame(summ).set_index('feed').T.to_csv('results/a1_timing_summary.csv')
# ---------------- A3
rows=[]
for f in FEEDS+['Blacksky']:
    b=B[B.feed==f].set_index('user').ts; pf=P[P.feed==f]; first_auth=pf.groupby('author').ts.min(); likers=set(L[L.feed==f].liker)
    auth=set(first_auth.index); e=set(b.index); common=first_auth.reindex(b.index).dropna()
    rows.append(dict(feed=f,endorsers=len(e),authors=len(auth),endorsers_who_authored_pct=100*len(e&auth)/len(e),authored_before_endorsing_pct=100*(common.values<b.reindex(common.index).values).sum()/len(e),
                     endorse_rate_among_authors_pct=100*len(e&auth)/max(len(auth),1),endorse_rate_among_nonauthor_likers_pct=100*len((e&likers)-auth)/max(len(likers-auth),1),
                     share_of_feed_posts_by_endorsers_pct=100*pf.author.isin(e).mean()))
A3=pd.DataFrame(rows).set_index('feed').T; A3.to_csv('results/a3_contributors.csv'); print(A3.round(1).to_string())
# ---------------- A4 (i) latency from first visible trace, all 11 feeds
D=pd.read_parquet(f'{O}/user_day.parquet',columns=['user','day']); fp=D.groupby('user').day.min(); del D
fl=(L.groupby('liker').m.min()//DAY); B['day']=B.m//DAY
fe=B.sort_values('m').groupby('user').day.apply(list)
lat=[]
for f,u,d in zip(B.feed.values,B.user.values,B.day.values):
    prev=[x for x in fe[u] if x<d]; cands=[fp.get(u,np.inf),fl.get(u,np.inf)]+([min(prev)] if prev else []); first=min(cands)
    lat.append(d-first if first<d else (0 if first==d else -1))
B['lat']=lat
cat=pd.cut(B.lat,[-2,-1,0,7,28,90,10000],labels=['no trace before or on that day','first trace same day','1-7 days','8-28 days','29-90 days','>90 days'])
A4=pd.crosstab(B.feed,cat,normalize='index').mul(100); A4['n']=B.groupby('feed').size(); A4.loc['ALL']=list(pd.Series(cat).value_counts(normalize=True).reindex(A4.columns[:-1]).mul(100).values)+[len(B)]
A4.to_csv('results/a4_latency.csv'); print(A4.round(1).to_string())
# (ii) bursts: another feed endorsed by the same user within 60 minutes / the same day
B=B.sort_values(['user','m']); g=B.groupby('user').m; gap=np.minimum((B.m-g.shift(1)).abs().fillna(1e12),(g.shift(-1)-B.m).abs().fillna(1e12))
B['burst60']=gap<=60; B['burstday']=gap<=DAY; nf=B.groupby('user').feed.nunique()
A4b=B.groupby('feed').agg(n=('user','size'),within_60min_of_another_feed_pct=('burst60',lambda x:100*x.mean()),within_1day_pct=('burstday',lambda x:100*x.mean()))
A4b.loc['ALL']=[len(B),100*B.burst60.mean(),100*B.burstday.mean()]; A4b['users_with_2plus_feeds_pct']=100*(nf>=2).mean(); A4b.to_csv('results/a4_bursts.csv'); print(A4b.round(1).to_string())
B[['feed','user','m','lat','burst60','burstday']].to_parquet(f'{O}/endorse_events.parquet')
# (iii) what endorsers do in the following 8 weeks, split by whether they had any earlier visible trace
M=np.load(f'{O}/panel_meta.npz'); users=M['users']; N=len(users); uix=pd.Series(np.arange(N),index=users)
dense=lambda n: np.asarray(sp.load_npz(f'{O}/{n}.npz').todense(),dtype=np.int32); act=dense('act'); la=dense('likes_ALL'); H=8; LASTW=62; rows=[]
for f in FEEDS:
    lk=dense('likes_'+f.strip('#')); it=dense('inter_'+f.strip('#')); startw=int(np.argmax(lk.sum(0)>0))
    b=B[(B.feed==f)]; w=(b.m//(7*DAY)).values; ok=(w>=startw+1)&(w<=LASTW-H); b=b[ok]; w=w[ok]; u=uix.reindex(b.user).values.astype(int)
    sl=lambda A: np.array([A[i,c+1:c+H+1].sum() for i,c in zip(u,w)]); L8,I8,A8,O8=sl(lk),sl(it),sl(act),sl(la-lk)
    typ=np.select([L8>0,I8>0,(A8>0)|(O8>0)],['likes feed posts','replies/reposts feed posts only','active elsewhere, no feed engagement'],'no observable activity')
    for nm,m in [('no earlier trace',b.lat.values<=0),('earlier trace',b.lat.values>0)]:
        r=dict(feed=f,group=nm,n=int(m.sum())); r.update({k:100*v for k,v in pd.Series(typ[m]).value_counts(normalize=True).items()}); rows.append(r)
A4c=pd.DataFrame(rows).fillna(0); A4c.to_csv('results/a4_notrace_types.csv',index=False); print(A4c.round(1).to_string(index=False))
# (iv) posts carried by a feed that ask readers to like / pin that feed
t=P.text.fillna(''); m=t.str.contains(r'(?i)\b(?:like|pin|follow|save)\b')&t.str.contains(r'(?i)\bfeed\b')&t.str.contains(r'(?i)visib|discover|top 50|please|remember|welcome|keep')
S=P[m].groupby('feed').agg(posts=('post','size'),authors=('author','nunique')); S.to_csv('results/a4_solicit.csv'); print(S.to_string())
