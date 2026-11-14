"""A6 (PLAN.md 6.7): (a) among endorsers, who goes on to like the feed's posts in the next 8 weeks? Logistic regression,
odds ratios with 95% CI, per feed. Model 1 uses only information from before the endorsement; model 2 adds whether the
user liked a post of the feed within one hour of endorsing. (b) What the feed's creator sees: in a given week, what share
of the people who have liked the feed like any of its posts, and what share of the feed's likes comes from them."""
import numpy as np, pandas as pd, scipy.sparse as sp, statsmodels.api as sm
O='data/derived'; FEEDS=['Science','News','#UkrainianView']; H=8; LASTW=62; EP=pd.Timestamp('2023-01-01'); DAY=1440
M=np.load(f'{O}/panel_meta.npz'); users=M['users']; first_post=M['first'].astype(int); eng=M['eng']; N=len(users); uix=pd.Series(np.arange(N),index=users)
dense=lambda n: np.asarray(sp.load_npz(f'{O}/{n}.npz').todense(),dtype=np.int32); act=dense('act')
Ev=pd.read_parquet(f'{O}/endorse_events.parquet'); L=pd.read_parquet(f'{O}/post_likes.parquet'); L['m']=((L.ts-EP).dt.total_seconds()//60).astype(np.int64)
P=pd.read_parquet(f'{O}/feed_posts.parquet',columns=['feed','author','ts'])
rows=[]; crows=[]
for f in FEEDS:
    lk=dense('likes_'+f.strip('#')); startw=int(np.argmax(lk.sum(0)>0)); tot=lk.sum(0); eff=int(np.argmax(tot>=500))
    e=Ev[Ev.feed==f].copy(); e['w']=e.m//(7*DAY); e=e[(e.w>=startw+1)&(e.w<=LASTW-H)]; u=uix.reindex(e.user).values.astype(int); w=e.w.values
    y=np.array([lk[i,c+1:c+H+1].sum()>0 for i,c in zip(u,w)]).astype(float)
    lw=np.array([(lk[i,max(c-8,0):c]>0).sum() for i,c in zip(u,w)]); a4=np.array([act[i,max(c-4,0):c].sum() for i,c in zip(u,w)])
    prev=Ev.merge(e[['user','m']].rename(columns={'m':'m0'}),on='user'); prev=prev[(prev.feed!=f)&(prev.m<prev.m0)].groupby('user').feed.nunique(); nprev=prev.reindex(e.user).fillna(0).values
    fa=P[P.feed==f].groupby('author').ts.min(); fam=((fa-EP).dt.total_seconds()//60); contrib=(fam.reindex(e.user).values<e.m.values)
    Lf=L[L.feed==f].sort_values('m'); T={k:g.m.values for k,g in Lf.groupby('liker')}
    sess=np.array([0 if T.get(k) is None else int(np.searchsorted(T[k],t+61)-np.searchsorted(T[k],t-60))>0 for k,t in zip(e.user.values,e.m.values)]).astype(float)
    X=pd.DataFrame({'prior like-weeks 1-3 (ref 0)':((lw>=1)&(lw<=3)).astype(float),'prior like-weeks 4-8 (ref 0)':(lw>=4).astype(float),
                    'log activity, previous 4 weeks':np.log1p(a4),'no earlier visible trace':(e.lat.values<=0).astype(float),
                    'log days since first trace':np.log1p(np.clip(e.lat.values,0,None)),'endorsed 1 other feed before':(nprev==1).astype(float),'endorsed 2+ other feeds before':(nprev>=2).astype(float),
                    'another feed endorsed within 60 min':e.burst60.values.astype(float),'authored a post in this feed before':np.nan_to_num(contrib).astype(float),
                    'mostly English posts':(eng[u]>0.5).astype(float),'endorsed after public opening (wk>=57)':(w>=57).astype(float),'cohort week (per 10 weeks)':(w-w.mean())/10})
    for model,extra in [('M1 pre-endorsement only',[]),('M2 + in-session like',['liked a post of this feed within 1h of endorsing'])]:
        Xm=X.copy()
        if extra: Xm[extra[0]]=sess
        keep=[k for k in Xm.columns if Xm[k].std()>0 and not (set(np.unique(Xm[k]))<={0.,1.} and (Xm[k].sum()<20 or y[Xm[k].values==1].min()==y[Xm[k].values==1].max()))]
        dropped=[k for k in Xm.columns if k not in keep]
        if dropped: print(f,model,'dropped (fewer than 20 cases or no outcome variation):',dropped)
        Xm=Xm[keep]; fit=sm.Logit(y,sm.add_constant(Xm)).fit(disp=0,maxiter=200); ci=fit.conf_int()
        for k in Xm.columns: rows.append(dict(feed=f,model=model,term=k,n=len(y),base_rate=y.mean(),OR=np.exp(fit.params[k]),lo=np.exp(ci.loc[k,0]),hi=np.exp(ci.loc[k,1]),p=fit.pvalues[k],pseudo_r2=fit.prsquared,share_with_term=Xm[k].mean() if set(np.unique(Xm[k]))<={0.,1.} else np.nan))
    print(f,'n',len(y),'base rate',y.mean().round(3),'in-session share',sess.mean().round(3),'y|session',y[sess==1].mean().round(3),'y|no session',y[sess==0].mean().round(3),flush=True)
    # (b) creator's view, week by week
    ew=np.full(N,999); b_all=Ev[Ev.feed==f]; ew[uix.reindex(b_all.user).values.astype(int)]=(b_all.m//(7*DAY)).values
    for c in range(eff+8,LASTW+1):
        stock=ew<=c; col=lk[:,c]; liker=col>0
        crows.append(dict(feed=f,week=c,endorsers_so_far=int(stock.sum()),pct_endorsers_liking_this_week=100*(liker&stock).sum()/max(stock.sum(),1),
                          pct_of_likes_from_endorsers=100*col[stock].sum()/max(col.sum(),1),pct_of_likers_who_endorsed=100*(liker&stock).sum()/max(liker.sum(),1),likers=int(liker.sum())))
A=pd.DataFrame(rows); A.to_csv('results/a6_who_engages.csv',index=False); C=pd.DataFrame(crows); C.to_csv('results/a6_creator_view.csv',index=False)
pd.set_option('display.width',220); pd.set_option('display.max_rows',200)
A['ci']=A.apply(lambda r: f"{r.OR:.2f} [{r.lo:.2f},{r.hi:.2f}]"+('*' if r.p<0.05 else ''),axis=1)
print(A.pivot_table(index='term',columns=['model','feed'],values='ci',aggfunc='first').to_string())
print(A.groupby(['feed','model'])[['n','base_rate','pseudo_r2']].first().round(3).to_string())
print(C.groupby('feed')[['pct_endorsers_liking_this_week','pct_of_likes_from_endorsers','pct_of_likers_who_endorsed']].agg(['median','min','max']).round(1).to_string())
