"""Event-time profile of revealed engagement around the explicit endorsement (liking a feed generator).
Target outcome: likes on posts surfaced by the endorsed feed. Within-user placebo: likes on posts of the other feeds.
Treated-only; no control group yet (see 05_*)."""
import pandas as pd, numpy as np
O='data/derived'
B=pd.read_parquet(f'{O}/feed_likes.parquet'); L=pd.read_parquet(f'{O}/post_likes.parquet'); L=L[L.ts<='2024-04-25']
FEEDS=['Science','Blacksky','News','#UkrainianView']; W=8
out=[]
for f in FEEDS:
    start=L[L.feed==f].ts.min()
    b=B[(B.feed==f)&(B.ts>=start+pd.Timedelta(weeks=W))&(B.ts<=pd.Timestamp('2024-03-18')-pd.Timedelta(weeks=W))]
    b=b.set_index('user').ts.rename('t0'); n=len(b)
    m=L[L.liker.isin(b.index)].merge(b,left_on='liker',right_index=True)
    m['wk']=np.floor((m.ts-m.t0).dt.total_seconds()/(7*86400)).astype(int)
    m=m[(m.wk>=-W)&(m.wk<W)]; m['target']=m.feed==f
    for tgt,g in m.groupby('target'):
        c=g.groupby('wk').agg(likes=('post','size'),users=('liker','nunique'))
        for wk in range(-W,W):
            out.append(dict(feed=f,outcome='target' if tgt else 'other_feeds',wk=wk,n=n,
                            likes_per_user=c.likes.get(wk,0)/n,pct_active=100*c.users.get(wk,0)/n))
    # day-level around t0 for the target
    g=m[m.target]; d=np.floor((g.ts-g.t0).dt.total_seconds()/86400).astype(int)
    print(f,'n=',n,'target likes/user by day -3..+3:',[round(((d==k).sum())/n,3) for k in range(-3,4)])
E_=pd.DataFrame(out); E_.to_csv('results/rq2_event_profile_treated.csv',index=False)
pd.set_option('display.width',250)
print(E_.pivot_table(index='wk',columns=['feed','outcome'],values='likes_per_user').round(3))
print(E_.pivot_table(index='wk',columns=['feed','outcome'],values='pct_active').round(1))
