"""RQ1: how much do explicit feed endorsements (liking the feed generator) overlap with revealed
engagement (liking posts the feed surfaced)? Descriptive only."""
import pandas as pd, numpy as np
O='data/derived'
B=pd.read_parquet(f'{O}/feed_likes.parquet'); L=pd.read_parquet(f'{O}/post_likes.parquet')
L=L[L.ts<='2024-04-25']
FEEDS=['Science','Blacksky','News','#UkrainianView']
rows=[]
for f in FEEDS:
    b=B[B.feed==f].set_index('user').ts; l=L[L.feed==f]
    start=l.ts.min()
    cnt=l.groupby('liker').size()
    E=set(b.index); K=set(cnt.index)
    m=l.merge(b.rename('t0'),left_on='liker',right_index=True)
    pre=m[m.ts<m.t0].groupby('liker').size(); post=m[m.ts>=m.t0].groupby('liker').size()
    # endorsers whose endorsement falls inside the window where the feed archive exists, with >=30d on both sides
    ok=b[(b>=start+pd.Timedelta(days=30))&(b<=pd.Timestamp('2024-02-17'))].index
    r=dict(feed=f,endorsers=len(E),post_likers=len(K),
           pct_endorsers_ever_like=100*len(E&K)/len(E),
           pct_endorsers_like_before=100*len(set(pre.index))/len(E),
           pct_endorsers_like_after=100*len(set(post.index))/len(E),
           pct_likers_who_endorse=100*len(E&K)/len(K),
           pct_top1pct_likers_who_endorse=100*np.mean([u in E for u in cnt[cnt>=cnt.quantile(.99)].index]),
           top1pct_threshold=cnt.quantile(.99),
           n_window=len(ok),
           pct_window_like_30d_before=100*np.mean([((m.liker==u)&(m.ts<m.t0)&(m.ts>=m.t0-pd.Timedelta(days=30))).any() for u in ok[:0]]) if False else np.nan)
    mw=m[m.liker.isin(ok)]; d=(mw.ts-mw.t0).dt.total_seconds()/86400
    r['pct_window_like_30d_before']=100*mw[(d<0)&(d>=-30)].liker.nunique()/len(ok)
    r['pct_window_like_30d_after']=100*mw[(d>=0)&(d<30)].liker.nunique()/len(ok)
    r['pct_window_like_0_1d']=100*mw[(d>=0)&(d<1)].liker.nunique()/len(ok)
    r['pct_window_no_like_ever']=100*(1-mw.liker.nunique()/len(ok))
    rows.append(r)
T=pd.DataFrame(rows).set_index('feed').T
pd.set_option('display.width',200); print(T.round(2)); T.to_csv('results/rq1_gap_table.csv')
# share of all likes on feed posts that come from endorsers
for f in FEEDS:
    l=L[L.feed==f]; E=set(B[B.feed==f].user)
    print(f,'share of likes from endorsers: %.2f%%'%(100*l.liker.isin(E).mean()),'| endorsers as share of likers: %.2f%%'%(100*len(E&set(l.liker))/l.liker.nunique()))
