"""RQ7 step 1: authored English posts of all feed endorsers within +-8 weeks of any of their endorsement weeks (<=30 per user-week)."""
import gzip, json, glob, numpy as np, pandas as pd, datetime
from multiprocessing import Pool
O='data/derived'; B=pd.read_parquet(f'{O}/feed_likes.parquet'); EP=pd.Timestamp('2023-01-01'); B['w']=((B.ts-EP).dt.days//7).astype(int)
W={}
for u,w in zip(B.user,B.w): W.setdefault(int(u),set()).update(range(w-8,w+9))
E0=datetime.date(2023,1,1).toordinal()
def work(f):
    out=[]
    with gzip.open(f,'rt') as fh:
        for l in fh:
            d=json.loads(l); u=d['user_id']; ws=W.get(u)
            if ws is None or d['lang']!=['eng'] or len(d['text'])<20: continue
            s=str(d['date']); w=(datetime.date(int(s[:4]),int(s[4:6]),int(s[6:8])).toordinal()-E0)//7
            if w in ws: out.append((u,w,d['text'][:500]))
    return out
if __name__=='__main__':
    with Pool(16) as p: R=[x for r in p.map(work,sorted(glob.glob(f'{O}/texts/*.gz'))) for x in r]
    D=pd.DataFrame(R,columns=['user','w','text']); n0=len(D)
    D=D.sample(frac=1,random_state=0).groupby(['user','w']).head(30).reset_index(drop=True)
    D.to_parquet(f'{O}/rq7_texts.parquet'); print(n0,len(D),D.user.nunique())
