"""A8 step 1: English authored posts (no reposts, >=20 chars) of every user in the text sample (all feed endorsers + a fixed random
sample of 80,000 feed-post likers), weeks 12..63, at most 8 randomly chosen posts per user-week. -> data/derived/a8_texts.parquet"""
import gzip, json, glob, datetime, numpy as np, pandas as pd
from multiprocessing import Pool
O='data/derived'; E0=datetime.date(2023,1,1).toordinal(); CAP=8
def work(f):
    out=[]
    with gzip.open(f,'rt') as fh:
        for l in fh:
            d=json.loads(l)
            if d['lang']!=['eng'] or len(d['text'])<20: continue
            s=str(d['date']); w=(datetime.date(int(s[:4]),int(s[4:6]),int(s[6:8])).toordinal()-E0)//7
            if 12<=w<=63: out.append((d['user_id'],w,d['text'][:500]))
    D=pd.DataFrame(out,columns=['user','w','text']); n0=len(D)
    D=D.sample(frac=1,random_state=0).groupby(['user','w']).head(CAP); return D,n0
if __name__=='__main__':
    with Pool(16) as p: R=p.map(work,sorted(glob.glob(f'{O}/texts/*.gz')))
    D=pd.concat([r[0] for r in R]).reset_index(drop=True); n0=sum(r[1] for r in R)
    D.to_parquet(f'{O}/a8_texts.parquet'); print('english posts',n0,'kept',len(D),'users',D.user.nunique(),'user-weeks',len(D.drop_duplicates(['user','w'])))
