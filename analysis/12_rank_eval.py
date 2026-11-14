"""Line B evaluation. Scores every impression (1 liked + 19 not-liked Science-feed posts) with:
 random, recency, popularity*, implicit-mean, implicit-knn, author-affinity, and (if profiles exist) explicit profiles + hybrids.
 *popularity uses final like counts, so it leaks the future; reported only as a reference.
History budgets K in {5,10,20,50,all} = the K most recent elicitation likes.
usage: 12_rank_eval.py [profiles.jsonl ...]   -> results/rankB_*.csv"""
import sys, json, pickle, numpy as np, pandas as pd
O='data/derived'; rng=np.random.default_rng(5)
P=pd.read_parquet(f'{O}/rank_posts.parquet'); EMB=np.load(f'{O}/rank_posts_emb.npy').astype(np.float32)
task=pickle.load(open(f'{O}/rank_task.pkl','rb')); users=sorted(task)
ts=P.ts.values.astype('int64'); lc=P.like_count.fillna(0).values; auth=P.author.values
def metrics(scores):   # scores[0] is the positive
    r=1+(scores[1:]>scores[0]).sum()+0.5*(scores[1:]==scores[0]).sum()
    return dict(mrr=1/r,ndcg10=(1/np.log2(1+r) if r<=10 else 0),hit1=float(r<1.5),auc=1-(r-1)/19)
def evaluate(scorer,name,K=None,subset=None):
    rows=[]
    for u in (subset or users):
        t=task[u]; h=t['hist'] if K is None else t['hist'][-K:]
        ctx=scorer.prep(u,h) if hasattr(scorer,'prep') else None; ms=[]
        for pos,neg in t['imps']:
            c=np.array([pos]+list(neg)); ms.append(metrics(scorer(u,h,c,ctx)))
        m=pd.DataFrame(ms).mean().to_dict(); m.update(user=u,method=name,K=K or 0,endorser=t['endorser']); rows.append(m)
    return rows
class Rand:
    def __call__(s,u,h,c,ctx): return rng.random(len(c))
class Rec:
    def __call__(s,u,h,c,ctx): return ts[c].astype(float)
class Pop:
    def __call__(s,u,h,c,ctx): return lc[c].astype(float)
class ImpMean:
    def prep(s,u,h): v=EMB[h].mean(0); return v/np.linalg.norm(v)
    def __call__(s,u,h,c,ctx): return EMB[c]@ctx
class ImpKnn:
    def prep(s,u,h): return EMB[h]
    def __call__(s,u,h,c,ctx): return (EMB[c]@ctx.T).max(1)
class Author:
    def prep(s,u,h): return pd.Series(auth[h]).value_counts().to_dict()
    def __call__(s,u,h,c,ctx): return np.array([ctx.get(a,0) for a in auth[c]],float)+1e-6*rng.random(len(c))
class Profile:      # explicit profile = list of statement embeddings per user
    def __init__(s,emb,agg='max',shuffle=False):
        s.emb=emb; s.agg=agg; s.map={u:u for u in emb}
        if shuffle:
            k=list(emb); p=k[1:]+k[:1]; s.map=dict(zip(k,p))
    def prep(s,u,h): return s.emb[s.map[u]]
    def __call__(s,u,h,c,ctx):
        m=EMB[c]@ctx.T; return m.max(1) if s.agg=='max' else m.mean(1)
class Hybrid:
    def __init__(s,a,b,w=0.5): s.a=a; s.b=b; s.w=w
    def prep(s,u,h): return tuple(x.prep(u,h) if hasattr(x,'prep') else None for x in (s.a,s.b))
    def __call__(s,u,h,c,ctx):
        z=lambda x: (x-x.mean())/(x.std()+1e-9)
        return (1-s.w)*z(s.a(u,h,c,ctx[0]))+s.w*z(s.b(u,h,c,ctx[1]))
if __name__=='__main__':
    R=[]
    for nm,sc in [('random',Rand()),('recency',Rec()),('popularity*',Pop())]: R+=evaluate(sc,nm)
    for K in [5,10,20,50,None]:
        for nm,sc in [('implicit-mean',ImpMean()),('implicit-knn',ImpKnn()),('author-affinity',Author()),('implicit-mean+author',Hybrid(ImpMean(),Author()))]: R+=evaluate(sc,nm,K)
    D=pd.DataFrame(R); D.to_csv('results/rankB_baselines.csv',index=False)
    pd.set_option('display.width',200); print(D.groupby(['method','K'])[['auc','mrr','ndcg10','hit1']].mean().round(4).to_string())
