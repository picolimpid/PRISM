"""Line B main table: explicit (LLM-inferred) profiles vs implicit history vs hybrid vs shuffled-profile control, same 600 users.
Also each method combined with recency (z-sum) because recency alone is a strong exposure proxy. Paired bootstrap over users."""
import json, pickle, numpy as np, pandas as pd, importlib.util
spec=importlib.util.spec_from_file_location('ev','analysis/12_rank_eval.py'); ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
S=pd.read_parquet('data/derived/llm_statements.parquet'); SE=np.load('data/derived/llm_statements_emb.npy').astype(np.float32)
rng=np.random.default_rng(1); R=[]
homog=[]
for K in (5,20):
    prof={}
    for kind in ('topic','value'):
        g=S[(S.K==K)&(S.kind==kind)]; prof[kind]={u:SE[ix.values] for u,ix in g.groupby('user').groups.items()}
        # homogenisation: mean cosine between different users' profile centroids
        C=np.stack([v.mean(0)/np.linalg.norm(v.mean(0)) for v in prof[kind].values()]); sim=C@C.T; homog.append(dict(K=K,kind=kind,cross_user_cos=(sim.sum()-len(C))/(len(C)*(len(C)-1))))
    sub=sorted(set(prof['topic'])&set(prof['value'])); 
    both={u:np.vstack([prof['topic'][u],prof['value'][u]]) for u in sub}
    M={'implicit-mean':ev.ImpMean(),'implicit-knn':ev.ImpKnn(),'explicit-topic':ev.Profile(prof['topic']),'explicit-value':ev.Profile(prof['value']),
       'explicit-topic+value':ev.Profile(both),'explicit-topic (shuffled user)':ev.Profile({u:prof['topic'][u] for u in sub},shuffle=True),
       'hybrid knn+topic':ev.Hybrid(ev.ImpKnn(),ev.Profile(prof['topic'])),'hybrid knn+value':ev.Hybrid(ev.ImpKnn(),ev.Profile(prof['value'])),
       'hybrid knn+shuffled topic':ev.Hybrid(ev.ImpKnn(),ev.Profile({u:prof['topic'][u] for u in sub},shuffle=True)),'recency':ev.Rec()}
    for nm,sc in list(M.items()):
        R+=ev.evaluate(sc,nm,K,subset=sub)
        if nm!='recency': R+=ev.evaluate(ev.Hybrid(ev.Rec(),sc),'recency + '+nm,K,subset=sub)
D=pd.DataFrame(R); D.to_csv('results/rankB_explicit.csv',index=False)
T=D.groupby(['K','method'])[['auc','mrr','ndcg10']].mean().round(4); pd.set_option('display.width',200); pd.set_option('display.max_rows',200); print(T.to_string()); print(pd.DataFrame(homog))
def paired(a,b,K,m='auc'):
    x=D[(D.method==a)&(D.K==K)].set_index('user')[m]; y=D[(D.method==b)&(D.K==K)].set_index('user')[m]; d=(x-y).dropna().values
    bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]; return f'{a} - {b} (K={K}): {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]'
out=[]
for K in (5,20):
    for a,b in [('hybrid knn+topic','implicit-knn'),('hybrid knn+value','implicit-knn'),('hybrid knn+shuffled topic','implicit-knn'),('explicit-topic','implicit-knn'),('explicit-value','implicit-knn'),
                ('explicit-topic','explicit-topic (shuffled user)'),('explicit-value','explicit-topic'),('recency + hybrid knn+topic','recency + implicit-knn')]:
        out.append(paired(a,b,K))
print('\n'.join(out)); open('results/rankB_paired.txt','w').write('\n'.join(out))
# direct LLM ranking: hit@5 (chance 0.25)
L=pd.DataFrame([json.loads(l) for l in open('data/derived/llm_direct.jsonl')]); L['hit5']=[p in t for p,t in zip(L.pos,L.top)]; L['valid']=L.top.str.len()==5
print(L.groupby('mode').agg(n=('hit5','size'),valid=('valid','mean'),hit5=('hit5','mean')))
