"""Fairness check for the hybrid: pick the explicit weight w on half of the users, test on the other half (K=20)."""
import numpy as np, pandas as pd, importlib.util
spec=importlib.util.spec_from_file_location('ev','analysis/12_rank_eval.py'); ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
S=pd.read_parquet('data/derived/llm_statements.parquet'); SE=np.load('data/derived/llm_statements_emb.npy').astype(np.float32); rng=np.random.default_rng(2)
g=S[(S.K==20)&(S.kind=='topic')]; prof={u:SE[ix.values] for u,ix in g.groupby('user').groups.items()}; us=sorted(prof); rng.shuffle(us); A,Bu=us[:300],us[300:]
res={}
for w in [0,0.05,0.1,0.2,0.3,0.5]:
    sc=ev.Hybrid(ev.ImpKnn(),ev.Profile(prof),w); res[w]=(pd.DataFrame(ev.evaluate(sc,'h',20,subset=A)).auc.mean(),pd.DataFrame(ev.evaluate(sc,'h',20,subset=Bu)).set_index('user').auc)
    print(w,'tune',round(res[w][0],4),'test',round(res[w][1].mean(),4))
best=max(res,key=lambda w:res[w][0]); d=(res[best][1]-res[0][1]).values; bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]
out=f'best w on tuning half = {best}; held-out dAUC vs implicit-knn = {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]'; print(out); open('results/rankB_hybrid_weight.txt','w').write(out)
