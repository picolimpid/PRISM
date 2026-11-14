"""Line C (BELA / Mem-PAL style controlled testbed): synthetic users with a KNOWN explicit profile acting on REAL Science-feed posts.
Each synthetic user owns 3 interest statements (sampled from real LLM-inferred topic statements of different real users).
Choice model: utility(post) = max cosine(post, statements)/tau + Gumbel noise; in each impression of 20 real posts the user likes the argmax.
This is a parametric simulator, not an LLM role-play, so the LLM that infers profiles did not also generate the behaviour.
Conditions (history budget K in {5,20}; noise tau in {0.02 low, 0.05 high}):
  implicit-knn | inferred explicit (Qwen3-8B reads the K liked posts) | oracle explicit (the planted statements) | hybrids | shuffled.
Fidelity = cosine between inferred-profile centroid and planted-profile centroid; we test fidelity -> utility at the user level."""
import json, re, os, numpy as np, pandas as pd, requests, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr
O='data/derived'; rng=np.random.default_rng(21)
P=pd.read_parquet(f'{O}/rank_posts.parquet'); EMB=np.load(f'{O}/rank_posts_emb.npy').astype(np.float32)
ok=np.flatnonzero((P.feed=='Science')&(P.lang=='eng')&(P.text.str.len()>=40)); TXT=P.text.str.replace(r'\s+',' ',regex=True).str.slice(0,300).values
S=pd.read_parquet(f'{O}/llm_statements.parquet'); SE=np.load(f'{O}/llm_statements_emb.npy').astype(np.float32); pool=np.flatnonzero((S.K==20)&(S.kind=='topic'))
NU=300; NIMP=40
def simulate(tau):
    U=[]
    for u in range(NU):
        st=rng.choice(pool,3,replace=False); V=SE[st]
        like=lambda c: c[np.argmax((EMB[c]@V.T).max(1)/tau+rng.gumbel(size=len(c)))]
        hist=[like(rng.choice(ok,20,replace=False)) for _ in range(20)]
        imps=[]
        for _ in range(NIMP):
            c=rng.choice(ok,20,replace=False); p=like(c); imps.append((p,[x for x in c if x!=p]))
        U.append(dict(st=st,hist=hist,imps=imps))
    return U
def chat(prompt):
    r=requests.post('http://localhost:9111/v1/chat/completions',json=dict(model='qwen3-8b',messages=[{'role':'user','content':prompt}],temperature=0,max_tokens=300,chat_template_kwargs={'enable_thinking':False}),timeout=300)
    return r.json()['choices'][0]['message']['content']
def infer(h):
    p='Below are posts that one Bluesky user liked.\n'+'\n'.join(f'- {TXT[i]}' for i in h)+'\n\nWrite exactly 3 short statements (at most 15 words each) describing the TOPICS and subject areas this user is interested in.\nReturn only a JSON list of 3 strings.'
    try: return [str(x) for x in json.loads(re.search(r'\[.*\]',chat(p),re.S).group(0))][:3]
    except Exception: return []
def auc(score_fn,U,i):
    a=[]
    for p,neg in U[i]['imps']:
        c=np.array([p]+neg); s=score_fn(c); r=1+(s[1:]>s[0]).sum()+0.5*(s[1:]==s[0]).sum(); a.append(1-(r-1)/19)
    return np.mean(a)
z=lambda x:(x-x.mean())/(x.std()+1e-9)
sims={tau:simulate(tau) for tau in (0.02,0.05)}
jobs=[(tau,i,K) for tau in sims for i in range(NU) for K in (5,20)]
path=f'{O}/synthetic_inferred.jsonl'
if not os.path.exists(path):
    with ThreadPoolExecutor(48) as ex: outs=list(ex.map(lambda j: infer(sims[j[0]][j[1]]['hist'][-j[2]:]),jobs))
    T=[(tau,i,K,s) for (tau,i,K),st in zip(jobs,outs) for s in st]; D=pd.DataFrame(T,columns=['tau','i','K','text']); D.to_parquet(f'{O}/synthetic_statements.parquet')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OPENSSL_CONF='/usr/lib/ssl/openssl.cnf',HF_HUB_OFFLINE='1')
    subprocess.run(['/mnt/disk1/hanqim2/.conda/envs/specjev/bin/python','analysis/10_embed.py',f'{O}/synthetic_statements.parquet','text',f'{O}/synthetic_statements_emb.npy'],env=env,check=True,capture_output=True)
    open(path,'w').write('done')
D=pd.read_parquet(f'{O}/synthetic_statements.parquet'); DE=np.load(f'{O}/synthetic_statements_emb.npy').astype(np.float32)
rows=[]
for tau,U in sims.items():
    for K in (5,20):
        g=D[(D.tau==tau)&(D.K==K)]; inf={i:DE[ix.values] for i,ix in g.groupby('i').groups.items()}
        for i in range(NU):
            if i not in inf: continue
            H=EMB[U[i]['hist'][-K:]]; V=SE[U[i]['st']]; I=inf[i]; J=inf[(i+1)%NU] if (i+1)%NU in inf else I
            knn=lambda c: (EMB[c]@H.T).max(1); ex=lambda W: (lambda c: (EMB[c]@W.T).max(1))
            cen=lambda W: W.mean(0)/np.linalg.norm(W.mean(0))
            rows.append(dict(tau=tau,K=K,i=i,implicit=auc(knn,U,i),inferred=auc(ex(I),U,i),oracle=auc(ex(V),U,i),shuffled=auc(ex(J),U,i),
                hyb_inferred=auc(lambda c: z(knn(c))+z(ex(I)(c)),U,i),hyb_oracle=auc(lambda c: z(knn(c))+z(ex(V)(c)),U,i),hyb_shuffled=auc(lambda c: z(knn(c))+z(ex(J)(c)),U,i),
                fidelity=float(cen(I)@cen(V)),fid_matched=float((I@V.T).max(1).mean())))
R=pd.DataFrame(rows); R.to_csv('results/synC_users.csv',index=False)
T=R.groupby(['tau','K'])[['implicit','inferred','oracle','shuffled','hyb_inferred','hyb_oracle','hyb_shuffled','fidelity']].mean().round(4); print(T.to_string()); T.to_csv('results/synC_summary.csv')
for (tau,K),g in R.groupby(['tau','K']):
    b=lambda d:[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]
    d1=(g.hyb_inferred-g.implicit).values; d2=(g.hyb_oracle-g.implicit).values
    print(f'tau={tau} K={K}: hyb_inferred-implicit {d1.mean():+.4f} [{np.percentile(b(d1),2.5):+.4f},{np.percentile(b(d1),97.5):+.4f}] | hyb_oracle-implicit {d2.mean():+.4f} [{np.percentile(b(d2),2.5):+.4f},{np.percentile(b(d2),97.5):+.4f}]'
          f' | fidelity~inferred AUC rho={spearmanr(g.fidelity,g.inferred)[0]:.3f} | fidelity~(hyb_inferred-implicit) rho={spearmanr(g.fidelity,g.hyb_inferred-g.implicit)[0]:.3f} p={spearmanr(g.fidelity,g.hyb_inferred-g.implicit)[1]:.3g}')
