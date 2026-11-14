"""Line C, second pass: does "an inferred explicit profile beats raw history" survive when the simulated world is NOT built in
the scorer's own embedding space and when choices also depend on things a topic profile cannot express?
(Borrowing BELA's separation between the model that defines ground truth and the model that uses it.)
Synthetic users act on real Science-feed posts. Utility of a candidate within an impression of 20 posts:
    U = S * [ w_topic * z(topic match) + w_author * z(favourite author) + w_recency * z(post time) ] + Gumbel noise
z() standardises within the impression. Conditions:
    C0  topic match = bge cosine (same space as the scorer), weights (1, 0, 0)        -> the first pass, re-parametrised
    C1  topic match = TF-IDF cosine (lexical, different from the scorer), (1, 0, 0)    -> encoder mismatch only
    C2  TF-IDF topic, weights (0.5, 0.3, 0.2)
    C3  TF-IDF topic, weights (0.25, 0.5, 0.25)                                        -> mostly non-topical, as real likes seem to be
All rankers score with bge embeddings, as on the real data: implicit-knn, profile inferred by Qwen3-8B from K liked posts,
oracle (planted) profile, another user's inferred profile, hybrids, and implicit-knn + author counts."""
import json, re, os, subprocess, numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
O='data/derived'; rng=np.random.default_rng(32); NU=300; NIMP=40; SIG=2.5
P=pd.read_parquet(f'{O}/rank_posts.parquet'); EMB=np.load(f'{O}/rank_posts_emb.npy').astype(np.float32)
ok=np.flatnonzero(((P.feed=='Science')&(P.lang=='eng')&(P.text.str.len()>=40)).values); TXT=P.text.str.replace(r'\s+',' ',regex=True).str.slice(0,300).values
AUTH=P.author.values; TS=P.ts.values.astype('datetime64[m]').astype(np.int64).astype(np.float64)
S=pd.read_parquet(f'{O}/llm_statements.parquet'); SE=np.load(f'{O}/llm_statements_emb.npy').astype(np.float32); pool=np.flatnonzero(((S.K==20)&(S.kind=='topic')).values)
tf=TfidfVectorizer(sublinear_tf=True,stop_words='english',min_df=3).fit(TXT[ok]); XP=tf.transform(TXT).tocsr(); XS=tf.transform(S.text.values).tocsr()
ac=pd.Series(AUTH[ok]).value_counts(); ap=(ac/ac.sum()).values; authors=ac.index.values
z=lambda x:(x-x.mean())/(x.std()+1e-9)
COND={'C0 bge topic only':('bge',(1,0,0)),'C1 lexical topic only':('tfidf',(1,0,0)),'C2 lexical 0.5 / author 0.3 / recency 0.2':('tfidf',(.5,.3,.2)),'C3 lexical 0.25 / author 0.5 / recency 0.25':('tfidf',(.25,.5,.25))}
def simulate(kind,w):
    U=[]
    for u in range(NU):
        st=rng.choice(pool,3,replace=False); fav=set(rng.choice(authors,40,replace=False,p=ap).tolist())
        def like(c):
            t=(EMB[c]@SE[st].T).max(1) if kind=='bge' else np.asarray((XP[c]@XS[st].T).max(1).todense()).ravel()
            a=np.array([AUTH[i] in fav for i in c],float)
            return c[np.argmax(SIG*(w[0]*z(t)+w[1]*z(a)+w[2]*z(TS[c]))+rng.gumbel(size=len(c)))]
        hist=[like(rng.choice(ok,20,replace=False)) for _ in range(20)]; imps=[]
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
def auc(score_fn,imps):
    a=[]
    for p,neg in imps:
        c=np.array([p]+neg); s=score_fn(c); r=1+(s[1:]>s[0]).sum()+0.5*(s[1:]==s[0]).sum(); a.append(1-(r-1)/19)
    return np.mean(a)
sims={k:simulate(*v) for k,v in COND.items()}
jobs=[(k,i,K) for k in sims for i in range(NU) for K in (5,20)]
sp=f'{O}/synthetic2_statements.parquet'
if not os.path.exists(sp):
    with ThreadPoolExecutor(48) as ex: outs=list(ex.map(lambda j: infer(sims[j[0]][j[1]]['hist'][-j[2]:]),jobs))
    pd.DataFrame([(k,i,K,s) for (k,i,K),st in zip(jobs,outs) for s in st],columns=['cond','i','K','text']).to_parquet(sp)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OPENSSL_CONF='/usr/lib/ssl/openssl.cnf',HF_HUB_OFFLINE='1')
    subprocess.run(['/mnt/disk1/hanqim2/.conda/envs/specjev/bin/python','analysis/10_embed.py',sp,'text',f'{O}/synthetic2_statements_emb.npy'],env=env,check=True,capture_output=True)
# NOTE: the simulation is re-drawn from the same seed on every run, so the stored statements match the simulated histories.
D=pd.read_parquet(sp); DE=np.load(f'{O}/synthetic2_statements_emb.npy').astype(np.float32); rows=[]
for k,U in sims.items():
    for K in (5,20):
        g=D[(D.cond==k)&(D.K==K)]; inf={i:DE[ix.values] for i,ix in g.groupby('i').groups.items()}
        for i in range(NU):
            if i not in inf: continue
            hist=U[i]['hist'][-K:]; Hm=EMB[hist]; V=SE[U[i]['st']]; I=inf[i]; J=inf[(i+1)%NU] if (i+1)%NU in inf else I
            hc=pd.Series(AUTH[hist]).value_counts(); knn=lambda c: (EMB[c]@Hm.T).max(1); ex=lambda Wm: (lambda c: (EMB[c]@Wm.T).max(1))
            au=lambda c: np.array([hc.get(AUTH[x],0) for x in c],float); cen=lambda Wm: Wm.mean(0)/np.linalg.norm(Wm.mean(0)); im=U[i]['imps']
            rows.append(dict(cond=k,K=K,i=i,implicit=auc(knn,im),inferred=auc(ex(I),im),oracle=auc(ex(V),im),shuffled=auc(ex(J),im),
                             hyb_inferred=auc(lambda c: z(knn(c))+z(ex(I)(c)),im),hyb_oracle=auc(lambda c: z(knn(c))+z(ex(V)(c)),im),
                             implicit_author=auc(lambda c: z(knn(c))+z(au(c)),im),fidelity=float(cen(I)@cen(V))))
R=pd.DataFrame(rows); R.to_csv('results/synC2_users.csv',index=False); out=[]
for (k,K),g in R.groupby(['cond','K']):
    b=lambda d: np.percentile([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)],[2.5,97.5])
    r=dict(cond=k,K=K,n=len(g),**{c:g[c].mean() for c in ['implicit','inferred','oracle','shuffled','hyb_inferred','hyb_oracle','implicit_author']})
    for nm,d in [('inferred-implicit',g.inferred-g.implicit),('hyb_inferred-implicit',g.hyb_inferred-g.implicit),('hyb_oracle-implicit',g.hyb_oracle-g.implicit),('implicit_author-implicit',g.implicit_author-g.implicit)]:
        lo,hi=b(d.values); r[nm]=d.mean(); r[nm+'_lo']=lo; r[nm+'_hi']=hi
    r['rho_fidelity_inferredAUC']=spearmanr(g.fidelity,g.inferred)[0]; out.append(r)
T=pd.DataFrame(out); T.to_csv('results/synC2_summary.csv',index=False); pd.set_option('display.width',250)
print(T[['cond','K','n','implicit','inferred','oracle','shuffled','hyb_inferred','implicit_author']].round(3).to_string(index=False))
for r in T.itertuples(): print(f"{r.cond} K={r.K}: inferred-implicit {r._11:+.3f} [{r._12:+.3f},{r._13:+.3f}] | hybrid-implicit {r._14:+.3f} [{r._15:+.3f},{r._16:+.3f}] | implicit+author - implicit {r._20:+.3f} [{r._21:+.3f},{r._22:+.3f}] | rho(fidelity, inferred AUC) {r.rho_fidelity_inferredAUC:.2f}")
