"""Line B: explicit profiles and direct LLM ranking with a local Qwen3-8B (vLLM, port 9111, thinking off, temperature 0).
 profiles: for 600 users x K in {5,20} x kind in {topic,value}: 6 short statements inferred from the K most recent elicitation likes.
 direct  : for 300 users x 5 impressions x {zero-shot, few-shot K=20}: LLM returns the 5 candidates the user is most likely to like.
Outputs data/derived/llm_profiles.jsonl, data/derived/llm_direct.jsonl (resumable)."""
import json, pickle, re, os, sys, numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
O='data/derived'; rng=np.random.default_rng(9)
P=pd.read_parquet(f'{O}/rank_posts.parquet'); TXT=P.text.str.replace(r'\s+',' ',regex=True).str.slice(0,300).values
task=pickle.load(open(f'{O}/rank_task.pkl','rb')); users=sorted(task); U600=[int(u) for u in rng.choice(users,600,replace=False)]; U300=U600[:300]
def chat(prompt,max_tokens=400):
    r=requests.post('http://localhost:9111/v1/chat/completions',json=dict(model='qwen3-8b',messages=[{'role':'user','content':prompt}],temperature=0,max_tokens=max_tokens,
        chat_template_kwargs={'enable_thinking':False}),timeout=300)
    return r.json()['choices'][0]['message']['content']
INS={'topic':'Write exactly 6 short statements (at most 15 words each) describing the TOPICS and subject areas this user is interested in.',
     'value':'Write exactly 6 short statements (at most 15 words each) describing what this user VALUES in a post: the qualities, tone, purpose or style that make them like it. Do not list topics.'}
def prof(job):
    u,K,kind=job; h=task[u]['hist'][-K:]
    p='Below are posts that one Bluesky user liked.\n'+'\n'.join(f'- {TXT[i]}' for i in h)+f'\n\n{INS[kind]}\nReturn only a JSON list of 6 strings.'
    try:
        out=chat(p); m=re.search(r'\[.*\]',out,re.S); st=[str(x) for x in json.loads(m.group(0))][:8]
    except Exception as e: st=[]
    return dict(user=u,K=K,kind=kind,statements=st)
def direct(job):
    u,j,mode=job; pos,neg=task[u]['imps'][j]; c=[pos]+list(neg); order=np.random.default_rng(u*100+j).permutation(20); c=[c[i] for i in order]
    ctx='' if mode=='zero' else 'Posts this Bluesky user liked recently:\n'+'\n'.join(f'- {TXT[i]}' for i in task[u]['hist'][-20:])+'\n\n'
    p=ctx+('Candidate posts:\n' if mode!='zero' else 'Candidate posts from a science feed on Bluesky:\n')+'\n'.join(f'[{k+1}] {TXT[i]}' for k,i in enumerate(c))+ \
      ('\n\nWhich 5 candidates is this user most likely to like? ' if mode!='zero' else '\n\nWhich 5 candidates is a typical user most likely to like? ')+'Return only a JSON list of 5 candidate numbers, most likely first.'
    try:
        out=chat(p,80); top=[int(x) for x in json.loads(re.search(r'\[.*?\]',out,re.S).group(0))][:5]
    except Exception: top=[]
    posk=int(np.where(order==0)[0][0])+1
    return dict(user=u,imp=j,mode=mode,top=top,pos=posk)
def run(fn,jobs,path,key):
    done=set()
    if os.path.exists(path): done={tuple(json.loads(l)[k] for k in key) for l in open(path)}
    jobs=[j for j in jobs if tuple(j) not in done]; print(path,'todo',len(jobs),flush=True)
    with open(path,'a') as f, ThreadPoolExecutor(48) as ex:
        for i,r in enumerate(ex.map(fn,jobs)):
            f.write(json.dumps(r)+'\n')
            if i%500==0: print(i,flush=True)
run(prof,[(u,K,k) for u in U600 for K in (5,20) for k in ('topic','value')],f'{O}/llm_profiles.jsonl',('user','K','kind'))
run(direct,[(u,j,m) for u in U300 for j in range(min(5,len(task[u]['imps']))) for m in ('zero','few')],f'{O}/llm_direct.jsonl',('user','imp','mode'))
