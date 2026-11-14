"""A8 step 2: embed a8_texts with local bge-small-en-v1.5 and keep only the cosine similarity of each post to the centroid of
English Science-feed posts and of English News-feed posts (same centroids as 18_rq7_writing.py).
usage: CUDA_VISIBLE_DEVICES=<gpu> python 28_a8_embed_sim.py <part> <nparts>  -> data/derived/a8_sim_part<part>.npy"""
import sys, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
O='data/derived'; part,nparts=int(sys.argv[1]),int(sys.argv[2]); M='/mnt/disk1/hanqim2/models/bge-small-en-v1.5'
P=pd.read_parquet(f'{O}/rank_posts.parquet',columns=['feed','lang']); PE=np.load(f'{O}/rank_posts_emb.npy').astype(np.float32); C=[]
for f in ['Science','News']:
    c=PE[((P.feed==f)&(P.lang=='eng')).values].mean(0); C.append(c/np.linalg.norm(c)); print(f,'centroid from',int(((P.feed==f)&(P.lang=='eng')).sum()),'posts')
C=torch.tensor(np.stack(C)).cuda()
T=pd.read_parquet(f'{O}/a8_texts.parquet',columns=['text']).text.iloc[part::nparts].tolist()
tok=AutoTokenizer.from_pretrained(M); model=AutoModel.from_pretrained(M).half().cuda().eval()
order=np.argsort([len(t) for t in T])[::-1]; out=np.empty((len(T),2),np.float32)
with torch.no_grad():
    for i in range(0,len(T),1024):
        idx=order[i:i+1024]; x=tok([T[j] for j in idx],padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda')
        e=torch.nn.functional.normalize(model(**x).last_hidden_state[:,0].float(),dim=-1); out[idx]=(e@C.T).cpu().numpy()
        if (i//1024)%200==0: print(i,len(T),flush=True)
np.save(f'{O}/a8_sim_part{part}.npy',out); print('done',out.shape,out.mean(0))
