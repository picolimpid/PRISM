"""Embed texts with a local bge-small-en-v1.5 (CLS pooling, L2-normalised). usage: 10_embed.py in.parquet text_col out.npy"""
import sys, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
M='/mnt/disk1/hanqim2/models/bge-small-en-v1.5'
tok=AutoTokenizer.from_pretrained(M); model=AutoModel.from_pretrained(M).half().cuda().eval()
d=pd.read_parquet(sys.argv[1]); T=d[sys.argv[2]].fillna('').astype(str).tolist(); out=[]
order=np.argsort([len(t) for t in T])[::-1]
with torch.no_grad():
    for i in range(0,len(T),512):
        b=[T[j][:1000] for j in order[i:i+512]]
        x=tok(b,padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda')
        e=torch.nn.functional.normalize(model(**x).last_hidden_state[:,0].float(),dim=-1); out.append(e.cpu().numpy())
E_=np.concatenate(out); R=np.empty_like(E_); R[order]=E_; np.save(sys.argv[3],R.astype(np.float16)); print(R.shape)
