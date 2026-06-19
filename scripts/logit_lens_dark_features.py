"""Logit lens for every 'Top in layer' (dark-red) DAS feature in the task x feature plot:
per (task, layer) the highest-ranked feature among the task's top-32, unembedded (logit
lens = embed_out @ direction) to top-5/bottom-5 tokens. Saves results/mtdas_dark_feature_logitlens.json.
Run on sc:  uv run python scripts/logit_lens_dark_features.py
"""
import pickle, glob, json
import numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer

R='results'
d=pickle.load(open(f'{R}/pythia1b_multitask_das.pkl','rb'))
dd=d['das_dim']
rots={int(L):d['das_rotations'][L].float() for L in d['das_rotations']}   # per-layer [2048,64]
feats=d['feature_vectors']
EXCLUDE={'syntaxgym/filler_gap_embed_4','syntaxgym/agr_gender'}   # matches the plot
tasks=[t for t in d['tasks'] if t not in EXCLUDE]
N=32

f=glob.glob('/nlp/scr/aryaman/.cache/huggingface/hub/models--EleutherAI--pythia-1b/snapshots/*/model.safetensors')[0]
with safe_open(f, framework='pt') as st:
    WU=st.get_tensor('embed_out.weight').float()           # unembed [vocab, 2048]
    WE=st.get_tensor('gpt_neox.embed_in.weight').float()   # input embed [vocab, 2048]
tok=AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')
decode=lambda idxs:[tok.decode([int(i)]) for i in idxs]

out={}
for t in tasks:
    fv=np.asarray(feats[t]); order=np.argsort(-fv)
    rank={int(fi):r for r,fi in enumerate(order)}
    sel=order[:N]
    bylayer={}   # layer -> feature idx with min within-task rank
    for fi in sel:
        L=int(fi)//dd
        if L not in bylayer or rank[int(fi)]<rank[int(bylayer[L])]:
            bylayer[L]=int(fi)
    entries=[]
    for L in sorted(bylayer):
        fi=bylayer[L]; dim=fi%dd
        direction=rots[L][:,dim]
        lg=WU@direction          # logit lens (unembed)
        le=WE@direction          # embed lens (input embeddings)
        entries.append({'layer':L,'dim':int(dim),'rank_in_task':int(rank[fi]),
                        'score':float(fv[fi]),
                        'top5':decode(lg.topk(5).indices.tolist()),
                        'bottom5':decode(lg.topk(5,largest=False).indices.tolist()),
                        'top5_embed':decode(le.topk(5).indices.tolist()),
                        'bottom5_embed':decode(le.topk(5,largest=False).indices.tolist())})
    out[t.replace('syntaxgym/','')]=entries
json.dump(out, open(f'{R}/mtdas_dark_feature_logitlens.json','w'), indent=2)
print('saved', f'{R}/mtdas_dark_feature_logitlens.json',
      '| tasks', len(out), '| dark features', sum(len(v) for v in out.values()))
