"""Task x feature heatmap (shared-basis layer,dim features), but per task select the
TOP-N features where N = #features needed to reach >=90% accuracy in the power-of-2 eval.
Cells colored by within-task rank (0 = top feature). One plot. Run on sc."""
import pickle
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

R=Path('results'); OUT=Path('paper/figs'); OUT.mkdir(parents=True,exist_ok=True)
d=pickle.load(open(R/'pythia1b_multitask_das.pkl','rb'))
acc=pickle.load(open(R/'pythia1b_multitask_das_acc.pkl','rb'))['per_task']
tasks=d['tasks']; feats=d['feature_vectors']
short=lambda t: t.replace('syntaxgym/','')

# N[task] = smallest power-of-2 k with learned_acc >= 0.90 (fallback: max k)
def k90(t):
    ev=acc[t]; 
    for k,a in zip(ev['ks'],ev['learned_acc']):
        if a>=0.9: return k
    return ev['ks'][-1]
N={t:k90(t) for t in tasks}

F=np.stack([np.asarray(feats[t]) for t in tasks])  # [29, nfeat]
nfeat=F.shape[1]
rank=np.empty_like(F); sel=np.zeros_like(F,bool)
for i,t in enumerate(tasks):
    order=np.argsort(-F[i]); rank[i,order]=np.arange(nfeat)
    sel[i,order[:min(N[t],nfeat)]]=True   # top-N[task]
keep=sel.sum(0)>0
S=sel[:,keep]; Rk=rank[:,keep]
A=np.where(S,Rk,np.nan); vmax=float(np.nanmax(A))

M=np.array(d['overlap_matrix']); np.fill_diagonal(M,1.0)
D=1-M; D=(D+D.T)/2; np.fill_diagonal(D,0)
torder=dendrogram(linkage(squareform(D,checks=False),'average'),no_plot=True)['leaves']
forder=dendrogram(linkage(S.T.astype(float),'average'),no_plot=True)['leaves'] if S.shape[1]>2 else list(range(S.shape[1]))
A=A[np.ix_(torder,forder)]

fig,ax=plt.subplots(figsize=(7.5,5))
cmap=plt.get_cmap('viridis_r').copy(); cmap.set_bad('#f0f0f0')
im=ax.imshow(A,aspect='auto',cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest')
ax.set_yticks(range(len(torder))); ax.set_yticklabels([f'{short(tasks[i])} (N={N[tasks[i]]})' for i in torder],fontsize=5)
ax.set_xlabel(f'shared-basis features (clustered; {keep.sum()} used)')
ax.set_title('Top-N features per task (N = # for >=90% acc), colored by within-task rank')
cb=fig.colorbar(im,ax=ax,fraction=0.025); cb.set_label('rank within task (0=top)',fontsize=7)
plt.tight_layout(); fig.savefig(OUT/'eda_task_feature_acc90.png',dpi=150)
print(f'Saved eda_task_feature_acc90.png; features used={keep.sum()} vmax={vmax:.0f}')
