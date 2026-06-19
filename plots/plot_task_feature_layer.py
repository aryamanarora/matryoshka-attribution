"""Task x feature heatmap; per task top-32 features. Features ordered along X by layer
(dashed vlines, layer# ticks). Tasks in CausalGym order, faceted/divided by category.
Two reds: most-important feature per (task,layer) vs the rest. Excludes dense outliers."""
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from plotnine import (ggplot, aes, geom_tile, geom_vline, facet_grid, scale_x_continuous,
                      scale_fill_manual, labs, theme_set, theme, theme_bw, element_text,
                      element_blank, element_rect)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color='#000', family='Inter'),
        axis_title=element_text(size=7),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
        panel_spacing_y=0.008,
        strip_background=element_blank(), strip_text=element_text(size=6),
        legend_title=element_text(size=7), legend_text=element_text(size=6),
        legend_key_size=8, legend_position='top',
    )
)

CATS=['Agreement','Licensing','Garden','GSS','Long']
CAT={}
for t in ['agr_gender','agr_sv_num_subj-relc','agr_sv_num_obj-relc','agr_sv_num_pp','agr_refl_num_subj-relc','agr_refl_num_obj-relc','agr_refl_num_pp']: CAT[t]='Agreement'
for t in ['npi_any_subj-relc','npi_any_obj-relc','npi_ever_subj-relc','npi_ever_obj-relc']: CAT[t]='Licensing'
for t in ['garden_mvrr','garden_mvrr_mod','garden_npz_obj','garden_npz_obj_mod','garden_npz_v-trans','garden_npz_v-trans_mod']: CAT[t]='Garden'
for t in ['gss_subord','gss_subord_subj-relc','gss_subord_obj-relc','gss_subord_pp']: CAT[t]='GSS'
for t in ['cleft','cleft_mod','filler_gap_embed_3','filler_gap_embed_4','filler_gap_hierarchy','filler_gap_obj','filler_gap_pp','filler_gap_subj']: CAT[t]='Long'

R=Path('results')
d=pickle.load(open(R/'pythia1b_multitask_das.pkl','rb'))
EXCLUDE={'syntaxgym/filler_gap_embed_4','syntaxgym/agr_gender'}
tasks=[t for t in d['tasks'] if t not in EXCLUDE]   # CausalGym order
feats=d['feature_vectors']; dd=d['das_dim']; short=lambda t: t.replace('syntaxgym/','')
N=32

F=np.stack([np.asarray(feats[t]) for t in tasks]); nfeat=F.shape[1]
rank=np.empty_like(F); sel=np.zeros_like(F,bool)
for i,t in enumerate(tasks):
    order=np.argsort(-F[i]); rank[i,order]=np.arange(nfeat); sel[i,order[:min(N,nfeat)]]=True
keep=np.where(sel.sum(0)>0)[0]; S=sel[:,keep]
layer_col=keep//dd; dim_col=keep%dd
fo=np.lexsort((dim_col, layer_col)); layer_sorted=layer_col[fo]

names=[short(t) for t in tasks]
rows=[]
for ti,t in enumerate(tasks):
    bylayer=defaultdict(list)
    for col in range(len(keep)):
        if S[ti,col]: bylayer[int(layer_col[col])].append(col)
    topcol={L:min(cols,key=lambda c:rank[ti,keep[c]]) for L,cols in bylayer.items()}
    for x,col in enumerate(fo):
        if not S[ti,col]: continue
        kind='Top in layer' if col==topcol[int(layer_col[col])] else 'Other'
        rows.append({'task':names[ti],'category':CAT[short(t)],'x':x,'kind':kind})
df=pd.DataFrame(rows)
df['task']=pd.Categorical(df['task'],categories=names[::-1],ordered=True)   # CausalGym order, top->bottom
df['category']=pd.Categorical(df['category'],categories=CATS,ordered=True)
df['kind']=pd.Categorical(df['kind'],categories=['Top in layer','Other'],ordered=True)

bnd=[x-0.5 for x in range(1,len(layer_sorted)) if layer_sorted[x]!=layer_sorted[x-1]]
centres={}
for x,L in enumerate(layer_sorted): centres.setdefault(int(L),[]).append(x)
breaks=[np.mean(v) for L,v in sorted(centres.items())]; labs_x=[str(L) for L in sorted(centres)]

p=(ggplot(df,aes('x','task',fill='kind'))
   + geom_tile()
   + geom_vline(xintercept=bnd, linetype='dashed', size=0.25, color='#777777')
   + facet_grid('category ~ .', scales='free_y', space='free_y')
   + scale_x_continuous(expand=(0,0), breaks=breaks, labels=labs_x)
   + scale_fill_manual(values={'Top in layer':'#cb181d','Other':'#fcae91'})
   + labs(x='Layer', y='', fill='')
   + theme(figure_size=(6.5,3.8),
           panel_background=element_rect(fill='#f0f0f0'),
           axis_text_y=element_text(size=5)))
p.save('plots/task_feature_layer.pdf', dpi=300)
p.save('plots/task_feature_layer.png', dpi=150)
print('saved')
