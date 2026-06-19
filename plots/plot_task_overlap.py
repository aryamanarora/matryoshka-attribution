"""Pairwise task-similarity heatmap: # of top-32 DAS features shared between every pair of
tasks. Tasks in CausalGym order, dashed lines at category boundaries. Saves
plots/task_overlap.pdf. Run on sc: uv run python plots/plot_task_overlap.py"""
import pickle
from pathlib import Path
import numpy as np, pandas as pd
from plotnine import (ggplot, aes, geom_tile, geom_text, geom_vline, geom_hline,
                      scale_x_discrete, scale_y_discrete, scale_fill_gradient, labs,
                      theme_set, theme, theme_bw, element_text, element_blank, coord_equal)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color='#000000', family='Inter'),
        axis_title=element_text(size=7),
        axis_text=element_text(size=5),
        axis_text_x=element_text(size=5, rotation=90, hjust=1.0, vjust=0.5),
        panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
        legend_title=element_text(size=7), legend_text=element_text(size=6),
        legend_key_size=8, legend_position='right',
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
EXCLUDE=set()  # include the 2 outlier tasks
tasks=[t for t in d['tasks'] if t not in EXCLUDE]   # CausalGym order
feats=d['feature_vectors']; short=lambda t: t.replace('syntaxgym/','')
N=32

names=[short(t) for t in tasks]
top={t:set(np.argsort(-np.asarray(feats[t]))[:N].tolist()) for t in tasks}

rows=[]
for ti,a in enumerate(tasks):
    for tj,b in enumerate(tasks):
        rows.append({'a':names[ti],'b':names[tj],'n':len(top[a]&top[b])})
df=pd.DataFrame(rows)
df['a']=pd.Categorical(df['a'],categories=names,ordered=True)            # x: left->right
df['b']=pd.Categorical(df['b'],categories=names[::-1],ordered=True)      # y: top->bottom

# category-boundary lines (between consecutive tasks in different categories)
bnd=[i for i in range(1,len(tasks)) if CAT[short(tasks[i])]!=CAT[short(tasks[i-1])]]
vb=[i+0.5 for i in bnd]                       # x boundaries
hb=[len(tasks)-i+0.5 for i in bnd]            # y boundaries (reversed axis)

p=(ggplot(df,aes('a','b',fill='n'))
   + geom_tile(color='white', size=0.3)
   + geom_text(aes(label='n'), size=3.2, color='#000000')
   + geom_vline(xintercept=vb, color='#000000', size=0.4)
   + geom_hline(yintercept=hb, color='#000000', size=0.4)
   + scale_x_discrete(expand=(0,0))
   + scale_y_discrete(expand=(0,0))
   + scale_fill_gradient(low='#ffffff', high='#cb181d', name='Shared')
   + labs(x='', y='')
   + coord_equal()
   + theme(figure_size=(6.5,6.0)))
p.save('plots/task_overlap.pdf', dpi=300)
p.save('plots/task_overlap.png', dpi=150)
print('saved | tasks:', len(tasks), '| max off-diag:',
      df[df['a'].astype(str)!=df['b'].astype(str)]['n'].max())
