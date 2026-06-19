"""Multitask DAS: accuracy vs k, faceted by task category. Faint per-task lines +
category-mean (stat_summary), learned vs random. Reproducible; run on sc.
    uv run python plots/plot_mtdas_acc.py
"""
import pickle
import numpy as np
import pandas as pd
_SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
def log_superscript(breaks):
    return ['10' + str(int(round(np.log10(b)))).translate(_SUP) for b in breaks]
from plotnine import (ggplot, aes, geom_line, stat_summary, facet_wrap, scale_x_log10,
                      scale_color_manual, labs, theme_set, theme, theme_bw,
                      element_text, element_line, element_blank)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(2.4, 1.7),  # ~70% — LaTeX upscales so text/lines feel larger
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

CAT = {}
for t in ['agr_gender','agr_sv_num_subj-relc','agr_sv_num_obj-relc','agr_sv_num_pp',
          'agr_refl_num_subj-relc','agr_refl_num_obj-relc','agr_refl_num_pp']: CAT[t]='Agreement'
for t in ['npi_any_subj-relc','npi_any_obj-relc','npi_ever_subj-relc','npi_ever_obj-relc']: CAT[t]='Licensing'
for t in ['garden_mvrr','garden_mvrr_mod','garden_npz_obj','garden_npz_obj_mod',
          'garden_npz_v-trans','garden_npz_v-trans_mod']: CAT[t]='Garden path'
for t in ['gss_subord','gss_subord_subj-relc','gss_subord_obj-relc','gss_subord_pp']: CAT[t]='Gross syntactic state'
for t in ['cleft','cleft_mod','filler_gap_embed_3','filler_gap_embed_4','filler_gap_hierarchy',
          'filler_gap_obj','filler_gap_pp','filler_gap_subj']: CAT[t]='Long-distance'
CATS = ['Agreement','Licensing','Garden path','Gross syntactic state','Long-distance']

d = pickle.load(open('results/pythia1b_multitask_das_acc.pkl','rb'))
rows = []
for tk, ev in d['per_task'].items():
    t = tk.replace('syntaxgym/',''); ks = ev['ks']
    for order, key in [('learned','learned_acc'),('random','random_acc')]:
        for k, a in zip(ks, ev[key]):
            rows.append({'task':t,'category':CAT[t],'k':k,'order':order,'accuracy':a})
df = pd.DataFrame(rows)
df['category'] = pd.Categorical(df['category'], CATS, ordered=True)
df['order'] = pd.Categorical(df['order'], ['learned','random'], ordered=True)
df['to'] = df['task'] + '_' + df['order'].astype(str)

p = (ggplot(df, aes('k','accuracy', color='order'))
     + geom_line(aes(group='to'), alpha=0.18, size=0.35)
     + stat_summary(fun_y=np.mean, geom='line', size=0.45)
     + stat_summary(geom='pointrange', fun_data='mean_se', size=0.25, fatten=2)
     + facet_wrap('~category', nrow=1)
     + scale_x_log10(labels=log_superscript)
     + scale_color_manual(values={'learned': '#e41a1c', 'random': '#999999'})
     + labs(x='Subspace dims $k$', y='Accuracy', color='')
     + theme(figure_size=(5.5, 1.6)))
p.save('plots/mtdas_acc.pdf', dpi=300)
p.save('plots/mtdas_acc.png', dpi=150)
print('saved plots/mtdas_acc.{pdf,png}; rows', len(df))
