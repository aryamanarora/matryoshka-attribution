import pickle
import numpy as np, pandas as pd
from plotnine import (ggplot, aes, geom_line, stat_summary, facet_wrap, scale_x_log10,
                      scale_color_brewer, labs, theme_bw, theme_set, theme,
                      element_text, element_line, element_blank)
theme_set(theme_bw(base_size=9)+theme(text=element_text(family='Inter'),
          panel_grid_minor=element_blank(), strip_background=element_blank()))

CAT={}
for t in ['agr_gender','agr_sv_num_subj-relc','agr_sv_num_obj-relc','agr_sv_num_pp',
          'agr_refl_num_subj-relc','agr_refl_num_obj-relc','agr_refl_num_pp']: CAT[t]='Agreement'
for t in ['npi_any_subj-relc','npi_any_obj-relc','npi_ever_subj-relc','npi_ever_obj-relc']: CAT[t]='Licensing'
for t in ['garden_mvrr','garden_mvrr_mod','garden_npz_obj','garden_npz_obj_mod',
          'garden_npz_v-trans','garden_npz_v-trans_mod']: CAT[t]='Garden path effects'
for t in ['gss_subord','gss_subord_subj-relc','gss_subord_obj-relc','gss_subord_pp']: CAT[t]='Gross syntactic state'
for t in ['cleft','cleft_mod','filler_gap_embed_3','filler_gap_embed_4','filler_gap_hierarchy',
          'filler_gap_obj','filler_gap_pp','filler_gap_subj']: CAT[t]='Long-distance'
CATS=['Agreement','Licensing','Garden path effects','Gross syntactic state','Long-distance']

d=pickle.load(open('results/pythia1b_multitask_das_acc.pkl','rb'))
rows=[]
for tk,ev in d['per_task'].items():
    t=tk.replace('syntaxgym/',''); ks=ev['ks']
    for order,key in [('learned','learned_acc'),('random','random_acc')]:
        for k,a in zip(ks,ev[key]):
            rows.append({'task':t,'category':CAT[t],'k':k,'order':order,'accuracy':a})
df=pd.DataFrame(rows)
df['category']=pd.Categorical(df['category'],CATS,ordered=True)
df['order']=pd.Categorical(df['order'],['learned','random'],ordered=True)
df['to']=df['task']+'_'+df['order'].astype(str)

p=(ggplot(df,aes('k','accuracy',color='order'))
   + geom_line(aes(group='to'),alpha=0.18,size=0.4)
   + stat_summary(fun_y=np.mean,geom='line',size=1.1)
   + facet_wrap('~category',nrow=1)
   + scale_x_log10()
   + scale_color_brewer(type='qual',palette='Set1')
   + labs(x='k (subspace dims, log)',y='accuracy',color='',
          title='Multitask DAS: accuracy vs k by task category (bold = category mean)')
   + theme(figure_size=(14,2.8),legend_position='top'))
p.save('paper/figs/mtdas_acc_categories.pdf',dpi=300)
p.save('paper/figs/mtdas_acc_categories.png',dpi=150)
print('saved; rows',len(df))
