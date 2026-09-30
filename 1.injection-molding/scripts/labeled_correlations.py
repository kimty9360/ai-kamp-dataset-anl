"""Descriptive correlations of deduplicated labeled data, stratified by product/side."""
from pathlib import Path
import argparse,hashlib,json,zipfile
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_sources import META
ROOT=Path(__file__).resolve().parents[1]


def draw_correlation_heatmap(matrix, title):
 """Annotate coefficients to 2 decimals; undefined constant columns omitted."""
 variable = matrix.columns[matrix.notna().any(axis=0)]
 values = matrix.loc[variable, variable].to_numpy()
 fig, ax = plt.subplots(figsize=(18, 16))
 im = ax.imshow(values, vmin=-1, vmax=1, cmap='RdBu_r')
 ax.set_xticks(range(len(variable)), variable, rotation=90, fontsize=9)
 ax.set_yticks(range(len(variable)), variable, fontsize=9)
 for i in range(len(variable)):
  for j in range(len(variable)):
   value = values[i, j]
   label = f'{value:.2f}' if np.isfinite(value) else '—'
   if label == '-0.00': label = '0.00'
   ax.text(j, i, label, ha='center', va='center', fontsize=8,
           color='white' if np.isfinite(value) and abs(value) >= .6 else 'black')
 fig.colorbar(im, ax=ax, shrink=.7)
 ax.set_title(title + '\nValues rounded to 2 decimals; constant columns omitted', fontsize=13)
 fig.tight_layout()
 return fig

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/correlation_v1');out=ap.parse_args().output
 if out.exists():raise FileExistsError('Use a new output directory')
 source=ROOT/'sources/04. Dataset_Molding.zip';sha=hashlib.sha256(source.read_bytes()).hexdigest()
 with zipfile.ZipFile(source) as z:
  with z.open('dataset/labeled_data.csv') as f: raw=pd.read_csv(f)
 assert not raw.drop_duplicates().duplicated('_id').any()
 d=raw.drop_duplicates('_id').copy();features=[c for c in d if c not in META]
 assert len(features)==36 and all(pd.api.types.is_numeric_dtype(d[c]) for c in features)
 d['product']=d.PART_NAME.str.extract(r'^(CN7|RG3|JX1|SP2)');d['side']=d.PART_NAME.str.strip().str.extract(r'(LH|RH)$');d['date']=pd.to_datetime(d.TimeStamp).dt.date;d['defect']=d.PassOrFail.map({'Y':0,'N':1})
 out.mkdir(parents=True);(out/'figures').mkdir();(out/'matrices').mkdir()
 coverage=raw.groupby(['PART_NAME','EQUIP_CD','EQUIP_NAME']).size().rename('raw_rows').reset_index().merge(d.groupby(['PART_NAME','EQUIP_CD','EQUIP_NAME']).agg(unique_ids=('_id','size'),defects=('defect','sum')).reset_index(),on=['PART_NAME','EQUIP_CD','EQUIP_NAME'])
 coverage.to_csv(out/'product_inventory.csv',index=False)
 s14=d[d.EQUIP_CD.eq('S14') & d['product'].isin(['CN7','RG3'])]
 cohorts={'all_products_reference':d,'s14_pooled_reference':s14}
 for product in ['CN7','RG3']:
  g=s14[s14['product']==product];cohorts[product.lower()]=g
  for side in ['LH','RH']:cohorts[product.lower()+'_'+side.lower()]=g[g.side==side]
 summaries=[];pairs=[];targets=[];constants=[]
 for name,g in cohorts.items():
  summaries.append({'cohort':name,'records':len(g),'defects':int(g.defect.sum()),'dates':g.date.nunique(),'event_candidates':len(g[['EQUIP_CD','TimeStamp']].drop_duplicates()),'interpretation':'mixed product/equipment reference only' if 'reference' in name else 'descriptive within product; sides retained or separated'})
  variable=[c for c in features if g[c].nunique()>1]
  constants.extend({'cohort':name,'feature':c,'value':float(g[c].iloc[0])} for c in features if c not in variable)
  for method in ['pearson','spearman']:
   matrix=g[features].corr(method=method);matrix.to_csv(out/f'matrices/{name}_{method}.csv')
   for a,b in combinations(variable,2):pairs.append({'cohort':name,'method':method,'feature_a':a,'feature_b':b,'correlation':matrix.loc[a,b],'records':len(g)})
   for f in features:
    value=g[f].corr(g.defect,method=method) if g[f].nunique()>1 and g.defect.nunique()>1 else np.nan
    targets.append({'cohort':name,'method':method,'feature':f,'correlation_with_defect':value,'records':len(g),'defects':int(g.defect.sum()),'note':'descriptive only; binary-target Pearson equals point-biserial'})
   if name in ['cn7','rg3']:
    fig=draw_correlation_heatmap(matrix, f'{name.upper()} {method} / {len(g)} unique records / both sides');fig.savefig(out/f'figures/{name}_{method}.png',dpi=160);plt.close(fig)
  # Remove day-by-side means for sensitivity to changing operating conditions.
  if 'reference' not in name:
   residual=g[features+['defect']]-g.groupby(['date','side'])[features+['defect']].transform('mean')
   for f in features:
    value=residual[f].corr(residual.defect) if f in variable and residual[f].std()>1e-12 and residual.defect.std()>1e-12 else np.nan
    targets.append({'cohort':name,'method':'pearson_within_day_side','feature':f,'correlation_with_defect':value,'records':len(g),'defects':int(g.defect.sum()),'note':'Correlation of residuals after date+side mean subtraction; exploratory, not causal or predictive validation'})
 pd.DataFrame(summaries).to_csv(out/'cohort_summary.csv',index=False)
 pd.DataFrame(constants).to_csv(out/'constant_columns.csv',index=False)
 pd.DataFrame(pairs).to_csv(out/'feature_pairs.csv',index=False)
 pd.DataFrame(targets).to_csv(out/'target_correlations.csv',index=False)
 # Inspect paired records; a database join can also duplicate process values.
 events=[]
 for product,g in s14.groupby('product'):
  group=g.groupby(['EQUIP_CD','TimeStamp']);pair_ids=[];mixed=0;identical=0;paired=0
  for key,h in group:
   if len(h)==2 and set(h.side)=={'LH','RH'}:
    paired+=1; mixed+=int(h.defect.nunique()>1)
    same=bool(h[features].nunique(dropna=False).le(1).all());identical+=int(same)
  events.append({'product':product,'records':len(g),'event_candidates':group.ngroups,'exactly_two_opposite_side_records':paired,'paired_same_36_process_values':identical,'paired_different_labels':mixed,'confirmed_cavity_count':False})
  # One observation per distinct event+process vector; do not average differing parts.
  eventview=g.drop_duplicates(['EQUIP_CD','TimeStamp']+features)
  for method in ['pearson','spearman']:
   em=eventview[features].corr(method=method);em.to_csv(out/f'matrices/{product.lower()}_event_process_{method}.csv')
   delta=(em-g[features].corr(method=method)).abs().to_numpy()
   events[-1][method+'_max_abs_event_weight_change']=float(np.nanmax(delta))
  events[-1]['event_process_rows']=len(eventview)
 pd.DataFrame(events).to_csv(out/'paired_event_audit.csv',index=False)
 verification={'source_sha256':sha,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==sha,'raw_rows':len(raw),'deduplicated_rows':len(d),'process_features':features,'cohorts':list(cohorts),'label':'Y=0,N=1','null_correlation':'undefined for constant columns; not zero','exclusions':'IDs, timestamps, production serials, equipment codes and Reason are not numeric predictors','limitations':['Descriptive full-data EDA; no untouched test or causal claims.','Repeated paired events and sparse labels invalidate naive independent-sample inference.','JX1/SP2 each have only one unique labeled record; no within-product correlation possible.','No p-values or statistical significance claims; within-day adjustment not full confounding control.'],'models_trained':False}
 (out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
 print(pd.DataFrame(events).to_string(index=False));print(pd.DataFrame(summaries).to_string(index=False))
if __name__=='__main__':main()
