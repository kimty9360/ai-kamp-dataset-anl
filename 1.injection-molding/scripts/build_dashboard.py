"""Build an offline HTML explorer of labeled_data. Embedded records stay local."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import pandas as pd
import numpy as np
from plotly.offline import get_plotlyjs
from review_sources import META
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'artifacts/dashboard_v1');a=p.parse_args()
 source=ROOT/'sources/04. Dataset_Molding.zip';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
 with zipfile.ZipFile(source) as z:
  with z.open('dataset/labeled_data.csv') as f: raw=pd.read_csv(f)
 assert not raw.drop_duplicates().duplicated('_id').any()
 d=raw.drop_duplicates('_id').sort_values(['TimeStamp','PART_NAME']).reset_index(drop=True)
 features=[c for c in d if c not in META];assert len(features)==36
 # Invalid labels/timestamps/non-finite numbers must not silently become valid observations.
 assert d.PassOrFail.isin(['Y','N']).all(), 'Unexpected quality label'
 assert d['_id'].notna().all(), 'Missing record ID'
 pd.to_datetime(d.TimeStamp,errors='raise')
 assert d.TimeStamp.notna().all(), 'Missing timestamp'
 for c in features: d[c]=pd.to_numeric(d[c],errors='raise')
 assert not np.isinf(d[features].to_numpy()).any(), 'Infinite process value'
 def profile(frame):
  zeros=[c for c in features if len(frame) and frame[c].notna().all() and frame[c].eq(0).all()]
  constants=[c for c in features if len(frame)>=2 and frame[c].notna().all() and frame[c].nunique()==1 and c not in zeros]
  missing=[c for c in features if frame[c].isna().all()]
  return {'records':len(frame),'all_zero':zeros,'nonzero_constant':{c:float(frame[c].iloc[0]) for c in constants},'all_missing':missing,'default_feature_count':len(features)-len(set(zeros+constants+missing))}
 audit={'numeric_missing_cells':int(d[features].isna().sum().sum()),'missing_by_column':d[features].isna().sum().to_dict(),'all_products':profile(d),'by_product':{g:profile(x) for g,x in d.groupby(d.PART_NAME.str.split().str[0])},'policy':'Exact ID duplicates removed; raw units and outliers retained. Display/export excludes all-zero, fully missing and constant columns based on selected product/side across all dates. One-row cohorts are not classified as nonzero constant. Source preserved.'}

 dictionary=pd.read_csv(ROOT/'docs/process_column_dictionary.csv').set_index('column')
 records=[]
 for _,r in d.iterrows():
  part=r.PART_NAME.strip();product=part.split()[0];side=part[-2:]
  records.append({'id':r['_id'],'time':r.TimeStamp,'product':product,'side':side,'equipment':r.EQUIP_CD,'equipmentName':r.EQUIP_NAME,'part':part,'reviewFlag':bool(product=='CN7' and r.TimeStamp[:10] in ('2020-10-29','2020-10-30') and r.Clamp_Open_Position<100 and r.Injection_Time-r.Filling_Time>6),'label':int(r.PassOrFail=='N'),'reason':None if pd.isna(r.Reason) else r.Reason,'values':[None if pd.isna(r[c]) else float(r[c]) for c in features]})
 def category(c):
  if 'Temperature' in c:return '온도'
  if 'Pressure' in c:return '압력'
  if 'Position' in c:return '위치'
  if 'Time' in c:return '시간'
  return '속도·회전'
 meta=[{'name':c,'label':dictionary.loc[c,'meaning_ko'],'unit':dictionary.loc[c,'unit_from_guide'],'note':dictionary.loc[c,'caution_ko'],'category':category(c)} for c in features]
 audit['review_flag_count']=sum(r['reviewFlag'] for r in records)
 audit['review_rule']='CN7 on 2020-10-29/30, Clamp_Open_Position < 100 and Injection_Time - Filling_Time > 6; retrospective review flag, not a defect label'
 data={'audit':audit,'features':meta,'records':records,'rawRows':len(raw),'uniqueRows':len(d),'duplicates':len(raw)-len(d),'sourceHash':source_hash,'source':'labeled_data.csv','builtAt':pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
 template=(ROOT/'dashboard/template.html').read_text()
 payload=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
 html=template.replace('/* DASHBOARD_STYLE */',(ROOT/'dashboard/style.css').read_text()).replace('/* PLOTLY_LIBRARY */',get_plotlyjs()).replace('/* DASHBOARD_DATA */',payload).replace('/* DASHBOARD_CODE */',(ROOT/'dashboard/app.js').read_text())
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'index.html').write_text(html)
 counts=d.assign(product=d.PART_NAME.str.split().str[0],defect=d.PassOrFail.eq('N')).groupby('product').agg(records=('_id','size'),defects=('defect','sum')).to_dict('index')
 manifest={'source_sha256':source_hash,'raw_rows':len(raw),'unique_rows':len(d),'duplicate_rows_removed':len(raw)-len(d),'process_features':len(features),'counts':counts,'offline':True,'embedded_row_data':True,'models_trained':False,'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,'build_sources_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'dashboard/template.html',ROOT/'dashboard/style.css',ROOT/'dashboard/app.js']}}
 (a.output/'preprocessing_report.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print('Built',a.output/'index.html','bytes=',len(html.encode()),'records=',len(records),'features=',len(features))
if __name__=='__main__':main()
