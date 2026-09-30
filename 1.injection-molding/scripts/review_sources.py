"""Read-only ZIP audit. No training, scaling, deduplication, or label replacement."""
from pathlib import Path
from itertools import combinations
import argparse, hashlib, json, zipfile
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
META = {'Unnamed: 0','_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL',
        'PART_NO','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason','ERR_FACT_QTY'}

def counts(s):
    return {str(k): int(v) for k,v in s.fillna('<missing>').value_counts(dropna=False).items()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=PROJECT/'artifacts/source_review_v1')
    args=ap.parse_args(); out=args.output; out.mkdir(parents=True, exist_ok=True)
    source=PROJECT/'sources/04. Dataset_Molding.zip'
    summary=[]; details={}; frames={}; features=[]; products=[]; intervals=[]; reasons=[]
    with zipfile.ZipFile(source) as z:
        for member in z.namelist():
            if not member.endswith('.csv'): continue
            name=Path(member).name
            with z.open(member) as f: d=pd.read_csv(f, low_memory=False)
            info={'file':name,'rows':len(d),'columns':len(d.columns),
                  'missing_cells':int(d.isna().sum().sum()),
                  'duplicate_rows_all_columns':int(d.duplicated().sum()),
                  'duplicate_rows_without_export_index':int(d.drop(columns=['Unnamed: 0'],errors='ignore').duplicated().sum())}
            detail={'columns':list(d.columns)}
            for c in ['PassOrFail','Reason','EQUIP_CD','EQUIP_NAME']:
                if c in d: detail[c]=counts(d[c])
            if '_id' in d:
                info['unique_ids']=d['_id'].nunique(); info['missing_ids']=int(d['_id'].isna().sum())
                changed=d.drop(columns=['Unnamed: 0'],errors='ignore').groupby('_id',dropna=False).nunique(dropna=False)
                detail['repeated_id_varying_columns']={c:int((changed[c]>1).sum()) for c in changed if (changed[c]>1).any()}
                detail['ids_with_any_varying_values']=int((changed>1).any(axis=1).sum())
                if 'PassOrFail' in d:
                    detail['unique_id_label_counts_not_training_dedup']=counts(d.drop_duplicates('_id')['PassOrFail'])
                    detail['unique_id_reason_counts_not_training_dedup']=counts(d.drop_duplicates('_id')['Reason'])
            if 'TimeStamp' in d:
                t=pd.to_datetime(d.TimeStamp,errors='coerce'); info.update(time_min=str(t.min()),time_max=str(t.max()),invalid_timestamps=int(t.isna().sum()),timestamp_sorted=bool(t.is_monotonic_increasing))
                for keys,g in d.groupby(['EQUIP_CD','PART_NAME'],dropna=False):
                    ts=pd.to_datetime(g.TimeStamp,errors='coerce').dropna().drop_duplicates().sort_values()
                    delta=ts.diff().dt.total_seconds().dropna()
                    intervals.append({'file':name,'EQUIP_CD':keys[0],'PART_NAME':keys[1], 'rows':len(g),'unique_timestamps':len(ts),'min_seconds':delta.min(),'p50_seconds':delta.median(),'p95_seconds':delta.quantile(.95),'max_seconds':delta.max(),'intervals_le_0_2s':int((delta<=.2).sum()),'intervals':len(delta)})
                    products.append({'file':name,'EQUIP_CD':keys[0],'PART_NAME':keys[1],'rows':len(g),'unique_ids':g['_id'].nunique(),'time_min':str(ts.min()),'time_max':str(ts.max())})
                if 'Reason' in d:
                    for dedup,g in [('all_rows',d),('unique_id',d.drop_duplicates('_id'))]:
                        table=g.groupby(['PART_NAME','PassOrFail','Reason'],dropna=False).size().reset_index(name='rows')
                        table['file']=name; table['count_unit']=dedup; reasons.extend(table.to_dict('records'))
                if 'ERR_FACT_QTY' in d:
                    keys=['EQUIP_CD','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NO']
                    batch=d.groupby(keys,dropna=False).agg(rows=('ERR_FACT_QTY','size'),count_values=('ERR_FACT_QTY','nunique'),min_count=('ERR_FACT_QTY','min'),max_count=('ERR_FACT_QTY','max'))
                    detail['ERR_FACT_QTY']={'min':float(d.ERR_FACT_QTY.min()),'max':float(d.ERR_FACT_QTY.max()),'zero_rows':int((d.ERR_FACT_QTY==0).sum()),'candidate_aggregate_keys':keys,'groups':len(batch),'groups_constant_count':int((batch.count_values==1).sum()),'groups_repeated_rows':int((batch.rows>1).sum()),'groups_varying_count':int((batch.count_values>1).sum()),'note':'Candidate grouping only; do not sum repeated per-row counts or treat as individual labels.'}
                    batch.reset_index().to_csv(out/'aggregate_count_audit.csv',index=False)
                frames[name]=d
            for c in d.columns:
                if c in META or not pd.api.types.is_numeric_dtype(d[c]): continue
                s=d[c]; features.append({'file':name,'column':c,'nunique':s.nunique(),'missing':int(s.isna().sum()),'zero_count':int((s==0).sum()),'min':s.min(),'p01':s.quantile(.01),'median':s.median(),'p99':s.quantile(.99),'max':s.max()})
            summary.append(info); details[name]=detail
            print(f'{name}: {len(d):,} rows, {len(d.columns)} columns',flush=True)
    overlap=[]
    for a,b in combinations(frames,2):
        da,db=frames[a],frames[b]; common=set(da['_id'].dropna()) & set(db['_id'].dropna())
        overlap.append({'file_a':a,'file_b':b,'shared_ids':len(common)})
        if a=='labeled_data.csv' and b=='moldset_labeled.csv':
            labels=da[['_id','PassOrFail']].drop_duplicates().merge(db[['_id','PassOrFail']],on='_id',suffixes=('_source','_processed'),validate='one_to_one')
            details['label_mapping']=labels.groupby(['PassOrFail_source','PassOrFail_processed']).size().reset_index(name='rows').to_dict('records')
    # Same equipment/timestamp is a candidate event, not a verified physical shot ID.
    d=frames['labeled_data.csv'].drop_duplicates('_id')
    candidate=d.groupby(['EQUIP_CD','TimeStamp'],dropna=False).agg(rows=('_id','size'),parts=('PART_NAME','nunique'),labels=('PassOrFail','nunique'))
    details['labeled_candidate_events']={'groups':len(candidate),'multiple_parts':int((candidate.parts>1).sum()),'mixed_labels':int((candidate.labels>1).sum()),'not_verified_shot_id':True}
    daily=d.assign(date=pd.to_datetime(d.TimeStamp).dt.strftime('%Y-%m-%d'))
    daily.groupby(['EQUIP_CD','PART_NAME','date','PassOrFail','Reason'],dropna=False).size().reset_index(name='unique_records').to_csv(out/'labeled_daily_counts.csv',index=False)
    keys=['EQUIP_CD','PART_NAME','TimeStamp']
    common=d.merge(frames['unlabeled_data.csv'],on=keys,suffixes=('_labeled','_unlabeled'))
    details['labeled_unlabeled_candidate_overlap']={'keys':keys,'matched_pairs':len(common),'labeled_ids_matched':int(common['_id_labeled'].nunique()),'note':'No shared IDs does not rule out repeated events or temporal leakage.'}
    process=[c for c in d if c not in META]
    if len(common):
        left=common[[c+'_labeled' for c in process]].to_numpy(dtype=float)
        right=common[[c+'_unlabeled' for c in process]].to_numpy(dtype=float)
        details['labeled_unlabeled_candidate_overlap'].update(process_columns=len(process),all_process_exact_pairs=int(np.equal(left,right).all(axis=1).sum()),all_process_close_pairs=int(np.isclose(left,right,rtol=1e-6,atol=1e-5).all(axis=1).sum()),max_absolute_process_difference=float(np.max(np.abs(left-right))))
    zero_by_equipment=[]
    for name,frame in frames.items():
        for equipment,g in frame.groupby('EQUIP_CD'):
            for col in frame:
                if col not in META and pd.api.types.is_numeric_dtype(g[col]):
                    zero_by_equipment.append({'file':name,'EQUIP_CD':equipment,'column':col,'rows':len(g),'zero_fraction':float(g[col].eq(0).mean()),'nunique':g[col].nunique()})
    pd.DataFrame(zero_by_equipment).to_csv(out/'sensor_coverage.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'file_inventory.csv',index=False)
    pd.DataFrame(features).to_csv(out/'feature_statistics.csv',index=False)
    pd.DataFrame(products).to_csv(out/'product_coverage.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'timestamp_intervals.csv',index=False)
    pd.DataFrame(reasons).to_csv(out/'reason_counts.csv',index=False)
    pd.DataFrame(overlap).to_csv(out/'id_overlap.csv',index=False)
    details['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    details['method']={'pandas':pd.__version__,'numpy':np.__version__,'timestamps':'No timezone assumed. Intervals between distinct times within equipment and PART_NAME. Gaps retained.','duplicates':'All columns and export-index-excluded exact row counts; unique-ID views descriptive only. Source unchanged.'}
    (out/'audit.json').write_text(json.dumps(details,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
