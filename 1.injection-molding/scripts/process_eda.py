"""S14 CN7/RG3 time-aware descriptive EDA; no model training or causal claims."""
from pathlib import Path
import argparse, hashlib, json, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['Injection_Time','Filling_Time','Plasticizing_Time','Cycle_Time',
          'Max_Injection_Pressure','Max_Injection_Speed','Cushion_Position',
          'Barrel_Temperature_5','Mold_Temperature_3','Mold_Temperature_4']
REASONS={'가스':'Gas','미성형':'Incomplete fill','초기허용불량':'Startup allowance'}

def prepare(d):
    d=d[(d.EQUIP_CD=='S14') & d.PART_NAME.str.match(r'^(CN7 W/S SIDE|RG3 MOLD.*W/SHLD)',na=False)].copy()
    d['timestamp']=pd.to_datetime(d.TimeStamp,errors='raise')
    d['product']=d.PART_NAME.str.extract(r'^(CN7|RG3)')
    d['side']=d.PART_NAME.str.strip().str.extract(r'(LH|RH)$')
    assert d.side.notna().all()
    d['date']=d.timestamp.dt.strftime('%Y-%m-%d')
    return d.sort_values(['product','side','timestamp','_id']).reset_index(drop=True)

def assign_periods(d):
    """60/20/20 by observed dates, floor boundaries, independent of outcome."""
    d=d.copy();d['period']=''; rules={}
    for product,g in d.groupby('product'):
        dates=sorted(g.date.unique());n=len(dates)
        if n<3: raise ValueError('At least three observed dates required')
        a=max(1,min(n-2,int(n*.6)));b=max(a+1,min(n-1,int(n*.8)))
        mapping={date:('train' if i<a else 'validation' if i<b else 'test') for i,date in enumerate(dates)}
        d.loc[g.index,'period']=g.date.map(mapping)
        rules[product]={key:[date for date in dates if mapping[date]==key] for key in ['train','validation','test']}
    return d,rules

def segments(d,gap_minutes=10):
    d=d.copy(); out=[]
    for keys,g in d.groupby(['product','side']):
        g=g.sort_values('timestamp').copy()
        g['gap_seconds']=g.timestamp.diff().dt.total_seconds()
        g['segment']=(g.gap_seconds.gt(gap_minutes*60)|g.gap_seconds.isna()).cumsum()
        g['minutes_since_observed_segment_start']=(g.timestamp-g.groupby('segment').timestamp.transform('min')).dt.total_seconds()/60
        out.append(g)
    return pd.concat(out,ignore_index=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/process_eda_v1')
    args=ap.parse_args();o=args.output
    if o.exists():raise FileExistsError(f'Choose a new output directory: {o}')
    o.mkdir(parents=True);(o/'figures').mkdir();(o/'rows').mkdir()
    source=ROOT/'sources/04. Dataset_Molding.zip';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    with zipfile.ZipFile(source) as z:
        with z.open('dataset/labeled_data.csv') as f:raw=pd.read_csv(f)
        # Reject varying values within an ID rather than silently taking the first label.
        assert not raw.drop_duplicates().duplicated('_id').any()
        d=prepare(raw.drop_duplicates('_id'))
        with z.open('dataset/unlabeled_data.csv') as f:
            u=pd.concat([prepare(c) for c in pd.read_csv(f,chunksize=100000)],ignore_index=True)
    d['defect']=d.PassOrFail.eq('N').astype(int)
    assert d.PassOrFail.isin(['Y','N']).all()
    d,rules=assign_periods(d);d=segments(d)
    d.to_csv(o/'rows/labeled_view.csv',index=False)
    daily=d.groupby(['product','side','date']).agg(records=('_id','size'),defects=('defect','sum'),events=('timestamp','nunique')).reset_index()
    daily['observed_defect_fraction']=daily.defects/daily.records;daily.to_csv(o/'daily_counts.csv',index=False)
    d.groupby(['product','side','date','Reason'],dropna=False).size().reset_index(name='records').to_csv(o/'daily_reasons.csv',index=False)
    session=d.groupby(['product','side','segment']).agg(start=('timestamp','min'),end=('timestamp','max'),records=('_id','size'),defects=('defect','sum'),preceding_gap_seconds=('gap_seconds','first')).reset_index()
    # groupby.first skips NaN: explicitly preserve unknown gap at first observed segment.
    first=d.sort_values('timestamp').groupby(['product','side','segment'],sort=False).head(1)
    session=session.drop(columns='preceding_gap_seconds').merge(first[['product','side','segment','gap_seconds']],on=['product','side','segment'],validate='one_to_one').rename(columns={'gap_seconds':'preceding_gap_seconds'})
    session.to_csv(o/'observed_segments.csv',index=False)
    sensitivity=[];near=[]
    for gap in [5,10,30,60]:
        s=segments(d,gap)
        for product,g in s.groupby('product'):
            sensitivity.append({'gap_minutes':gap,'product':product,'side_segments':len(g[['side','segment']].drop_duplicates()),'records':len(g)})
    pd.DataFrame(sensitivity).to_csv(o/'gap_sensitivity.csv',index=False)
    # This is proximity to observation starts, not proof of machine restarts.
    d['first_10_observed_minutes']=d.minutes_since_observed_segment_start.le(10)
    for keys,g in d.groupby(['product','side','first_10_observed_minutes']):
        near.append(dict(zip(['product','side','first_10_observed_minutes'],keys))|{'records':len(g),'defects':int(g.defect.sum()),'observed_defect_fraction':float(g.defect.mean())})
    pd.DataFrame(near).to_csv(o/'segment_start_counts.csv',index=False)
    # Day/side-matched descriptive comparisons; no label ranking or statistical testing.
    comparison=[]
    for keys,g in d.groupby(['product','side','date']):
        normal=g[g.defect==0]
        for reason,bad in g[g.defect==1].groupby('Reason'):
            for f in FEATURES:
                q1,q3=normal[f].quantile([.25,.75]);iqr=q3-q1
                comparison.append(dict(zip(['product','side','date'],keys))|{'reason':reason,'feature':f,'normal_n':len(normal),'defect_n':len(bad),'normal_median':normal[f].median(),'defect_median':bad[f].median(),'raw_median_difference':bad[f].median()-normal[f].median(),'normal_iqr':iqr,'difference_over_normal_iqr':(bad[f].median()-normal[f].median())/iqr if len(normal)>=10 and iqr>0 else np.nan})
    pd.DataFrame(comparison).to_csv(o/'same_day_side_comparison.csv',index=False)
    # Distribution coverage includes past unlabeled only in each prospective training pool.
    eligibility=[];shift=[]
    for product,g in d.groupby('product'):
        uu=u[u["product"]==product].copy(); cutoff=g[g.period=='train'].timestamp.max()
        before=uu.timestamp.le(cutoff)
        # Conservative event key omits side: keep paired parts together.
        protected=set(g.timestamp)
        overlap=uu.timestamp.isin(protected)
        eligible=uu[before & ~overlap]
        eligible[['_id','timestamp','product','side']].to_csv(o/f'rows/{product.lower()}eligible_unlabeled.csv',index=False)
        eligibility.append({'product':product,'train_end':str(cutoff),'all_unlabeled':len(uu),'after_train_end':int((~before).sum()),'overlap_with_any_labeled_event':int(overlap.sum()),'excluded_overlap_before_train_end':int((before&overlap).sum()),'eligible_unlabeled':len(eligible),'confirmed_normal':False})
        assert eligible.timestamp.le(cutoff).all() and not eligible.timestamp.isin(protected).any()
        for side in ['LH','RH']:
            for name,frame in [('eligible_unlabeled',eligible),('labeled_train',g[g.period=='train']),('labeled_all_descriptive',g)]:
                h=frame[frame.side==side]
                for f in FEATURES:
                    shift.append({'product':product,'side':side,'source':name,'feature':f,'records':len(h),'p05':h[f].quantile(.05),'median':h[f].median(),'p95':h[f].quantile(.95),'zero_fraction':h[f].eq(0).mean()})
    pd.DataFrame(eligibility).to_csv(o/'unlabeled_eligibility.csv',index=False)
    pd.DataFrame(shift).to_csv(o/'distribution_context.csv',index=False)
    u.groupby(['product','side',u.timestamp.dt.to_period('M').astype(str)]).size().reset_index(name='records').rename(columns={'timestamp':'month'}).to_csv(o/'unlabeled_monthly_counts.csv',index=False)
    split=d.groupby(['product','period']).agg(records=('_id','size'),defects=('defect','sum'),event_candidates=('timestamp','nunique'),start=('timestamp','min'),end=('timestamp','max')).reset_index()
    split['normals']=split.records-split.defects
    split['both_classes_present']=split.defects.gt(0)&split.normals.gt(0)
    split.to_csv(o/'chronological_feasibility.csv',index=False)
    d.groupby(['product','period','Reason'],dropna=False).size().reset_index(name='records').to_csv(o/'period_reasons.csv',index=False)
    for product,g in d.groupby('product'):
        assert g.groupby('timestamp').period.nunique().max()==1
        bounds=[g[g.period==p].timestamp for p in ['train','validation','test']]
        assert bounds[0].max()<bounds[1].min() and bounds[1].max()<bounds[2].min()
        fig,axes=plt.subplots(4,1,figsize=(13,10),sharex=True)
        for ax,f in zip(axes,['Filling_Time','Max_Injection_Pressure','Barrel_Temperature_5','Mold_Temperature_3']):
            for side,color in [('LH','tab:blue'),('RH','tab:orange')]:
                h=g[g.side==side]
                ax.scatter(h.timestamp,h[f],s=5,alpha=.3,color=color,label=side)
            for reason,marker in [('가스','x'),('미성형','^'),('초기허용불량','s')]:
                h=g[g.Reason==reason]
                ax.scatter(h.timestamp,h[f],s=45,marker=marker,color='crimson',label=REASONS[reason],zorder=3)
            for stamp in [bounds[1].min(),bounds[2].min()]:ax.axvline(stamp,color='gray',linestyle='--',linewidth=.8)
            ax.set_ylabel(f);ax.grid(alpha=.2)
        axes[0].legend(ncol=5,fontsize=8)
        axes[0].set_title(product+' / observed process values and defects (descriptive; dashed lines: candidate period starts)')
        fig.autofmt_xdate();fig.tight_layout();fig.savefig(o/f'figures/{product.lower()}_timeline.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,side in zip(axes,['LH','RH']):
            h=g[g.side==side]; tab=pd.crosstab(h.date,h.Reason.fillna('Normal')).reindex(columns=list(REASONS),fill_value=0)
            tab.rename(columns=REASONS).plot.bar(stacked=True,ax=ax)
            ax.set_title(product+' '+side+' / defect records');ax.set_ylabel('Deduplicated records');ax.tick_params(axis='x',rotation=60)
        fig.tight_layout();fig.savefig(o/f'figures/{product.lower()}_defect_days.png',dpi=150);plt.close(fig)
    protocol={'source_sha256':source_hash,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'S14; exact CN7 W/S SIDE and RG3 W/SHLD family; sides retained','gap_minutes':10,'gap_meaning':'Observation gap only; NOT confirmed restart','split_rule':'Observed unique dates, floor 60/80 percent boundaries, at least one day per partition; labels not used to choose boundaries','period_dates':rules,'status':'Exploratory feasibility, NOT pristine holdout; all labels already inspected during EDA','unlabeled_rule':'At or before training end, exclude any labeled event timestamp of same product/S14 regardless of side; labels unknown','models_trained':False,'pandas':pd.__version__,'numpy':np.__version__}
    (o/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    checks={'source_unchanged':True,'unique_labeled_ids':bool(d._id.is_unique),'scope_records':len(d),'scope_defects':int(d.defect.sum()),'chronological_boundaries_valid':True,'event_candidates_not_split_between_periods':True,'eligible_unlabeled_excludes_all_labeled_event_candidates':True,'models_trained':False}
    (o/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(split.to_string(index=False));print(pd.DataFrame(eligibility).to_string(index=False));print('Saved:',o)

if __name__=='__main__':main()
