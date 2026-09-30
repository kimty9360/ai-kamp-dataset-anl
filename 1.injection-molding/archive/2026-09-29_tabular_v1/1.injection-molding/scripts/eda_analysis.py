"""Read-only EDA and saved-OOF error analysis. No model fitting or feature selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT=Path(__file__).resolve().parents[1]
FEATURE_EXCLUSIONS=['Unnamed: 0','PassOrFail']
KEYS=['source_file','row_position']
FOCUS_MODELS=['logistic','xgboost_balanced','extratrees','catboost_balanced','lightgbm_balanced']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def add_inspection_selection(oof,budget=.10):
    result=oof.copy()
    result['selected_top10']=False
    for _,part in result.groupby(['scenario','seed','fold','model'],sort=False):
        selected=part.sort_values(['p_class_1','tie_key'],ascending=[False,True],kind='stable').head(math.ceil(len(part)*budget)).index
        result.loc[selected,'selected_top10']=True
    return result


def error_counts(y,p):
    y=np.asarray(y);p=np.asarray(p)
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum())
    return dict(rows=len(y),positives=int((y==1).sum()),tp=tp,fp=fp,fn=fn,tn=tn,
                recall=tp/(tp+fn) if tp+fn else np.nan,
                precision=tp/(tp+fp) if tp+fp else np.nan,
                fnr=fn/(tp+fn) if tp+fn else np.nan,
                fpr=fp/(fp+tn) if fp+tn else np.nan)


def descriptive_auc(y,x):
    if len(np.unique(y))<2:return np.nan
    return float(roc_auc_score(y,x))


def validate_oof(oof,raw,cfg):
    unique=['scenario','seed','model',*KEYS]
    if oof.duplicated(unique).any():raise ValueError('Duplicate OOF record')
    if raw.duplicated(KEYS).any():raise ValueError('Duplicate raw record key')
    joined=oof.merge(raw,on=KEYS,how='left',validate='many_to_one',suffixes=('','_raw'),indicator=True)
    if not joined['_merge'].eq('both').all():raise ValueError('OOF row missing from raw')
    for col in ['PassOrFail','Unnamed: 0','product_group','feature_group','label_conflict']:
        if not joined[col].eq(joined[col+'_raw']).all():raise ValueError('OOF/raw mismatch: '+col)
    expected={(s,seed,m) for s in cfg['scenarios'] for seed in cfg['outer_seeds'] for m in cfg['models']}
    if set(oof.groupby(['scenario','seed','model']).groups)!=expected:raise ValueError('Missing OOF experiment')
    for (scenario,seed,model),part in oof.groupby(['scenario','seed','model']):
        target=raw if scenario=='pooled' else raw[raw.product_group==scenario]
        if set(map(tuple,part[KEYS].to_numpy()))!=set(map(tuple,target[KEYS].to_numpy())):
            raise ValueError('OOF coverage differs from raw')
        if not np.isfinite(part.p_class_1).all() or not part.p_class_1.between(0,1).all():
            raise ValueError('Invalid OOF probability')
    return joined.drop(columns=['_merge'])


def main(baseline,out,source_check):
    if out.exists():raise FileExistsError(f'Refusing to overwrite: {out}')
    manifest=json.loads((baseline/'manifest.json').read_text())
    cfg=json.loads((baseline/'config.json').read_text())
    if manifest['status']!='complete':raise ValueError('Baseline incomplete')
    data={};hashes={};frames=[]
    for group in ['cn7','rg3']:
        for kind in ['labeled','unlabeled']:
            name=f'moldset_{kind}_{group}.csv';path=PROJECT/'dataset'/name
            hashes[name]=sha(path);data[(group,kind)]=pd.read_csv(path)
            if kind=='labeled':
                if hashes[name]!=manifest['input_sha256'][name]:raise ValueError('Baseline raw mismatch')
                df=data[(group,kind)].copy();df['source_file']=name;df['row_position']=np.arange(len(df));df['product_group']=group
                df['row_block']=np.minimum(9,np.floor(np.arange(len(df))*10/len(df)).astype(int))
                frames.append(df)
    features=[c for c in data[('cn7','labeled')] if c not in FEATURE_EXCLUSIONS]
    inputs=[c for c in features if c not in cfg['drop_features']]
    raw=pd.concat(frames,ignore_index=True)
    raw['feature_group']=raw.groupby(inputs,sort=True).ngroup()
    conflicts=raw.groupby('feature_group').PassOrFail.nunique()>1
    raw['label_conflict']=raw.feature_group.isin(conflicts.index[conflicts])
    oof=add_inspection_selection(pd.read_csv(baseline/'oof_predictions.csv'))
    joined=validate_oof(oof,raw,cfg)
    # Confirm inspection selection reproduces baseline metric definitions.
    fm=pd.read_csv(baseline/'fold_metrics.csv');fm=fm[fm.policy=='inner_f1'].set_index(['scenario','seed','fold','model'])
    for key,part in oof.groupby(['scenario','seed','fold','model']):
        recall=float(part.loc[part.selected_top10,'PassOrFail'].sum()/part.PassOrFail.sum())
        if not np.isclose(recall,fm.loc[key,'recall_at_10pct'],atol=1e-12):raise AssertionError('Ranking policy mismatch')
    source=json.loads(source_check.read_text()) if source_check else {'all_csv_bytes_match':None,'label_semantics':'unconfirmed'}
    if source_check:
        for record in source['files']:
            if not record['bytes_match'] or record['local_sha256']!=hashes[record['file']]:raise ValueError('Source verification stale/mismatched')
    out.mkdir(parents=True);(out/'rows').mkdir();(out/'figures').mkdir()
    if source_check:shutil.copyfile(source_check,out/'source_check.json')
    summary=[];feature_rows=[];block_rows=[];pairs=[];drift=[];outliers=[];pair_gaps=[]
    for group in ['cn7','rg3']:
        df=raw[raw.product_group==group];y=df.PassOrFail
        g=df.groupby('feature_group').agg(size=('PassOrFail','size'),positive_count=('PassOrFail','sum'),min_row=('row_position','min'),max_row=('row_position','max'))
        conflict=(g.positive_count>0)&(g.positive_count<g['size'])
        pos=df[df.PassOrFail==1]
        summary.append(dict(product_group=group,rows=len(df),class_0=int((y==0).sum()),class_1=int(y.sum()),class_1_rate=float(y.mean()),
                            unique_patterns=len(g),conflicting_groups=int(conflict.sum()),conflicting_class_1=int(pos.label_conflict.sum()),
                            first_class_1_row=int(pos.row_position.min()),last_class_1_row=int(pos.row_position.max()),
                            class_1_in_first_decile=int(((y==1)&(df.row_block==0)).sum()),first_decile_rows=int((df.row_block==0).sum())))
        for gap,count in (g.max_row-g.min_row).value_counts().sort_index().items():pair_gaps.append(dict(product_group=group,row_gap=int(gap),groups=int(count)))
        for block,part in df.groupby('row_block'):
            block_rows.append(dict(product_group=group,row_block=int(block),rows=len(part),class_1=int(part.PassOrFail.sum()),class_1_rate=float(part.PassOrFail.mean())))
        first=df[df.row_block==0]
        for col in features:
            auc=descriptive_auc(y,df[col]);first_auc=descriptive_auc(first.PassOrFail,first[col])
            feature_rows.append(dict(product_group=group,feature=col,constant=df[col].nunique()==1,
                class_0_mean=float(df.loc[y==0,col].mean()),class_1_mean=float(df.loc[y==1,col].mean()),
                mean_gap=float(df.loc[y==1,col].mean()-df.loc[y==0,col].mean()),
                auc_high_value=auc,auc_direction_free=max(auc,1-auc),rank_biserial=2*auc-1,
                first_decile_auc_high_value=first_auc,
                first_decile_auc_direction_free=max(first_auc,1-first_auc) if np.isfinite(first_auc) else np.nan,
                class_0_median=float(df.loc[y==0,col].median()),class_1_median=float(df.loc[y==1,col].median()),
                overall_min=float(df[col].min()),overall_max=float(df[col].max())))
        for method in ['pearson','spearman']:
            corr=df[inputs].corr(method=method)
            corr.to_csv(out/f'{group}_{method}_correlation.csv')
            for i,a in enumerate(inputs):
                for b in inputs[i+1:]:
                    value=corr.loc[a,b]
                    if abs(value)>=.95:pairs.append(dict(product_group=group,method=method,feature_a=a,feature_b=b,correlation=float(value)))
        for label,part in df.groupby('PassOrFail'):
            for cutoff in [3,5]:
                extreme=part[inputs].abs().gt(cutoff).any(axis=1)
                outliers.append(dict(product_group=group,label=int(label),abs_z_cutoff=cutoff,rows=len(part),extreme_rows=int(extreme.sum()),extreme_fraction=float(extreme.mean())))
        unlabeled=data[(group,'unlabeled')]
        for col in features:
            a=df[col];b=unlabeled[col]
            drift.append(dict(product_group=group,feature=col,ks_statistic=float(ks_2samp(a,b).statistic),
                wasserstein_standardized=float(wasserstein_distance(a,b)),labeled_unique=int(a.nunique()),unlabeled_unique=int(b.nunique()),
                unlabeled_outside_labeled_range=float(((b<a.min())|(b>a.max())).mean()),
                labeled_mean=float(a.mean()),unlabeled_mean=float(b.mean()),labeled_std=float(a.std(ddof=0)),unlabeled_std=float(b.std(ddof=0))))
    summary=pd.DataFrame(summary);feature_table=pd.DataFrame(feature_rows);blocks=pd.DataFrame(block_rows)
    correlated=pd.DataFrame(pairs,columns=['product_group','method','feature_a','feature_b','correlation'])
    drift=pd.DataFrame(drift);outlier_table=pd.DataFrame(outliers)
    for name,table in [('data_summary',summary),('feature_class_summary',feature_table),('row_blocks',blocks),('correlated_pairs',correlated),
                       ('distribution_shift',drift),('outlier_summary',outlier_table),('duplicate_row_gaps',pd.DataFrame(pair_gaps))]:
        table.to_csv(out/f'{name}.csv',index=False)
    raw[raw.label_conflict][KEYS+['Unnamed: 0','product_group','feature_group','PassOrFail',*features]].to_csv(out/'rows/conflict_records.csv',index=False)
    # All-seed summaries are recorded per seed: no duplicated rows counted as independent observations.
    cohort_rows=[];block_errors=[]
    for (scenario,seed,model),part in joined.groupby(['scenario','seed','model']):
        for product,product_part in part.groupby('product_group'):
            for cohort,subset in [('all',product_part),('conflict',product_part[product_part.label_conflict]),('no_conflict',product_part[~product_part.label_conflict])]:
                for policy,column in [('inner_f1','prediction_inner_f1'),('top10','selected_top10')]:
                    cohort_rows.append(dict(scenario=scenario,product_group=product,seed=seed,model=model,cohort=cohort,policy=policy,
                        **error_counts(subset.PassOrFail,subset[column])))
            if scenario==product and seed==42 and model!='dummy_prior':
                for block,subset in product_part.groupby('row_block'):
                    block_errors.append(dict(product_group=product,model=model,row_block=int(block),**error_counts(subset.PassOrFail,subset.prediction_inner_f1)))
    pd.DataFrame(cohort_rows).to_csv(out/'error_cohorts_by_seed.csv',index=False)
    pd.DataFrame(block_errors).to_csv(out/'errors_by_row_block_seed42.csv',index=False)
    individual=joined[joined.scenario==joined.product_group].copy()
    individual['false_negative']=(individual.PassOrFail==1)&(individual.prediction_inner_f1==0)
    individual['false_positive']=(individual.PassOrFail==0)&(individual.prediction_inner_f1==1)
    repeated=individual.groupby([*KEYS,'product_group','model'],as_index=False).agg(
        label=('PassOrFail','first'),label_conflict=('label_conflict','first'),seed_count=('seed','nunique'),
        fn_count=('false_negative','sum'),fp_count=('false_positive','sum'),top10_count=('selected_top10','sum'),
        score_mean=('p_class_1','mean'),score_std=('p_class_1','std'),score_min=('p_class_1','min'),score_max=('p_class_1','max'))
    repeated.to_csv(out/'rows/repeated_record_errors.csv',index=False)
    positive=repeated[repeated.label==1]
    stable=[]
    for (product,model),part in positive.groupby(['product_group','model']):
        stable.append(dict(product_group=product,model=model,class_1_rows=len(part),
            missed_all_seeds=int((part.fn_count==part.seed_count).sum()),missed_any_seed=int((part.fn_count>0).sum()),
            outside_top10_all_seeds=int((part.top10_count==0).sum()),inside_top10_all_seeds=int((part.top10_count==part.seed_count).sum())))
    stable=pd.DataFrame(stable);stable.to_csv(out/'stable_misses.csv',index=False)
    # Common errors: same original row across models and seeds, not independent trials.
    real_models=[m for m in cfg['models'] if m!='dummy_prior']
    common=[]
    for (source_file,row),part in positive[positive.model.isin(real_models)].groupby(KEYS):
        common.append(dict(source_file=source_file,row_position=int(row),product_group=part.product_group.iloc[0],
            label_conflict=bool(part.label_conflict.iloc[0]),models_miss_all_seeds=int((part.fn_count==part.seed_count).sum()),
            models_miss_any_seed=int((part.fn_count>0).sum()),models_outside_top10_all_seeds=int((part.top10_count==0).sum()),
            all_models_miss_all_seeds=bool((part.fn_count==part.seed_count).all()),model_count=len(part)))
    common=pd.DataFrame(common);common.to_csv(out/'rows/class1_consensus.csv',index=False)
    selected=individual[(individual.seed==42)&individual.model.isin(real_models)]
    overlap=[]
    for product,part in selected.groupby('product_group'):
        sets={m:set(p.loc[p.false_negative,'row_position']) for m,p in part.groupby('model')}
        for a in real_models:
            for b in real_models:
                union=sets[a]|sets[b];intersection=sets[a]&sets[b]
                overlap.append(dict(product_group=product,model_a=a,model_b=b,fn_intersection=len(intersection),fn_union=len(union),
                                    fn_jaccard=len(intersection)/len(union) if union else np.nan))
    pd.DataFrame(overlap).to_csv(out/'fn_overlap_seed42.csv',index=False)
    focus=individual[(individual.seed==42)&individual.model.isin(FOCUS_MODELS)]
    focus[KEYS+['Unnamed: 0','product_group','model','seed','fold','PassOrFail','label_conflict','p_class_1','threshold','prediction_inner_f1','selected_top10','row_block',*features]].to_csv(out/'rows/focus_oof_with_features.csv',index=False)
    bins=[]
    bounds=[-np.inf,-2,-1,0,1,2,np.inf];bin_labels=['<-2','[-2,-1)','[-1,0)','[0,1)','[1,2)','>=2']
    for (product,model),part in focus.groupby(['product_group','model']):
        for col in inputs:
            bucket=pd.cut(part[col],bounds,right=False,labels=bin_labels)
            for label in bin_labels:
                sub=part[bucket==label]
                bins.append(dict(product_group=product,model=model,feature=col,bin=label,seed=42,**error_counts(sub.PassOrFail,sub.prediction_inner_f1)))
    pd.DataFrame(bins).to_csv(out/'error_feature_bins_seed42.csv',index=False)
    make_figures(raw,features,inputs,feature_table,blocks,drift,selected,real_models,out)
    write_report(summary,feature_table,blocks,drift,stable,common,source,out)
    if any(sha(PROJECT/'dataset'/name)!=digest for name,digest in hashes.items()):raise AssertionError('Raw data changed')
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'baseline_run':baseline.name,'positive_label':1,'label_semantics':'unconfirmed',
                'input_sha256':hashes,'oof_sha256':sha(baseline/'oof_predictions.csv'),'code_sha256':sha(Path(__file__)),
                'baseline_config_sha256':sha(baseline/'config.json'),'oof_rows_validated':len(oof),
                'individual_class1_rows':int(raw.PassOrFail.sum()),'model_fit_performed':False,
                'oof_join_check':'passed','inspection_metric_reproduction':'passed','raw_unchanged':'passed',
                'exploratory_limit':'Whole-data feature/segment exploration is not independent validation. No causal or time-order inference.'}
    (out/'manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(__file__,out/'eda_analysis_snapshot.py')
    print(summary.to_string(index=False));print('EDA complete:',out)


def make_figures(raw,features,inputs,feature_table,blocks,drift,selected,models,out):
    figures=out/'figures'
    fig,axes=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    for col,group in enumerate(['cn7','rg3']):
        df=raw[raw.product_group==group];b=blocks[blocks.product_group==group]
        axes[0,col].bar(b.row_block,b.class_1,color='#d4584a');axes[0,col].set_xticks(range(10),[f'{i*10}-{(i+1)*10}%' for i in range(10)],rotation=45)
        axes[0,col].set_title(f'{group.upper()}: class 1 count by FILE ROW decile');axes[0,col].set_ylabel('Class 1 rows')
        feat=feature_table[(feature_table.product_group==group)&~feature_table.constant].sort_values('auc_direction_free',ascending=False).feature.iloc[0]
        axes[1,col].scatter(df.row_position,df[feat],s=5,alpha=.35,label='All rows')
        pos=df[df.PassOrFail==1];axes[1,col].scatter(pos.row_position,pos[feat],s=35,marker='x',color='red',label='Class 1')
        axes[1,col].set_title(feat);axes[1,col].set_xlabel('File row position (not verified time)');axes[1,col].set_ylabel('Provided standardized value');axes[1,col].legend()
    fig.savefig(figures/'01_row_order.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for i,group in enumerate(['cn7','rg3']):
        df=raw[raw.product_group==group]
        top=feature_table[(feature_table.product_group==group)&~feature_table.constant].nlargest(3,'auc_direction_free')
        for j,row in enumerate(top.itertuples()):
            ax=axes[i,j]
            for label,color in [(0,'#377eb8'),(1,'#e24a33')]:
                values=np.sort(df.loc[df.PassOrFail==label,row.feature].to_numpy())
                ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=f'class {label} (n={len(values)})',color=color)
            ax.set_title(f'{group.upper()} | {row.feature}\nDescriptive direction-free AUC={row.auc_direction_free:.3f}')
            ax.set_xlabel('Provided standardized value');ax.set_ylabel('Cumulative fraction within class');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Whole-data exploratory distributions (not CV performance)')
    fig.savefig(figures/'02_class_distributions.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(24,12),constrained_layout=True)
    for ax,group in zip(axes,['cn7','rg3']):
        corr=raw.loc[raw.product_group==group,inputs].corr(method='spearman')
        im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(inputs)),inputs,rotation=90,fontsize=7);ax.set_yticks(range(len(inputs)),inputs,fontsize=7);ax.set_title(group.upper()+' Spearman correlation')
    fig.colorbar(im,ax=axes,shrink=.5);fig.savefig(figures/'03_correlations.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(17,6),constrained_layout=True)
    for ax,group in zip(axes,['cn7','rg3']):
        part=drift[drift.product_group==group].nlargest(8,'ks_statistic').sort_values('ks_statistic')
        ax.barh(part.feature,part.ks_statistic,color='#4987ad');ax.set_xlim(0,1);ax.set_title(group.upper()+' labeled vs unlabeled');ax.set_xlabel('KS distribution distance (not p-value)')
    fig.savefig(figures/'04_distribution_shift.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(16,13),constrained_layout=True)
    for i,group in enumerate(['cn7','rg3']):
        part=selected[(selected.product_group==group)&(selected.PassOrFail==1)]
        for j,column in enumerate(['prediction_inner_f1','selected_top10']):
            matrix=part.pivot(index='row_position',columns='model',values=column).reindex(columns=models).sort_index()
            ax=axes[i,j];ax.imshow(matrix.to_numpy(dtype=int),vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
            ax.set_xticks(range(len(models)),models,rotation=60,ha='right',fontsize=8)
            conflicts=part.groupby('row_position').label_conflict.first()
            ax.set_yticks(range(len(matrix)),[f'{r}{" *" if conflicts.loc[r] else ""}' for r in matrix.index],fontsize=8)
            ax.set_title(f'{group.upper()} | '+('Class 1 prediction' if j==0 else 'Selected in fold top 10%'))
            ax.set_ylabel('Original row (0-based); * = conflicting label group')
    fig.suptitle('OOF seed 42 | Actual class 1 rows only | Green = found; red = missed')
    fig.savefig(figures/'05_class1_error_matrix.png',dpi=150);plt.close(fig)


def write_report(summary,feature_table,blocks,drift,stable,common,source,out):
    lines=['# EDA 및 OOF 오류 분석', '',
           '**분석 범위:** 원본 확인, 클래스 분포, 행 순서, 특성 관계, labeled/unlabeled 분포 차이, 저장된 baseline_v2 OOF 오류. 모델 학습·데이터 수정·파생변수 추가는 하지 않았습니다.', '',
           '## 공식 데이터 확인', '',
           f"- 공식 ZIP과 로컬 CSV 4개 바이트 일치: **{source.get('all_csv_bytes_match', '미확인')}**.",
           ('- 검증한 ZIP은 CSV 4개만 포함하고 별도 데이터 사전은 없습니다. 첨부 HWPX의 키워드 검색 결과는 source_check.json을 참고하세요.' if source.get('archive_members') else '- 이 실행에는 공식 첨부 검증 기록이 없습니다.'),
           '- 라벨의 불량 방향, 파일 구성·중복 이유, 전처리 이력은 여전히 미확인입니다. 키워드 검색만으로 다른 설명 자료의 존재 여부까지 판단하지 않습니다.',
           f"- 공식 출처: {source.get('notice_url','https://www.kamp-ai.kr/noticeDetail?NOTICE_SEQ=86')}", '',
           '## 데이터 구조와 행 순서', '',
           '| 데이터 | 행 | class_1 | 고유 패턴 | 충돌 그룹 | 충돌에 속한 class_1 | 앞 10%의 class_1 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.product_group} | {r.rows} | {r.class_1} | {r.unique_patterns} | {r.conflicting_groups} | {r.conflicting_class_1} | {r.class_1_in_first_decile} |')
    lines += ['', '- CN7 class_1은 파일 앞부분에 집중되어 있습니다. 행 번호는 시간/배치로 검증되지 않았으며 모델 입력에 사용하지 않습니다.',
              '- 특정 구간의 공정 조건과 라벨 분포가 함께 달라지는 것은 선택·수집·정렬 효과일 수 있습니다. 공정 원인이나 미래 성능으로 단정하지 않습니다.',
              '- 중복이 공식 원본에 존재한다는 사실은 확인됐지만, 실제 반복 생산인지 복제/익명화/가공 결과인지는 알 수 없습니다.', '',
              '## 단일 변수의 클래스 구분 (전체 데이터 탐색)', '',
              '아래 AUC는 학습 모델의 CV 점수가 아닙니다. 전체 라벨을 보고 변수와 방향을 비교했으므로 과대평가될 수 있고 다중 비교를 수행했습니다. p-value/인과 해석은 하지 않습니다.', '',
              '| 데이터 | 변수 | 방향 무관 AUC | class_1 평균 - class_0 평균 | 앞 10% 안의 방향 무관 AUC |',
              '|---|---|---:|---:|---:|']
    for group in ['cn7','rg3']:
        for r in feature_table[(feature_table.product_group==group)&~feature_table.constant].nlargest(5,'auc_direction_free').itertuples():
            lines.append(f'| {group} | {r.feature} | {r.auc_direction_free:.3f} | {r.mean_gap:.3f} | {r.first_decile_auc_direction_free:.3f} |')
    lines += ['', '- 방향 무관 AUC=max(AUC,1-AUC): 0.5에 가까우면 해당 변수 하나로 순위를 구분하기 어렵습니다. 큰 값은 탐색적 연관성이지 검증 성능이 아닙니다.',
              '- 첫 10% 안의 AUC는 파일 위치 집중에 대한 기술적 민감도 분석입니다. 별도 독립 검증이 아니며 표본이 작습니다.',
              '- 표준화된 변수 차이는 섭씨·초·압력 단위 차이가 아닙니다. 상관 높은 변수를 자동 삭제하지 않았습니다.', '',
              '## labeled / unlabeled 분포 차이', '',
              '| 데이터 | 변수 | KS 거리 | 미라벨 중 라벨 범위 밖 비율 |', '|---|---|---:|---:|']
    for group in ['cn7','rg3']:
        for r in drift[drift.product_group==group].nlargest(4,'ks_statistic').itertuples():
            lines.append(f'| {group} | {r.feature} | {r.ks_statistic:.3f} | {r.unlabeled_outside_labeled_range:.1%} |')
    lines += ['', '- 평균과 표준편차가 같아도 분포 모양은 다를 수 있습니다. 파일별 표준화 정황 때문에 동일 수치가 동일 원 단위를 뜻하지 않습니다.',
              '- KS는 분포 차이의 크기만 기술합니다. 중복/의존성을 무시한 p-value 유의성 주장은 하지 않습니다.',
              '- 극단값(abs(z)>3/5)은 점검 대상으로만 집계했고 class_1을 포함한 행을 제거하지 않았습니다.', '',
              '## 반복적으로 놓치는 class_1 (제품군별 모델)', '',
              '같은 원본 행에 대한 세 seed의 OOF 결과를 요약합니다. 세 번의 독립 생산 관측으로 세지 않습니다. 아래 모델은 대표 비교용이며 전체 설정은 stable_misses.csv에 있습니다.', '',
              '| 데이터 | 모델 | class_1 수 | 세 반복 모두 FN | 세 반복 모두 상위 10% 밖 | 세 반복 모두 상위 10% 안 |',
              '|---|---|---:|---:|---:|---:|']
    for r in stable[stable.model.isin(FOCUS_MODELS)].itertuples():
        lines.append(f'| {r.product_group} | {r.model} | {r.class_1_rows} | {r.missed_all_seeds} | {r.outside_top10_all_seeds} | {r.inside_top10_all_seeds} |')
    for group in ['cn7','rg3']:
        part=common[common.product_group==group]
        lines.append(f"\n- {group.upper()}: Dummy를 제외한 10개 설정·세 반복 모두 FN인 원본 class_1은 {int(part.all_models_miss_all_seeds.sum())}행입니다.")
    lines += ['', '- FN은 선택된 판정 임계값 아래에 있다는 뜻이고, 검사 상위 10% 밖이라는 뜻과 다릅니다.',
              '- 오류 조건 표는 seed 42의 OOF와 사전 고정된 표준화 값 구간을 사용합니다. 양성이 없는 구간의 Recall은 0 대신 결측으로 표시합니다.',
              '- 전체 라벨과 오류를 보고 탐색한 조건이므로 확정 규칙이나 독립적으로 검증된 실패조건으로 주장하지 않습니다.', '',
              '## 다음 실험 후보 (아직 실행하지 않음)', '',
              '1. 공식 문의/추가 설명에서 라벨 방향, 중복 생성 이유, 파일 정렬 및 수집 시점을 확인합니다.',
              '2. CN7에서 공정값 구분이 파일 앞부분 안에서도 유지되는지 확인하고 해당 구간의 FN/FP를 살펴봅니다. 행 번호를 파생변수로 넣지 않습니다.',
              '3. 온도 채널의 상대 편차, 최대/평균 측정값의 차이를 소수 묶음으로 검토합니다. 현재 동일 입력의 라벨 충돌은 이러한 파생변수로 해결되지 않습니다.',
              '4. 탐색으로 정한 후보는 고정된 그룹 검증의 처리 전후 비교로 확인하고, 이후 같은 평가에 반복 적응한 선택 편향을 별도로 관리합니다.',
              '5. RG3의 분리 신호가 약한 이유를 모델 종류만으로 단정하지 않고 데이터 수집·라벨 구조와 함께 검토합니다.', '',
              '## 산출물 안내', '',
              '- data_summary.csv, row_blocks.csv, duplicate_row_gaps.csv: 데이터 구조/파일 순서.',
              '- feature_class_summary.csv, correlated_pairs.csv, *_correlation.csv: 클래스 분포와 변수 관계.',
              '- distribution_shift.csv, outlier_summary.csv: 분포 차이/극단값 진단.',
              '- error_cohorts_by_seed.csv: 충돌 여부·제품군별 오류; seed별 원본 행을 한 번씩 집계.',
              '- stable_misses.csv, fn_overlap_seed42.csv, error_feature_bins_seed42.csv: 반복 미탐·모델 겹침·구간별 오류.',
              '- rows/: 원본 행과 결합한 진단 표. 로컬 보관이며 공개 Git 추적에서 제외.',
              '- figures/: 분포/행 순서/상관/오류 그림. manifest.json: 원본·OOF·코드 해시와 검증 결과.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir',type=Path,default=PROJECT/'artifacts/baseline_v2')
    p.add_argument('--output-dir',type=Path,default=PROJECT/'artifacts/eda_v1')
    p.add_argument('--source-check',type=Path)
    a=p.parse_args();main(a.baseline_dir,a.output_dir,a.source_check)
