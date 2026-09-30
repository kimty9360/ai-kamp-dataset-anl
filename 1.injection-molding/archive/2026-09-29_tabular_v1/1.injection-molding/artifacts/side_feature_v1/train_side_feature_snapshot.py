"""Controlled RH indicator ablation; frozen baseline partitions and parameters."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from train_baseline import PROJECT, sha, make_model, probability, select_threshold, metrics


def join_metadata(rows, metadata):
    keys=['source_file','row_position']
    if metadata.duplicated(keys).any(): raise ValueError('Duplicate metadata key')
    result=rows.merge(metadata[keys+['side','PassOrFail','TimeStamp','EQUIP_CD']],on=keys,how='left',validate='one_to_one',suffixes=('','_metadata'),sort=False)
    if not result.side.isin(['LH','RH']).all(): raise ValueError('Missing/invalid side')
    if not np.array_equal(result.PassOrFail,result.PassOrFail_metadata): raise ValueError('Metadata label mismatch')
    if not result[keys].equals(rows[keys]): raise ValueError('Row order changed')
    return result


def main(base,archive,out):
    if out.exists(): raise FileExistsError(out)
    cfg=json.loads((base/'config.json').read_text())
    splits=json.loads((base/'splits.json').read_text())
    old_manifest=json.loads((base/'manifest.json').read_text())
    features=json.loads((base/'preflight.json').read_text())['retained_features']
    protected={str(p):sha(p) for p in base.rglob('*') if p.is_file()}
    raw_paths=[PROJECT/'dataset'/f'moldset_{kind}_{g}.csv' for kind in ['labeled','unlabeled'] for g in ['cn7','rg3']]
    protected.update({str(p):sha(p) for p in raw_paths})
    metadata=[]; raw_frames=[]
    with zipfile.ZipFile(archive) as z:
        intermediate=pd.read_csv(z.open('dataset/moldset_labeled.csv'))
        for group in ['cn7','rg3']:
            name=f'moldset_labeled_{group}.csv'
            p=PROJECT/'dataset'/name
            if sha(p)!=old_manifest['input_sha256'][name]: raise ValueError('Raw changed')
            df=pd.read_csv(p)
            source=intermediate[intermediate.PART_NAME.str.startswith(group.upper())].iloc[:len(df)].copy().reset_index(drop=True)
            cols=[c for c in df if c not in ['Unnamed: 0','PassOrFail']]
            np.testing.assert_allclose(StandardScaler().fit_transform(source[cols]),df[cols],rtol=0,atol=1e-10)
            np.testing.assert_array_equal(source.PassOrFail,df.PassOrFail)
            source['side']=source.PART_NAME.str.extract(r'(LH|RH)$',expand=False)
            source['source_file']=name;source['row_position']=np.arange(len(source))
            metadata.append(source[['source_file','row_position','side','PassOrFail','TimeStamp','EQUIP_CD']])
            df['source_file']=name;df['row_position']=np.arange(len(df))
            raw_frames.append(df)
    meta=pd.concat(metadata,ignore_index=True)
    raw=pd.concat(raw_frames,ignore_index=True)
    prepared={}
    for scenario in cfg['scenarios']:
        rows=pd.read_csv(base/f'{scenario}_rows.csv')
        joined=join_metadata(rows,meta)
        # Explicit row-key alignment; only process inputs and known product/side identities enter model.
        frame=rows[['source_file','row_position']].merge(raw,on=['source_file','row_position'],how='left',validate='one_to_one',sort=False)
        np.testing.assert_array_equal(frame.PassOrFail,rows.PassOrFail)
        x=frame[features].copy()
        if scenario=='pooled': x['product_is_rg3']=rows.product_group.eq('rg3').astype(int)
        augmented=x.assign(part_is_rh=joined.side.eq('RH').astype(int))
        prepared[scenario]=(rows,joined,x,augmented,rows.PassOrFail.to_numpy())
    out.mkdir(parents=True)
    protocol={'experiment':'side_feature_v1','reference_run':base.name,'models':cfg['models'],'scenarios':cfg['scenarios'],'added_feature':'part_is_rh: RH=1 LH=0 from PART_NAME','positive_label':1,'label_semantics':'0=pass,1=fail; verified source Y/N mapping','selection':'no hyperparameter or model selection; only one feature added','group_definition':'original process feature groups; NOT recomputed after adding side','threshold':'baseline inner group OOF F1 rule','independent_test':False,'source_archive_sha256':sha(archive),'raw_and_baseline_sha256':protected,'code_sha256':sha(Path(__file__)),'baseline_helper_sha256':sha(PROJECT/'scripts/train_baseline.py'),'assumption':'side known at prediction time; post-cycle pre-inspection','leakage_exclusions':['Reason','PassOrFail','TimeStamp','_id','PART_FACT_SERIAL'],'metadata_reconstruction_tolerance':1e-10}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    shutil.copyfile(__file__,out/'train_side_feature_snapshot.py')
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    old_oof=pd.read_csv(base/'oof_predictions.csv')
    baseline_metrics=pd.read_csv(base/'fold_metrics.csv').assign(variant='baseline')
    new_scores=[]; predictions=[]; verified=0; fit_count=0
    for split in splits:
        scenario=split['scenario'];rows,joined,x,aug,y=prepared[scenario]
        tr,va=np.array(split['train_indices']),np.array(split['validation_indices'])
        assert not set(rows.feature_group.iloc[tr]) & set(rows.feature_group.iloc[va])
        # A same-equipment/same-timestamp pair must not cross partitions either.
        time_groups=joined.EQUIP_CD.astype(str)+'|'+joined.TimeStamp.astype(str)
        assert not set(time_groups.iloc[tr]) & set(time_groups.iloc[va])
        inner=[]
        for fold in split['inner_splits_relative_to_train']:
            it,iv=np.array(fold['train']),np.array(fold['validation'])
            assert not set(rows.feature_group.iloc[tr[it]]) & set(rows.feature_group.iloc[tr[iv]])
            assert not set(time_groups.iloc[tr[it]]) & set(time_groups.iloc[tr[iv]])
            inner.append((it,iv))
        for name in cfg['models']:
            if split['seed']==cfg['outer_seeds'][0] and split['fold']==0:
                check=make_model(name,cfg,y[tr]);check.fit(x.iloc[tr],y[tr])
                old=old_oof[(old_oof.scenario==scenario)&(old_oof.seed==split['seed'])&(old_oof.fold==0)&(old_oof.model==name)]
                wanted=pd.MultiIndex.from_frame(rows.iloc[va][['source_file','row_position']])
                old=old.set_index(['source_file','row_position']).loc[wanted]
                np.testing.assert_allclose(probability(check,x.iloc[va]),old.p_class_1,rtol=0,atol=1e-12)
                verified+=1
            cutoff=0.5
            if name!='dummy_prior':
                inner_p=np.full(len(tr),np.nan)
                for it,iv in inner:
                    model=make_model(name,cfg,y[tr[it]])
                    model.fit(aug.iloc[tr[it]],y[tr[it]])
                    inner_p[iv]=probability(model,aug.iloc[tr[iv]])
                assert np.isfinite(inner_p).all()
                cutoff=select_threshold(y[tr],inner_p)
            model=make_model(name,cfg,y[tr]);model.fit(aug.iloc[tr],y[tr]);p=probability(model,aug.iloc[va]);fit_count+=1
            common=dict(scenario=scenario,seed=split['seed'],fold=split['fold'],model=name,variant='with_side',selected_threshold=cutoff)
            for policy,t in [('fixed_0.5',0.5),('inner_f1',cutoff)]:
                new_scores.append({**common,'policy':policy,**metrics(y[va],p,p>=t,rows.tie_key.to_numpy()[va],cfg['ranking_budgets'])})
            pred=rows.iloc[va][['source_file','row_position','product_group','PassOrFail','feature_group','tie_key']].copy()
            predictions.append(pred.assign(**common,part_is_rh=aug.part_is_rh.to_numpy()[va],p_class_1=p,prediction_inner_f1=(p>=cutoff).astype(int)))
        print('Completed',split['key'],flush=True)
    all_scores=pd.concat([baseline_metrics,pd.DataFrame(new_scores)],ignore_index=True)
    all_scores.to_csv(out/'fold_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'oof_predictions.csv',index=False)
    keys=['ap','roc_auc','f1','precision','recall','recall_at_10pct','precision_at_10pct']
    summary=all_scores.groupby(['scenario','model','variant','policy'])[keys].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index();summary.to_csv(out/'summary.csv',index=False)
    pairing=['scenario','model','seed','fold','policy']
    paired=pd.DataFrame(new_scores).merge(baseline_metrics[pairing+keys],on=pairing,validate='one_to_one',suffixes=('','_baseline'))
    for k in keys:paired[k+'_delta']=paired[k]-paired[k+'_baseline']
    paired.to_csv(out/'paired_comparison.csv',index=False)
    table=summary[summary.policy.eq('inner_f1')].pivot(index=['scenario','model'],columns='variant',values=[k+'_mean' for k in ['ap','f1','recall_at_10pct']])
    table.columns=['_'.join(c) for c in table.columns];table=table.reset_index()
    table.to_csv(out/'comparison.csv',index=False)
    lines=['# 좌우 부품 정보 추가 실험','', 'baseline_v2의 모델 6계열·11설정, CN7/RG3/pooled, 시드3개×외부3폴드를 유지했다. 하이퍼파라미터 변경 없이 part_is_rh 한 열만 추가했다. 기존 결과는 수정하지 않았다.', '', '라벨 1은 원자료 Y/N 및 불량 사유와 대조한 불량이다. 좌우 정보는 원자료 PART_NAME에서 복구했으며 24개 공정값과 라벨의 순서별 재현으로 정합성을 확인했다. 원래 공정 그룹을 유지했고 동일 설비·동일 시각도 외부/내부 폴드 사이에 겹치지 않음을 검사했다.', '', '표는 9개 외부 폴드의 평균이다. F1 임계값은 각 외부 학습 데이터의 내부 그룹 OOF에서만 선택했다. AP와 Recall@10%는 분류 임계값과 무관하다.', '', '|데이터|모델|AP 기존→좌우|F1 기존→좌우|Recall@10% 기존→좌우|','|---|---|---:|---:|---:|']
    for r in table.to_dict('records'):
        vals=[f"{r[k+'_mean_baseline']:.4f} → {r[k+'_mean_with_side']:.4f}" for k in ['ap','f1','recall_at_10pct']]
        lines.append('|'+r['scenario']+'|'+r['model']+'|'+'|'.join(vals)+'|')
    lines+=['','## 해석 제한','','전체 데이터에서 발견한 좌우 차이를 이용한 개발 실험이다. 반복 평가를 독립 최종 성능이나 통계적 유의성으로 해석하지 않는다. RG3에서 불량이 RH에만 나타나는 관계는 다른 기간의 독립 데이터로 검증해야 한다. CN7 불량의 특정 시간대 집중, 제공 파일별 표준화의 한계는 그대로다. 실제 운영에서 좌우 정보가 검사 전에 확보되어야 한다. 모델·미라벨 제출 파일은 생성하지 않았다.','','baseline 전체 파일과 원본 네 CSV의 해시 불변, 원공정/동시각 그룹 분리, 모델별 기준 확률 재현을 verification.json에 기록했다. summary.csv는 평균·표준편차, paired_comparison.csv는 동일 폴드의 전후 차이다.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    for path,value in protected.items():assert sha(Path(path))==value,path
    # Dummy has no feature dependence: every metric must remain unchanged.
    dummy=paired[paired.model.eq('dummy_prior')]
    for k in keys:np.testing.assert_allclose(dummy[k],dummy[k+'_baseline'],rtol=0,atol=1e-12)
    (out/'verification.json').write_text(json.dumps({'status':'passed','baseline_and_raw_unchanged':True,'baseline_reference_refits':verified,'new_outer_fits':fit_count,'old_process_groups_preserved':True,'same_equipment_timestamp_groups_disjoint':True,'dummy_metrics_unchanged':True,'independent_test':False},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline-dir',type=Path,default=PROJECT/'artifacts/baseline_v2');p.add_argument('--archive',type=Path,default=PROJECT/'04. Dataset_Molding.zip');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):main(a.baseline_dir,a.archive,a.output_dir)
