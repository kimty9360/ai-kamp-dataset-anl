"""Verify an expanded run preserves original baseline scores and predictions."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from train_baseline import PROJECT, make_model, probability, sha


def main(run, reference):
    cfg=json.loads((run/'config.json').read_text())
    old_cfg=json.loads((reference/'config.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='complete': raise AssertionError('Run incomplete')
    for name,digest in manifest['input_sha256'].items():
        if sha(PROJECT/'dataset'/name)!=digest: raise AssertionError('Raw changed')
    if json.loads((run/'splits.json').read_text())!=json.loads((reference/'splits.json').read_text()):
        raise AssertionError('Splits differ')
    columns=['scenario','seed','fold','model','policy']
    old=pd.read_csv(reference/'fold_metrics.csv').sort_values(columns).reset_index(drop=True)
    new=pd.read_csv(run/'fold_metrics.csv')
    new=new[new.model.isin(old_cfg['models'])].sort_values(columns).reset_index(drop=True)
    scores=[c for c in old if c not in ['train_seconds_including_inner_cv','predict_seconds']]
    pd.testing.assert_frame_equal(old[scores],new[scores],check_exact=False,rtol=1e-12,atol=1e-12)
    keys=['scenario','seed','model','source_file','row_position']
    old_p=pd.read_csv(reference/'oof_predictions.csv').sort_values(keys).reset_index(drop=True)
    all_new=pd.read_csv(run/'oof_predictions.csv')
    new_p=all_new[all_new.model.isin(old_cfg['models'])].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(old_p,new_p,check_exact=False,rtol=1e-12,atol=1e-12)
    # Independently check complete scenario/model/seed Cartesian product and coverage.
    expected={(s,m,seed) for s in cfg['scenarios'] for m in cfg['models'] for seed in cfg['outer_seeds']}
    actual=set(all_new.groupby(['scenario','model','seed']).groups)
    if actual!=expected: raise AssertionError('Missing model/scenario/seed results')
    for (s,m,seed),part in all_new.groupby(['scenario','model','seed']):
        expected_rows=pd.read_csv(run/f'{s}_rows.csv')
        if set(zip(part.source_file,part.row_position))!=set(zip(expected_rows.source_file,expected_rows.row_position)):
            raise AssertionError('Wrong OOF rows')
        if part.duplicated(['source_file','row_position']).any(): raise AssertionError('Duplicate OOF')
    # Same-seed refit smoke check for all six added configurations on a real fold.
    split=json.loads((run/'splits.json').read_text())[0]
    if split['key']!='cn7_42_0': raise AssertionError('Unexpected first split')
    tr=np.array(split['train_indices']);va=np.array(split['validation_indices'])
    raw=pd.read_csv(PROJECT/'dataset/moldset_labeled_cn7.csv')
    y=raw.PassOrFail.to_numpy()
    refits=[]
    with threadpool_limits(limits=4):
        for name in cfg['models']:
            if name in old_cfg['models']: continue
            saved=joblib.load(run/'models'/f'cn7_42_0_{name}.joblib')
            x=raw[saved['features']]
            model=make_model(name,cfg,y[tr]);model.fit(x.iloc[tr],y[tr])
            np.testing.assert_allclose(probability(model,x.iloc[va]),probability(saved['model'],x.iloc[va]),rtol=1e-12,atol=1e-12)
            refits.append(name)
    result={'summary':'기존 5개 설정의 모든 fold 지표·OOF 예측 재현, 추가 6개 설정의 실제 fold 동일 seed 재학습 일치.',
            'reference_run':reference.name,'raw_hashes':'passed','splits':'identical',
            'original_fold_metric_rows':len(old),'original_oof_rows':len(old_p),
            'metric_and_prediction_tolerance':1e-12,'new_model_refits':refits,
            'expected_model_scenario_seed_combinations':len(expected),'oof_coverage':'passed',
            'all_outer_model_reload':manifest['reload_check']}
    (run/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'artifacts/baseline_v2')
    parser.add_argument('--reference-dir',type=Path,default=PROJECT/'artifacts/baseline_v1')
    args=parser.parse_args();main(args.run_dir,args.reference_dir)
