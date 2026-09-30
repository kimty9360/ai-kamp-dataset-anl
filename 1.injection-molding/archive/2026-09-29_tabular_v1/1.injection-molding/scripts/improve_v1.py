"""Small preregistered ablations using baseline outer/inner group partitions."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from train_baseline import PROJECT, sha, make_model, probability, select_threshold, metrics

VARIANTS = ['base', 'temperature_differences', 'pressure_difference', 'shallower_tree']
MODELS = ['xgboost_balanced', 'catboost_balanced']

def transform(x, variant):
    x = x.copy()
    if variant == 'temperature_differences':
        x['mold_z_difference'] = x.Mold_Temperature_3 - x.Mold_Temperature_4
        for i in range(1, 6):
            x[f'barrel_z_difference_{i}_{i+1}'] = x[f'Barrel_Temperature_{i}'] - x[f'Barrel_Temperature_{i+1}']
    elif variant == 'pressure_difference':
        x['back_pressure_z_difference'] = x.Max_Back_Pressure - x.Average_Back_Pressure
    elif variant not in ('base', 'shallower_tree'):
        raise ValueError(variant)
    return x

def main(base, out):
    if out.exists():
        raise FileExistsError(out)
    cfg = json.loads((base/'config.json').read_text())
    manifest = json.loads((base/'manifest.json').read_text())
    for name, digest in manifest['input_sha256'].items():
        assert sha(PROJECT/'dataset'/name) == digest, name
    splits = json.loads((base/'splits.json').read_text())
    features = json.loads((base/'preflight.json').read_text())['retained_features']
    reference = pd.read_csv(base/'oof_predictions.csv')
    prepared = {}
    for scenario in ('cn7', 'rg3'):
        df = pd.read_csv(PROJECT/'dataset'/f'moldset_labeled_{scenario}.csv')
        rows = pd.read_csv(base/f'{scenario}_rows.csv')
        assert np.array_equal(rows.row_position, np.arange(len(df)))
        assert np.array_equal(rows.PassOrFail, df.PassOrFail)
        groups = df.groupby(features, sort=True).ngroup().to_numpy()
        # Verify the equivalence classes of stored groups against raw features.
        mapping = pd.DataFrame({'raw': groups, 'saved': rows.feature_group})
        assert mapping.groupby('raw').saved.nunique().max() == 1
        assert mapping.groupby('saved').raw.nunique().max() == 1
        prepared[scenario] = (df[features], df.PassOrFail.to_numpy(), rows)
    out.mkdir(parents=True)
    protocol = {'variants': VARIANTS, 'models': MODELS, 'scenarios': ['cn7','rg3'],
                'selection': 'highest inner OOF AP; ties prefer earlier variant',
                'threshold': 'inner OOF F1 only', 'target': 'class_1, semantics unconfirmed',
                'deployment_assumption': 'after cycle completion, before quality inspection',
                'ranking_budget': 0.1, 'independent_test': False,
                'feature_units': 'differences of provided standardized values, not physical units',
                'shallower_tree': {'max_depth_or_depth': 2},
                'source_hashes': {p.name:sha(p) for p in [base/'config.json',base/'splits.json',base/'oof_predictions.csv']},
                'raw_hashes': manifest['input_sha256'], 'code_sha256': sha(Path(__file__)),
                'baseline_helper_sha256': sha(PROJECT/'scripts/train_baseline.py')}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    shutil.copyfile(__file__,out/'improve_v1_snapshot.py')
    scores, selections, predictions = [], [], []
    for split in splits:
        scenario = split['scenario']
        if scenario not in prepared: continue
        x, y, rows = prepared[scenario]
        tr, va = np.array(split['train_indices']), np.array(split['validation_indices'])
        assert not set(rows.feature_group.iloc[tr]) & set(rows.feature_group.iloc[va])
        for name in MODELS:
            candidates = []
            for variant in VARIANTS:
                xx = transform(x, variant)
                cc = copy.deepcopy(cfg)
                if variant == 'shallower_tree':
                    cc['xgboost']['max_depth'] = 2
                    cc['catboost']['depth'] = 2
                inner_p = np.full(len(tr), np.nan)
                for inner in split['inner_splits_relative_to_train']:
                    it, iv = np.array(inner['train']), np.array(inner['validation'])
                    assert not set(rows.feature_group.iloc[tr[it]]) & set(rows.feature_group.iloc[tr[iv]])
                    m = make_model(name, cc, y[tr[it]])
                    m.fit(xx.iloc[tr[it]], y[tr[it]])
                    inner_p[iv] = probability(m, xx.iloc[tr[iv]])
                assert np.isfinite(inner_p).all()
                cutoff = select_threshold(y[tr], inner_p)
                inner_ap = average_precision_score(y[tr], inner_p)
                m = make_model(name, cc, y[tr])
                m.fit(xx.iloc[tr], y[tr])
                p = probability(m, xx.iloc[va])
                if variant == 'base':
                    old = reference[(reference.scenario == scenario)&(reference.seed == split['seed'])&(reference.fold == split['fold'])&(reference.model == name)].sort_values('row_position')
                    np.testing.assert_array_equal(old.row_position.to_numpy(), va)
                    np.testing.assert_allclose(old.p_class_1, p, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(old.threshold, cutoff, rtol=0, atol=1e-12)
                info = dict(scenario=scenario, seed=split['seed'], fold=split['fold'], model=name, variant=variant, inner_ap=inner_ap, threshold=cutoff)
                candidates.append((inner_ap, variant, p, cutoff))
                for policy, threshold in [('fixed_0.5',0.5),('inner_f1',cutoff)]:
                    scores.append({**info,'policy':policy,**metrics(y[va],p,p>=threshold,rows.tie_key.to_numpy()[va],cfg['ranking_budgets'])})
                predictions.append(pd.DataFrame({**info,'row_position':va,'PassOrFail':y[va],'p_class_1':p}))
            best = max(candidates, key=lambda c:c[0])
            ap, variant, p, cutoff = best
            selections.append(dict(scenario=scenario,seed=split['seed'],fold=split['fold'],model=name,selected_variant=variant,inner_ap=ap))
            for policy, threshold in [('fixed_0.5',0.5),('inner_f1',cutoff)]:
                scores.append(dict(scenario=scenario,seed=split['seed'],fold=split['fold'],model=name,variant='inner_selected',inner_ap=ap,threshold=cutoff,policy=policy,**metrics(y[va],p,p>=threshold,rows.tie_key.to_numpy()[va],cfg['ranking_budgets'])))
            print(split['key'],name,'selected:',variant,flush=True)
    results = pd.DataFrame(scores)
    results.to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'selections.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'oof_predictions.csv',index=False)
    keys = ['ap','f1','precision','recall','recall_at_10pct']
    summary = results.groupby(['scenario','model','variant','policy'])[keys].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/'summary.csv',index=False)
    reference_scores = results[results.variant=='base'][['scenario','seed','fold','model','policy',*keys]]
    paired = results.merge(reference_scores,on=['scenario','seed','fold','model','policy'],suffixes=('','_base'))
    for k in keys: paired[k+'_delta'] = paired[k]-paired[k+'_base']
    paired.to_csv(out/'paired_comparison.csv',index=False)
    lines = ['# 소규모 개선 실험 v1','', 'CN7·RG3 각각 XGBoost/CatBoost balanced를 비교했습니다. 기존 baseline_v2의 원본·외부/내부 그룹 분할을 그대로 사용하며 base 예측과 임계값 일치를 확인했습니다.', '', '파생변수 두 묶음과 깊이 2 설정을 각각 단독 비교했습니다. 조합 탐색은 하지 않았습니다. inner_selected는 외부 평가 점수를 보지 않고 내부 OOF AP로 후보를 선택한 절차입니다. 임계값도 내부 OOF에서 F1 기준으로 선택했습니다.', '', '전체 EDA를 보고 후보를 만든 개발 실험입니다. 외부 폴드도 이전에 분석했으므로 독립 최종 성능이 아닙니다. 표준편차는 반복 분할의 변동이며 신뢰구간이 아닙니다. 라벨 방향·원단위·사용 가능 시점은 미확인입니다.', '', '|데이터|모델|설정|AP|F1 (내부 임계값)|Recall@10%|', '|---|---|---|---:|---:|---:|']
    for row in summary.reset_index().query("policy == 'inner_f1'").itertuples():
        lines.append(f'|{row.scenario}|{row.model}|{row.variant}|{row.ap_mean:.4f}|{row.f1_mean:.4f}|{row.recall_at_10pct_mean:.4f}|')
    lines += ['', 'AP와 Recall@10%는 임계값 변경으로 달라지지 않습니다. fixed_0.5/inner_f1 분류 성능은 summary.csv에서 비교하세요. 운영 비용·허용 오탐률이 정해지지 않아 최종 임계값은 확정하지 않았습니다.', '', '상세 재현: protocol.json, fold_metrics.csv, paired_comparison.csv, selections.csv. 최종 배포 모델·미라벨 제출 예측은 생성하지 않았습니다.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    for name, digest in manifest['input_sha256'].items(): assert sha(PROJECT/'dataset'/name)==digest
    (out/'verification.json').write_text(json.dumps({'status':'passed','baseline_predictions_and_thresholds':'identical within 1e-12','raw_unchanged':True,'group_disjoint':True,'outer_fits':144,'independent_evaluation':False},indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-dir',type=Path,default=PROJECT/'artifacts/baseline_v2')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4): main(args.baseline_dir,args.output_dir)
