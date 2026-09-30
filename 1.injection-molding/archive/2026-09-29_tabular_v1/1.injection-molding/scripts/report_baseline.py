"""Generate a model-agnostic report and figure from saved baseline metrics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
LABELS = {'dummy_prior': 'Dummy', 'logistic': 'Logistic', 'logistic_balanced': 'Logistic weighted',
          'xgboost': 'XGBoost', 'xgboost_balanced': 'XGBoost weighted',
          'extratrees': 'ExtraTrees', 'extratrees_balanced': 'ExtraTrees weighted',
          'catboost': 'CatBoost', 'catboost_balanced': 'CatBoost weighted',
          'lightgbm': 'LightGBM', 'lightgbm_balanced': 'LightGBM weighted'}


def main(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    cfg = json.loads((run / 'config.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete run')
    summary = pd.read_csv(run / 'summary.csv')
    summary = summary[summary.policy == 'inner_f1'].copy()
    folds = pd.read_csv(run / 'fold_metrics.csv')
    scenarios = [s for s in cfg['scenarios'] if s in summary.scenario.unique()]
    models = [m for m in cfg['models'] if m in summary.model.unique()]
    order = {(s, m): i for i, (s, m) in enumerate((s, m) for s in scenarios for m in models)}
    summary['_order'] = [order[(s, m)] for s, m in zip(summary.scenario, summary.model)]
    summary = summary.sort_values('_order')
    fold_count = cfg['outer_folds'] * len(cfg['outer_seeds'])
    lines = [f"# {cfg['run_name']} 학습·평가 결과", '',
             'positive=1인 class_1 탐색 결과입니다. 불량 라벨 의미는 미확인입니다. 모델 점수는 확률 보정되지 않았습니다.', '',
             '## 실험 조건', '',
             f"- 데이터 시나리오: {', '.join(scenarios)}. pooled는 두 파일 통합 + 제품군 indicator입니다.",
             f"- {cfg['outer_folds']}-fold × {len(cfg['outer_seeds'])} seeds {cfg['outer_seeds']}. 동일 특성 그룹의 학습/평가 분리.",
             f"- 모델 설정 {len(models)}개: {', '.join(models)}. CPU 실행. 파생변수 추가·대규모 튜닝 없음.",
             f'- 아래 값은 외부 {fold_count}개 fold 평균±표준편차입니다. 표준편차는 신뢰구간이 아닙니다.',
             '- F1/Recall/Precision은 외부 train 내부 그룹 OOF에서 선택한 임계값으로 계산하며 Dummy는 0.5를 유지합니다.',
             '- AP와 Recall@10%는 임계값과 무관합니다. 검사 건수는 ceil(n*0.1). 동점은 정답과 독립적인 고정 순열로 처리합니다.']
    reference = manifest.get('reference_check')
    if reference:
        lines.append(f"- 기준 실행 {reference['run']}: 원본 SHA-256, 특성, 외부·내부 split 및 기존 설정 일치를 학습 전 확인했습니다.")
    verification_path = run / 'verification.json'
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        lines.append(f"- 추가 검증: {verification.get('summary', 'verification.json 참조')}")
    lines += ['', '## 지표별 가장 높은 평균 (탐색 결과)', '',
              '같은 평가에서 후보를 비교한 순위입니다. 통계적 우월성이나 최종 선정 결과가 아닙니다. AP는 정확도가 아닙니다.', '',
              '| 데이터 | 지표 | 모델 | 평균 ± SD |', '|---|---|---|---:|']
    for scenario in scenarios:
        part = summary[summary.scenario == scenario]
        for metric, label in [('ap', 'AP'), ('f1', 'F1'), ('recall_at_10pct', 'Recall@10%')]:
            row = part.loc[part[metric + '_mean'].idxmax()]
            lines.append(f"| {scenario} | {label} | {row['model']} | {row[metric+'_mean']:.4f} ± {row[metric+'_std']:.4f} |")
    lines += ['', '## 전체 모델 비교', '',
              '| 데이터 | 모델 | AP 평균±SD | F1 평균±SD | Precision | Recall | Recall@10% |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.scenario} | {row.model} | {row.ap_mean:.4f} ± {row.ap_std:.4f} | {row.f1_mean:.4f} ± {row.f1_std:.4f} | {row.precision_mean:.4f} | {row.recall_mean:.4f} | {row.recall_at_10pct_mean:.4f} |')
    lines += ['', '## 해석과 한계', '',
              '- 모델별 초기 설정은 고정했지만 알고리즘마다 유효 복잡도와 가중치 구현은 다릅니다. 최적 튜닝 성능 비교가 아닙니다.',
              '- 주 비교 지표는 AP이며 F1과 검사 우선순위 지표도 함께 봅니다. 가장 좋은 단일 fold만 선택하지 않습니다.',
              '- scenario별 class_1 비율과 분할이 다르므로 CN7/RG3/pooled의 AP를 단순 비교해 통합 효과를 확정하지 않습니다.',
              '- Dummy AP는 fold class_1 비율이며, 무작위 검사 기대 Recall@10%는 약 10%입니다. 저장된 Dummy 순위는 고정 난수의 한 실현값입니다.',
              '- 가중치를 적용해도 성능 개선을 보장하지 않습니다. 미보정 점수를 실제 불량 발생확률로 해석하지 않습니다.',
              '- 희귀 클래스 수가 적어 내부 F1 임계값도 불안정할 수 있습니다. 임계값 0.5 결과는 fold_metrics.csv와 summary.csv에서 비교할 수 있습니다.',
              '- 같은 입력에 반대 라벨이 있는 행은 현재 입력만으로 구별할 수 없습니다. 임의로 삭제하거나 라벨 오류로 확정하지 않았습니다.',
              '- 반복 CV는 새 독립 표본이 아닙니다. 모델 선택 이후 독립 최종 평가나 배포 성능을 보장하지 않습니다.',
              '- 파일별 표준화 이력과 라벨 의미가 미확인입니다. 시간/배치 구조를 확인하면 검증 설계를 재검토해야 합니다.', '',
              '## 선택 임계값 범위', '', '| 데이터 | 모델 | 최소 | 최대 |', '|---|---|---:|---:|']
    for (scenario, model), part in folds[folds.policy == 'inner_f1'].groupby(['scenario', 'model']):
        lines.append(f'| {scenario} | {model} | {part.selected_threshold.min():.6f} | {part.selected_threshold.max():.6f} |')
    lines += ['', '## 검증 및 산출물', '',
              f"- 외부 모델 {manifest['outer_models']}개. 학습 루프 약 {manifest['elapsed_seconds']:.1f}초(CPU, 내부 임계값 학습 포함).",
              '- 모든 외부 모델 저장·재로드 예측 일치와 원본 SHA-256 불변 확인.',
              '- 내부/외부 그룹 겹침 없음, 모든 fold 양 클래스 존재, 반복·모델당 원본 각 행 OOF 1회 확인.',
              '- summary.csv: 모델/시나리오/임계값 정책별 fold 평균·표준편차.',
              '- fold_metrics.csv: 각 fold의 지표, 선택 임계값, 실행시간.',
              '- repeat_oof_metrics.csv: 반복별 전체 OOF 및 pooled 모델의 제품군별 지표. fold 간 점수 척도 차이에 유의.',
              '- oof_predictions.csv: 원본 행·그룹·충돌 여부·점수·판정. models/: 개발용 fold 모델.',
              '- config.json, splits.json, manifest.json, 코드 snapshot: 재현 정보.',
              '- comparison.png: AP/F1/Recall@10%의 평균±표준편차. training.log: 학습 출력.', '',
              '## 다음 단계', '',
              '1. 전체 baseline 표를 확인한 뒤 CN7/RG3별 클래스 분포와 모델 오류 사례를 EDA로 분석합니다.',
              '2. 후보 모델이 공통으로 놓치는 행과 서로 다르게 예측하는 행을 살펴봅니다.',
              '3. 그 근거로 전처리/파생변수 후보를 정하고 동일 그룹 분할에서 처리 전후를 비교합니다.',
              '4. 개선 근거가 확인된 후보만 튜닝·확률 보정·최종 모델 선정으로 진행합니다.']
    (run / 'report.md').write_text('\n'.join(lines) + '\n')
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(18, max(5, len(scenarios)*5.6)), squeeze=False, constrained_layout=True)
    palette = plt.get_cmap('tab20')
    for row_idx, scenario in enumerate(scenarios):
        part = summary[summary.scenario == scenario].set_index('model').reindex(models).dropna(subset=['ap_mean'])
        labels = [LABELS.get(m, m) for m in part.index]
        for col, (metric, label) in enumerate([('ap', 'Average precision'), ('f1', 'F1 (inner threshold)'), ('recall_at_10pct', 'Recall at top 10%')]):
            ax = axes[row_idx, col]
            positions = np.arange(len(part))
            ax.barh(positions, part[metric+'_mean'], xerr=part[metric+'_std'].fillna(0), capsize=2,
                    color=[palette(models.index(m) % 20) for m in part.index])
            ax.set_yticks(positions, labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_title(f'{scenario.upper()} | {label}')
            ax.set_xlim(0, max(1.05, float((part[metric+'_mean']+part[metric+'_std'].fillna(0)).max())+0.03))
            ax.grid(axis='x', alpha=.2)
            if metric == 'recall_at_10pct':
                ax.axvline(.1, color='gray', linestyle='--', linewidth=1)
    fig.suptitle(f"{cfg['run_name']} | Class 1 | {cfg['outer_folds']} folds x {len(cfg['outer_seeds'])} seeds | Mean +/- SD (not CI)")
    fig.savefig(run / 'comparison.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=PROJECT/'artifacts/baseline_v2')
    main(parser.parse_args().run_dir)
