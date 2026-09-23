"""Generate reviewable baseline report and figure from saved metrics only."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]


def main(run):
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='complete': raise ValueError('Incomplete run')
    summary=pd.read_csv(run/'summary.csv')
    summary=summary[summary.policy=='inner_f1']
    folds=pd.read_csv(run/'fold_metrics.csv')
    lines=['# 첫 baseline 학습·평가 결과', '',
           'positive=1인 class_1 탐색 결과입니다. 불량 라벨 의미는 미확인입니다. 모델 점수는 확률 보정되지 않았습니다.', '',
           '## 실험 조건', '',
           '- CN7, RG3 개별 학습 및 CN7+RG3 통합 학습. 통합 모델만 제품군 indicator 사용.',
           '- 3-fold × 3 seeds [42, 1729, 2026], 동일 특성 그룹의 학습/평가 분리. 각 scenario 내 모든 모델이 같은 분할 사용.',
           '- Dummy, Logistic 기본/가중치, XGBoost 기본/가중치. CPU 실행. 대규모 튜닝 없음.',
           '- 아래 값은 외부 9개 fold 평균±표준편차입니다. 표준편차는 신뢰구간이 아닙니다.',
           '- F1/Recall/Precision은 외부 train 내부의 그룹 OOF에서 F1을 최대화한 임계값으로 계산. Dummy는 0.5 유지.',
           '- AP와 Recall@10%는 위 임계값과 무관한 점수/순위 지표입니다. 검사 건수는 ceil(n*0.1).', '',
           '## 모델 비교', '',
           '| 데이터 | 모델 | AP 평균±SD | F1 평균±SD | Precision | Recall | Recall@10% |',
           '|---|---|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.scenario} | {row.model} | {row.ap_mean:.4f} ± {row.ap_std:.4f} | {row.f1_mean:.4f} ± {row.f1_std:.4f} | {row.precision_mean:.4f} | {row.recall_mean:.4f} | {row.recall_at_10pct_mean:.4f} |')
    lines += ['', '## 해석과 한계', '',
              '- CN7에서는 Logistic 계열 AP가 높고 가중치 XGBoost의 Recall@10%가 높습니다. 모델 순위는 운영 목표에 따라 달라집니다.',
              '- RG3에서는 전체적으로 AP가 낮고 변동이 큽니다. class_1 정답의 충돌, 누락 변수, 파일 전처리 이력 등을 우선 확인해야 하며 원인을 아직 확정하지 않습니다.',
              '- 통합 모델 결과만으로 제품군별 모델보다 열등하다고 확정할 수 없습니다. scenario 사이 분할은 서로 다르고 유병률도 다르며, 동일 행 분할의 paired 비교는 후속 작업입니다.',
              '- Dummy AP는 fold class_1 비율입니다. Dummy Recall@10%는 고정 무작위 동점 순서의 한 실현값이므로 무작위 검사 기대 약 10%와 별개입니다.',
              '- 내부 threshold 최적화도 희귀 클래스 수가 적어 불안정합니다. 보정이나 독립 운영 임계값 검증 없이 현장에 적용하지 않습니다.',
              '- 분할 간 score 척도가 다를 수 있으므로 기본 비교는 fold별 지표입니다. 반복별 합친 OOF 지표는 별도 CSV에 저장했습니다.',
              '- 같은 입력의 상반된 라벨은 현재 입력만으로 구별할 수 없습니다. 이를 자동으로 라벨 오류로 단정하거나 삭제하지 않았습니다.',
              '- 표본 반복은 새로운 독립 데이터가 아닙니다. 탐색적 모델 비교이며 최종 모델/독립 최종 평가/미라벨 제출 예측은 아닙니다.', '',
              '## 선택 임계값 범위', '', '| 데이터 | 모델 | 최소 | 최대 |', '|---|---|---:|---:|']
    for (scenario,model), part in folds[folds.policy=='inner_f1'].groupby(['scenario','model']):
        lines.append(f'| {scenario} | {model} | {part.selected_threshold.min():.6f} | {part.selected_threshold.max():.6f} |')
    lines += ['', '## 검증 및 산출물', '',
              f"- 외부 모델 {manifest['outer_models']}개와 내부 임계값 학습 완료. 학습 루프 시간 약 {manifest['elapsed_seconds']:.1f}초(CPU).",
              '- 모든 외부 모델 저장/재로드 예측 일치, 입력 SHA-256 불변 확인.',
              '- 내부/외부 그룹 겹침 없음, 모든 fold 양 클래스 존재, 반복·모델당 원본 각 행 OOF 1회 확인.',
              '- `fold_metrics.csv`: 외부 fold별 지표와 실행시간, 0.5/내부 F1 임계값 정책 비교.',
              '- `summary.csv`: fold 평균·표준편차.',
              '- `repeat_oof_metrics.csv`: 반복별 전체 OOF 및 통합 모델의 제품군별 지표.',
              '- `oof_predictions.csv`: 행·그룹·충돌 추적 및 모델별 확률/판정.',
              '- `splits.json`, `config.json`, `manifest.json`, 코드 snapshot, `models/`: 재현 정보.',
              '- `comparison.png`: AP와 Recall@10% 비교. 막대는 fold 평균, 오차막대는 fold 표준편차.', '',
              '## 다음 분석', '',
              '1. CN7의 Logistic과 가중치 XGBoost가 잡는/놓치는 행 비교.',
              '2. RG3 충돌 라벨과 class_1 공정 분포, 원본 데이터 설명 확인.',
              '3. 같은 외부 분할로 제품군별/통합 모델을 비교하고 임계값 안정성 검토.',
              '4. 개선 근거가 있는 경우에만 튜닝·후보 모델·확률 보정 확대.']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    fig, axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for col, scenario in enumerate(['cn7','rg3','pooled']):
        part=summary[summary.scenario==scenario].set_index('model').loc[['dummy_prior','logistic','logistic_balanced','xgboost','xgboost_balanced']]
        for row,metric in enumerate(['ap','recall_at_10pct']):
            ax=axes[row,col]
            ax.bar(np.arange(len(part)),part[metric+'_mean'],yerr=part[metric+'_std'],capsize=3,color=['#999999','#3678aa','#7bb4d9','#da8232','#eab87c'])
            ax.set_xticks(np.arange(len(part)),['Dummy','LR','LR weighted','XGB','XGB weighted'],rotation=35,ha='right')
            ax.set_title(scenario.upper())
            ax.set_ylabel('Average precision' if row==0 else 'Recall at top 10%')
            ax.set_ylim(0,1.2 if row else 1)
            if row==1: ax.axhline(.1,color='gray',linestyle='--',linewidth=1,label='Random expectation ~10%')
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('Class 1 baseline | Group CV: 3 folds x 3 seeds | Mean +/- SD (not CI)')
    fig.savefig(run/'comparison.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'artifacts/baseline_v1')
    main(parser.parse_args().run_dir)
