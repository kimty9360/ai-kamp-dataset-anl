# baseline_v2: 전체 baseline 모델 비교

이 실험은 baseline_v1의 모델 후보만 확장한다. 새로운 파생변수, 데이터 정제, 튜닝, SMOTE, 확률 보정, early stopping은 추가하지 않는다. 양성 클래스 1의 불량 의미는 미확인이다.

## 고정 및 검증

- CN7 / RG3 / pooled, 23개 공정 특성, pooled에만 제품군 indicator 추가.
- 3 outer folds × seeds [42, 1729, 2026], 각 outer train 안에서 3 inner group folds로 F1 임계값 선택.
- baseline_v1과 원본 SHA-256, 특성 정의, 외부·내부 split 인덱스, tie seed, 지표 정의 및 기존 모델 설정이 정확히 일치하는지 **학습 전에 검사**한다. 불일치하면 중단한다.
- 기존 Dummy/Logistic/XGBoost도 다시 실행하여 결과가 재현되는지 확인한다. v1 산출물은 덮어쓰지 않는다.
- 반복 fold 평균±표준편차는 신뢰구간이 아니다. 시나리오 간 비교는 동일한 평가행 분할의 paired 비교가 아니다.
- 기존 프로토콜의 데이터 보존, 누수 방지, OOF coverage 및 모델 재로드 검사를 그대로 유지한다.

## 모델 설정 (성능을 확인하기 전에 고정)

총 6개 계열 / 11개 설정: Dummy 1개와 Logistic, XGBoost, ExtraTrees, CatBoost, LightGBM 각각 기본/가중치 2개.

| 추가 모델 | 초기 설정 | 가중치 버전 |
|---|---|---|
| ExtraTrees | 150 trees, max_depth=8, min_samples_leaf=3, max_features=1.0 | class_weight=balanced |
| CatBoost | 150 iterations, depth=3, learning_rate=0.05, l2_leaf_reg=5 | auto_class_weights=Balanced |
| LightGBM | 150 trees, max_depth=3, num_leaves=7, learning_rate=0.05, min_child_samples=20, reg_lambda=5 | class_weight=balanced |

가중치는 각 모델이 실제로 fit하는 train fold에서 계산한다. 외부 평가 라벨로 가중치나 임계값을 정하지 않는다.
모두 CPU, 최대 4 threads를 사용한다. 트리 수·깊이를 일부 맞춰도 알고리즘별 유효 복잡도와 가중치 구현은 같지 않다. 이 비교는 동일 계산량이나 최적 튜닝 성능을 보장하는 벤치마크가 아니라 고정 초기 설정 비교다.
정확한 설정은 `configs/baseline_v2.json`에 저장한다.

## 실행

프로젝트 루트, kamp-base에서:

```bash
python -m unittest discover -s 1.injection-molding/tests -v
python 1.injection-molding/scripts/train_baseline.py --config 1.injection-molding/configs/baseline_v2.json
python 1.injection-molding/scripts/report_baseline.py --run-dir 1.injection-molding/artifacts/baseline_v2
```

`reference_run=baseline_v1`은 해당 실행의 로컬 split/manifest를 필요로 한다. 공개 GitHub에는 split 파일을 포함하지 않으므로 새 환경에서는 원본 데이터를 준비하고 baseline_v1을 새 경로에 재실행한 다음 reference_run에 그 실행 폴더 이름을 지정한다. 기준 실험의 원본 데이터와 설정은 동일해야 한다.
기준 설정 자체를 바꾸는 후속 실험은 별도 config/run 이름을 사용하고 reference_run을 제거한다. 그런 실험은 동일 조건의 baseline_v2 재현이라고 주장하지 않는다.

산출물은 `artifacts/baseline_v2/`: 전체 비교표, fold/반복별 지표, OOF, 모델, 설정, split, 실행 정보, 보고서·그래프. 결과 확인 후 모델 선정/EDA/오류 분석으로 이어가며 최종 배포 모델은 이번 범위에 포함하지 않는다.
