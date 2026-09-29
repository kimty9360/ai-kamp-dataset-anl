# baseline_v2 학습·평가 결과

positive=1인 class_1 탐색 결과입니다. 불량 라벨 의미는 미확인입니다. 모델 점수는 확률 보정되지 않았습니다.

## 실험 조건

- 데이터 시나리오: cn7, rg3, pooled. pooled는 두 파일 통합 + 제품군 indicator입니다.
- 3-fold × 3 seeds [42, 1729, 2026]. 동일 특성 그룹의 학습/평가 분리.
- 모델 설정 11개: dummy_prior, logistic, logistic_balanced, xgboost, xgboost_balanced, extratrees, extratrees_balanced, catboost, catboost_balanced, lightgbm, lightgbm_balanced. CPU 실행. 파생변수 추가·대규모 튜닝 없음.
- 아래 값은 외부 9개 fold 평균±표준편차입니다. 표준편차는 신뢰구간이 아닙니다.
- F1/Recall/Precision은 외부 train 내부 그룹 OOF에서 선택한 임계값으로 계산하며 Dummy는 0.5를 유지합니다.
- AP와 Recall@10%는 임계값과 무관합니다. 검사 건수는 ceil(n*0.1). 동점은 정답과 독립적인 고정 순열로 처리합니다.
- 기준 실행 baseline_v1: 원본 SHA-256, 특성, 외부·내부 split 및 기존 설정 일치를 학습 전 확인했습니다.
- 추가 검증: 기존 5개 설정의 모든 fold 지표·OOF 예측 재현, 추가 6개 설정의 실제 fold 동일 seed 재학습 일치.

## 지표별 가장 높은 평균 (탐색 결과)

같은 평가에서 후보를 비교한 순위입니다. 통계적 우월성이나 최종 선정 결과가 아닙니다. AP는 정확도가 아닙니다.

| 데이터 | 지표 | 모델 | 평균 ± SD |
|---|---|---|---:|
| cn7 | AP | logistic_balanced | 0.5484 ± 0.1353 |
| cn7 | F1 | logistic | 0.5007 ± 0.1307 |
| cn7 | Recall@10% | catboost_balanced | 0.9667 ± 0.1000 |
| rg3 | AP | lightgbm_balanced | 0.0383 ± 0.0229 |
| rg3 | F1 | logistic_balanced | 0.0391 ± 0.0188 |
| rg3 | Recall@10% | lightgbm_balanced | 0.1692 ± 0.0808 |
| pooled | AP | logistic | 0.1854 ± 0.0646 |
| pooled | F1 | logistic | 0.2018 ± 0.1015 |
| pooled | Recall@10% | catboost | 0.3576 ± 0.1025 |

## 전체 모델 비교

| 데이터 | 모델 | AP 평균±SD | F1 평균±SD | Precision | Recall | Recall@10% |
|---|---|---:|---:|---:|---:|---:|
| cn7 | dummy_prior | 0.0140 ± 0.0053 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0455 |
| cn7 | logistic | 0.5428 ± 0.1387 | 0.5007 ± 0.1307 | 0.9111 | 0.3968 | 0.7876 |
| cn7 | logistic_balanced | 0.5484 ± 0.1353 | 0.2991 ± 0.2515 | 0.3796 | 0.2873 | 0.8209 |
| cn7 | xgboost | 0.1918 ± 0.1475 | 0.1692 ± 0.1064 | 0.1001 | 0.6487 | 0.6283 |
| cn7 | xgboost_balanced | 0.4329 ± 0.2387 | 0.2033 ± 0.1644 | 0.1350 | 0.5722 | 0.9101 |
| cn7 | extratrees | 0.5066 ± 0.1236 | 0.4822 ± 0.1151 | 0.7593 | 0.3968 | 0.8376 |
| cn7 | extratrees_balanced | 0.3669 ± 0.2385 | 0.2385 ± 0.0594 | 0.1556 | 0.6653 | 0.8598 |
| cn7 | catboost | 0.4883 ± 0.1285 | 0.4677 ± 0.2018 | 0.7222 | 0.3746 | 0.7280 |
| cn7 | catboost_balanced | 0.4773 ± 0.1883 | 0.2667 ± 0.1819 | 0.2087 | 0.4616 | 0.9667 |
| cn7 | lightgbm | 0.4154 ± 0.2547 | 0.2183 ± 0.0991 | 0.1367 | 0.6595 | 0.9074 |
| cn7 | lightgbm_balanced | 0.4744 ± 0.2110 | 0.1952 ± 0.1677 | 0.1241 | 0.5479 | 0.9185 |
| rg3 | dummy_prior | 0.0212 ± 0.0038 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0441 |
| rg3 | logistic | 0.0265 ± 0.0096 | 0.0270 ± 0.0190 | 0.0144 | 0.4355 | 0.0862 |
| rg3 | logistic_balanced | 0.0247 ± 0.0098 | 0.0391 ± 0.0188 | 0.0218 | 0.5093 | 0.0545 |
| rg3 | xgboost | 0.0264 ± 0.0203 | 0.0259 ± 0.0220 | 0.0134 | 0.4222 | 0.0464 |
| rg3 | xgboost_balanced | 0.0383 ± 0.0293 | 0.0260 ± 0.0342 | 0.0151 | 0.1898 | 0.1488 |
| rg3 | extratrees | 0.0281 ± 0.0143 | 0.0288 ± 0.0241 | 0.0158 | 0.3103 | 0.0575 |
| rg3 | extratrees_balanced | 0.0297 ± 0.0162 | 0.0212 ± 0.0217 | 0.0110 | 0.3616 | 0.0810 |
| rg3 | catboost | 0.0241 ± 0.0081 | 0.0315 ± 0.0157 | 0.0166 | 0.5356 | 0.0487 |
| rg3 | catboost_balanced | 0.0289 ± 0.0101 | 0.0263 ± 0.0200 | 0.0138 | 0.3750 | 0.0646 |
| rg3 | lightgbm | 0.0340 ± 0.0288 | 0.0242 ± 0.0245 | 0.0130 | 0.2535 | 0.1094 |
| rg3 | lightgbm_balanced | 0.0383 ± 0.0229 | 0.0354 ± 0.0518 | 0.0228 | 0.1922 | 0.1692 |
| pooled | dummy_prior | 0.0176 ± 0.0044 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0513 |
| pooled | logistic | 0.1854 ± 0.0646 | 0.2018 ± 0.1015 | 0.7822 | 0.2096 | 0.3179 |
| pooled | logistic_balanced | 0.1609 ± 0.0601 | 0.1247 ± 0.0983 | 0.2875 | 0.0970 | 0.2709 |
| pooled | xgboost | 0.1451 ± 0.0717 | 0.1549 ± 0.0729 | 0.1364 | 0.2702 | 0.3532 |
| pooled | xgboost_balanced | 0.1062 ± 0.0597 | 0.1242 ± 0.0820 | 0.1589 | 0.1399 | 0.2881 |
| pooled | extratrees | 0.1807 ± 0.0691 | 0.1927 ± 0.0911 | 0.6169 | 0.1479 | 0.3411 |
| pooled | extratrees_balanced | 0.1375 ± 0.0853 | 0.1110 ± 0.0733 | 0.1709 | 0.1877 | 0.2679 |
| pooled | catboost | 0.1825 ± 0.0677 | 0.1896 ± 0.1071 | 0.6817 | 0.2158 | 0.3576 |
| pooled | catboost_balanced | 0.1459 ± 0.0720 | 0.1118 ± 0.1038 | 0.3155 | 0.1311 | 0.2451 |
| pooled | lightgbm | 0.1521 ± 0.0828 | 0.0662 ± 0.0684 | 0.0670 | 0.1338 | 0.3496 |
| pooled | lightgbm_balanced | 0.1139 ± 0.0645 | 0.1221 ± 0.0973 | 0.1905 | 0.1594 | 0.2976 |

## 해석과 한계

- 모델별 초기 설정은 고정했지만 알고리즘마다 유효 복잡도와 가중치 구현은 다릅니다. 최적 튜닝 성능 비교가 아닙니다.
- 주 비교 지표는 AP이며 F1과 검사 우선순위 지표도 함께 봅니다. 가장 좋은 단일 fold만 선택하지 않습니다.
- scenario별 class_1 비율과 분할이 다르므로 CN7/RG3/pooled의 AP를 단순 비교해 통합 효과를 확정하지 않습니다.
- Dummy AP는 fold class_1 비율이며, 무작위 검사 기대 Recall@10%는 약 10%입니다. 저장된 Dummy 순위는 고정 난수의 한 실현값입니다.
- 가중치를 적용해도 성능 개선을 보장하지 않습니다. 미보정 점수를 실제 불량 발생확률로 해석하지 않습니다.
- 희귀 클래스 수가 적어 내부 F1 임계값도 불안정할 수 있습니다. 임계값 0.5 결과는 fold_metrics.csv와 summary.csv에서 비교할 수 있습니다.
- 같은 입력에 반대 라벨이 있는 행은 현재 입력만으로 구별할 수 없습니다. 임의로 삭제하거나 라벨 오류로 확정하지 않았습니다.
- 반복 CV는 새 독립 표본이 아닙니다. 모델 선택 이후 독립 최종 평가나 배포 성능을 보장하지 않습니다.
- 파일별 표준화 이력과 라벨 의미가 미확인입니다. 시간/배치 구조를 확인하면 검증 설계를 재검토해야 합니다.

## 선택 임계값 범위

| 데이터 | 모델 | 최소 | 최대 |
|---|---|---:|---:|
| cn7 | catboost | 0.089604 | 0.726452 |
| cn7 | catboost_balanced | 0.259641 | 0.846792 |
| cn7 | dummy_prior | 0.500000 | 0.500000 |
| cn7 | extratrees | 0.169111 | 0.447778 |
| cn7 | extratrees_balanced | 0.037035 | 0.648670 |
| cn7 | lightgbm | 0.014816 | 0.237692 |
| cn7 | lightgbm_balanced | 0.186685 | 0.950658 |
| cn7 | logistic | 0.194269 | 0.932047 |
| cn7 | logistic_balanced | 0.480329 | 0.999992 |
| cn7 | xgboost | 0.020064 | 0.092274 |
| cn7 | xgboost_balanced | 0.042543 | 0.930306 |
| pooled | catboost | 0.009318 | 0.682905 |
| pooled | catboost_balanced | 0.311217 | 0.915754 |
| pooled | dummy_prior | 0.500000 | 0.500000 |
| pooled | extratrees | 0.029142 | 0.454573 |
| pooled | extratrees_balanced | 0.256917 | 0.764393 |
| pooled | lightgbm | 0.025415 | 0.281460 |
| pooled | lightgbm_balanced | 0.279272 | 0.868396 |
| pooled | logistic | 0.007630 | 0.899132 |
| pooled | logistic_balanced | 0.880379 | 0.998072 |
| pooled | xgboost | 0.012311 | 0.092489 |
| pooled | xgboost_balanced | 0.290715 | 0.909377 |
| rg3 | catboost | 0.001795 | 0.054574 |
| rg3 | catboost_balanced | 0.001368 | 0.543999 |
| rg3 | dummy_prior | 0.500000 | 0.500000 |
| rg3 | extratrees | 0.000000 | 0.111554 |
| rg3 | extratrees_balanced | 0.000000 | 0.617348 |
| rg3 | lightgbm | 0.001903 | 0.088643 |
| rg3 | lightgbm_balanced | 0.005078 | 0.796299 |
| rg3 | logistic | 0.000239 | 0.118050 |
| rg3 | logistic_balanced | 0.000006 | 0.847941 |
| rg3 | xgboost | 0.010093 | 0.058322 |
| rg3 | xgboost_balanced | 0.003241 | 0.699808 |

## 검증 및 산출물

- 외부 모델 297개. 학습 루프 약 83.8초(CPU, 내부 임계값 학습 포함).
- 모든 외부 모델 저장·재로드 예측 일치와 원본 SHA-256 불변 확인.
- 내부/외부 그룹 겹침 없음, 모든 fold 양 클래스 존재, 반복·모델당 원본 각 행 OOF 1회 확인.
- summary.csv: 모델/시나리오/임계값 정책별 fold 평균·표준편차.
- fold_metrics.csv: 각 fold의 지표, 선택 임계값, 실행시간.
- repeat_oof_metrics.csv: 반복별 전체 OOF 및 pooled 모델의 제품군별 지표. fold 간 점수 척도 차이에 유의.
- oof_predictions.csv: 원본 행·그룹·충돌 여부·점수·판정. models/: 개발용 fold 모델.
- config.json, splits.json, manifest.json, 코드 snapshot: 재현 정보.
- comparison.png: AP/F1/Recall@10%의 평균±표준편차. training.log: 학습 출력.

## 다음 단계

1. 전체 baseline 표를 확인한 뒤 CN7/RG3별 클래스 분포와 모델 오류 사례를 EDA로 분석합니다.
2. 후보 모델이 공통으로 놓치는 행과 서로 다르게 예측하는 행을 살펴봅니다.
3. 그 근거로 전처리/파생변수 후보를 정하고 동일 그룹 분할에서 처리 전후를 비교합니다.
4. 개선 근거가 확인된 후보만 튜닝·확률 보정·최종 모델 선정으로 진행합니다.
