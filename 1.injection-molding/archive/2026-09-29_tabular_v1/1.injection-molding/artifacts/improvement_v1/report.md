# 소규모 개선 실험 v1

CN7·RG3 각각 XGBoost/CatBoost balanced를 비교했습니다. 기존 baseline_v2의 원본·외부/내부 그룹 분할을 그대로 사용하며 base 예측과 임계값 일치를 확인했습니다.

파생변수 두 묶음과 깊이 2 설정을 각각 단독 비교했습니다. 조합 탐색은 하지 않았습니다. inner_selected는 외부 평가 점수를 보지 않고 내부 OOF AP로 후보를 선택한 절차입니다. 임계값도 내부 OOF에서 F1 기준으로 선택했습니다.

전체 EDA를 보고 후보를 만든 개발 실험입니다. 외부 폴드도 이전에 분석했으므로 독립 최종 성능이 아닙니다. 표준편차는 반복 분할의 변동이며 신뢰구간이 아닙니다. 라벨 방향·원단위·사용 가능 시점은 미확인입니다.

|데이터|모델|설정|AP|F1 (내부 임계값)|Recall@10%|
|---|---|---|---:|---:|---:|
|cn7|catboost_balanced|base|0.4773|0.2667|0.9667|
|cn7|catboost_balanced|inner_selected|0.4541|0.2278|0.9397|
|cn7|catboost_balanced|pressure_difference|0.5078|0.2632|0.9444|
|cn7|catboost_balanced|shallower_tree|0.4992|0.2470|0.9101|
|cn7|catboost_balanced|temperature_differences|0.4618|0.2620|0.9053|
|cn7|xgboost_balanced|base|0.4329|0.2033|0.9101|
|cn7|xgboost_balanced|inner_selected|0.3990|0.2086|0.8878|
|cn7|xgboost_balanced|pressure_difference|0.4245|0.1808|0.8693|
|cn7|xgboost_balanced|shallower_tree|0.4139|0.1459|0.9101|
|cn7|xgboost_balanced|temperature_differences|0.3315|0.2225|0.8693|
|rg3|catboost_balanced|base|0.0289|0.0263|0.0646|
|rg3|catboost_balanced|inner_selected|0.0269|0.0176|0.0507|
|rg3|catboost_balanced|pressure_difference|0.0278|0.0320|0.0744|
|rg3|catboost_balanced|shallower_tree|0.0269|0.0241|0.1014|
|rg3|catboost_balanced|temperature_differences|0.0264|0.0245|0.0646|
|rg3|xgboost_balanced|base|0.0383|0.0260|0.1488|
|rg3|xgboost_balanced|inner_selected|0.0280|0.0271|0.1128|
|rg3|xgboost_balanced|pressure_difference|0.0382|0.0316|0.1874|
|rg3|xgboost_balanced|shallower_tree|0.0351|0.0177|0.1400|
|rg3|xgboost_balanced|temperature_differences|0.0290|0.0431|0.0989|

AP와 Recall@10%는 임계값 변경으로 달라지지 않습니다. fixed_0.5/inner_f1 분류 성능은 summary.csv에서 비교하세요. 운영 비용·허용 오탐률이 정해지지 않아 최종 임계값은 확정하지 않았습니다.

상세 재현: protocol.json, fold_metrics.csv, paired_comparison.csv, selections.csv. 최종 배포 모델·미라벨 제출 예측은 생성하지 않았습니다.
