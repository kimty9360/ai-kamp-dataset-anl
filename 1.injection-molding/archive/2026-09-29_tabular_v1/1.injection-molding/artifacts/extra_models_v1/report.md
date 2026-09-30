# 추가 모델 비교: 좌우 정보 포함

기존 11설정과 SVM 2설정·Random Forest 2설정·GaussianNB·DNN 1설정씩을 합쳐 총 17설정이다. CN7/RG3/pooled에서 기존 3 seeds×3 folds와 내부 그룹 분할을 그대로 사용했다. 새 162개 외부 모델을 학습했다.

DNN은 sklearn MLPClassifier로 구현한 은닉층 64·32의 완전연결 신경망이다. 고정 200 epoch로 학습하며 random early stopping 검증을 만들지 않는다. TensorFlow/Keras 가이드북 모델이나 미라벨 의사라벨링을 재현한 것이 아니다. 수렴 진단은 fit_diagnostics.csv에 저장한다. 고정 epoch 종료 경고는 숨겨진 실패가 아니라 해당 학습 예산의 한계다.

SVM은 내부 무작위 확률 보정을 사용하지 않는다. decision_function에 sigmoid를 적용한 순위 점수로 AP/ROC-AUC/Recall@10%를 평가하며, 0.5는 margin=0이다. 이 점수는 보정된 불량 확률이 아니므로 SVM의 Brier/logloss는 보고하지 않는다. 임계값은 모든 모델에서 기존 내부 OOF F1 규칙으로 정한다.

표는 9개 외부 폴드 평균. F1/Accuracy는 inner_f1 정책이다. 표준편차는 summary.csv에서 확인한다.

|데이터|모델|AP|F1|Recall@10%|Accuracy|
|---|---|---:|---:|---:|---:|
|cn7|catboost_balanced|0.5541|0.3116|0.9000|0.9700|
|cn7|logistic|0.5517|0.4971|0.9090|0.9882|
|cn7|logistic_balanced|0.5494|0.4196|0.9005|0.9829|
|cn7|catboost|0.5430|0.4218|0.8074|0.9816|
|cn7|extratrees|0.5283|0.3648|0.8667|0.9766|
|cn7|dnn_mlp|0.5090|0.3977|0.6418|0.9829|
|cn7|random_forest|0.5003|0.4749|0.7534|0.9879|
|cn7|lightgbm_balanced|0.4966|0.2664|0.8815|0.9681|
|cn7|svm_rbf|0.4645|0.4134|0.6050|0.9893|
|cn7|random_forest_balanced|0.4323|0.2173|0.8519|0.9568|
|cn7|xgboost_balanced|0.4234|0.2512|0.8481|0.9615|
|cn7|gaussian_nb|0.3850|0.4413|0.6712|0.9882|
|cn7|lightgbm|0.3715|0.2689|0.8630|0.9537|
|cn7|svm_rbf_balanced|0.3423|0.1837|0.7765|0.9626|
|cn7|extratrees_balanced|0.3281|0.2737|0.8664|0.9502|
|cn7|xgboost|0.1974|0.1764|0.6394|0.8960|
|cn7|dummy_prior|0.0140|0.0000|0.0455|0.9860|
|pooled|logistic|0.2185|0.1909|0.4108|0.9532|
|pooled|catboost|0.2103|0.2007|0.3798|0.9628|
|pooled|extratrees|0.2038|0.1725|0.4042|0.9579|
|pooled|gaussian_nb|0.1905|0.2070|0.3570|0.9650|
|pooled|random_forest|0.1881|0.1847|0.3506|0.9717|
|pooled|svm_rbf|0.1849|0.1237|0.2965|0.9705|
|pooled|logistic_balanced|0.1740|0.1291|0.4430|0.9739|
|pooled|catboost_balanced|0.1658|0.1334|0.3639|0.9430|
|pooled|dnn_mlp|0.1459|0.1668|0.3463|0.9657|
|pooled|extratrees_balanced|0.1438|0.1108|0.4609|0.9312|
|pooled|lightgbm|0.1431|0.1197|0.3944|0.9457|
|pooled|xgboost|0.1421|0.1401|0.4214|0.9209|
|pooled|lightgbm_balanced|0.1421|0.1148|0.3804|0.9220|
|pooled|xgboost_balanced|0.1214|0.1227|0.4106|0.9487|
|pooled|random_forest_balanced|0.1200|0.1193|0.3870|0.9287|
|pooled|svm_rbf_balanced|0.0809|0.0917|0.3607|0.9103|
|pooled|dummy_prior|0.0176|0.0000|0.0513|0.9824|
|rg3|xgboost_balanced|0.0753|0.0609|0.2603|0.8063|
|rg3|svm_rbf|0.0709|0.0658|0.2430|0.8765|
|rg3|lightgbm_balanced|0.0696|0.0677|0.2373|0.7916|
|rg3|extratrees_balanced|0.0616|0.0566|0.2173|0.8167|
|rg3|catboost_balanced|0.0572|0.0678|0.2033|0.7375|
|rg3|random_forest_balanced|0.0567|0.0539|0.1995|0.7642|
|rg3|lightgbm|0.0566|0.0499|0.1960|0.8393|
|rg3|logistic_balanced|0.0563|0.0598|0.1144|0.7600|
|rg3|catboost|0.0542|0.0625|0.1339|0.6404|
|rg3|dnn_mlp|0.0531|0.0400|0.2475|0.8923|
|rg3|logistic|0.0521|0.0525|0.1122|0.7296|
|rg3|extratrees|0.0515|0.0645|0.1886|0.6861|
|rg3|xgboost|0.0488|0.0487|0.1117|0.6557|
|rg3|svm_rbf_balanced|0.0481|0.0487|0.2033|0.8556|
|rg3|random_forest|0.0477|0.0173|0.1110|0.7462|
|rg3|gaussian_nb|0.0445|0.0494|0.0785|0.7259|
|rg3|dummy_prior|0.0212|0.0000|0.0441|0.9788|

이 결과는 제한된 기본 설정의 개발 비교다. 같은 데이터로 EDA와 여러 모델 선택을 반복했으므로 가장 높은 평균을 독립 최종 성능으로 주장하지 않는다. 가이드북의 평가 불량수·분할·준지도 방식과 다르다. 데이터 표준화 및 CN7 시간대 집중의 한계도 유지된다. 기존 baseline·side_feature 파일은 해시로 변경 없음을 확인한다.
