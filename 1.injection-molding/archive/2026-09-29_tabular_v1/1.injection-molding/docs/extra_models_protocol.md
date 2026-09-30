# 가이드북 등장 모델의 추가 지도학습 비교

## 범위

SVM(RBF) 기본/balanced, Random Forest 기본/balanced, Gaussian Naive Bayes, DNN을 추가한다. 기존 side_feature_v1의 11설정과 합쳐 17설정을 비교한다. 기존 결과는 읽기만 하며 원본·baseline·side_feature의 해시 불변을 확인한다.

입력은 공정값 23개와 part_is_rh이며 pooled는 product_is_rg3도 포함한다. 원자료에서의 좌우 복구와 표준화 재현 검증, baseline_v2의 외부/내부 그룹 분할, seed와 tie_key, 내부 F1 임계값 규칙을 유지한다. 같은 설비·같은 시각 기록도 폴드 사이에 겹치지 않는지 확인한다. CatBoost balanced의 시드42·첫 외부 폴드를 제품별/pooled에서 재학습하여 기존 좌우 실험의 예측과 일치하는지 확인한다.

## 고정 설정

- SVM: RBF, C=1, gamma=scale. 기본 및 class_weight=balanced. 확률 보정을 위한 내부 무작위 교차검증을 피하기 위해 probability=False. sigmoid(decision_function)를 순위 점수로 사용한다. 임계값0.5는 원래 SVM의 margin0과 같고 내부 F1 임계값도 비교한다. 보정된 확률이 아니므로 Brier/logloss는 비워둔다.
- Random Forest: 200 trees, max_depth8, min_samples_leaf3, max_features=sqrt, 기본/balanced. 기본설정의 소규모 비교이며 최적 Random Forest라고 주장하지 않는다.
- GaussianNB: var_smoothing1e-9.
- DNN: scikit-learn MLPClassifier, 은닉층64·32(ReLU), 이진 출력, Adam, learning_rate0.001, batch64, L2 alpha0.0001, 고정200 epochs, random_state42. 조기 종료용 행단위 랜덤 분할을 만들지 않는다. 클래스 가중치·재샘플링은 적용하지 않는다. 고정 학습 예산의 ConvergenceWarning·손실·반복횟수는 fit_diagnostics.csv에 기록한다.

RF 외 모델은 학습 폴드 안에서 상수열 제거와 StandardScaler를 fit한다. RF는 상수열만 제거한다. 제공된 파일의 사전 표준화 한계는 유지된다.

## 해석과 가이드북 차이

이번 DNN은 두 은닉층의 완전연결 지도학습 신경망이며 Keras 코드, 가이드북 구조·튜닝·의사라벨링을 재현한 것이 아니다. 가이드북의 학습/평가 표본수·분할·샘플링 방식 차이는 그대로여서 성능표를 일대일 재현했다고 주장하지 않는다. 새 알고리즘이 동일 조건에서 기존 모델보다 나은지 알아보는 비교다.

독립 평가 데이터 없이 같은 데이터에서 반복 선택한 개발 결과다. AP를 주 지표로 정렬하고 F1, Recall@10%, 정밀도/재현율, Accuracy와 폴드 간 변동을 함께 본다. 종합 평균 점수를 만들지 않는다.

## 실행

```bash
conda activate kamp-base
python 1.injection-molding/scripts/compare_extra_models.py --output-dir 1.injection-molding/artifacts/extra_model_runs/new_run
```

기존 경로는 덮어쓰지 않는다. 노트북은 05_extra_models.ipynb, 기본 RUN_TRAINING=False로 저장 결과를 확인한다. 원본 ZIP·CSV와 baseline_v2·side_feature_v1 로컬 산출물이 재학습에 필요하다. 저장 모델·미라벨 제출 예측은 생성하지 않는다.
