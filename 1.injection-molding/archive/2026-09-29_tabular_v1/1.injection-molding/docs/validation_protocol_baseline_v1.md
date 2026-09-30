# baseline_v1: 첫 탐색 학습 프로토콜

학습 전 고정한 설정은 `configs/baseline_v1.json`이다. 정답 의미가 확인되지 않아 positive=1인 class_1 이진분류로만 해석한다.

- CN7, RG3 개별 모델과 통합 모델을 비교한다. 통합 모델에만 제품군 indicator를 추가한다.
- ID는 제외하고 양쪽 labeled에서 0인 Clamp_Open_Position을 제외한다. 나머지 상수 제거와 Logistic 스케일링은 각 학습 fold에서 fit한다.
- 남은 23개 공정 특성의 정확한 일치로 제품군에 걸쳐 보수적으로 그룹을 만든다. 그룹에 라벨이나 ID를 넣지 않는다. 충돌 라벨과 원본 행은 보존한다.
- 외부 3-fold × seed [42, 1729, 2026]. 모델 간 외부/내부 split을 공유한다. 두 클래스 존재, 그룹 겹침 없음, 평가 행당 반복별 1회 coverage를 검사한다. 모든 분할은 학습 전에 저장한다.
- Dummy prior, Logistic 기본/균형 가중치, XGBoost 기본/균형 가중치: 5개 모델 설정. 가중치는 각 학습 fold에서 계산한다.
- XGBoost는 CPU hist, 깊이 3, 150 trees 등 고정 설정을 사용한다. 튜닝/early stopping/SMOTE/확률 보정은 하지 않는다. Logistic max_iter=2000.
- 기본 임계값 0.5 결과와 내부 3-fold 그룹 OOF에서 F1을 최대화한 임계값 결과를 모두 기록한다. F1 동률이면 높은 임계값을 선택한다. Dummy는 비정보적 기준선으로 0.5를 유지한다.
- 내부 threshold 최적화의 F1은 성능 추정으로 사용하지 않는다. 외부 평가에서만 성능을 보고한다. 희귀 클래스가 적어 threshold가 불안정할 수 있다.
- 주 지표 AP; F1(binary), Recall, Precision, MCC, ROC-AUC, Brier, Log loss, 혼동행렬, FNR/FPR, Recall/Precision/Lift@K를 저장한다. K는 백분율이며 검사 건수는 ceil(n*K).
- 동점은 정답과 무관하게 사전 생성한 tie seed 314159의 순열로 처리한다. Dummy의 개별 fold 검사 성과는 이 무작위 순서의 한 실현값이며 이론적 무작위 기대와 구분한다.
- 기본 비교표는 9개 외부 fold의 평균±표준편차다. fold 점수는 독립 관측이 아니므로 이를 신뢰구간으로 해석하지 않는다. 별도로 반복별 전체 OOF 및 통합 모델의 제품군별 성능을 저장한다. 서로 다른 fold 모델 점수의 척도 차이가 전체 OOF ranking에 영향을 줄 수 있다.
- OOF 확률은 보정되지 않은 모델 점수다. 실제 불량확률이나 미래 제품군 성능으로 주장하지 않는다. 그룹 분할은 새 공정 패턴으로의 일반화를 평가하며 생산시간·배치 기반 검증을 대체하지 않는다.
- 이 실험은 개발용 CV 비교이며 모델 선택 후 독립적인 최종 성능을 보장하지 않는다. 예측은 labeled OOF만 생성하며 unlabeled 제출파일이나 전체 데이터 최종 모델은 만들지 않는다.

## 실행 및 산출물

프로젝트 루트에서 `conda activate kamp-base` 후:

```bash
python -m unittest discover -s 1.injection-molding/tests -v
python 1.injection-molding/scripts/train_baseline.py
```

`artifacts/baseline_v1/`에 config, 원본·코드 해시/패키지 버전 manifest, 코드 snapshot, split, 행 추적표, fold별 모델, OOF 예측, fold 지표, 평균/표준편차, 반복별 지표를 저장한다. 기존 run 디렉터리는 덮어쓰지 않으며 재실행은 `--output-dir`에 새 경로를 지정한다.

검증: raw hash 일치, 모든 fold 그룹 분리/클래스 존재, OOF coverage, 확률 범위, 모든 외부 모델 저장·재로드 예측 일치를 확인한다.
