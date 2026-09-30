# 좌우 부품 구분 추가 실험

## 목적과 비교 조건

baseline_v2의 23개 공정 입력에 part_is_rh(RH=1, LH=0)를 추가한다. pooled에는 기존 product_is_rg3도 유지한다. Dummy·Logistic·XGBoost·ExtraTrees·CatBoost·LightGBM의 기존 11설정 모두 비교한다. 학습률·깊이·가중치 방식 등 모델 설정은 변경하지 않는다.

동일한 CN7 1211행·RG3 1182행 및 pooled 2393행을 사용한다. 원자료의 나머지 214행은 학습에 추가하지 않는다. label 1은 원자료 Y/N과 불량사유 대조로 불량으로 해석한다. Reason은 입력에 사용하지 않는다.

## 좌우 정보의 정합성

원자료 ZIP의 moldset_labeled.csv에서 CN7 앞 1211행, RG3 전체 1182행을 각각 표준화하여 현재 CSV의 24개 공정값과 라벨을 재현한다. 모든 공정값 절대오차 1e-10 이내 및 라벨 완전 일치를 요구한다. PART_NAME 끝의 LH/RH로 side를 얻는다. 파일명과 행 번호를 함께 키로 사용하여 제품군 간 동일 행 번호 충돌을 방지한다. ID·Reason·시각·생산일련번호는 모델 입력에서 제외한다.

이 과정의 StandardScaler는 원자료 연결 검증을 위한 것이다. 모델 입력은 기존 baseline CSV 그대로이며, 이번 실험이 파일별 표준화 자체의 한계를 해결하는 것은 아니다.

## 분할·임계값

baseline_v2/splits.json의 외부/내부 분할을 그대로 사용한다. 좌우 입력을 추가한 후 그룹을 재계산하면 동일 생산 기록이 분리될 수 있으므로 기존 원공정 그룹을 유지한다. 동일 설비·동일 시각 기록도 외부/내부 분할에 겹치지 않는지 검사한다. 다만 시간순 검증은 아니다.

3 seeds × 3 outer folds × 11 settings × 3 scenarios = 새 외부 모델 297회다. 각 설정의 분류 임계값은 내부 그룹 OOF에서 F1 최대값으로 선택하고 0.5 고정 정책도 보고한다. Recall@10%는 각 외부 폴드에서 기존 tie_key로 동점을 처리해 ceil(n×0.1)개를 고른다.

기존 baseline 지표는 저장된 결과에서 읽는다. 실행 환경 재현 점검으로 시드42·외부fold0의 3시나리오×11설정 기준 모델을 다시 학습하고 기존 확률과 절대오차1e-12 이내인지 확인한다. Dummy의 모든 비교 지표는 좌우 정보 추가 전후 일치해야 한다.

## 재현

```bash
conda activate kamp-base
python 1.injection-molding/scripts/train_side_feature.py --output-dir 1.injection-molding/artifacts/side_feature_runs/new_run
```

기존 폴더 덮어쓰기는 거부한다. baseline_v2 전체 파일 및 원본 네 CSV의 전후 SHA256을 검사한다. 새 결과는 side_feature_v1에 보존하고 반복 실행은 side_feature_runs에 별도 저장한다. 기본 결과 확인은 notebooks/04_side_feature.ipynb의 RUN_TRAINING=False로 한다.

## 해석

평균·표준편차는 같은 데이터를 반복 분할한 개발 결과이며 독립 테스트 성능이나 신뢰구간이 아니다. 전체 EDA에서 발견한 좌우 패턴으로 만든 후보이므로 별도 기간/배치 데이터에서 최종 확인이 필요하다. 특히 RG3에서 관찰한 RH 불량 집중이 미래에도 유지된다고 단정하지 않는다. 공정 종료 후 검사 전에 좌우 부품 구분이 가능하다는 조건의 실험이다.

제출 예측·배포용 모델은 이번 범위에 포함하지 않는다. 기존 baseline_v1/v2 및 improvement_v1 결과는 덮어쓰지 않는다.
