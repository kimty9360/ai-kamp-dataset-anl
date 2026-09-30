# EDA 및 OOF 오류 분석

**분석 범위:** 원본 확인, 클래스 분포, 행 순서, 특성 관계, labeled/unlabeled 분포 차이, 저장된 baseline_v2 OOF 오류. 모델 학습·데이터 수정·파생변수 추가는 하지 않았습니다.

## 공식 데이터 확인

- 공식 ZIP과 로컬 CSV 4개 바이트 일치: **True**.
- 검증한 ZIP은 CSV 4개만 포함하고 별도 데이터 사전은 없습니다. 첨부 HWPX의 키워드 검색 결과는 source_check.json을 참고하세요.
- 라벨의 불량 방향, 파일 구성·중복 이유, 전처리 이력은 여전히 미확인입니다. 키워드 검색만으로 다른 설명 자료의 존재 여부까지 판단하지 않습니다.
- 공식 출처: https://www.kamp-ai.kr/noticeDetail?NOTICE_SEQ=86&page=1&GROUP_SEL=&SEARCH_SEL=&SEARCH_TXT=

## 데이터 구조와 행 순서

| 데이터 | 행 | class_1 | 고유 패턴 | 충돌 그룹 | 충돌에 속한 class_1 | 앞 10%의 class_1 |
|---|---:|---:|---:|---:|---:|---:|
| cn7 | 1211 | 17 | 606 | 11 | 11 | 17 |
| rg3 | 1182 | 25 | 591 | 25 | 25 | 0 |

- CN7 class_1은 파일 앞부분에 집중되어 있습니다. 행 번호는 시간/배치로 검증되지 않았으며 모델 입력에 사용하지 않습니다.
- 특정 구간의 공정 조건과 라벨 분포가 함께 달라지는 것은 선택·수집·정렬 효과일 수 있습니다. 공정 원인이나 미래 성능으로 단정하지 않습니다.
- 중복이 공식 원본에 존재한다는 사실은 확인됐지만, 실제 반복 생산인지 복제/익명화/가공 결과인지는 알 수 없습니다.

## 단일 변수의 클래스 구분 (전체 데이터 탐색)

아래 AUC는 학습 모델의 CV 점수가 아닙니다. 전체 라벨을 보고 변수와 방향을 비교했으므로 과대평가될 수 있고 다중 비교를 수행했습니다. p-value/인과 해석은 하지 않습니다.

| 데이터 | 변수 | 방향 무관 AUC | class_1 평균 - class_0 평균 | 앞 10% 안의 방향 무관 AUC |
|---|---|---:|---:|---:|
| cn7 | Mold_Temperature_3 | 0.894 | 1.234 | 0.614 |
| cn7 | Mold_Temperature_4 | 0.892 | 1.386 | 0.676 |
| cn7 | Plasticizing_Position | 0.870 | 1.518 | 0.573 |
| cn7 | Max_Back_Pressure | 0.852 | -1.497 | 0.572 |
| cn7 | Average_Back_Pressure | 0.845 | -1.360 | 0.612 |
| rg3 | Barrel_Temperature_5 | 0.635 | 0.455 | nan |
| rg3 | Plasticizing_Time | 0.561 | 0.177 | nan |
| rg3 | Barrel_Temperature_3 | 0.548 | -0.140 | nan |
| rg3 | Cycle_Time | 0.546 | 0.008 | nan |
| rg3 | Clamp_Close_Time | 0.546 | 0.185 | nan |

- 방향 무관 AUC=max(AUC,1-AUC): 0.5에 가까우면 해당 변수 하나로 순위를 구분하기 어렵습니다. 큰 값은 탐색적 연관성이지 검증 성능이 아닙니다.
- 첫 10% 안의 AUC는 파일 위치 집중에 대한 기술적 민감도 분석입니다. 별도 독립 검증이 아니며 표본이 작습니다.
- 표준화된 변수 차이는 섭씨·초·압력 단위 차이가 아닙니다. 상관 높은 변수를 자동 삭제하지 않았습니다.

## labeled / unlabeled 분포 차이

| 데이터 | 변수 | KS 거리 | 미라벨 중 라벨 범위 밖 비율 |
|---|---|---:|---:|
| cn7 | Clamp_Open_Position | 0.552 | 100.0% |
| cn7 | Plasticizing_Time | 0.515 | 51.5% |
| cn7 | Cushion_Position | 0.510 | 0.0% |
| cn7 | Plasticizing_Position | 0.510 | 0.0% |
| rg3 | Clamp_Open_Position | 0.633 | 100.0% |
| rg3 | Cycle_Time | 0.624 | 11.6% |
| rg3 | Injection_Time | 0.542 | 0.0% |
| rg3 | Cushion_Position | 0.441 | 0.0% |

- 평균과 표준편차가 같아도 분포 모양은 다를 수 있습니다. 파일별 표준화 정황 때문에 동일 수치가 동일 원 단위를 뜻하지 않습니다.
- KS는 분포 차이의 크기만 기술합니다. 중복/의존성을 무시한 p-value 유의성 주장은 하지 않습니다.
- 극단값(abs(z)>3/5)은 점검 대상으로만 집계했고 class_1을 포함한 행을 제거하지 않았습니다.

## 반복적으로 놓치는 class_1 (제품군별 모델)

같은 원본 행에 대한 세 seed의 OOF 결과를 요약합니다. 세 번의 독립 생산 관측으로 세지 않습니다. 아래 모델은 대표 비교용이며 전체 설정은 stable_misses.csv에 있습니다.

| 데이터 | 모델 | class_1 수 | 세 반복 모두 FN | 세 반복 모두 상위 10% 밖 | 세 반복 모두 상위 10% 안 |
|---|---|---:|---:|---:|---:|
| cn7 | catboost_balanced | 17 | 8 | 0 | 14 |
| cn7 | extratrees | 17 | 11 | 1 | 10 |
| cn7 | lightgbm_balanced | 17 | 3 | 0 | 13 |
| cn7 | logistic | 17 | 11 | 3 | 11 |
| cn7 | xgboost_balanced | 17 | 3 | 1 | 14 |
| rg3 | catboost_balanced | 25 | 7 | 21 | 0 |
| rg3 | extratrees | 25 | 6 | 21 | 0 |
| rg3 | lightgbm_balanced | 25 | 11 | 17 | 2 |
| rg3 | logistic | 25 | 5 | 21 | 1 |
| rg3 | xgboost_balanced | 25 | 11 | 17 | 1 |

- CN7: Dummy를 제외한 10개 설정·세 반복 모두 FN인 원본 class_1은 2행입니다.

- RG3: Dummy를 제외한 10개 설정·세 반복 모두 FN인 원본 class_1은 0행입니다.

- FN은 선택된 판정 임계값 아래에 있다는 뜻이고, 검사 상위 10% 밖이라는 뜻과 다릅니다.
- 오류 조건 표는 seed 42의 OOF와 사전 고정된 표준화 값 구간을 사용합니다. 양성이 없는 구간의 Recall은 0 대신 결측으로 표시합니다.
- 전체 라벨과 오류를 보고 탐색한 조건이므로 확정 규칙이나 독립적으로 검증된 실패조건으로 주장하지 않습니다.

## 다음 실험 후보 (아직 실행하지 않음)

1. 공식 문의/추가 설명에서 라벨 방향, 중복 생성 이유, 파일 정렬 및 수집 시점을 확인합니다.
2. CN7에서 공정값 구분이 파일 앞부분 안에서도 유지되는지 확인하고 해당 구간의 FN/FP를 살펴봅니다. 행 번호를 파생변수로 넣지 않습니다.
3. 온도 채널의 상대 편차, 최대/평균 측정값의 차이를 소수 묶음으로 검토합니다. 현재 동일 입력의 라벨 충돌은 이러한 파생변수로 해결되지 않습니다.
4. 탐색으로 정한 후보는 고정된 그룹 검증의 처리 전후 비교로 확인하고, 이후 같은 평가에 반복 적응한 선택 편향을 별도로 관리합니다.
5. RG3의 분리 신호가 약한 이유를 모델 종류만으로 단정하지 않고 데이터 수집·라벨 구조와 함께 검토합니다.

## 산출물 안내

- data_summary.csv, row_blocks.csv, duplicate_row_gaps.csv: 데이터 구조/파일 순서.
- feature_class_summary.csv, correlated_pairs.csv, *_correlation.csv: 클래스 분포와 변수 관계.
- distribution_shift.csv, outlier_summary.csv: 분포 차이/극단값 진단.
- error_cohorts_by_seed.csv: 충돌 여부·제품군별 오류; seed별 원본 행을 한 번씩 집계.
- stable_misses.csv, fn_overlap_seed42.csv, error_feature_bins_seed42.csv: 반복 미탐·모델 겹침·구간별 오류.
- rows/: 원본 행과 결합한 진단 표. 로컬 보관이며 공개 Git 추적에서 제외.
- figures/: 분포/행 순서/상관/오류 그림. manifest.json: 원본·OOF·코드 해시와 검증 결과.
