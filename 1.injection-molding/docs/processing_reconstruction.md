# labeled_data → moldset_labeled → 제품별 2차 가공본 계산 검증

현재 제공된 ZIP을 직접 계산해 대조했다. **동일한 결과를 만드는 연산은 확인했지만, 원 작성자의 정확한 코드와 행 선택 의도까지 확인한 것은 아니다.** 가이드북 17–20쪽에서 `moldset_labeled.csv`는 시간·공정값이 남은 1차 가공 자료이고, 제품명 접미사가 붙은 `moldset_labeled_cn7.csv`, `moldset_labeled_rg3.csv`는 2차 가공 자료다.

## 1. labeled_data.csv → moldset_labeled.csv

- 시작: 7,996행 × 45열.
- `labeled_data.csv`의 **파일 순서상 앞 2,607행**을 선택한다. 시간순으로 새로 정렬하거나, 중복 제거 후 앞 2,607개를 선택하는 것이 아니다.
- `PassOrFail`을 Y→0, N→1로 바꾼다.
- `PART_NAME`에 대응하는 부품 번호 `PART_NO`를 추가한다. LH/RH 별 참조값이지 온도·압력으로 계산되는 값은 아니다.
- 원래 행 위치를 `Unnamed: 0`에 보존한다.
- CN7 1,425행을 먼저, RG3 1,182행을 뒤에 배치한다. 제품 안의 기존 순서는 유지한다.
- 결과: **2,607행 × 47열 전 셀이 제공된 moldset_labeled.csv와 정확히 일치**한다. 공정값에 평균·필터·이동평균·표준화 연산을 할 필요가 없다.

| PART_NAME | 추가되는 PART_NO |
|---|---|
| CN7 W/S SIDE MLD'G LH | 86131AA000 |
| CN7 W/S SIDE MLD'G RH | 86141AA000 |
| RG3 MOLD'G W/SHLD, LH | 86131T1000 |
| RG3 MOLD'G W/SHLD, RH | 86141T1000 |

0부터 세는 원본 행 위치로 최종 순서는 `0~1210 → 2393~2606 → 1211~2392`다. 이것이 제품별 묶음 순서다. `Unnamed: 0`을 새로 0부터 다시 번호 매기지 않아야 그 열까지 일치한다.

가이드북 19쪽은 데이터베이스에서 불러온 속성 차이로 `PART_NO`가 일부 파일에 없다고 설명한다. 위 표는 제공된 중간본에서 확인한 대응 관계이며, 원 작성자가 실제로 이 딕셔너리를 써서 추가했다는 증거는 아니다. **왜 앞 2,607행을 선택했는지는 미확인**이다.

## 2. moldset_labeled.csv → 제품별 파일

가이드북 62쪽은 제품 번호로 분리하고 불필요 열을 제거하는 코드, 69–70쪽은 StandardScaler 및 CSV 저장 코드를 보여 준다. 실제 제공 파일을 맞추려면 다음과 같다.

| 단계 | CN7 | RG3 |
|---|---:|---:|
| 중간본에서 제품별 행 선택 | 1,425행 | 1,182행 |
| 최종 파일에 들어간 행 | 제품별 기존 순서의 **앞 1,211행** | 전체 1,182행 |
| 제외된 행 | 214행, 그중 불량 10개 | 없음 |
| 공정 입력 열 | 24개 | 24개 |
| 최종 열 수 | 공정 24 + 라벨 1 + 인덱스 1 = 26 | 동일 |

CN7 뒤 214행은 원본의 2393~2606행에 해당하며 2020-10-27의 기록이다. 69쪽 예제는 CN7 양품 1,398개 + 불량 27개 = 1,425개를 표시하지만, ZIP 최종 CN7은 1,211개다. **이 추가 제외를 설명하는 근거는 확인되지 않았다.** 문서만 기계적으로 따라 전체 1,425행에 scaler를 fit하면 현재 최종 CSV와 다르다.

제거하는 21열:

- 문맥/검사 정보 9개: `_id`, `TimeStamp`, `PART_FACT_PLAN_DATE`, `PART_FACT_SERIAL`, `PART_NO`, `PART_NAME`, `EQUIP_CD`, `EQUIP_NAME`, `Reason`.
- 공정 열 12개: `Switch_Over_Position`, `Barrel_Temperature_7`, `Mold_Temperature_1`, `Mold_Temperature_2`, `Mold_Temperature_5`부터 `Mold_Temperature_12`까지.

좌우·시간 정보가 제거되면서 서로 다른 기록이 공정 입력만 보면 같아질 수 있다. 이것은 값을 평균해서 같게 만든다는 뜻이 아니다. `Clamp_Open_Position`은 최종 열에 남아 있지만 표준화 후 상수 0이다.

## 3. 실제 숫자 계산: 표준화

각 제품의 **최종 선택된 행들**에서 각 공정 열의 평균 μ와 모집단 표준편차 σ를 따로 계산한다.

```text
μ = sum(x_i) / N
σ = sqrt(sum((x_i - μ)^2) / N)     # ddof=0
z_i = (x_i - μ) / σ
```

CN7과 RG3는 평균·표준편차를 공유하지 않는다. 라벨과 내보내기 인덱스는 표준화하지 않는다. 상수 열은 scikit-learn이 나눗셈의 scale을 1로 처리하므로 모든 값이 0이 된다. pandas의 기본 `std()`는 ddof=1이므로 직접 계산할 때는 `std(ddof=0)`을 써야 한다.

CN7 첫 행의 `Injection_Time` 예:

```text
원래 값 x = 9.59000015258789
평균 μ    = 9.548150324801785
표준편차 σ = 0.024923620337679496

z = (9.59000015258789 - 9.548150324801785) / 0.024923620337679496
  ≈ 1.679123145799
```

제공된 CSV의 값 `1.6791231457990747`과 수치 정밀도 범위에서 일치한다. 이 값은 **1.679초가 아니라 해당 CN7 자료 평균보다 약 1.679 표준편차 위**라는 뜻이다.

모든 24개 공정 열의 모든 행을 대조한 최대 절대 오차:

- CN7: 약 `1.77e-13`
- RG3: 약 `4.16e-13`

부동소수점 계산·CSV 표현의 아주 작은 오차 수준이다. 라벨과 인덱스는 정확히 일치한다. 파일 바이트가 완전히 동일하다는 뜻은 아니다.

## 4. 코드와 검증 결과

- [전체 재현 스크립트](../scripts/reconstruct_molding_processing.py)
- [검증 기록](../artifacts/processing_reconstruction_v1/verification.json)
- [열별 평균·표준편차](../artifacts/processing_reconstruction_v1/scaler_parameters.csv)
- [열별 첫 행 계산 예시](../artifacts/processing_reconstruction_v1/calculation_examples.csv)

```bash
conda activate kamp-base
python 1.injection-molding/scripts/reconstruct_molding_processing.py --output 1.injection-molding/artifacts/processing_reconstruction_rerun
```

새 출력 경로를 지정한다. 원자료와 이전 결과를 변경하지 않으며, 학습 모델을 만들지 않는다. 여기서 StandardScaler의 fit은 제공 데이터의 변환을 검증하기 위한 통계 계산이다.

이번 재현 범위는 `labeled_data → moldset_labeled → moldset_labeled_cn7/rg3`다. 별도 AE용 `supervised_label_cn7.csv`와 미라벨 파일의 전체 변환 경로를 이번 스크립트로 검증했다는 뜻은 아니다.

**이 계산을 새 학습 파이프라인에 그대로 복사하지 않는다.** 제공 파일을 맞추려고 전체 선택 행으로 평균·표준편차를 계산했지만, 새 미래 예측 실험은 먼저 시간 분할하고 학습 구간으로만 전처리를 fit해야 한다. 또한 이번 공정 분석에서는 시간·설비·좌우를 문맥 정보로 보존한다.
