# RG3 집중 개선: 후보·가설·결과 해석

## 모델과 목표

CN7은 좌우 추가 CatBoost balanced를 기준으로 고정한다. RG3는 좌우 추가 XGBoost balanced와 RBF SVM(C=1)을 주력 후보로 둔다. pooled는 이번 범위에서 제외한다. 목표 Recall95%는 보장이 아니라 내부 검증에서 임계값을 고를 때의 제약이다. 외부 Precision·Recall·검사 비율·FN을 모두 보고한다.

## 순서

1. 원자료로 현재 행을 수치 재현하고 LH/RH·불량 사유·생산일을 연결한다. RH만의 양품/불량 단일 변수 구분력과 seed42 기존 OOF의 사유별 미탐을 살펴본다. Reason·날짜는 입력에 넣지 않는다.
2. 원단위 입력을 복구한다. 기존 입력23개와 side만 유지한다. SVM의 스케일러는 각 학습fold에서 fit한다. XGBoost는 별도 스케일러 없이 원단위를 사용한다.
3. 원단위에서 금형온도차1, 인접 배럴온도차5, 배럴온도범위·표준편차2, 최대/평균 배압차1, 최대/평균 스크루속도차1의 총10개를 추가한다. 실제 센서 측정 정의가 완전히 검증된 것은 아니므로 음수 차이를 오류로 삭제하지 않는다. 특히 최대/평균 명칭에 단순한 대소 제약을 가정하지 않는다.
4. 후보를 미리 고정해 소규모로 비교한다. XGB는 깊이2·min_child_weight10·lambda10으로 보수화한 후보와 가중치 sqrt(neg/pos) 후보. SVM은 C3 후보와 balanced 가중치 후보. 데이터 재샘플링은 하지 않는다. 내부 학습표본으로 가중치를 계산한다.
5. 내부 OOF Recall>=95%를 만족하는 임계값 중 Precision이 높은 것을 고른다. 동률이면 높은 임계값. 모델 후보도 내부 precision95가 높은 순서, 동률이면 내부AP로 선택한다. 외부 점수로 고른 최선 후보와 구별하기 위해 inner_selected 절차의 외부 결과를 별도 보고한다. 고정0.5·inner_f1도 보존하고 검사 예산5/10/20% 발견율을 확인한다.

## 평가 보존

baseline_v2의 같은 원공정 그룹·3seeds×3folds·내부3folds를 사용한다. 동일 설비/시각도 서로 겹치지 않는지 검사한다. 원단위·파생변수로 그룹을 재정의하지 않는다. 27개 기준 후보의 외부 확률과 내부F1 임계값은 이전 결과와 1e-12 이내 일치를 요구한다. 원본 및 baseline·side·extra_models 기존 파일 해시 불변을 검사한다.

## 재현

```bash
conda activate kamp-base
python 1.injection-molding/scripts/focus_rg3.py --output-dir 1.injection-molding/artifacts/rg3_focus_runs/new_run
```

06_rg3_focus.ipynb의 기본 RUN_TRAINING=False로 결과를 읽는다. True는 새 폴더로 실행한다. 기존 폴더 덮어쓰기는 거부한다. 현재 데이터셋 전체를 이미 탐색했으므로 독립 미래 성능을 주장할 수 없다. CN7 초기 구간의 집중과 RG3 불량25개라는 표본 한계도 유지된다. 새 생산 기간 확보 전 최종 모델로 확정하지 않는다.
