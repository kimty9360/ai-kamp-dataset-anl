# 사출성형기 품질 예측

현재 상태: Dummy·Logistic·XGBoost·ExtraTrees·CatBoost·LightGBM의 전체 baseline 비교 완료(11개 설정). 라벨 의미 미확인으로 class_1 예측 성능만 보고한다.

## Jupyter에서 셀 단위로 실행

WSL Ubuntu 터미널에서 실행한다. 현재 환경은 Windows Anaconda가 아닌 WSL의 Miniforge/Conda에 설치되어 있다.

```bash
cd /home/kimty/projects/kamp-ai
conda activate kamp-base
jupyter lab
```

터미널에 표시된 localhost 접속 주소를 Windows 브라우저에서 열고 `1.injection-molding/notebooks/01_baseline.ipynb`를 선택한다.
커널은 `Python (kamp-base)`를 선택한다. VS Code WSL에서도 같은 파일을 열고 이 커널을 선택할 수 있다.

`Shift+Enter`로 위에서 아래로 실행한다. 기본은 기존 결과 확인 모드이며, 설정 셀의 `RUN_TRAINING = True`로 바꾸면 새로 학습한다.
새 결과는 `artifacts/notebook_runs/<실행시각_ID>/results/`에 저장되므로 기존 결과를 덮어쓰지 않는다.
하이퍼파라미터는 노트북의 `config`에서 수정한다. 새 학습 없이 설정만 바꾸면 기존 결과는 바뀌지 않으며, 결과 셀에 실제 학습에 사용된 설정을 표시한다.
JupyterLab에서 단일 셀 실행은 Shift+Enter, 전체 실행은 Run → Run All Cells, 중단은 Kernel → Interrupt Kernel을 사용한다.

Windows Anaconda Navigator에서 별도 서버를 실행하면 현재 WSL의 커널 및 패키지를 자동으로 공유하지 않는다. 현재 환경을 그대로 사용하려면 위 WSL 명령 또는 VS Code WSL 경로를 사용한다.

- [baseline 결과 보고서](artifacts/baseline_v2/report.md)
- [비교 그림](artifacts/baseline_v2/comparison.png)
- [검증 프로토콜](docs/validation_protocol_baseline_v2.md)
- [실험 설정](configs/baseline_v2.json)
- [데이터 진단](artifacts/audit/data_audit.md)

상위 `kamp-ai` 프로젝트 루트에서 공통 환경 `kamp-base`를 활성화하고 실행한다.

```bash
conda activate kamp-base
python -m unittest discover -s 1.injection-molding/tests -v
python 1.injection-molding/scripts/data_audit.py
python 1.injection-molding/scripts/train_baseline.py --config 1.injection-molding/configs/baseline_v2.json --output-dir 1.injection-molding/artifacts/baseline_v2_rerun
python 1.injection-molding/scripts/verify_baseline.py --run-dir 1.injection-molding/artifacts/baseline_v2_rerun
python 1.injection-molding/scripts/report_baseline.py --run-dir 1.injection-molding/artifacts/baseline_v2_rerun
```

기존 실행 결과는 덮어쓰지 않는다. `dataset/`은 원본이며 수정하지 않는다.
폴드별 모델은 개발 검증 산출물이며 배포용 최종 모델이 아니다. 전체 데이터 재학습 및 미라벨 제출 예측은 아직 수행하지 않았다.

첫 5개 설정 비교는 [baseline_v1](artifacts/baseline_v1/report.md)에 보존합니다.
확장 baseline_v2는 baseline_v1의 원본·분할·기존 설정을 그대로 사용합니다.
GitHub에서 새로 복제한 환경은 기준 split 재생성 방법을 검증 프로토콜에서 확인하세요.

## EDA 및 오류 분석

[02_eda.ipynb](notebooks/02_eda.ipynb)를 `Python (kamp-base)` 커널로 열고 위에서부터 실행합니다. 기본값 `RUN_EDA = False`는 저장된 분석 결과를 읽습니다. `RUN_EDA = True`는 기존 baseline의 OOF 예측으로 분석을 재생성하며 모델을 재학습하지 않습니다. 결과는 `artifacts/eda_runs/`의 새 폴더에 저장됩니다.

- [분석 보고서](artifacts/eda_v1/report.md)
- [분석 범위와 재현 방법](docs/eda_protocol.md)

공식 원본 다운로드 검증은 선택 사항(`DOWNLOAD_OFFICIAL_SOURCE = True`)이며 기본 실행에는 네트워크가 필요하지 않습니다. 행별 상세 표와 OOF 예측은 로컬 파일이므로 GitHub 복제본에서는 baseline과 EDA를 재실행해야 상세 사례까지 볼 수 있습니다. 보고서의 `nan`은 계산 불가를 뜻하며, RG3 앞 10%에는 class_1이 없어 해당 AUC를 계산할 수 없습니다.

## 소규모 개선 실험

[03_improvement.ipynb](notebooks/03_improvement.ipynb)에서 기존 설정, 온도·배압 차이 변수, 얕은 트리를 비교합니다. 기본은 저장 결과 확인이며 RUN_EXPERIMENT=True일 때만 새 폴더로 학습합니다.

- [6단계 실행 계획](docs/improvement_plan.md)
- [데이터 의미의 근거와 미확인 사항](docs/data_semantics_evidence.md)
- [첫 개선 실험 결과](artifacts/improvement_v1/report.md)

일부 후보의 개별 지표는 올랐지만 내부 검증으로 후보를 선택한 절차의 AP는 네 조합 모두 낮아졌습니다. 이번 실험을 전반적인 성능 개선으로 판단하지 않고 baseline을 유지합니다.

## 원자료 연결 확인

[원자료 연결·좌우 부품 보고서](artifacts/lineage_v1/report.md)에서 원자료의 Y/N 라벨, 현재 CSV의 수치 재현, 충돌 그룹의 좌우 관계를 확인했습니다. 0=양품·1=불량 해석을 뒷받침하며, 다음 후보는 좌우 정보를 추가하되 기존 공정 그룹을 유지하는 실험입니다. `scripts/audit_lineage.py`로 재현할 수 있습니다. 원자료 ZIP과 행별 연결 표는 로컬 보관합니다.

## 좌우 부품 정보 추가 학습

[04_side_feature.ipynb](notebooks/04_side_feature.ipynb)에서 baseline_v2와 좌우 정보 추가 결과를 비교합니다. 6개 모델 계열·11설정, CN7/RG3/pooled를 동일 분할·설정으로 평가합니다. 기본값 RUN_TRAINING=False는 저장 결과만 표시합니다.

- [좌우 추가 결과 보고서](artifacts/side_feature_v1/report.md)
- [전후 성능 비교표](artifacts/side_feature_v1/comparison.csv)
- [실험 프로토콜](docs/side_feature_protocol.md)

기존 baseline과 첫 파생변수 실험은 별도로 보존합니다. 새 실험은 part_is_rh 한 열만 추가하며, 좌우를 추가해도 기존 공정 그룹을 분리하지 않습니다.

## SVM·Random Forest·GaussianNB·DNN 추가 비교

[05_extra_models.ipynb](notebooks/05_extra_models.ipynb)에서 기존 좌우 모델11설정과 신규6설정, 총17설정을 비교합니다. DNN은 지도학습 완전연결 신경망이며 가이드북 준지도학습 재현과 구분합니다.

- [결과 보고서](artifacts/extra_models_v1/report.md)
- [전체 비교표](artifacts/extra_models_v1/comparison.csv)
- [설정·검증 방법](docs/extra_models_protocol.md)

기본 RUN_TRAINING=False는 저장 결과를 읽습니다. 이전 baseline·개선·좌우 실험은 별도 보존합니다.

## RG3 집중 개선과 Recall95% 정책

[06_rg3_focus.ipynb](notebooks/06_rg3_focus.ipynb)에서 RH 불량사유·시간대, 원단위·파생변수·소규모 튜닝, 높은 Recall에 필요한 검사량을 확인합니다. 기본 RUN_TRAINING=False입니다.

- [결과 보고서](artifacts/rg3_focus_v1/report.md)
- [실험 계획](docs/rg3_focus_protocol.md)

이번 실험에서 높은 Recall과 높은 Precision을 동시에 확보했다고 결론 내리지 않았습니다. 원단위 복구만으로는 성능이 달라지지 않았고, 내부 선택 절차도 기존 AP보다 낮았습니다. 기존 주력 후보는 비교 기준으로 유지합니다.

RG3 집중 실험의 핵심 판단은 [결과 해설](artifacts/rg3_focus_v1/findings.md)에 정리했습니다.
