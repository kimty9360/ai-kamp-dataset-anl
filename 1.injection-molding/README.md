# 사출성형기 품질 예측

현재 상태: 첫 그룹 CV baseline 학습 완료. 라벨 의미 미확인으로 class_1 예측 성능만 보고한다.

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

- [baseline 결과 보고서](artifacts/baseline_v1/report.md)
- [비교 그림](artifacts/baseline_v1/comparison.png)
- [검증 프로토콜](docs/validation_protocol_baseline_v1.md)
- [실험 설정](configs/baseline_v1.json)
- [데이터 진단](artifacts/audit/data_audit.md)

상위 `kamp-ai` 프로젝트 루트에서 공통 환경 `kamp-base`를 활성화하고 실행한다.

```bash
conda activate kamp-base
python -m unittest discover -s 1.injection-molding/tests -v
python 1.injection-molding/scripts/data_audit.py
python 1.injection-molding/scripts/train_baseline.py --output-dir 1.injection-molding/artifacts/baseline_v1_rerun
python 1.injection-molding/scripts/report_baseline.py --run-dir 1.injection-molding/artifacts/baseline_v1_rerun
```

기존 실행 결과는 덮어쓰지 않는다. `dataset/`은 원본이며 수정하지 않는다.
폴드별 모델은 개발 검증 산출물이며 배포용 최종 모델이 아니다. 전체 데이터 재학습 및 미라벨 제출 예측은 아직 수행하지 않았다.
