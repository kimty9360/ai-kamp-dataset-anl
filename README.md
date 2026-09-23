# KAMP AI 경진대회 작업 공간

## 공통 개발환경

과제들이 공유하는 Conda 환경 `kamp-base`의 정의는 이 루트에서 관리합니다.
실제 환경은 Conda 설치 경로에 있으며, 환경 정의 파일 이동은 설치된 환경에 영향을 주지 않습니다.

- `environment-base.yml`: Linux/WSL용 Conda 및 pip 패키지 버전 스냅샷. 환경 재생성의 기본 파일입니다.
- `requirements-base-lock.txt`: 설치된 Python 패키지 버전 스냅샷. Conda의 시스템 라이브러리까지 대체하지는 않습니다.
- `1.injection-molding/`: 사출성형 데이터 및 분석
- `5.resource-optimization/`: 자원 최적화 데이터 및 분석

기존 환경 사용:

```bash
conda activate kamp-base
python -c "import sys; print(sys.executable); print(sys.version)"
```

새 환경 생성 시 프로젝트 루트에서 실행:

```bash
conda env create -f environment-base.yml
```

VS Code의 Python 인터프리터와 노트북 커널은 `kamp-base`를 선택합니다.
공통 패키지는 루트 환경 파일에서 관리하고, 과제별 추가 의존성이 생기면 해당 과제에 기록합니다.
향후 GPU 라이브러리 버전 변경이 필요한 경우 별도 환경 정의를 추가해 현재 baseline 환경을 보존합니다.

현재 파일은 버전 스냅샷이며 플랫폼 독립적인 해시 lock은 아닙니다.
2026-09-23에 YAML 파싱과 설치 버전 일치를 점검했으며, 새 환경을 만드는 재설치 검증은 수행하지 않았습니다.

## GPU 확인 기록 (2026-09-23)

샌드박스 밖에서 `/usr/lib/wsl/lib/nvidia-smi` 실행 성공:

- NVIDIA GeForce RTX 5060 Ti, VRAM 8151 MiB
- NVIDIA-SMI 610.43.02 / KMD 610.47 / CUDA UMD 13.3

기본 에이전트 샌드박스에서는 NVML 접근이 차단되었지만 권한 확장 실행에서는 정상 조회되었습니다.
이는 GPU 인식 확인이며, 개별 학습 라이브러리의 CUDA 연산 호환성 검증은 별도로 필요합니다.
CUDA UMD 표시는 설치된 CUDA Toolkit 버전을 뜻하지 않습니다.

## 현재 결과

제6회 K-인공지능 제조데이터 분석 경진대회 과제 중 사출성형기 데이터의 첫 baseline 비교까지 진행했습니다.

- [실행 안내 및 Jupyter 노트북 사용법](1.injection-molding/README.md)
- [Baseline 결과 보고서](1.injection-molding/artifacts/baseline_v1/report.md)
- [성능 비교 그래프](1.injection-molding/artifacts/baseline_v1/comparison.png)
- [실험 설정](1.injection-molding/configs/baseline_v1.json)
- [검증 프로토콜](1.injection-molding/docs/validation_protocol_baseline_v1.md)

## GitHub 저장 범위

공통 환경 정의, 설계서, 코드, 테스트, 출력이 제거된 노트북, 데이터 진단 요약, baseline 성능표·그래프를 저장합니다.
대회 원본 데이터, 모델 바이너리, 행별 예측·그룹표, split 인덱스, 반복 실행 및 임시 파일은 저장하지 않습니다.
이 파일들은 로컬에 보관하며 코드로 재생성할 수 있습니다. 전체 디스크 백업을 대신하는 저장소는 아닙니다.

복제한 환경에서는 [데이터 준비 안내](1.injection-molding/dataset/README.md)에 따라 데이터를 배치하고,
노트북의 `RUN_TRAINING = True`로 새로 실행하면 행별 예측·모델을 포함한 전체 산출물이 생성됩니다.
학습 없이 결과만 읽으려면 저장된 report.md, summary.csv, comparison.png를 확인하세요.
