# 사출성형 제조데이터 — 공정·시간 중심으로 다시 분석

현재 단계는 **가이드북·원자료 진단과 S14 CN7/RG3 공정·시간 EDA 완료**다. 기존 모델 비교를 보존하고, 시간·설비·부품·좌우·불량 사유를 복구한 분석을 새로 시작한다. 아직 새 모델 학습이나 성능 개선을 주장하는 단계는 아니다.

## 현재 작업 요약

[전처리 재설계 전 체크포인트](docs/current_status.md) — 완료한 작업, 확인된 사실, 남은 불확실성, 다음 전처리 계획과 재현 방법을 정리했습니다.

[CN7 10/29~30 진단](artifacts/cn7_oct29_30_audit/report.md): 검토 18건과 상관계수 민감도, 평균 RPM 배율 문제. 원본·라벨은 보존하며 대시보드에서 포함/제외 비교가 가능합니다.

## HTML 대시보드

[브라우저에서 열기](http://127.0.0.1:8765/) — 차종·좌우·날짜 선택, 공정 컬럼 36개 중 제품별 상수 제외 후 기본 24개의 시간 그래프, LH/RH 좌우 비교·실제 불량 표시·온도 이동 중앙값/표준편차, 수치 상관계수, 기록 상세·CSV 저장. 노트북 실행 없이 사용할 수 있다. [사용·재실행 안내](dashboard/README.md).

## 최신 분석

- [상관계수 노트북](notebooks/02_labeled_correlations.ipynb) / [분석 보고서](artifacts/correlation_v1/report.md): 제품·좌우별 Pearson/Spearman, JX1/SP2 처리, 캐비티 수의 미확인 사항.

- [01_process_eda.ipynb](notebooks/01_process_eda.ipynb): 날짜·좌우·불량 사유와 원단위 공정값. 원하는 날짜 확대 가능.
- [공정·시간 EDA 보고서](artifacts/process_eda_v1/report.md): 초기허용불량 관측, 미라벨 분포 차이, 시간 분할의 표본 부족.
- 이번에는 새 모델을 학습하지 않았다. 다음은 운전 조건·센서 사용 구간 조사와 개발용 시간 백테스트 설계다.

- [가공 데이터 계산 경로 재현](docs/processing_reconstruction.md): labeled_data에서 중간본·표준화 CN7/RG3가 되는 계산과 미확인 선택 이유.

## 먼저 읽을 자료

1. [가이드북 상세 분석](docs/guidebook_review.md): 공정 원리 → 제공 자료 → 불량 유형 → AE/준지도 예제 → 새 분석 순서.
2. [원자료 진단 결과](artifacts/source_review_v1/report.md): 실제 파일 크기, 중복, 시간 간격, 불량 사유, 미라벨과의 중복 후보.
3. [00_source_review.ipynb](notebooks/00_source_review.ipynb): 표와 기간·불량 기록을 셀 단위로 확인.
4. [전체 48종 열 설명](docs/process_column_dictionary.csv): 원단위·의미·해석 유의점.
5. [기존 실험 보관본](archive/2026-09-29_tabular_v1/README.md): baseline부터 RG3 집중 실험까지 모두 보존.

## 폴더 구성

```text
1.injection-molding/
  sources/                     제공 ZIP·가이드북, 변경하지 않음
  docs/                        새 공정·데이터 분석 문서
  scripts/review_sources.py    읽기 전용 데이터 진단
  notebooks/00_source_review.ipynb
  notebooks/01_process_eda.ipynb
  scripts/process_eda.py        시간·공정 EDA 재현
  artifacts/process_eda_v1/     그래프·비교·분할 가능성 점검
  artifacts/source_review_v1/  새 진단 결과 (모델 평가 결과 아님)
  archive/2026-09-29_tabular_v1/
    1.injection-molding/        기존 폴더 구조와 파일 전체
    manifest.json              로컬 SHA-256 보존 검증 목록
```

## Jupyter에서 확인

WSL 터미널에서:

```bash
cd /home/kimty/projects/kamp-ai
conda activate kamp-base
jupyter lab
```

터미널의 접속 주소를 브라우저에서 열고 `1.injection-molding/notebooks/00_source_review.ipynb`를 선택한다. 커널은 `Python (kamp-base)`다. 위에서 아래로 실행한다. 기본 실행은 저장된 요약표를 읽으며 학습하지 않는다. 로그인 토큰은 실행 중인 Jupyter 터미널에서 확인한다.

다시 진단할 때:

```bash
python 1.injection-molding/scripts/review_sources.py --output 1.injection-molding/artifacts/source_review_rerun
```

기존 실험을 찾을 때는 `archive/2026-09-29_tabular_v1/1.injection-molding/notebooks/` 및 `artifacts/`를 확인한다. 예전 절대 경로는 달라졌고, 과거 노트북의 재학습은 현재 작업이 아니다. 폴더 이동 전에 열어 둔 노트북 탭은 닫고 새 위치에서 다시 연다.

시간 EDA 결과, 단순 날짜 60/20/20 분할은 불량 사례가 부족해 최종 평가에 적합하지 않았다. 다음은 미라벨의 운전 조건·센서 사용 구간 조사와 개발용 시간 백테스트 설계다. 0.2초 원시 파형, 모든 불량 유형의 라벨, 불량 확률이 이미 확보됐다고 가정하지 않는다.

공정 EDA 재현: `python 1.injection-molding/scripts/process_eda.py --output 1.injection-molding/artifacts/process_eda_rerun`. 새 출력 폴더를 지정한다.
