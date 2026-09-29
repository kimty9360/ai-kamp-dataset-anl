# EDA 및 오류 분석 재현

## 분석 범위

원본 CSV 4개와 baseline_v2의 저장된 OOF 예측을 사용한다. 모델 재학습, 라벨 변경, 행 삭제, 파생변수 생성은 하지 않는다. 라벨 의미가 확인되지 않았으므로 class_1로 표현한다.

입력 CSV와 baseline manifest의 해시를 확인하고, OOF의 원본 행·라벨·그룹 연결 및 실험별 행 누락·중복을 검사한다. 상위 10%는 기존 baseline과 같이 각 outer fold 안에서 ceil(n × 0.1)개를 선택하고, 동점은 저장된 라벨 독립 tie_key로 처리한다. 재계산한 Recall@10%가 baseline과 일치하는지 검사한다.

클래스 분포·상관·단일 변수 AUC는 전체 데이터의 탐색 통계이며 CV 성능이 아니다. 세 seed의 오류는 같은 관측에 대한 반복 예측으로 취급한다. 반복 오류 표는 제품군별 개별 모델의 세 seed를 집계하고, 상세 오류 그림과 특성 구간 표는 seed 42를 사용한다. 구간 내 양성이 없으면 Recall을 결측으로 둔다. 파일 행 순서는 시간으로 해석하거나 모델 입력에 넣지 않는다.

## 실행

프로젝트 루트에서 `kamp-base` 환경으로 실행한다. 출력 폴더가 이미 있으면 덮어쓰지 않는다.

```bash
python 1.injection-molding/scripts/eda_analysis.py --baseline-dir 1.injection-molding/artifacts/baseline_v2 --output-dir 1.injection-molding/artifacts/eda_runs/manual_run --source-check 1.injection-molding/artifacts/eda_v1/source_check.json
```

공식 원본 확인이 필요하면 별도로 실행한다. 다운로드는 로컬 CSV를 덮어쓰지 않는다.

```bash
python 1.injection-molding/scripts/verify_source.py --download --output /tmp/kamp_source_check.json
```

`--source-check`는 선택 사항이다. 지정하면 검증 당시의 로컬 CSV 해시와 현재 파일의 일치 여부도 확인한다. 공식 파일과의 바이트 일치는 데이터 의미·라벨 방향·중복 생성 이유를 검증하지 않는다.

## 산출물과 해석

`artifacts/eda_v1/report.md`에 요약, CSV에 통계, figures에 그림, manifest.json에 입력·OOF·코드 해시와 검증 결과를 저장한다. rows는 로컬 행별 상세 자료이며 Git 추적에서 제외한다. 반복 실행도 Git에서 제외한다.

`notebooks/02_eda.ipynb`는 기본적으로 저장된 결과를 표시한다. `RUN_EDA = True`로 새 분석을 실행할 수 있고, `PRODUCT`, `MODEL`, `SEED`, `ROW_POSITION`, `FEATURE`로 상세 사례를 탐색한다. 모델 선택은 해당 상세 표가 제공하는 설정 안에서 한다.

이번 EDA는 기존 평가 라벨을 참고한 탐색이다. 여기서 고른 전처리·파생변수를 같은 검증에 반복 적용하면 선택 편향이 생길 수 있다. 다음 실험은 후보를 적게 정하고 기존 그룹 분할에서 전후를 비교하며, 최종 성능은 별도 독립 평가가 필요하다.
