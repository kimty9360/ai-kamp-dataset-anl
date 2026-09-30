# 기존 표준화 데이터 실험 보관본

2026-09-29, 원단위·시간·공정 중심으로 분석을 다시 시작하면서 이전 사출 과제 파일 전체를 보관했다.

- `1.injection-molding/` 아래에 이전 폴더 구조 그대로 보존했다. baseline, EDA, improvement, side_feature, extra_models, rg3_focus, 데이터, 설계서, 노트북, 모델 및 행별 산출물을 포함한다.
- 790개 파일, 273,323,519바이트. 이동 전후 SHA-256 및 크기가 모두 동일함을 검증했다.
- 로컬 `manifest.json`에 각 파일의 상대 경로·크기·SHA-256을 기록했다.
- 루트 안내서의 이전 내용은 `workspace_README_before.md`에 따로 복사했다. 이 파일은 위 790개 목록에 포함되지 않는다.
- 기존 파일 내용을 새 결론에 맞게 수정하지 않았다. 과거 문서의 ‘현재’, ‘라벨 미확인’, ‘RG3 불량은 RH에만 존재’ 등은 당시 데이터·실험 범위에 대한 기록이다.

읽을 결과: [baseline_v2](1.injection-molding/artifacts/baseline_v2/report.md), [EDA](1.injection-molding/artifacts/eda_v1/report.md), [좌우 추가](1.injection-molding/artifacts/side_feature_v1/report.md), [추가 모델](1.injection-molding/artifacts/extra_models_v1/report.md), [RG3 집중 실험](1.injection-molding/artifacts/rg3_focus_v1/report.md).

이동으로 이전 절대 경로 링크나 설정의 경로는 달라졌다. 이 보관본은 결과 열람·보존 목적이며 전체 재실행 호환성을 검증한 것은 아니다. 재실험이 필요하면 별도 작업 공간에 보관된 `1.injection-molding/`을 복원해 상대·절대 경로를 점검한다. 역사적 설정·매니페스트를 소급 수정하지 않는다.

실행 출력이 있는 역사적 노트북·원본 데이터·모델·행별 산출물은 로컬에 보존한다. GitHub에 올리기 전에는 노트북의 원자료 출력 제거 여부를 별도로 검토해야 한다. `.gitignore`는 로컬 데이터와 출력이 담긴 보관 노트북의 신규 추가를 막는다. 이미 Git에 추적된 파일을 자동으로 추적 해제하는 기능은 없다.
