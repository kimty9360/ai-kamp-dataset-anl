# 원본 데이터 준비

이 공개 저장소에는 대회 데이터 원본을 포함하지 않습니다.
KAMP 공식 공지의 `1. 사출성형기 AI 데이터셋.zip`을 내려받아 다음 파일을 이 폴더에 배치하세요.

공식 출처: https://www.kamp-ai.kr/noticeDetail?NOTICE_SEQ=86&page=1&GROUP_SEL=&SEARCH_SEL=&SEARCH_TXT=

- `moldset_labeled_cn7.csv`
- `moldset_labeled_rg3.csv`
- `moldset_unlabeled_cn7.csv`
- `moldset_unlabeled_rg3.csv`

공식 데이터 이용 조건을 확인하고 원본을 수정하지 마세요. `scripts/data_audit.py`로 스키마와 SHA-256을 확인할 수 있습니다.
