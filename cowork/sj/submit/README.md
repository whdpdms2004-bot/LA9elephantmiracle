# 제출 파일 관리 규칙

## 디렉터리와 파일명

```text
submit/
└── YYYY-MM-DD/
    ├── submit_NNN.zip
    └── SUBMISSION_LOG.md
```

- 제출 ZIP 이름은 전역 제출 회수를 3자리로 표시한 `submit_NNN.zip`을 사용한다.
- `submit_NNN.zip`은 14자로 대회 제한인 30자 미만이다.
- 날짜가 바뀌어도 회수는 초기화하지 않는다. 예: 다음 제출은 `submit_004.zip`.
- 실제 데이콘에 업로드한 시점을 한 회로 센다. 실행 오류도 회수에 포함한다.
- 채점 또는 오류가 확정된 파일은 덮어쓰지 않는다.
- 수정본은 새로운 회수로 저장하고 이전 시도의 오류 원인을 로그에 남긴다.

## 날짜별 로그 필수 항목

각 `SUBMISSION_LOG.md`에는 다음을 기록한다.

1. 파일명과 SHA-256
2. 모델·피처 버전과 Trackman 사용 여부
3. 시간 검증 분할
4. 하이퍼파라미터 탐색 과정
5. 최종 전체 학습 범위
6. fold별 Brier/BSS/AUC/예측 평균
7. 리더보드 점수 또는 실행 오류
8. 이전 제출과의 차이
9. 다음 실험에서 확인할 가설

## 제출 전 체크리스트

- 파일명 30자 미만
- ZIP 최상위가 `model/`, `script.py`, `requirements.txt`
- 추가 최상위 폴더 없음
- Linux 권한과 ZIP CRC 정상
- `Path(__file__).resolve().parent` 기준 절대경로 사용
- 평가 서버 패키지 버전과 추론 API 호환성 확인
- 5행 샘플 추론 성공 및 확률 유한성/범위 확인
- Trackman 사용 시 target season보다 이전 시즌만 사용했는지 확인
- 최종 제출 모델은 2024를 포함한 전체 train으로 재학습했는지 확인

## 보관 명령

새 ZIP을 만든 뒤 다음 명령으로 검증과 보관을 동시에 수행한다.

```powershell
python submit\archive_submission.py <원본.zip> 4 --date 2026-08-06
```

이미 존재하는 회수는 덮어쓰지 않으며, 출력된 SHA-256과 학습/검증/점수를 같은 날짜의 `SUBMISSION_LOG.md`에 기록한다.

## 현재 다음 회수

`submit_005.zip`
