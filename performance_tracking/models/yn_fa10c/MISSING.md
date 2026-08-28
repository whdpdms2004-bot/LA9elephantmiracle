# yn_fa10c — 학습 코드 누락 해소

2026-08-28에 기존 canonical 학습 경로를 이 폴더로 이식했다. 과거에 기록된
학습 코드 누락은 해소됐으며, 현재 재현 진입점은 다음과 같다.

| 목적 | 파일 |
|---|---|
| 63엔트리 최종 ZIP 재생성 | train_fa10c.py |
| 71피처 공용 구현 | features.py, pipeline.py |
| val2022·val2023 정직 예측 | build_val_predictions.py |
| 기존 제출과 피처·lookup 대조 | verify_pipeline.py |
| 환경·실행 순서 | README.md |

최종 모델은 71피처 = 기존 68 + 오염보정 3이며, 20개 LightGBM,
20개 numeric CatBoost, 20개 team-ID CatBoost와 meta.json을 생성한다.
isotonic은 2019~2023 학습 모델의 2024 holdout 예측으로 적합하고 상한 0.80을
적용한다. 전체 모델은 2019~2024로 다시 학습한다.

TrackMan 실험 피처는 최종 구성에서 기각됐으므로 재현 코드는
trackman_history.csv를 읽지 않는다. 이는 제출 로그의 TrackMan 0건 판정과 일치한다.

## 검증 상태 — 실행한 것과 아직 안 한 것

sj 의 요구는 "돌 것이다" 가 아니라 "돌렸다" 이므로 현재 상태를 그대로 적는다.

| 항목 | 상태 |
|---|---|
| 71피처 이름·순서, stats, cutoff=2024 lookup, 학습-추론 피처 값 대조 | 실행 통과 (`verify_pipeline.py`) |
| 데이터·피처 계약 | 실행 통과 (`train_fa10c.py --check-only`) |
| 피처 생성 · LGB20 · teamCB20 학습 경로 | val2022·val2023 생성으로 80회 실행 |
| numeric CatBoost 학습 | **미실행** |
| 전체학습 모드(홀드아웃 없음) · isotonic 적합 · 63엔트리 ZIP 조립 | **미실행** |
| 재생성 `model/` 과 기존 ZIP 63엔트리 내용 해시 대조 | **미실행** |

즉 **누락은 코드 차원에서 해소됐고, 끝까지 돌려서 같은 산출물이 나오는지는 아직
확인되지 않았다.** 전체 재학습은 시드별 체크포인트를 쓰므로 중단 후 재실행할 수 있고,
완료되면 이 표와 `models/yn_fa10c.md` §6 을 갱신한다.

참고로 이 폴더가 아닌 yn 로컬의 **기존 canonical 경로**
(`experiments/feature_engineering_20260819/build_submission.py` + `apply_cap.py`)는
63/63 엔트리 내용 해시 일치가 이미 검증돼 있다. 그 경로가 실제 제출본
`submit_fa10c.zip` 을 만든 코드다. 이 폴더는 그것을 팀 저장소 규격으로 이식한 것이며,
**두 경로가 같은 산출물을 내는지는 위 표의 마지막 줄에서 확정된다.**
