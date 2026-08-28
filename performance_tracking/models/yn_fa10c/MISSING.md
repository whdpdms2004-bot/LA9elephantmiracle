# yn_fa10c — 학습 코드 누락 해소 (재현 검증 완료)

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

## 검증 상태 — 전부 실행했다 (2026-08-28)

sj 의 요구는 "돌 것이다" 가 아니라 "돌렸다" 이므로 실행 결과만 적는다.

| 항목 | 결과 |
|---|---|
| 데이터·피처 계약 | 통과 (`train_fa10c.py --check-only`) |
| 71피처 이름·순서, stats, cutoff=2024 lookup, 학습-추론 피처 값 | 통과 (`verify_pipeline.py`) |
| **전체 재학습 120모델** (Stage 1 60 + Stage 2 60) | **완료. 4시간 50분** |
| **63엔트리 ZIP 조립** | **완료** (54,001,622 bytes) |
| **기존 ZIP 과 엔트리 내용 해시 대조** | 63/63 이름 일치, 내용 23 일치 / 40 불일치 |
| **★ 두 패키지 `script.py` 실제 실행 후 예측 대조** | **최대절대차 0.000e+00, 3,000/3,000 완전 동일** |

### 해시 40개가 갈린 것을 어떻게 읽나

    일치 23   lgb_booster_*.txt 20 · meta.json · script.py · requirements.txt
    불일치 40 cb_model_*.cbm 20 · cb_team_model_*.cbm 20   (파일 크기는 전부 동일)

CatBoost `.cbm` 은 같은 모델이라도 생성 메타데이터가 박혀 해시가 갈린다. 크기가
1바이트도 다르지 않고, 두 패키지를 실제로 실행한 예측이 3,000행 전부 비트 단위로
같으므로 **기능적으로 동일한 산출물**이다. ZIP 파일 자체의 SHA-256 은 압축 순서·
타임스탬프에 좌우되므로 애초에 재현 기준이 아니다 — 기준은 위 표의 마지막 두 줄이다.
대조는 엔트리 내용 해시와, 두 ZIP 을 각각 풀어 같은 모사 test 로 script.py 를
실행한 예측을 비교하는 방식으로 했다. 모사 test 는 train.csv 의 2024 행 3,000개이고
열 이름만 test.csv 에서 읽었다 — 실제 테스트셋 규모는 가정하지 않았다.

`meta.json` 이 **바이트 단위로 일치**한다는 점을 따로 적어둔다. 여기에 stats ·
category_levels · **isotonic 좌표** · cutoff=2024 lookup · cap 근거가 전부 들어 있다.
Stage 1(2019~2023 학습 → 2024 홀드아웃)을 처음부터 다시 돌려 뽑은 isotonic 이
좌표까지 그대로 나왔다.

    Stage 1 raw Brier        0.2477694719   등록 val2024 Brier 0.247769 와 일치
    Stage 1 isotonic 후      0.2476066466   SUBMISSION_LOG 자체보고 0.247607 과 일치
    보정 후 예측평균          0.4861033      2024 실제 성공률 0.4861

참고로 yn 로컬의 **기존 canonical 경로**
(`experiments/feature_engineering_20260819/build_submission.py` + `apply_cap.py`)가
실제 제출본 `submit_fa10c.zip` 을 만든 코드다. 이 폴더는 그것을 팀 저장소 규격으로
이식한 것이고, **두 경로가 같은 산출물을 낸다는 것이 위 표로 확정됐다.**
