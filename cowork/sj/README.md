# LG Aimers 9기 Phase 2 — 투구 제구 성공 확률 예측

최종 업데이트: 2026-08-13 KST  
목적: 새로운 팀원/Agent가 기존 실험을 다시 해석하느라 시간을 쓰지 않고, 검증·피처 실험·제출 제작을 바로 이어갈 수 있게 하는 인수인계 문서

## 1. 문서 사용법과 현재 결론

이 파일과 `requirements-dev.txt`만으로 다음이 가능하도록 구성했다.

- 데이터 파일의 역할, 컬럼 의미, 범주/범위와 결측 구조 파악
- 누수 없이 2023/2024 순방향 검증 구성
- 핵심 EDA 수치와 현재까지의 모델 성능 확인
- pandas 기반 EDA, 피처 생성, XGBoost 기준선 학습
- TrackMan의 500구·strict-as-of 집계 설계
- 새로운 실험과 제출 후보의 기록·비교

현재 결론은 다음과 같다.

- 가장 강한 신호는 선수 ID 자체가 아니라 **시즌 drift + 투수 과거 성공률 + 투수×타자손×count 반응**이다.
- 전체 AUC가 약 0.55라서 복잡한 분류 경계보다 calibration, shrinkage, residual ensemble이 더 중요했다.
- Public 최고는 `submit_013`의 **895.404000081**이다.
- 내부 Val2024 최고는 `submit_021`의 **836.502924**지만 Public 미검증이며 개선 신뢰구간이 0을 포함한다.
- 다음 우선순위는 TrackMan 완전 미사용 clean validation, nested selection, R residual 강화, Outside component의 작은 가중치 연결이다.

### 선택적으로 확인할 상세 산출물

아래 링크는 근거 원본이며 이 README를 이해하기 위한 필수 선행자료는 아니다.

1. [`status.md`](status.md): 현재 최고 성능과 바로 다음 작업
2. [`submit/2026-08-13/SUBMISSION_LOG.md`](submit/2026-08-13/SUBMISSION_LOG.md): 최신 ZIP의 설계·검증·해시
3. [`experiment/model_optimization/pitcher_cluster_matchup/reports/reverse20_submission_metrics.json`](experiment/model_optimization/pitcher_cluster_matchup/reports/reverse20_submission_metrics.json): 최신 시스템의 정확한 Val2024 결과
4. [`experiment/model_optimization/VALIDATION_LOG.md`](experiment/model_optimization/VALIDATION_LOG.md): 과거 실험 레지스트리
5. [`experiment/control_success_eda/EDA_REPORT.md`](experiment/control_success_eda/EDA_REPORT.md): 전체 EDA
6. [`experiment/control_success_feature_template/README.md`](experiment/control_success_feature_template/README.md): 협업용 피처 코드 규격

## 2. 문제와 평가 지표

- 각 투구의 `control_success=1` 확률을 예측한다.
- 학습 Target은 `control_success`이며 성공 1, 실패 0이다.
- 실패는 가운데 몰림, 존에서 크게 벗어남, 포수 요구 방향과 반대인 경우를 포함한다.
- 공식 평가는 Brier Skill Score다.

```text
Brier = mean((prediction - target)^2)
null_brier = target_mean * (1 - target_mean)
BSS = max(0, 100000 * (1 - Brier / null_brier))
```

모델 비교 과정에서는 0으로 잘린 BSS보다 **Brier와 normalized Brier**를 우선한다. AUC만 높이는 모델이나 0.5 임계값 F1을 높이는 모델이 반드시 좋은 확률 모델은 아니다.

## 3. 데이터

데이터는 프로젝트 루트의 `data/`에 둔다.

```text
data/
├── train.csv                  # 1,475,092행 × 49열, 2019~2024
├── test.csv                   # 로컬은 형식 확인용 5행, 평가 서버는 245,789행
├── sample_submission.csv      # row_id, control_success
└── trackman_history.csv       # 1,793,078행 × 30열, 2019~2024
```

전체 train 메모리는 pandas 기본 dtype 기준 약 **818.7MB**, TrackMan은 약 **1,261.3MB**다. 28GB RAM에서는 둘을 동시에 로드할 수 있지만 여러 복사본을 만들면 메모리가 빠르게 증가하므로 필요한 컬럼만 읽고 `float32/int32/category` 변환을 권장한다.

### 3.1 데이터 무결성 요약

| 항목 | 결과 |
|---|---:|
| train Target 1 / 0 | 772,603 / 702,489 |
| 전체 성공률 | 0.52376598 |
| train `row_id` 고유성 | 100% |
| 입력 피처가 완전히 같은 중복 행 | 0 |
| train/test 입력 컬럼 일치 | 일치 |
| 투수 / 타자 수 | 792 / 830 |
| 투수팀 / 타자팀 수 | 13 / 13 |
| 500구 이상 투수 | 465 |
| TrackMan 투수 / 타자 수 | 906 / 913 |
| 메인↔TrackMan raw 투수 ID 직접 교집합 | 0 |
| TrackMan `trackman_id` 고유성 | 100% |

메인 데이터의 행 순서는 시즌 및 투수 누적수와 강하게 연결된다. 다음 행의 누적 성공률에서 현재 Target을 복원할 수 있는 현상도 확인했지만, 실제 test 행은 독립이며 다른 test 행 사용이 명시적으로 금지되어 있다. 이 현상은 **누수 진단**이지 사용할 수 있는 피처가 아니다.

### 3.2 train/test 입력 피처 48개

| 묶음 | 컬럼 |
|---|---|
| 식별·시점 | `row_id`, `season`, `game_month`, `game_dayofweek` |
| 경기 상황 | `inning`, `top_bottom`, `game_type`, `balls_before`, `strikes_before`, `outs_before` |
| 점수 | `run_top_before`, `run_bot_before`, `run_total_before`, `score_diff_home`, `score_diff_pitcher_team` |
| 주자 | `runner_on_1b`, `runner_on_2b`, `runner_on_3b`, `num_runners_on`, `base_state` |
| 승리·중요도 | `home_win_expectancy`, `away_win_expectancy`, `li` |
| 선수·팀 | `pitcher_id`, `batter_id`, `pitcher_hand`, `batter_hand`, `pitcher_team_id`, `batter_team_id` |
| 투수 누적 | `asof_pitcher_n`, `asof_pitcher_success_rate`, `asof_pitcher_reverse_rate`, `asof_pitcher_middle_rate`, `asof_pitcher_ball_rate`, `asof_pitcher_strike_rate` |
| 투수 최근 경기 | `asof_pitcher_prev1_game_success_rate`, `asof_pitcher_prev3_game_success_rate`, `asof_pitcher_prev5_game_success_rate`, `asof_pitcher_prev1_game_middle_rate`, `asof_pitcher_prev3_game_middle_rate`, `asof_pitcher_prev5_game_middle_rate` |
| 타자 누적 | `asof_batter_n`, `asof_batter_success_rate`, `asof_batter_middle_rate` |
| 구종 구성 | `asof_pitcher_pitchmix_n`, `asof_pitcher_fastball_rate`, `asof_pitcher_breaking_rate`, `asof_pitcher_offspeed_rate` |

`train.csv`에만 Target `control_success`가 추가된다. `row_id`는 조인·제출 순서 보존용이지 시간이나 수치 피처로 사용하지 않는다.

### 3.3 TrackMan 원본

TrackMan은 경기/투구 식별자, 투수·타자와 좌우, 구종, 그리고 다음 물리량을 제공한다.

- 구속: `rel_speed`, `zone_speed`
- 회전: `spin_rate`
- 무브먼트: `induced_vert_break`, `horz_break`
- 릴리스: `extension`, `rel_height`, `rel_side`
- 구종: `tagged_pitch_type`, `auto_pitch_type`, `pitch_type_group`

메인 데이터의 `pitcher_id`와 TrackMan의 `pitcher_trackman_id`는 그대로 일치하지 않으므로 crosswalk가 필요하다. 현재 실험 규칙은 **시즌당 500구 이상인 투수-시즌만 사용**하며, 예측 시즌보다 과거인 시즌만 집계한다.

### 3.4 타입·범위·결측 요약

| 피처군 | dtype/범위 | 결측 |
|---|---|---:|
| `season` | int, 2019~2024 | 0% |
| `game_month` | int, 3~10 | 0% |
| `inning` | int, 1~13 | 0% |
| count/out | balls 0~3, strikes 0~2, outs 0~2 | 0% |
| `home/away_win_expectancy` | float, 0~100 | 0% |
| `li` | float, 0~10.83 | 0% |
| 투수 누적 rate 5개 | float, 0~1 | 각 792행, 0.0537% |
| 투수 최근 1/3/5경기 6개 | float, 0~1 | 각 29,185행, 1.9785% |
| 타자 누적 rate 2개 | float, 0~1 | 각 830행, 0.0563% |
| 투수 구종 rate 3개 | float, 0~1 | 각 792행, 0.0537% |

결측은 오류가 아니라 대체로 첫 투구·첫 경기의 cold start를 나타낸다. 단순 평균 대치만 하지 말고 `is_missing`, 표본량, shrinkage reliability를 함께 둔다.

알려진 결정적 관계는 다음과 같다.

```text
run_total_before = run_top_before + run_bot_before
score_diff_home = run_bot_before - run_top_before
num_runners_on = runner_on_1b + runner_on_2b + runner_on_3b
base_state = 세 runner flag의 문자열 인코딩
away_win_expectancy ≈ 100 - home_win_expectancy
asof_pitcher_pitchmix_n = asof_pitcher_n
```

트리 모델은 중복 피처를 유지해도 작동하지만 선형/신경망/거리 기반 모델에서는 하나만 남기거나 표준화·정규화를 명시해야 한다.

## 4. 절대 지켜야 할 시간·누수 규칙

- Val2023: 2019~2022 학습 → 2023 검증.
- Val2024: 2019~2023 학습 → 2024 검증.
- 최종 2025 추론 모델: 설정을 확정한 뒤 2019~2024 전체로 재학습.
- 2024 검증에서 Target 집계, 임베딩, 클러스터, calibration, blend weight는 모두 2023 이전/순방향 OOF만 사용한다.
- TrackMan은 예측 시즌 `S`에 대해 `season < S`만 허용한다. 즉 Val2024에는 2019~2023만, 최종 2025에는 2019~2024만 사용할 수 있다.
- 사용자가 별도로 요구한 **Val2024 TrackMan 완전 미사용 기준선**이 아직 최신 시스템과 완전히 분리되지 않았다. 다음 Agent가 우선 재구축해야 한다.
- `test.csv`의 다른 행을 이용한 빈도, 집계, 순위, normalization, target encoding은 금지한다. 실제 테스트의 행 구성은 로컬 5행 샘플과 다르다.
- Stateful 피처는 fold 안에서 fit하고 validation/test에는 transform만 한다.
- 같은 선수의 미래 시즌이나 validation Target으로 만든 profile/cluster를 validation에 연결하지 않는다.

## 5. 현재 모델 시스템

최신 `submit_021`은 단일 거대 모델이 아니라 기본 확률에 시간 안전 residual expert를 작은 크기로 더하는 구조다.

```text
Enhanced CatBoost 3-seed ┐
                         ├─ beta-calibrated base ensemble
Enhanced XGBoost 3-seed ┘
             + Insight-adjusted XGBoost
             + pitcher/batter matchup residual
             + reverse batter-cluster residual 20-seed
             + R count×inning×hand context residual
             + F XGB/CatBoost/TabM partial-pooling expert
             = final probability
```

주요 수치:

- base Cat/XGB 가중치: Cat 0.51644, XGB 0.48356.
- Insight XGB와 base의 probability blend: 0.6085 / 0.3915.
- 일반 matchup residual scale: 0.25.
- Reverse 20-seed: 좌타자 K=4, 우타자 K=6, smoothing=1000, half-life=1, Ridge alpha=1000.
- `submit_020`: reverse scale 0.55.
- `submit_021`: reverse scale 0.40.
- R context: count × inning bucket × 투수손 × 타자손, smoothing=5000, scale=1.15.
- F expert: 기준 예측을 0.462 남기고 XGB 0.105, CatBoost 0.133, TabM 0.30의 residual을 반영.
- F TabM: BASE43, 2023~2024 F 55,696행, Brier loss, 6 epochs, seed 20260813.

실제 배포 산식과 정확한 모델 목록의 최종 근거는 `submit_021.zip/model/metadata.json`이다. ZIP을 풀지 않고 확인하려면 다음처럼 읽을 수 있다.

```powershell
tar -xOf submit/2026-08-13/submit_021.zip model/metadata.json
```

## 6. 현재 사용 피처

### 6.1 Enhanced 209개

원본 입력에서 `row_id`를 제외한 피처에 다음을 추가한다.

| 피처군 | 핵심 피처/방식 |
|---|---|
| 상황 상호작용 | `count_state`, `runner_out_state`, `handedness_matchup`, `score_abs`, `late_inning`, `high_leverage` |
| 표본량 변환 | 투수/타자/구종 누적수의 `log1p` |
| 최근 변화 | 누적 성공·middle 대비 최근 1/3/5경기 delta |
| 구성 일관성 | ball+strike rate gap, failure component sum/gap |
| 결측·수축 | 각 as-of rate의 missing flag, prior=200 Bayesian smoothing, reliability |
| 신인/희소성 | 투수·타자 `n=0`, `n<=25/100/500/1000` 플래그 |
| 최근 요약 | 성공률·middle률의 mean/std/range |
| crosswalk 신뢰도 | match season 수, similarity, margin, 메인/TrackMan 표본량 |
| TrackMan 가용성 | eligible seasons, total pitches, last season/gap/count, confidence flags |
| TrackMan 물리량 | 8개 물리량의 latest/recent/between-season mean·std 및 latest-recent delta |
| TrackMan 구종 | fastball/breaking/offspeed/other의 latest/recent/between-season rate·delta |

TrackMan 물리량 8개는 `rel_speed`, `spin_rate`, `induced_vert_break`, `horz_break`, `extension`, `rel_height`, `rel_side`, `zone_speed`다.

### 6.2 Insight-adjusted 211개

Enhanced 209개에 아래 두 피처를 더한다.

- `pitcher_success_adjusted_smoothed_200`
- `batter_success_adjusted_smoothed_200`

시즌 성공률 drift를 반영한 adjusted prior를 사용한다. 상수와 계산 근거는 최신 ZIP `metadata.json`의 `insight_prior_constants`가 권위 있는 값이다.

### 6.3 F 전용 210개

Insight-adjusted에서 route를 정의하는 `game_type`을 빼고 사용한다. F 전용 모델은 표본이 작고 시즌 구조 변화가 커서 완전 hard dispatch가 아니라 base에 residual을 작게 섞는다.

### 6.4 TabM BASE43

TabM은 raw ID와 TrackMan을 제외했다. 사용 피처는 다음 묶음이다.

- 상황 24개: season/month/day/inning, 공수·game type, count/out/score/runner, win expectancy, leverage, 좌우.
- as-of 19개: 투수 누적 6개, 최근 경기 성공/middle 6개, 타자 누적 3개, pitchmix 표본량과 구종률 4개.
- 코드 기준: `experiment/model_optimization/tabm_context/run_tabm_temporal.py`의 `determine_base_features()`.

Raw `pitcher_id`/`batter_id`를 TabM embedding에 직접 넣는 T2는 temporal overfit이 확인되어 현재 제출에서는 제외했다.

### 6.5 클러스터/Residual 피처

- 투수 유형과 타자 유형은 좌우를 분리한 profile에서 생성한다.
- reverse residual은 `season × pitcher_hand × batter_hand × count_state` 기준값을 제거한 뒤 학습한다.
- 타자 클러스터는 좌타자 K=4, 우타자 K=6.
- 투수 유형 × 타자 유형 pair residual은 recency weight와 smoothing을 적용한다.
- 새 선수나 unknown pair는 0 residual로 돌아가 base 확률을 보존한다.
- 20개 KMeans seed를 동일 가중 평균해 군집 초기값 변동을 줄인다.

피처의 **정확한 전체 컬럼 목록**은 최신 ZIP의 `model/metadata.json > feature_sets`와 `model/metadata.json > trackman_columns`를 사용한다. 문서에 복사된 목록보다 이 메타데이터를 우선한다.

## 7. EDA와 모델링에서 얻은 핵심 인사이트

### 7.1 Target과 시즌 drift

| 시즌 | 행 수 | 성공률 | 전 시즌 대비 해석 |
|---:|---:|---:|---|
| 2019 | 237,413 | 0.564670 | 가장 높음 |
| 2020 | 244,087 | 0.532712 | -3.20%p |
| 2021 | 247,088 | 0.532762 | 안정 |
| 2022 | 247,472 | 0.528920 | -0.38%p |
| 2023 | 245,525 | 0.499957 | -2.90%p |
| 2024 | 253,507 | 0.486105 | -1.39%p |

2019→2024에 **-7.86%p** 하락했다. 동일 투수 중 양 시즌 100구 이상 집단에서도 2022→2023 가중 평균 -2.61%p, 2023→2024 -1.00%p가 나타나 단순 선수 교체만으로 설명되지 않는다. 따라서 `season`, recency weighting, 시즌 prior 이동, calibration이 필수다.

연속 시즌 투수 교집합은 246~289명이고, 다음 시즌 신규 투수 비율은 약 24.3~33.1%다. raw ID embedding은 cold start fallback이 없으면 불안정하다.

### 7.2 F/R 구조 변화

| 시즌 | F n | F 성공률 | R n | R 성공률 |
|---:|---:|---:|---:|---:|
| 2019 | 25,786 | 0.689250 | 211,627 | 0.549490 |
| 2020 | 23,213 | 0.587774 | 220,874 | 0.526925 |
| 2021 | 25,861 | 0.703840 | 221,227 | 0.512763 |
| 2022 | 30,448 | 0.708749 | 217,024 | 0.503691 |
| 2023 | 25,686 | 0.472904 | 219,839 | 0.503118 |
| 2024 | 30,010 | 0.459280 | 223,497 | 0.489707 |

F는 전체 161,004행, R은 1,314,088행이다. F의 과거 전체 성공률은 0.6033이지만 2023년부터 완전히 다른 regime으로 바뀌었다. F는 전체 비중이 작아도 residual expert 효과가 컸지만 과거 전체 평균이나 hard dispatch를 쓰면 위험하다. R은 2024의 88.16%이므로 R에서 아주 작은 Brier 개선이 전체 점수에 크게 반영된다.

### 7.3 상황별 분포

카운트별 성공률:

| count | n | 성공률 | count | n | 성공률 |
|---|---:|---:|---|---:|---:|
| 0-0 | 380,996 | 0.526567 | 1-0 | 154,878 | 0.524852 |
| 0-1 | 181,775 | **0.534105** | 1-1 | 150,304 | 0.528456 |
| 0-2 | 89,281 | 0.518531 | 1-2 | 139,598 | 0.524442 |
| 2-0 | 54,187 | 0.518888 | 2-1 | 80,348 | 0.521755 |
| 2-2 | 119,702 | 0.520985 | 3-0 | 18,060 | 0.507309 |
| 3-1 | 35,425 | 0.504390 | 3-2 | 70,538 | **0.499603** |

좌우 조합별 성공률:

| 투수손×타자손 코드 | n | 성공률 |
|---|---:|---:|
| 1×1 | 170,292 | 0.490927 |
| 1×2 | 211,059 | **0.537504** |
| 2×1 | 525,455 | 0.530666 |
| 2×2 | 568,286 | 0.522124 |

이닝이 진행될수록 성공률이 1~3회 0.530956 → 4~6회 0.524868 → 7~9회 0.515629 → 연장 0.504862로 하락한다. 월도 3월 0.537506에서 10월 0.509171로 하락한다. 단, 시즌 구성과 game type이 섞인 marginal 통계이므로 반드시 `season × game_type`을 통제해 재확인한다.

주자 수의 단독 효과는 작다: 0명 0.523047, 1명 0.526022, 2명 0.523425, 만루 0.516134. LI도 단조롭지 않다: `<=0.5` 0.516875, `0.5~1` 0.527103, `1~2` 0.528970, `2~3` 0.524914, `>3` 0.518124. 단순 선형 효과보다 비선형 interaction이 적합하다.

### 7.4 과거 이력과 calibration

단변량 표본 분석에서 강한 순서는 대체로 다음과 같다.

| 피처 | 단변량 AUC strength | 방향 |
|---|---:|---|
| `asof_pitcher_success_rate` | 0.0487 | 높을수록 성공 증가 |
| `prev5_game_success_rate` | 0.0457 | 높을수록 성공 증가 |
| `asof_pitcher_reverse_rate` | 0.0454 | 높을수록 성공 감소 |
| `prev3_game_success_rate` | 0.0431 | 높을수록 성공 증가 |
| `prev1_game_success_rate` | 0.0373 | 높을수록 성공 증가 |
| `asof_batter_success_rate` | 0.0300 | 높을수록 성공 증가 |
| `season` | 0.0281 | 최근일수록 성공 감소 |

`asof_pitcher_success_rate` 자체의 전체 AUC는 약 0.5488이지만 극단값은 심하게 과신한다. 예를 들어 예측 구간 0.70~0.75의 실제 성공률은 0.6607, 0.90~0.95는 0.7018, 정확히 1에 가까운 집단은 0.5887이었다. 작은 표본에 Bayesian smoothing과 reliability를 적용해야 한다.

누적 표본량별 raw 성공률도 크게 다르지만 이는 선수 경험 효과와 시즌 drift가 혼재한다.

| 표본량 | 투수 기준 성공률 | 타자 기준 성공률 |
|---|---:|---:|
| 0 | 0.551768 | 0.602410 |
| 1~10 | 0.544706 | 0.574325 |
| 11~50 | 0.548602 | 0.574481 |
| 51~200 | 0.537794 | 0.561618 |
| 201~1000 | 0.530428 | 0.543568 |
| >1000 | 0.519001 | 0.512773 |

### 7.5 결측과 cold start

- 전체 행의 97.976%는 as-of 결측이 없다.
- 최근 1/3/5경기 피처 6개가 모두 비는 패턴이 28,259행, 1.916%다.
- 투수 첫 투구의 누적 rate 결측은 792행이다.
- 타자 첫 상대의 누적 rate 결측은 830행이다.
- 결측 행 성공률이 더 높아 결측 플래그 자체가 신호다. 타자 누적 결측은 0.6024, 존재는 0.5237이다.

권장 처리:

```text
smoothed_rate = (n * observed_rate + prior_strength * prior_rate) / (n + prior_strength)
reliability = n / (n + prior_strength)
```

현재 enhanced 모델은 기본적으로 `prior_strength=200`을 사용하며 결측이면 prior로 fallback한다. 선수-유형 pair residual에는 훨씬 강한 smoothing 1000을 사용한다.

### 7.6 TrackMan EDA

| 시즌 | 행 수 | 투수 | 타자 | 경기 |
|---:|---:|---:|---:|---:|
| 2019 | 255,957 | 394 | 423 | 909 |
| 2020 | 279,126 | 431 | 441 | 921 |
| 2021 | 301,032 | 425 | 425 | 989 |
| 2022 | 307,637 | 432 | 436 | 1,031 |
| 2023 | 315,100 | 447 | 435 | 1,039 |
| 2024 | 334,226 | 460 | 472 | 1,091 |

물리량 분포:

| 피처 | 결측률 | 평균 | 표준편차 | p01 | 중앙값 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| `rel_speed` | 0.425% | 135.964 | 9.372 | 111.605 | 137.412 | 152.434 |
| `spin_rate` | 0.695% | 2169.969 | 338.072 | 1010.711 | 2223.170 | 2857.310 |
| `induced_vert_break` | 0.561% | 25.784 | 24.329 | -42.908 | 30.105 | 61.821 |
| `horz_break` | 0.576% | 9.454 | 25.891 | -45.269 | 12.992 | 55.093 |
| `extension` | 0.430% | 1.759 | 0.167 | 1.362 | 1.760 | 2.146 |
| `rel_height` | 0.425% | 1.703 | 0.250 | 0.605 | 1.761 | 2.023 |
| `rel_side` | 0.425% | 0.264 | 0.532 | -0.899 | 0.431 | 1.130 |
| `zone_speed` | 0.442% | 124.590 | 8.386 | 102.287 | 125.915 | 139.375 |

구종군 비중은 2019 fastball 57.68% / breaking 25.83% / offspeed 14.48%에서 2024 fastball 46.78% / breaking 32.48% / offspeed 20.18%로 이동했다. 따라서 TrackMan 집계에도 시즌 recency와 latest-minus-recent 변화량이 필요하다.

TrackMan에는 극소수 잘못된 count/out 값이 있다: balls 범위 초과 1행, strikes 1행, outs 95행, inning 1행. 물리량은 삭제보다 pitcher-season×pitch-group 내부 winsorization과 missing indicator를 우선한다. 전체 행 기준 강한 IQR 제거는 투수 arm-slot 같은 실제 유형 차이를 지울 수 있다.

### 7.7 모델링으로 확인된 결론

- 2023은 거의 null Brier 수준으로 특히 어려웠다. 2024 하나에만 맞춘 튜닝은 위험하다.
- F가 `submit_007 → submit_013` 개선의 약 절반을 만들었으므로 별도 expert는 가치가 있지만 강한 hard route는 위험하다.
- XGBoost는 대략 18~24 leaves 부근이 안정적이었고 무작정 큰 모델은 overfit했다.
- 직접 Brier XGB와 순수 F1 loss는 실패했다. F1 혼합도 이득이 작거나 없었다.
- TrackMan 직접 embedding/대규모 injection은 검증 이득이 없었다. 신뢰도와 500구 gate가 필수다.
- hard cluster 자체보다 soft residual과 seed bagging이 상대적으로 안정적이었다.
- F TabM 20-seed는 단독 분산은 줄였지만 2023 expanding-month에서 모든 epoch의 최적 추가 가중치가 0이라 기각했다.
- 내부 Val에서는 CatBoost가 XGB보다 높았지만 Public은 XGB가 더 높았다. 작은 내부 BSS 차이를 확정적 순위로 해석하면 안 된다.

## 8. 성능 이력

| 모델/제출 | 설계 | Val2024 BSS | Public BSS |
|---|---|---:|---:|
| Cat V1 | CatBoost 기준선 | 750.571 | 838.492 |
| XGB V1 | XGBoost 기준선 | 745.131 | 873.075 |
| `submit_013` | reverse 타자 클러스터 3-seed | 812.704 | **895.404** |
| `submit_015` | 013 + R context residual | 830.523 | - |
| `submit_017` | 015 + F XGB/Cat expert | 833.342 | - |
| `submit_019` | 017 + F TabM | 835.861 | - |
| `submit_020` | 019 + reverse 20-seed, scale 0.55 | 835.795 | - |
| `submit_021` | 019 + reverse 20-seed, scale 0.40 | **836.503** | - |

주의: 과거 문서의 `submit_019=837.214`는 실제 추론 산식과 다른 계산이었다. exact parity 평가값 **835.861235**가 맞다.

최신 exact Val2024 세부 결과:

| 시스템 | 전체 Brier | 전체 BSS | R BSS | F BSS | 예측 평균 |
|---|---:|---:|---:|---:|---:|
| `submit_017` 재구성 | 0.24772518 | 833.3417 | 832.8604 | 511.0342 | 0.485817 |
| `submit_019` 재구성 | 0.24771889 | 835.8612 | 832.8604 | **532.4434** | 0.485735 |
| `submit_020` | 0.24771905 | 835.7945 | 832.7908 | 532.3985 | 0.485730 |
| `submit_021` | **0.24771728** | **836.5029** | **834.0723** | 528.8141 | 0.486347 |

Val2024 Target 평균은 0.48610492다. 021은 R을 개선했지만 F는 019보다 나빠졌다. 전체 향상은 R의 큰 모수에서 발생했다. 021의 019 대비 +0.642 BSS는 3·4·10월 개선, 5·7·9월 악화로 월별 일관성이 없고 투수 cluster bootstrap CI도 0을 포함한다.

단순 통계 기준선:

| 예측 | 범위 | Brier | AUC |
|---|---|---:|---:|
| 전체 평균 상수 0.523766 | 전체 train | 0.24943518 | 0.5000 |
| `asof_pitcher_success_rate`, global fill | 전체 train | 0.24818305 | 0.54883 |
| `asof_batter_success_rate`, global fill | 전체 train | 0.24931627 | 0.53055 |
| 2019~2023 평균 상수 | Val2024 | 0.25187505 | 0.5000 |
| 투수 as-of rate, 과거 평균 fill | Val2024 | 0.25027446 | 0.53417 |

Public 결과가 내부 검증과 다르게 움직였다. Cat V1은 Val 750.571로 XGB V1 745.131보다 높았지만 Public은 Cat 838.492, XGB 873.075였다. 따라서 모델 선택은 단일 Val 차이뿐 아니라 fold 안정성, residual 상관, calibration, Public probe를 함께 본다.

## 9. 개발 환경과 requirements

Python 3.11 권장. 현재 작업 환경의 재현용 패키지는 루트 [`requirements-dev.txt`](requirements-dev.txt)에 고정했다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

PyTorch는 GPU/CUDA에 맞는 wheel을 **별도로 먼저 설치**하는 편이 안전하다. 현재 로컬은 `torch 2.5.1+cu121`, 평가 서버 기본 환경은 `torch 2.7.1+cu128`이다. `requirements-dev.txt`는 잘못된 대형 CUDA wheel을 자동으로 받지 않도록 torch를 포함하지 않는다.

최신 제출 ZIP의 `requirements.txt`는 다음 네 개만 포함한다.

```text
xgboost==3.1.1
catboost==1.2.8
tabm==0.0.3
rtdl-num-embeddings==0.0.12
```

평가 서버 기본 설치 패키지인 numpy/pandas/scipy/sklearn/joblib/torch는 제출 requirements에 중복 기재하지 않는다.

### 9.1 설치 확인

```powershell
python -c "import numpy,pandas,sklearn,xgboost,catboost,lightgbm,optuna,pyarrow,matplotlib,seaborn; print('analysis environment OK')"
python -c "import torch,tabm; print(torch.__version__, torch.cuda.is_available())"
```

PyTorch가 없어도 아래 기본 EDA와 XGBoost 분석은 가능하다. TabM을 실행할 때만 GPU에 맞는 PyTorch가 필요하다.

### 9.2 Jupyter 시작

```powershell
python -m ipykernel install --user --name lgaimers --display-name "Python (LGAIMERS)"
jupyter notebook
```

노트북의 작업 디렉터리는 반드시 프로젝트 루트 `C:\Users\isj67\Desktop\LGAIMERS`로 둔다.

## 10. README만으로 시작하는 데이터 분석

### 10.1 안전한 로딩과 기본 검사

```python
from pathlib import Path
import gc
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\isj67\Desktop\LGAIMERS")
DATA = ROOT / "data"
TARGET = "control_success"

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

assert train.shape == (1_475_092, 49)
assert test.shape[1] == 48
assert TARGET in train and TARGET not in test
assert train["row_id"].is_unique and test["row_id"].is_unique
assert set(train[TARGET].unique()) <= {0, 1}
assert train.drop(columns=TARGET).columns.tolist() == test.columns.tolist()

print(train.shape, train.memory_usage(deep=True).sum() / 2**20)
print(train[TARGET].value_counts(dropna=False))
print(train[TARGET].mean())
```

메모리가 부족할 때는 ID와 작은 정수부터 줄인다.

```python
for c in ["season", "game_month", "game_dayofweek", "inning",
          "balls_before", "strikes_before", "outs_before",
          "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on"]:
    train[c] = pd.to_numeric(train[c], downcast="integer")
for c in train.select_dtypes("float64"):
    train[c] = train[c].astype("float32")
for c in ["top_bottom", "game_type", "base_state"]:
    train[c] = train[c].astype("category")
gc.collect()
```

### 10.2 한 번에 만드는 기본 프로파일

```python
def profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in frame.columns:
        s = frame[c]
        row = {
            "column": c,
            "dtype": str(s.dtype),
            "missing_n": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "nunique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            row.update({"min": s.min(), "p01": q.loc[0.01], "p25": q.loc[0.25],
                        "median": q.loc[0.5], "p75": q.loc[0.75],
                        "p99": q.loc[0.99], "max": s.max()})
        rows.append(row)
    return pd.DataFrame(rows)

train_profile = profile(train)
display(train_profile.sort_values(["missing_pct", "nunique"], ascending=False))
```

### 10.3 시간 drift와 조건부 성공률

```python
def target_table(df, keys):
    return (df.groupby(keys, observed=True)[TARGET]
              .agg(n="size", successes="sum", success_rate="mean")
              .reset_index()
              .sort_values(keys))

tables = {
    "season": target_table(train, ["season"]),
    "season_game_type": target_table(train, ["season", "game_type"]),
    "count": target_table(train, ["balls_before", "strikes_before"]),
    "hands": target_table(train, ["pitcher_hand", "batter_hand"]),
    "inning": target_table(train.assign(
        inning_bucket=pd.cut(train["inning"], [0, 3, 6, 9, np.inf],
                              labels=["1-3", "4-6", "7-9", "10+"])),
        ["season", "inning_bucket"]),
}
for name, table in tables.items():
    print(f"\n[{name}]")
    display(table)
```

시즌을 섞은 전체 평균만 보고 결론을 내리지 않는다. 모든 가설은 최소한 `season`, `game_type`으로 다시 나누고 표본 수 `n`을 함께 본다.

### 10.4 시각화

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
season_rate = target_table(train, ["season"])
sns.lineplot(data=season_rate, x="season", y="success_rate", marker="o")
plt.axhline(train[TARGET].mean(), color="gray", ls="--")
plt.title("Control success rate by season")
plt.tight_layout()
plt.show()

count_rate = target_table(train, ["balls_before", "strikes_before"])
pivot = count_rate.pivot(index="balls_before", columns="strikes_before",
                         values="success_rate")
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlBu")
plt.title("Control success by count")
plt.tight_layout()
plt.show()
```

### 10.5 Brier/BSS와 그룹 성능

```python
from sklearn.metrics import roc_auc_score, log_loss

def probability_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    brier = np.mean((p - y) ** 2)
    null = y.mean() * (1 - y.mean())
    return {
        "n": len(y), "target_mean": y.mean(), "pred_mean": p.mean(),
        "brier": brier, "normalized_brier": brier / null,
        "bss_raw": 100000 * (1 - brier / null),
        "auc": roc_auc_score(y, p), "logloss": log_loss(y, p),
    }

def group_metrics(frame, pred_col, group_col):
    return pd.DataFrame([
        {group_col: key, **probability_metrics(g[TARGET], g[pred_col])}
        for key, g in frame.groupby(group_col, observed=True)
    ])
```

실험 기록에는 전체뿐 아니라 R/F, 월, 투수손×타자손, 신규/기존 선수별 Brier를 저장한다. BSS는 그룹의 Target 평균이 다르면 절대값이 직접 비교되지 않으므로 delta Brier도 함께 기록한다.

### 10.6 누수 없는 순방향 split

```python
FOLDS = {
    2023: (train["season"] < 2023, train["season"] == 2023),
    2024: (train["season"] < 2024, train["season"] == 2024),
}

for valid_season, (tr_mask, va_mask) in FOLDS.items():
    assert train.loc[tr_mask, "season"].max() < valid_season
    assert train.loc[va_mask, "season"].nunique() == 1
    print(valid_season, tr_mask.sum(), va_mask.sum())
```

동일 시즌 안에서 상세 선택이 필요하면 2023년을 월별 expanding으로 쓴다. 예: 3~4월 학습→5월 검증, 3~5월 학습→6월 검증. 2024는 설정 고정 뒤 마지막 한 번만 여는 gate로 취급한다.

### 10.7 기본 stateless 파생 피처

```python
RATE_SPECS = [
    ("pitcher_success", "asof_pitcher_success_rate", "asof_pitcher_n"),
    ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("batter_success", "asof_batter_success_rate", "asof_batter_n"),
    ("batter_middle", "asof_batter_middle_rate", "asof_batter_n"),
]

def add_stateless_features(df, priors, strength=200.0):
    out = df.copy()
    out["count_state"] = (out["balls_before"].astype(str) + "-" +
                          out["strikes_before"].astype(str))
    out["handedness_matchup"] = (out["pitcher_hand"].astype(str) + "_" +
                                  out["batter_hand"].astype(str))
    out["score_abs"] = out["score_diff_pitcher_team"].abs()
    out["late_inning"] = (out["inning"] >= 7).astype("int8")
    out["high_leverage"] = (out["li"] >= 2).astype("int8")
    out["log1p_asof_pitcher_n"] = np.log1p(out["asof_pitcher_n"])
    out["log1p_asof_batter_n"] = np.log1p(out["asof_batter_n"])
    out["pitcher_success_delta_prev1"] = (
        out["asof_pitcher_prev1_game_success_rate"] - out["asof_pitcher_success_rate"])
    out["pitcher_success_delta_prev3"] = (
        out["asof_pitcher_prev3_game_success_rate"] - out["asof_pitcher_success_rate"])
    out["pitcher_success_delta_prev5"] = (
        out["asof_pitcher_prev5_game_success_rate"] - out["asof_pitcher_success_rate"])

    for name, rate_col, n_col in RATE_SPECS:
        n = out[n_col].astype(float)
        rate = out[rate_col].fillna(priors[name]).astype(float)
        out[f"{name}_is_missing"] = out[rate_col].isna().astype("int8")
        out[f"{name}_smoothed_{int(strength)}"] = (
            n * rate + strength * priors[name]) / (n + strength)
        out[f"{name}_reliability_{int(strength)}"] = n / (n + strength)
    return out

# prior는 반드시 train fold에서만 계산한다.
tr = train[train["season"] < 2024]
priors = {
    "pitcher_success": tr[TARGET].mean(),
    "pitcher_reverse": tr["asof_pitcher_reverse_rate"].median(),
    "pitcher_middle": tr["asof_pitcher_middle_rate"].median(),
    "batter_success": tr[TARGET].mean(),
    "batter_middle": tr["asof_batter_middle_rate"].median(),
}
```

### 10.8 빠른 XGBoost temporal baseline

이 코드는 TrackMan과 validation Target을 사용하지 않는 분석용 시작점이다. 최종 최고 모델 재현 코드가 아니라 새 피처의 방향을 빠르게 확인하는 기준선이다.

```python
from xgboost import XGBClassifier

DROP = ["row_id", TARGET]
CATS = ["top_bottom", "game_type", "base_state", "count_state",
        "handedness_matchup"]

def fit_category_maps(fit_df, columns):
    return {c: {v: i for i, v in enumerate(fit_df[c].dropna().unique())}
            for c in columns}

def encode(frame, feature_cols, maps):
    x = frame[feature_cols].copy()
    for c, mapping in maps.items():
        x[c] = x[c].map(mapping).fillna(-1).astype("int32")
    return x.replace([np.inf, -np.inf], np.nan)

base_train = add_stateless_features(train, priors, strength=200)
tr_mask = base_train["season"] < 2024
va_mask = base_train["season"] == 2024
feature_cols = [c for c in base_train.columns if c not in DROP]
maps = fit_category_maps(base_train.loc[tr_mask], [c for c in CATS if c in feature_cols])
X_tr = encode(base_train.loc[tr_mask], feature_cols, maps)
X_va = encode(base_train.loc[va_mask], feature_cols, maps)
y_tr = base_train.loc[tr_mask, TARGET]
y_va = base_train.loc[va_mask, TARGET]

model = XGBClassifier(
    n_estimators=3000, learning_rate=0.03, max_depth=5,
    min_child_weight=100, subsample=0.85, colsample_bytree=0.80,
    reg_alpha=1.0, reg_lambda=10.0,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", max_bin=256, n_jobs=6,
    random_state=20260813, early_stopping_rounds=100,
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
p_va = model.predict_proba(X_va)[:, 1]
print(probability_metrics(y_va, p_va))
```

이 baseline에서도 raw ID를 연속형 숫자로 보는 문제가 있다. 첫 비교에서는 `pitcher_id`, `batter_id`, team ID를 제외한 모델과 함께 평가한다. CatBoost에서는 이 네 ID와 문자열 컬럼을 categorical로 지정하되 validation 이전 fold만으로 학습한다.

### 10.9 TrackMan strict-as-of 집계 골격

```python
TM_NUM = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
          "extension", "rel_height", "rel_side", "zone_speed"]

tm = pd.read_csv(DATA / "trackman_history.csv", parse_dates=["game_date"])

# 명백한 구조 오류만 제거. 물리량 전체 outlier 일괄 삭제는 하지 않는다.
tm = tm[
    tm["balls_before"].between(0, 3) &
    tm["strikes_before"].between(0, 2) &
    tm["outs_before"].between(0, 2) &
    tm["inning"].between(1, 30)
].copy()

season_n = (tm.groupby(["pitcher_trackman_id", "season"], observed=True)
              .size().rename("season_pitch_n").reset_index())
eligible = season_n.query("season_pitch_n >= 500")
tm500 = tm.merge(eligible, on=["pitcher_trackman_id", "season"], how="inner")

def trackman_profile_for_prediction(tm500, prediction_season):
    past = tm500[tm500["season"] < prediction_season].copy()
    numeric = (past.groupby("pitcher_trackman_id", observed=True)[TM_NUM]
                   .agg(["mean", "std"]))
    numeric.columns = [f"tm_{a}_{b}" for a, b in numeric.columns]
    pitchmix = (pd.crosstab(past["pitcher_trackman_id"], past["pitch_type_group"],
                            normalize="index")
                  .add_prefix("tm_pitch_group_").add_suffix("_rate"))
    return numeric.join(pitchmix, how="outer").reset_index()

tm_val2024 = trackman_profile_for_prediction(tm500, 2024)  # 2019~2023만
tm_final2025 = trackman_profile_for_prediction(tm500, 2025)  # 2019~2024
```

이 결과를 메인 데이터에 붙이려면 독립적으로 검증된 crosswalk가 필요하다. raw ID는 교집합이 0이므로 숫자나 문자열 유사성만으로 직접 조인하지 않는다. crosswalk의 confidence, matched season 수, 표본량과 unknown fallback을 피처로 함께 둔다.

### 10.10 새 피처 채택 기준

| 단계 | 통과 조건 |
|---|---|
| 무결성 | 행 수·순서·`row_id` 보존, inf 없음 |
| 누수 | validation 시즌/Target/test 분포 미사용 |
| 단변량 | 시즌별 방향, 결측 및 표본량 확인 |
| Ablation | 동일 seed/parameter에서 baseline 대비 Brier 감소 |
| 안정성 | 2023와 2024 방향 일치 또는 구조 차이 설명 가능 |
| Residual | 기존 예측과 다른 오차를 줄이며 correlation이 과도하지 않음 |
| Calibration | 전체/R/F/month별 prediction mean과 target mean 확인 |
| 제출 | 전체 245,789행 시간·메모리·경로 smoke 통과 |

모든 실험은 최소한 `experiment`, `features`, `train_seasons`, `valid_season`, `seed`, `params`, `n`, `brier`, `bss_raw`, `auc`, `pred_mean`, `elapsed_sec`를 CSV/JSON에 남긴다.

## 11. 주요 코드 위치

| 목적 | 위치 |
|---|---|
| EDA 재생성 | `experiment/control_success_eda/run_eda.py` |
| 협업 피처 예제 | `experiment/control_success_feature_template/feature_example.py`, `.ipynb` |
| 공통 temporal 피처 | `experiment/model_optimization/v2_temporal_features.py` |
| Optuna family 탐색 | `experiment/model_optimization/run_optuna_family.py` |
| Enhanced seed OOF | `experiment/model_optimization/build_enhanced_seed_oof.py` |
| Insight 피처 ablation | `experiment/model_optimization/benchmark_insight_features.py` |
| 클러스터/매치업 전체 | `experiment/model_optimization/pitcher_cluster_matchup/` |
| Reverse 20-seed OOF | `pitcher_cluster_matchup/src/run_reverse_seedbag20.py` |
| Reverse 20-seed 최종 산출물 | `pitcher_cluster_matchup/src/build_reverse20_artifacts.py` |
| 최신 exact Val2024 | `pitcher_cluster_matchup/src/evaluate_reverse20_submissions.py` |
| 최신 ZIP 제작 | `pitcher_cluster_matchup/src/build_reverse20_submissions.py` |
| TabM 실험 | `experiment/model_optimization/tabm_context/` |
| 실패유형 component 모델 | `experiment/model_optimization/failure_experts/` |
| 검증 레지스트리 생성 | `experiment/model_optimization/build_validation_registry.py` |

표의 `pitcher_cluster_matchup/...` 경로는 모두 `experiment/model_optimization/` 아래다.

## 12. 재실행 순서

### 12.1 데이터/환경 확인

```powershell
python -c "import pandas as pd; print(pd.read_csv('data/train.csv', nrows=5).shape)"
python experiment/control_success_eda/run_eda.py
python experiment/control_success_feature_template/feature_example.py --nrows 20000 --output-dir experiment/control_success_feature_template/smoke_outputs
```

### 12.2 Reverse 20-seed 재검증

```powershell
python experiment/model_optimization/pitcher_cluster_matchup/src/run_reverse_seedbag20.py
python experiment/model_optimization/pitcher_cluster_matchup/src/build_reverse20_artifacts.py
python experiment/model_optimization/pitcher_cluster_matchup/src/evaluate_reverse20_submissions.py
```

첫 스크립트는 기존 seed cache가 있으면 재사용한다. 캐시를 지우거나 덮어쓰지 말고 새 실험은 별도 폴더/이름으로 만든다.

### 12.3 ZIP 제작

```powershell
python experiment/model_optimization/pitcher_cluster_matchup/src/build_reverse20_submissions.py
```

현재 스크립트의 날짜와 번호는 2026-08-13의 020/021로 고정되어 있다. **현재 파일을 덮어쓰지 말고**, 새 날짜 폴더와 다음 제출 번호로 `OUTPUT_DIR`, `SPECS`를 변경한 뒤 실행한다.

### 12.4 제출 ZIP 검사

- ZIP 루트에 `model/`, `script.py`, `requirements.txt`만 두고 추가 최상위 폴더를 만들지 않는다.
- 파일명은 30자 미만, 규칙은 `submit/날짜/submit_NNN.zip`.
- `script.py`는 `Path(__file__).resolve().parent` 기준으로 `model`, `data`, `output`을 찾는다. CWD 상대경로 `model/model.pt` 또는 `/app/model/model.pt`를 하드코딩하지 않는다.
- 결과는 반드시 `output/submission.csv`에 `row_id`, `control_success` 순서로 저장한다.
- 확률의 finite/범위, 행 수와 순서, ZIP CRC, 모델 파일 누락을 확인한다.
- 실제 서버 제한: 설치 10분, 추론 10분, 6 vCPU, 28GB RAM, L4 22.4GB, 인터넷 없음.

## 13. 완료·기각·보류 실험

### 적용 완료

- XGB/CatBoost seed bag + beta calibration.
- 시즌 drift adjusted success smoothing.
- R count/inning/hand residual.
- F XGB/CatBoost/TabM partial pooling.
- reverse batter cluster seed bag 3/20.
- TrackMan strict-as-of 500구 profile.

### 기각 또는 약함

- 직접 Brier objective XGB.
- 순수 F1 loss와 강한 F1 혼합.
- 무작정 확대한 XGB.
- raw player ID를 넣은 TabM T2.
- TrackMan 직접 embedding/과도한 injection.
- hard F/R dispatch.
- F TabM 20-seed 추가 앙상블.

### 아직 최종 시스템에 미연결

- Outside failure component Optuna: 101회 시도, best trial 97, Val2022 component BSS 1586.007, Val2023 1790.799.
- Middle/Reverse failure component gate.
- failure component는 보조 label을 재구성해 사용하므로 대회 규정과 문제 취지에 대한 별도 검토가 필요하다.

## 14. 알려진 위험과 문서 신뢰도

- 최신 시스템의 TrackMan은 strict prior-season이라 미래 누수는 막았지만, 사용자의 문자 그대로인 “Val2024에서는 TrackMan 미사용” 기준선은 아직 없다.
- `validation_registry.csv`는 실험 수가 많지만 최신 residual 시스템의 exact 산식을 모두 담지 못한다. 최신 비교에는 `reverse20_submission_metrics.json`을 우선한다.
- 2024를 반복 확인한 실험이 많아 selection bias가 있다. 다음 선택은 2022→2023, 2023 월별 expanding, 2023→2024를 함께 보는 nested/rolling 기준이 필요하다.
- `submit_021`의 019 대비 +0.642 BSS는 투수 단위 신뢰구간이 0을 포함하고 월별 방향도 일관되지 않는다.
- 로컬 test는 5행뿐이다. 245,789행 smoke는 행 반복과 강제 R/F 혼합으로 속도·기본 분기만 검증했으며 실제 categorical breadth를 보장하지 않는다.
- failure component label은 다음 행의 누적률 변화를 이용해 99.89% 정도 복원했다. 본 Target 자체는 거의 일치하지만 organizer 검증 시 설명 가능성과 허용 여부가 위험 요소다.

## 15. 다음 Agent의 권장 작업 순서

1. `submit_021`과 `submit_020` Public 점수를 받아 `SUBMISSION_LOG.md`와 `status.md`에 기록한다.
2. Public 결과로 reverse20과 scale 0.40/0.55의 방향을 결정한다. 내부 +0.6만으로 확정하지 않는다.
3. **TrackMan을 완전히 뺀 Val2024 clean baseline**을 동일 모델/동일 피처 ablation으로 재생성한다.
4. 2022→2023와 2023 월별 expanding을 selection fold, 2024를 최종 gate로 사용하는 nested pipeline을 고정한다.
5. Outside expert를 작은 residual scale grid로 연결하되 OOF prediction만으로 scale을 정한다.
6. R 대규모 표본에서 count×hand×pitcher reliability 보정을 더 미세하게 찾는다.
7. 새 아이디어는 전체 재학습 전에 feature-only ablation → OOF residual correlation → 작은 blend → exact parity 순으로 검증한다.

## 16. 협업 피처 전달 규격

새 피처는 [`experiment/control_success_feature_template/feature_example.py`](experiment/control_success_feature_template/feature_example.py)를 복사해 만든다.

- 파일: `feat_<이름>_<주제>.py`
- prefix: `<이름>_<주제>__`
- 입력: 원본과 같은 행 단위 DataFrame.
- 출력: `row_id + 신규 피처`.
- 행 수, `row_id`, 순서를 그대로 유지.
- 일반 모듈에서 Target 사용 금지.
- test 행 간 집계 금지.
- Stateful builder는 CV fold 내부에서만 fit.
- 결과 공유: train/test parquet, feature summary CSV, manifest JSON, 코드 해시, temporal ablation.
- 병합은 `row_id` one-to-one merge로 하고 `concat(axis=1)`은 사용하지 않는다.

노트북은 설명·검토용이며 최종 재생성 로직은 반드시 `.py`에 둔다.

## 17. 인수인계 완료 기준

새 Agent가 아래를 확인하면 작업을 이어갈 준비가 된 것이다.

- [ ] `data/` 네 파일 존재와 schema 확인.
- [ ] `requirements-dev.txt` 환경에서 핵심 import 확인.
- [ ] `status.md`와 최신 날짜의 `SUBMISSION_LOG.md` 읽음.
- [ ] Val2024/TrackMan 규칙과 test row-independence 규칙 이해.
- [ ] `submit_021` metadata에서 실제 feature/model/blend 확인.
- [ ] 새 실험 번호와 출력 폴더를 기존 산출물과 다르게 지정.
- [ ] Brier, 전체 BSS, R/F별 BSS, 월별 Brier, prediction mean을 함께 기록.
- [ ] 최종 ZIP을 서버와 같은 경로 구조로 smoke test.

## 부록 A. 메인 데이터 전체 컬럼 사전

| 컬럼 | 종류 | 분석 시 의미/주의점 |
|---|---|---|
| `row_id` | 식별자 | 제출 조인 전용. 순번·시간 피처로 사용 금지 |
| `season` | 시점 | 2019~2024 train, 2025 test. 가장 큰 drift 축 |
| `game_month` | 시점 | 월, train 3~10 |
| `game_dayofweek` | 시점 | 월요일 0~일요일 6 |
| `inning` | 상황 | 투구 직전 이닝, train 1~13 |
| `top_bottom` | 범주 | `T` 초, `B` 말 |
| `game_type` | 범주 | `F`, `R`; 의미를 임의 해석하지 말고 별도 regime 코드로 사용 |
| `balls_before` | 상황 | 투구 직전 볼 0~3 |
| `strikes_before` | 상황 | 투구 직전 스트라이크 0~2 |
| `outs_before` | 상황 | 투구 직전 아웃 0~2 |
| `run_top_before` | 점수 | 초 공격 팀의 현재 점수 |
| `run_bot_before` | 점수 | 말 공격 팀의 현재 점수 |
| `run_total_before` | 점수 | 두 팀 합계; 위 두 점수에서 완전 유도 가능 |
| `score_diff_home` | 점수 | 홈 팀 기준 점수 차; 두 점수에서 완전 유도 가능 |
| `score_diff_pitcher_team` | 점수 | 투수 팀 기준 점수 차 |
| `runner_on_1b` | 주자 | 1루 주자 여부 0/1 |
| `runner_on_2b` | 주자 | 2루 주자 여부 0/1 |
| `runner_on_3b` | 주자 | 3루 주자 여부 0/1 |
| `num_runners_on` | 주자 | 세 runner flag 합, 완전 유도 가능 |
| `base_state` | 주자 | `___`, `1__`, `_2_`, `__3`, `12_`, `1_3`, `_23`, `123` |
| `home_win_expectancy` | 상황 | 홈 기대 승률 0~100 |
| `away_win_expectancy` | 상황 | 원정 기대 승률 0~100, 홈 값과 거의 보완 |
| `li` | 상황 | leverage index, 클수록 중요한 상황 |
| `pitcher_id` | 범주 ID | 익명 투수 ID 792개. 숫자 크기에 순서 의미 없음 |
| `batter_id` | 범주 ID | 익명 타자 ID 830개. 숫자 크기에 순서 의미 없음 |
| `pitcher_hand` | 범주 | 좌우 유형 코드 1/2; 의미를 데이터 설명 밖으로 확장하지 않음 |
| `batter_hand` | 범주 | 좌우 유형 코드 1/2 |
| `pitcher_team_id` | 범주 ID | 투수 팀 13개 |
| `batter_team_id` | 범주 ID | 타자 팀 13개 |
| `asof_pitcher_n` | 과거 이력 | 해당 투구 직전까지 투수 누적 투구 수 |
| `asof_pitcher_success_rate` | 과거 이력 | 투수 누적 제구 성공률 |
| `asof_pitcher_reverse_rate` | 과거 이력 | 투수 누적 의도 반대성 투구 비율 |
| `asof_pitcher_middle_rate` | 과거 이력 | 투수 누적 가운데/위험 코스 비율 |
| `asof_pitcher_ball_rate` | 과거 이력 | 투수 누적 볼성 결과 비율 |
| `asof_pitcher_strike_rate` | 과거 이력 | 투수 누적 스트라이크성 결과 비율 |
| `asof_pitcher_prev1_game_success_rate` | 최근 이력 | 직전 1경기 제구 성공률 |
| `asof_pitcher_prev3_game_success_rate` | 최근 이력 | 직전 3경기 제구 성공률 |
| `asof_pitcher_prev5_game_success_rate` | 최근 이력 | 직전 5경기 제구 성공률 |
| `asof_pitcher_prev1_game_middle_rate` | 최근 이력 | 직전 1경기 가운데/위험 코스 비율 |
| `asof_pitcher_prev3_game_middle_rate` | 최근 이력 | 직전 3경기 가운데/위험 코스 비율 |
| `asof_pitcher_prev5_game_middle_rate` | 최근 이력 | 직전 5경기 가운데/위험 코스 비율 |
| `asof_batter_n` | 과거 이력 | 타자가 직전까지 상대한 누적 투구 수 |
| `asof_batter_success_rate` | 과거 이력 | 타자가 상대한 누적 투구의 제구 성공률 |
| `asof_batter_middle_rate` | 과거 이력 | 타자가 상대한 누적 투구의 가운데/위험 코스 비율 |
| `asof_pitcher_pitchmix_n` | 과거 이력 | 투수 구종 이력 표본 수; `asof_pitcher_n`과 동일 |
| `asof_pitcher_fastball_rate` | 과거 이력 | fastball 계열 누적 사용 비율 |
| `asof_pitcher_breaking_rate` | 과거 이력 | breaking 계열 누적 사용 비율 |
| `asof_pitcher_offspeed_rate` | 과거 이력 | offspeed 계열 누적 사용 비율 |
| `control_success` | Target | train에만 존재. 성공 1, 실패 0 |

## 부록 B. TrackMan 전체 컬럼 사전

| 컬럼 | 분석 시 의미/주의점 |
|---|---|
| `trackman_id` | 로그 행 고유 ID |
| `season` | 2019~2024 |
| `game_date` | `MM/DD/YYYY`, 날짜로 변환해 as-of 제한 가능 |
| `game_month` | 경기 월 |
| `game_dayofweek` | 월 0~일 6 |
| `trackman_game_id` | TrackMan 경기 ID, 메인 `row_id`와 직접 대응하지 않음 |
| `pitch_no` | 경기 내 투구 번호 |
| `inning` | 이닝 |
| `top_bottom` | `Top`/`Bottom`; 메인의 `T`/`B`와 표기가 다름 |
| `balls_before` | 투구 직전 볼 |
| `strikes_before` | 투구 직전 스트라이크 |
| `outs_before` | 투구 직전 아웃 |
| `pitch_of_pa` | 타석 내 투구 순번 |
| `pitcher_trackman_id` | TrackMan 투수 ID, 메인 투수 ID와 직접 불일치 |
| `batter_trackman_id` | TrackMan 타자 ID, 메인 타자 ID와 직접 불일치 |
| `pitcher_hand` | `Left`/`Right` |
| `batter_hand` | `Left`/`Right` |
| `pitcher_team` | 팀 문자열, 시즌에 따라 명칭 변경 가능 |
| `batter_team` | 팀 문자열, 시즌에 따라 명칭 변경 가능 |
| `tagged_pitch_type` | 수동/태깅 구종, 17종 |
| `auto_pitch_type` | 자동 분류 구종, 11종, 72행 결측 |
| `pitch_type_group` | `fastball`, `breaking`, `offspeed`, `other` |
| `rel_speed` | 릴리스 구속 |
| `spin_rate` | 회전수 |
| `induced_vert_break` | 유도 수직 무브먼트 |
| `horz_break` | 수평 무브먼트 |
| `extension` | 릴리스 확장 거리 |
| `rel_height` | 릴리스 높이 |
| `rel_side` | 릴리스 좌우 위치 |
| `zone_speed` | 홈플레이트 부근 구속 |

## 부록 C. 제출 환경 요약

```text
OS: Ubuntu 22.04.5 LTS
Python: 3.11.15
CPU/RAM: 6 vCPU / 28GB
GPU: NVIDIA L4 22.4GiB, CUDA 12.8
설치 제한: 10분
추론 제한: 10분 / 245,789행
인터넷: 패키지 설치 이후 불가
ZIP: 압축 10GB 이하, 해제 32GB 이하
필수 루트: model/, script.py, requirements.txt
입력: data/ 읽기 전용
출력: output/submission.csv
```
