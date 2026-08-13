# 미사용 피처 점검 및 활용 계획

최종 갱신: 2026-08-12  
점검 기준: 현재 최상위 후보 `submit_018.zip`

## 1. 결론

메인 `train/test` 입력 48개 중 모델이 빠뜨린 실질 입력 컬럼은 없다. `row_id`만 의도적으로 제외했고, 나머지 47개는 모두 현재 전체 모델에 들어간다. F 전용 모델은 `game_type=F`가 상수이므로 `game_type`까지 제외한다.

남은 성능 여지는 다음 세 곳에 있다.

1. TrackMan에서 아직 예측 통계로 쓰지 않은 날짜·경기·투구순번·타석순번 정보
2. 이미 사용 중인 메인 컬럼의 새로운 상황 상호작용
3. 너무 크게 넣어 성능이 하락했던 구종별·군집 피처를 10~40개의 압축 도메인 피처로 재설계

가장 먼저 시도할 것은 `TM_WORKLOAD`, `TM_SITUATIONAL_STYLE`, `MAIN_CONTEXT_COMPACT` 세 묶음이다.

## 2. 현재 사용 범위

| 데이터 | 원본 입력 | 현재 사용 | 미사용 또는 제한적 사용 |
|---|---:|---:|---|
| `train/test` | 48개 | 47개 | `row_id`만 제외 |
| F 전용 모델 | 48개 | 46개 | `row_id`, 상수인 `game_type` 제외 |
| TrackMan | 30개 | 20개를 통계 또는 ID 매칭에 사용 | 10개 완전 미사용, 9개는 매칭에만 사용 |
| TrackMan 물리값 | 8개 | 투수-시즌 mean/std, 최신·최근·시즌간 변동 | quantile·공분산·상황별 편차는 미사용 |
| TrackMan 구종 | 3개 | `pitch_type_group` 4종 비율 | 상세 tagged/auto 구종과 두 분류기의 불일치는 미사용 |

현재 주력 피처 수는 enhanced 209개, insight-adjusted 211개다. TrackMan은 검증 시즌 이전 자료만 사용하고 투수-시즌 500구 이상 조건을 지킨다.

## 3. TrackMan 원본 컬럼 사용 감사

### 3.1 예측 통계에 직접 사용

- `pitcher_trackman_id`, `season`, `pitch_type_group`
- `rel_speed`, `spin_rate`, `induced_vert_break`, `horz_break`
- `extension`, `rel_height`, `rel_side`, `zone_speed`

현재 집계는 투수-시즌별 평균·표준편차, 최신 시즌, recency 평균, 시즌간 표준편차, 4개 구종군 비율이다.

### 3.2 ID crosswalk에만 사용하고 예측 통계에는 미사용

- `game_month`, `game_dayofweek`, `inning`, `top_bottom`
- `balls_before`, `strikes_before`, `outs_before`
- `pitcher_hand`, `batter_hand`

이 값들은 메인 투수 ID와 TrackMan 투수 ID를 익명 상태 분포로 매칭할 때만 사용된다. 투수의 카운트별·상대 손별·이닝 역할별 패턴으로는 아직 활용하지 않았다.

### 3.3 완전 미사용

| 컬럼 | 판단 | 활용안 |
|---|---|---|
| `trackman_id` | 행 식별자라 예측 가치 없음 | 계속 제외 |
| `game_date` | 유효, 결측 0% | 등판 간격, 월별 drift, 최근 시즌 후반 상태 |
| `trackman_game_id` | ID 자체는 제외, 그룹 키는 유효 | 투수별 경기 수, 경기당 투구 수, 선발/구원 역할 |
| `pitch_no` | 단독 값은 경기 전체 순번일 가능성 | 경기 내 상대적 투구 시점과 workload 보조치로만 사용 |
| `pitch_of_pa` | 유효, 결측 0%, 중앙값 3 | 타석 초반/후반 구사 패턴과 물리 변화 |
| `batter_trackman_id` | 직접 main ID 연결은 불안정 | 상대 다양성, 동일 타자 집중도, 손 유형 보조 집계 |
| `pitcher_team` | 시즌 이동으로 drift 위험 | crosswalk 품질 확인용, 모델 직접 입력은 보류 |
| `batter_team` | 시즌 이동으로 drift 위험 | 상대 다양성 보조 집계만 검토 |
| `tagged_pitch_type` | 17종, 결측 0% | 상세 레퍼토리 비율·entropy·희귀 구종 수 |
| `auto_pitch_type` | 11종, 결측 0.004% | tagged와 별도 레퍼토리 및 불일치율 |

TrackMan은 1,793,078행, 5,980경기, 투수 906명이다. 투수-시즌 중앙값은 551구·18경기라 500구 조건 안에서도 workload 집계가 가능하다. tagged와 auto의 정확 일치율은 55.07%에 불과해, 둘 중 하나를 정답처럼 택하기보다 두 분류의 비율과 불일치율을 품질·레퍼토리 피처로 쓰는 편이 안전하다.

## 4. 아직 만들지 않은 메인 데이터 상호작용

원본값은 모두 사용 중이지만 아래 표현은 현재 주력 트리에 직접 들어가지 않는다.

| 묶음 | 신규 피처 예시 | 예상 역할 |
|---|---|---|
| 카운트 압력 | `three_ball`, `two_strike`, `full_count`, `balls-strikes`, 투수 우세/열세 | 공격적 제구와 유인구 상황 분리 |
| 이닝·점수 | 이닝 bucket, 동점/1점차/대량차, `late_close` | R 문맥 잔차를 트리 내부에서 학습 |
| 주자 압력 | 득점권, 만루, 2아웃 득점권, force 상황 | 가운데·outside 실패 위험 분리 |
| 투수 관점 승률 | 투수 홈/원정 여부, 투수팀 win expectancy, 50%와의 거리 | 홈 기준 기대승률을 투수 관점으로 정렬 |
| 중요도 | `log1p(li)`, LI quantile/bucket, LI×late/close | 극단 LI의 비선형 효과 |
| 구종 구성 | pitchmix entropy, 최대 구종군 비율, fastball-breaking 차이 | 단순 세 비율보다 제구 스타일을 압축 |
| 표본 균형 | pitcher/batter log-count 차이·최솟값, 둘 중 cold-start 여부 | 신뢰도와 상성 불확실성 표현 |
| R/F 문맥 | `game_type × count_state × inning_bucket × hand matchup` | 현재 효과가 확인된 R lookup을 모델 피처로 일반화 |

최근 성공률 gap·momentum을 대량 추가하는 방식은 다시 사용하지 않는다. 기존 실험에서 `INSIGHT_GAP` BSS 687.28, `INSIGHT_MOMENTUM` 754.96으로 기준 763.54보다 낮았다. 필요하면 F 또는 failure expert 안에서 2~4개만 제한적으로 사용한다.

## 5. 만들어 두었지만 최종 모델에 채택하지 않은 피처

| 피처군 | 기존 결과 | 결정 |
|---|---:|---|
| TrackMan 구종군별 전체 259개 | Val2024 760.71 | 과대 확장 중단 |
| 구종군 compact mean | 764.94 | 기준 768.12보다 낮아 단독 채택 안 함 |
| fastball 전용 구종군 | 764.23 | 단독 채택 안 함 |
| TrackMan available/unavailable 별도 모델 | 678.85 | hard gating 중단 |
| hard pitcher cluster ID | 780.52 | seed·희소성 문제로 제외 |
| soft/style cluster blend | 약 799~801 | 신호는 있으나 이후 R 문맥 보정이 더 강함 |
| 공동 SVD 상성 | 815.08 | 다양성은 있으나 R 안정성 확인 전 보류 |
| success adjusted prior 2개 | 784.56 | 성능 개선되어 이미 현재 모델에 사용 |

핵심은 “미사용 피처를 전부 추가”하는 것이 아니다. 구종군별 259개처럼 차원을 크게 늘리면 과적합했다. 기존 신호를 10~40개의 해석 가능한 요약으로 압축해야 한다.

## 6. 신규 피처군 설계

### P0. `MAIN_CONTEXT_COMPACT` — 20~30개

- 카운트 압력 6~8개
- 이닝·점수·LI 상호작용 6~8개
- 득점권·주자 압력 4~6개
- 투수 관점 기대승률 2~4개
- R/F×카운트×손 상호작용 category 2~4개

비용이 가장 낮고, 이미 R 문맥 lookup에서 효과가 확인된 구조를 일반화한다.

### P0. `TM_WORKLOAD` — 20~35개

투수-시즌마다 다음을 계산한다.

- 경기 수, 경기당 투구 수 mean/std/median/p90/max
- 등판 간격 mean/std/최근값
- 등판당 이닝 범위, 초반 이닝 비율, 후반 이닝 비율
- 월별 투구량 slope와 시즌 후반/전체 비율
- `pitch_of_pa` 평균, 4구 이상 타석 비율
- 최신 시즌·recency 평균·시즌간 변동·결측 및 신뢰도

모든 cutoff `S`는 시즌 `<S`만 사용하고, 투수-시즌 500구 이상 규칙을 유지한다.

### P1. `TM_SITUATIONAL_STYLE` — 25~45개

- 좌/우 타자 상대 구종군 비율 차이
- 투수 우세/열세/풀카운트 구종군 비율 차이
- 초반/후반 이닝 구속·무브먼트·릴리스 편차
- 타석 1구째와 4구 이후의 구속·구종 entropy 차이

각 split은 raw mean 대신 투수 전체 평균 대비 residual로 만들고, cell 100구 미만은 shrink한다. 상황별 현재 투구 정보가 아니라 과거 TrackMan 스타일 요약만 사용한다.

### P1. `TM_REPERTOIRE_COMPACT` — 15~30개

- tagged/auto 각각의 entropy, 유효 구종 수, 최다 구종 비율
- tagged-auto 불일치율
- fastball 대비 breaking/offspeed 구속 차이
- 구종군 간 IVB/HB 거리와 릴리스 높이·좌우 차이
- arm-side 정규화 movement와 release-side
- 릴리스 좌표 공분산 또는 2차원 분산 크기

구종군별 원본 통계 259개를 그대로 넣지 않고 물리적 contrast만 사용한다.

### P2. `PLAYER_UNCERTAINTY_AND_MATCHUP` — 10~20개

- 투수·타자 표본 수의 최소값, 기하평균, 로그 차이
- 양쪽 success/middle posterior의 차이와 합의도
- pitcher cluster posterior entropy와 top-2 margin
- continuous embedding cosine, 동일/이종 유형 여부

과거 gap 피처의 실패를 고려해 전체 모델보다 F/middle/reverse expert에서 먼저 검증한다.

## 7. 검증 순서

| 순서 | 실험 | 비교 |
|---:|---|---|
| 1 | baseline + `MAIN_CONTEXT_COMPACT` | 현재 211개 기준 |
| 2 | baseline + `TM_WORKLOAD` | TrackMan available/unavailable 동시 확인 |
| 3 | baseline + `TM_REPERTOIRE_COMPACT` | 기존 pitchgroup 259개와 비교 |
| 4 | baseline + `TM_SITUATIONAL_STYLE` | R/F 및 failure 유형별 분해 |
| 5 | P0/P1 상위 피처의 2-way 조합 | 단일군 이득이 양 fold에서 확인된 경우만 |
| 6 | failure expert 및 F expert에 선택 피처 추가 | 전체 모델과 잔차 상관 확인 |

탐색은 Val2022·Val2023에서 선택하고 상위 후보만 Val2024에 게이트한다. Val2024에서는 TrackMan 2024를 절대 사용하지 않는다.

## 8. 채택 기준

- Val2023·Val2024 중 한 fold에서 큰 악화가 없어야 한다.
- 전체 BSS +2 이상 또는 기존 앙상블에 추가했을 때 +1 이상을 우선한다.
- TM 미보유군을 악화시키지 않아야 하며, TM 보유군 개선만 있는 경우 soft reliability interaction으로 제한한다.
- R/F, middle/reverse/outside, 좌우 손별 평균 오차를 함께 기록한다.
- 현재 제출 추론 21.2초이므로 신규 피처 계산을 포함해 180초 이내를 유지한다.

## 9. 제외·주의

- `row_id`, `trackman_id`는 계속 제외한다.
- 평가 `test.csv` 내부 행으로 선수별·팀별·월별 집계를 만들지 않는다.
- 현재 투구 구종이나 현재 투구 TrackMan 물리값은 사용할 수 없다.
- TrackMan 팀 ID를 모델 범주로 직접 넣는 것은 시즌 이동과 crosswalk 오류 위험 때문에 후순위다.
- batter TrackMan ID를 main batter ID에 억지로 1:1 연결하지 않는다. 먼저 상대 다양성 집계로만 사용한다.
- 상세 구종군 259개와 hard cluster ID는 재투입하지 않는다.

