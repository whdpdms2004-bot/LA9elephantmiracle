# 피처 카탈로그

## A. Direct: test.csv에 그대로 있는 피처

| 세부 family | 열 | 처리 원칙 |
|---|---|---|
| 시점 | `season`, `game_month`, `game_dayofweek` | raw `season` 제거는 2023/2024 부호가 뒤집혀 기각. 미래 시즌 Target을 이용한 보정 금지 |
| 경기 상태 | `inning`, `top_bottom`, `game_type`, count, outs | 원본 유지 + 행 단위 상태 조합 |
| 점수/주자 | 득점, 점수차, runner flag, `base_state` | 결정적으로 중복인 열은 선형/거리 모델에서 정리 |
| 경기 중요도 | win expectancy, `li` | 비선형 중심성, 접전/후반 상호작용 후보 |
| 선수/팀 | pitcher/batter/team ID와 손 | ID 숫자 자체의 서열 의미는 없으므로 범주 처리 또는 시간 안전 lookup 사용 |
| 투수 누적 | `asof_pitcher_*` | 가장 강한 직접 신호. 표본량 기반 수축/신뢰도 필요 |
| 최근 경기 | prev1/3/5 success/middle | 추세, 가속도, 변동성, career 대비 변화 후보 |
| 타자 누적 | `asof_batter_*` | 투수와의 차이/곱 및 손 상성 후보 |
| 구종 구성 | pitchmix n, fastball/breaking/offspeed rate | entropy, concentration, count/손 상호작용 후보 |

`row_id`는 모델 피처에서 제외한다.

## B. Row-derived: 평가 행에서 파생 가능한 피처

| family | 예시 | 근거/가설 | 상태 |
|---|---|---|---|
| 기본 상태 | count state, hand matchup, runner-out state, late/high leverage | 이미 검증된 기준 피처 | 사용 중 |
| Bayesian reliability | smoothed rate, `n/(n+K)`, missing/cold-start flag | 소표본 극단값의 과신 완화 | 사용 중 |
| 최근 폼 | career 대비 prev1/3/5 delta, trend/std | 최근 컨디션과 장기 기량 분리 | 사용 중 |
| 명시적 2차항 | 투수x타자 성공률, 플래툰xcount, ratex표본량 | 축 정렬 트리가 곱을 비효율적으로 근사 | P1 통과: +3.03/+3.47 결합 BSS |
| 경기 압력 | count margin, full count, close/late game, LIx접전 | 같은 투수 기량도 제약 조건에 따라 제구 의사결정이 달라질 수 있음 | V75 |
| 폼 곡률 | prev1-prev3, prev3-prev5, acceleration, abs shock | 단순 trend가 놓치는 방향 전환과 변동폭 | V75 |
| 프로파일 기하 | 실패 구성 entropy/concentration, pitch mix entropy | 평균이 같은 투수도 실패/구종 구성의 집중도가 다름 | V75 |
| 상황 상호작용 | reverse/ball/success/prev5 × balls/strikes, pitchmix×count | 두 fold 2D EB 감사에서 추가 신호. GBDT screen 대기 | V75 |
| 학습 lookup C1 | pitcher/batter x 상대손, pitcher x 상대손 x count/inning 계층 차감 EB | validation은 fold 이전이지만 학습행은 fold 학습 전체 lookup | **탐색 전용**: 신호는 재현됐으나 자기 시즌 Target 포함으로 제출 제외 |
| 학습 lookup F1 | C1과 같은 45열을 각 학습행 시즌보다 이전 시즌만으로 생성 | 학습·검증·2025 추론의 시간 규칙을 동일하게 맞춤 | **최종 채택**: Cat 보정 3-fold 평균 BSS 1054.01 |
| 잔차 신뢰도 | `asof_pitcher_n` 고정 구간별 `logit(C1)-logit(B0)` 강도 | 저표본과 베테랑에서 새 계층 피처 신뢰도가 다름 | 탐색 결과 보존, C1 의존으로 최종 제외 |

## C. TrackMan pitcher representation

모든 표현은 예측 시즌 `S`에 대해 `trackman.season < S`만 사용한다.

| family | 표현 | 새 정보 | 우선순위 |
|---|---|---|---:|
| 기존 요약 | 물리량 mean/std, 최근값, 시즌간 변화, pitch-group rate | 이미 production base에 다수 존재 | 기준/중복 감사 |
| release consistency | `(rel_height, rel_side, extension)` 공분산의 trace/determinant/eigen-ratio, robust dispersion | 단변량 std가 놓치는 다변량 릴리스 타이트함 | 1 |
| pitch-group consistency | fastball/breaking/offspeed별 release·velocity·movement dispersion과 그룹 간 separation | 서로 다른 목표/구종을 한 분산으로 섞는 문제 완화 | 1 |
| mechanics coupling | velocity-release, velocity-movement, release-movement correlation | 릴리스 변화가 구속/무브먼트와 함께 움직이는 패턴 | 2 |
| target-free embedding | 위 통계 벡터의 fold-fit PCA/SVD, 결측은 학습 통계로 처리 | 투수 ID 대신 물리 스타일의 저차원 좌표 | 1 |
| similar-pitcher prior | embedding 거리 기반 이웃의 과거 Target residual, 강한 수축 | 저표본 main 투수의 fallback 보강 | 3, 반드시 순방향 OOF |

Residual PCA12는 새 정보 계약은 통과했지만 Val2023/Val2024 증분 부호가 안정하지 않아
현 최종 후보에는 넣지 않는다. TrackMan raw/current-pitch 실제 측정값은 어떤 경우에도 쓰지 않는다.

raw `pitcher_trackman_id` 자체를 main `pitcher_id`처럼 사용하지 않는다. crosswalk 신뢰도와 가용성은 별도 피처로 둔다.
