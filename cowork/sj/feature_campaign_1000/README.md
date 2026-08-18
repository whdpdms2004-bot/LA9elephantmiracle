# Feature campaign toward BSS 1000

작성일: 2026-08-18 KST

## 기준점

- Public 최고: `submit_032`, BSS 979.
- 최신 미제출 후보: `submit_036`, strict forward-OOF F1 272피처의 단독 CatBoost.
- 목표: 외부 데이터, test 행간 통계, 미래 시즌 정보를 쓰지 않고 단독 ML 시스템으로 Public BSS 1000 이상.

## 피처 대분류

1. `direct`: 평가 행에 그대로 제공되는 47개 입력 열. `row_id`는 제출 순서 보존에만 쓰고 모델 입력에서 제외한다.
2. `row_derived`: 한 평가 행의 값과 학습 데이터에서 미리 고정한 상수/lookup만으로 계산하는 파생 피처.
3. `trackman_pitcher`: 예측 시즌보다 과거인 TrackMan 로그만으로 만든 투수 표현. main ID와 TrackMan ID의 raw 교집합이 없으므로 학습 데이터에서 만든 crosswalk가 필요하다.

상세 family와 채택 조건은 `feature_catalog.md`에 기록한다.

## 검증 원칙

- Val2022: 2019~2021 학습 -> 2022 검증.
- Val2023: 2019~2022 학습 -> 2023 검증.
- Val2024: 2019~2023 학습 -> 2024 검증.
- 학습 기반 lookup, target encoding, 임베딩, calibration은 fold마다 다시 fit한다.
- 평가 행 파생은 반드시 행 단위 함수여야 한다. test 전체의 평균, 순위, 빈도, rolling, lag를 쓰지 않는다.
- 후보 선별은 빠른 screen 뒤 full seedbag으로 두 fold를 재확인한다.
- 최종 채택은 두 fold에서 성분 단독 BSS와 production-base 결합 Delta BSS가 함께 개선되고, 개선폭이 잡음 바닥을 넘을 때만 한다.
- 리더보드 점수는 상수, 피처, 가중치 선택에 사용하지 않는다.

## 실험 실행 순서

1. `v75_feature_family_screen.py --stage screen`
   - production과 동일한 F행 가중치만 사용한다.
   - `submit_035`의 2차 피처가 최종 학습 조건에서도 재현되는지 먼저 확인한다.
   - 경기 맥락, 최근 폼, 확률 프로파일 기하, 명시적 상호작용 family를 빠르게 비교한다.
2. screen 상위 family를 `--stage confirm --arms ...`로 Val2023/Val2024 full seedbag 확인.
3. TrackMan raw 로그에서 strict-as-of release consistency 및 target-free pitcher embedding을 생성한다.
4. 채택 family만 합쳐 최종 재학습한다.

위 순서는 완료했다. 현재 재현 순서는 `train_final_f1_cat.py` ->
`report_f1_validation.py` -> `verify_final_f1.py` -> `build_f1_package.py`다.
공식 raw data부터 TrackMan cutoff, 행 피처 캐시, 실패 성분 학습 라벨, 최종 가중치를
한 번에 재생성하려면 `reproduce_final_f1.py`를 사용한다. 학습 환경 버전은
`requirements_training.txt`에 고정했다.

## 현재 확인된 위험

- `v71_second_order.py`는 짧은 등판 행 가중치 0.5를 썼지만 `v73_build035.py`의 최종 학습은 F행 가중치 0.2만 쓴다. 증분 효과를 production 조건에서 재확인해야 한다.
- 기존 2차 피처 이름 생성은 여러 `prev1/3/5` 차이를 같은 이름으로 덮어쓴다. 다만 원래 `component_features`에 해당 delta가 이미 있으므로 정보 누락이라기보다 중복 열 가중 효과에 가깝다. 새 실험은 이름 충돌을 금지한다.
- TrackMan 물리 피처는 기존 base에 이미 다수 포함되어 있다. 새 TrackMan 후보는 평균/표준편차 재투입보다 release 공분산, pitch-group 조건부 일관성, target-free 저차원 표현처럼 기존에 없는 정보만 시험한다.

## 2026-08-18 진행 결과

### 원본 47열 감사

`audit_direct_features.py`로 2019~2023에서만 단변량 lookup을 만들고 Val2024에 적용했다.
상수 예보 대비 상위 Delta BSS는 다음과 같다.

| 열 | Delta BSS |
|---|---:|
| `asof_pitcher_ball_rate` | +134.56 |
| `asof_pitcher_reverse_rate` | +114.14 |
| `asof_pitcher_success_rate` | +101.86 |
| `asof_pitcher_prev5_game_success_rate` | +83.35 |
| `asof_pitcher_offspeed_rate` | +57.75 |
| `asof_pitcher_prev3_game_success_rate` | +49.25 |
| `balls_before` | +25.58 |

원시 `pitcher_id`, `batter_id`, 단순 표본량, `game_type` lookup은 시즌 전이가 나빠 음수였다.
이는 ID를 그대로 외우는 방향보다 수축된 이력과 카운트 상호작용을 우선해야 한다는 근거다.

### 행 단위 파생 family 빠른 선별

Val2024, XGB+CatBoost 2시드, 250 rounds의 동일 실행 내 P0 대비 결과다.

| arm | family | Delta BSS vs P0 | 단독 BSS vs P0 |
|---|---|---:|---:|
| P1 | submit_035 2차항 | +4.53 | +12.03 |
| P2 | 경기 압력 | +4.40 | +5.59 |
| P6 | P1 + 경기 압력 | **+8.36** | **+12.66** |
| P7 | P1 + 최근 폼 곡률 | +5.08 | +14.68 |
| P8 | P1 + 프로파일 기하 | +4.79 | +11.88 |
| P9 | 전체 | +7.43 | +12.61 |

P6는 Val2023에서 결합 -1.08로 뒤집혀 기각했다. P1은 production 조건의
8시드·400 rounds에서 두 fold 모두 재현되어 최종 성분 후보로 승격했다.

| fold | P1 단독 BSS 변화 | production-base 결합 Delta BSS 변화 |
|---:|---:|---:|
| 2023 | +16.79 | +3.03 |
| 2024 | +5.15 | +3.47 |

### 두 fold 직접 신호와 상호작용 감사

Val2023/Val2024에서 모두 양수인 직접 신호는 투수 ball/reverse/success rate,
prev3/5 success, balls, win expectancy/LI, breaking/offspeed rate였다.
반대로 `game_type`은 두 fold 단변량 lookup에서 각각 -1214/-869 Delta BSS로 전이하지 않았다.

학습 시즌에서만 만든 10-bin 2차원 EB 진단에서는 다음 상호작용이 두 fold 모두
각 단변량보다 추가 신호를 보였다.

| 상호작용 | 최소 synergy BSS |
|---|---:|
| reverse rate × balls | +159.25 |
| ball rate × balls | +132.83 |
| reverse rate × strikes | +130.89 |
| prev5 success × balls | +125.72 |
| ball rate × strikes | +122.62 |
| pitcher hand × batter hand | +80.27 |

이 수치는 최종 모델 이득이 아니라 후보 순위용 진단이다. 실제 채택은 GBDT arm의
두 fold full confirm으로만 결정한다.

### 원시 season ablation

8 seeds, 400 rounds의 동일 조건에서 `season` 제거는 regime에 따라 부호가 뒤집혔다.

| fold | 단독 BSS 변화 | 결합 Delta BSS 변화 |
|---:|---:|---:|
| 2023 | +465.27 | +77.07 |
| 2024 | -301.32 | -30.99 |

따라서 시즌 제거는 기각한다. `season`을 0부터 다시 번호화한 arm은 원본과 사실상
같았으므로 숫자의 절대 크기도 원인이 아니다. 시즌 축은 유지하고 행 단위 상호작용과
TrackMan 표현으로 특정 regime 의존성을 낮춘다.

### TrackMan target-free 표현

`v76_trackman_release_embedding.py`가 cutoff 2023/2024/2025 산출물을 만들었다.

| cutoff | 사용 가능한 마지막 시즌 | TrackMan 투수 | main 매핑 투수 | PCA 12 설명분산 |
|---:|---:|---:|---:|---:|
| 2023 | 2022 | 433 | 269 | 65.05% |
| 2024 | 2023 | 491 | 295 | 64.60% |
| 2025 | 2024 | 559 | 336 | 64.57% |

Val2024 행 커버리지는 60.24%다. PCA12, release core, P1/경기 압력과의 결합을 다음 screen에 넣는다.

기존 tm500 72열로 5-fold Ridge 재구성한 결과 raw PCA12의 평균 R2는 0.692였다.
중복을 줄이기 위해 기존 tm500로 설명되는 부분을 target-free Ridge로 제거한 뒤 residual PCA12를 추가했다.
Residual PCA의 평균 CV R2는 -0.115이고 12차원 모두 0.5 미만이어서 기존 요약과 구별되는 표현임을 확인했다.
시점/키/유한값/행 독립성 계약은 residual PCA를 포함해 모두 통과했다.

### 단일 XGBoost 재현

기존 V2R200+TM500 209피처에 direct products/context 18열을 추가한 D0를
서로 다른 두 기존 XGBoost 설정에서 비교했다.

| 고정 설정 | Val2023 D0-B0 | Val2024 D0-B0 |
|---|---:|---:|
| local trial 93, screen 1000 trees | +16.37 | +12.09 |
| robust trial 136, full early stopping | +7.16 | +9.08 |

D0는 두 fold와 두 하이퍼파라미터에서 모두 양수여서 단일 ML 채택 후보로 올린다.
`game_type`과 raw 선수/팀 ID를 함께 제거한 arm은 Val2024 -56.33으로 기각했다.

Residual TrackMan을 D0에 더한 T0의 증분은 local 설정에서 +0.51/+8.08,
robust 설정에서 +45.27/-8.07이었다. 모델별·fold별 부호가 안정하지 않아 단독 채택은 보류한다.

별도 attention 피처는 Val2023에서 결합 +65.82였지만 Val2024에서 -46.16으로
완전히 뒤집혀 기각했다. 표현력이 큰 피처일수록 한 regime을 강하게 외우는 위험을 재확인했다.

### 계층 차감 피처 C1과 모델 독립 재현 — 탐색용 결과

`pitcher/batter x 상대손`, `pitcher x 상대손 x count/inning`을 fold 이전 Target으로만
집계하고, 바로 아래 계층의 EB 주효과를 뺀 45열을 만들었다. 여기에 D0 18열을 합친
`C1`을 탐색 후보로 만들었다.

| 모델 | Val2023 C1-B0 | Val2024 C1-B0 |
|---|---:|---:|
| XGBoost robust CPU | +29.52 | +11.46 |
| CatBoost robust GPU | +23.63 | +28.42 |

더 오래된 Val2022에서도 XGBoost C1-B0가 +66.75여서 세 validation fold에서 모두 양수다.
수축 강도 K를 100/600/1200으로 바꾸고, 투수·타자손과 count/inning K를 분리한 실험은
GPU Val2024에서 더 높기도 했지만 시드 및 Val2023 부호가 불안정했다.

사후 코드 감사에서 C1의 **검증행 lookup은 fold 이전 시즌만 사용하지만 학습행 lookup은
fold 학습 전체로 한 번 만들어져 자신의 시즌 Target이 섞임**을 확인했다. 검증 라벨 누수는
아니어도 최종 학습/추론 분포와 코드 검증 근거가 약하므로 C1은 제출 후보에서 제외했다.

### production-base 결합과 순방향 가중치 검증 — 탐색용

P1 성분 예측 뒤에 `logit(C1)-logit(B0)` 잔차를 더했다. 전 행 고정 0.15는 현행
P0 기준 +9.79/+7.93 BSS였다. 한 행의 `asof_pitcher_n`만 보는 고정 구간 가중치로
신뢰도를 나누면 다음과 같다.

| 선택 방식 | Val2022 | Val2023 | Val2024 |
|---|---:|---:|---:|
| 2022에서만 5구간 가중치 선택 | +13.31 | +11.14 | +5.05 |
| 2024에서 log8 가중치 선택, 2023 전이 확인 | - | +11.50 | +11.23 |

2022에서만 고른 값이 손대지 않은 두 미래 fold에서 모두 양수라 C1 잔차의 순방향
일반화 근거가 있다. 반면 P1 성분 비중까지 구간별로 공동 최적화한 +39.42/+11.85
결과는 단일 연도 선택 시 다른 연도가 악화되어 과적합으로 판정하고 채택하지 않는다.

이 결과는 C1 탐색 피처의 신호 확인용으로만 보존하고 제출에는 쓰지 않는다.

### strict forward-OOF F1과 최종 단독 CatBoost

F1은 C1과 같은 45개 계층 차감 피처를 학습행에도 엄격히 순방향으로 만든다.
2019년은 0 fallback, 2020~2024년 각 행은 해당 시즌보다 이전 시즌만 사용한다.
검증 시즌과 2025 추론 lookup도 각각 `< fold`, `2019~2024`로 고정했다.

| fold | raw BSS | 학습 Target 추세 보정 BSS |
|---:|---:|---:|
| 2022 | 2313.374 | 2301.262 |
| 2023 | 57.736 | 64.024 |
| 2024 | 782.456 | 796.734 |

- 보정 3-fold 단순 평균: **1054.006700**
- 2022~2024 연결 OOF BSS: **1179.566787**
- 최종 모델: enhanced 209 + D0 18 + F1 hierarchy 45 = **272피처** 단독 CatBoost
- 최종 2025 학습: 2019~2024 전체, GPU, 2595 iterations, seed 20262843
- 패키지: `cowork/sj/submit/2026-08-18/submit_036.zip`
- 행 독립성 최대 차이 0.0, 245,789행 benchmark 5.87초, 오프라인 smoke 통과

오프라인 평균은 목표를 넘었지만 Public 1000은 아직 제출로 확인하지 않았으며
기존 Public 최고 979에서 목표 도달을 주장하지 않는다.

### TrackMan 표현의 최종 판정

Residual PCA12는 기존 TrackMan 요약과 다른 정보이고 계약 검사는 통과했지만,
단일 XGBoost 증분이 설정에 따라 `+45.27/-8.07`로 뒤집혔다. P1+C1과의 결합에서도
두 fold 동시 개선 게이트를 넘지 못해 현재 최종 세트에는 포함하지 않는다.
