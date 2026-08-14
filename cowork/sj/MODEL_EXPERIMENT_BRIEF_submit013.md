# submit_013 기준 모델 실험 정리

> 팀원 공유용. 기준일 2026-08-12. 원본 근거는 문서 맨 아래 "원본 파일" 참고.

---

## 0. 세 줄 요약

1. **제출본 `submit_013`이 현재 유일하게 Public LB로 검증된 최고점이다. Public LB `895.404000081`, 내부 Val2024 BSS `812.7040`.**
2. 점수를 올린 것은 "모델을 더 크게/더 좋게" 만든 게 아니라, **① 성공률 누적치를 직전 시즌 실제값에 맞춰 재보정한 피처 → ② 시간 순방향 OOF 앙상블 → ③ 투수유형×타자유형 상성 residual 보정** 이 3단 적층이다.
3. 특히 **"포수 요구 반대 방향(reverse) 실패를 성공률과 섞지 않고 독립 잔차로 따로 모델링한 것"** 이 단일 최대 상승 요인이다 (806.49 → 810.10 → 812.70).

---

## 1. 먼저 맞춰야 할 용어와 규칙

### 1-1. 점수 지표

| 지표 | 정의 | 방향 |
|---|---|---|
| `Brier` | 확률 예측의 평균제곱오차 | 낮을수록 좋음 |
| `normalized Brier` | `Brier / 0.25` (=상수 0.5 예측 대비 비율) | 낮을수록 좋음 |
| `BSS` | 대회 점수. normalized Brier가 1보다 낮은 만큼 커짐 | **높을수록 좋음** |
| `ΔBrier` | 기준 모델 대비 Brier 변화 | **음수일수록 개선** |

BSS는 소수 4째 자리 Brier 차이가 수백 점으로 증폭되는 지표다. 그래서 아래 표에서 `ΔBrier -0.00003`처럼 작아 보이는 값이 BSS 수 점~수십 점을 만든다. **순위력(AUC)보다 예측 평균의 연도별 편향(calibration)이 점수를 더 많이 좌우한다.**

### 1-2. 검증 프로토콜 (전 실험 공통)

| 표기 | 학습 | 검증 |
|---|---|---|
| `Val2023` (=F23) | 2022년 이하 | 2023년 |
| `Val2024` (=F24) | 2023년 이하 | 2024년 |
| 최종 제출 | 2019~2024 전체 재학습 | 2025 test 추론 |

- **랜덤 분할 금지.** 성공률이 2019년 56.47% → 2024년 48.61%로 계속 하락하는 시계열이라 무작위 분할은 낙관 편향이 생긴다.
- **TrackMan 규칙**: 검증연도보다 **이전 시즌만** 사용, **투수-시즌 500구 이상**만 개별 표현 생성. (2024 검증행 TrackMan 가용률 60.24%)
- 최종 2025 lookup만 2024 TrackMan까지 사용. **test 전체를 재집계·재군집하는 transductive 방식은 전부 기각.**
- 최근 시즌 가중: 각 fold의 예측 시즌 기준 지수형 recency weight, `half_life`는 Optuna 탐색 대상.

### 1-3. 왜 내부 812점인데 LB는 895점인가

내부는 2024 검증, LB는 2025 test다. 데이터가 다르므로 절대값을 직접 비교하면 안 된다. 지금까지 관측된 관계는 **내부와 LB가 같은 방향으로 움직인다**는 것뿐이다 (013: 내부 +2.4 상승 → LB +22.3 상승). 반대 사례도 있었다 — 001 CatBoost는 내부에서 003 XGBoost보다 5.44점 높았으나 LB에서는 34.58점 낮았다. **그래서 내부 1점 차이로 제출본을 바꾸지 않는다.**

---

## 2. 실험 지도 — 무엇을 어떤 순서로 시도했나

Val2024 BSS 기준 전체 추이. **`submit_013`은 8단계까지의 누적 결과다.**

| # | 방법 | Val2024 BSS | Val2023 ΔBrier | Val2024 ΔBrier | 판정 |
|---:|---|---:|---:|---:|---|
| 1 | XGBoost V2R200 + strict TM500 (단일 기준선) | 774.484 | — | — | 기준선 |
| 2 | 성공률 사전값·신뢰도 보정 피처 (insight) | 784.557 | — | — | 단일 모델 최고 |
| 3 | 시간 순방향 OOF 앙상블 (V1 14모델 풀) | 789.673 | — | — | 앙상블 기준선 |
| 4 | performance 앙상블 + insight blend | 801.147 | — | — | `submit_007` |
| 5 | 투수·타자 **성공률** 상성 Ridge 보정 | 806.488 | -0.996e-5 | -0.411e-5 | `submit_010` |
| 6 | **reverse** 실패 독립 보정 추가 | 810.098 | -0.922e-5 | -3.075e-5 | `submit_011` |
| 7 | reverse 전용 타자 군집 | 810.257 | -1.046e-5 | -3.030e-5 | `submit_012` |
| 8 | reverse 군집 **3-seed correction 평균** | **812.704** | **-1.436e-5** | **-3.676e-5** | **`submit_013` → LB 895.404** |
| 9 | 대형 XGB expert 3-way 혼합 | 813.432 | -1.453e-5 | -4.443e-5 | `submit_014`, 미제출 |
| 10 | 공동 SVD 임베딩·군집 (안정형) | 814.291 | -3.261e-5 | -3.334e-5 | 미제출 |
| 11 | 공동 SVD 임베딩·군집 (공격형) | 815.083 | -0.758e-5 | -2.443e-5 | 미제출 |

읽는 법: **1→4는 "모델·피처·앙상블" 구간이고, 5→8은 "이미 만든 확률 위에 얹는 보정층" 구간이다.** 후자의 상승폭(+11.6)이 전자 마지막 단계의 상승폭보다 크다.

---

## 3. 모델 계열별 실험과 세팅

### 3-1. 1세대 — V1 63피처 단일 모델 (LB 검증된 유일한 기준선)

TrackMan·투수 임베딩 미사용, 메인 train의 경기 상황 + 선수/팀 ID + 투구 직전 `asof_*` 이력 + row-local 파생만 사용. 누수 없는 63개 피처.

| 제출 | 모델 | Optuna | 선택 trial | Val2023 BSS | Val2024 BSS | **Public LB** |
|---|---|---:|---:|---:|---:|---:|
| `submit_001` | CatBoostClassifier | 140 trial | 71 | 6.71 | 750.571 | **838.4920422492** |
| `submit_003` | XGBoost native Booster | 140 trial | 24 | 0.00 | 745.131 | **873.0751046509** |
| `submit_004` | XGB 7개 + Cat 6개 OOF 앙상블 | — | — | 54.01 | **789.673** | 미제출 |

**하이퍼파라미터**

| | CatBoost trial 71 | XGBoost trial 24 |
|---|---|---|
| half_life | 1.7324464557 | 0.4731162635 |
| 최종 트리 수 | 152 | 952 |
| 깊이/구조 | depth 7 | lossguide, max_depth 6, max_leaves 27 |
| learning_rate | 0.0521742 | 0.00619882 |
| 정규화 | L2 2.3071, Bayesian bootstrap, border_count 128 | gamma 0.56675, alpha 1.01823, **lambda 224.595** |
| min_child_weight | — | 617.1884 |
| subsample / colsample | — | 0.96959 / bytree 0.60405, bylevel 0.96186 |
| max_bin | — | 512 |

**여기서 얻은 교훈 2개 (지금까지 유지되는 원칙)**

- `half_life 0.473`처럼 **최근 시즌을 매우 강하게 반영한 설정이 2025 분포에 더 잘 맞았다.** 이후 모든 study의 half_life 탐색 범위를 짧게(0.25~0.90) 잡은 근거.
- 제출 002는 `XGBClassifier.load_model/predict_proba` + 평가서버 `scikit-learn==1.8.0` 조합에서 `TypeError: _estimator_type undefined`로 채점 실패했다. **이후 모든 XGBoost 추론은 `Booster.load_model` + `DMatrix` 네이티브 경로만 쓴다.**

### 3-2. 2세대 — V2 피처 + strict TrackMan (3계열 동시 재탐색)

동일 피처 세트 `V2R200_TM500_ALL`로 3개 계열을 독립 Optuna 탐색했다. 목적함수는 Val2023·Val2024 normalized Brier 가중값 + 최악 fold penalty.

| 계열 | study | trial 수 | best trial | Val2023 BSS | Val2024 BSS | 판정 |
|---|---|---:|---:|---:|---:|---|
| XGBoost | `xgboost_v2r200_tm500_robust` | 160 | 136 | 0.00 | 747.298 | 다양성 후보 |
| CatBoost | `catboost_v2r200_tm500_robust` | 80 | 69 | 47.59 | 744.628 | **Val2023 안정성 최고** |
| LightGBM | `lightgbm_v2r200_tm500_robust` | 60 | 55 | 0.00 | 697.484 | **기각** |
| XGBoost (2024 특화) | `xgboost_v2r200_tm500_local_2024` | 100 | **93** | — | **774.484** | **이후 모든 실험의 anchor** |

**계열별 best params**

| | XGB t136 (robust) | CatBoost t69 | LightGBM t55 |
|---|---|---|---|
| grow / depth | depthwise, max_depth 4 | depth 9 | max_depth 10, num_leaves 379 |
| n_estimators | 4803 (조기중단 266) | 4487 (조기중단 1838) | 1466 (조기중단 137) |
| learning_rate | 0.03911943 | 0.00782040 | 0.04762026 |
| 정규화 | gamma 8.6606, alpha 27.0208, lambda 1.1746 | l2_leaf_reg 124.8867, random_strength 0.00043758 | alpha 1.6177, lambda 6.8e-06, min_split_gain 1.9922 |
| min_child | min_child_weight 15.8862 | — | min_child_samples 4997 |
| subsample / colsample | 0.882197 / 0.614718, 0.774361 | Bayesian, bagging_temp 2.7358 | 0.619426 (freq 3) / 0.573382 |
| max_bin | 128 | border_count 254 | 63 |
| half_life | 0.340786 | 1.674602 | 0.975069 |

> LightGBM은 Val2024가 697점으로 두 계열보다 47~50점 낮아 이후 파이프라인에서 제외했다. **현재 사용 계열은 XGBoost + CatBoost 2개다.**

### 3-3. anchor 모델 — `xgboost_v2r200_tm500_local_2024` trial 93

`submit_007`~`submit_013`의 **본체 단일 모델**이다. 이 트라이얼 하나의 파라미터가 이후 모든 실험의 고정 기준이 된다.

```
study         : xgboost_v2r200_tm500_local_2024  (100 trials)
best_trial    : 93   (normalized Brier 0.9922551627)
tree_method   : hist
grow_policy   : lossguide  (study 고정)
objective     : binary:logistic / eval_metric logloss / early_stopping_rounds 220

half_life          : 0.5259510183
n_estimators       : 4431        (조기중단: V2피처 2468 / insight피처 2641)
learning_rate      : 0.0030626838
max_depth          : 7
max_leaves         : 18
min_child_weight   : 626.0593078
subsample          : 0.9736716317
colsample_bytree   : 0.7907533297
colsample_bylevel  : 0.8524268909
gamma              : 0.1155162057
reg_alpha          : 0.2421771226
reg_lambda         : 505.8374114
max_bin            : 512
```

**중요**: insight 피처로 바꾼 뒤 Optuna를 25회 더 돌렸지만(`xgboost_insight_success_local_2024`) **여전히 trial 93 파라미터가 최고였다. 즉 지금 병목은 하이퍼파라미터가 아니라 피처와 보정층이다.** 팀원이 HPO에 시간 쓰는 것보다 피처·보정 쪽이 기대수익이 크다.

### 3-4. insight 피처 실험 — 무엇이 774 → 784를 만들었나

가설: "누적 성공률(`asof_rate`)은 베테랑일수록 과거 편향이 크다. `asof_pitcher_n > 1000`에서 실제보다 평균 **1.49%p 높았다.**" → 직전 시즌 실제값으로 재보정한다.

핵심 공식 두 개:

```
gap_component  = actual_rate_(S-1) − mean_asof_rate_(S-1)
adjusted_rate  = sigmoid( logit(asof_rate) + logit_shift_component )
smoothed       = (rate × n + prior × k) / (n + k)
```

모든 피처 버전을 **trial 93 고정** 상태로 ablation했다 (동일 파라미터·동일 fold, 피처만 교체).

| feature_version | 피처 수 | Val2024 BSS | 기준선 대비 | 판정 |
|---|---:|---:|---:|---|
| `INSIGHT_BASE` (=V2R200_TM500) | 209 | 774.484 | 0.00 | 기준선 |
| **`INSIGHT_SUCCESS_ADJUSTED`** | **211** | **784.557** | **+10.07** | **채택 → submit_007/013 본체** |
| `INSIGHT_PRIOR_SUCCESS` | 221 | 783.523 | +9.04 | 채택 → submit_008 |
| `INSIGHT_SUCCESS_LAST` | — | 780.818 | +6.33 | 하위 |
| `INSIGHT_PRIOR_LAST` | — | 779.730 | +5.25 | 하위 |
| `INSIGHT_PRIOR` (동적 prior 42개) | 251 | 778.507 | +4.02 | 불필요 피처 과다 |
| `INSIGHT_PRIOR_REVERSE` (역방향 단독) | — | 774.644 | +0.16 | **단순 시즌 prior로는 효과 없음** |
| `INSIGHT_PRIOR_MIDDLE` (가운데 단독) | — | 768.806 | -5.68 | 기각 |
| `INSIGHT_GAP` | 223 | 687.280 | -87.20 | 기각 |
| `INSIGHT_ALL` (전부 투입) | 292 | 679.716 | -94.77 | **기각. 많이 넣으면 망한다** |

**해석**: 개선의 원천은 딱 **2개 피처**(`INSIGHT_SUCCESS_ADJUSTED`)다. 211 - 209 = 2. 피처를 292개까지 늘린 `INSIGHT_ALL`은 기준선보다 95점 낮다. **피처 추가는 반드시 ablation으로 검증한다.**

또한 `INSIGHT_PRIOR_REVERSE`가 +0.16에 그쳤다는 점이 중요하다 — **reverse는 "피처"로 넣으면 효과가 없고, "독립 residual 보정층"으로 넣어야 효과가 있다** (3-6 참조). 이것이 011~013의 설계 근거다.

같은 피처를 CatBoost로도 확인했다: `INSIGHT_BASE` t39 = 781.779 / `INSIGHT_PRIOR` t39 = 766.533. **CatBoost는 insight 피처로 오히려 하락하므로 CatBoost에는 적용하지 않는다.**

### 3-5. 앙상블 — performance / robust 두 트랙

단일 모델 교체가 아니라 **잔차가 다른 모델을 시간 OOF로 혼합**하는 구조다.

- 후보 풀: 39개 (V1 14개 + enhanced 계열)
- 혼합 공간: **logit space**
- 가중치: 비음수·합 1 제약 + L2 penalty
- 상관 제한 / top_k로 유사 모델 중복 방지
- 확률 보정 비교: none / logit-shift / Platt / Beta / isotonic / cohort_count_logit
- **선택은 2023 OOF로 적합 → 2024로 검증, 배포는 가장 최근 2024 OOF로 재적합**

| 트랙 | L2 | top_k | 상관 상한 | calibration | Val2024 BSS |
|---|---:|---:|---:|---|---:|
| performance | 1e-4 | 8 | 0.98 | **beta** (coef 1.137875 / -1.122039, intercept -0.019686) | **780.318** |
| robust | 1e-2 | 5 | 0.999 | **cohort_count_logit** (global offset -0.033782, 12개 그룹 offset, ridge 0.001) | 760.560 |

배포 시 두 트랙 모두 동일한 2개 모델로 수렴했다: `enh__cat_robust_t62_seedbag3` + `enh__xgb_recent_t35_seedbag3`
- performance 가중치 **0.516440 / 0.483560**
- robust 가중치 0.500318 / 0.499682

`submit_013`이 쓰는 것은 **performance 트랙**이다.

### 3-6. 보정층 — 여기가 806 → 813의 전부

이 층의 원리: **XGBoost에 새 피처를 넣는 게 아니라, 이미 나온 확률에 작은 residual을 더한다.**

왜 그래야 하는가 — 같은 정보를 피처로 직접 넣었을 때의 점수:

| 투입 방식 | Val2024 BSS |
|---|---:|
| matchup 피처를 XGBoost에 직접 추가 | 748.96 ~ 761.01 |
| hard cluster ID + style 직접 투입 | 780.52 |
| soft 확률 + style 직접 투입 | 781.31 |
| centroid style만 직접 투입 | 784.21 |
| **(비교) 아무것도 안 넣은 기준선** | **784.56** |
| **residual 보정층으로 얹기** | **806 ~ 815** |

즉 **군집·상성 정보는 피처로 넣으면 전부 손해이고, 상성 통계의 smoothing 계층으로 쓸 때만 이득이다.** 이게 이 프로젝트에서 가장 반복 검증된 인사이트다.

**탐색 규모** (이 층에만 들어간 실험 수)

| 대상 | 탐색한 구조 수 |
|---|---|
| 투수 군집 | 56개 구조 × seed 5 (표현 physical/physical+control × PCA 8/16 × KMeans/GMM × 좌우 K 7종 × cutoff 3종) |
| 성공 타자 군집 | K 4종 × smoothing 2종 × 반감기 3종 |
| reverse 전용 타자 군집 | 60개 구조, Ridge 포함 **180개 검증** |
| reverse seed 안정화 | seed 5개의 **모든 부분집합 31개** × Ridge alpha 3종 |
| (이후) 멀티뷰 / 공동 SVD | 48 + 54 + 108개 |

**투수 군집 선정 결과** — intrinsic 안정성 1위(physical PCA8 KMeans, seed ARI 0.9023)와 downstream 1위가 달랐다. downstream을 채택했다.

| 투수 군집 | 단독 BSS | performance 혼합 BSS |
|---|---:|---:|
| **physical+control PCA8 + diagonal GMM (좌2/우4)** | **783.712** | **800.824** |
| physical PCA8 + KMeans (좌2/우4) | 783.015 | 800.385 |
| physical+control + KMeans (좌2/우4) | 783.020 | 799.808 |
| physical + GMM (좌2/우4) | 781.935 | 799.254 |

**reverse 독립 보정의 핵심 아이디어**: reverse 발생률에서 `시즌 × 투수손 × 타자손 × 볼카운트` 평균을 **먼저 제거**하고, 남은 잔차만 투수유형×타자유형으로 집계한다. 이렇게 하면 상황 효과와 상성 효과가 섞이지 않는다.

**seed 평균을 넣은 이유**: KMeans 초기값에 따라 correction이 흔들렸다. 단일 seed 17이 robust 목적함수는 근소하게 좋았지만, **seed 17/2026/4099의 correction을 평균한 쪽이 2024 혼합 성능이 더 높았고 초기값 의존성도 낮았다.** 그래서 013을 최종안으로 했다.

---

## 4. `submit_013` 최종 구조 (한 장 요약)

```
[입력] 메인 train (2019~2024, 1,475,092행) + TrackMan (투수-시즌 500구↑, 2024까지)
   │
   ├─(A) performance 앙상블 ─────────────── 가중치 0.3915
   │      enh__cat_robust_t62_seedbag3   0.516440
   │      enh__xgb_recent_t35_seedbag3   0.483560
   │      logit space 혼합 + beta calibration
   │
   └─(B) corrected insight 모델 ────────── 가중치 0.6085
          XGBoost trial 93 / INSIGHT_SUCCESS_ADJUSTED (211열)
             ↓  p_base
          + 성공 상성 correction × 0.25
             투수 GMM(좌2/우4) × 타자 KMeans(좌3/우4)
             smoothing 1000, half-life 1.0, Ridge alpha 10
          + reverse 상성 correction × 0.55
             투수 GMM(좌2/우4) × reverse 전용 타자 KMeans(좌4/우6)
             center: season × pitcher_hand × batter_hand × count_state
             smoothing 1000, half-life 1.0, Ridge alpha 1000
             seed 17 / 2026 / 4099 correction 평균
             ↓  단독 Val2024 BSS 799.2716
   │
[출력] 최종 확률  →  Val2024 BSS 812.7040  /  Public LB 895.404000081
```

### 컴포넌트별 세팅 (재현용)

| 컴포넌트 | 설정 |
|---|---|
| 투수 프로필 | TrackMan 물리 + 과거 제구 성향, cutoff별 2020~2025 생성, **286개 열** |
| 투수 군집 | 좌/우 분리, combined(구위+제구) **PCA 8차원**, **diagonal GMM**, 좌 2 / 우 4 |
| 성공 타자 군집 | 타석 손 방향 분리 **KMeans**, 좌 3 / 우 4 (스위치 타자는 실제 타석 손 방향별 분리) |
| reverse 타자 군집 | 손 방향 분리 **KMeans**, 좌 4 / 우 6, seed 17·2026·4099 |
| 성공 상성 | smoothing **1000**, half-life **1.0년**, Ridge **alpha 10**, correction scale **0.25** |
| reverse 상성 | smoothing **1000**, half-life **1.0년**, Ridge **alpha 1000**, correction scale **0.55** |
| reverse centering | `season × pitcher_hand × batter_hand × count_state` 평균 제거 후 집계 |
| 외부 혼합 가중치 | corrected insight **0.6085** : performance 앙상블 0.3915 |
| 2025 frozen artifact | pitcher lookup 792행 / batter-hand lookup 862행 / pair table **84셀** + Ridge imputer·scaler·coef JSON |
| 커버리지 | 2024 TrackMan 500구 프로필 57.34%, 투수유형×타자유형 pair 72.78% |
| 누수 방지 | 2024 검증 = 2023 이하만 / 2025 산출물 = 2024 이하만. test 재집계·재군집 없음 |

### 제출 검증 (전 제출본 공통 체크리스트)

- ZIP 최상위 구조 `model/`, `script.py`, `requirements.txt`
- `script.py` 위치 기준 상대경로 (`model/`, `data/`, `output/`)
- ZIP CRC + 모델 파일 존재 검사
- **245,789행 전체 로컬 추론: 16.48초** (10분 제한 대비 충분)
- 확률 유효 범위 검사
- 파일명 14자 (30자 제한 통과)
- 정방향 batch / 역순 batch / 1행씩 실행 결과 완전 동일 (batch 의존성 없음 확인)
- 크기 109,707,177 bytes / SHA256 `AB0CE11E5BD2B71E7FB570B05A860C83CC663C7BCCAEFC73F1F20D97E065DE92`

---

## 5. 어떻게 비교했나 — 채택 규칙

점수만 보고 고르지 않았다. 실제 사용한 채택 기준은 아래 순서다.

| 순위 | 기준 | 이유 |
|---:|---|---|
| 1 | **Val2023 · Val2024 ΔBrier가 둘 다 음수** | Val2024 단일 최대화는 연도 특화 위험. 방향 일치가 전이 가능성의 최소 조건 |
| 2 | Val2024 BSS | 대회 점수 프록시 |
| 3 | Val2024 예측 평균 오차 절댓값 | calibration 편향이 BSS 손실의 주원인 |
| 4 | Val2023 normalized Brier | 안정성 |
| 5 | 기존 모델과 예측 상관 < 0.98~0.985 | 앙상블 다양성. 상관 높으면 가중치가 0으로 수렴해 무의미 |
| 6 | 모델 수 / 추론 시간 | 10분 추론 제한 |

**피처 단계 채택 조건** (Phase A 기준): Val2024 BSS +3 이상, 또는 +1 이상이면서 Val2023 normalized Brier 개선, 또는 단독 개선은 작아도 예측 상관 <0.985이고 앙상블 BSS +1 이상.

**나중에 추가된 조건 (중요)**: 전체 ΔBrier만 보면 안 된다. R/F를 분리하니 013의 correction은 **Val2023 R에서 ΔBrier +0.000035로 악화**였다 (F가 -0.000438로 크게 개선해서 전체가 가려졌다). 그래서 이후 후보는 **R의 Val2023·Val2024가 둘 다 개선되는지 별도 확인**한다.

---

## 6. 점수 총정리 — `submit_007` ~ `submit_013`

| 파일 | 설계 | 성공 scale | reverse scale | 혼합 가중치 | 단독 BSS | **혼합 Val2024 BSS** | Val2023 ΔBrier | Val2024 ΔBrier | 추론 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `submit_007` | adjusted 2피처 + performance 앙상블 | — | — | 0.5284 | 784.557 | 801.147 | — | — | 16.35초 |
| `submit_008` | prior_success 12피처 + robust 앙상블 | — | — | 0.6090 | 783.523 | 799.630 | — | — | 16.01초 |
| `submit_009` | 성공 상성 보수형 | 0.60 | — | 0.5580 | — | 805.564 | -1.216e-5 | -1.281e-5 | 15.90초 |
| `submit_010` | 성공 상성 공격형 | 1.00 | — | 0.5320 | 786.275 | 806.488 | -0.996e-5 | -0.411e-5 | 15.87초 |
| `submit_011` | 성공 + reverse 독립 (기존 타자군집 재사용) | 0.10 | 0.55 | 0.6000 | 796.867 | 810.098 | -0.922e-5 | -3.075e-5 | 16.09초 |
| `submit_012` | reverse 전용 타자 군집 | 0.25 | 0.65 | 0.5980 | 796.685 | 810.257 | -1.046e-5 | -3.030e-5 | 16.14초 |
| **`submit_013`** | **reverse 군집 3-seed 평균** | **0.25** | **0.55** | **0.6085** | **799.272** | **812.704** | **-1.436e-5** | **-3.676e-5** | 16.48초 |

**Public LB 결과**

| 파일 | Public LB | 이전 최고 대비 |
|---|---:|---:|
| `submit_001` (CatBoost V1) | 838.4920422492 | — |
| `submit_003` (XGBoost V1) | 873.0751046509 | +34.5830624017 |
| **`submit_013`** | **895.404000081** | **+22.3288954301** |

리더보드 1위 참고 점수 약 1100. 현재 차이 약 205.

---

## 7. `submit_013`의 실체 — R/F 분해

`game_type=R`이 2024 데이터의 **88.16%**, `F`가 11.84%다.

| 2024 모델 | 전체 BSS | R BSS | F BSS | R 평균오차 | F 평균오차 |
|---|---:|---:|---:|---:|---:|
| adjusted 단일 | 784.557 | 784.625 | 457.969 | +0.443%p | +1.015%p |
| `submit_007` | 801.147 | 806.073 | 438.213 | +0.437%p | +1.012%p |
| **`submit_013`** | **812.704** | **812.655** | **487.087** | **+0.224%p** | **+0.591%p** |

`submit_007 → submit_013` 개선의 출처:

| 집단 | 표본 비중 | 집단 ΔBrier | **전체 개선 기여율** |
|---|---:|---:|---:|
| R | 88.16% | -0.000016 | **50.23%** |
| F | 11.84% | -0.000121 | **49.77%** |

**즉 013은 R에서도 개선됐지만, F는 행당 개선폭이 R보다 약 7.5배 커서 11.84%의 표본으로 전체 상승의 절반을 만들었다.** F는 2022→2023에 성공률이 70.87% → 47.29%로 급락한 구조적 단절이 있는 구간이라, 이 절반은 상대적으로 불안정한 지분이다. 그래서 이후 개선은 **R 쪽에서 찾는 것이 우선**이다.

R에 아직 남아 있는 오차:

| 구간 | 행 수 | 실제 | 예측 | 오차 |
|---|---:|---:|---:|---:|
| count 3-0 | 2,736 | 48.76% | 46.96% | -1.80%p |
| count 3-2 | 10,403 | 45.91% | 47.20% | +1.29%p |
| count 2-1 | 11,832 | 47.78% | 48.91% | +1.13%p |
| count 0-1 | 27,768 | 50.88% | 49.76% | -1.12%p |
| 만루 | 7,215 | 47.60% | 48.63% | +1.04%p |
| 좌투수-좌타자 | 28,390 | 45.81% | 46.37% | +0.56%p |

또한 R의 최근 악화는 reverse보다 middle 쪽이다: 2023→2024 R 성공률 -1.34%p인데 **reverse는 -0.80%p 감소, middle은 +2.08%p 증가**. reverse를 더 밀기보다 middle·볼카운트를 봐야 한다는 근거.

---

## 8. 기각된 접근 — 다시 시도하지 말 것

| 접근 | 결과 | 왜 실패했나 |
|---|---|---|
| 군집/상성을 XGBoost 피처로 직접 투입 | BSS 748.96~761.01 | 트리가 희소 조합에 과적합. 보정층으로만 유효 |
| hard cluster ID 투입 | 780.52 | 범주 ID 자체는 정보가 아님 |
| `INSIGHT_ALL` (292 피처 전부) | 679.72 | 피처 과다. **-94.77** |
| `INSIGHT_GAP` | 687.28 | 예측 평균 편향 +0.0109로 악화 |
| 전체 temporal target encoding | 391.08 | 최근 분포 변화에 대규모 과적합 |
| 단일 모델 거대화 32→128 leaves | 780.80 → 726.37 | 명확한 과적합. 20~24 leaves가 상한 |
| 1024-bin 64-leaf | 758.76 | 동일 |
| 실패유형 다중분류로 success 대체 | softmax 732.43 | 주 모델 대체 불가. auxiliary/expert로만 |
| TrackMan 가용/미가용 별도 모델 | 하락 | 표본 분할 + 가용 투수의 경력 선택 편향 |
| TrackMan 구종별 세부 확장 | 760.71~764.94 (요약형 768.12~769.69) | 저차원 안정 요약이 우수 |
| 타자 GMM | 두 fold 동시 개선 조합 없음 | 희소 군집 |
| 큰 K (좌6/우12, 좌8/우20 등) | 악화 | 1명짜리 군집 + seed 불안정 |
| LightGBM | 697.48 | XGB/Cat보다 47~50점 낮음 |
| CatBoost + insight prior | 766.53 | CatBoost에는 insight 피처가 역효과 |
| seedbag 무조건 확대 | F24 포화/하락 | 모델 수 늘리기 자체는 이득 없음 |
| test batch 통계로 변환 (transductive) | — | 누수·batch 의존 위험으로 격리 (`large_xgb/rejected_transductive/`) |
| 단일 fold 최고 probe (BSS 810.73) | F23에서 +0.0000128 악화 | **F23 재현 실패로 제출 우선순위 제외** |

---

## 9. `submit_013` 이후 진행 상황 (예고)

013 이후 만든 후보 3개다. **전부 Public LB 미제출 상태이므로 아직 "더 좋다"고 말할 수 없다.** 013이 계속 안전 기준점이다.

| 파일 | 기반 | 추가한 것 | Val2024 BSS | 상태 |
|---|---|---|---:|---|
| `submit_014` | 013 구조 보존 | **24-leaf diverse XGB expert 추가** (anchor 0.10 + large 0.90 고정, success 0.200 / reverse 0.575 재학습, 3-way 가중치 0.35955 / 0.25167 / 0.38878) | 813.432 | 고위험 비교 후보 — F23/F24 최적 외부 가중치 차이가 큼 |
| `submit_015` | **013** | **R 문맥 잔차 보정** (`볼카운트 × 4이닝구간 × 투·타 손`, 192셀 lookup, smoothing 5000, scale 1.15, `game_type=R`에만 적용) | **830.523** (R 832.860) | 유력 |
| `submit_016` | **014** | 위와 동일한 R 문맥 보정 | **831.321** (R 833.466) | 내부 최고 |

R 문맥 보정이 왜 큰가: 7절에서 본 R의 볼카운트 오차를 직접 겨냥한다. 단년 집계(822~825)보다 **2개 시즌 OOF 잔차 동일 가중 결합(830.523)**이 안정적이었고, `game_type=R`에만 적용하므로 **F 예측과 F Brier는 전혀 바뀌지 않는다** (F의 불안정한 지분을 건드리지 않는다는 뜻). TrackMan과 투수 군집도 사용하지 않는다.

미제출 내부 후보:

| 후보 | Val2024 BSS | 특징 |
|---|---:|---|
| 공동 SVD 공격형 (λ100, dim4, unit, 투수 4/8, 타자 6/8) | 815.083 | 내부 최고였으나 seed 상관 F23 0.897 — 불안정 |
| 공동 SVD 안정형 seed97 (투수 3/6, 타자 3/4) | 814.831 | seed 상관 F23 0.978 / F24 0.985 |
| 공동 SVD 안정형 5-seed 평균 | 814.711 | 권장 안전형 |

공동 SVD가 유망한 이유는 점수가 아니라 **기존 reverse correction과 2024 상관이 약 0.09로 낮다**는 점이다 (멀티뷰는 0.887이라 가중치가 0으로 수렴해 버렸다). 상관이 낮은 신호만 앙상블에서 실제 가치가 있다.

**현재 제출 우선순위: `016` → `015` → 안전 기준점 `013`**

---

## 10. 팀원 확인 포인트

1. **`submit_013`이 기준선이라는 합의.** 내부 점수 1~2점 차이로 제출본을 교체하지 않는다. LB 검증이 끝난 것은 013뿐이다.
2. **새 아이디어는 "피처 추가"보다 "보정층 추가"로 설계한다.** 피처 직접 투입은 748~784, 보정층은 806~831. 이 격차는 우연이 아니다.
3. **새 후보를 가져올 때 반드시 같이 제출할 숫자**: Val2023 ΔBrier, Val2024 ΔBrier, **R만 분리한 두 연도 ΔBrier**, 기존 correction과의 예측 상관, 245,789행 추론 시간.
4. **HPO는 후순위.** trial 93 파라미터가 25회 재탐색에서도 최고였다.
5. 다음 우선 과제: R 전용 middle residual → 볼카운트 보정 → 공동 SVD의 R/F 재분해.

---

## 11. 원본 파일

| 내용 | 경로 |
|---|---|
| 제출 기록·해시·LB | `submit/2026-08-12/SUBMISSION_LOG.md`, `submit/2026-08-06/SUBMISSION_LOG.md` |
| 전체 진행 상태 | `status.md` |
| 군집·상성 실험 전체 | `experiment/model_optimization/pitcher_cluster_matchup/RESULTS.md` |
| Val 성능 추이 요약 | `experiment/model_optimization/MODEL_VAL_SUMMARY.md` |
| 인사이트 총정리 | `experiment/model_optimization/INSIGHTS_SUMMARY.md`, `CURRENT_MODELING_SUMMARY.md` |
| 대형 모델 실험 | `experiment/model_optimization/LARGE_MODEL_RESULTS.md` |
| 전처리·R 문맥 | `experiment/model_optimization/PREPROCESSING_MODELING_RESULTS.md`, `PREPROCESSING_REVIEW.md` |
| 검증 레지스트리 (1,648행 · fold2024 실험 128종) | `experiment/model_optimization/validation_registry.csv` |
| 피처 ablation | `experiment/model_optimization/insight_feature_ablation_results*.csv` |
| Optuna best params | `experiment/model_optimization/{xgboost,catboost,lightgbm}_*_best.json` |
| Optuna DB | `experiment/model_optimization/*.db` |
| submit_013 manifest | `experiment/model_optimization/pitcher_cluster_matchup/final/reverse_seedbag_submission_manifest.json` |
| 앙상블 선택 결과 | `experiment/model_optimization/enhanced_ensemble_selection.json` |
| 2025 frozen artifact | `experiment/model_optimization/pitcher_cluster_matchup/final/robust_matchup_v1/` |
