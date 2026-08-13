# 성능 최우선 모델·Optuna 최적화 계획

## 0. 결론

최종 모델은 한 종류로 끝내지 않는다. **CatBoost + LightGBM + XGBoost + 신경망/임베딩 모델**을 시간 순방향 OOF로 학습하고, 마지막에 **비음수 가중 블렌딩과 확률 보정**을 적용한다.

우선순위는 다음과 같다.

1. 시간 순방향 CV와 Brier 목적함수를 고정한다.
2. CatBoost, LightGBM, XGBoost를 각각 독립 Optuna study로 튜닝한다.
3. 기본 피처, OOF 투수 임베딩, Trackman 요약 피처를 단계적으로 ablation한다.
4. 상위 설정을 전체 데이터와 여러 seed로 재학습한다.
5. OOF 예측만 사용해 블렌딩하고, 마지막에 확률을 보정한다.

현재 임베딩 단독 2024 BSS는 363.23으로 공식 베이스라인 기준 549.51보다 낮다. 따라서 임베딩은 단독 주력 모델이 아니라 부스팅 모델의 추가 피처 및 앙상블 다양성 확보 수단으로 사용한다.

---

## 1. 검증과 Optuna 목적함수

### 1.1 시간 순방향 fold

무작위 K-fold는 사용하지 않는다. 같은 선수의 미래 정보와 시즌 분포가 과거 fold로 섞이면 리더보드 성능을 과대평가할 수 있다.

| fold | 학습 시즌 | 검증 시즌 | 역할 |
|---|---|---|---|
| F1 | 2019~2021 | 2022 | 오래된 시즌 안정성 |
| F2 | 2019~2022 | 2023 | 최근 체계 변화 적응성 |
| F3 | 2019~2023 | 2024 | 2025 일반화의 가장 중요한 대리 지표 |

상위 후보 최종 확인 때는 2024를 상·하반기 또는 월 단위로 한 번 더 나눠 최근 구간 안정성을 확인한다. 이 추가 split은 모델 선택의 보조 지표이며, 공식 test 행끼리 통계를 만드는 데 사용하지 않는다.

### 1.2 직접 최소화할 값

공식 점수는 기준 Brier 대비 개선율이므로, Optuna에서는 잘린 BSS가 아니라 아래 **정규화 Brier 비율**을 최소화한다.

```text
NB_f = Brier_f / (r_f * (1-r_f))
```

`r_f`는 해당 검증 fold의 실제 성공률이다. BSS의 `max(0, ...)`를 목적함수에 그대로 넣으면 나쁜 trial이 전부 0으로 뭉쳐 TPE가 차이를 학습하지 못하므로 사용하지 않는다.

최종 robust objective:

```text
weighted_mean = 0.15*NB_2022 + 0.30*NB_2023 + 0.55*NB_2024
objective     = 0.75*weighted_mean + 0.25*max(NB_2022, NB_2023, NB_2024)
```

- 2025 평가와 가장 가까운 2024에 가장 큰 가중치를 준다.
- 최악 fold를 25% 반영해 특정 시즌에만 맞는 설정을 막는다.
- 참고 지표로 fold별 Brier, BSS, logloss, AUC, `예측 평균-정답 평균`, calibration slope/intercept를 모두 저장한다.

### 1.3 최근성 가중치

각 fold의 학습 행 가중치는 fold 검증 연도를 기준으로 계산한다.

```text
weight = 0.5 ** (season_age / half_life)
```

- `half_life`: Optuna에서 0.35~8.0 시즌, log scale
- 선택지: 최근 1/2/3/4/전체 시즌 window
- fold마다 검증 시즌 이후 정보는 절대 사용하지 않는다.
- `class_weight`, oversampling은 기본적으로 사용하지 않는다. 양성률을 바꾸면 확률 보정과 Brier가 악화될 수 있다.

---

## 2. 단계별 모델 후보군

### Stage A. 확률 기준선

| 후보 | 역할 | 우선순위 | 비고 |
|---|---|---:|---|
| 상수 확률 | metric sanity check | 필수 | fold 학습 구간 성공률만 사용 |
| Logistic Regression / Elastic Net | calibration anchor | 높음 | rate 피처와 상황 피처의 매끈한 확률 모델 |
| 공식 RandomForest 복원 | 제출 기준선 | 필수 | 공식 점수와 로컬 CV 차이 확인 |
| ExtraTrees | 비선형 bagging 다양성 | 중간 | 단독 최강보다 앙상블용 |
| HistGradientBoosting | 가벼운 비선형 기준 | 낮음 | 부스팅 파이프라인 검증용 |

기준선은 Optuna를 크게 돌리지 않는다. Logistic은 `C`, `l1_ratio`, 최근성만 50~100 trial, ExtraTrees는 100~200 trial이면 충분하다.

### Stage B. 주력 표형 데이터 모델

#### 1순위: CatBoost

이유:

- `pitcher_id`, `batter_id`, 팀, 경기 유형처럼 고유값이 많은 범주형 피처를 원형 그대로 다룰 수 있다.
- 결측값과 범주형 상호작용을 한 모델에서 처리한다.
- 선수 ID의 ordered target statistics가 일반 one-hot보다 효율적일 가능성이 높다.

주의:

- `row_id`는 제외한다.
- ID와 범주형은 문자열 또는 category로 명시한다.
- 시간 순서 데이터에서는 `has_time=True/False`를 별도 study로 비교한다. 설정 의미가 커서 한 study 안에서 무분별하게 섞지 않는다.
- GPU CatBoost는 사용자 pruning callback 제약이 있으므로 native early stopping + fold 단위 Optuna pruning을 사용한다.

#### 2순위: LightGBM

이유:

- 147만 행에서 탐색 속도가 빠르고 수치형 rate·count·임베딩 피처에 강하다.
- CatBoost와 오차 패턴이 달라 블렌딩 이득이 기대된다.

주의:

- leaf-wise 성장이라 `num_leaves`, `min_child_samples`, `max_depth`의 조합을 강하게 제한한다.
- `pitcher_id`, `batter_id`는 category 코드로 처리하되 train에서 만든 frozen category 사전을 test에 적용한다.
- 드문 ID를 `UNK`로 모으는 빈도 임계값도 별도 탐색한다.

#### 3순위: XGBoost hist

이유:

- histogram tree의 강한 정규화와 grow policy가 LightGBM과 다른 다양성을 만든다.
- dense OOF 임베딩과 연속형 피처 결합에 안정적이다.

주의:

- 고유값이 큰 범주형은 frozen ordinal/frequency encoding 또는 `enable_categorical` 버전을 별도 비교한다.
- GPU 메모리와 DMatrix 복제 비용을 확인한다.

### Stage C. 표현 학습 모델

| 후보 | 입력 | 목적 | 우선순위 |
|---|---|---|---:|
| 현재 Two-tower MLP | 상황 + asof + Trackman + 투수/cohort | 임베딩 및 비선형 상호작용 | 높음 |
| Direct Brier MLP | 같은 입력, success 직접 예측 | 안전한 신경망 앙상블 | 높음 |
| 3-head conditional MLP | reverse/middle/far | 보조 과제 표현 학습 | 규정 확인 후 |
| FT-Transformer / TabTransformer | 수치 + 범주 embedding | 고차 상호작용 | 중간 |
| DeepSets/Set Transformer | 과거 Trackman pitch set | raw history encoder | 연구 후보 |

신경망은 단독 점수보다 **OOF embedding과 예측 다양성**에 초점을 둔다. 복원 보조 라벨이 허용되지 않으면 direct head만 최종 후보에 남긴다.

### Stage D. 스태킹과 보정

최종 후보:

- CatBoost 상위 3~5개 seed/model
- LightGBM 상위 3~5개
- XGBoost 상위 2~4개
- Logistic/ExtraTrees 1~2개
- Direct MLP 및 투수 임베딩 모델 2~4개

스태커는 먼저 복잡한 2차 모델보다 아래 순서로 진행한다.

1. 비음수 가중 평균, 가중치 합 1
2. logit 공간 비음수 가중 평균
3. Ridge/ElasticNet stacker
4. 충분한 OOF가 있을 때만 작은 LightGBM stacker

Brier에서는 과적합된 스태커보다 제약된 평균이 안정적인 경우가 많다. 최종 보정은 blend 이후에 적용한다.

---

## 3. 피처 단계와 ablation

Optuna가 수백 개 피처 조합을 한 study에서 동시에 고르게 하지 않는다. 피처군별 독립 실험으로 성능 기여를 먼저 확정한다.

| 버전 | 피처군 | 목적 |
|---|---|---|
| V0 | 공식 기본 + asof | 재현 가능한 안전 기준 |
| V1 | V0 + 카운트/주자/손잡이 교호작용 | 현재 투구 상황 표현 |
| V2 | V1 + count log/smoothing/uncertainty | 적은 표본 rate 신뢰도 반영 |
| V3 | V2 + OOF 투수 임베딩 48차원 | 과거 성향 표현 |
| V4 | V3 + 이전 완료 시즌 Trackman 집계 | 구속·회전·무브먼트·구종군 성향 |
| V5 | V4 + 최근 1/3/5경기 차이·추세 | 단기 form 반영 |
| V6-exp | V5 + 보조 실패 확률/embedding | 운영진 허용 시만 |

핵심 파생 피처 후보:

- `log1p(asof_*_n)`
- Beta smoothing rate: `(success + alpha*prior)/(n+alpha)`와 posterior variance
- 장기 성공률과 최근 1/3/5경기 성공률 차이
- middle 장기율과 최근 middle율 차이
- ball/strike rate 합 및 불일치량
- 투수-타자 손잡이 matchup
- 카운트 상태 categorical (`0-0`, `3-2` 등)
- 주자/아웃/점수차/leverage interaction
- 투수별 Trackman 평균뿐 아니라 표준편차, 분위수, 구종군별 비율과 시즌 간 변화량
- 신인: 시즌 시작 전/해당 시점 누적 100개 이하 여부, 1~25/26~100 구간, 팀/손잡이/cohort fallback

2019~2020 OOF 임베딩은 현재 0 fallback이므로, V3 실험에서 `oof_available`, `season`, cold-start cohort를 반드시 함께 넣는다.

---

## 4. Optuna 탐색 공간

### 4.1 공통 sampler와 pruner

```python
TPESampler(
    seed=2026,
    multivariate=True,
    group=True,
    constant_liar=True,   # 병렬 worker 사용 시
    n_startup_trials=40,
    n_ei_candidates=48,
)
```

- 모델군마다 study를 분리한다. 서로 조건부 공간이 완전히 다른 모델을 한 study의 `model_family`로 섞으면 TPE 효율이 낮아진다.
- SQLite 또는 PostgreSQL storage를 사용하고 `load_if_exists=True`로 중단 후 재개 가능하게 한다.
- boosting은 native early stopping을 우선 사용한다.
- fold가 끝날 때 `trial.report(running_objective, step=fold_idx)`를 호출하고 Median/Patient pruner를 적용한다.
- 신경망은 epoch마다 report하고 Hyperband pruner를 적용한다.
- CatBoost GPU trial은 Optuna callback 대신 fold 단위 pruning만 사용한다.

### 4.2 CatBoost 탐색

| 파라미터 | 범위 |
|---|---|
| `iterations` | 800~8000, 실제 종료는 early stopping |
| `learning_rate` | 0.003~0.12, log |
| `depth` | 4~10 |
| `l2_leaf_reg` | 1e-3~300, log |
| `random_strength` | 1e-4~30, log |
| `bootstrap_type` | Bayesian / Bernoulli / MVS |
| `bagging_temperature` | 0~10, Bayesian에서만 |
| `subsample` | 0.5~1.0, Bernoulli/MVS에서만 |
| `rsm` | 0.55~1.0 |
| `border_count` | 64 / 128 / 254 |
| `one_hot_max_size` | 2 / 8 / 32 / 128 |
| `leaf_estimation_iterations` | 1~10 |
| `od_wait` | 100~500 |
| `half_life` | 0.35~8.0, log |

1차는 `grow_policy=SymmetricTree`로 고정해 안정적 공간을 찾는다. 2차에서 상위 설정 주변만 `Depthwise`/`Lossguide`, `min_data_in_leaf`, `max_leaves<=64`를 별도 study로 비교한다.

### 4.3 LightGBM 탐색

| 파라미터 | 범위 |
|---|---|
| `n_estimators` | 1000~12000, early stopping |
| `learning_rate` | 0.002~0.08, log |
| `num_leaves` | 15~511, log integer |
| `max_depth` | 4~12 또는 -1 |
| `min_child_samples` | 100~10000, log integer |
| `min_sum_hessian_in_leaf` | 1e-3~100, log |
| `subsample` | 0.5~1.0 |
| `subsample_freq` | 1~10 |
| `colsample_bytree` | 0.5~1.0 |
| `reg_alpha` | 1e-8~100, log |
| `reg_lambda` | 1e-8~300, log |
| `min_split_gain` | 0~2 |
| `max_bin` | 63 / 127 / 255 / 511 |
| `cat_smooth` | 1~100, log |
| `cat_l2` | 1e-3~100, log |
| `half_life` | 0.35~8.0, log |

제약: `max_depth>0`이면 `num_leaves <= 2**max_depth`로 clip/재표현한다. 147만 행에서는 아주 작은 leaf를 피하도록 `min_child_samples` 하한을 크게 둔다.

### 4.4 XGBoost 탐색

| 파라미터 | 범위 |
|---|---|
| `n_estimators` | 1000~12000, early stopping |
| `learning_rate` | 0.002~0.10, log |
| `max_depth` | 3~12 |
| `min_child_weight` | 1~2000, log |
| `subsample` | 0.5~1.0 |
| `colsample_bytree` | 0.5~1.0 |
| `colsample_bylevel` | 0.6~1.0 |
| `gamma` | 1e-8~30, log |
| `reg_alpha` | 1e-8~100, log |
| `reg_lambda` | 1e-6~300, log |
| `max_bin` | 64 / 128 / 256 / 512 |
| `grow_policy` | depthwise / lossguide |
| `max_leaves` | 16~256, lossguide에서만 |
| `half_life` | 0.35~8.0, log |

### 4.5 신경망 탐색

| 파라미터 | 범위 |
|---|---|
| `pitcher_embedding_dim` | 8 / 16 / 24 / 32 / 48 |
| `trackman_tower_dim` | 16 / 24 / 32 / 48 / 64 |
| `hidden_dim` | 64 / 96 / 128 / 192 / 256 |
| `n_layers` | 2~5 |
| `dropout` | 0~0.40 |
| `learning_rate` | 1e-5~3e-3, log |
| `weight_decay` | 1e-8~0.1, log |
| `batch_size` | 2048 / 4096 / 8192 / 16384 |
| `shrinkage_tau` | 10~1000, log |
| `brier_weight` | 0.5~1.0 |
| `bce_weight` | 0~0.5 |
| `aux_component_weight` | 0~0.5, 허용 시만 |

focal loss와 class balancing은 순위 지표에는 유용할 수 있지만 확률 자체를 왜곡할 수 있으므로 기본 후보에서 제외한다.

### 4.6 블렌딩·보정 탐색

OOF 예측을 캐시한 뒤 1000~5000 trial을 매우 싸게 돌린다.

- 각 모델 raw weight를 `0~1`로 제안한 뒤 합이 1이 되도록 정규화
- 모델군별 weight cap도 비교: 한 family 최대 0.7~0.9
- probability 평균 vs logit 평균
- calibration 후보:
  - none
  - affine-logit/Platt: `sigmoid(a*logit(p)+b)`
  - beta calibration: `sigmoid(a*log(p)+b*log(1-p)+c)`
  - isotonic
- calibrator는 각 시간 fold 안에서 모델 학습에 쓰지 않은 과거 calibration slice로만 적합한다.
- isotonic은 표본은 많지만 season shift에 과민할 수 있으므로 fold 간 개선이 일관될 때만 채택한다.

---

## 5. 탐색 예산

시간 제한이 없더라도 처음부터 모든 행으로 수천 trial을 돌리면 비효율적이다. 아래 successive-fidelity 방식으로 진행한다.

| 단계 | 데이터/검증 | trial 수 | 통과 기준 |
|---|---|---:|---|
| S0 코드 검증 | 5~10% 표본, F3 일부 | 모델당 10~20 | 누수·오류·속도 확인 |
| S1 광역 탐색 | 25~35% 시간층화 표본, F2+F3 | 모델당 150~300 | 상위 15% |
| S2 정밀 탐색 | 전체 행, F1~F3 | 모델당 250~500 | 상위 20 설정 |
| S3 재평가 | 전체 행, 3 seeds | 모델당 상위 20 | 평균+최악 fold |
| S4 최종 후보 | 전체 행, 5 seeds | family별 상위 3~5 | OOF pool 저장 |
| S5 blend/calibration | 캐시된 OOF | 1000~5000 | robust objective |

권장 총량은 부스팅 약 1000~1800 trial, 신경망 200~400 trial, blend 2000 trial 이상이다. 성능이 50~100 trial 동안 갱신되지 않으면 탐색 공간을 상위 구간 중심으로 좁혀 새 study를 시작한다.

---

## 6. 최종 선택 규칙

단순히 Optuna `best_trial` 하나를 고르지 않는다.

1. robust objective 상위 20개 추출
2. 3개 seed 재학습 후 평균과 표준편차 계산
3. 2024 성능, 최악 fold, calibration slope를 함께 확인
4. 상관계수 0.995 이상인 중복 모델은 대부분 제거
5. 잔차 상관이 낮은 모델을 우선해 OOF blend 후보 구성
6. 블렌딩 후 cross-fitted calibration 적용
7. 마지막으로 전체 2019~2024 재학습 및 2025용 frozen lookup 결합
8. 245,789행, 10분, 28GB RAM 조건으로 제출 zip 벤치마크

최종 채택 조건:

- 모든 시간 fold에서 상수 확률보다 개선
- 2024 fold의 BSS가 공식 기준선 재현 모델보다 개선
- calibration 후 특정 fold만 좋아지고 다른 fold가 크게 나빠지지 않음
- test 다른 행의 빈도·분포·순서를 전혀 사용하지 않음
- 보조 실패 라벨 모델은 운영진 허용 답변이 있을 때만 최종 앙상블에 포함

---

## 7. 실행 순서

1. `optuna_max_performance_plan.ipynb`의 설정·metric·fold 셀 실행
2. V0 피처로 공식 RF와 CatBoost/LGBM/XGB 시간 CV 기준선 생성
3. S1 광역 탐색 실행
4. V1~V5를 상위 CatBoost/LGBM 설정으로 ablation
5. 확정 피처로 S2~S4 전체 탐색
6. 신경망 direct head와 OOF 임베딩 재튜닝
7. OOF 예측 저장 후 blend/calibration study 실행
8. 전체 재학습, 제출 패키지 생성, 자원·행 독립성 검증

