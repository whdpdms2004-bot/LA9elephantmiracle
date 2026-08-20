# 3WAY 최종 확률 모델링 계획

작성일: 2026-08-19  
기준선: `submit_037`, Public 958.2563447143, 서버 추론 8초

현재 전체 최고 기록: `submit_034`, Public **983.5821977654**  
현재 3WAY 최고 기록: `submit_040`, Public **976.9464685456**
(`submit_037` 대비 **+18.6901238313**, 전체 최고까지 **6.6357292198**). 이 점수는 완료된 실험 기록으로만 보존하며
후속 calibration 계수·offset·blend weight를 역산하는 데 사용하지 않는다.

> 2026-08-20 초기 실행 메모: 기존 screen OOF는 `eval_set`과
> `use_best_model=True`로 채점 fold 라벨을 조기 종료에 사용했다. 규정 위반은 아니지만
> 최종 고정 900회 모델과 분포가 다르므로 meta 학습용으로 확정하지 않는다. 모든
> 결합·보정 실험은 `build_honest_oof.py`가 만드는 고정 900회 strict-forward OOF로
> 다시 통과해야 한다.

## 1. 목표

두 문제를 순서대로 푼다.

1. 각 way의 최적 ensemble이 내는 `p_m`, `p_r`, `p_o`로
   `P(M ∪ R ∪ O)`를 추정하고, 그 여집합으로 최종 `p_success`를 만든다.
2. 장기적으로 세 way를 따로 학습한 뒤 합치는 구조를 넘어, ensemble의 지식을 보존한
   공동 확률모델로 재학습·미세조정한다.

Public 958.256은 기준선 기록으로만 사용한다. Public 점수로 결합 계수, offset,
calibration 값을 정하거나 역산하지 않는다.

## 2. 고정할 수학 구조

```text
M  = middle
R  = reverse
O  = outside = failure AND !M AND !R
Y  = control_success

P(Y=1) = 1 - [P(M) + P(R) + P(O) - P(M AND R)]
```

따라서 순수 3입력 모델과 구조 보조입력 모델을 분리한다.

| 트랙 | 입력 | 목적 |
|---|---|---|
| `PURE3` | `p_m, p_r, p_o` | 세 주변확률에서 빠진 `p_mr`을 추정한 뒤 합집합 계산 |
| `STRUCT4` | `p_m, p_r, p_o, p_mr` | 네 확률을 일관된 공동 상태분포로 만들고 합집합 계산 |

`STRUCT4`가 실전 주력이다. `PURE3`는 세 주변확률만으로 교집합을 얼마나 복원할 수
있는지 측정한다. 어떤 트랙도 `p_success=f(p_m,p_r,p_o)`라는 의미 없는 자유 회귀를
주 모델로 사용하지 않는다.

## 3. 가장 먼저 만들 데이터: 시간순 nested OOF

메타모델보다 이 데이터가 먼저다. 각 행의 teacher 확률은 반드시 그 행의 시즌보다
이전 데이터만 학습한 모델에서 나와야 한다.

### OOF 생성표

| 예측 시즌 | teacher 학습 시즌 | 용도 |
|---:|---|---|
| 2021 | 2019~2020 | 초기 OOF / 강건성 |
| 2022 | 2019~2021 | **주 학습 제외**, regime stress 및 낮은 가중치 실험만 |
| 2023 | 2019~2022 | 메타 학습·검증 |
| 2024 | 2019~2023 | 최종 의사결정 fold |
| 2025 | 2019~2024 | 최종 제출 추론 |

2022는 시간상 누수가 있는 것은 아니지만 기존 실험에서 2024 순위와 역신호였다.
따라서 기본 정책은 제외하고, `weight=0.10~0.25`의 강건성 실험에서만 사용한다.

### 각 OOF 행에 저장할 값

```text
row_id, season, game_month
y_success, y_m, y_r, y_o, y_mr

p_m_mean,  p_m_logit_mean,  p_m_logit_std,  p_m_members...
p_r_mean,  p_r_logit_mean,  p_r_logit_std,  p_r_members...
p_o_mean,  p_o_logit_mean,  p_o_logit_std,  p_o_members...
p_mr_mean, p_mr_logit_mean, p_mr_logit_std, p_mr_members...

p_identity
teacher_recipe_version
```

way 내부 ensemble 가중치는 그 OOF 행의 라벨로 정하지 않는다. 예를 들어 2024
teacher ensemble 가중치는 2023 이하의 inner OOF에서 고정하고 2024에 적용한다.

### nested 평가

| outer 평가 | 메타모델 학습 | 판정 |
|---|---|---|
| 2023 | 2021 OOF, 2022는 별도 옵션 | 방향성·붕괴 여부 |
| 2024 | 2021+2023 OOF, 2022는 별도 옵션 | 주 의사결정 |

최종 2025 메타모델은 2021+2023+2024 OOF로 학습한다. 2022 포함 여부는 outer
평가에서 미리 정한 정책을 그대로 쓴다.

## 4. 단기 트랙 A: 세 사건의 합집합 확률 만들기

### A0. 현재 구조식 기준선

```text
p0 = clip(1 - (p_m + p_r - p_mr + p_o))
```

모든 실험은 A0의 **raw Brier/BSS**를 넘어야 한다. centered BSS만 좋아지는 모델은
채택하지 않는다.

### A1. PURE3 bounded overlap 추정기

세 확률만으로 먼저 교집합을 추정한다.

```text
p_mr_hat = g(logit(p_m), logit(p_r), logit(p_o),
             p_m*p_r, min(p_m,p_r), ensemble_uncertainty)

p_success = clip(1 - (p_m + p_r + p_o - p_mr_hat))
```

`g`는 자유로운 최종 결합기가 아니라 **M과 R의 조건부 의존성 모델**이다. 후보는
다음 순서로만 넓힌다.

1. ridge logistic
2. 제약된 2차 logistic
3. 단조 3D lattice

`p_mr_hat`에는 Fréchet 범위를 사후 clip으로만 처리하지 않고 모델 구조에 넣는다.

```text
lower = max(0, p_m + p_r - 1)
upper = min(p_m, p_r)
p_mr_hat = lower + (upper-lower)*sigmoid(h(features))
```

이 방식은 최종 성공 라벨보다 먼저 실제 `y_mr`를 맞추므로 의미가 분명하고, 자유
결합기가 구조를 뒤집는 것을 막는다.

### A2. STRUCT4 공동상태 reconciliation

네 teacher 확률을 다섯 상호배타 상태로 바꾼다.

```text
q = [q_success, q_m_only, q_r_only, q_mr, q_outside]
sum(q) = 1, q >= 0

p_m  = q_m_only + q_mr
p_r  = q_r_only + q_mr
p_o  = q_outside
p_mr = q_mr
p_failure = 1 - q_success
```

초기 공동상태는 teacher 확률로 만든다.

```text
q_mr      <- p_mr
q_m_only  <- p_m - p_mr
q_r_only  <- p_r - p_mr
q_outside <- p_o
q_success <- 1 - (p_m + p_r - p_mr + p_o)
```

서로 따로 학습된 teacher 때문에 음수나 합 불일치가 생기면, 행별 constrained simplex
projection 또는 작은 softmax reconciler로 가장 가까운 유효 공동분포를 구한다. OOF
학습 손실은 다음처럼 상태와 주변확률을 함께 맞춘다.

```text
L = CE(q, true_5_state)
  + lambda_marginal * [Brier(q_m_only+q_mr, y_m)
                     + Brier(q_r_only+q_mr, y_r)
                     + Brier(q_outside, y_o)]
  + lambda_teacher * distance(q, q_teacher_initial)
```

이 모델이 내는 최종확률은 별도 head가 아니라 반드시 `q_success`다.

### A3. 단조 overlap lattice/GAM

A1의 ridge overlap이 양 fold에서 이길 때만 교집합 의존성 모델에 비선형성을 연다.

- `p_m`, `p_r`과 함께 가능한 교집합 범위가 커지는 구조 유지
- 출력은 항상 Fréchet 범위 안
- 입력 knot는 학습 OOF에서만 고정
- 최대 3~5 knot/축

### A4. 확률 보정의 위치

scalar `p_success`만 따로 보정하면 공동분포 일관성이 깨진다. 따라서 주 경로에서는
최종 성공확률에 beta calibration을 바로 붙이지 않는다. 보정이 필요하면 다음 둘 중
하나만 허용한다.

1. 각 주변 teacher를 OOF에서 먼저 보정한 뒤 공동분포 계산
2. 5-state logits를 함께 보정하고 softmax로 다시 정규화

어느 경우든 outer 2023/2024 raw Brier가 모두 개선되고 test 분포를 전혀 보지 않아야
한다.

### A5. way masking으로 한 성분 불안정에 대비

세 way 중 하나의 ensemble이 특정 시즌·선수군에서 갑자기 흔들려도 최종 확률이 같이
붕괴하지 않도록, meta/student 학습에 **way 단위 structured masking**을 넣는다. 이는
개별 숫자를 무작위로 0으로 만드는 feature dropout이 아니다. 한 way에 속한
`mean/logit/std/member predictions`를 한꺼번에 가리고 `mask_k=1`을 명시하는
component dropout이다.

MR은 M·R의 교집합이므로 다음 종속 규칙을 적용한다.

```text
mask M  -> p_m과 p_mr 계열을 함께 mask
mask R  -> p_r과 p_mr 계열을 함께 mask
mask O  -> p_o 계열만 mask
mask MR -> p_mr만 가리는 별도 교집합 stress arm
```

가린 확률을 `0`으로 대체하면 “사건이 절대 발생하지 않는다”는 잘못된 의미가 된다.
대체값은 rolling OOF 학습 구간에서 고정한 사건 prior 또는 학습 가능한 missing token을
사용하고, mask indicator를 반드시 함께 준다. 출력은 마스킹 상황에서도 항상
`[SUCCESS, M_ONLY, R_ONLY, MR, OUTSIDE]` 5-state softmax로 만들어 확률합 1과
포함–배제 일관성을 보장한다.

#### 학습 분포

초기 mask schedule은 다음 범위에서 OOF로만 선택한다.

| 학습 행 | 비율 후보 | 목적 |
|---|---:|---|
| clean, mask 없음 | 60~75% | 정상 입력 성능 보존 |
| M/R/O 중 정확히 하나 mask | 20~30% | 주 강건성 학습 |
| 두 way mask | 0~5% | 극단 stress, 과도 의존 방지 |
| 세 way 전부 mask | 0% | 식별 정보가 없어 학습하지 않음 |

clean 행과 masked 복제 행을 함께 쓰며 손실은 다음처럼 둔다.

```text
L = L_clean_state
  + lambda_mask * L_masked_state
  + lambda_consistency * KL(q_masked || stopgrad(q_clean))
```

`L_masked_state`는 실제 5-state 라벨을 사용하고, consistency는 정상 예측을 완전히
복제시키는 강제가 아니라 작은 regularizer로만 둔다. 한 way가 다른 way를 억지로
대신해 정상 성능을 깎지 않도록 `lambda_mask`와 mask 비율은 nested OOF에서 정한다.

#### 비교군과 stress test

1. masking 없음
2. 숫자만 prior로 대체, indicator 없음
3. way masking + indicator
4. way masking + indicator + clean/masked consistency
5. ensemble member dropout만 적용

검증에서는 단순 hard mask 외에 각 way별 logit bias `±0.25/±0.50`, temperature 왜곡,
member 일부 탈락, ensemble dispersion 증가를 넣는다. 정상·M 불안정·R 불안정·O
불안정의 평균뿐 아니라 **최악 시나리오 Brier**를 기록한다.

채택 조건은 다음을 모두 만족해야 한다.

- clean outer 2024 raw BSS가 masking 없는 같은 모델 대비 2 이상 악화되지 않음
- one-way stress 평균 Brier와 worst-case Brier가 모두 개선
- M/R/O 어느 한 축에서도 masking 없는 모델보다 더 크게 붕괴하지 않음
- R/F별 성능과 월별 Brier에 새로운 집중 손실이 없음
- 모든 mask 패턴에서 finite, `[0,1]`, 5-state 합 1, 포함–배제 오차 0

실제 추론에서는 무작위 masking을 하지 않는다. OOF에서 미리 정한 **행 단위** 품질
조건(비유한값, 모델 자산 누락, 같은 way 멤버 간 과도한 dispersion 등)이 발생할 때만
해당 way를 mask한다. 임계값은 학습 OOF에서 고정하며 test 배치의 평균·분포·순위를
사용하지 않는다. 정상 행에서는 clean 경로를 그대로 쓴다.

### A6. 진단 비교군: direct residual

현재 항등식에 ridge 잔차를 더하는 직접 성공확률 모델은 **진단 비교군**으로만 둔다.
이 모델이 좋아져도 공동분포 모델이 같은 개선을 설명하지 못하면 최종 채택하지 않는다.

### A7. 금지할 설계

- 한 outer fold에서 학습하고 같은 fold에서 성능 보고
- 2024 라벨로 결합기를 맞춘 뒤 2024 BSS를 보고
- test prediction mean/std/rank를 이용한 보정
- Public 점수로 offset/weight 역산
- 사건 의미를 무시한 unconstrained GBDT나 깊은 MLP를 첫 결합기로 사용

## 5. 중기 트랙 B: 공동 상태 CatBoost

세 주변확률을 각각 맞춘 뒤 교집합을 빼는 대신, 처음부터 다섯 상호배타 상태를
예측한다.

```text
class 0: SUCCESS
class 1: M_ONLY
class 2: R_ONLY
class 3: MR
class 4: OUTSIDE
```

CatBoost `MultiClass`가 내는 softmax 확률을 `pi`라 하면:

```text
p_success = pi[SUCCESS]
p_m       = pi[M_ONLY] + pi[MR]
p_r       = pi[R_ONLY] + pi[MR]
p_o       = pi[OUTSIDE]
p_mr      = pi[MR]
```

이 구조는 포함–배제 일관성을 자동으로 보장한다.

### 비교군

1. 같은 F1 공통 피처의 5-state CatBoost
2. way별 최적 피처의 합집합을 쓴 5-state CatBoost
3. 5-state CatBoost와 기존 STRUCT4 공동분포의 OOF blend
4. 기존 `p_identity`를 baseline으로 넣은 success residual CatBoost

residual CatBoost는 원본 피처를 자유롭게 다 쓰지 않고 다음부터 시작한다.

```text
p_m, p_r, p_o, p_mr
각 way ensemble dispersion
asof 표본수/신뢰도
balls_before, strikes_before, inning, batter_stand, pitcher_hand
```

피처를 단계적으로 늘리고, `p_identity` 대비 OOF 증분만 본다.

## 6. 장기 트랙 C: ensemble 증류 + 공동 미세조정

### 권장 구조

```text
입력 피처
  -> 수치 embedding + 범주 embedding
  -> [shared experts]
     [middle-private expert]
     [reverse-private expert]
     [outside-private expert]
  -> task별 gate (MMoE)
  -> 5-state coherent softmax head
  -> p_success = softmax[SUCCESS]
```

기본 trunk는 두 후보만 비교한다.

1. 작은 MLP/MMoE
2. TabM식 parameter-efficient ensemble trunk

attention/Transformer는 이 둘이 실패한 뒤에만 연다.

### 학습 손실

모든 teacher target은 시간순 OOF만 사용한다.

```text
L_state   = CE(5-state softmax, true_state)
L_success = Brier(p_success, y_success)
L_way     = BCE(p_m,y_m) + BCE(p_r,y_r) + BCE(p_o,y_o) + BCE(p_mr,y_mr)
L_distill = sum_k Brier(p_student_k, p_teacher_oof_k)
L_mask    = masked 5-state supervised loss + clean/masked consistency

L_total = a*L_state + b*L_success + c*L_way + d*L_distill + e*L_mask
```

초기 실험은 `(a,b,c,d)=(1,2,0.5,1)`로 시작하되, 최종값을 수동 고정하지 않는다.
GradNorm으로 task별 gradient 크기를 관찰하고 균형을 맞춘다. 공유층에서 task gradient
cosine이 반복적으로 음수면 그때 PCGrad를 비교한다.

### 미세조정 순서

1. **Teacher 준비:** 각 way ensemble의 rolling OOF mean/std 생성
2. **Warm start:** 실제 5-state 라벨 + teacher soft probability로 trunk와 private
   expert 학습
3. **Joint training:** 모든 head를 열고 `L_total` 학습
4. **Mask robustness:** one-way structured masking을 열고 clean/masked 혼합 미세조정
5. **Success tuning:** trunk 하단과 private expert를 동결하고 success/state head만
   작은 learning rate로 Brier 미세조정
6. **Partial unfreeze:** 마지막 shared block만 1/10 learning rate로 해제
7. **Seed/TabM member 평균:** outer fold에서 미리 정한 방식으로만 평균

CatBoost는 이 구조로 end-to-end 역전파할 수 없으므로 기존 CatBoost ensemble은
frozen teacher다. "파인튜닝"은 CatBoost 자체의 leaf를 억지로 업데이트하는 것이
아니라, teacher 지식을 받은 공동 student를 최종 성공확률에 맞게 단계적으로
미세조정하는 의미로 사용한다.

## 7. 판정 지표와 채택 관문

모든 표에 다음을 함께 기록한다.

- raw Brier / raw BSS
- centered BSS는 진단용만
- R/F별 BSS
- 월별 Brier
- prediction mean과 target mean의 차이
- calibration intercept/slope
- 각 way 및 현재 identity와의 logit 상관
- seed 또는 ensemble member 간 분산

### 단기 결합기 채택

다음을 모두 만족해야 한다.

1. outer 2024 raw BSS가 A0보다 **최소 +5**
2. outer 2023이 A0 대비 **-5 아래로 악화되지 않음**
3. outer 2024 월별 Brier 8개 중 최소 5개 개선
4. R/F 중 한쪽을 +20 이상 희생해 전체만 좋아지는 현상 없음
5. 세 가지 meta seed/초기화 중 개선 방향 일치

### 장기 공동모델 채택

1. 단독 raw BSS가 current identity를 양 fold에서 넘음
2. 또는 단독은 동률이지만 current identity와의 logit 상관이 충분히 낮아 정직한 OOF
   blend에서 +5 이상
3. 5-state 확률합과 논리 관계 오차가 기계적으로 0
4. 추론 10분 제한, 행 독립성, 오프라인 실행 통과

한 조건이라도 실패하면 현재 `submit_037` 구조식을 유지한다.

## 8. 실행 순서

### Phase 0 — OOF 자산 고정

- [ ] way별 최종 ensemble recipe와 멤버 목록 고정
- [ ] `eval_set`, `use_best_model`, early stopping이 없는지 소스 검사
- [ ] 2021/2023/2024 rolling OOF 생성
- [ ] 2022 stress OOF 별도 생성
- [ ] `meta_oof.parquet`와 manifest 저장
- [ ] 각 teacher의 TrackMan cutoff 검증
- [ ] 동일 행 단독/배치 teacher 확률 일치 확인

GPU 필요: **있음.** 기존 way ensemble을 각 시간 fold에서 재생성해야 한다. 동시에
여러 GPU 작업을 돌리지 않는다.

### Phase 1 — 저비용 결합

- [ ] A0 identity 재현
- [ ] A1 PURE3 bounded overlap ridge
- [ ] A2 STRUCT4 5-state reconciliation
- [ ] way별 hard mask/logit drift/member dropout stress baseline 생성
- [ ] 2022 제외/저가중치 정책 비교
- [ ] outer 2023/2024 고정 리포트

GPU 필요: **없음.** CPU에서 충분하다.

### Phase 2 — 구조 모델

- [ ] A3 단조 lattice/GAM
- [ ] 5-state CatBoost
- [ ] identity baseline residual CatBoost
- [ ] 정직한 OOF blend

GPU 필요: CatBoost 모델에 **있음**. lattice/GAM은 CPU 가능하다.

### Phase 3 — 공동 딥러닝

- [ ] 작은 MLP/MMoE 5-state student
- [ ] OOF teacher distillation
- [ ] way structured masking + mask indicator
- [ ] clean/masked consistency와 mask 비율 ablation
- [ ] clean·평균 stress·worst stress 동시 게이트
- [ ] GradNorm, 필요 시 PCGrad
- [ ] TabM trunk 비교
- [ ] 동결 -> head tuning -> partial unfreeze

GPU 필요: **있음.** 모델 하나씩 순차 실행한다.

### Phase 4 — 2025 최종화

- [ ] 선택 규칙을 동결한 뒤 2019~2024 teacher 전체 재학습
- [ ] meta/student는 rolling OOF로만 학습
- [ ] 2025는 `test.csv` 각 행과 고정 model asset만 사용
- [ ] B1 6단계 전체 재검증 및 새 제출 번호 생성

## 9. 규정 방어선

- 학습/OOF: 각 prediction season보다 이전 시즌만 사용
- 최종 teacher: 2019~2024만 사용
- TrackMan: 각 fold의 target season 미만, 최종은 2024 이하만
- 2025 TrackMan 사용 금지
- test 행 간 집계, mean/std/rank, calibration 금지
- meta knot/weight/offset/prior는 학습 OOF에서만 결정
- mask prior·품질 임계값·mask schedule도 학습 OOF에서만 결정
- 추론 중 mask 여부는 현재 행의 고정 품질 조건만 사용하고 test 배치 통계를 보지 않음
- Public 958.256을 모델 선택 또는 값 조정에 사용하지 않음
- 제출 `script.py`는 추론 전용
- 모든 lookup과 meta weight는 `model/`에 동봉
- 단독 행과 전체 배치 예측 일치 검증

## 10. 바로 시작할 최소 실험

첫 구현은 아래 둘이면 된다.

1. `PURE3 bounded overlap ridge`와 기존 별도 `p_mr` 비교
2. `STRUCT4 5-state reconciliation`과 `5-state CatBoost` 비교

첫 번째는 세 주변확률만으로 빠진 교집합을 얼마나 복원할 수 있는지 검사한다. 두
번째는 개별 teacher 확률을 하나의 유효한 공동분포로 만드는 것과 원본 피처에서
공동분포를 직접 예측하는 것을 비교한다. 둘 중 하나가 outer 2023/2024 관문을 넘은
뒤에만 MMoE/TabM 파인튜닝으로 넘어간다.

### 초기 탐색에서 확인된 것

기존 screen OOF로 방향만 탐색했으며 제출 판정 수치로 사용하지 않는다.

| 방법 | Val2024 raw BSS | 기준선 대비 | 판정 |
|---|---:|---:|---|
| 기존 M/R/O/MR 포함–배제 | 828.59 | 0 | 기준선 |
| 세 확률에서 bounded MR 재추정 | 822.63 | -5.96 | 탈락 |
| 사건별 약한 logit 보정 후 합집합 | 842.42 | +13.83 | 정직한 OOF 재검증 대기 |

현재 별도 MR teacher는 이미 모든 행에서 Fréchet 범위를 만족했고 simplex projection도
예측을 바꾸지 않았다. 따라서 다음 우선순위는 MR 재추정보다 **M/R/O/MR teacher의
확률 보정과 5-state 공동학습**이다.

## 11. 예정 산출물

이 계획을 실행할 때 다음 파일을 새로 만든다.

```text
meta_fusion/
├─ RESEARCH_NOTES.md
├─ MODELING_PLAN.md
├─ src/
│  ├─ build_meta_oof.py
│  ├─ fit_overlap.py
│  ├─ fit_joint_reconciler.py
│  ├─ train_joint_state_cat.py
│  ├─ train_joint_student.py
│  └─ evaluate_meta.py
└─ outputs/
   ├─ meta_oof_manifest.json
   ├─ fusion_results.csv
   └─ joint_results.csv
```

## 12. 2026-08-20 초기 제출본 이후 실행 결정

초기 조사에서 가장 작은 안전한 변경인 **사건별 약한 logit 보정 후 정확한 확률
합집합 계산**을 `submit_040`으로 패키징했다. 기존 screen OOF 기준 Val2024 raw
BSS는 828.59에서 842.42로 +13.83이었지만, teacher의 조기 종료가 검증 fold 라벨을
참조해 최종 고정 900회 teacher와 분포가 다르다. 따라서 이 수치는 탐색 신호이고
성능 확정값이 아니다.

실제 Public은 `submit_037` 958.2563447143에서 `submit_040` 976.9464685456으로
**+18.6901238313** 향상됐다. 사건별 확률 보정 방향이 유효하다는 외부 확인이지만,
Public 결과를 이용해 수치 보정을 추가하지 않는다. 다음 모델의 모든 선택은 아래
honest OOF 관문으로만 결정한다.

다음 작업 순서는 아래처럼 고정한다.

1. GPU가 비는 즉시 `build_honest_oof.py`를 재개한다. 완료된
   `outputs/honest_oof/middle_2023.npy`는 재사용한다.
2. fixed-900 OOF2023에서 보정기를 학습하고 OOF2024에서 위 +5 채택 관문을 다시
   검사한다. 실패하면 새 보정안을 폐기하고 3WAY 최고인 `submit_040`을 유지한다.
3. 통과하면 2021 OOF를 추가해 2023 outer 검증까지 수행한 뒤 보정 recipe를 고정한다.
4. 그 다음에만 5-state CatBoost를 학습하고, 마지막으로 MMoE/TabM 공동 student를
   teacher 증류 방식으로 비교한다.

CPU에서 이미 확인한 `PURE3 bounded overlap`은 -5.96 BSS로 탈락했다. 현재 MR
teacher가 전 행에서 Fréchet 범위를 만족하므로, 장기 최적화의 우선순위는 새로운 MR
근사기가 아니라 **각 사건의 확률보정 → 5-state 공동분포 → 공동 student 증류**다.
