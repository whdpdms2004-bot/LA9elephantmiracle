# TrackMan 전처리 스펙

작성: 2026-08-12 / 근거: `experiment/pitcher_embedding/trackman500_cutoff.py` 코드 감사(P0-2), 임베딩 산출물 실측 감사(P0-3), EDA 수치

현재 파이프라인에서 **확인된 전처리 결함 4개를 먼저 고정**하고, 그다음 신규 피처·임베딩이 따를 표준 전처리 순서를 정의한다.

---

## 1. 현재 파이프라인의 확인된 결함

### 결함 1 — 물리량이 원시 절대값으로 집계된다 (2022 측정 체계 단절 미반영)

`trackman500_cutoff.py:310-316`:

```python
season_metric = rows.groupby(["pitcher_trackman_id", "season"])[TM_METRICS].agg(["mean", "std"])
```

`TM_METRICS` 8개 전부가 **정규화 없이** 원시 단위로 평균·표준편차를 낸다. 그 뒤 `tm500_latest_*`, `tm500_recent_*`(EWM half-life 2), `tm500_between_*_std`를 만든다.

문제가 되는 지점은 특히 `tm500_between_*_std`다. 동일 투수 145명(양 시즌 500구+) 기준 2021→2022 리그 전체가 이동했다.

| 지표 | 2021→2022 변화 | 비고 |
|---|---:|---|
| `extension` | **−0.148** | 투수의 **98.6%** 가 감소 |
| `induced_vert_break` | −3.079 | |
| fastball 비중 | −12.08%p | 분류 체계 변화 |
| offspeed 비중 | +8.31%p | |

즉 2021과 2022 시즌을 함께 가진 투수의 `tm500_between_extension_std`는 **투수 개인의 변동성이 아니라 리그 전체 측정계 이동을 재고 있다.** 이 피처는 사실상 "2022 이전 데이터를 갖고 있는가"의 대리변수다.

**조치**: §3-2의 `season × pitch_type_group` robust z 변환을 집계 **이전**에 적용한다. 절대값 버전과 z 버전을 동시에 넣지 말고 ablation으로 하나만 남긴다.

### 결함 2 — crosswalk가 hard 1:1이고 정보를 두 번 버린다

`trackman500_cutoff.py:273-287`:

```python
crosswalk = vote_table.drop_duplicates("pitcher_id", keep="first")      # 1차 절단
crosswalk = crosswalk.sort_values(...).drop_duplicates("pitcher_trackman_id", keep="first")  # 2차 절단
```

`similarity ≥ 0.80` **및** `margin ≥ 0.02`를 통과한 뒤에도 두 번의 `drop_duplicates`로 잘려나간다. 결과 커버리지는 투수 419/792(52.90%), as-of + 500구 게이트 후 2024 행 **60.24%**.

cosine 0.79는 0.79만큼의 증거다. 버릴 이유가 없다.

**조치**: §4의 soft crosswalk.

### 결함 3 — fingerprint가 "팀 일정 + 보직"을 주로 인코딩한다

`MAIN_FINGERPRINT_COLUMNS`에 `game_month`, `game_dayofweek`가 들어간다. 이 두 변수는 **같은 팀 투수 전원이 공유**한다. `inning`은 보직(선발/불펜)을 인코딩한다. 상태 공간 크기는 `12 × 7 × 21 × 2 × 4 × 3 × 3 × 2 = 254,016`인데 투수-시즌 표본은 500~3,000구다 → 극단적으로 희소하고, 유사도가 팀·보직 신호에 지배될 수 있다.

검증은 hand 일치 335/336뿐인데, hand는 2범주라 우연 일치 확률이 50%다. 팀 동료를 잘못 매칭해도 hand는 맞을 수 있다.

**조치**: §4-2의 보강 검증 3종(보직 프로파일, 등판 수, 워크로드 궤적).

### 결함 4 — 임베딩 산출물의 3개 치명적 결함 (P0-3 실측)

`pitcher_season_embedding_oof.csv`(2,260행 × 58열) 직접 측정 결과.

| # | 결함 | 실측 |
|---|---|---|
| 4a | 2019·2020 pitcher-season 711건(31.5%)이 48차원 **전부 0**. 행 기준 481,500행 = 학습의 **32.64%** | 시즌 대리변수화 |
| 4b | `trackman_embedding` 24차원이 `tm_available=0`인 투수에도 채워짐. 미가용 686건의 unique 벡터가 **4개**뿐 | 임베딩 24차원만으로 `tm_available`를 **AUC 0.9983** 으로 예측 가능 = 완전 분리 |
| 4c | latent 축이 fold마다 회전 (`torch.manual_seed(SEED + fold + dim)`, 정렬 단계 없음) | **YoY 안정성이 사실상 0** (아래) |

**YoY 안정성 실측** — 같은 투수의 연속 시즌 임베딩:

| 그룹 | 2021→22 코사인 | 2022→23 | 2023→24 | 차원별 평균 YoY R² |
|---|---:|---:|---:|---:|
| `pitcher_embedding` (16d, 타깃 지도) | −0.005 | +0.054 | +0.076 | **0.0035 ~ 0.0049** |
| `trackman_embedding` (24d) | +0.024 | +0.004 | +0.049 | 0.070 ~ 0.115 |
| `cohort_embedding` (8d) | **−0.203** | +0.166 | **−0.267** | 0.063 ~ 0.119 |

참고선: Stuff+ YoY R² **0.702**, Kirby Index **0.50**, 팀 자체 Kirby 재현 0.26~0.34.

`cohort_embedding`의 코사인이 시즌마다 **부호가 뒤집힌다**. 이는 신경망 재학습마다 latent 축이 임의 회전하는 전형적 증상이다. 2025 예측은 2024까지의 임베딩을 외삽해야 하는데, **축이 정렬되지 않은 임베딩은 원리적으로 외삽이 불가능하다.**

또한 `cohort_embedding`은 `nn.Embedding(5, dim)` — 5개 값의 lookup이므로 `experience_cohort` 컬럼 하나와 정보량이 동일하다. 8차원 전부 낭비.

**조치**: §5의 임베딩 전처리 규약. 특히 **fold 간 Procrustes 정렬을 필수 단계로 승격**한다.

---

## 2. 원자료 정제 (모든 파이프라인 공통, 최초 1회)

`trackman_history.csv` 1,793,078행 × 30열.

### 2-1. 이상치 제거 규칙

`trackman_schema_profile.csv` 실측 기준.

| 컬럼 | 결측률 | 물리적으로 불가능한 값 | 처리 |
|---|---:|---|---|
| `extension` | 0.4303% | **min −0.387** (음수 불가) | `< 0.5` → NaN |
| `rel_height` | 0.4248% | min 0.0971 (10cm) | `< 0.8` → NaN |
| `rel_speed` | 0.4248% | min 71.68 km/h | `< 90` → NaN (이글 볼/측정오류) |
| `spin_rate` | 0.6952% | min 434.9 | `< 800` → NaN |
| `zone_speed` | 0.4418% | — | `zone_speed > rel_speed` → 둘 다 NaN |
| `induced_vert_break` | 0.5612% | max 153.33 | `\|IVB\| > 90` → NaN |
| `horz_break` | 0.5763% | max 103.70 | `\|HB\| > 90` → NaN |

카운트/구조 이상치:

| 항목 | 실측 | 처리 |
|---|---|---|
| `balls_before == 4` | 1건 | 행 제거 |
| `strikes_before == 3` | 1건 | 행 제거 |
| `outs_before ∈ {3,4}` | 95건 | 행 제거 |
| `inning == 0` | 1건 | 행 제거 |
| `(trackman_game_id, pitch_no)` 중복 | 2쌍 / 4행 | 첫 행만 유지 |

**중요**: 이상치를 대체값으로 채우지 않는다. NaN으로 두고 집계에서 `nan`-aware 함수를 쓴다. 투수-시즌 단위 집계는 표본이 500구 이상이므로 0.5% 결측은 무해하다.

### 2-2. 구종 표준화

`tagged_pitch_type` 17종 vs `auto_pitch_type` 11종, 문자열 정확 일치 **55.1%**.

- 모델링에는 `pitch_type_group` 4종(fastball / breaking / offspeed / other)만 사용한다. 세부 구종 확장은 이미 BSS 760.71~764.94로 악화 확인.
- `other`는 2019 2.00% → 2024 0.57%로 감소. **`other`는 별도 집계하지 말고 제외하거나 `breaking`에 병합** (표본 14,713구, 전체 0.82%).
- **불일치율 자체는 피처로 남긴다** (Tier C8). `pitch_type_group` 수준으로 매핑한 뒤 불일치 여부를 센다.

### 2-3. 주 패스트볼 정의 (Tier D1 차분의 기준점)

```
투수-시즌별 primary_fastball = pitch_type_group == 'fastball' 중 사용 빈도 최대인 auto_pitch_type
  단 fastball 표본 < 50구인 투수-시즌은 차분 피처 전부 NaN
```

구종군 물리 기준값(`trackman_physical_by_pitch_group.csv`):

| group | n | rel_speed | spin_rate | IVB | HB | extension |
|---|---:|---:|---:|---:|---:|---:|
| fastball | 931,100 | 142.70 | 2245.84 | 41.73 | 14.08 | 1.802 |
| breaking | 512,845 | 127.53 | 2334.98 | 0.396 | −5.324 | 1.683 |
| offspeed | 326,803 | 130.47 | **1708.29** | 20.72 | 19.26 | 1.757 |
| other | 14,713 | 125.55 | 2135.27 | 12.13 | 14.16 | 1.765 |

---

## 3. 정규화 (결함 1 대응)

### 3-1. 좌우 미러 정규화 (반드시 먼저)

투수 손별 표본이 극단적으로 불균형하다 (P0-6 실측).

| | 투수 수 | 행 비중 |
|---|---:|---:|
| 좌투 | **211** | 25.85% |
| 우투 | 596 | 74.15% |

`horz_break`, `rel_side`는 손에 따라 부호가 뒤집히는 양이다. 표현 학습 전에 미러 정규화하면 좌우 투수를 같은 좌표계에 놓을 수 있다.

```python
sign = np.where(hand == 'Left', -1.0, 1.0)
tm['horz_break_m'] = tm['horz_break'] * sign
tm['rel_side_m']   = tm['rel_side']   * sign
# batter_hand 도 함께 스왑해 '같은 방향 타자' 로 통일
tm['batter_same_side'] = (tm['batter_hand'] == tm['pitcher_hand']).astype(int)
```

**이 변환의 두 번째 용도가 표현 학습 증강이다** (§5-4).

### 3-2. `season × pitch_type_group` robust z (핵심)

```python
KEY = ['season', 'pitch_type_group']
for col in TM_METRICS_M:                       # 미러 정규화 후 컬럼
    med = tm.groupby(KEY)[col].transform('median')
    iqr = tm.groupby(KEY)[col].transform(lambda s: s.quantile(.75) - s.quantile(.25))
    tm[col + '_rz'] = (tm[col] - med) / (iqr.clip(lower=1e-6) / 1.349)
```

median/IQR을 쓰는 이유: §2-1 이후에도 남는 꼬리에 평균/표준편차가 끌려간다. `/1.349`는 정규분포에서 IQR을 SD로 환산하는 계수다.

**왜 `season × pitch_type_group`인가**: 2022 단절은 (a) 측정계 이동과 (b) 구종 분류 재편이 동시에 일어났다. 시즌만으로 정규화하면 구종 비중 변화(fastball −12.08%p)가 잔류한다. 두 축을 함께 잡아야 한다.

**주의**: 정규화 기준(median/IQR)도 **fold별로 학습 시즌만 사용해 계산**해야 한다. 검증 시즌의 분포를 보고 정규화하면 누수다. 단, 정규화는 각 시즌 내부에서 이뤄지므로 검증 시즌 행의 정규화에는 **그 시즌 자신의 통계를 쓸 수 없다** → 직전 시즌 통계를 이월(carry-forward)하거나, 리그 평균 궤적을 선형 외삽한다. 이 선택 자체를 ablation 대상으로 둔다.

### 3-3. Empirical Bayes 축소

비율과 SD 추정은 표본수에 따라 리그 평균으로 축소한다. 팀이 smoothing 200 부근에서 최적을 찾았으므로 같은 규모에서 시작한다.

```python
# 비율 (구종 사용률, 불일치율 등): beta-binomial
#   a0, b0 는 리그 분포의 method-of-moments 추정
p_shrunk = (k + a0) / (n + a0 + b0)

# 분산/SD: chi-square 축소, m0 in {100, 200, 500} 그리드
var_shrunk = (n * var_hat + m0 * var_league) / (n + m0)
sd_shrunk  = np.sqrt(var_shrunk)
```

### 3-4. 절대 수준을 피처로 넣지 않는다 (P0-5 실측 근거)

P0-5에서 투수 수준 **절대 확률**을 피처로 넣으면 성능이 무너지는 것을 직접 확인했다.

| 투입 | 2024 AUC | 2024 BSS | pred_mean (target 0.4861) |
|---|---:|---:|---:|
| BASE (상황 24 + asof 19) | 0.5435 | **567.0** | 0.4959 |
| BASE + oracle (pitcher, season) 확률 | 0.5185 | **−1163.1** | 0.4819 |
| BASE + oracle (batter, season) 확률 | 0.5171 | **−2008.6** | 0.4916 |
| BASE + oracle (pitcher, season, count, bhand) | 0.5517 | **864.6** | 0.4882 |
| BASE + oracle (pitcher, batter, season) | 0.5516 | 812.0 | 0.4832 |

**완벽한 정보(oracle)를 넣어도 절대 확률 형태면 성능이 파괴된다.** 이유: 학습 시즌(성공률 0.489~0.565)과 검증 시즌의 수준이 달라, 트리가 학습한 임계값이 검증에서 계통 편향을 만든다. 반면 셀이 잘게 쪼개진 조건부 형태(count·batter_hand 포함)는 절대 수준이 희석되어 개선된다.

이것이 팀이 "군집·matchup 피처 직접 투입 748~761"로 실패한 메커니즘의 정체다.

> **전처리 원칙 (하드 규칙)**: TrackMan 유래 피처는 **절대 확률·절대 수준으로 투입하지 않는다.** 반드시 `시즌 × 구종군`(물리량) 또는 `시즌 × 투수손 × 타자손 × 볼카운트`(비율·잔차) 평균을 제거한 **잔차 형태**로 투입한다.

---

## 4. Crosswalk 재설계 (결함 2·3 대응)

### 4-1. soft crosswalk

hard 1:1을 후보 분포에 대한 기대값으로 대체한다.

```python
# 시즌·손 제약 하의 유사도 행렬 S (기존 코드 재사용)
# 상위 k=5 후보만 남기고 온도 tau 로 softmax
top = np.argsort(S, axis=1)[:, -k:]
w = softmax(S[rows, top] / tau, axis=1)
w[S[rows, top] < s_min] = 0.0          # s_min = 0.60 정도로 완화
w /= w.sum(1, keepdims=True).clip(1e-9)

profile[pitcher] = (w[:, :, None] * TM_PROFILE[top]).sum(1)   # 기대 프로파일

# 동반 신뢰도 피처 (반드시 함께 투입)
cw_top1_sim      = S[rows, top[:, -1]]
cw_margin        = S[rows, top[:, -1]] - S[rows, top[:, -2]]
cw_entropy       = -(w * np.log(w.clip(1e-12))).sum(1)     # 후보 분포 불확실성
cw_eff_candidates = np.exp(cw_entropy)                      # 유효 후보 수
```

기대 커버리지: 투수 52.90% → 80% 이상, as-of·500구 게이트 후 행 커버리지 60.24% → 75~85%.

**리스크**: 서로 다른 투수를 섞으면 신호가 희석된다. `cw_entropy`와 `cw_eff_candidates`를 함께 넣어 트리가 신뢰도를 학습하게 하고, `tau`와 `s_min`을 ablation한다.

### 4-2. fingerprint 보강 + 매칭 검증 3종

현행 fingerprint(월·요일·이닝·초말·카운트·타자손)는 팀 일정과 보직에 지배될 수 있다(결함 3). 다음을 **추가 채널**로 넣는다.

| 채널 | 근거 | 계산 |
|---|---|---|
| **경기당 투구 수 분포** | main의 `asof_pitcher_n`은 통산 누적이므로, 학습 데이터 내 같은 투수 행을 정렬해 증분을 내면 등판별 투구 수를 복원할 수 있다. TrackMan은 `trackman_game_id`로 직접 계산된다. **보직·지속력은 팀 일정과 독립인 개인 특성** | 등판별 투구 수의 `[p10, p50, p90, mean, n_outings]` 5차원 코사인 |
| **볼카운트 전이 행렬** | 투수의 카운트 운용은 개인 특성 | 12×12 전이 확률 flatten 후 코사인 |
| **`pitch_of_pa` 분포** | TrackMan에만 있음 — main에는 없어 채널로 쓸 수 없다 | 검증용으로만 |

> `asof_pitcher_n` 증분은 **학습 데이터 내부에서만** 사용한다. crosswalk는 train으로만 만들므로 규칙 위반이 아니다(금지 대상은 test 내부 순서·누적). 단, 이 값을 **모델 입력 피처로 만들면** 팀이 이미 확인한 타깃 복원 누수(99.9463% 행에서 복원 가능)에 걸리므로 절대 금지.

**매칭 검증 3종** (hand 일치만으로는 부족):

1. 보직 일치 — 매칭된 TrackMan 투수의 선발/불펜 비율이 main 투수의 `inning==1` 비율과 일치하는가
2. 등판 수 일치 — TrackMan 등판 수 vs main의 추정 등판 수 (상대오차 20% 이내)
3. 시즌 간 일관성 — 같은 main 투수가 여러 시즌에서 같은 TrackMan ID로 매칭되는가 (현행 419명 중 평균 2.43시즌, 고신뢰 100% 일치 → 이 지표는 이미 양호)

---

## 5. 임베딩 전처리 규약 (결함 4 대응)

### 5-1. 하드 규칙 6개

| # | 규칙 | 근거 |
|---|---|---|
| E1 | **NaN을 0으로 채우지 않는다.** 미가용은 NaN + `has_trackman=0` | 결함 4a: 481,500행(32.64%) 0 벡터 → 시즌 대리변수화 |
| E2 | **TrackMan 가용/미가용을 같은 열에서 섞지 않는다.** 폴백 값을 임베딩 차원에 쓰지 않는다 | 결함 4b: `tm_available` 예측 AUC 0.9983 |
| E3 | **`control_success`를 임베딩 학습 타깃으로 쓰지 않는다** | 결함 4c + 팀의 temporal TE 실패(BSS 361.06) |
| E4 | **fold 간 latent 축을 Procrustes로 정렬한다** | YoY 코사인 ≈ 0, `cohort_embedding` 부호 진동 |
| E5 | **`experience_cohort`처럼 이미 컬럼으로 존재하는 저카디널리티 변수에 임베딩 차원을 쓰지 않는다** | `nn.Embedding(5, dim)` = 8차원 낭비 |
| E6 | **투수-시즌 as-of 산출 + fold별 재적합.** scaler·PCA·커널 σ·인코더 전부 학습 시즌만으로 | 기존 규칙 유지 |

### 5-2. Procrustes 정렬 (E4 구현)

fold별로 인코더를 재학습하면 latent 공간이 임의 직교변환만큼 달라진다. GBDT는 축의 절대 위치로 분할하므로 정렬이 없으면 학습 fold의 규칙이 검증 fold에 전이되지 않는다.

```python
# 기준 fold(가장 이른 fold)의 임베딩을 앵커로 고정
# 공통 투수(anchor 교집합)에 대해 직교 Procrustes 로 회전 행렬을 구해 적용
from scipy.linalg import orthogonal_procrustes
common = anchor.index.intersection(target.index)
Rrot, _ = orthogonal_procrustes(target.loc[common].values, anchor.loc[common].values)
target_aligned = target.values @ Rrot
```

**검증 게이트**: 정렬 후 같은 투수의 연속 시즌 코사인이 **0.60 이상**, 차원별 YoY R² **0.30 이상**이어야 임베딩을 채택한다. Kirby Index(0.50)를 하한 참고선으로 삼는다. 현행 임베딩은 R² 0.0035~0.115로 이 게이트를 전혀 통과하지 못한다.

### 5-3. 무학습 임베딩을 먼저 쓴다

정렬 문제를 근본적으로 회피하는 방법은 **학습을 하지 않는 것**이다. 결정론적 임베딩은 fold와 무관하게 같은 좌표계를 갖는다.

| 방법 | 정렬 필요? | 이유 |
|---|---|---|
| 분위수·모멘트 signature | **불필요** | 좌표계가 물리 단위로 고정 |
| Kernel Mean Embedding (RFF) | **불필요** | `W`, `b`를 시드 고정하면 같은 사상 |
| Sliced-Wasserstein 분위수 | **불필요** | 랜덤 방향을 시드 고정 |
| SVD / NMF | 필요 (부호·순서) | 부호를 첫 성분 기준으로 고정 + Procrustes |
| GMM 파라미터 | 필요 (컴포넌트 순서) | 컴포넌트를 velocity 순으로 정렬 |
| DeepSets / Contrastive / AE | **필수** | 매 학습마다 자유 회전 |

> **순서**: 무학습 → SVD/GMM → 학습 기반. 앞 단계가 lift를 내면 뒤 단계는 그 lift를 넘어야 한다.

### 5-4. 표현 학습에서만 증강을 쓴다 (P0-6 근거)

학습곡선 실측:

| 학습 행 | n | 2024 AUC | 2024 BSS |
|---|---:|---:|---:|
| 12.5% | 152,698 | 0.5388 | 347.1 |
| 25% | 305,396 | 0.5423 | 536.5 |
| 50% | 610,792 | 0.5434 | 557.3 |
| **100%** | **1,221,585** | **0.5435** | **567.0** |
| 2022~2023만 | 492,997 | 0.5420 | 518.5 |

50% → 100%에서 **AUC +0.0001, BSS +9.7**. 완전히 포화했다.

> **결론: 타깃 모델에 대한 데이터 증강은 무의미하다.** 표본 수는 병목이 아니다. mixup·SMOTE·pseudo-labeling에 시간을 쓸 이유가 없다 (게다가 pseudo-labeling은 "평가 데이터 전체를 보고 만든 사후 보정값" 금지 조항에 걸린다).

**단 하나의 예외가 표현 학습이다.** TrackMan 표현 학습에서는 투수당 표본이 실제 병목이고 좌투수는 211명뿐이다. 여기서만 증강이 정당하다.

| 증강 | 방법 | 용도 |
|---|---|---|
| **좌우 미러** | §3-1의 부호 반전 + 타자손 스왑 | 좌투수 표현 학습 표본을 사실상 3.8배 (211 → 807 상당) |
| **서브셋 샘플링** | 같은 투수-시즌에서 128구 부분집합을 여러 개 추출 | contrastive positive pair 생성, 학습 표본 수십 배 |
| **측정 노이즈 주입** | 각 물리량에 시즌×구종군 IQR의 5~10% 가우시안 | contrastive augmentation, 측정 노이즈에 대한 불변성 학습 |
| **등판 드롭아웃** | 등판 단위로 랜덤 제외 후 프로파일 재계산 | 프로파일 추정 분산에 대한 강건성 |

미러 증강은 (좌투, 좌타) 셀에서 특히 값을 한다 — P0-6 실측으로 이 셀이 가장 어렵다.

| 손 조합 | n | 비중 | 성공률 |
|---|---:|---:|---:|
| 좌투–좌타 | 170,292 | 11.54% | **0.4909** (최저) |
| 좌투–우타 | 211,059 | 14.31% | 0.5375 |
| 우투–좌타 | 525,455 | 35.62% | 0.5307 |
| 우투–우타 | 568,286 | 38.53% | 0.5221 |

팀 문서의 "2024 좌투수–좌타자 28,390행 오차 +0.56%p"와 일치한다.

---

## 6. 표준 전처리 순서 (요약)

```
[원자료]  trackman_history.csv 1,793,078 x 30
   │
   ├─ 1. 이상치 → NaN / 구조 이상 행 제거        (§2-1)   -99행
   ├─ 2. pitch_type_group 4종으로 축약, other 병합/제외  (§2-2)
   ├─ 3. 좌우 미러 정규화 (horz_break, rel_side 부호)   (§3-1)
   ├─ 4. season x pitch_type_group robust z             (§3-2)  ← fold별 학습 시즌만
   ├─ 5. primary_fastball 결정 → FB 대비 차분           (§2-3)
   │
   ├─ 6. 투수-시즌 단위 집계
   │      ├─ Tier A 릴리스 일관성 (문맥 회귀 잔차 SD)
   │      ├─ Tier B 등판 내 drift (pitch_no slope)
   │      ├─ Tier C 아스널 구조
   │      ├─ Tier D Stuff+ 입력 차분
   │      └─ Tier E 조건부 반응 프로파일  ← 최우선
   ├─ 7. Empirical Bayes 축소 (m0 그리드)              (§3-3)
   ├─ 8. Tier별 PCA/SVD 압축 → 8~16차원
   │
   ├─ 9. soft crosswalk 로 main pitcher_id 에 결합      (§4-1)
   │      + cw_top1_sim / cw_margin / cw_entropy / cw_eff_candidates
   ├─ 10. as-of 게이트: 시즌 S 행은 S 미만 시즌만, 500→300구 완화 실험
   │
   └─ 11. 투입: 절대 수준이 아니라 잔차 형태로만        (§3-4 하드 규칙)
           1순위 correction 평활 계층(M2) / 2순위 residual expert(M3) / 최하 GBDT 원시 피처(M1)
```
