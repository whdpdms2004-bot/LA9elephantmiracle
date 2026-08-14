# TrackMan 기반 중간단계 피처 카탈로그

작성: 2026-08-12 / 목적: `trackman_history.csv`(2019~2024, 1,793,078행 × 30열)에서 현재 미사용 상태인 신호를 **중간단계 피처(intermediate feature)**로 뽑아내는 후보를 전부 열거하고, 각각의 계산식·as-of 규칙·커버리지·예상 효과·리스크를 고정한다.

현재 투입 중인 72개 피처는 8개 물리량의 `{mean, std} × {latest, EWM, between-season std}` + 구종군 비율뿐이다(→ `00_ASSESSMENT.md` §6-1). 아래는 그 밖의 축이다.

---

## 0. 설계 제약 (모든 피처에 공통 적용)

### 0-1. main 테이블에 없는 것 — 피처 설계를 제한하는 결정적 사실

`train.csv` / `test.csv`에는 **경기 식별자, 경기 날짜, 경기 내 투구 번호가 없다.** 있는 것은 `season`, `game_month`, `game_dayofweek`, `inning`, `top_bottom`, 카운트, `asof_pitcher_n`(통산 누적 투구 수), `asof_pitcher_prev{1,3,5}_game_*`뿐이다.

또한 평가 규칙상 **test 행끼리의 순서·누적을 쓸 수 없다.** 따라서:

- ❌ main 행에 "이 경기 몇 번째 투구인지"를 붙일 수 없다 (test에서 계산 불가)
- ❌ main 행에 "직전 등판 후 며칠 쉬었는지"를 붙일 수 없다 (날짜 없음)
- ✅ **TrackMan에서 투수별 "경기 내 반응 곡선"을 뽑아 상수로 붙이고, main의 `inning`과 상호작용시키는 것은 가능하다**

이 마지막 항목이 이 카탈로그의 핵심 아이디어다. TrackMan에는 `trackman_game_id`, `pitch_no`, `game_date`가 있으므로 **등판 내 시간 구조를 투수 수준의 파라미터로 압축**할 수 있고, 그 파라미터는 main 행에 투수-시즌 lookup으로 안전하게 결합된다.

### 0-2. as-of / 누수 규칙

| 규칙 | 내용 |
|---|---|
| 시즌 cutoff | 시즌 S의 main 행은 **S보다 이전 시즌** TrackMan만 사용 (2019 행은 전부 미사용) |
| fold별 재적합 | crosswalk, scaler, PCA, 군집중심, 임베딩 인코더 전부 fold마다 재학습. F24 검증 시 2024 TrackMan은 crosswalk 재적합에도 금지 |
| 표본 게이트 | 현행 투수-시즌 500구 이상. 완화 실험은 §6 |
| 2025 | TrackMan 미제공 → 2019~2024 전체로 만든 frozen lookup 사용 |
| 결측 정책 | 절대 impute하지 않는다. NaN 유지 + `has_trackman` + crosswalk 신뢰도(`cw_mean_sim`, `cw_min_margin`) 동반 투입 |

### 0-3. 2022 측정 체계 단절 대응 (필수)

동일 투수 145명 기준 2021→2022: `extension` −0.148(98.6%가 감소), `IVB` −3.079, fastball 비중 −12.08%p. **원시 절대값을 시즌 간에 비교하는 모든 피처는 무효로 간주한다.**

모든 물리량은 다음으로 변환한 뒤 집계한다.

```python
# season x pitch_type_group 셀 안에서 robust z
key = ['season', 'pitch_type_group']
med = tm.groupby(key)[col].transform('median')
iqr = tm.groupby(key)[col].transform(lambda s: s.quantile(.75) - s.quantile(.25))
tm[col + '_rz'] = (tm[col] - med) / (iqr.clip(lower=1e-6) / 1.349)
```

절대값 버전과 rz 버전을 **동시에 투입하지 말고** ablation으로 하나만 남긴다(차원 절약이 이 데이터에서 반복적으로 이득이었다).

### 0-4. 표본수 축소 (empirical Bayes)

SD·비율 추정은 표본수에 따라 리그 평균으로 축소한다. 팀이 이미 smoothing 200 부근이 최적임을 확인했으므로 동일 철학을 적용한다.

```python
# 비율: beta-binomial shrinkage
p_hat = (k + a0) / (n + a0 + b0)          # a0, b0는 리그 분포에서 method-of-moments
# SD: 분산을 chi-square 축소
var_shrunk = (n * var_hat + m0 * var_league) / (n + m0)     # m0 = 200 기준으로 그리드
```

---

## 1. Tier A — 릴리스 일관성 (Kirby Index 축소판)

**왜 최우선인가**: 이 대회의 타깃은 "의도한 코스에 넣었는가"다. 이론적으로 가장 직접적인 물리 상관은 **릴리스 반복성**이다. TrackMan에 `rel_height`, `rel_side`, `extension`이 있으므로 릴리스 위치 축은 계산 가능하다.

**한계를 미리 고정**: 원본 Kirby Index는 vx0~az 운동학에서 역산한 **수직/수평 릴리스 각도(VRA/HRA)**를 쓰고, RF 중요도가 VRA 0.385 / HRA 0.306 / rel_x 0.190 / rel_z 0.119로 **각도가 지배적**이다. 우리 데이터에는 각도 원시값이 없다 → 중요도 하위 31%만 쓰는 축소판이다. 또한 MLB 344명 연구에서 릴리스 변동성(95% 신뢰타원)은 xFIP(R²=0.207)와는 유의했지만 **BB/9와는 거의 무관**했다. `control_success`는 BB 계열에 가깝다 → **효과 기대치를 낮게 잡고 시작한다.**

| ID | 피처 | 계산 |
|---|---|---|
| A1 | `rel_sd_{h,s,ext}__{fastball,breaking,offspeed}` | 투수×시즌×구종군 SD, EB 축소 |
| A2 | `rel_ellipse_area__g` | `2×2 cov(rel_height, rel_side)`의 `sqrt(det)` → 95% 타원 반경 |
| A3 | `rel_resid_sd__g` | `rel_* ~ f(balls, strikes, outs, inning, pitch_of_pa, batter_hand)` OLS 후 **잔차 SD** = 상황 통제 후 순수 반복성 |
| A4 | `rel_iqr_ratio__g` | `IQR / (p99−p01)` → 꼬리 두께 (돌발적 릴리스 이탈 빈도) |
| A5 | `rel_arsenal_spread` | 구종군 릴리스 중심 간 유클리드 거리 평균/최대 → 구종별로 릴리스가 갈리는가 (터널링 대용, 위치 불필요) |
| A6 | `rel_sd_by_hand_gap` | 좌타 상대 SD − 우타 상대 SD → 특정 타자 손에서 무너지는가 |

**A3이 이 Tier의 핵심이다.** 단순 SD는 "다양한 상황에서 던졌다"와 "제구가 흔들린다"를 구분하지 못한다. 잔차 SD는 구분한다.

**예상 효과**: 중. **비용**: 0.5일. **리스크**: 각도 부재로 약화, BB 계열 상관 약함.

---

## 2. Tier B — 등판 내 drift (신규 축, 가장 유망)

**아이디어**: main에는 `inning`이 있지만 "이 투수가 이닝이 갈수록 얼마나 무너지는가"는 없다. TrackMan의 `pitch_no` × `trackman_game_id`로 그 **민감도 계수**를 투수-시즌 상수로 뽑아 붙이고, `inning`·`asof_pitcher_n`과 상호작용시킨다. 인게임 구속 감소가 피로 신호라는 것은 확립된 사실이다.

| ID | 피처 | 계산 |
|---|---|---|
| B1 | `drift_slope_{rel_speed, extension, rel_height}` | 등판별 `y ~ pitch_no` OLS slope → 투수-시즌 중앙값 |
| B2 | `drift_sd_ratio` | `SD(후반 50%) / SD(전반 50%)` (pitch_no 중앙값 기준) → 후반 제구 붕괴율 |
| B3 | `drift_mix_shift` | 후반 fastball 비중 − 전반 fastball 비중 → 지치면 구종을 바꾸는가 |
| B4 | `outing_len_p50 / p90` | 등판별 투구 수 분포 → 선발/불펜 역할 및 지속력 |
| B5 | `drift_slope_x_inning` | **B1 × main의 `inning`** 상호작용 항 (또는 GBDT에 둘 다 주고 트리가 만들게) |
| B6 | `rest_response` | TrackMan `game_date` 차분으로 휴식일별 구속·SD 반응 → 투수별 계수. main에 날짜가 없으므로 **`game_dayofweek`·`game_month`와만 간접 결합** (약함, 후순위) |

**예상 효과**: 중상. main에 없는 정보를 순수 추가한다. **비용**: 0.5일. **리스크**: B1은 등판당 표본이 작아 slope 분산이 큼 → 등판 25구 이상만 사용 + EB 축소. B6은 결합 경로가 약해 우선순위 낮음.

---

## 3. Tier C — 아스널 구조

| ID | 피처 | 계산 | 근거 수치 |
|---|---|---|---|
| C1 | `velo_sep_fb_off` | `mean(FB rel_speed) − mean(offspeed rel_speed)` | FB 142.70 vs offspeed 130.47 |
| C2 | `velo_sep_fb_brk` | FB − breaking | breaking 127.53 |
| C3 | `move_dist_{fb_brk, fb_off, brk_off}` | (IVB, HB) 평면 구종군 중심 간 마할라노비스 거리 | FB (41.73, 14.08) / brk (0.396, −5.324) / off (20.72, 19.26) |
| C4 | `spin_gap_fb_off` | FB spin − offspeed spin | 2245.84 vs **1708.29** (가장 큰 대비) |
| C5 | `arsenal_entropy` | `H = −Σ pᵍ log₂ pᵍ`, `Gini = 1 − Σ pᵍ²` (구종군 4종) | — |
| C6 | `n_effective_pitches` | 사용률 5% 이상 구종군 수 | — |
| C7 | `arsenal_silhouette` | 구종군을 라벨로 본 물리공간 silhouette → 구종 경계 선명도 | — |
| C8 | `tag_auto_disagree_rate` | `tagged_pitch_type` vs `auto_pitch_type` 불일치율 (전체 문자열 일치 55.1%, 17종 vs 11종) | **거의 무료 + 독창적** |
| C9 | `speed_retention` | `zone_speed / rel_speed` → 항력·회전효율 프록시 | 125.92 / 137.41 |

**C8 주의**: 명명 규약 차이가 섞여 있으므로 반드시 `pitch_type_group` 매핑 후 그룹 수준 불일치만 센다. 그리고 연도별 태깅 운영 변화가 교란하므로 `season` 내 z로 정규화한다.

**예상 효과**: 중. C4·C9·C8이 가장 저비용. **비용**: 0.5일. **리스크**: 엔트로피 계열은 "전략 다양성"이지 커맨드가 아니므로 직결성이 약하다.

---

## 4. Tier D — Stuff+ 입력 세트 차용

공개된 Stuff+ 구현체(aStuff+)의 입력은 velo, IVB, HB, spin rate, spin axis, extension, 수직/수평 릴리스 위치 + **패스트볼 대비 velo/IVB/HB 차분 3개**이며, **plate location을 쓰지 않는다.** 750구 기준 YoY R² 0.702로 안정적이다.

우리는 **run value 타깃이 TrackMan에 없으므로 Stuff+ 점수 자체는 만들 수 없다.** 대신 **입력 피처 세트만 차용**한다. 특히 FB 대비 차분 3종(= C1~C3의 일반화)이 실질 payoff다.

| ID | 피처 | 계산 |
|---|---|---|
| D1 | `d_velo__g`, `d_ivb__g`, `d_hb__g` | 각 구종군 − 해당 투수 주 패스트볼 |
| D2 | `approx_vaa_proxy` | `zone_speed`, `IVB`, `extension`으로 근사 접근각. 낙하량 ≈ f(비행시간, IVB) → `atan2` 근사 |
| D3 | `spin_efficiency_proxy` | `sqrt(IVB² + HB²) / spin_rate` → 회전 대비 실제 무브먼트 |
| D4 | `release_consistency_x_stuff` | Tier A의 반복성 × D3 → "좋은 공을 반복해서 던지는가" |

**절대 하지 말 것**: `control_success`나 main의 `asof_*`를 타깃으로 Stuff+ 유사 회귀를 학습하는 것. 그 순간 학습된 target encoding이 되고, 팀은 이미 이 함정에서 BSS 361.06을 기록했다.

**예상 효과**: 중상 (D1). **비용**: 0.5일. **리스크**: D2는 각도 부재로 근사 오차가 크다 → ablation에서 빠지면 미련 없이 버린다.

---

## 5. Tier E — 조건부 반응 프로파일 (main 상황 변수와 직접 맞물리는 축)

TrackMan에 `balls_before`, `strikes_before`, `outs_before`, `pitch_of_pa`, `inning`, `batter_hand`가 있다. main도 같은 변수를 갖는다. → **투수별 "상황에 따라 무엇이 어떻게 변하는가"를 프로파일로 만들면 main 행의 상황 변수와 직접 결합된다.**

이것은 현재 72개 피처가 전혀 담지 못하는 축이다. 72개는 모두 상황 무관 marginal 통계다.

| ID | 피처 | 계산 |
|---|---|---|
| E1 | `mix_by_count` | 투수 × 볼카운트(12셀) × 구종군(4) 사용률 → 48차원 (→ §7 SVD로 압축) |
| E2 | `velo_by_count_slope` | `rel_speed_rz ~ (balls − strikes)` 계수 → 불리한 카운트에서 힘을 주는가 |
| E3 | `sd_by_pressure` | `3-2, 3-0, 만루` 상황의 릴리스 잔차 SD − 전체 SD → 압박 상황 붕괴도 (2024 R 최대 잔차 구간이 count 3-0 −1.80%p, 3-2 +1.29%p, 0-1 −1.12%p) |
| E4 | `mix_by_hand_gap` | 좌/우 타자별 구종 사용률 차 → 상성 correction의 입력으로 직결 |
| E5 | `first_pitch_profile` | `pitch_of_pa == 1` 서브셋의 구종·구속·SD → 초구 전략 |

**E3이 특히 매력적이다.** `R_FOCUS_RECHECK.md`가 지목한 2024 R 최대 잔차 구간이 정확히 볼카운트 극단이고, 현재는 그 구간을 사후 보정으로만 다룬다. **투수별 압박 취약도**를 사전 피처로 넣으면 사후 보정 없이 그 잔차를 설명할 여지가 있다.

**예상 효과**: 상. **비용**: 1일. **리스크**: 셀 희소성 → 12×4 셀에 EB 축소 필수, 그리고 원시 48차원 투입은 금지(§7로 압축).

---

## 6. Tier F — 커버리지 확대 (같은 피처의 효과를 1.3배로)

현행 커버리지: crosswalk 419/792명(52.90%), as-of + 500구 게이트 후 **2024 행 60.24%**.

| ID | 방안 | 기대 |
|---|---|---|
| F1 | 500구 게이트 → 300 / 200 / 100구 완화 + `log1p(n)`·EB 축소 동반 | 커버리지 60% → 75~85%. 팀이 smoothing 200에서 최적을 찾았듯 게이트도 완화가 이길 가능성이 있다 |
| F2 | **soft crosswalk** — cosine 임계 0.80 hard 1:1 매칭을 버리고, 상위 k 후보의 similarity-softmax 가중 평균으로 임베딩·통계를 구성 | 매칭 실패 47%를 "약한 증거"로 회수. 매칭 오류가 평균화되어 오히려 강건 |
| F3 | crosswalk 임계 0.80 → 0.70 + `cw_mean_sim`을 신뢰도 가중으로 사용 | F2의 저비용 버전 |
| F4 | 다중 시즌 합산 허용 (현행 금지) — 500구 미만 시즌들을 가중 합산, `season_gap` 페널티와 함께 | 2019·2020 커버리지(0% / 50.33%) 회복 |
| F5 | 팀 수준 fallback — 투수 매칭 실패 시 소속팀 평균 프로파일 (초기 팀 fingerprint crosswalk는 무효였지만, **투수 매칭 결과를 팀으로 집계**하는 역방향은 유효) | cold-start 투수 커버 |

**F2가 이 Tier의 핵심이다.** hard 매칭은 정보를 버린다. cosine 0.79로 탈락한 후보도 0.79만큼의 증거다.

**예상 효과**: 중상 (기존 피처의 이득을 그대로 1.25~1.4배). **비용**: 1일. **리스크**: soft 결합은 서로 다른 투수를 섞으므로 신호가 희석될 수 있다 → `cw_mean_sim` 가중과 `cw_entropy`(후보 분포의 불확실성)를 함께 투입해 트리가 신뢰도를 학습하게 한다.

---

## 7. 차원 관리 규칙 (반드시 지킬 것)

팀의 실증: **구종별 TrackMan 전개(차원 확장)는 760.71~764.94로 악화**, compact 요약은 767.03~769.69. 즉 이 데이터에서 차원 증가는 거의 항상 손해다.

따라서 위 Tier A~E에서 나오는 원시 피처는 **150~300개 규모**가 되는데, 그대로 투입하면 실패가 예정되어 있다. 압축 규칙:

1. **Tier 단위로 PCA/SVD 압축** — Tier별 8~16차원으로 줄인 뒤 투입. E1(48차원)은 특히 반드시 SVD.
2. **Tier 단위 ablation** — 한 Tier씩 추가하며 채택 기준(§8) 통과분만 유지. 전부 넣고 selection하는 방식은 금지.
3. **최종 목표는 기존 211개 피처에 +8~24개**. 그 이상이면 압축 실패로 본다.
4. **주입 방식은 GBDT 원시 피처가 1순위가 아니다** — 상세는 `02_EMBEDDING_METHODS.md` §3.

---

## 8. 채택 기준 (모든 후보 공통)

`00_ASSESSMENT.md` §4~5의 진단을 반영한 게이트다.

| 게이트 | 조건 |
|---|---|
| G1 하드 | **R-only Val2023 ΔBrier < 0 AND R-only Val2024 ΔBrier < 0** (전체 점수로 대체 불가) |
| G2 하드 | 전체 ΔBrier가 **paired 부트스트랩 95% CI에서 0을 포함하지 않음** |
| G3 하드 | F 집단 기여율이 60%를 넘지 않음 (F 편중 후보는 다양성 슬롯으로만) |
| G4 소프트 | **AUC 증가** — 이 프로젝트에서 처음 요구하는 기준. 보정 이득이 아니라 판별력 이득인지 구분한다 |
| G5 소프트 | TrackMan 가용 행 / 미가용 행 각각에서 악화 없음 (선택 편향 확인) |
| G6 운영 | 245,789행 추론 8분·RAM 24GB 이내, 오프라인, lookup만으로 재현 |

---

## 9. 우선순위 요약

| 순위 | 항목 | 예상 효과 | 비용 | 근거 |
|---:|---|---|---|---|
| 1 | **Tier E 조건부 반응 프로파일** (E1·E3·E4) | 상 | 1일 | 현재 완전 미탐색 + 2024 R 최대 잔차 구간과 직결 |
| 2 | **Tier F2 soft crosswalk** | 중상 | 1일 | 모든 TrackMan 피처의 이득을 1.25~1.4배 |
| 3 | **Tier B 등판 내 drift** (B1·B2·B5) | 중상 | 0.5일 | main에 없는 정보 순수 추가 |
| 4 | **Tier D1 FB 대비 차분** | 중상 | 0.5일 | Stuff+ 검증된 입력, YoY R² 0.702 |
| 5 | **0-3 시즌×구종 robust z 전환** | 중 (기존 72개 피처의 품질 개선) | 0.3일 | 2022 단절 미반영 상태 |
| 6 | **Tier A3 상황 통제 잔차 SD** | 중 | 0.5일 | Kirby 축소판, 이론적 직결성 최고 |
| 7 | **Tier F1 게이트 완화** | 중 | 0.3일 | 커버리지 60% → 80% |
| 8 | **Tier C4·C8·C9** | 중~낮 | 0.3일 | 저비용·독창 |
| 9 | Tier A2·A4·A5·A6 | 낮~중 | 0.3일 | Tier A 확장 |
| 10 | Tier C5·C6·C7, D2·D3, B6 | 낮 | 0.5일 | 여유 있을 때 |
