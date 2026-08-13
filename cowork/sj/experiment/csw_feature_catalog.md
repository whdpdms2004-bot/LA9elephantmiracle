# CSW 예측 — 파생 변수 카탈로그 (Feature Dictionary)

> 목적: 4개 노트북(전체/투수별 × 기본/파생)에서 쓸 **입력 변수 전체 목록과 생성법**.
> 계획 수립 전 단계 — 이 목록을 확정한 뒤 노트북 설계 md를 작성한다.
> 목표(label): `is_csw = description ∈ {called_strike, swinging_strike, swinging_strike_blocked}` (팀 확정 정의).
> 분할: **입력 2017–2018 / 테스트 2019.**

---

## 0. 절대 규칙 — 예측 시점 (모든 변수 공통)

> **"예측할 그 투구"에서 나온 값은 절대 입력하지 않는다.** 릴리스 각도·구속·회전·무브먼트·`plate_x/z`·`zone`·`sz_top/bot`·구종(pitch_type)·투구 결과 전부 금지.
> **허용**: (1) 투구 전 이미 확정된 경기 상황, (2) 그 투수/타자/포수의 **과거 투구**에서 계산한 값(항상 `shift(1)` 이후 집계).

이것은 팀의 기존 `train_csw_model.py` 규칙(`CURRENT_PITCH_LEAKAGE`, `STRICT_PREPITCH_DROP`)과 동일하다. 아래 모든 파생 변수는 이 규칙을 만족하도록 계산법을 명시한다.

> 참고: 앞서 만든 **xCSW(투구 단위)** 모델은 정반대로 현재 투구 물리·위치를 *입력으로* 쓴다. 이 카탈로그는 그 모델이 아니라 **투구 전 CSW 예측 모델**용이다. (혼동 금지)

---

## 1. 4개 시간 창(window) 정의 — 파생 변수의 핵심 축

투수(및 타자)의 과거 피칭을 아래 4개 창으로 나누고, 각 창에서 동일한 기초 지표들을 집계한다. 창은 **중첩(nested)** 되며 서로 다른 평활 스케일을 준다(문제 없음, 오히려 다중 스케일 신호).

| 창 | 코드 | 범위 (현재 투구 = 투수 p, 날짜 d, 경기 g) | 잡아내는 것 | 계산 근거 |
|---|---|---|---|---|
| **당일(이번 등판)** | `_day` | 같은 경기 `game_pk`에서 **현재 투구 이전** 공들 | 이 등판의 컨디션·구속 저하·이미 보여준 패턴 | 경기 내 `shift(1).expanding()` |
| **최근 2주** | `_2w` | `[d-14, d-1]` (당일 제외) | 최근 폼·부상 조짐·구속 추세 | 일자 집계 후 `rolling("14D", closed="left")` |
| **이번 시즌** | `_szn` | 시즌 시작 ~ `d-1` (당일 제외) | 시즌 누적 실력·안정된 성향 | 시즌 내 일자 `cumsum().shift(1)` |
| **지난 시즌** | `_pszn` | 직전 시즌 **전체** | 사전(prior) 실력·시즌 간 변화 기준선 | 시즌별 집계 후 `season→season+1` 매핑 |

- 2017은 `_pszn` 없음 → 결측(NaN). 트리 모델은 NaN 네이티브 처리(HistGradientBoosting) 또는 리그 평균으로 대체.
- 각 창은 **표본 수 컬럼**(`*_n_day`, `*_n_2w`, …)을 함께 만들어 신뢰도(min_periods)와 수축(shrinkage)에 사용.
- 더블헤더: `_day`는 `game_pk` 기준(= 이번 등판)으로 정의해 날짜 혼선을 피한다.

**기초 지표 × 창** 조합으로 변수가 생성된다. 예: `p_csw_rate_day`, `p_csw_rate_2w`, `p_csw_rate_szn`, `p_csw_rate_pszn`.

---

## 2. 변수 분류 체계 (기본 vs 파생)

| Tier | 구분 | 노트북 배정 | 설명 |
|---|---|---|---|
| **0** | 직접(raw) | 기본 + 파생 | 원본에 그대로 있고 투구 전 확정된 열 |
| **1** | 행 내 단순 변환 | 기본 + 파생 | 현재 상황만으로 즉시 계산(이력 불필요) |
| **2** | 파생(이력·창) | **파생 전용** | 과거 투구/타석에서 창별 집계·lag·수축 필요 |

기본 노트북 = **Tier 0 + Tier 1** 만. 파생 노트북 = **Tier 0 + 1 + 2** 전부.

---

## 3. Tier 0 — 직접(raw) 변수 [기본]

모두 투구 전 확정. 원본 열 그대로.

| 변수 | 원본 열 | 비고 |
|---|---|---|
| 투수 ID | `pitcher` | 전체 모델에선 인코딩 대상(§8), 투수별 모델에선 그룹 키 |
| 타자 ID | `batter` | 인코딩/타자 피처 조인 키 |
| 포수 ID | `fielder_2` | 프레이밍 피처 조인 키 |
| 구장 | `home_team` | 파크 효과 프록시 |
| 타자 좌우 | `stand` | |
| 투수 좌우 | `p_throws` | |
| 볼 / 스트라이크 | `balls`, `strikes` | |
| 아웃 | `outs_when_up` | |
| 이닝 / 초말 | `inning`, `inning_topbot` | |
| 주자 | `on_1b`, `on_2b`, `on_3b` | 존재여부(0/1)로 변환은 Tier 1 |
| 점수 | `home_score`, `away_score`, `bat_score`, `fld_score` | |
| 수비 배치 | `if_fielding_alignment`, `of_fielding_alignment` | 투구 전 확정 → 허용 |

> ⚠️ `sz_top`, `sz_bot`, `zone`, `plate_x/z`, 물리·릴리스 열은 Tier 0이지만 **현재 투구 측정치라 입력 금지**. 오직 과거 투구 집계(Tier 2)로만 사용.

---

## 4. Tier 1 — 행 내 단순 변환 [기본]

이력 없이 현재 상황만으로 계산. (대부분 기존 `build_statcast_strike_dataset.py`에 이미 구현됨.)

| 변수 | 정의 / 계산 |
|---|---|
| `score_diff_bat` | `bat_score - fld_score` |
| `base_state` (0–7) | `on1 + 2*on2 + 4*on3` (존재=1) |
| `runner_count`, `risp`, `bases_loaded` | 주자 수 / 2·3루 여부 / 만루 |
| `count_state` | `"{balls}-{strikes}"` 범주 |
| `two_strike`, `full_count` | `strikes==2` / `(3,2)` |
| `pitcher_ahead`, `batter_ahead` | 카운트 유불리 (기준 명시) |
| `matchup`, `same_handed_matchup` | `p_throws×stand` / 동일 손 여부 |
| `late_inning`, `extra_inning` | `inning>=7` / `>=10` |
| `close_game` | `|score_diff_bat| <= 2` |

---

## 5. Tier 2 — 파생 변수 (이력·창 기반) [파생 전용]

> 공통 계산 원칙: **정렬 → `shift(1)` → 창 집계**. 아래 각 항목의 "만드는 법"을 그대로 따르면 누수가 없다.
> 표기: `p_`=투수, `b_`=타자, `c_`=포수, `park_`=구장. 창 접미사 `_day/_2w/_szn/_pszn`.

### A. 투수 결과 성향 (Rate) — 창별
**기초 지표(각 창에서 과거 공들의 평균):**

| 기초 지표 | 정의(과거 공 기준) |
|---|---|
| `p_csw_rate` | CSW 비율 = mean(`is_csw`) |
| `p_called_rate` | 지켜본 공 중 콜스트라이크 = called/taken |
| `p_whiff_rate` | 스윙 중 헛스윙 = whiff/swing |
| `p_swstr_rate` | 전체 대비 헛스윙 = whiff/pitches |
| `p_zone_rate` | 존 통과율 = mean(zone∈1..9) |
| `p_chase_rate` | 존 밖 스윙 유도 = 존밖스윙/존밖 |
| `p_strike_rate` | 스트라이크(type=S) 비율 |
| `p_first_pitch_strike_rate` | 초구(strikes==0&balls==0) 스트라이크율 |
| `p_contact_rate` | 스윙 중 컨택 = 1 − whiff/swing |

→ 변수 = {지표}×{`_day`,`_2w`,`_szn`,`_pszn`} (예: `p_csw_rate_2w`).

**만드는 법**
- `_day`: `df.groupby(["game_pk","pitcher"])[metric].transform(lambda s: s.shift(1).expanding().mean())`
- `_2w`, `_szn`: 아래 §7의 "일자-창 패턴" (분자=이벤트 합, 분모=투구 수 합, 나눗셈).
- `_pszn`: §7의 "지난 시즌 매핑".
- 표본 적을 때 리그 평균 μ로 **수축**: `(n·x̄ + λμ)/(n+λ)` (λ≈100~300, 검증으로 튜닝).

근거: BaseballIQ의 rolling 성향 피처(단, 여기선 누수 없이 재정의), Baseball Scouting Lab의 결과 지표.

### B. 투수 아스널 — 구속·무브·회전·릴리스 (창별)
**기초 지표(과거 공의 평균 및 표준편차):** `release_speed`, `release_spin_rate`, `pfx_x`, `pfx_z`, `release_pos_x`, `release_pos_z`, `release_extension`.

| 변수 예 | 의미 |
|---|---|
| `p_velo_mean_{win}`, `p_velo_sd_{win}` | 평균 구속 / 구속 안정성 |
| `p_spin_mean_{win}` | 평균 회전수 |
| `p_pfxx_mean_{win}`, `p_pfxz_mean_{win}` | 평균 수평/수직 무브먼트 |
| `p_relx_mean_{win}`, `p_relz_mean_{win}` | 평균 릴리스 위치 |
| `p_release_dispersion_{win}` | √(SD_relx² + SD_relz²) 릴리스 일관성 |
| `p_velo_trend_day_vs_szn` | `p_velo_mean_day − p_velo_mean_szn` (등판 중 구속 저하=피로) |
| `p_velo_trend_2w_vs_pszn` | `p_velo_mean_2w − p_velo_mean_pszn` (작년 대비 구위 변화) |

**만드는 법**: 물리 열은 과거 공의 값 → §7 일자-창 평균/표준편차. 현재 공 물리값은 절대 미포함(`shift(1)` 필수).
근거: PyMC-BART whiff 모델의 핵심 피처(구속·수직/수평무브·회전), Baseball Scouting Lab(릴리스 위치).

### C. 투수 커맨드 (Kirby Index 계열, 창별)
릴리스 각도(VRA/HRA)의 **과거 표준편차**(작을수록 반복성↑). 현재 공 각도는 금지, 과거 포심 각도만.

| 변수 | 정의 | 만드는 법 |
|---|---|---|
| `p_cmd_vra_sd_{win}`, `p_cmd_hra_sd_{win}` | 과거 FF 릴리스 각도 SD | `kirby_index.add_release_angles()`로 과거 각도 → §7 창별 std |
| `p_cmd_dispersion_{win}` | √(vra_sd² + hra_sd²) | 위 둘 결합 |
| `p_kirby_index_pszn` | 지난 시즌 Kirby Index | `kirby_index.kirby_index_table()` (시즌×투수) → `_pszn` 매핑 |

근거: 팀의 `kirby_index.py`(`add_prepitch_ff_command_features`), FanGraphs Kirby Index. 이미 구현된 `cmd_ff_*`(rolling 40구)를 창 기반으로 확장.

### D. 구종 사용률 & 다양성 (창별, +상황별)
**기초:** 구종별 사용 비율(FF/SI/FC/SL/CU/CH/FS/KC/ST), 구종 엔트로피.

| 변수 예 | 의미 |
|---|---|
| `p_usage_FF_{win}` … `p_usage_ST_{win}` | 구종별 사용률 |
| `p_pitch_entropy_{win}` | `−Σ pₖ·log pₖ` (다양성/예측불가성) |
| `p_usage_FF_vs_R_{win}`, `p_usage_FF_vs_L_{win}` | 타자 손별 사용률 |
| `p_usage_breaking_2strk_{win}` | 2스트라이크 결정구(브레이킹) 사용률 |

**만드는 법**: 구종 원-핫 → §7 일자-창 평균(=사용률). 상황별은 그룹 키에 `stand` 또는 `two_strike` 추가. 엔트로피는 창별 사용률 벡터로 계산.
근거: BaseballIQ `stuff_diversity`(누수 없이 재정의), 기존 `*_usage_last_20`.

### E. 시퀀싱 / Lag (직전 공, 같은 등판)
현재 공 직전에 이미 일어난 것 → 허용.

| 변수 | 정의 |
|---|---|
| `prev_pitch_type_1/2/3` | 직전 1~3구 구종 |
| `prev_description` | 직전 공 결과(스트라이크/볼/파울 등) |
| `prev_release_speed`, `prev_pfx_x/z`, `prev_plate_x/z` | 직전 공 물리/위치 (**직전** 공은 과거 → 허용) |
| `prev_vaa_approx`, `prev_haa_approx` | 직전 공 접근각(근사) |
| `count_transition` | 직전 카운트→현재 카운트 전이 |
| `same_pitch_as_prev` | (예측 대상이 아니므로 사용 불가 — 현재 구종 필요) ❌ |

**만드는 법**: `df.groupby(["game_pk","pitcher"])[col].shift(1..3)`. (기존 구현 존재.)
근거: Baseball Scouting Lab(직전 투구의 구종·위치, 구종 간 속도·무브 차이).

### F. 타자 규율 성향 (창별)
타자의 과거 스윙 판단 성향. 투수 CSW에 직접 영향.

| 변수 | 정의(타자 과거 기준) |
|---|---|
| `b_chase_rate_{win}` | 존 밖 스윙률 |
| `b_zone_swing_rate_{win}` | 존 안 스윙률 |
| `b_whiff_rate_{win}` | 스윙 대비 헛스윙률 |
| `b_contact_rate_{win}` | 컨택률 |
| `b_called_strike_take_rate_{win}` | 지켜본 공 콜스트라이크 허용률 |
| `b_csw_against_rate_{win}` | 상대로 허용한 CSW율 |
| `b_k_rate_{win}`, `b_bb_rate_{win}` | 삼진율/볼넷율(타석 단위) |

**만드는 법**: 그룹 키 `batter`로 §7 일자-창. 타석 단위 지표(K/BB)는 PA 집계 후 창. `_day`는 이번 경기 이전 타석.
근거: 권장 아키텍처의 "상대 타선 chase·contact 성향", plate-discipline BART.

### G. 매치업 / 타순
| 변수 | 정의 / 만드는 법 |
|---|---|
| `times_faced_in_game` | 이번 경기 n번째 대결 (기존 구현) |
| `is_first_time_facing` | 이번 경기 첫 대결 여부 |
| `p_csw_rate_vs_stand_{win}` | 투수의 해당 타자 손 상대 과거 CSW율 (그룹 키 +`stand`) |
| `pair_csw_rate_pszn` | 이 투수–타자 조합 과거 CSW율(희소→수축) |

근거: 타순 효과(times through order), 좌우 스플릿.

### H. 워크로드 / 피로
| 변수 | 정의 / 만드는 법 |
|---|---|
| `pitcher_pitch_count_before` | 이번 등판 현재까지 투구 수 (기존) |
| `prev_game_pitch_count` | 직전 등판 총 투구 수 (경기 집계 후 `shift(1)`) |
| `rest_days` | 현재 등판일 − 직전 등판일 − 1 (기존) |
| `p_pitches_last_2w` | 최근 2주 총 투구 수(부하) = `_2w` 표본 수 재사용 |
| `is_starter` | 등판 시작 이닝/투구수 패턴으로 선발/불펜 추정(선택) |

근거: 휴식일·직전 부하(권장 아키텍처 입력 변수).

### I. 경기 중요도 (Leverage) — **train 전용 산출**
| 변수 | 정의 / 만드는 법 |
|---|---|
| `li_pa_shrunk`, `leverage_class` | 상황별 |ΔWE| 평균 정규화(기존 `_add_leak_safe_li`). **2019 테스트엔 2017–18로 만든 표만 조인** |

⚠️ `delta_home_win_exp`/`delta_run_exp` 자체는 현재 결과 → 직접 입력 금지, LI 표 산출에만.

### J. 포수 프레이밍
| 변수 | 정의 / 만드는 법 |
|---|---|
| `c_called_strike_take_rate_{win}` | 포수의 지켜본 공 콜스트라이크율 (그룹 키 `fielder_2`) |
| `c_framing_above_expected_pszn` | (고급) 위치 기대 콜확률 대비 초과 콜율. 기대콜은 앞서 만든 **xCSW 모델 C**로 계산 → 잔차 평균 |

⚠️ 테스트 연도 신규 포수는 결측 → 리그 평균 대체.
근거: called-strike 모델의 포수 변수(참고자료 5).

### K. 구장 (Park)
| 변수 | 정의 / 만드는 법 |
|---|---|
| `park_called_strike_rate_train` | `home_team`별 과거(train) 콜스트라이크율 |
| `park_csw_rate_train` | 구장별 CSW율 (train 전용, 테스트에 매핑) |

⚠️ **train 연도로만 산출** 후 test에 조인(랜덤분할 금지 원칙과 동일).
근거: PyMC-BART의 park random effect.

### (선택) L. 주심 (외부 조인 필요)
Statcast 기본 컬럼에 주심 ID 없음. Retrosheet 등 외부 조인 시 `ump_called_strike_rate_train`(train 전용) 추가 가능. 미조인 시 생략.

---

## 6. 기존 코드에 이미 있는가 (재사용 매핑)

| 카테고리 | 이미 구현(재사용) | 새로 만들 것 |
|---|---|---|
| Tier 0/1 | 전부 (`build_features`) | — |
| A 결과성향 | `pitcher_strike_rate_before/_last20` | `_day/_2w/_szn/_pszn` 창 + csw/called/whiff/zone/chase/first-pitch |
| B 아스널 | `*_last20_mean`, `release_dispersion_last20` | 창 기반 평균·SD, velo trend |
| C 커맨드 | `cmd_ff_*`(rolling 40), `kirby_index_table` | 창 기반 SD, `p_kirby_index_pszn` |
| D 구종사용 | `*_usage_last_20` (9종) | 창 기반 + 엔트로피 + 상황별 |
| E 시퀀싱 | `prev_pitch_type_1..3`, `prev_*`, lag | (거의 완비) |
| F 타자규율 | `batter_csw_rate_before/_last100`(train_csw_model) | 창 기반 chase/whiff/zone/K/BB |
| G 매치업 | `times_faced_in_game` | 좌우 스플릿, pair, first-time |
| H 워크로드 | `pitcher_pitch_count_before`, `prev_game_pitch_count`, `rest_days` | `p_pitches_last_2w` 등 |
| I 레버리지 | `li_pa_shrunk`, `leverage_class` | (완비) |
| J 포수 | — | `c_*` 프레이밍 |
| K 구장 | — | `park_*` (train 전용) |

→ 파생 노트북은 **기존 함수 재사용 + 4-창 확장 모듈** 신규 작성 조합.

---

## 7. 계산 공통 패턴 (pandas) — "만드는 법" 레퍼런스

**(0) 정렬 (항상 먼저)**
```python
df = df.sort_values(["game_date","game_pk","at_bat_number","pitch_number"], kind="stable")
```

**(1) 당일(이번 등판) `_day` — 경기 내 shift+expanding**
```python
g = df.groupby(["game_pk","pitcher"], sort=False, group_keys=False)
df["p_csw_rate_day"] = g["is_csw"].transform(lambda s: s.shift(1).expanding().mean())
df["p_velo_mean_day"] = g["release_speed"].transform(lambda s: s.shift(1).expanding().mean())
```

**(2) 일자-창 `_2w`, `_szn` — 투수×일자 집계 후 시간 롤링**
```python
daily = (df.groupby(["pitcher","game_year","game_date"])
           .agg(n=("is_csw","size"), csw=("is_csw","sum"),
                velo_sum=("release_speed","sum"))
           .reset_index().sort_values(["pitcher","game_date"]))

def per_pitcher(gp):
    gp = gp.set_index("game_date").sort_index()
    # 최근 2주 (당일 제외): closed="left"
    w2 = gp[["n","csw","velo_sum"]].rolling("14D", closed="left").sum()
    # 이번 시즌 누적 (당일 제외): 시즌별 cumsum 후 shift
    szn = (gp.groupby(gp["game_year"])[["n","csw","velo_sum"]]
             .cumsum().groupby(gp["game_year"]).shift(1))
    out = gp[["game_year"]].copy()
    out["p_csw_rate_2w"]   = w2["csw"]  / w2["n"]
    out["p_velo_mean_2w"]  = w2["velo_sum"] / w2["n"]
    out["p_csw_rate_szn"]  = szn["csw"] / szn["n"]
    out["p_velo_mean_szn"] = szn["velo_sum"] / szn["n"]
    out["p_n_2w"], out["p_n_szn"] = w2["n"], szn["n"]
    return out.reset_index()

feat = daily.groupby("pitcher", group_keys=True).apply(per_pitcher).reset_index()
feat = feat[[c for c in feat.columns if not c.startswith("level_")]]  # 잔여 인덱스 열 제거
df = df.merge(feat, on=["pitcher","game_date","game_year"], how="left")
```
> 표준편차 창은 합 대신 `sum(x)`, `sum(x²)`, `n`을 모아 `sd=√(Σx²/n − (Σx/n)²)`로 계산(롤링에서 직접 std가 어려울 때).

**(3) 지난 시즌 `_pszn` — 시즌 집계 후 season+1 매핑**
```python
szn_tot = (df.groupby(["pitcher","game_year"])
             .agg(n=("is_csw","size"), csw=("is_csw","sum")).reset_index())
szn_tot["p_csw_rate_pszn"] = szn_tot["csw"] / szn_tot["n"]
szn_tot["game_year"] = szn_tot["game_year"] + 1          # 다음 시즌에 붙임
df = df.merge(szn_tot[["pitcher","game_year","p_csw_rate_pszn"]],
              on=["pitcher","game_year"], how="left")
```

**(4) 수축(shrinkage) — 표본 적은 창 안정화**
```python
lam, mu = 200.0, df.loc[train_mask, "is_csw"].mean()
df["p_csw_rate_2w_sh"] = (df["p_n_2w"]*df["p_csw_rate_2w"] + lam*mu) / (df["p_n_2w"] + lam)
```

**(5) train 전용 집단통계(구장/포수/LI) — test 조인**
```python
park = (df[train_mask].groupby("home_team")["is_csw"].mean().rename("park_csw_rate_train"))
df = df.merge(park, on="home_team", how="left")   # 2019에도 2017–18 값만 매핑
```

**(6) 누수 안전망 (학습 직전 assert)**
```python
BANNED = CURRENT_PITCH_LEAKAGE | STRICT_PREPITCH_DROP  # 현재 물리/위치/구종/결과
assert not (BANNED & set(feature_cols)), "누수 열 잔존!"
```

---

## 8. 두 모델 설계에서의 변수 취급

| | ① 전체 한 모델 (투수=ID) | ② 투수별 모델 |
|---|---|---|
| 투수 식별 | `pitcher`를 **인코딩**: target/frequency encoding(**train 전용**) 또는 임베딩. 원-핫은 고카디널리티라 비권장 | 그룹 키 — **피처로 넣지 않음** |
| 투수 성향 피처(A~D) | 그대로 사용(투수 간 차이를 모델이 학습) | 사용하되 **투수 내 변동**만 의미 있음(당일/2주 추세가 핵심) |
| `_pszn`, `_szn` | 유효 | 데이터 적은 투수는 창이 자주 결측 → 수축·min_periods 중요 |
| 표본 | 많음(전 투수) | 투수별 소량 → **최소 투구 수 기준**(예: train 1,500+) 미만은 전체모델로 폴백 |
| 기대 | 안정·일반화 강함 | 개인 특화 가능하나 과적합·소표본 위험 |

> 투수별 모델에서도 §1의 4-창 피처를 **반드시 포함**(사용자 요구). 다만 투수별 모델은 "그 투수 내부의 시간 변화"를 학습하므로 `_day`·`_2w` 추세 변수의 가치가 상대적으로 커진다.

---

## 9. 기본 노트북용 최소 변수 셋 (요약)

Tier 0 + Tier 1 전부 + 시퀀싱 lag(E의 `prev_*`) + 워크로드 3종(`pitcher_pitch_count_before`, `rest_days`, `prev_game_pitch_count`) + `times_faced_in_game` + IDs(전체 모델은 인코딩, 투수별 모델은 제외). → **이력 창(A~D,F,J,K) 없이** 시작.

파생 노트북 = 위 + Tier 2 전체(A~K).

---

## 10. 다음 단계 (예고)

이 카탈로그 확정 후, **4개 노트북 설계·계획 md**를 작성한다:
1. 전체 한 모델 · 기본 정보만 2. 전체 한 모델 · 파생 3. 투수별 모델 · 기본 4. 투수별 모델 · 파생

각 노트북 공통: 로드(2017–18 train / 2019 test) → 피처 구성(위 셋) → 여러 모델(로지스틱/HGB/XGB/LGBM/(선택)CatBoost) 비교 → 최고 성능 선별 → 평가(ROC-AUC·PR-AUC·LogLoss·Brier·Calibration/ECE) + Baseline(리그평균·투수 shrinkage·전년도 CSW) 대비.

> **검토 요청**: 위 변수 목록에서 추가/제외할 것, λ·창 경계(2주=14일?), 투수별 모델 최소 투구 수 기준을 알려주면 반영해 계획 md로 확정하겠습니다.
