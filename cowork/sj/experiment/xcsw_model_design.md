# CSW% 예측 설계서 — 투구 단위 xCSW 분해 모델 (2안)

> 작성: 2026-07-23 · 참고자료 검토 + BaseballIQ 누수 분석 + 기존 파이프라인 연계 설계
> 범위: 참고자료가 권장하는 **2안 (투구 단위 xCSW)** 만 다룬다. 1안(다음 경기 CSW)은 향후 확장으로만 언급.
> 전제: 팀은 이미 **엄격한 투구 전(strict pre-pitch) `is_csw` 분류 모델**을 완성했다(`train_csw_model.py`, `build_statcast_strike_dataset.py`, `kirby_index.py`, status.md). 이 문서는 그것을 대체하지 않고 **보완**하는 새 모델을 설계한다.

---

## 0. 한 문장 요약

> **xCSW 모델은 "실제로 던져진 그 공(구속·무브먼트·위치)이 주어졌을 때 CSW가 될 확률"을 추정한다.** 이는 팀의 기존 pre-pitch 모델("공을 던지기 전에 CSW가 될 확률")과 **질문이 정반대**이며, 따라서 **누수 규칙도 정반대**다 — pre-pitch 모델이 금지하는 현재 투구의 물리·위치 값이, xCSW 모델에서는 **핵심 입력**이고 대신 **결과(description·events·타구질·WPA)만 금지**된다.

이 구분을 흐리면 두 모델 중 하나는 반드시 누수거나 무의미해진다. 문서 전체가 이 경계를 지키는 방법을 다룬다.

---

## 1. 두 모델의 관계 (가장 중요)

| 구분 | 기존: pre-pitch `is_csw` 모델 | 신규: 투구 단위 **xCSW** 모델 (이 문서) |
|---|---|---|
| 질문 | 공을 **던지기 전**, 이 투구가 CSW가 될까? | **던져진 공**(그 구속/무브/위치)이 CSW가 될까? |
| 성격 | 예측(prediction) · 시퀀싱/상황/커맨드 상태 | 기술·기대값(descriptive / expected) · "stuff + location" 품질 |
| 현재 투구 물리값 (구속·무브·회전·릴리스) | ❌ **금지** (릴리스 후에만 알 수 있음) | ✅ **필수 입력** |
| 현재 투구 위치 (`plate_x/z`, `zone`, `sz_top/bot`) | ❌ 금지 | ✅ 필수 입력 (특히 called-strike 모델의 핵심) |
| 현재 구종 `pitch_type` | ❌ 금지 (구종 결정 전) | ✅ 사용 (또는 구종별 층화) |
| 금지되는 것 | 위 물리/위치/구종 + 결과 전부 | **결과만**: `description`, `type`, `events`, `launch_*`, `estimated_woba*`, `delta_run_exp`, `delta_home_win_exp`, post-score |
| 검증 | 연도 분할 (2017–18 / 2019) | 동일 (연도 분할) |
| 대표 선행연구 | (자체 설계) | PyMC-BART whiff, CalledStrike GAM, plate-discipline BART |

**세 가지 연결 고리** (xCSW를 만들면 기존 모델과 이렇게 이어진다):

1. **품질 평가**: 투수·구종의 `xCSW`를 계산해 실제 CSW와 비교 → "운/맥락 보정된 진짜 CSW 실력". stuff 지표(Stuff+, PitchingBot)와 같은 계열.
2. **pre-pitch 모델의 입력**: 투수별 과거 `xCSW`를 **집계·shift** 하여 다음-경기/다음-투구 pre-pitch 모델의 피처로 투입 (누수 없음 — 과거 값만).
3. **다음 경기 CSW 예측(1안)의 재료**: 상대 타선·구장·구종 믹스로 기대 CSW를 조립하는 상향식(bottom-up) 경로 제공.

---

## 2. 참고자료 검토 요약

| # | 자료 | 핵심 기여 | 이 프로젝트에 반영 | 주의 |
|---|---|---|---|---|
| 1 | **BaseballIQ** (GitHub, Medium) | Statcast→메달리온(DuckDB)→XGBoost CSW→SHAP→Streamlit 엔드투엔드 구조 | 폴더 구조·수집·집계·대시보드 **구조만** 참고 | 모델 코드에 **심각한 목표 누수** (3장) — 성능·피처 설계는 반면교사 |
| 2 | **PyMC-Labs BART** (2026-02) | 스윙한 공의 **헛스윙 확률**을 BART로. baseline=구속·수직/수평 무브·회전, enhanced=릴리스(높이·좌우·익스텐션)+axis diff+platoon+구장 랜덤효과. 상관 ≈0.85, calibration 우선 | **모델 B (whiff\|swing)** 의 피처·검증(calibration/WAIC) 설계 근거 | CSW 전체가 아니라 whiff 성분만. 결과를 그대로 성능 근거로 쓰지 말고 **연도 holdout 재평가** |
| 3 | **CalledStrike** (R, bayesball) | taken pitch만으로 `logit(P(called strike))=s(plate_x, plate_z)` 이항 GAM, 존 확률 표면·heatmap | **모델 C (called strike\|take)** 의 뼈대. Python은 `pygam`/LightGBM/HGB/PyMC-BART로 재구현 | 위치 외 카운트·좌우·구종·`sz_top/bot`·주심 확장 |
| 4 | **Plate-discipline BART** (arXiv 2305.05752, Yee & Deshpande) | 3단계 분해: (i) P(called strike\|take) (ii) P(contact\|swing) (iii) 결과별 기대득점. 선수·주심·아웃·카운트·주자·점수차 맥락 | **분해 구조의 이론적 근거**. 우리는 A/B/C 3모델로 변형 | 논문은 스윙 의사결정 평가가 목적. 우리는 CSW 조립이 목적 (기대득점 파트는 선택) |
| 5 | **Baseball Scouting Lab** | Stuff/Location/Decision/xContact/xDamage/Called-strike를 **분리 모델**로. called-strike는 taken만, `plate_x/z`+카운트+좌우+VAA/HAA. AUC·Brier 병기 | 모델 분리 철학·평가지표(AUC+Brier) | 일부 모델이 **랜덤 분할** → 미래 예측 검증으로 쓰지 말 것. 반드시 **연도/날짜 분할** |
| 6 | **pybaseball** (jldbc) | Savant Statcast 투구 단위 수집 표준 도구 | 이미 사용 중 (2017–19, 22청크, 2.2M행 수집 완료) | Savant 403/요청제한 → 이미 월별 분할·캐싱·스킵·검증으로 대응 완료 |

세부 방법론은 각 자료에서 확인했으며, **모델 B는 (2)**, **모델 C는 (3)(5)**, **분해 골격은 (4)** 를 따른다.

---

## 3. BaseballIQ 데이터 누수 분석 (코드 직접 검토 결과)

### 3.1 판정

**Case 1 — 현재 경기 피처 → 현재 경기 CSW.** 심각한 목표 누수(target leakage). `models/train.py`에는 목표를 다음 경기로 옮기는 `shift(-1)`이 **없다**. README는 "Next start projected CSW"라고 홍보하지만 코드의 실제 목표는 **같은 행의 `csw_rate`** 다 (문서·코드 불일치).

### 3.2 근거 (코드 인용)

`models/train.py`:

```python
TARGET = "csw_rate"                 # pitcher_game_summary의 "그 경기" CSW율
FEATURE_COLS = [ ... "zone_rate", "chase_rate", "whiff_rate_delta",
                 "barrel_rate_allowed", "avg_xwoba_allowed", "stuff_diversity",
                 "velo_vs_30d_avg", "avg_spin", "avg_h_break", ... "total_pitches" ]
```

`pipeline/silver/feature_engineering.py` — 위 피처 대부분이 **`csv_rate`와 같은 `game_agg` CTE**, 즉 **동일 투수-경기**에서 계산된다:

```sql
SUM(... 'swinging_strike' ...) / SUM(... '%swing%' ...)           AS whiff_rate,
SUM(... IN ('called_strike','swinging_strike') ...) / COUNT(*)    AS csw_rate,  -- 목표
... zone_rate, chase_rate, barrel_rate_allowed, avg_xwoba_allowed, stuff_diversity ...
-- whiff_rate_delta = 현재 경기 whiff_rate - 30일 평균
```

결정적으로 **`csw_rate = (called_strike + swinging_strike)/pitches`** 이고 **`whiff_rate = swinging_strike/swings`** 이므로, 피처 `whiff_rate_delta`는 목표의 분자 `swinging_strike`를 **그대로 포함**한다 → 거의 순환 참조.

### 3.3 피처별 누수 판정 (15개)

| 피처 | 계산 시점 | 목표(csw_rate)와의 관계 | 누수 |
|---|---|---|---|
| `rolling_30d_avg_velo` | 과거 30일 (RANGE … 1 DAY PRECEDING) | 현재 경기 제외 | ✅ 안전 |
| `rolling_30d_whiff_rate` | 과거 30일 | 현재 경기 제외 | ✅ 안전 |
| `rolling_30d_csw_rate` | 과거 30일 | 현재 경기 제외 | ✅ 안전 |
| `whiff_rate_delta` | 현재 − 과거 30일 | **현재 whiff(=CSW 분자 성분) 포함** | ❌ 심각 |
| `velo_vs_30d_avg` | 현재 − 과거 30일 | 현재 경기 값 포함 | ❌ |
| `zone_rate` | 현재 경기 | 현재 경기 산출물 | ❌ |
| `chase_rate` | 현재 경기 | 현재 경기 산출물 | ❌ |
| `barrel_rate_allowed` | 현재 경기 | 현재 경기 결과 | ❌ |
| `avg_xwoba_allowed` | 현재 경기 | 현재 경기 결과 | ❌ |
| `stuff_diversity` | 현재 경기 pitch mix | 현재 경기 산출물 | ❌ |
| `avg_spin` / `avg_h_break` / `avg_v_break` | 현재 경기 | 현재 경기 물리 평균 | ❌ |
| `total_pitches` | 현재 경기 | 경기 종료 후 확정 | ❌ (사전 예측 관점) |
| `home_away` | `np.random.randint` **임시값** | 무관 | ⚠️ 순수 노이즈 |

**15개 중 누수 없는 피처는 3개(rolling_30d_*)뿐.** 나머지는 현재 경기 산출물이거나 노이즈다.

### 3.4 왜 TimeSeriesSplit이 못 막는가

`TimeSeriesSplit`은 **행(경기) 사이의 시간 순서**만 지킨다 — 학습 fold가 검증 fold보다 미래를 보지 않게 한다. 그러나 **한 행 내부의 피처↔목표 누수**(같은 경기에서 뽑은 값으로 그 경기 목표를 맞힘)는 전혀 막지 못한다. BaseballIQ가 "누수 방지"로 내세운 것이 정작 이 문제를 놓쳤다.

### 3.5 우리 프로젝트 시사점

- 구조(수집·Parquet·집계·SHAP·대시보드)는 참고하되 **피처/목표 시점 설계는 신뢰하지 않는다.**
- 우리 팀의 `build_statcast_strike_dataset.py`는 이미 `CURRENT_PITCH_LEAKAGE` 집합 + `integrity_report`의 `current_pitch_leakage` 검사로 이 부류를 **assert로 차단**하고 있다 — 방향이 정확하다.
- xCSW 모델은 물리·위치를 **의도적으로** 쓰므로, 같은 안전망을 **결과(outcome) 열 전용**으로 다시 정의해야 한다 (5장).

---

## 4. xCSW 분해 모델 설계 (2안)

### 4.1 분해 공식

각 투구 i에 대해:

```text
P(CSW_i) = P(swing_i) · P(whiff_i | swing_i)  +  P(take_i) · P(called_strike_i | take_i)
         = p_swing · p_whiff              +  (1 − p_swing) · p_cs_take
```

투수·경기·구종 단위로 `p_csw_i`를 평균 → **xCSW**.

### 4.2 세 확률 모델

| 모델 | 대상 투구 | 라벨(양성) | 핵심 피처(표면) | 선행연구 | 후보 알고리즘 |
|---|---|---|---|---|---|
| **A. P(swing)** | 전체 투구 | `is_swing` | 위치(`plate_x/z`, `sz_top/bot`), 카운트, 좌우, 구종, 구속·무브(맥락) | plate-discipline BART | HGB(기본) |
| **B. P(whiff\|swing)** | 스윙한 투구만 | `is_whiff` | **구속·수직/수평 무브·회전**, 릴리스(높이·좌우·익스텐션), VAA/HAA·axis, platoon, 구장, 위치 | PyMC-BART (2) | HGB / **PyMC-BART** / LightGBM |
| **C. P(called strike\|take)** | 지켜본 투구만 | `is_called_strike` | **`plate_x`, `plate_z`** (핵심), `sz_top/bot`, 카운트, 좌우, 구종, 주심·포수, VAA/HAA | CalledStrike GAM (3)(5) | HGB / `pygam` / PyMC-BART |

세 모델을 **독립 학습**한 뒤 4.1로 결합. 기본 알고리즘은 팀 스택과 일관되게 `HistGradientBoostingClassifier`(NaN·범주형 네이티브), 정밀 calibration/불확실성이 필요하면 모델 B/C를 **PyMC-BART**로 교체(참고자료 2·4 방식, posterior·credible interval 확보).

### 4.3 라벨 정의 (Statcast `description` 기준)

```python
WHIFF   = {"swinging_strike", "swinging_strike_blocked"}          # 팀 is_csw와 정합
CALLED  = {"called_strike"}
SWING   = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
           "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip"}
# TAKE = not SWING (called_strike, ball, blocked_ball, hit_by_pitch, pitchout, ...)

is_swing         = description ∈ SWING
is_whiff         = description ∈ WHIFF         # (스윙 부분집합)
is_called_strike = description ∈ CALLED        # (take 부분집합)
```

**정합성 검증(필수)**: 모델과 무관하게 데이터에서
`is_csw == (is_swing & is_whiff) | (~is_swing & is_called_strike)` 이 성립해야 한다.
팀의 `is_csw = {called_strike, swinging_strike, swinging_strike_blocked}` 와 위 정의의 유일한 경계는 **`missed_bunt`(스윙+헛스윙이지만 CSW 아님)** 뿐 → 극소수이며, xCSW 결합 시 whiff 성분에서 번트류를 분리하거나 무시 처리(status.md 논의사항 ④와 연결).

### 4.4 집계 (xCSW)

```text
xCSW(투수)        = mean_i p_csw_i            (해당 투수의 모든 투구)
xCSW(투수, 구종)  = 구종별 평균
xCSW(투수, 경기)  = 경기별 평균  → 실제 CSW와 비교, 그리고 pre-pitch 모델 입력용(과거 shift)
```

---

## 5. xCSW 전용 누수 규칙 (기존 pre-pitch 규칙과 반대)

### 5.1 허용/금지 표

| 범주 | 예시 열 | pre-pitch 모델 | **xCSW 모델** |
|---|---|---|---|
| 투구 물리 | `release_speed`, `release_spin_rate`, `pfx_x/z`, `vx0..az`, `release_pos_*`, `release_extension`, `spin_axis`, `effective_speed` | ❌ | ✅ 입력 |
| 투구 위치 | `plate_x`, `plate_z`, `zone`, `sz_top`, `sz_bot` | ❌ | ✅ 입력 (C의 핵심) |
| 현재 구종 | `pitch_type`, `pitch_name` | ❌ | ✅ 입력/층화 |
| 상황 | 카운트·주자·점수차·이닝·좌우·매치업 | ✅ | ✅ |
| **결과 (양쪽 모두 금지)** | `description`, `type`, `events`, `bb_type`, `launch_speed/angle`, `estimated_woba_using_speedangle`, `woba_value`, `delta_run_exp`, `delta_home_win_exp`, post-score | ❌ | ❌ **라벨 산출에만 사용, 피처 금지** |

### 5.2 왜 반대인가

pre-pitch 모델의 누수 정의(`CURRENT_PITCH_LEAKAGE`)는 "릴리스 후에만 알 수 있는 값 전부"다. xCSW는 **바로 그 값들이 입력**이다(그게 "stuff+location"의 정의). 따라서 xCSW의 누수는 오직 **투구 결과** — `description`은 라벨의 원천이므로 피처로 들어가면 100% 자기누수, `events`/`launch_*`/`woba`/`delta_run_exp`는 타구·득점 결과라 CSW와 강상관.

### 5.3 안전망 (코드로 강제)

```python
XCSW_OUTCOME_LEAKAGE = {
    "description", "type", "events", "bb_type",
    "launch_speed", "launch_angle", "hit_distance_sc",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value", "babip_value", "iso_value",
    "delta_run_exp", "delta_home_win_exp", "home_win_exp", "bat_win_exp",
    "post_home_score", "post_away_score", "post_bat_score", "post_fld_score",
    "is_csw", "is_swing", "is_whiff", "is_called_strike",   # 라벨 자신
}
# 학습 직전: assert not (XCSW_OUTCOME_LEAKAGE & set(feature_cols))
```

`zone`은 회색지대다: `plate_x/z`의 이산화라 위치 정보일 뿐이지만, **모델 A/C의 라벨(스윙/콜)과 지나치게 근접**할 수 있으니 `plate_x/z`를 쓰면 `zone`은 중복으로 제외 권장.

---

## 6. 기존 파이프라인 연계

### 6.1 재사용 (신규 수집 불필요)

- **원천 데이터**: `data/statcast_2017_2019_raw_csw.parquet` (2,201,095행 × 122열, 원본 121열 + `is_csw`). 물리·위치·`description` 전부 포함 → xCSW 피처/라벨 모두 여기서 생성.
- **정렬 키**: `SEQUENCE_COLUMNS = [game_date, game_pk, at_bat_number, pitch_number]`.
- **연도 분할**: `train_years=(2017,2018)`, `test_years=(2019,)` — 그대로.
- **물리 피처 헬퍼**: `kirby_index.add_release_angles()` 로 VRA/HRA 계산 (모델 B/C의 접근각 피처).
- **무결성 철학**: `integrity_report` 패턴을 xCSW 버전(결과 누수 검사)으로 이식.

### 6.2 신규 산출물 (제안)

```text
xcsw_pitch_model.ipynb        # 이 문서와 함께 제공하는 실행 골격
build_xcsw_dataset.py         # (후속) 피처/라벨 빌더 — build_statcast_...와 동일 규약
train_xcsw_model.py           # (후속) A/B/C 학습 + 결합 + 평가
reports/xcsw_model/           # metrics_*.json, calibration_*.csv, xcsw_by_pitcher.csv
```

---

## 7. 검증 전략

### 7.1 분할

랜덤 분할 금지. 연도 기반 holdout:

```text
Train: 2017–2018      Test: 2019 (현 보유 데이터)
→ (대회/추가 수집 시) Val 2024 · Test 2025 · 최종 실시간 2026 로 확장
```

세 모델 각각 train 연도로만 학습하고, LI·주심·포수 등 **집단 통계도 train 연도로만** 산출해 test에 조인(참고자료 5의 랜덤분할 함정 회피).

### 7.2 투구 단위 지표 (모델 A/B/C 및 결합 p_csw)

| 지표 | 목적 |
|---|---|
| Log loss | 확률 정확도(주지표) |
| Brier score | 확률 정확도(참고자료 5 병기) |
| ROC-AUC / PR-AUC | 순위 성능 (양성 희소한 B/C는 PR-AUC 중시) |
| Calibration curve | 예측확률=실제빈도 여부 (참고자료 2가 정확도보다 우선시) |
| Expected Calibration Error (ECE) | calibration 단일 수치 |

### 7.3 집계(경기·투수·구종) 지표

MAE · RMSE · R² · 실제 CSW와 xCSW 상관 · 투수별 calibration · 구종별 calibration.

### 7.4 Baseline (반드시 비교)

| # | Baseline | 정의 |
|---|---|---|
| 1 | 리그 평균 CSW | 전체 상수 (2017–18 ≈ 27.x%) |
| 2 | 투수 직전 30일 CSW | rolling 평균 |
| 3 | 투수 전년도 CSW | 시즌 이월 |
| 4 | 투수별 shrinkage 평균 | `(n·x̄ + λ·μ)/(n+λ)` |

투구 단위 확률에는 **위치만 쓰는 단일 GAM/로지스틱**(C의 최소판)과 **구속·무브만 쓰는 단일 모델**(B의 최소판)도 baseline으로 둔다. **분해 모델이 이들 rolling/단순 baseline을 확실히 이기지 못하면 예측력을 주장하지 않는다.**

---

## 8. 작업 로드맵 (status.md 이어받기)

기존 status.md의 단계 8~9(pre-pitch 모델)은 완료/로컬실행 대기 상태. xCSW는 다음 트랙으로 병행:

1. **라벨 3종 생성 + 정합성 검증** — `is_csw == (swing&whiff)|(take&called)` assert (4.3).
2. **xCSW 피처셋 + 결과-누수 안전망** 구축 (5.3).
3. **모델 A/B/C baseline** 학습 (HGB), 연도 분할.
4. **p_csw 결합 + calibration/ECE** 확인.
5. **투수·구종 xCSW 집계**, 실제 CSW와 상관·MAE.
6. **Baseline 4종 + 단순 단일모델** 대비 우위 검증.
7. (선택) 모델 B/C를 **PyMC-BART**로 교체해 posterior·partial dependence·변수중요도 확보.
8. (확장) xCSW를 **pre-pitch/다음경기(1안)** 모델의 과거 집계 피처로 투입.

`xcsw_pitch_model.ipynb`가 1~6을 합성 데이터로 end-to-end 실행하는 골격을 제공한다(로컬에서 실데이터 parquet로 스위치).

---

## 9. 리스크 · 논의사항

- **번트류 경계**(`missed_bunt`): whiff이지만 팀 CSW 정의엔 없음 → 결합식에서 분리/무시 결정 (status.md ④).
- **주심/포수 ID**: called-strike에 강력하지만 test 연도 신규 심판·이적 포수 → 미관측 처리(스무딩/기타 범주).
- **2017–2019 = TrackMan** 시대: 이후 Hawk-Eye와 측정 특성 상이 → 대회 데이터가 다른 시대면 물리 피처 분포 이동 점검.
- **PyMC-BART 비용**: 2.2M 투구 전량 MCMC는 무겁다 → 층화 샘플/구종별 분리 학습, 또는 HGB로 대량+BART로 정밀검증 이원화.
- **`zone` 중복**: `plate_x/z` 사용 시 제외 (5.3).
- **BaseballIQ의 `home_away` 노이즈 피처**는 반영하지 않는다.

---

### 부록: 참고 링크

- BaseballIQ: https://github.com/ivanrivasgr/baseballiq · 피처: `/pipeline/silver/feature_engineering.py` · 학습: `/models/train.py`
- PyMC-BART whiff 튜토리얼: https://www.pymc-labs.com/blog-posts/bayesian-additive-regression-tree-swinging-strikes · PyMC-BART: https://github.com/pymc-devs/pymc-bart
- CalledStrike: https://github.com/bayesball/CalledStrike · 소개: https://bayesball.github.io/Intro_to_CalledStrike_Package.html
- Plate-discipline BART: https://arxiv.org/abs/2305.05752 (Yee & Deshpande 2023)
- Baseball Scouting Lab: https://baseballscoutinglab.netlify.app/
- pybaseball: https://github.com/jldbc/pybaseball
