# CSW 예측 프로젝트 — 에이전트 인수인계 문서

> **읽는 대상**: 이 프로젝트를 이어받는 AI 에이전트 / 새 팀원
> **한 줄 요약**: 투구 전 정보만으로 개별 투구의 CSW를 예측. **최고 LogLoss 0.5721 / AUC 0.6378**.
> count-only 기준선(0.5790)은 넘지만 개선폭이 작고, **원인은 모델이 아니라 정보의 부재**(공의 위치를 모름)로 진단됨.

---

## 0. 즉시 파악해야 할 3가지

1. **과제의 천장이 낮다.** 현재 투구의 위치(`plate_x/z`)를 넣으면 AUC 0.613→0.739인데, 투구 전 피처 311개 전부로는 0.613→0.638. **신호의 대부분은 "그 공이 어디로 갔는가"에 있고 정의상 예측 시점에 알 수 없다.**
2. **이진 판정기로는 무가치, 확률 추정기로는 유효.** 정확도 0.7172 = "항상 아니다"(0.7171)와 동급, F1 0.470 vs "전부 CSW"(0.441). 그러나 **calibration은 거의 완벽**(예측 15%→실제 15%, 46%→45%).
3. **비율(CSW%) 예측에서는 단순 이동평균에 패배.** 경기 단위 R²: 이동평균 0.244 vs 우리 모델 0.075. 투구 단위 확률을 평균내면 투수 실력 신호가 희석되기 때문.

---

## 1. 데이터셋

| 항목 | 값 |
|---|---|
| 원천 | `../data/statcast_2017_2019_raw_csw.parquet` (Statcast, pybaseball 수집) |
| 규모 | 2,201,095행 × 122열 (정규시즌 필터 후 2,184,169) |
| 기간 | 2017–2019 (TrackMan 시대. 2020+ 는 Hawk-Eye로 측정 특성 다름) |
| 라벨 | `is_csw` = `description ∈ {called_strike, swinging_strike, swinging_strike_blocked}` |
| 기저 비율 | 27.1% (2017) / 27.5% (2018) / **28.3% (2019 test)** |
| **분할** | **train 2017–2018 / test 2019 고정** |
| 실험 표본 | **상위 40투수** (train 238,054 / test 94,150) — 샌드박스 RAM 3GB 제약 |

### 반드시 지킬 규칙
- **엄격 투구 전(strict pre-pitch)**: 예측 대상 투구의 물리·위치·릴리스각·구종·결과를 **입력 금지**. `csw_pipeline.BANNED`가 assert로 차단.
- **미래 정보 금지 함정**: `pitcher_days_until_next_game`, `batter_days_until_next_game`(이름이 rest_days와 비슷), `post_*_score`, `delta_*`, `home/bat_win_exp`.
- **전 연도 0% 컬럼**(사용 불가): `umpire`, `arm_angle`, `bat_speed`, `swing_length`, `spin_dir`, `spin_rate_deprecated` → **주심 정보는 외부(Retrosheet) 조인 없이는 불가**.
- **부분 결측은 정상**: 물리값 ~2%, `release_spin_rate` 3–4%. `launch_speed`·`estimated_woba`는 25%만(타구에만 존재).
- **누수 안전 패턴**: 정렬 → `shift(1)` → 창 집계. target encoding도 expanding.
- **평가는 prequential/adaptive**: 2019 진행 중 과거 2019 결과로 이력 피처는 갱신, **모델 파라미터는 2017–18 고정**.

---

## 2. 코드 구조

```
0723/
├── src/
│   ├── csw_pipeline.py    # 로드·라벨·기본/투수 피처·누수집합·모델zoo·평가·베이스라인
│   ├── pp_features.py     # PitchPredict 차용 + 타자이력 + 워크로드
│   └── plotstyle.py       # 한글 폰트 (Noto Sans CJK) — 그래프 그릴 때 반드시 import
├── prep_features.py       # 1단계: 기본+투수 피처 → cache/features.parquet (270열)
├── prep_g.py              # 2단계: +타자이력 +워크로드 → cache/features_g.parquet (324열)
├── run_cv.py              # A/B: 경기그룹 K-fold, 전체 vs 투수별
├── run_pitchtype.py       # C: 구종 예측 stacking
├── run_def.py             # D/E/F: EDA·ablation·CS/W 분해
├── run_optuna.py          # E: LightGBM Optuna 40 trials (SQLite 이어달리기)
├── run_g.py / run_h.py    # G/H: 타자·워크로드 격리 측정
├── run_final.py           # FULL 결합 (single/decomp/combine 블록)
├── run_i.py               # I: 계열별 Optuna 6종 + 앙상블
├── diagnose_ceiling.py    # 천장 진단 (위치·구위 추가 시 성능)
├── diagnose_rate.py       # 비율 예측 진단
├── report_f1.py / report_acc.py   # F1·정확도·calibration 보고
├── build_*.py             # 노트북 생성기
├── A~I *.ipynb            # 결과 노트북 (실행 완료, 그래프 포함)
└── out/                   # v2·v3·final·i·tabpfn — 지표 JSON·PNG·CSV
```

### 재현 순서
```bash
python prep_features.py 40      # 15~25s
python prep_g.py                # 15s
python run_final.py single && python run_final.py decomp && python run_final.py combine
python run_i.py lgbm 10   # 계열별. rf/et는 5, 나머지 10
python run_i.py combine
python build_i.py               # 노트북 재생성
```

**샌드박스 제약(중요)**: bash 호출당 **45초 제한**, RAM **3GB**. 그래서 (a) 무거운 계산은 스크립트로 분리해 **블록 체크포인트**, (b) Optuna는 **SQLite로 이어달리기**(`/tmp/optuna_*.db`), (c) 전처리 2단계 분리, (d) 학습은 서브샘플(N=45000). **백그라운드 프로세스는 호출 간 유지되지 않음.**
`out/`·`results/` 의 기존 파일은 **삭제·덮어쓰기 불가**(마운트 권한) → 새 결과는 새 디렉터리에 저장.

---

## 3. 피처셋 (FULL = 311개)

| 그룹 | 개수 | 내용 |
|---|---|---|
| `situation` | ~25 | 카운트·주자·점수차·이닝·좌우·구장·수비배치 |
| `basic_history` | ~10 | 나이·타순바퀴·휴식일·경기내 투구수·직전구종(lag) |
| `ids` | 2 | 투수/타자 expanding target encoding (수축 포함) |
| `pitcher_hist` | 126 | 6지표(csw·whiff·called·zone·chase·초구스트라이크) × **7창**(day·2w·szn·pszn·**car**·**l100**·**l500**) × (비율+분모+결측) |
| `arsenal` | ~30 | 구종별 평균구속(FF/SI/SL/CH)·패스트볼/브레이킹 사용률 × 4창 |
| `release_rep` | 10 | 릴리스 각도(VRA/HRA) 과거 SD — **"command"가 아니라 반복성 지표** |
| `batter_scout` | 30 | 구종 카테고리(fb/br/off)별 타자 chase·whiff·콜스트라이크허용·스윙률 |
| `batter_hist` | 42 | 4지표 × 4창(szn·car·pszn·l200) + **투수 좌우 스플릿** |
| `gameflow` | 17 | 직전 3구 존/스윙/CSW, 최근 5·15구 카테고리 비율·스트라이크율 |
| `prior_ab` / `lineup` | 5 | 삼진·볼넷·안타·홈런 직후 플래그, 타순 슬롯 |
| `workload` | 12 | 오늘 투구수, 평소 경기당 평균, **평소 대비 비율/초과/z-score**, 시즌 누적 |

핵심 설계: **지표별 분모 분리**(whiff=스윙수, called=테이크수, chase=존밖수) + **지표별 Beta-Binomial 수축**(λ: csw 200·whiff 150·called 150·zone 200·chase 120·fps 100) + **표본수·결측 지시자** 동반.

---

## 4. A~I 각 라운드

| 라운드 | 무엇을 했나 | 결과 (TEST 2019 LogLoss) | 배운 것 |
|---|---|---|---|
| **A** | 투수별 개별 모델(+임계 미달 폴백). GroupKFold(game_pk) | 0.5865 (XGB) | 완전분리는 **전체 모델보다 열등**. 부분 풀링이 대안 |
| **B** | 전체 단일 모델, 4계열 비교 | **0.5760** (HistGB) | 부스팅 우세. 로지스틱도 근접 → 대부분 선형으로 설명 |
| **C** | 구종 예측(7클래스, top-1 42.7%) → 확률을 피처로 stacking | CV↓ TEST↑ | **효과 없음**. 구종 예측 정확도 부족 |
| **D** | EDA·천장 진단·비율 진단 | — | ★ **핵심 라운드**. 위치=AUC 0.739 / 우리=0.638. 비율 예측은 이동평균에 패배 |
| **E** | PitchPredict 차용 피처 +137, Optuna 40 trials | 0.5734 | 피처↑ → AUC↑ / LogLoss는 정체(과적합 경향) |
| **F** | **CS/W 분해**: `P(swing)·P(whiff\|swing) + P(take)·P(called\|take)` | **0.5722** | ★ **구조 변경이 가장 효과적**. 두 메커니즘 분리가 유효 |
| **G** | 타자 정보 확장(73개) + 격리 측정 | **0.5720** | 순기여 −0.0022, 중요도 31%. **누적 ablation으론 안 보임 → 격리 필요** |
| **H** | 워크로드/피로 12개 | 순기여 −0.0008 | 평소 1.25배↑ 투구 시 CSW 0.21, 1.5배↑ 0.15 (신호는 뚜렷하나 표본 0.2%) |
| **I** | **계열 6종 각각 Optuna** + 앙상블 | **0.5721 / AUC 0.6378** | 계열 교체 이득 미미(최고↔최하 0.0076). 계열 앙상블 무효, **분해 결합만 유효** |

### I 라운드 계열별 결과
| 계열 | TEST LogLoss | AUC | ECE |
|---|---|---|---|
| lgbm | **0.5730** | 0.6345 | 0.0042 |
| hgb | 0.5736 | 0.6329 | 0.0032 |
| xgb | 0.5736 | 0.6337 | 0.0072 |
| et | 0.5760 | 0.6278 | 0.0096 |
| rf | 0.5765 | 0.6255 | 0.0130 |
| logreg | 0.5806 | 0.6198 | 0.0211 |

**최종 채택**: `lgbm + CS/W분해` 가중결합(w=0.3) → **LogLoss 0.5721 / AUC 0.6378 / ECE 0.0121**

---

## 5. 기준선 (항상 이것과 비교할 것)

| 기준선 | LogLoss | 비고 |
|---|---|---|
| 리그평균 상수 | 0.5957 | |
| **count-only** `P(CSW\|balls,strikes,stand,p_throws)` | **0.5790** | ★ **실질 기준선** |
| 투수 target encoding | 0.5947 | |
| 우리 최고 | **0.5721** | count-only 대비 −0.0069 |

이진 지표: 정확도 "항상 아니다" 0.7171 / 우리 0.7172 · F1 "전부 CSW" 0.4410 / 우리 0.4701

---

## 6. 시도했으나 효과 없던 것 (반복 금지)
- **투수별 완전분리 모델** — LightGBM 0.61~0.64로 크게 열등 (단 TabPFN은 0.576으로 양호)
- **구종 예측 stacking (C)** — CV만 개선, TEST 악화
- **계열 앙상블(top-k 평균)** — 단일 최고보다 나쁨
- **피처 대량 추가** — E에서 137개 추가해도 LogLoss 정체
- **경기그룹 번갈아 K-fold를 모델 선택에 사용** — 내삽 편향으로 순위 왜곡(XGB가 CV 2위→TEST 4위). **시간순 분할 권장**

## 7. 남은 유망 방향
1. **목표 전환 ①: xCSW(투구 품질 평가)** — 현재 투구 위치·구위를 *의도적으로* 입력. AUC 0.756 확인됨. 예측이 아니라 **"이 투수의 공이 얼마나 좋은가"** 를 운·수비 보정해 평가하는 지표(Stuff+ 계열). 첫 설계서의 권장안.
2. **목표 전환 ②: 경기 단위 CSW% 직접 모델링** — 투구 확률 평균이 아니라 비율을 직접. **목표는 명확: 이동평균 R² 0.244 이기기**.
3. **부분 풀링** — 전체 모델 + 투수별 오프셋/캘리브레이션 (완전분리보다 안정적)
4. **분해 모델 개별 튜닝** — 현재 세 모델이 파라미터를 공유. `P(swing)`·`P(whiff|swing)`·`P(called|take)`는 난이도가 달라 따로 튜닝할 가치 있음
5. **확률 보정** — isotonic/Platt. 분해 결합의 ECE 0.0121은 단일(0.0060)보다 나쁨
6. **TabPFN** — 로컬 GPU(RTX 4080)에서 검증됨. 투수별에서 LightGBM 대폭 상회(0.576 vs 0.61~0.64). `05_tabpfn_compare.ipynb`, 라이선스 토큰 필요
7. **표본 확대** — 현재 상위 40투수. 로컬(RAM 충분)에서 `python prep_features.py 0`으로 전체 가능

## 8. 미완료
- `run_cv.py`의 `per_pitcher_hgb`, `per_pitcher_logreg` (샌드박스 계산 제한)
- 지표별 λ 그리드 탐색(현재 고정값), 투수별 학습곡선(임계 500~5000)
- frozen vs adaptive 이중 평가

---

## 9. 참고자료 (검토 완료)
- **BaseballIQ** (github.com/ivanrivasgr/baseballiq) — 구조만 참고. **목표 누수 있음**: `models/train.py`가 현재 경기 피처로 같은 경기 CSW를 예측(`shift(-1)` 없음). 15개 피처 중 누수 없는 건 3개뿐. TimeSeriesSplit은 행 내부 누수를 못 막음
- **PyMC-Labs BART** — whiff 모델 피처(구속·무브·회전·릴리스·구장 랜덤효과), calibration 우선 철학
- **CalledStrike**(bayesball) — taken pitch만으로 `logit P = s(plate_x, plate_z)` GAM
- **Plate-discipline BART** (arXiv 2305.05752) — 3단계 분해의 이론적 근거 (F 라운드의 출처)
- **Baseball Scouting Lab** — 모델 분리 철학, AUC+Brier 병기. 일부 랜덤분할이라 미래 예측 검증엔 부적합
- **Pitch Predict** (Josh Mancuso, Analytics Vidhya 3부작) — 타자 스카우팅 리포트·게임플로우·직전타석 플래그의 출처 (E 라운드)
- **Kirby Index** (FanGraphs) — 릴리스 각도 반복성. 원 지표명은 command지만 실제론 릴리스 산포라 **재명명해 사용**

## 10. 산출물 위치
- 노트북: `A_perpitcher` `B_global` `C_pitchtype` `D_eda` `E_features_tuning` `F_cs_w_decomposition` `G_batter_features` `I_model_structures` `05_tabpfn_compare` (전부 실행 완료, 한글 그래프)
- 종합 그림: `out/v3/FINAL_SUMMARY.png` (6패널)
- 지표: `out/v2/`(천장·비율 진단) `out/v3/`(EDA·ablation·Optuna·F·G·H·F1·정확도·calibration) `out/final/`(FULL) `out/i/`(계열별)
- 기존 정리 MD: `ABC_RESULTS.md` `DEF_RESULTS.md` `RERUN_TOP60.md` `RESULTS_AND_REPORT_PLAN.md` `csw_feature_catalog.md`
