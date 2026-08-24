# CSW 예측 — 실행 결과 & 2페이지 보고서 계획

> 대상: `0723/` 4개 노트북(전체/투수별 × 기본/파생) 실행 결과 종합 + 결과 보고서 작성 계획.
> 데이터: `data/statcast_2017_2019_raw_csw.parquet` 중 **상위 40투수**(train 2017–18 = 238,054 · test 2019 = 94,150). 주심(umpire) 미사용.
> ⚠️ 아래 수치는 **속도를 위한 상위 40투수 데모**다. 결론의 방향성은 유효하나, 최종 보고서 숫자는 전체 투수로 재실행 후 확정한다.

---

## A. 무엇을 만들었나 (재현 방법)

```
0723/
├── src/csw_pipeline.py     # 로드·라벨·기본/파생 피처·인코딩·모델·Optuna·SHAP·평가·투수별
├── prep_features.py        # 피처 1회 생성→ cache/features.parquet, cache/meta.json
├── run_experiments.py      # 실험 1개 실행(블록 체크포인트) → results/<exp>/
├── build_notebooks.py      # results/를 로드하는 4개 노트북 생성
├── 01_global_basic.ipynb  02_global_derived.ipynb
├── 03_perpitcher_basic.ipynb  04_perpitcher_derived.ipynb
├── cache/   results/       # 산출물(지표 json, shap.png/csv, ablation)
```
재현:
```bash
python prep_features.py 40                 # 전체로 하려면 인자 크게(예: 1205) 또는 0
python run_experiments.py global_basic
python run_experiments.py global_derived
python run_experiments.py pitcher_basic
python run_experiments.py pitcher_derived
python build_notebooks.py                  # 결과 렌더 노트북 재생성
```

### 반영된 리뷰 필수사항
- **누수+prequential 통합**: 모든 이력/인코딩 = 전체기간 시간정렬 후 `shift(1)` expanding/rolling. train 내부 미래누수 없음 + **2019 = online/adaptive**(과거 2019 결과로 이력 갱신, 모델 파라미터는 2017–18 고정). target encoding(투수/타자)도 expanding.
- **지표별 분자/분모 분리 + 지표별 Beta-Binomial 수축**(csw/whiff/called/zone/chase/fps 각각 λ) + 표본수·결측 지시자.
- **Kirby → release-angle repeatability**(`p_relangle_*`, `p_release_repeatability_*`), FF 표본수·결측 포함. "command" 표현 제거.
- **구종별 아스널**(`p_ff_velo_*`,`p_fastball_velo_*`,`p_fastball_usage_*`,`p_breaking_usage_*`), 물리 n = 해당 측정 non-null.
- Basic / Basic-history / Historical 분리. `is_starter`=수비팀 첫 투수(첫 투구 시점 확정). 분산 clamp, ddof=0. **count-only 베이스라인** + **단계적 ablation**.

---

## B. 현재 결과 요약 (상위 40투수 데모, TEST=2019)

**Baseline (LogLoss)**: 리그평균 0.5957 · **count-only 0.5790** · pitcher_te 0.5947

| 실험 | LogLoss | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|
| 01 전체·기본 (tuned LGBM) | **0.5752** | 0.6288 | 0.3906 | 0.1946 | 0.009 |
| 02 전체·파생 (tuned LGBM) | 0.5761 | 0.6271 | 0.3865 | 0.1950 | 0.011 |
| 03 투수별·기본 (weighted) | 0.6097 | 0.5936 | 0.3534 | 0.2073 | 0.074 |
| 03 참조 전체모델 | 0.5783 | 0.6232 | 0.3855 | — | 0.018 |
| 04 투수별·파생 (weighted) | 0.6415 | 0.5705 | 0.3327 | 0.2179 | 0.108 |
| 04 참조 전체모델 | 0.5790 | 0.6226 | 0.3829 | — | 0.021 |

**SHAP 카테고리 기여(전체·파생, 평균 |SHAP|)**: situation 0.69 ≫ pitcher_hist 0.40 > basic_history 0.22 > arsenal 0.12 ≈ ids 0.11 ≫ release_rep 0.04
**Ablation(LogLoss)**: count_only 0.5793 → +situation 0.583 → … → +release_rep(full) 0.5809 — **거의 평탄**.

### 핵심 발견 (정직하게)
1. **카운트/상황이 지배적 신호.** count-only(0.579)가 이미 매우 강하고, 최고 모델(0.575)의 이득은 작지만 실재하며 리그평균·투수평균은 확실히 이긴다.
2. **파생이 기본을 넘지 못함(이 subset).** 전체·파생(0.576)이 전체·기본(0.575)보다 나아지지 않음 → 현재 파생 구성의 한계. SHAP상 release-angle repeatability 기여 최소.
3. **투수별 완전분리 < 전체모델.** 투수별은 LogLoss·ECE 모두 악화(특히 파생에서 과적합) → 리뷰 지적대로 **부분 풀링**(전체모델 + 투수 ID/이력 + 투수 보정) 권장.

---

## C. 평가 계획 (보고서 전 확정할 것)

1. **전체 투수 재실행**: `prep_features.py`를 전체(또는 train≥N)로. 상위40 결론이 유지되는지 확인.
2. **단계적 ablation(핵심 서사)**: 리그평균 → count-only → count+상황 → basic → +투수이력 → +타자이력 → +아스널 → +시퀀싱 → +포수/구장 → full. 각 단계 LogLoss/Brier/PR-AUC.
3. **지표별 λ 선택**: 2017 학습 / 2018 검증에서 λ∈{25,50,100,200,400} 비교(지표별). 2017–18 재학습 후 2019 최종. (2019는 λ 선택에 미사용.)
4. **투수별 학습곡선**: 임계 {500,1k,2k,3k,5k}구에서 시간검증 → 전체모델을 안정적으로 넘는 지점 탐색. 폴백 비율·가중/개별 성능 병기.
5. **Frozen vs Adaptive**: (a) 2018말 정보 고정 (b) 2019 과거로 이력 갱신 — 두 성능 함께 제시(운영 상황은 b).
6. **보조 분해 모델(분석용)**: `is_called_strike`, `is_swinging_strike`를 별도 학습 → 개선이 포수·take쪽인지 whiff쪽인지 설명. (최종 예측은 단일 `is_csw` 유지.)
7. **calibration**: 신뢰도 곡선 + ECE, 투수별 calibration 점검.

### 후순위/주의 (리뷰 반영)
- `pair_csw_rate_pszn`(투수-타자 조합), `park_csw_rate_train`(홈팀≠구장, 혼합효과), 수비시프트, 상황별 구종사용률 전조합 → 초기 제외, ablation으로만 확인. 넣을 경우 **시간안전 인코딩 + 강한 수축** 필수.
- 포수 framing residual = xCSW(모델 C) 기대콜 대비 잔차 → 별도 트랙(계산량 큼).

---

## D. 2페이지 결과 보고서 구조 (작성 계획)

> 형식: A4 2쪽(국문). 표 2~3개 + 그림 2개. 산출물은 `results/`에서 그대로 인용.

**[1쪽] 문제·데이터·방법**
1. 과제 정의: 투구 전 CSW 예측, 왜 CSW인가(투수 가치 신호), train 2017–18 / test 2019.
2. 데이터: Statcast 2017–19(TrackMan), 투구 단위, 라벨 `is_csw`, 주심 미사용.
3. 예측 시점·누수·**prequential/adaptive** 규칙(핵심 1문단) + 지표별 분모/수축·release repeatability 재명명.
4. 실험 설계 2×2 표(전체/투수별 × 기본/파생), 모델 비교(LogReg/HistGB/LGBM/XGB)+Optuna, 평가지표(LogLoss·Brier·ROC/PR-AUC·ECE)와 **baseline(리그·count-only·투수평균)**.

**[2쪽] 결과·해석·결론**
5. 표1: 4실험 TEST 성능 vs baseline (B절 표 사용).
6. 그림1: SHAP 카테고리 기여 막대(`results/global_derived/shap.png`).
7. 그림2: 단계적 ablation LogLoss 곡선(`ablation.csv`).
8. 핵심 발견 3가지(B절): ① 상황/카운트 지배 ② 파생의 한계 ③ 투수별<전체(부분풀링 권장).
9. 한계·다음 단계: 상위40 데모→전체 재실행, λ/학습곡선/frozen-vs-adaptive, 보조 분해모델, 포수/구장 시간안전 인코딩.

**작성 실행**: 위 구조로 docx 생성(제목·표·그림 임베드) — 요청 시 `docx` 스킬로 산출. 그림은 이미 `results/`에 있으므로 표/문구만 채우면 됨.

---

## E. 다음 액션 체크리스트
- [ ] `prep_features.py`를 전체 투수로 재실행 → 4실험 재산출
- [ ] 단계적 ablation 확장(타자이력·시퀀싱·포수/구장 그룹 추가)
- [ ] 지표별 λ 그리드(2018 검증) 확정
- [ ] 투수별 학습곡선 → 임계·폴백 정책 결정, 부분풀링 프로토타입
- [ ] 보조 분해모델(called/whiff) 기여 분석
- [ ] 2페이지 docx 보고서 생성(표/그림 임베드)
