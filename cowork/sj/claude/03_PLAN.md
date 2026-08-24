# 이후 분석 계획

작성: 2026-08-12 / 작업 디렉토리: `C:\Users\isj67\Desktop\LGAIMERS\claude`
전제 문서: `00_ASSESSMENT.md`(평가), `01_TRACKMAN_FEATURE_CATALOG.md`(피처 후보), `02_EMBEDDING_METHODS.md`(임베딩 방법)

> ## ⚠ 개정 안내 (2026-08-12, Phase 0 실행 후)
>
> **Phase 0을 실제로 실행한 결과 이 문서 §0·§4·§6·§10의 우선순위가 바뀌었다. 최신 우선순위와 일정은 `05_PHASE0_RESULTS.md` §7을 따른다.**
>
> 바뀐 핵심 3가지:
> 1. **투수-시즌 수준 표현의 판별력은 이미 소진됐다.** oracle 상한 AUC 0.5446 / BSS 720.9 < 팀의 현재 단일 모델 0.5498 / 784.6. 따라서 "TrackMan 임베딩으로 AUC 천장을 공격한다"는 이 문서의 전제는 **부분 철회**한다. Tier A·C·D와 무학습/contrastive 임베딩은 강등.
> 2. **헤드룸은 `투수 × 상황` 조건부에 있다.** oracle `(투수,시즌,타자손)` = BSS **917.7** (현재 최고 815.08 대비 +100). **Tier E와 M2/M4가 유일한 유망 축으로 승격.**
> 3. **구조 선택은 fold 간 전이되지 않는다** (spearman 0.06~0.12). 따라서 Phase 0의 P0-1은 예측 벡터 대신 탐색 그리드 순위 상관으로 이미 결론이 났고, **nested 선택 프로토콜이 모든 실험의 전제**가 된다.
>
> 추가로 `04_PREPROCESSING_SPEC.md`(전처리 스펙), `06_COMPUTE_STRATEGY.md`(컴퓨팅 활용)가 신설됐다. 아래 본문은 Phase 0 이전의 원안으로, 설계 근거와 미채택 후보 목록으로서 유지한다.

---

## 0. 전략 요약

| | |
|---|---|
| **그만할 것** | correction 하이퍼 재탐색(smoothing / alpha / scale / blend weight), 트리 용량 실험, seedbag, 군집 K 재탐색, Optuna 재탐색 — 4개 축 모두 문서상 포화 확인 |
| **할 것** | TrackMan을 "72개 요약통계"에서 "조건부 반응 프로파일 + 분포 임베딩"으로 전환하고, **AUC 천장(0.550)** 을 공격한다 |
| **주입 원칙** | GBDT 원시 피처 직접 투입은 4번 시도해 4번 실패(748~781, 임베딩 395). **correction의 평활 계층(M2)** 을 1순위로 한다 |
| **판정 원칙** | 단일 fold 점수 비교 금지. **paired 부트스트랩 CI + R-only 양 fold 게이트**를 통과한 것만 후보로 승격 |
| **안전선** | `submit_013`(Public 895.404)은 어떤 경우에도 유지. 신규 후보는 추가 슬롯에서만 검증 |

---

## 1. 작업 디렉토리 규약

```
LGAIMERS/claude/
  00_ASSESSMENT.md                 # 현황 평가 (완료)
  01_TRACKMAN_FEATURE_CATALOG.md   # 피처 후보 카탈로그 (완료)
  02_EMBEDDING_METHODS.md          # 임베딩 방법 서치 (완료)
  03_PLAN.md                       # 본 문서
  RESULTS.md                       # 단일 진실원. 모든 실험 결과를 여기에만 누적
  src/
    p0_diagnostics.py              # Phase 0
    tm_normalize.py                # 시즌x구종 robust z + EB 축소 유틸
    tm_conditional_profile.py      # Tier E
    tm_outing_drift.py             # Tier B
    tm_arsenal.py                  # Tier C, D
    tm_release_consistency.py      # Tier A
    crosswalk_soft.py              # Tier F2
    emb_kme.py                     # N2
    emb_svd_strategy.py            # N5
    emb_sw.py                      # N3
    emb_contrastive.py             # L1
    inject_kernel_correction.py    # M2
    inject_matrix_factorization.py # M4
    eval_paired_ci.py              # 공통 판정 유틸
  outputs/                         # 피처 parquet, 임베딩 lookup, 지표 csv
  reports/                         # fold별 ablation 결과
```

**규칙 3개**
1. 모든 실험 결과는 `experiment/model_optimization/validation_registry.csv` 스키마에 append하고, 요약은 `claude/RESULTS.md`에만 쓴다. 기존 요약 문서 4개(`CURRENT_MODELING_SUMMARY` / `INSIGHTS_SUMMARY` / `MODEL_VAL_SUMMARY` / `R_FOCUS_RECHECK`)는 **읽기 전용 아카이브로 동결**한다. (현재 공동 SVD 안전형 점수가 문서마다 814.291 / 814.711 / 814.831로 다르게 적혀 있다.)
2. 기존 `experiment/` 코드는 수정하지 않는다. 필요한 것은 import 또는 복사.
3. 모든 스크립트는 `--fold {2022,2023,2024,final}` 인자를 받고 fold별로 재적합한다.

---

## 2. 공통 검증 프로토콜

```
Fold 정의 (기존 유지)
  F22: 2019~2021 학습 → 2022 검증   |  TrackMan ≤ 2021
  F23: 2019~2022 학습 → 2023 검증   |  TrackMan ≤ 2022
  F24: 2019~2023 학습 → 2024 검증   |  TrackMan ≤ 2023
  Final: 2019~2024 전체 재학습 → 2025 예측  |  TrackMan ≤ 2024

기록 항목 (fold x {ALL, R, F} 전부)
  brier, normalized_brier, bss, auc, pred_mean, target_mean, mean_gap,
  paired_delta_brier vs 기준, bootstrap 95% CI, tm_available/unavailable 분해
```

### 채택 게이트 (G1~G6)

| | 조건 | 성격 |
|---|---|---|
| G1 | **R-only F23 ΔBrier < 0 AND R-only F24 ΔBrier < 0** | 하드 |
| G2 | 전체 ΔBrier의 **paired 부트스트랩 95% CI가 0을 포함하지 않음** | 하드 |
| G3 | F(game_type) 기여율 ≤ 60% | 하드 |
| G4 | **AUC 증가** (판별력 이득 여부 구분) | 소프트 — 미달이면 "보정형"으로 분류 |
| G5 | TrackMan 가용 / 미가용 각각에서 악화 없음 | 소프트 |
| G6 | 245,789행 추론 ≤ 8분, RAM ≤ 24GB, 오프라인 lookup 재현 | 하드 |

### 중단(kill) 기준

- 한 Phase에서 **2개 연속 후보가 G2 실패** → 그 Phase 종료하고 다음으로 이동
- Phase 총 소요가 계획의 2배 초과 → 중단
- 임베딩 계열이 **N1(분위수·모멘트 signature) 대비 lift를 못 내면** 해당 방법 폐기

---

## 3. Phase 0 — 진단 고정 (0.5일, 최우선)

**이걸 먼저 하지 않으면 이후 모든 실험의 판정 기준이 없다.**

| # | 작업 | 산출물 | 왜 |
|---|---|---|---|
| P0-1 | **paired ΔBrier 표준오차·부트스트랩 CI 계산** — 기존 상위 8개 후보(805.56~815.08) 전부에 대해 쌍별로 | `reports/p0_candidate_ci.csv` | 추정상 인접 후보 차이가 1~2 se다. **CI가 겹치면 "815.08이 812.70보다 좋다"는 전제 자체가 무효**이고, 제출 우선순위를 점수 대신 구조적 안정성으로 다시 정해야 한다 |
| P0-2 | 기존 72개 TrackMan 피처가 **원시 절대값인지 시즌×구종 z인지** 코드 확인 (`build_trackman500_asof_features.py`, `trackman500_cutoff.py`) | `reports/p0_tm_feature_audit.md` | 2022 단절(extension −0.148, IVB −3.079, FB −12.08%p)이 미반영이면 기존 피처 품질부터 개선 여지가 있다 |
| P0-3 | **임베딩 실패 원인 3개 재현 확인** — (a) 2019·2020 481,500행(32.64%) 전부 0, (b) `trackman_embedding` 24차원이 `tm_available=0`인 46%에도 값 출력, (c) 타깃 지도 학습 | `reports/p0_embedding_postmortem.md` | 이 3개가 확정되면 임베딩 계열을 재개할 근거가 된다. (a)만 고쳐도 AUC 0.5405 → V1 수준 복귀 가능성 |
| P0-4 | **AUC 상한·하한 실측** — ① `asof_*`만, ② +기존 TrackMan 72, ③ +투수 ID one-hot(누수 상한, 진단용), ④ 상황변수만 | `reports/p0_auc_ceiling.csv` | "0.550이 데이터의 한계인가, 우리 표현의 한계인가"를 구분. ③이 크게 높으면 투수 표현 개선 여지가 남았다는 직접 증거 |
| P0-5 | **F23 fold의 성격 정량화** — 상수 0.5 예측, oracle 평균 이동, 순위 신호만의 기여를 분해 | `reports/p0_f23_decomposition.md` | F23이 "모델 품질"이 아니라 "평균 편향"을 재는 fold임을 수치로 고정하고, 목적함수 가중을 유지할지 결정 |

**P0 게이트**: P0-1 결과에 따라 Phase 1~5의 채택 임계(G2)를 확정한다. P0-4 ③이 0.560 미만이면 판별력 공략의 기대값이 낮으므로 계획을 축소하고 Phase 1·2만 수행한다.

---

## 4. Phase 1 — 중간단계 피처 (2일)

`01_TRACKMAN_FEATURE_CATALOG.md`의 Tier를 우선순위대로 하나씩 추가하며 ablation.

| 순서 | 대상 | 산출물 | 예상 |
|---|---|---|---|
| 1-1 | **시즌×구종 robust z 전환** (§0-3) + EB 축소(§0-4) 유틸 | `src/tm_normalize.py`, 기존 72피처의 z 버전 | 기존 피처 품질 개선. 절대값 vs z 중 하나만 남긴다 |
| 1-2 | **Tier E 조건부 반응 프로파일** — E1(구종×카운트×타자손, SVD 8~16으로 압축), E3(압박 상황 잔차 SD), E4(좌우 사용률 차) | `outputs/tm_conditional_profile_{fold}.parquet` | **최대 기대.** 2024 R 최대 잔차 구간(count 3-0 −1.80%p, 3-2 +1.29%p, 0-1 −1.12%p)과 직결 |
| 1-3 | **Tier B 등판 내 drift** — B1(pitch_no slope), B2(후반/전반 SD 비), B5(inning 상호작용) | `outputs/tm_outing_drift_{fold}.parquet` | main에 없는 정보 순수 추가 |
| 1-4 | **Tier D1 FB 대비 차분** + C4(spin gap), C9(speed retention) | `outputs/tm_arsenal_{fold}.parquet` | Stuff+ 검증 입력(YoY R² 0.702) |
| 1-5 | **Tier A3 상황 통제 잔차 SD** + A2(타원 면적), A5(구종 간 릴리스 응집도) | `outputs/tm_release_consistency_{fold}.parquet` | 이론적 직결성 최고, 단 각도 부재로 기대치 하향 |
| 1-6 | **Tier C8** tagged/auto 불일치율 | 위 arsenal parquet에 포함 | 거의 무료 |

**차원 규칙**: Tier별로 PCA/SVD 압축 후 **최종 추가 8~24개**를 목표. 그 이상이면 압축 실패로 간주(구종별 전개가 760.71~764.94로 악화한 전례).

**투입 순서**: 각 Tier를 (a) GBDT 피처로 추가 → G1~G5 판정, (b) 실패 시 Phase 3의 M2/M3 경로로 재시도.

---

## 5. Phase 2 — 커버리지 확대 (1일)

현행 2024 행 커버리지 60.24%가 모든 TrackMan 이득을 0.6배로 희석한다.

| 순서 | 작업 | 산출물 |
|---|---|---|
| 2-1 | **F1 게이트 완화** — 500 → 300 / 200 / 100구, `log1p(n)` + EB 축소 동반 | `reports/p2_gate_sweep.csv` |
| 2-2 | **F2 soft crosswalk** — cosine 상위 k 후보의 softmax 가중 평균으로 프로파일 구성. `cw_entropy`(후보 분포 불확실성) 동반 투입 | `outputs/crosswalk_soft_{fold}.parquet` |
| 2-3 | **F4 다중 시즌 가중 합산** — 500구 미만 시즌들을 `season_gap` 페널티와 함께 합산 | 2019(0%)·2020(50.33%) 회복 |
| 2-4 | **F5 팀 fallback** — 투수 매칭 실패 시 매칭 성공 투수들의 팀 평균 프로파일 | cold-start 커버 |

**기대**: 커버리지 60% → 75~85%. Phase 1의 이득이 그대로 1.25~1.4배가 된다.
**주의**: `has_trackman` 자체가 "출장 많은 주력 투수" 대리변수일 수 있다(가용 행 `asof_pitcher_n` 중앙값 4,440 vs 미가용 837). 커버리지가 늘면 이 편향 구조가 변하므로 **G5를 반드시 확인**한다.

---

## 6. Phase 3 — 임베딩 + 커널 평활 주입 (2일)

| 순서 | 작업 | 산출물 |
|---|---|---|
| 3-1 | **N1 확장 signature** (분위수 p10~p90, skew 추가) — 이후 모든 임베딩의 기준선 | `outputs/emb_signature_{fold}.parquet` |
| 3-2 | **N2 Kernel Mean Embedding (RFF)** — D=256 → PCA 8~16. 무학습·결정론적, 1시간 | `outputs/emb_kme_{fold}.parquet` |
| 3-3 | **N5 전략 SVD** — `pitcher × (구종군 × 볼카운트 × 타자손)` rate 행렬 → TruncatedSVD 8~16 | `outputs/emb_svd_strategy_{fold}.parquet` |
| 3-4 | **N3 Sliced-Wasserstein** — 랜덤 방향 64 × 분위수 9 → PCA 8~16 | `outputs/emb_sw_{fold}.parquet` |
| 3-5 | **M2 커널 평활 correction** — 현행 correction의 hard 군집(좌2~8 / 우4~20 + seed 3개 평균)을 **임베딩 커널 이웃 `w = exp(−‖eₚ−eₚ'‖²/2σ²)`** 로 교체 | `src/inject_kernel_correction.py` |
| 3-6 | **M3 residual expert** — 임베딩만 쓰는 소형 모델의 시간 OOF를 앙상블 멤버로. 기존 correction과의 상관을 먼저 확인(공동 SVD는 0.09로 성공, 멀티뷰는 0.887로 가중치 0 수렴) | `reports/p3_expert_correlation.csv` |
| 3-7 | **M1 재시도** — 단, 4~8차원만 + NaN 유지 + `has_trackman` 분리 | — |

**M2가 이 Phase의 핵심이다.** 팀이 검증한 유일한 성공 패턴(군집을 평활 계층으로 → 806~815)의 직접 연장이며, hard K 탐색(좌2~8 × 우4~20 × seed 5 = 수백 조합)이 σ 하나의 그리드로 대체되어 **선택 과적합 표면적이 크게 줄어든다.**

---

## 7. Phase 4 — Contrastive 임베딩 (2일, Phase 3 결과에 따라 조건부)

**진입 조건**: Phase 3에서 M2 또는 M3가 G1·G2를 통과했을 때만 착수. 실패했다면 임베딩 계열의 주입 경로 자체가 막힌 것이므로 Phase 5로 건너뛴다.

| 순서 | 작업 |
|---|---|
| 4-1 | **L1 same-pitcher-different-game contrastive** — 같은 투수의 다른 등판에서 뽑은 pitch subset(128구) 쌍을 positive, 다른 투수를 negative → NT-Xent. 인코더는 DeepSets(`mean+max+sd` pooling) |
| 4-2 | **pretext는 TrackMan 1.79M 전량** — crosswalk 미매칭 투수도 라벨이 필요 없으므로 학습 표본이 2배 (906명 전부) |
| 4-3 | hard negative를 **같은 구속대·같은 손**으로 제한 — "구속 절대값" 같은 식별력 높지만 커맨드 무관한 축의 지배를 막는다 |
| 4-4 | 검증: **시간 분할 + 투수 holdout 이중**. 투수 holdout에서 무너지면 base rate 암기 |
| 4-5 | YoY 안정성 측정 — 같은 투수의 t / t+1 임베딩 코사인. Stuff+ YoY R² 0.702, Kirby 0.50이 참고선 |
| 4-6 | M2/M3로 주입 |

---

## 8. Phase 5 — middle 잔차 인자화 (1일)

Phase 3~4와 병행 가능. `R_FOCUS_RECHECK.md`가 1순위로 지목했으나 미착수 상태다.

- R 2023→2024 성공률 −1.34%p 중 **reverse −0.80%p(감소), middle +2.08%p(증가)**
- 현행 correction의 주력은 reverse. **middle은 독립 상성 실험이 없다** (reverse만 60구조·180검증 수행)
- 작업: reverse와 동일한 파이프라인으로 `시즌×투수손×타자손×볼카운트` 평균 제거 후 middle 잔차 → 임베딩 커널 평활(M2) → Ridge 보정
- **M4 확장**: 투수×타자 middle 잔차 행렬의 공동 SVD (reverse는 이미 815.08, middle은 미시행)

---

## 9. Phase 6 — 동결·제출 (1일)

| 순서 | 작업 |
|---|---|
| 6-1 | G1~G6 통과 후보만 2024까지 전체 재학습 → frozen artifact |
| 6-2 | 정방향 / 역순 / 1행 예측 일치 검증 (<1e-7), 245,789행 추론 시간·RAM |
| 6-3 | ZIP 구조·CRC·파일명 길이·SHA256 기록 |
| 6-4 | **제출 슬롯 배분** — ① 안전: `submit_013` 유지, ② 신규 최고: G1~G4 전부 통과분, ③ 다양성: F 기여율이 낮고 R 양 fold 개선이 큰 후보 (Private에서 F 체제가 또 바뀔 때의 보험) |

---

## 10. 일정 (누적 영업일 기준, 병렬 없음 가정)

| 일 | Phase | 산출 |
|---:|---|---|
| 0.5 | Phase 0 | 판정 기준 확정. **P0-1 결과가 이후 전체 우선순위를 바꿀 수 있다** |
| 1.5 | Phase 1 (1-1 ~ 1-3) | 정규화 전환 + Tier E + Tier B |
| 2.5 | Phase 1 (1-4 ~ 1-6) | Tier D·C·A |
| 3.5 | Phase 2 | 커버리지 60% → 80% |
| 5.0 | Phase 3 | 무학습 임베딩 4종 + M2/M3 주입 |
| 6.0 | Phase 5 | middle 잔차 (Phase 3과 병행 시 −0.5일) |
| 8.0 | Phase 4 | contrastive (조건부) |
| 9.0 | Phase 6 | 동결·제출 |

---

## 11. 기대 효과와 근거 (정직한 추정)

| Phase | 기대 BSS 변화 | 근거 / 불확실성 |
|---|---|---|
| Phase 0 | 0 (판정 기준만) | 다만 P0-1이 **현재 후보 순위를 무효화할 수 있어 가장 가치가 높다** |
| Phase 1 | +5 ~ +25 | 기존 TrackMan 전체 기여가 +10.8(758.92→769.69)이었다. Tier E는 완전 미탐색 축이라 상방이 있지만 불확실 |
| Phase 2 | Phase 1 이득의 ×1.25~1.4 | 커버리지 산술. 비교적 확실 |
| Phase 3 | +5 ~ +20 | M2는 검증된 패턴의 연장이라 하방이 얕다. 상방은 임베딩 품질에 의존 |
| Phase 4 | 0 ~ +15 | 가장 불확실. Phase 3 성공 시에만 착수 |
| Phase 5 | +3 ~ +12 | reverse가 같은 방식으로 +2.4(810.10→812.70)를 냈고 middle은 최근 악화의 주범 |

**중요**: 위 숫자는 전부 추정이며, 실제로 확실한 것은 **"보정 계층 재탐색의 기대값이 0에 가깝다"** 는 것뿐이다. 그래서 계획의 축을 판별력으로 옮긴다.

---

## 12. 확인이 필요한 열린 항목

| # | 항목 | 영향 |
|---|---|---|
| 1 | **대회 마감일과 남은 제출 슬롯 수** | Phase 4(contrastive) 착수 여부, 제출 슬롯 배분 |
| 2 | **DACON 보조 라벨 질의 답변** — `asof_*` 차분으로 복원한 reverse/middle을 학습 라벨로 쓰는 것이 허용되는가 (`pitcher_embedding/SUBMISSION_STATUS.md`에 질문 초안 있음, 미게시) | 허용 안 되면 현재 correction 구조의 근거가 흔들린다. **가장 큰 규칙 리스크이고 아직 미해결** |
| 3 | 팀 내 역할 분담 (Phase 1·2와 Phase 3·4는 독립 병렬 가능) | 일정 절반 단축 가능 |
| 4 | 외부 데이터 허용 범위 — KBO 공개 기록으로 crosswalk를 보강할 수 있는가 | 커버리지 52.90%의 근본 해결 경로 |
