# sj 모델링 흐름과 단계별 비교군

작성일: 2026-08-20  
현재 전체 최고: `submit_034` Public **983.5821977654**  
현재 3WAY 최고: `submit_040` Public **976.9464685456**  
목적: `sj`에서 무엇을 어떤 비교군으로 검증했고, 다음 단계로 왜 넘어갔는지 한 문서로 요약한다.

> 이 문서는 실험 흐름 요약이다. 정확한 수치·코드·규정 근거는 각 단계의 원본 문서를
> 따른다. Public 점수는 완료된 실험 기록으로만 쓰며 후속 계수나 offset을 역산하는 데
> 사용하지 않는다.

## 전체 흐름

```text
문제·누수 감사
→ 1WAY 직접 모델 기준선
→ 전처리 단일 평가
→ 피처 마스킹·제거
→ 전처리 조합 탐색
→ TrackMan·계층 피처
→ 모델·학습 방식 튜닝
→ 시간 OOF·보정
→ residual expert
→ 앙상블
→ 1WAY 실패성분 분해
→ 3WAY 타깃별 최적화
→ 사건확률 합집합·보정
→ 2019~2024 재학습·제출 검증
```

## 0. 문제 정의·EDA·누수 감사

| 항목 | 내용 |
|---|---|
| 목적 | 타깃 의미, 시간 변화, 사용할 수 있는 정보의 경계를 확정 |
| 진행 | 시즌·R/F·카운트·좌우·결측·cold-start 분포 분석, Target 복원 구조 감사 |
| 비교군 | 랜덤 분할 vs 시즌 순방향 분할, 전체 지표 vs R/F·월별 지표, raw rate vs 표본 수로 수축한 rate |
| 핵심 결과 | 2019→2024 성공률 하락, R/F regime 차이, `row_id`와 다음 행 누적률 사용 금지 확인 |
| 다음 결정 | Val2023=`2019~2022→2023`, Val2024=`2019~2023→2024`; Brier/raw BSS를 주 지표로 고정 |

원본: `experiment/control_success_eda/EDA_REPORT.md`, `README.md`

## 1. 1WAY 직접 성공확률 기준선

| 항목 | 내용 |
|---|---|
| 목적 | `control_success`를 직접 예측하는 강한 기준선 구축 |
| 진행 | 공식 행 피처와 시간 안전 과거 이력으로 모델별 동일 fold 평가 |
| 비교군 | CatBoost, XGBoost, LightGBM, TabM/MLP; raw 47열 vs enhanced 피처 |
| 추가 비교 | 전체 모델 vs R/F 분리 모델, raw ID vs 빈도·수축 처리, 작은 모델 vs 대형 모델 |
| 핵심 결과 | CatBoost/XGBoost가 주력. 대형 트리와 완전 R/F 분리는 과적합, TabM은 F 보조 expert에 제한적으로 사용 |

## 2. 고정 모델 기반 단일 전처리 평가

이 단계의 주 평가기는 MLP가 아니라 **고정 CatBoost**였다. MLP/TabM은 별도 모델군
실험이다.

| 항목 | 내용 |
|---|---|
| 통제 조건 | fold 2024, CatBoost 900회·depth 8·동일 seed/피처 기준선 |
| 진행 | 전처리 원자 하나만 추가하거나 교체하고 `bss_centered` 변화 측정 |
| 비교군 | `id_frequency`, `temporal_cyclic`, `rate_geometry`, `rate_multiscale`, `count_multiscale`, `context_robust`, `trackman_quality`, compact 계열 |
| 대표 결과 | `id_frequency` +7.94, `temporal_cyclic` +4.75; `trackman_quality`는 단독 -3.11 |
| 결론 | 단일 평가는 방향 탐색용이며 단독 탈락 후보도 조합 단계에 일부 남김 |

원본: `preprocess_lab/RESULTS.md`

## 3. 피처 마스킹·제거·대체 평가

| 항목 | 내용 |
|---|---|
| 목적 | 기존 신호의 필요성, 중복, ID·TrackMan 의존성 확인 |
| 진행 | 피처군을 통째로 제거하거나 compact/빈도 표현으로 교체 |
| 비교군 | raw ID vs `id_frequency` vs `drop_ids`; 전체 TrackMan vs compact vs `no_trackman`; 전체 component vs compact vs `no_component`; season 유지 vs 제거/순번화 |
| 대표 결과 | `drop_ids` -57.06, `no_component` -25.95, `no_trackman` -5.71; ID는 제거가 아니라 빈도 표현이 유효 |
| 결론 | “차원이 크면 제거”가 아니라 정보 보존 방식과 시간 전이를 기준으로 판정 |

신경망 쪽에서는 Brier, positive soft-F1, macro soft-F1, 혼합 loss도 비교했다. 순수
soft-F1은 확률이 극단으로 붕괴했고, 전체 학습에서는 **Brier loss**가 최선이었다.

## 4. 전처리 조합 탐색

| 항목 | 내용 |
|---|---|
| 목적 | 단독 성능으로 보이지 않는 상호보완 조합 탐색 |
| 진행 | 빔 폭 3 × 4라운드, 총 76조합; 매 라운드 상위 부분집합만 확장 |
| 비교군 | 단일 원자, 2~4개 부분집합, `all_additive`, `all_compact` |
| 최고 | `id_frequency + temporal_cyclic + trackman_quality` = +16.52 |
| 대조 | `id_frequency` 단독 +7.94, 8개 전체 +3.27, 12개 전체 +2.78 |
| 결론 | 단독으로 나쁜 피처도 조합에서 살 수 있고, 모두 넣는 방식은 실패 |

## 5. TrackMan·임베딩·계층 피처

| 트랙 | 비교군/진행 | 결과 |
|---|---|---|
| TrackMan 집계 | 전체 요약, compact, 구종별 확장, 500구 strict-as-of | compact 요약은 일부 유효, 구종별 고차원 확장은 불안정 |
| crosswalk | hard 매칭, 가용성·유사도·margin 피처, 미가용 fallback | ID 직접 사용 금지, 신뢰도와 가용성 분리 |
| 표현학습 | PCA/SVD 12·24차원, target-free residual PCA, supervised pitcher embedding | fold·모델별 부호가 뒤집혀 최종 주력에서 제외 또는 보조화 |
| 계층 피처 | 투수×타자손, 카운트×손, 이닝×손, 투수·타자 residual | 단순 집계보다 EB 수축 및 하위 계층 주효과 차감이 유효 |
| strict F1 | 학습행도 자기 시즌 이전 데이터만 사용하는 45개 hierarchy 피처 | 시간 계약은 통과했지만 후속 모델 선택에는 fold 강건성 재확인 필요 |

TrackMan은 예측 시즌보다 이전 시즌만 사용한다. Val2024는 2023 이하, 최종 2025는
2024 이하로 고정한다.

## 6. 모델군·하이퍼파라미터·학습 방식 평가

| 축 | 비교군 | 판정 |
|---|---|---|
| 트리 모델 | CatBoost, XGBoost, LightGBM | CatBoost/XGBoost 중심 채택 |
| 신경망 | MLP multi-head, TabM, FT-Transformer, attention | 단독 주 모델로 GBDT를 넘지 못함; 일부 F/residual 보조 후보 |
| 크기 | depth/leaves/iteration 증가 | 20~24 leaves 부근까지 소폭, 32~128은 과적합 |
| 수축 | smoothing K, prior, reliability | 중간 수축 강도가 유효, 과도한 수축은 악화 |
| 학습 가중치 | 전체 동일, F 축소, F 제거, 짧은 등판 제거/축소 | 제거보다 낮은 가중치가 유효 |
| 목적함수 | Brier, soft-F1, macro soft-F1, 혼합형 | 확률 예측에는 Brier 유지 |
| 탐색 방식 | 단일 fold 최고값 vs 여러 fold·seed 확인 | 단일 fold 최고값은 과적합 위험으로 채택 금지 |

## 7. 시간순 OOF·확률 보정

| 항목 | 내용 |
|---|---|
| 진행 | 이전 시즌만 학습한 OOF 확률을 저장하고 Brier, BSS, 평균 오차, R/F·월별 성능 평가 |
| 비교군 | 무보정, Platt, isotonic, beta/logit, season prior, scalar offset |
| 초기 1WAY 결과 | 자유 사후 보정은 다음 시즌으로 잘 전이되지 않아 대부분 기각 |
| 유지한 방식 | 학습 데이터 시즌 추세로 정한 prior/base score, 강한 수축, 고정 recipe |
| 규칙 | test 평균·분포·순위 또는 Public 점수로 보정값을 정하지 않음 |

OOF teacher가 validation label로 early stopping을 선택하면 meta 성능이 낙관적일 수
있으므로, 최종 비교는 fixed-iteration honest OOF로 다시 확인한다.

## 8. Residual·상성 expert

| 비교군 | 진행 | 결과 |
|---|---|---|
| 전역 모델 vs 투수·타자 유형 residual | 기준 예측 오차를 군집 셀에서 EB/Ridge로 수축 | hard cluster 직접 입력보다 residual smoothing이 유효 |
| reverse expert | 타자 군집, count×inning×손, 여러 K/seed 비교 | R의 큰 비중을 작은 residual로 보정 |
| F expert | 글로벌 단독, F 전용 XGB/CatBoost/TabM, hard dispatch, 부분 풀링 | 완전 분리보다 글로벌을 남긴 축소 residual이 안정적 |
| 공동 SVD | 기존 correction과의 상관·R/F 성능 비교 | 다양성은 좋았지만 fold 안정성 부족 후보는 보류 |

## 9. 앙상블

| 항목 | 내용 |
|---|---|
| 멤버 | CatBoost/XGBoost seed bag, 성분 라인, insight 모델, R/F residual expert |
| 비교군 | 단일 최고, 단순 평균, 고정 전역 가중치, 투구량 구간 가중치, 자유 최적화 |
| 평가 | OOF Brier 증분, 오차/확률 상관, R/F 손실, seed 안정성 |
| 성공 | 서로 다른 잔차를 가진 모델의 작은 가중치 결합, 투구량 구간별 신뢰도 |
| 실패 | 같은 fold에서 자유롭게 고른 탐욕/해석적 가중치, 상관이 높은 팀 모델 결합 |
| 원칙 | 단독 모델 선택과 앙상블 멤버 선택을 분리; 앙상블은 정확도와 다양성을 함께 확인 |

## 10. 1WAY 실패성분 분해

| 항목 | 내용 |
|---|---|
| 목적 | 성공을 직접 맞히는 대신 실패 원인을 따로 예측해 다른 오차 방향 확보 |
| 성분 | middle, reverse, middle∩reverse, outside-ball, outside-zone |
| 진행 | 다섯 성분이 공통 111피처를 사용하고 각 모델 확률을 포함–배제로 결합 |
| 비교군 | direct success 모델, 성분 분해 모델, 성분 스태킹, 더 세밀한 분할, 카운트별 전문가 |
| 결과 | direct 모델은 단독이 강하지만 성분 분해가 base와 덜 상관되어 결합 이득이 큼 |
| 기각 | 과도한 세분화·카운트별 분리는 표본만 줄고 성능 악화 |

```text
P(success) = 1 - (p_m + p_r - p_mr + p_ob + p_oz)
```

## 11. 3WAY 타깃별 독립 최적화

| 항목 | 내용 |
|---|---|
| 목적 | 1WAY의 공통 피처 제약을 없애 각 실패 사건에 맞는 전처리·모델을 선택 |
| 실제 사건 | middle, reverse, outside와 교집합 middle∩reverse |
| 진행 | 타깃별 단일 전처리 → 타깃별 빔 서치 → 계층 축 → 모델/seed ensemble |
| 비교군 | 타깃 공통 피처 vs 타깃별 피처; M/R/B vs M/R/O; identity vs logistic/GBDT 결합기 |
| 대표 결과 | middle–reverse 전처리 순위 Spearman -0.215; 타깃별 최적화 필요성 확인 |
| 결합 결과 | 학습형 logistic/GBDT 결합기는 fold 전이 실패, 포함–배제 identity가 최선 |

`ball`은 실패 유형과 겹치는 속성이어서 합집합을 닫지 못한다. 따라서 최종 경로는
`middle/reverse/outside/MR`을 사용한다.

## 12. 사건확률 합집합과 재보정

| 비교군 | Val2024 raw BSS | 판정 |
|---|---:|---|
| 기존 M/R/O/MR 포함–배제 | 828.59 | 기준 |
| M/R/O 세 확률에서 bounded MR 재추정 | 822.63 | 기각 |
| 사건별 약한 logit 보정 후 합집합 | 842.42 | 초기 채택 |

```text
p_failure = p_middle + p_reverse + p_outside - p_middle_and_reverse
p_success = clip(1 - p_failure, 0, 1)
```

진행 방식은 각 사건확률을 OOF에서 따로 보정하고, `p_mr`을 Fréchet 범위 안에 둔 뒤
합집합의 여집합을 계산하는 것이다. 자유로운 최종 성공확률 회귀는 사용하지 않는다.

Public 결과:

| 제출 | 구성 | Public |
|---|---|---:|
| `submit_034` | 1WAY + 짧은 등판 학습 가중치 | **983.5821977654** |
| `submit_035` | 1WAY + 2차 상호작용 | **981.8287622303** |
| `submit_037` | 원래 3WAY 포함–배제 | 958.2563447143 |
| `submit_040` | 사건별 약한 보정 + 동일 합집합 | **976.9464685456** |
| 040−037 | 3WAY 내부 향상 | **+18.6901238313** |

따라서 `submit_040`은 **3WAY 최고**지만 `sj` 전체 최고는 `submit_034`다. 3WAY가
1WAY 최고를 넘으려면 추가로 **+6.6357292198**가 필요하다.

## 13. 최종 재학습·패키징·규정 검증

| 항목 | 진행/비교 |
|---|---|
| 최종 학습 | 설정 동결 후 2019~2024 전체 재학습; 탐색 fold 모델과 구분 |
| 추론 | 현재 행과 고정 모델/lookup만 사용; 행 간 집계 없음 |
| TrackMan gate | cutoff 2023→최대 2022, 2024→2023, 2025→2024 확인 |
| 행 독립성 | `predict(row_i 단독) == predict(전체)[i]` 기계 검증 |
| 코드 감사 | 행 간 통계, 네트워크, 학습 코드, 절대 경로 전수 스캔 |
| 패키지 | ZIP 루트 `model/`, `script.py`, `requirements.txt`; CRC·오프라인 smoke |
| 시간 | 실제 245,789행으로 10분 제한 확인 |
| 기록 | SHA-256, Brier/BSS, R/F, 월별 Brier, prediction mean, Public 결과 기록 |

`submit_040`은 행 독립성 최대차 0.0, 오프라인 실행 및 245,789행 약 10.94초를
통과했다.

## 핵심 결론

1. 가장 큰 구조적 개선은 **실패성분 분해**와 **타깃별 피처 최적화**였다.
2. 단일 전처리 최고값보다 **부분집합 조합**이 중요했고, 모든 피처 투입은 실패했다.
3. 신경망·대형 모델·자유 결합기보다 중간 크기 GBDT와 구조적 합집합이 안정적이었다.
4. TrackMan은 과거 시즌만 사용한 compact/신뢰도 표현이 안전했고, 고차원 직접 투입은 불안정했다.
5. 최종 확률은 임의 회귀값이 아니라 `M ∪ R ∪ O`의 여집합이어야 한다.
6. 다음 개선은 fixed-iteration honest OOF에서 사건별 calibration을 재검증한 뒤,
   coherent 5-state 모델과 공동 student 증류 순서로 진행한다.

## 원본 문서 안내

- 전체 EDA·기준선: `cowork/sj/README.md`
- 접근법 유형화: `cowork/sj/claude/24_APPROACH_TYPOLOGY.md`
- 1WAY 최종 상태: `cowork/sj/claude/26_1WAY_STATUS.md`
- 전처리 실험: `cowork/sj/preprocess_lab/RESULTS.md`
- 신규 피처 캠페인: `cowork/sj/feature_campaign_1000/README.md`
- 모델 최적화: `cowork/sj/experiment/model_optimization/CURRENT_MODELING_SUMMARY.md`
- 임베딩 loss: `cowork/sj/experiment/pitcher_embedding/F1_LOSS_RESULTS.md`
- 3WAY 결과: `cowork/sj/three_way/RESULTS.md`
- 확률 결합 계획: `cowork/sj/three_way/meta_fusion/MODELING_PLAN.md`
- 최신 제출 로그: `cowork/sj/submit/2026-08-20/SUBMISSION_LOG.md`
