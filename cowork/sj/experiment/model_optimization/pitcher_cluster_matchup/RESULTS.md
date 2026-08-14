# 투수 임베딩·클러스터링 및 타자 상성 실험 결과

업데이트: 2026-08-12  
검증 기준: F23=`≤2022 → 2023`, F24=`≤2023 → 2024`  
TrackMan 규칙: 검증연도 미만 시즌, 투수-시즌 500구 이상만 사용

최신 Public LB: `submit_013` **895.404000081**

## 1. 현재 결론

투수 군집 ID를 XGBoost에 직접 넣는 방식은 개선되지 않았다. 반면 군집을 **투수 유형×타자 유형 상성 통계의 smoothing 단위**로 사용하고, 기존 확률에 작은 Ridge residual을 더하는 방식은 효과가 있었다.

현재 최고 권장 조합:

- 투수 프로필: TrackMan 물리 + 과거 제구 성향
- 투수 군집: hand 분리, diagonal GMM, Left 2 / Right 4
- 성공 타자 군집: hand 분리 KMeans, Left 3 / Right 4
- reverse 전용 타자 군집: hand 분리 KMeans, Left 4 / Right 6
- 성공 상성: 반감기 1년, smoothing 1000, Ridge alpha 10, correction 0.25
- reverse 상성: 상황별 reverse 평균 제거, 반감기 1년, smoothing 1000, Ridge alpha 1000, correction 0.55
- reverse KMeans seed: 17/2026/4099 correction 평균
- 적용 방식: 기존 adjusted 확률에 두 correction을 더한 뒤 performance 모델과 0.6085:0.3915로 혼합
- 2024 혼합 BSS: **812.7040**

## 2. 데이터 감사

- 메인 hand `1=Left`, `2=Right`로 TrackMan과 대응했다.
- crosswalk 336명 중 335명 일치, 1명 불일치가 있어 hand를 ID 추론에 사용하지 않고 별도 값으로 보존했다.
- cutoff별 투수 프로필 2020~2025를 생성했다.
- 프로필은 물리·구종·제구·상황 split을 포함한 286개 열이다.
- 2024 실제 행 기준 TrackMan 500구 프로필 coverage는 57.34%였다.
- 2024 투수유형×타자유형 pair coverage는 72.78%였다.

## 3. 투수 군집 탐색

- 표현: physical / physical+control
- PCA: 8 / 16
- 알고리즘: KMeans / diagonal GMM
- hand별 K: `(2,4)`, `(3,6)`, `(4,8)`, `(5,10)`, `(6,12)`, `(8,16)`, `(8,20)`
- cutoff: 2023, 2024, 2025
- 총 56개 구조, seed 5개 비교

intrinsic 기준 가장 안정적인 구조는 physical PCA8 + KMeans `(2,4)`였으며 평균 seed ARI는 0.9023이었다. 그러나 downstream에서는 physical+control PCA8 + GMM `(2,4)`가 가장 좋았다.

| 투수 군집 | 2024 단독 BSS | performance 혼합 BSS |
|---|---:|---:|
| physical+control GMM `(2,4)` | **783.7118** | **800.8235** |
| physical KMeans `(2,4)` | 783.0154 | 800.3845 |
| physical+control KMeans `(2,4)` | 783.0195 | 799.8083 |
| physical GMM `(2,4)` | 781.9346 | 799.2541 |

기존 adjusted 단일 784.5568 및 기존 최고 혼합 801.1471보다 낮으므로 군집 자체는 채택하지 않는다.

## 4. 군집 피처 ablation

physical+control GMM `(2,4)`에서 직접 투입 방식을 분리했다.

| 투입 방식 | 2024 BSS | 결론 |
|---|---:|---|
| centroid style만 | 784.2139 | 가장 덜 악화, 기존 대비 -0.343 |
| soft 확률+style | 781.3138 | 불확실성 피처가 방해 |
| hard ID+style | 780.5185 | hard cluster ID는 제외 |

군집은 예측 피처보다 상성 집계의 계층으로 사용하는 것이 적합하다.

## 5. 타자 군집과 상성 residual

타자 KMeans K를 `(3,4)`, `(4,6)`, `(6,8)`, `(8,12)`로 비교하고 smoothing 500/1000, 반감기 1/2/무가중을 탐색했다.

### 최고 2024 probe

- 투수 GMM `(2,4)` + 타자 KMeans `(3,4)`
- smoothing 500, 반감기 1년
- 2023 residual로 Ridge 학습 후 2024 적용
- adjusted 단독: 784.5568
- matchup correction: 794.5420
- performance 혼합: **810.7283**

다만 동일 구조가 2022→2023에서 Brier를 `+0.0000128` 악화시켜, 이 조합은 성능 probe로만 보존하고 최종 권장안에서는 제외한다.

### 강건 후보

- 투수 GMM `(2,4)` + 타자 KMeans `(3,4)`
- smoothing 1000, 반감기 1년, Ridge alpha 10
- F23 Brier 변화: 약 `-0.000010`
- F24 Brier 변화: 약 `-0.000004`
- 2024 corrected 단독 BSS: **786.2754**
- 2024 performance 혼합 BSS: **806.4875**
- 혼합 시 corrected 모델 가중치: 0.532

기존 최고 혼합 801.1471 대비 약 `+5.34 BSS`이며 두 시간 전이에서 방향이 일치한다.

## 6. 포수 요구 반대 방향(reverse) 독립 실험

3번 실패는 전체 성공률과 섞지 않고 별도 residual로 구성했다. reverse 발생률에서 `시즌×투수손×타자손×볼카운트` 평균을 제거한 뒤, 투수 유형×타자 유형별 잔차를 집계했다.

### 기존 성공률 타자 군집 재사용

- reverse smoothing 2000, 반감기 1년, Ridge alpha 1000
- 성공 correction 0.10 + reverse correction 0.55
- F23 ΔBrier: `-0.00000922`
- F24 ΔBrier: `-0.00003075`
- 2024 단독 BSS: 796.8674
- performance 혼합 BSS: **810.0982**

### reverse 전용 타자 군집

- 탐색: KMeans/GMM × K `(2,3)~(8,12)` × smoothing 1000/2000/4000 × 반감기 1/2
- 총 60개 군집 구조, Ridge 포함 180개 검증
- 최고: KMeans Left 4 / Right 6, smoothing 1000, 반감기 1년, Ridge alpha 10000
- 성공 correction 0.25 + reverse correction 0.65
- F23 ΔBrier: `-0.00001046`
- F24 ΔBrier: `-0.00003030`
- 2024 단독 BSS: 796.6850
- performance 혼합 BSS: **810.2572**

reverse 전용 타자 군집은 성공률 기준 군집보다 robust 목적함수가 좋았고, F23·F24가 모두 같은 개선 방향을 보였다. 3번 실패를 별도 케이스로 다루자는 가설이 검증된 결과다.

### reverse 군집 seed 안정화

- KMeans seed: 17, 43, 97, 2026, 4099
- Ridge alpha: 1000/10000/100000
- 모든 seed 부분집합 31개와 공통 correction scale을 비교
- 최종: seed 17/2026/4099 correction 평균, Ridge alpha 1000
- 성공 correction 0.25 + reverse correction 0.55
- F23 ΔBrier: `-0.00001436`
- F24 ΔBrier: `-0.00003676`
- 2024 단독 BSS: 799.2716
- performance 혼합 가중치: 0.6085
- 2024 혼합 BSS: **812.7040**

단일 seed 17의 강건 목적함수가 근소하게 가장 좋았지만, seed 17/2026/4099 평균은 2024 혼합 성능이 더 높고 군집 초기값 의존성을 낮추므로 최종안으로 채택했다.

## 7. 실패한 방식

- matchup 피처를 XGBoost에 직접 추가: BSS 748.96~761.01
- 타자 GMM: 두 fold 동시 개선 조합 없음
- physical-only 또는 KMeans 투수 군집으로 상성 구성: 2024 악화
- hard cluster ID 직접 투입: 성능 악화
- 큰 K: 소수 군집과 seed 불안정성 증가

## 8. 2025 frozen artifact

`final/robust_matchup_v1/`에 다음을 생성했다.

- 2025 pitcher lookup: 792행
- batter-hand lookup: 862행
- pitcher type×batter type pair table: 84셀
- Ridge imputer/scaler/coef JSON
- CSV 기반 frozen inference 모듈
- sample test 5행 smoke feature
- reverse pair table 및 reverse 전용 batter lookup
- reverse Ridge imputer/scaler/coef JSON

최종 artifact는 main/TrackMan 2024까지만 사용한다. test 전체 분포를 집계하거나 재군집하지 않는다. 정방향 batch, 역순 batch, 1행씩 실행한 결과가 완전히 동일했다.

## 9. 제출 후보

| 파일 | 설계 | F23 ΔBrier | F24 ΔBrier | 2024 혼합 BSS | 역할 |
|---|---|---:|---:|---:|---|
| `submit_009.zip` | 성공 상성 60% shrink | -0.00001216 | -0.00001281 | 805.5638 | 안정·다양성 |
| `submit_010.zip` | 성공 상성 100% | -0.00000996 | -0.00000411 | 806.4875 | 성공 상성 공격형 |
| `submit_011.zip` | 성공 + reverse 상성 | -0.00000922 | -0.00003075 | 810.0982 | reverse 검증형 |
| `submit_012.zip` | 성공 + reverse 전용 타자 군집 | -0.00001046 | -0.00003030 | 810.2572 | 이전 최고 |
| `submit_013.zip` | reverse 전용 타자 군집 3-seed 평균 | -0.00001436 | -0.00003676 | **812.7040** | 최종 최우선 |

모든 ZIP은 245,789행 전체 로컬 추론, CRC, 루트 구조, 확률 유효성 검사를 통과했다. 첫 제출은 013이 우선이다. 두 번째 제출은 정보 다양성을 중시하면 009, reverse 재현성을 중시하면 011을 사용한다.

### Public LB 확인

- `submit_013`: **895.404000081**
- 이전 최고 XGBoost V1: 873.0751046509
- 상승폭: **+22.3288954301**
- 해석: reverse를 별도 잔차로 모델링한 효과가 2025 평가 데이터에도 전이됐다. F23/F24 동시 개선을 기준으로 고른 seed 평균화 역시 유효했다.

## 10. 다음 실행 순서

1. `submit_013.zip` Public LB 895.404 기록 완료
2. 남은 제출 슬롯은 `submit_012`에 즉시 쓰지 않고 개선된 reverse 후보에 우선 배정
3. reverse count-bucket interaction과 pair-table ensemble을 F23/F24로 검증
4. LB 결과를 `SUBMISSION_LOG.md`에 즉시 기록하고 최종 2개를 확정

## 11. 주요 파일

- 계획: `../PITCHER_CLUSTER_MATCHUP_PLAN.md`
- 프로필 감사: `reports/profile_audit.json`
- 군집 전체 탐색: `reports/cluster_summary_stage1.csv`
- 군집 downstream: `reports/cluster_feature_validation_stage1.csv`
- KMeans 상성 24조합: `reports/matchup_robust_screen_kmeans24.csv`
- reverse 기본 탐색: `reports/reverse_matchup_screen.csv`
- reverse 전용 타자 군집 180개: `reports/reverse_batter_cluster_screen.csv`
- 성공+reverse 결합: `reports/dual_reverse_batter_tuning.json`
- reverse seed 탐색: `reports/reverse_batter_seed_summary.json`
- 2025 artifact: `final/robust_matchup_v1/manifest.json`
- 제출 기록: `../../../submit/2026-08-12/SUBMISSION_LOG.md`

## 12. 투수·타자 임베딩 및 공동 군집 심화 실험

### 탐색 범위

- 기존 투수 군집 56개: physical/combined × KMeans/GMM × PCA 8/16 × 좌·우 K
- 멀티뷰 투수 군집 48개: 물리 임베딩과 제구 임베딩을 별도 PCA 후 가중 결합
- 공동 SVD 54개: 투수×타자 reverse 잔차 행렬의 수축 100/500 × 차원 4/8/16 × 투수 K 3종 × 타자 K 3종
- 안정형 공동 SVD 108개: unit normalization/quantile normalization 추가
- 상위 2개 공동 SVD에 대해 seed 17/43/97/2026/4099 및 모든 seed 부분집합 검증

### 핵심 결과

| 후보 | 구조 | F23 ΔBrier | F24 ΔBrier | 2024 최종 BSS | 최소 투수/타자 군집 |
|---|---|---:|---:|---:|---:|
| 기존 `submit_013` 재현 | reverse 타자군집 3-seed | -0.00001436 | -0.00003676 | 812.7040 | - |
| 공격형 공동 SVD | λ100, dim4, unit, 투수 4/8, 타자 6/8 | -0.00000758 | -0.00002443 | **815.0828** | 14 / 7 |
| 안전형 공동 SVD seed97 | λ100, dim4, unit, 투수 3/6, 타자 3/4 | -0.00003120 | -0.00001853 | **814.8309** | 20 / 21 |
| 안전형 공동 SVD 5-seed | 같은 구조, correction 평균 | 약 -0.000029 | 약 -0.000019 | **814.7109** | 20 / 21 |
| 안전형 robust 설정 | 성공 0.25, 기존 0.65, SVD 0.175 | -0.00003261 | -0.00003334 | 814.2908 | 20 / 21 |

### 해석

- 기존 combined GMM PCA8 좌2/우4가 기존 56개 중 여전히 1위였다. K를 늘리면 희소 pair cell과 연도 불안정성이 증가했다.
- 멀티뷰는 robust 목적함수에서 개선됐지만 기존 reverse correction과 2024 상관이 `0.887`이라 최종 최적화에서는 가중치가 0에 수렴했다.
- 공동 SVD correction은 기존 reverse correction과 2024 상관이 약 `0.09`로 낮아 실질적인 보완 신호가 됐다.
- 표준화 SVD 군집은 모든 조합에 1명짜리 군집이 생겼다. unit normalization 후 최소 군집이 7~21명으로 개선되면서 BSS도 상승했다.
- 공격형은 seed 간 평균 상관이 F23 `0.897`, F24 `0.913`; 안전형은 F23 `0.978`, F24 `0.985`로 훨씬 안정적이다.
- 따라서 실제 다음 제출은 안전형을 우선하고 공격형은 비교 후보로 둔다.

### 결과 파일

- 기존 투수 군집 전수: `reports/deep_pitcher_cluster_best.csv`
- 멀티뷰 투수 군집: `reports/multiview_pitcher_cluster_best.csv`
- 공동 SVD 기본 탐색: `reports/joint_svd_cluster_best.csv`
- 최종 혼합 기준 공동 SVD 전수: `reports/joint_svd_outer_search_with_stability.csv`
- 안정형 SVD 전수: `reports/joint_svd_stable_search_with_stability.csv`
- seed 안정성: `reports/joint_svd_seed_stability.json`
- 실험 코드: `src/joint_svd_cluster_search.py`, `src/joint_svd_outer_search.py`, `src/joint_svd_seed_stability.py`

## 13. 대형 XGBoost 용량 확장

- 18-leaf 기준 BSS 784.5568에서 20-leaf는 785.4658, 24-leaf diverse는 785.0671로 개선됐다.
- 32/48/64/96/128 leaves의 BSS는 각각 780.8043/774.0135/761.8727/744.4284/726.3679로, 단일 거대화는 명확히 과적합했다.
- 24-leaf diverse는 AUC 0.550120으로 가장 높아 교체 모델이 아니라 추가 expert로 사용했다.
- seedbag은 F24에서 포화 또는 하락해 seed 0을 유지했다.
- 모든 fold에 동일한 anchor 0.10 + large 0.90을 적용하고, 대형 OOF 잔차 기준으로 success 0.20/reverse 0.575 correction을 재학습했다.
- 기존 `submit_013` corrected insight를 보존한 3-way 혼합의 2024 BSS는 **813.4317**로 812.7040 대비 **+0.7277**이다.
- F23/F24 최적 외부 가중치 차이가 커 `submit_014`는 고위험 비교 후보이며, `submit_013`이 계속 안전 기준이다.
- 상세 결과: `../LARGE_MODEL_RESULTS.md`
