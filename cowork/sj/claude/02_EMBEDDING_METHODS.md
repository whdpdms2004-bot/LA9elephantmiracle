# TrackMan 임베딩 방식 서치 정리

작성: 2026-08-12 / 목적: 가변길이 투구 집합(pitch set/sequence)을 투수-시즌 수준 고정길이 벡터로 바꾸는 방법을 조사·평가하고, 이 대회 조건에 맞춰 순위를 정한다.

핵심 전제 두 가지:
1. 기존 임베딩 시도가 실패한 원인은 임베딩 계열이 아니라 **주입 방식과 3개의 구조적 결함**이다 (→ `00_ASSESSMENT.md` §7).
2. 이 문제에서 임베딩의 가치 기준은 "재구성 품질"이 아니라 **시즌 간 안정성**이다. 2025는 TrackMan이 없으므로 2024까지의 임베딩을 그대로 외삽해야 한다.

---

## 1. 안정성이 유일한 통화다

2025 예측은 2024까지의 표현을 외삽하는 것이므로, **YoY 재현성이 낮은 임베딩은 학습 fold에서 아무리 좋아도 2025에서 죽는다.** 공개 지표의 YoY 안정성 참고값:

| 지표 | YoY 안정성 | 성격 |
|---|---|---|
| Stuff+ (aStuff+, 750구 기준) | R² ≈ **0.702** | 물리량 기반 → 안정 |
| Kirby Index (커맨드) | R² ≈ 0.50 | 릴리스 반복성 |
| Location+ | R² ≈ 0.39 | 위치 커맨드 → 불안정 |
| 팀 자체 Kirby 재현 (Statcast 2017→19) | r 0.51 / 0.58 (R² 0.26~0.34) | 원문보다 낮음 |

**따라서 방법 선택 기준에 "YoY 안정성"을 1급 축으로 넣는다.** 이것이 아래 §2에서 contrastive 계열을 높게 평가하는 이유다(안정성이 손실함수에 직접 인코딩됨).

또한 GBDT가 최종 소비자라는 점도 중요하다. tabular에서 트리가 딥러닝을 이기는 것이 지배적 결과이므로(Grinsztajn et al. 2022), **딥 임베딩은 "좋은 통계량" 대비 lift를 증명할 의무가 있고, 증명 못 하면 버린다.**

---

## 2. 방법 카탈로그

### 2-1. 무학습 분포 임베딩 — 가성비 최상, 먼저 할 것

| ID | 방법 | 내용 | 비용 | 안정성 | 비고 |
|---|---|---|---|---|---|
| **N1** | **분위수·모멘트 signature** | 투수×시즌×구종군별 각 물리량의 `[mean, sd, p10/25/50/75/90, skew]` + 구종 usage. EB 축소 | 0.5일 | 상 | **기준선. 모든 딥 방법이 이걸 넘어야 한다.** 현행 72개는 이것의 일부(mean/sd만) |
| **N2** | **Kernel Mean Embedding (RFF)** | 투구 벡터 `x`를 random Fourier feature `z(x) = √(2/D)·cos(Wx+b)`로 사상 후 투수별 평균. 사실상 "학습 없는 DeepSets" | **1시간** | 상 | RBF 커널의 평균 임베딩. 결정론적·순열불변·재현 100%. **가장 먼저 시도할 것** |
| **N3** | **Sliced-Wasserstein / 분위수함수 임베딩** | 랜덤 방향 L개(예: 64)로 투영 → 각 방향의 분위수 벡터(예: 9개) → 576차원 → PCA 8~16 | 0.5일 | 상 | 리그 평균을 reference로 두면 OT 거리를 유클리드로 근사. 해석 가능 |
| **N4** | **투수-시즌 GMM 파라미터** | 6D 물리공간(velo, IVB, HB, spin, ext, rel_side)에 K-component GMM → weight/mean/cov-trace를 피처화 | 1일 | 중상 | xCTRL이 plate location에 동일 기법 적용해 YoY r=0.65 달성. 우리는 release/movement 공간에 이식 |
| **N5** | **SVD/NMF on `pitcher × (구종군 × 볼카운트 × 타자손)` rate 행렬** | 24~96차원 count/rate 행렬 → TruncatedSVD 8~32 | 0.5일 | 중상 | Tier E1(48차원) 압축의 정석. NMF는 "구종 배합 topic"으로 해석 가능. **팀의 공동 SVD 성공(815.08)과 같은 계열** |
| **N6** | **PCA per-pitch → 집계** | 물리 8차원을 PCA 4~6으로 줄인 뒤 투수별 `[mean, sd, 분위수]` | 0.2일 | 상 | AE를 시도하기 전 반드시 이걸 먼저. lift 없으면 AE 중단 |
| **N7** | **구종 전이 행렬 / SGT** | `구종군(t) → 구종군(t+1)` 4×4 전이확률 flatten (+ 카운트 조건부). Sequence Graph Transform 계열 | 0.3일 | 중상 | 3.59M구·1,523투수 연구에서 13개 시퀀스 클러스터 도출. usage%와 중복 가능성 있음 |

### 2-2. 학습 기반 — 조건부로만

| ID | 방법 | 내용 | 비용 | 안정성 | 판정 |
|---|---|---|---|---|---|
| **L1** | **Contrastive: same-pitcher-different-game** | 같은 투수의 서로 다른 등판에서 뽑은 두 pitch subset을 positive, 다른 투수를 negative → NT-Xent. 인코더는 DeepSets | 2일 | **최상** | **★ 이 문제의 최적 pretext.** 손실함수가 곧 "등판 간 재현되는 시그니처만 남겨라"이므로 §1이 요구하는 안정성이 목적함수에 내재. 경기별 캘리브레이션 드리프트·상대·구장 효과가 자동 제거 |
| **L2** | **DeepSets** | per-pitch MLP φ → `mean+max+sd` pooling → ρ | 1.5일 | 중상 | sum-pooling DeepSets는 "학습된 통계량" = N1의 상위집합. N1 대비 lift 증명 필요 |
| **L3** | **등판 내 시퀀스 GRU + next-pitch pretext** | 등판 시퀀스로 다음 구종/구속/카운트 전이 예측 → hidden state 또는 학습된 pitcher token | 2일 | 중 | 커맨드는 시퀀스 의존적(피로·리듬)이라 N1이 못 잡는 축. TrackMan에 결과 라벨이 없어 pretext가 약하지만 카운트 전이는 가능 |
| **L4** | **Multi-task supervised ID 임베딩 (재설계)** | 문맥(count, pitch_no, 타자 손, 이닝) + pitcher ID embedding → **구종/구속/IVB/HB/extension** 예측. 타깃은 절대 `control_success`가 아님 | 1일 | 중 | 기존 실패본의 수정판. "문맥 조건부 잔차"를 학습하므로 단순 평균과 다른 정보를 담는다. cold start 주의 |
| **L5** | **AE / VAE per-pitch → 집계** | 물리벡터 오토인코더 latent를 투수별 집계 | 0.5일 | 중 | N6(PCA)이 먼저. 차이가 없으면 폐기 |
| **L6** | **Set Transformer / Perceiver** | ISAB + PMA pooling | 2~3일 | 중 | 투수 물리량 집합은 원소 간 고차 상호작용이 약해 attention 이득이 작다. **ROI 낮음 → 후순위** |
| **L7** | **Tabular SSL (VIME / SCARF / SubTab / TabNet)** | per-pitch 마스킹·손상 후 복원 | 1~2일 | 중 | 라벨 희소 상황에서 이득이 크고, 여기선 pitch 단위 정보가 풍부하고 소비자가 GBDT라 이득이 얇다. per-pitch 임베딩을 평균내면 N1으로 붕괴 → **후순위** |

### 2-3. 제외 권고

| 방법 | 제외 이유 |
|---|---|
| **타깃 지도 ID 임베딩** (`control_success`로 학습) | 사실상 학습된 target encoding. 팀이 이미 전체 temporal TE로 **BSS 361.06** 기록. 기존 실패본 `pitcher_embedding_00~15`가 정확히 이 형태 |
| **그래프 임베딩** (node2vec / metapath2vec / GNN on 투수-타자 이분그래프) | 야구 대결 그래프는 리그·팀·일정 구조를 반영할 뿐 → 임베딩이 "소속팀·연도" 대리변수가 되어 과적합 |
| **Pitch tunneling** | release angle + approach angle + **plate location** 필수. 원저자도 "터널링 대부분은 위치의 산물"이라 명시. 대체는 Tier A5(구종 간 릴리스 응집도) |
| **Kirby Index 원형** | VRA/HRA 계산에 vx0~az 운동학 필요, TrackMan에 없음. 대체는 Tier A(릴리스 위치 SD 축소판) |
| **Stuff+ 점수 자체** | run value 타깃이 TrackMan에 없음. 대체는 Tier D1(입력 피처 세트만 차용) |

---

## 3. 주입 방식 — 여기가 진짜 승부처

팀의 실증 데이터가 명확한 방향을 가리킨다.

| 주입 방식 | 팀 실측 결과 |
|---|---|
| 군집 ID를 GBDT에 직접 투입 | 780.52 (hard ID+style), 781.31 (soft 확률), 784.21 (centroid만) — **전부 기준선 784.56 미달** |
| matchup 피처를 GBDT에 직접 투입 | **748.96 ~ 761.01** — 크게 미달 |
| 임베딩 48차원을 GBDT에 직접 투입 | **395.38** (AUC 0.5405 < V1 0.5479) — 참사 |
| 군집을 **correction의 평활 계층**으로 사용 | **806.49 ~ 815.08** — 유일한 성공 패턴 |

**결론: 이 파이프라인에서 "표현을 GBDT 원시 피처로 넣는다"는 방식은 4번 시도해서 4번 실패했다.** 임베딩도 같은 방식으로 넣으면 5번째 실패가 된다.

### 주입 방식 4안

| ID | 방식 | 설명 | 우선순위 |
|---|---|---|---|
| **M1** | GBDT 원시 피처 | 임베딩 차원을 그대로 feature로 | **최하.** 재시도하되 **4~8차원으로만**, 그리고 `has_trackman` 분리·2019/2020 NaN 처리를 반드시 수정한 뒤 |
| **M2** | **correction의 평활 단위** | 현행 correction은 투수 군집 × 타자 군집 셀에서 reverse/success 잔차를 집계하고 Ridge로 보정한다. **군집(hard 할당)을 임베딩 커널 이웃(soft)으로 교체**한다: `w(p, p') = exp(−‖e_p − e_p'‖²/2σ²)` 가중 평균으로 잔차를 평활 | **★ 1순위.** 검증된 성공 패턴의 직접 연장. hard K 선택 문제(좌2~8/우4~20 탐색)가 σ 하나로 대체되어 탐색 공간도 줄어든다 |
| **M3** | **residual expert** | 임베딩만 쓰는 소형 모델의 시간 OOF 예측을 앙상블 멤버로 추가 | **2순위.** 공동 SVD가 기존 correction과 상관 **0.09**로 이득을 본 전례가 있다. 상관이 낮으면 약한 모델도 앙상블에서 값을 한다 |
| **M4** | **상성 행렬 인자화** | 투수×타자 reverse/middle 잔차 행렬을 임베딩으로 저차원 근사 | **2순위.** 이미 공동 SVD로 부분 시행(815.08). **middle 잔차 행렬은 미시행** — R의 최근 악화 주범이 middle이므로 여기가 비어 있다 |

### M2 구현 스케치

```python
# fold별: 학습 시즌 TrackMan → 투수 임베딩 e_p (as-of, 정규화)
# 1) 잔차 준비: 시간 OOF 기준예측 대비 reverse/middle/success 잔residual
# 2) 커널 가중 평활 (hard 군집 대체)
K = np.exp(-cdist(E_p, E_p, 'sqeuclidean') / (2 * sigma**2))    # sigma: 그리드
K = K / K.sum(1, keepdims=True)
r_smooth = K @ r_raw                          # 투수축 평활
# 타자축도 동일하게 (타자 임베딩은 상대한 투구의 TrackMan 프로파일 평균으로 대체 구성)
# 3) smoothing 강도 n_eff 기반 EB 축소 → Ridge(alpha) 보정 → 기존 blend에 가중 투입
```

핵심 이점: hard 군집은 경계에서 정보를 잃고 K와 seed에 민감하다(팀이 seed 3~5개 평균으로 대응 중). 커널 평활은 그 두 문제를 동시에 없앤다.

---

## 4. 최종 랭킹 (이 대회 기준)

| 순위 | 항목 | 조합 | 예상 효과 | 비용 | 핵심 리스크 |
|---:|---|---|---|---|---|
| 1 | **N2 KME + M2 커널 평활** | 무학습 임베딩 → correction 평활 | 상 | 1일 | σ 선택 과적합 → fold별 고정 |
| 2 | **N5 SVD(구종×카운트×타자손) + M4** | 전략 성향 압축 → middle 잔차 인자화 | 상 | 1일 | 노출 편향 → row-normalize |
| 3 | **N1 확장 signature (분위수·skew)** | 현행 mean/sd에 분위수 추가 | 중상 | 0.5일 | 차원 증가 → Tier 단위 PCA |
| 4 | **N3 Sliced-Wasserstein** | 분위수함수 임베딩 → M2/M3 | 중상 | 0.5일 | 차원 폭증 |
| 5 | **L1 Contrastive (DeepSets + NT-Xent)** | 안정성 내재 학습 → M2/M3 | 중상 | 2일 | 식별축 편향 (구속 절대값이 지배) → hard negative를 같은 구속대·같은 팀으로 제한 |
| 6 | **N4 GMM 파라미터** | 분포 형태 → M1(저차원)/M3 | 중 | 1일 | 소표본 투수에서 컴포넌트 붕괴 |
| 7 | **M3 residual expert** | 위 임베딩 중 상관 낮은 것 | 중 | 0.5일 | 약한 모델의 가중치가 0으로 수렴 (멀티뷰 SVD 사례: 상관 0.887 → 가중치 0) |
| 8 | **N7 구종 전이 행렬** | 4×4 flatten + 카운트 조건부 | 중~낮 | 0.3일 | usage%와 중복 |
| 9 | **L4 multi-task ID 임베딩 (재설계)** | 타깃을 물리량으로 교체 | 중 | 1일 | cold start (2025 신인) |
| 10 | **L2 DeepSets** | N1 대비 lift 증명용 | 중 | 1.5일 | N1을 못 넘으면 폐기 |
| 11 | N6 PCA → L5 AE | 순서대로 | 낮~중 | 0.7일 | 압축 손실 |
| 12 | **L3 시퀀스 GRU** | 피로·리듬 축 | 중 | 2일 | pretext 약함 |

**제외**: L6 Set Transformer, L7 Tabular SSL, 그래프 임베딩, 타깃 지도 ID 임베딩, 터널링.

---

## 5. 공통 구현 원칙 (기존 실패 재발 방지)

| 원칙 | 이유 |
|---|---|
| **pretext는 TrackMan 1.79M 전량으로 학습** | crosswalk 미매칭 투수도 라벨이 필요 없는 pretext에는 쓸 수 있다. 표현 학습 표본을 2배로 늘리는 무료 이득 |
| **산출은 투수-시즌 as-of, fold별 재적합** | 인코더·scaler·PCA·커널 σ 전부 fold 학습 시즌만으로 |
| **NaN을 0으로 채우지 않는다** | 기존 실패의 결함 1: 2019·2020 행 481,500개(32.64%)가 전부 0 → 시즌 대리변수화 |
| **TrackMan 가용/미가용을 임베딩 공간 안에서 섞지 않는다** | 기존 실패의 결함 2: 미가용 투수 46%에도 24차원이 채워졌다. 미가용은 NaN + `has_trackman=0` |
| **타깃(`control_success`)을 임베딩 학습에 쓰지 않는다** | 기존 실패의 결함 3. 학습된 target encoding = BSS 361.06의 재현 |
| **검증은 시간 분할 + 투수 holdout 이중** | 임베딩이 투수 base rate를 암기했는지 확인. 투수 holdout에서 무너지면 암기다 |
| **`has_trackman` 플래그 자체의 A/B** | 매칭 성공이 "출장 많은 주력 투수"와 상관 → 플래그가 대리변수로 과적합할 수 있다. 2024 실측: 가용 152,725행 성공률 49.29%·`asof_pitcher_n` 중앙값 4,440 vs 미가용 100,782행 47.58%·837 |
| **한 Tier/한 방법씩 추가하고 CI로 판정** | `01_TRACKMAN_FEATURE_CATALOG.md` §8의 G1~G6 게이트 |

---

## 6. 참고 자료

**표현학습**
- Deep Sets — https://arxiv.org/abs/1703.06114
- Set Transformer — https://arxiv.org/abs/1810.00825
- Set Norm (집합 정규화) — https://arxiv.org/abs/2206.11925
- Entity Embeddings of Categorical Variables — https://arxiv.org/abs/1604.06737
- Why tree-based models still outperform deep learning on tabular data — https://arxiv.org/abs/2207.08815
- VIME — https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html
- SCARF — https://arxiv.org/abs/2106.15147
- SubTab — https://arxiv.org/abs/2110.04361
- TabNet — https://arxiv.org/pdf/1908.07442
- Sliced Wasserstein Kernels — https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kolouri_Sliced_Wasserstein_Kernels_CVPR_2016_paper.pdf
- Distribution Regression with Sliced-Wasserstein Kernels (ICML 2022) — https://proceedings.mlr.press/v162/meunier22b.html
- Signature method primer — https://arxiv.org/pdf/2006.00873
- metapath2vec — https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf

**야구 적용**
- (batter|pitcher)2vec (Sloan 2018) — https://www.sloansportsconference.com/research-papers/batter-pitcher-2vec-statistic-free-talent-modeling-with-neural-player-embeddings
- Sequence Graph Transform on pitch sequencing — https://cdn.prod.website-files.com/5f1af76ed86d6771ad48324b/606e515d0e2589d01b6946ba_ArnavPrasad-DecodingMLB-RPpaper.pdf
- aStuff+ (입력 피처·YoY R² 0.702) — https://medium.com/@adamsalorio/introducing-my-stuff-model-2840f196cf01
- FanGraphs Stuff+ / Location+ primer — https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/
- Kirby Index — https://blogs.fangraphs.com/introducing-the-kirby-index-a-new-way-to-quantify-command/
- Revisiting the Kirby Index — https://blogs.fangraphs.com/revisiting-the-kirby-index/
- xCTRL (GMM 기반 위치 정확도, YoY r=0.65) — https://wsb.wharton.upenn.edu/introducing-xctrl-a-probabilistic-approach-to-pitch-location-accuracy/
- Release metrics → location 예측 (NCAA 2.2M구, 평균오차 15cm) — https://link.springer.com/article/10.1007/s12283-025-00497-5
- 릴리스 변동성과 성적 (344명, BB/9와 무관) — https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1447665/full
- Pitch mix variation (엔트로피·Gini) — https://community.fangraphs.com/pitch-mix-variation-and-ways-to-measure-it/
- Quantifying Pitch Tunneling — https://medium.com/@maxwellresnick/quantifying-pitch-tunneling-acc0cfcdff02
- VBGMM 구종 분류 (TrackMan) — https://link.springer.com/article/10.1007/s42081-020-00079-8
- In-game velocity changes / 피로 — https://fantasy.fangraphs.com/in-game-velocity-changes-when-fatigue-attacks/
- Beta-binomial shrinkage (야구 비율 축소) — http://varianceexplained.org/r/beta_binomial_baseball/
