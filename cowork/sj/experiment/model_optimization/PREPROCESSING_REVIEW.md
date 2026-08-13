# 전처리·이상치 처리 검토

## 1. 결론

| 대상 | 현재 판단 | 적용 원칙 |
|---|---|---|
| 메인 학습 데이터 | 논리 범위 위반이 없어 행 삭제 불필요 | 유효한 희귀 상황을 이상치로 오인하지 않고 원본 보존 |
| 누적 rate | 0·1 극단값은 오류보다 저표본 문제 | 제거·winsorize 대신 표본 수 기반 smoothing |
| 누적 count | 긴 오른쪽 꼬리가 있으나 정상적인 경력 차이 | raw와 `log1p`를 함께 사용 |
| `li`·점수차 | 큰 값도 실제 경기 상황일 가능성이 높음 | raw 유지, log/capped 파생값만 추가 비교 |
| TrackMan 투구 로그 | 소수의 극단 측정값이 존재 | 행 전체 삭제 대신 metric별 temporal winsorization 또는 결측 처리 |
| 투수 임베딩·군집 | 이상치보다 217차원 대비 투수 수 부족이 더 큰 문제 | feature 축소, robust scaling, missing indicator, seed 안정성 평가 |
| 공동 SVD | 입력 residual이 bounded이고 이미 수축됨 | 추가 이상치 삭제보다 빈도·희소 pair shrinkage 유지 |
| 테스트 추론 | 테스트 전체 분포 사용 금지 | 모든 기준값·imputer·scaler·cluster를 학습 데이터에서 고정 |

## 2. 데이터 품질 감사

### 메인 데이터

다음 항목의 위반은 모두 0건이었다.

- 볼·스트라이크·아웃 카운트 범위
- 주자 플래그와 `num_runners_on` 일치
- `run_total_before = run_top_before + run_bot_before`
- 모든 `asof_*_rate`의 0~1 범위
- 승리확률 0~100, `li ≥ 0`
- `asof_pitcher_pitchmix_n = asof_pitcher_n`

따라서 메인 데이터에서는 **이상치 행 삭제를 하지 않는다.** 점수차, 연장 이닝, 높은 LI는 드물더라도 정상 경기 상황이므로 삭제하면 오히려 일반화가 나빠질 수 있다.

### TrackMan 원시 측정값

| 변수 | 최소 | 0.1% | 99.9% | 최대 | 과거 기준으로 2024 clipping 대상 |
|---|---:|---:|---:|---:|---:|
| rel_speed | 71.68 | 103.13 | 156.10 | 160.66 | 0.161% |
| spin_rate | 434.90 | 768.37 | 3,129.38 | 3,695.92 | 0.320% |
| induced_vert_break | -81.81 | -54.19 | 71.73 | 153.33 | 0.249% |
| horz_break | -78.92 | -55.77 | 62.37 | 103.70 | 0.150% |
| extension | -0.39 | 1.23 | 2.26 | 3.85 | 0.265% |
| rel_height | 0.10 | 0.32 | 2.11 | 2.51 | 0.008% |
| rel_side | -2.29 | -1.14 | 1.31 | 1.67 | 0.017% |
| zone_speed | 62.75 | 94.22 | 142.61 | 148.30 | 0.191% |

극단값 비중은 작지만 현재 TrackMan 집계는 투구별 값을 그대로 `mean/std`로 요약한다. 특히 extension 음수, induced break 153 같은 값은 투수 프로필 평균과 표준편차를 흔들 수 있다.

권장 방식:

1. 각 검증 cutoff보다 과거인 TrackMan만 사용한다.
2. source season×pitch type group별 0.1~99.9% 또는 robust z ±6 기준을 학습한다.
3. 이상 metric만 winsorize하거나 NaN으로 바꾸고 **투구 행 전체는 삭제하지 않는다.**
4. clipped 비율을 투수-시즌 품질 피처로 남긴다.
5. mean/std 외에 median/IQR을 compact 후보로 비교한다.

## 3. 현재 전처리에서 발견한 중요 문제

### 저장된 투수 프로필 캐시 노후화

| cutoff | 임베딩 대상 투수 | 최신 failure component 결측 투수 | 결측률 |
|---:|---:|---:|---:|
| Val2023용 cutoff 2023 | 210 | **181** | **86.19%** |
| Val2024용 cutoff 2024 | 209 | 26 | 12.44% |
| 2025 추론용 cutoff 2025 | 225 | 0 | 0.00% |

저장된 cutoff 2023 프로필에서 2022년 reverse·middle·outside residual이 통째로 빠져 있었다. 현재 생성 코드로 다시 계산하면 2022년 최신 시즌 390명 중 389명에서 정상 복원된다. 즉 생성 로직 자체보다 **과거 캐시가 최신 failure label보다 먼저 만들어진 문제**다.

영향:

- Val2023 투수 군집과 이를 사용한 상성 안정성 평가는 반드시 재실행해야 한다.
- Val2024도 일부 영향을 받으므로 함께 재실행한다.
- 2025 최종 프로필에는 해당 결측이 없지만, 모델 선택 근거인 과거 검증이 바뀔 수 있다.
- 공동 SVD는 메인 failure label에서 직접 계산하므로 이 캐시 문제의 직접 영향은 없다.

조치:

- `build_profiles.py`에 schema version, failure label coverage, 고표본 결측 fail-fast 검사를 추가했다.
- 기존 결과를 덮어쓰기 전에 clean profile v2로 검증 실험을 다시 수행한다.

## 4. 투수 임베딩·군집 전처리 감사

현재 파이프라인은 `median imputation → RobustScaler(10~90%) → ±5 clip → PCA → hand별 GMM`이다.

| cutoff/손 | 투수 수 | 입력 차원 | ±5 clip 비율 | 최소 군집 | seed ARI |
|---|---:|---:|---:|---:|---:|
| 2024/좌 | 49 | 217 | 0.000% | 12 | 0.738 |
| 2024/우 | 160 | 217 | 0.040% | **2** | 0.659 |
| 2025/좌 | 56 | 217 | 0.017% | 7 | 0.539 |
| 2025/우 | 169 | 217 | 0.014% | 11 | 0.981 |

clip되는 셀은 0.04% 이하인데도 clip을 제거했을 때 현재 군집과의 ARI가 2024 우투수 0.310, 2025 우투수 0.409였다. 소수 극단값보다 **217차원/160명 수준의 고차원 불안정성**이 핵심 문제다.

또한 StandardScaler는 일부 cutoff에서 1명짜리 군집과 낮은 seed 안정성을 만들었다. Quantile-normal은 군집 크기를 균형화하기도 했지만 현재 군집과 크게 달랐고 cutoff별 seed 안정성이 일정하지 않았다. 따라서 intrinsic 지표만 보고 교체하지 않고 downstream Brier로 선택한다.

### 개선 방향

| 후보 | 처리 | 기대 효과 |
|---|---|---|
| E0 | clean profile v2 + 현재 robust5 | 캐시 수정 후 새로운 기준선 |
| E1 | eligible 집단에서 결측률 30% 초과 피처 제거 | 중앙값으로 채워진 무정보 차원 제거 |
| E2 | cutoff 전체에서 공통인 20~40개 compact 피처만 사용 | p/n 비율과 군집 변동 감소 |
| E3 | median imputation + missing indicator | 구종 부재와 센서 결측을 구별 |
| E4 | robust5 vs robust3 vs quantile-normal | 이상치 민감도와 군집 균형 비교 |
| E5 | season×pitch group robust z + latest raw 병행 | 2022 측정 체계 단절 완화 |

우선순위는 더 강한 clipping이 아니라 **clean cache → 피처 축소 → missing indicator → scaler 비교**다.

## 5. 모델 계열별 권장 전처리

### XGBoost·CatBoost

| 피처 종류 | 권장 처리 |
|---|---|
| 범주형·ID | 학습 fold에서만 mapping하고 미등록 값은 unknown 처리 |
| 누적 count | raw + `log1p(count)` |
| 누적 rate | raw + empirical-Bayes smoothing + reliability |
| 최근 1·3·5경기 | 누적 대비 delta, positive/negative 분리, 표본 기반 shrinkage |
| 결측 | native missing 유지 + 의미 있는 cold-start flag 추가 |
| LI·점수차 | raw 유지, `log1p(li)`·capped 파생값을 추가 ablation |
| 중복 피처 | 완전 중복은 제거 후보로 비교 |
| 스케일링 | 트리 모델에는 표준화 불필요 |

rate 0·1을 단순 clip하면 신인의 표본 부족 신호와 실제 성향을 섞는다. 이미 성능이 확인된 smoothing 방식이 더 적절하다.

### 신경망 투수 임베딩

현재는 `median → mean/std`이며 z-score clipping과 missing indicator가 없다. TrackMan 및 최근 경기 rate의 극단값에 트리 모델보다 민감하다.

권장 실험:

- count는 `log1p` 후 robust scaling
- rate는 smoothed rate와 reliability를 입력
- 연속값은 train fold 기준 median/IQR 또는 mean/std 후 z ±5
- 결측값 0 대체와 별개로 feature별 missing indicator 제공
- TrackMan tower에는 `tm_available`, source season gap, clipped-rate 품질 피처 제공
- 모든 scaler는 각 Val 학습 구간에서만 fit

### 투수×타자 공동 SVD

- reverse residual은 범위가 제한되고 `matrix_lambda`와 reliability weighting으로 이미 수축된다.
- 개별 행 이상치 제거는 필요하지 않다.
- unit normalization은 최소 군집 크기와 성능을 개선해 유지한다.
- 추가 후보는 degree normalization, 최소 유효 표본 수, pair reliability 조정이다.
- SVD·군집·scaler는 cutoff 이전 이력으로만 fit하고 test에서 재학습하지 않는다.

### Ridge residual 보정

현재 `median imputation → StandardScaler → Ridge → correction ±0.05`는 입력이 4~8개이고 대부분 bounded이므로 적절하다. `known/reliability`가 이미 결측 정보를 표현하므로 우선 유지한다.

## 6. 실험 및 채택 기준

| 단계 | 실험 | 비교 기준 |
|---:|---|---|
| 1 | clean profile v2 재생성 | component coverage와 leakage audit 통과 |
| 2 | current/compact/missing-indicator 임베딩 | Val2023·Val2024 전체 및 game_type=R ΔBrier |
| 3 | TrackMan raw/winsor/season-z 집계 | 단일 BSS, 앙상블 BSS, 예측 상관 |
| 4 | robust5/robust3/quantile 군집 | 최소 군집, seed ARI, downstream Brier |
| 5 | 신경망 robust preprocessing | Brier, calibration gap, cold-start 성능 |

채택 조건:

- Val2023와 Val2024 전체 Brier가 모두 악화되지 않을 것
- 모수가 큰 `game_type=R`에서 두 검증연도 모두 악화되지 않을 것
- 군집 최소 크기 10명 권장, 5명 미만 군집은 제외 또는 상위 계층으로 병합
- 성능 차이가 작으면 더 적은 피처와 더 안정적인 seed 결과를 선택
- 최종 전처리 상수는 2024년까지의 학습 데이터로 다시 fit해 frozen artifact로 저장

## 7. 생성된 감사 파일

- `audit_preprocessing.py`
- `audit_embedding_preprocessing.py`
- `preprocess_train_numeric_audit.csv`
- `preprocess_train_logical_checks.csv`
- `preprocess_trackman_numeric_audit.csv`
- `preprocess_trackman_season_audit.csv`
- `preprocess_trackman_transfer_clip.csv`
- `preprocess_profile_numeric_audit.csv`
- `preprocess_embedding_variants.csv`
- `preprocess_embedding_variant_ari.csv`
- `preprocess_embedding_columns.csv`

