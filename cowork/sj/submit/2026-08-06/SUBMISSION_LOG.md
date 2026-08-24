# 2026-08-06 제출 기록

## 요약

| 회수 | 파일 | 상태 | 모델 | Trackman | 최종 학습 | 2024 Val BSS | Public LB |
|---:|---|---|---|---|---|---:|---:|
| 001 | `submit_001.zip` | 채점 완료 | CatBoost V1 trial 71 | 미사용 | 2019~2024 전체 | 750.5711 | **838.4920422492** |
| 002 | 보관 ZIP 없음 | 실행 오류 | XGBoost V1 trial 24, sklearn wrapper | 미사용 | 2019~2024 전체 | 745.1314 | 오류 |
| 003 | `submit_003.zip` | 채점 완료 | XGBoost V1 trial 24, native Booster | 미사용 | 2019~2024 전체 | 745.1314 | **873.0751046509** |
| 004 | `submit_004.zip` | 제출 후보 | V1 OOF XGBoost+CatBoost ensemble | 미사용 | 2019~2024 전체 | **789.6734** | 대기 |

리더보드 1위 참고 점수: 약 1100  
현재 최고 팀 점수: 873.0751046509  
1위와 차이: 약 226.92

## 제출 001 — CatBoost 안정성 최고

### 파일

- 파일: `submit_001.zip`
- 크기: 212,841 bytes
- SHA-256: `780B03483EAD1D750A101321C9DE16EF9D0FBEB1A5C7BCE35B72642EC879BE36`
- 패키지: `catboost==1.2.8`
- 제출 결과: Public LB `838.4920422492`

### 설계

- 피처: 누수 없는 V1 63개
- 입력: 메인 train의 경기 상황, 선수/팀 ID, 투구 직전 `asof_*` 이력과 row-local 파생 피처
- Trackman/투수 임베딩: 사용하지 않음
- 모델: CatBoostClassifier
- 확률 보정/앙상블: 없음, 단일 모델 확률
- Optuna: CatBoost 전체 140 trial
- 선택 기준: 2023·2024 normalized Brier 가중 목적값과 최악 fold penalty
- 선택 trial: 71
- half-life: 1.7324464557
- 검증 최대 iterations: 5,033
- 2024 early-stop 최적 tree: 151, 최종 학습 tree: 152
- 주요 설정: learning rate 0.0521742, depth 7, L2 2.3071, Bayesian bootstrap, border count 128

### 학습 과정

1. Optuna fold 2023: 2019~2022 학습 → 2023 검증
2. Optuna fold 2024: 2019~2023 학습 → 2024 검증
3. 각 fold는 예측 시즌을 기준으로 과거 시즌에 지수형 최근 가중치를 적용
4. trial 선택 후 2019~2024 전체 1,475,092행 재학습
5. 최종 재학습 가중치는 2025를 기준으로 계산하여 2024를 가장 강하게 반영

### 검증 결과

| Fold | Brier | normalized Brier | BSS | AUC | Target mean | Pred mean |
|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 0.24998323 | 0.99993294 | 6.7064 | 0.520390 | 0.499957 | 0.501236 |
| 2024 | 0.24793195 | 0.99249429 | **750.5711** | 0.548944 | 0.486105 | 0.493928 |

### 해석

- 로컬 2024 BSS 750.57보다 Public LB가 약 87.92 높았다.
- Trackman 없이도 공식 기준 549.51과 기존 임베딩 모델을 넘어섰다.
- 2024 예측 평균이 실제보다 약 0.78%p 높아 이후 확률 보정 실험 가치가 있다.

## 제출 002 — XGBoost sklearn wrapper 실행 오류

### 상태

- 모델과 전체 학습은 제출 003과 동일하다.
- 평가 서버 실행 중 `TypeError: _estimator_type undefined` 발생
- 원인: `xgboost==3.1.1`의 `XGBClassifier.load_model/predict_proba` 경로와 평가 서버 `scikit-learn==1.8.0` estimator tag 호환성
- 채점되지 않았으며 Public LB 점수 없음
- 원본 ZIP은 수정 과정에서 덮어써져 별도 보관하지 못했다. 오류 메시지와 설계는 이 로그에 보존한다.

## 제출 003 — XGBoost native 수정본

### 파일

- 파일: `submit_003.zip`
- 크기: 854,940 bytes
- SHA-256: `4C238156E382489C7E67E4F19DDAB8185C68EE7A47675B8644723646FF5D8191`
- 패키지: `xgboost==3.1.1`
- 제출 결과: Public LB `873.0751046509`

### 설계

- 피처: 제출 001과 같은 V1 63개
- Trackman/투수 임베딩: 사용하지 않음
- 모델: XGBoost trial 24
- Optuna: XGBoost 전체 140 trial
- half-life: 0.4731162635
- 탐색 최대 estimators: 4,045
- 2024 early-stop 최적 tree: 951, 최종 학습 tree: 952
- grow policy: lossguide, max depth 6, max leaves 27
- learning rate: 0.00619882
- min child weight: 617.1884
- subsample: 0.96959
- column sample by tree/level: 0.60405 / 0.96186
- regularization: gamma 0.56675, alpha 1.01823, lambda 224.59509
- max bin: 512
- 확률 보정/앙상블: 없음, 단일 모델 확률

### 학습 과정

1. Optuna fold 2023: 2019~2022 학습 → 2023 검증
2. Optuna fold 2024: 2019~2023 학습 → 2024 검증
3. trial 선택 후 2019~2024 전체 1,475,092행 재학습
4. 2025 기준 최근 가중치 적용
5. 범주 매핑은 전체 train에서 동결하고 테스트 미등록 범주는 -1 처리
6. 제출 002의 sklearn wrapper를 제거하고 `xgboost.Booster + DMatrix` 네이티브 추론으로 변경

### 검증 결과

| Fold | Brier | normalized Brier | BSS | AUC | Target mean | Pred mean |
|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 0.25053940 | 1.00215760 | 0.0000 | 0.527763 | 0.499957 | 0.527594 |
| 2024 | 0.24794554 | 0.99254869 | **745.1314** | 0.548928 | 0.486105 | 0.494819 |

### 수정 검증

- 5행 샘플 추론 성공
- 수정 전 로컬 예측 범위와 정확히 동일: 0.40293086~0.47894430
- `XGBClassifier()` 호출 없음
- native `Booster.load_model`과 `DMatrix` 사용
- ZIP CRC와 Linux 권한 정상
- 파일명 14자, 30자 제한 통과

### 리더보드 결과와 해석

- Public LB: **873.0751046509**
- 제출 001 CatBoost보다 `+34.5830624017`
- 로컬 2024 BSS보다 `+127.9437312290`
- 로컬 2024에서는 CatBoost가 XGBoost보다 5.44점 높았지만, 실제 2025에서는 XGBoost가 34.58점 높았다.
- half-life 0.4731로 최근 시즌을 매우 강하게 반영한 설정이 2025 분포에 더 잘 맞았을 가능성이 높다.
- 다음 탐색은 XGBoost의 짧은 half-life와 최근 시즌 성능을 우선하되, CatBoost는 오차 다양성을 위한 앙상블 후보로 유지한다.

## 제출 004 — V1 3-fold OOF 앙상블

### 파일

- 파일: `submit_004.zip`
- 크기: 70,572,208 bytes (67.30 MiB)
- SHA-256: `76A852F63FDA5469AE167E4DC0BA972EDA652C784E00A765F622D56904BFC52C`
- 패키지: `xgboost==3.1.1`, `catboost==1.2.8`
- 제출 결과: 대기

### 후보 생성 과정

1. XGBoost 140 trial과 CatBoost 140 trial에서 각각 다중 fold 목적값 상위 4개와 2024 상위 4개의 합집합을 선택했다.
2. XGBoost 8개, CatBoost 6개를 2022·2023·2024 fold에 독립 재학습했다.
3. `2022→2023`, `2023→2024` 두 전이에서 probability/logit blend를 비교했다.
4. 비음수·합 1 가중치에 L2 `1e-4`를 적용하고 none/logit-shift/Platt/Beta/isotonic 보정을 비교했다.
5. 최적 구조는 probability blend + L2 `1e-4` + logit shift였다.
6. 최종 deployment 가중치와 보정은 가장 최근 2024 OOF로 다시 적합했다.

### OOF 검증 결과

| 전이 | Brier | normalized Brier | BSS | AUC | Target mean | Pred mean | Mean gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022→2023 | 0.24986498 | 0.99945991 | 54.0089 | 0.527247 | 0.499957 | 0.502311 | +0.002354 |
| 2023→2024 | **0.24783427** | **0.99210327** | **789.6734** | **0.549493** | 0.486105 | 0.486550 | +0.000445 |

- 기존 단일 모델 최고 2024 BSS 751.90 대비 `+37.77`
- 제출 003 XGBoost trial 24의 2024 BSS 745.13 대비 `+44.54`
- 보정 후 2024 예측 평균 오차가 약 0.045%p로 감소했다.
- 다중 fold 선택 objective: `0.9956344557`

### 최종 전체 학습

- 학습 데이터: 2019~2024 전체 1,475,092행
- Trackman/투수 임베딩: 사용하지 않음
- 피처: V1 63개
- 각 모델은 자신의 half-life로 2025 기준 최근 가중치를 다시 계산했다.
- OOF 재학습에서 얻은 2024 최적 tree 수로 전체 데이터를 학습했다.
- 최종 활성 모델: XGBoost 7개 + CatBoost 6개
- 가족별 총 가중치: XGBoost 약 51.2%, CatBoost 약 48.8%
- 확률 보정: `logit(p) - 0.0351021769`

| Family | Trial | Weight | Half-life | Trees |
|---|---:|---:|---:|---:|
| XGBoost | 35 | 0.068412 | 0.620658 | 110 |
| XGBoost | 36 | 0.138982 | 1.003584 | 149 |
| XGBoost | 47 | 0.011047 | 0.671848 | 142 |
| XGBoost | 24 | 0.154834 | 0.473116 | 974 |
| XGBoost | 34 | 0.038833 | 0.687833 | 149 |
| XGBoost | 113 | 0.053589 | 0.684875 | 138 |
| XGBoost | 139 | 0.045214 | 0.548145 | 845 |
| CatBoost | 71 | 0.017346 | 1.732446 | 143 |
| CatBoost | 136 | 0.133204 | 0.763690 | 1,014 |
| CatBoost | 94 | 0.061111 | 2.422125 | 722 |
| CatBoost | 80 | 0.034984 | 0.701093 | 1,009 |
| CatBoost | 116 | 0.049333 | 2.149090 | 204 |
| CatBoost | 114 | 0.193110 | 1.749676 | 718 |

### 실행 검증

- 5행 샘플 추론 성공
- 245,789행 CPU 모사 추론: **8.30초**
- 처리량: 약 29,623행/초
- 10분 제한 대비 충분한 여유
- 모사 예측 범위: 0.418309~0.473936
- ZIP 구조·CRC·Linux 권한·절대경로 검사 통과
- 파일명 14자, 30자 제한 통과

## 제출 005 전 성능 실험 기록

아래 결과는 아직 ZIP을 만들지 않은 연구 단계이며 공통 원본은 `experiment/model_optimization/validation_registry.csv`에 있다.

| 실험 | 2023 normalized Brier/BSS | 2024 BSS | Trackman | 판단 |
|---|---:|---:|---|---|
| XGB V2 temporal 전체 | 1.003293 / 0 | 391.0796 | 없음 | 대규모 하락, 폐기 |
| XGB V2 row selected-200 | 1.002523 / 0 | 758.9178 | 없음 | V1 대비 개선, 채택 |
| 실패원인 cross-year blend | 2023에서 가중치 선택 | 760.6500 | 없음 | 소폭 개선, 다양성 후보 |
| XGB V2 row + strict TM500 전체 | 1.002154 / 0 | **768.1219** | 시즌당 500구, fold 이전만 | enhanced HPO 기준 |
| Cat V2 row + strict TM500 전체 | 0.999994 / 0.6020 | 744.5017 | 시즌당 500구, fold 이전만 | 안정형 후보 |

Trackman 누수 감사:

- 2024 검증행은 최대 2023 Trackman/crosswalk 증거만 사용한다.
- 학습의 각 시즌 S 행도 S 이전 Trackman 스냅샷만 사용한다.
- 2024 검증행의 Trackman 피처 가용률은 60.2449%다.
- 최종 2025 lookup에서만 2024 Trackman까지 사용한다.

### 이어갈 판단

1. `submit_004.zip`은 첫 번째 최종 후보로 보존한다.
2. `xgboost_v2r200_tm500_robust` 160 trial과 2024-only recent study를 완료한다.
3. enhanced 상위 XGB와 소수 CatBoost를 시간 순방향 OOF로 다시 학습한다.
4. Trackman 통계 대비 투수 임베딩의 실제 증분을 확인한다.
5. 새 후보 ZIP은 `submit_005.zip`부터 만들되, 실제 제출은 성능형과 저상관 안정형 2개만 한다.
6. 모든 새 제출은 학습 과정, fold별 Brier/normalized Brier/BSS/AUC/평균 편향, Public LB를 이 문서에 기록한다.

## 최종 강화 앙상블 005·006

생성 시각: 2026-08-07T00:06:32.165057+09:00

| 회수 | 목적 | 시간 OOF 설계 | 2024 BSS | 모델 수 | 245,789행 로컬 추론 | SHA256 |
|---:|---|---|---:|---:|---:|---|
| 005 | performance | 2023 OOF 적합→2024 검증, 2024 OOF로 배포 재적합 | 780.3183 | 2 | 13.2초 | `91EC4BEDE53F9245448E56976F4D5B739C0FC128225F90C32B4012B6793776C1` |
| 006 | robust | 2023 OOF 적합→2024 검증, 2024 OOF로 배포 재적합 | 760.5595 | 2 | 12.8초 | `5D9F84B902C1889B3B21BA7635A6C14F4750854C67735BA1FF9D9C44EC78C078` |

- 두 모델 모두 2019~2024 전체 학습 데이터를 재학습했다.
- Trackman은 2024년까지의 과거 로그 중 시즌 500구 이상 투수-시즌만 사용했다.
- `script.py`는 실행 파일 위치를 기준으로 `model/`, `data/`, `output/`을 해석한다.
- ZIP 내부 최상위 구조와 모든 모델 파일 존재 여부, CRC, 245,789행 추론을 검증했다.
