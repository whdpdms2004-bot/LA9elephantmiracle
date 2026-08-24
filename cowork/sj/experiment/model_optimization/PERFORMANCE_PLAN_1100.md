# BSS 1100+ 성능 향상 계획

기준일: 2026-08-06  
현재 최고 로컬 시간 검증: CatBoost 2024 BSS 751.90  
현재 리더보드 1위 참고 점수: 약 1100

## 1. 시간 절단 원칙

예측 대상 시즌을 `s`라고 할 때 모든 학습 산출물은 다음 규칙을 지킨다.

| 산출물 | 허용 데이터 |
|---|---|
| 메인 학습 라벨 | `season < s` |
| Trackman 원시 행 | `season < s` |
| Trackman 투구 수 500구 판정 | 완료된 `season < s`만 사용 |
| Main↔Trackman crosswalk | 두 데이터 모두 `season < s`만 사용 |
| 결측 대체·스케일러·PCA/임베딩 | `season < s`만 사용 |
| 범주 인코딩·target encoding | `season < s`의 메인 학습 데이터만 사용 |

따라서 2024 검증에서는 2019~2023만 사용하고, 2024 Trackman은 crosswalk 생성에도 사용하지 않는다. 최종 2025 추론에서만 2019~2024 Trackman과 메인 학습 데이터를 사용할 수 있다.

현재 먼저 만든 CatBoost/XGBoost 제출 ZIP 2개는 Trackman을 전혀 사용하지 않아 이 원칙에 안전하다.

## 2. Trackman 500구 규칙

Trackman 기반 개별 투수 표현의 최소 단위는 `(pitcher_trackman_id, season)`이다.

1. 시즌별 Trackman 투구 수가 **500개 이상**인 투수-시즌만 임베딩 학습과 통계 집계에 사용한다.
2. 500개 미만 시즌의 원시 투구는 개별 임베딩 입력에서 완전히 제외한다.
3. 한 투수에게 과거 500구 이상 시즌이 하나도 없으면 개별 Trackman 벡터를 주지 않고 `NO_TM500` fallback과 availability/count 피처만 사용한다.
4. 같은 시즌의 최종 투구 수로 현재 시즌 행을 분류하지 않는다. 500구 판정은 반드시 이미 끝난 과거 시즌에 대해서만 한다.
5. 2024 검증용 아티팩트는 2019~2023 중 500구 이상 시즌만, 2025 제출용 아티팩트는 2019~2024 중 500구 이상 시즌만 사용한다.

데이터 현황:

| 시즌 | 500구 이상 Trackman 투수 | 해당 투구 수 |
|---:|---:|---:|
| 2019 | 180 | 210,407 |
| 2020 | 200 | 226,477 |
| 2021 | 239 | 258,408 |
| 2022 | 244 | 262,337 |
| 2023 | 252 | 270,357 |
| 2024 | 276 | 295,212 |

전체 Trackman 행의 약 84.95%가 500구 이상 투수-시즌에 포함된다. 2024 검증 시점에는 과거 기준 491명, 최종 2025 시점에는 559명이 최소 한 번의 500구 이상 시즌을 가진다.

## 3. 발견된 기존 파이프라인 문제

- 양호: `build_lagged_trackman`은 원시 Trackman을 `season < cutoff`로 자른다.
- 양호: 시즌 순방향 OOF 임베딩의 supervised 학습 라벨은 `season < target_season`만 사용한다.
- 수정 필요: 기존 Main↔Trackman crosswalk는 2019~2024 전체 시즌의 매칭 투표로 한 번 생성된다. 2024 검증용 crosswalk에 2024 경기 상태가 간접 사용될 수 있다.
- 수정 필요: 기존 임베딩 eligibility와 shrinkage가 100구 기준이다.
- 수정 필요: 기존 Trackman 집계는 과거의 500구 미만 시즌도 함께 평균에 포함한다.

기존 V3 임베딩 결과는 엄격한 새 기준으로 폐기하고 다시 생성한다.

## 4. 검증 설계

세 개의 완전 독립 cutoff 아티팩트를 만든다.

| fold | 모델 학습/Trackman/crosswalk | 검증 |
|---|---|---|
| F22 | 2019~2021 | 2022 |
| F23 | 2019~2022 | 2023 |
| F24 | 2019~2023 | 2024 |
| FINAL | 2019~2024 | 2025 테스트 |

모델 선택 목적값은 normalized Brier를 사용한다.

`objective = 0.20 × F22 + 0.30 × F23 + 0.50 × F24 + 0.20 × max(F22,F23,F24)`

마지막 항은 최악 시즌 붕괴를 막기 위한 penalty이며, 실제 가중치는 합이 1이 되도록 정규화한다. BSS가 0으로 잘리는 경우에도 내부 선택은 unclipped normalized Brier로 수행한다.

## 5. 성능 향상 실험 순서

### A. 리더보드 기준점 확보

먼저 생성한 두 제출 결과를 기록한다.

1. CatBoost trial 71, 안정성 최고
2. XGBoost trial 24, 2024 최고

두 점수로 로컬 2024 BSS와 리더보드 점수의 방향성, 모델 계열별 일반화 차이를 확인한다.

### B. Trackman 없는 V2 피처

Trackman과 무관하게 먼저 1100점 격차를 줄인다.

- 누적 표본 수 기반 Beta/Binomial smoothing
- 누적 성공률과 최근 1·3·5경기 성공률의 신뢰도 가중 결합
- cold start, `n≤25`, `n≤100`, `n≤500` 플래그
- pitcher/batter/team/count/matchup의 과거 시즌 전용 target encoding
- 최근 완료 시즌과 전체 과거의 이중 target encoding
- `game_type × season`, count, 주자, 이닝, 손잡이 상호작용
- 중복 파생 변수 ablation

CatBoost와 XGBoost를 각각 최소 200~300 trial 탐색한다. 단일 2024 최고뿐 아니라 3-fold 안정성 상위 후보를 보존한다.

### C. Trackman 500 통계 피처

임베딩 전에 해석 가능한 집계부터 검증한다.

- 구속·회전·수직/수평 무브먼트·익스텐션·릴리스 위치·zone speed의 시즌별 평균/표준편차/분위수
- 구종군 비율, 구종 다양성 entropy
- 직전 eligible 시즌, 최근 2개 eligible 시즌 가중 평균, 전체 eligible 시즌 평균
- 시즌 간 변화량과 안정성
- `tm500_available`, eligible 시즌 수, 마지막 eligible 시즌과의 간격

V2 대비 F22/F23/F24 모두에서 개선되거나 가중 objective가 명확히 개선될 때만 채택한다.

### D. Trackman 500 투수 임베딩

두 표현을 분리 비교한다.

1. 비지도 임베딩: eligible 투수-시즌의 물리 피처와 구종 구성 재구성/대조학습
2. 과거 라벨 supervised 임베딩: cutoff 이전 메인 라벨만 이용해 control failure 구조를 학습

후보 모델:

- MLP autoencoder 16/32/64차원
- variational autoencoder
- 투수-시즌 Set Transformer 또는 attention pooling
- physical tower + main-history tower의 two-tower fusion

임베딩은 16·32·64차원을 비교하며, 원본 통계 피처와 임베딩을 각각 단독/동시 투입해 증분을 확인한다. Trackman이 없는 투수는 0 벡터와 availability 플래그를 사용하고 별도 cohort embedding으로 처리한다.

### E. 최종 모델과 앙상블

- CatBoost: 범주형 ID와 target statistics
- XGBoost: 명시적 temporal target encoding
- LightGBM: 범주 처리와 leaf-wise 다양성 후보
- PyTorch tabular model: 수치 tower + ID embedding + Trackman embedding

각 모델의 F22/F23/F24 OOF 확률을 저장한다. 가중치는 비음수·합 1 제약으로 이전 시즌에서 학습하고 다음 시즌에서 검증한다. probability/logit blend와 none/logit-shift/Platt/Beta calibration을 비교한다.

## 6. 채택 기준과 목표

| 단계 | 최소 채택 조건 | 목표 2024 BSS |
|---|---|---:|
| V1 | 현재 기준 | 750+ |
| V2 안전 피처 | 두 시즌 이상 개선, 최신 시즌 악화 없음 | 850~950 |
| Trackman 500 통계 | 가중 objective 개선 | 900~1000 |
| Trackman 500 임베딩 | 통계 피처 대비 추가 개선 | 950~1050 |
| OOF 앙상블·보정 | 두 전이에서 재현 | 1050~1150+ |

Trackman 추가가 다시 성능을 낮추면 최종 제출에서 과감히 제외한다. 목표는 Trackman 사용 자체가 아니라 숨은 2025 Brier 개선이다.

## 7. 약 100시간 실행 예산

| 구간 | 작업 | 예산 |
|---|---|---:|
| 1 | cutoff별 누수 감사·crosswalk·500구 캐시 | 6시간 |
| 2 | Trackman 없는 V2 피처/ablation | 14시간 |
| 3 | CatBoost/XGBoost V2 Optuna | 24시간 |
| 4 | Trackman 500 통계/임베딩 탐색 | 28시간 |
| 5 | 전체 OOF 재학습·앙상블·보정 | 16시간 |
| 6 | 전체 재학습·추론 벤치마크·ZIP 검증 | 8시간 |
| 7 | 실패 재시도와 최종 후보 비교 | 4시간 |

장시간 탐색은 단계별 DB와 중간 산출물을 저장해 컴퓨터가 꺼져도 완료 trial부터 이어서 실행한다.

## 8. 다음 작업

1. cutoff별 crosswalk를 만드는 재사용 Python 모듈 작성
2. Trackman 500 eligible cache와 누수 assertion 작성
3. F22/F23/F24 Trackman 통계 피처 생성
4. V1 + Trackman 500 통계의 고정 CatBoost/XGBoost ablation
5. 개선이 확인되면 임베딩 모델 학습 시작
6. 두 초기 제출의 리더보드 결과를 실험 테이블에 반영
