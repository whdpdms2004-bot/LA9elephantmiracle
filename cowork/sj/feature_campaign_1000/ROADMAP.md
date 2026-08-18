# BSS 1000 피처 캠페인 로드맵

작성일: 2026-08-18 KST

## 목표와 종료 조건

- 외부 데이터 없이 `train.csv`, 한 평가 행의 `test.csv`, 예측 시즌 이전 TrackMan만 사용한다.
- 전처리·피처 엔지니어링·단독 ML 예측으로 Public BSS 1000 이상을 확인한다.
- 리더보드 결과는 도달 여부 확인에만 쓰고 피처, 상수, 보정값, blend weight 선택에는 쓰지 않는다.
- Public 1000을 넘더라도 Val2023/Val2024 순방향 검증, 행 독립성, TrackMan 시점 게이트와 재현 파이프라인이 통과해야 완료로 판정한다.

## 고정 검증 계약

| 단계 | 학습 | 검증 | 목적 |
|---|---|---|---|
| 빠른 screen | 2019~2023 | 2024 | 많은 family의 방향 확인 |
| 시간 안정성 | 2019~2022 | 2023 | 다른 regime에서도 부호 재현 |
| full confirm | 각 fold 이전 시즌 | 2023·2024 | 8 seeds, production rounds로 채택 판정 |
| final fit | 2019~2024 | 2025 test | 확정 설정 한 번만 적용 |

채택 조건은 두 fold 모두에서 기존 arm보다 성분 단독 BSS와 base 결합 Delta BSS가 함께 개선되고,
증분이 반복 실행 잡음 바닥인 3 BSS를 넘는 것이다. 한 fold만 좋아지는 후보는 보류하거나 regime 원인을 먼저 설명한다.

## 피처 트랙

### 1. Direct 입력 정리

1. 원본 47열의 결측, 범위, 신규 범주, 시즌별 안정성을 감사한다.
2. 숫자 ID를 순서형 수치로 해석하는 위험을 제거 ablation으로 확인한다.
3. 미래 시즌에 외삽할 수 없는 `season` 원시 축을 제거하거나 학습 데이터로 고정한 추세 표현으로 대체한다.
4. 직접 신호는 단순 중요도가 아니라 Val2023/Val2024 양쪽의 단변량 Brier 부호로 우선순위를 정한다.

현재 우선 신호: 투수 success/reverse/ball rate, prev3/5 success, balls, win expectancy/LI, pitch mix.
현재 제거 후보: raw season, game_type, raw batter/team/player ID, 단순 누적 표본량.

### 2. 행 단위 파생

1. count·주자·이닝·접전·LI 상태를 명시적으로 교차한다.
2. career와 prev1/3/5의 수준, 기울기, 가속도, shock를 분리한다.
3. 투수·타자 rate에는 표본량 기반 shrinkage와 reliability를 함께 둔다.
4. pitch mix와 실패 방향의 entropy/concentration을 만들되 합이 0인 행을 안전 처리한다.
5. 강한 직접 신호끼리만 제한적으로 2차항을 추가해 feature dilution을 막는다.

현재 채택 피처는 `F1`이다. enhanced 209열에 D0 18열과 strict forward-OOF
계층 차감 EB 45열을 더한다. C1은 validation lookup만 안전하고 학습행 lookup이
자기 시즌 Target을 포함해 최종 후보에서 제외했다.

### 3. TrackMan 투수 표현

1. 예측 시즌 `S`마다 `trackman.season < S`만 남긴다.
2. 기존 평균·표준편차와 중복되지 않는 release 공분산, 구종군별 일관성, mechanics coupling을 만든다.
3. target-free PCA와 기존 tm500에 대한 residual PCA를 비교한다.
4. main 투수와의 crosswalk 가용성·신뢰도를 별도 피처로 둔다.
5. target-aware 유사 투수 prior는 모든 학습 행에 순방향 OOF를 만들 수 있을 때만 시험한다.

Residual PCA12는 기존 tm500 72열로부터의 평균 5-fold 재구성 R2가 -0.115이고,
Val2024 매핑 행 커버리지는 60.24%다. 두 fold 예측력 부호가 불안정해 보류한다.

## 반복 순서

1. 시즌/불안정 raw 열 제거 arm을 screen한다.
2. 이긴 전처리를 새 기준선으로 고정한다.
3. `P1 + context`, audit 상위신호×count, TrackMan residual PCA를 한 family씩 추가한다.
4. 가장 좋은 두 조합만 Val2023/2024 full confirm한다.
5. component별 이득과 R/F·월별 Brier를 분해해 특정 regime 과적합을 제거한다.
6. 최종 2019~2024 재학습, inference-only 패키지 생성, 행 독립성·오프라인 smoke test를 수행한다.
7. 제출 후 1000 미만이면 리더보드 역산 없이 다음 미검증 family로 1번부터 반복한다.

## 실험 대기열

| 우선순위 | 실험 | 비교 기준 | 상태 |
|---:|---|---|---|
| 1 | raw season 제거 | 현행 111 | **기각**: 2023 +77.07, 2024 -30.99 Delta BSS |
| 2 | season 유지 + game_type/raw ID 제거 | 현행 | 준비됨 |
| 3 | season 유지 + P1 + context | 현행 | **기각**: Val2023 부호 반전 |
| 4 | 위 조합 + TrackMan residual PCA12 | 3번 | **보류**: 단일 XGB 증분 +45.27/-8.07 |
| 5 | 상위 direct signal × count | 3번 | 성분 screen 단독 악화로 기각 |
| 6 | 단일 XGBoost D0 파생항 | 기존 209피처 | **F1에 포함**: 2 folds × 2 설정 모두 +7~16 |
| 7 | 계층 차감 C1 full confirm | 기존 209피처 | **탐색 통과·제출 제외**: 학습행 자기 시즌 lookup 발견 |
| 8 | strict forward-OOF F1 | 기존 209피처 | **채택**: Cat 보정 BSS 2301.26/64.02/796.73 |
| 9 | F1 전용 XGB/Cat 하이퍼파라미터 탐색 | F1 기준 | **완료**: 기존 robust Cat이 최종 승자 |
| 10 | 최종 재학습·제출 패키지 | F1 Cat | **완료**: raw-data wrapper + `submit_036`, B1 검증 통과, Public 대기 |

## 현재 다음 단계

1. `submit_036.zip`을 제출하고 Public이 1000 이상인지 확인한다.
2. Public 결과는 도달 판정에만 사용하고 offset·피처·가중치를 역산하지 않는다.
3. 1000 미만이면 아직 채택하지 않은 새 family를 순방향 OOF로만 추가해 반복한다.

## 중단·실패 처리

- 실험 예측은 stage/arm/fold별 cache로 저장하고 동일 코드·설정일 때만 명시적으로 재사용한다.
- GPU에 다른 사용자 작업이 있으면 해당 프로세스를 중단하지 않는다.
- 한 후보가 실패해도 결과를 삭제하지 않고 다음 family의 중복 방지 근거로 남긴다.
- 제출본을 만들 때는 `cowork/RULES.md`와 `AGENTS.md` B1의 6단계를 처음부터 다시 수행한다.
