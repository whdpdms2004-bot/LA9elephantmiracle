# F1 계열 loss 실험 결과

## 결론

- 순수 soft-F1 loss는 사용하지 않는다. 성공 확률을 거의 전부 1로 보내 Brier Skill Score가 0이 됐다.
- macro soft-F1도 단독 사용하지 않는다. 양·음성 클래스를 대칭 처리해도 확률이 극단화되어 BSS가 0이었다.
- `Brier + lambda * macro soft-F1` 혼합형은 작은 표본 스크리닝에서는 좋아졌지만 전체 학습에서 재현되지 않았다.
- 현재 단독 모델 기준 최선은 기존 Brier loss다. F1 혼합형은 제출 후보가 아니라 소량 블렌드용 보조 expert로만 보류한다.

## 공통 검증 조건

- 모델: 32차원 strict TrackMan500 multi-task pitcher embedding network
- 목표: `control_success` 주 head + reverse/middle/outside 보조 head
- Val2023: 2022년까지 학습하고 2023년 평가
- Val2024: 2023년까지 학습하고 2024년 평가
- 검증 연도 TrackMan 사용 금지, 과거 시즌 500구 이상 투수만 TrackMan tower 활성화
- 동일 architecture, batch size 8,192, learning rate 0.0008, half-life 1.0, seed 사용
- 대회 선택 지표는 BSS이며 F1은 진단 지표로만 사용

## 목적함수

`soft-F1`은 배치 내 soft confusion matrix로 계산했다.

- positive soft-F1: 성공 클래스 F1만 최적화
- macro soft-F1: 성공·실패 클래스 soft-F1의 평균
- 혼합형: `Brier + lambda * macro soft-F1`

F1은 임계값 기반 비분해 지표이므로 XGBoost 행 단위 objective로 직접 넣지 않고 PyTorch에서 미분 가능한 surrogate로 실험했다.

## Val2024 빠른 스크리닝

학습 표본 300,000개, 3 epoch의 동일 조건이다.

| main loss | lambda | BSS | F1@0.5 | 예측 평균 | 판단 |
|---|---:|---:|---:|---:|---|
| Brier | 0 | 113.265 | 0.5761 | 0.5134 | 기준 |
| positive soft-F1 | - | 0 | 0.6542 | 0.9889 | 전부 1에 가까운 붕괴 |
| macro soft-F1 | - | 0 | 0.4546 | 0.3762 | 극단 확률, 탈락 |
| Brier + macro soft-F1 | 0.01 | 115.643 | 0.5749 | 0.5133 | 소폭 상승 |
| Brier + macro soft-F1 | 0.03 | 120.043 | 0.5728 | 0.5131 | 상승 |
| Brier + macro soft-F1 | 0.10 | 132.598 | 0.5662 | 0.5122 | 상승 |
| Brier + macro soft-F1 | 0.30 | **140.624** | 0.5490 | 0.5096 | 스크리닝 최고 |
| Brier + macro soft-F1 | 0.50 | 105.801 | 0.5310 | 0.5070 | 하락 시작 |

스크리닝에서는 lambda 0.30이 최고였지만 F1@0.5는 오히려 감소했다. BSS 개선은 클래스 F1 자체의 상승보다 예측 평균 보정 효과에 가까웠다.

## 전체 학습 결과

| loss | Val2023 BSS | Val2023 F1@0.5 | Val2024 BSS | Val2024 F1@0.5 | Val2024 예측 평균 |
|---|---:|---:|---:|---:|---:|
| Brier | 0 | **0.6084** | **297.701** | **0.4971** | 0.4966 |
| Brier + 0.10 macro soft-F1 | 미실행 | 미실행 | 274.964 | 0.4919 | 0.4948 |
| Brier + 0.30 macro soft-F1 | 0 | 0.5835 | 177.612 | 0.4785 | 0.4909 |

전체 데이터 8 epoch에서는 혼합형이 기존 Brier를 넘지 못했다. lambda가 커질수록 예측 평균은 2024 정답 평균 0.4861에 가까워졌지만 AUC와 분리력이 약해져 최종 Brier가 나빠졌다.

## 블렌드 확인

- 기존 Brier embedding + F1 혼합 embedding 자체의 최적 혼합은 F1 비중 0%, 즉 기존 Brier 단독이었다.
- F1 혼합 embedding을 일부 XGBoost OOF에 1.5~4.0% 섞으면 Val2024 BSS가 약 +0.04~+0.54 개선되는 후보가 있었다.
- 가장 큰 관측 이득은 XGBoost trial 50에서 약 +0.543 BSS였으나 단일 연도·사후 가중치 탐색 결과라 제출 채택 근거로 부족하다.

## 최종 판단 및 다음 조건

1. 주 loss는 계속 Brier를 사용한다.
2. positive/macro soft-F1 단독 모델은 폐기한다.
3. 혼합형은 `lambda <= 0.10`, 1~3% 소량 블렌드 범위만 보조 후보로 보관한다.
4. 실제 채택 전 Val2022/Val2023/Val2024 공통 고정 가중치와 seed 반복 검증이 필요하다.
5. F1을 다시 쓴다면 batch별 soft-F1 대신 EMA confusion statistics, calibration head 분리, epoch early stopping을 함께 실험한다.

## 산출물

- 학습 코드: `experiment/pitcher_embedding/train_strict_multitask_embedding.py`
- 스크리닝: `experiment/pitcher_embedding/outputs/f1_loss_screen/`
- 전체 결과: `experiment/pitcher_embedding/outputs/f1_loss_full/`
