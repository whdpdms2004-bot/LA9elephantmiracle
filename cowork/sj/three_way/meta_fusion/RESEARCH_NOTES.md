# 3WAY 최종 확률 결합·공동 재학습 조사 노트

작성일: 2026-08-19  
범위: `cowork/sj/three_way`의 기존 결과와 제출본 `submit_037`을 기준으로 한 다음 단계 설계

## 1. 먼저 확인한 현재 상태

### 현재 제출본의 실제 구조

`submit_037`은 다음 네 CatBoost 모델을 2019~2024 전체로 다시 학습해 사용한다.

| 출력 | 의미 | 피처 수 | 2025 학습 기반 prior |
|---|---|---:|---:|
| `p_m` | middle 실패 확률 | 186 | 0.18654480 |
| `p_r` | reverse 실패 확률 | 297 | 0.26131076 |
| `p_o` | outside 실패 확률 | 230 | 0.11614189 |
| `p_mr` | middle과 reverse가 동시에 발생할 확률 | 186 | 0.03437079 |

최종값은 학습 결합기가 아니라 다음 구조식이다.

```text
p_success = clip(1 - (p_m + p_r - p_mr + p_o))
```

Public 결과는 958.2563447143, 서버 추론 시간은 8초다. Val2024는 Brier
0.247736119, raw BSS 828.590, centered BSS 840.23이었다.

### 중요한 표현 정리

`outputs/`에는 way 내부 평균·로짓 회귀·시드 배깅 실험이 많이 남아 있지만,
`submit_037`에 실제로 들어간 최종 네 모델은 각각 단일 CatBoost다. 따라서 다음
연구에서 말하는 "이미 앙상블된 3WAY"는 아래처럼 명확한 인터페이스로 다시 정의해야
한다.

```text
way별 teacher ensemble
  -> 평균 확률
  -> 멤버 간 표준편차 또는 로짓 분산
  -> 사용 멤버와 고정 가중치
```

최종 결합 모델은 이 인터페이스만 받고, 개별 멤버를 임의로 다시 고르지 않는다.

## 2. 확률 구조: 세 확률만으로는 항등식이 완성되지 않는다

복원 라벨에서 다음이 성립한다.

- 실패 = `middle OR reverse OR outside`
- `outside`는 정의상 `!middle AND !reverse`와 함께 만들어져 M/R과 겹치지 않는다.
- 3중 겹침은 없다.
- 남는 교집합은 `middle AND reverse`다.

따라서 정확한 포함–배제식은 다음이다.

```text
P(success | x)
= 1 - P(M OR R OR O | x)
= 1 - [P(M|x) + P(R|x) + P(O|x) - P(M AND R|x)]
```

여기서 핵심은 `P(M)`, `P(R)`, `P(O)`라는 세 주변확률만으로는 일반적으로
`P(M AND R)`를 알 수 없다는 점이다. 세 확률만을 받는 모델이 만들 수 있는 최적값은
대수적인 항등식이 아니라 다음 조건부 기대값이다.

```text
E[control_success | p_m, p_r, p_o]
```

즉 선택지는 두 개다.

1. `p_mr`를 구조 보조입력으로 계속 유지한다. 현재 방식이며 가장 안전하다.
2. 정말 세 확률만 쓰려면 `p_mr = g(p_m, p_r, p_o)`를 OOF 데이터에서 학습하거나,
   최종 성공확률을 직접 학습한다.

세 확률의 합에 임의 상수를 곱하는 방식은 교집합이 행마다 달라지는 문제를 해결하지
못한다.

## 3. 기존 결합 실험에서 배운 것

`combine_2024.csv`와 `three_way_combine.csv`를 다시 읽었다.

| 방식 | Val2024 raw BSS | 해석 |
|---|---:|---|
| 현재 M/R/O/MR 항등식 | **828.59** | 현재 기준선 |
| M/R/B 로짓 회귀 | 541.46 | 2023에서 학습한 사상이 2024로 잘 전이되지 않음 |
| M/R/O 로짓 회귀 | 462.07 | 자유 결합이 구조식을 훼손 |
| M/R/O 얕은 GBDT | -36.81 | 비선형 결합기의 심한 시간 과적합 |
| 과거 별도 overlap 모델 기반 재결합 | 455.78 | way 모델 세대·설정 차이까지 섞인 결과 |

결론은 "학습 결합기는 안 된다"가 아니라 다음에 가깝다.

- 한 시즌의 OOF만으로 자유도가 큰 결합기를 맞추면 시간 이동에 취약하다.
- 현재 항등식은 강한 구조적 prior가 아니라 **사건 합집합을 계산하는 본체**다.
- 다음 모델은 성공확률을 임의로 회귀하기보다 빠진 교집합과 공동 상태 확률을
  학습해야 한다.
- 모델 선택과 결합기 학습에는 중첩된 시간순 OOF가 필요하다.

입력 벡터가 3차원이라는 사실은 부차적이다. 본질은 세 점수의 수치 조합이 아니라
`M`, `R`, `O` 중 하나 이상이 발생할 조건부 확률, 즉 `P(M ∪ R ∪ O | x)`를
계산하는 것이다.

## 4. 외부 방법론 조사

### 4.1 OOF 스태킹과 시간순 검증

Super Learner는 교차검증 예측을 이용해 후보 학습기들의 가중 조합을 학습하는
손실 기반 스태킹 방법이다. 이 프로젝트에서는 일반 K-fold가 아니라 시간순 OOF로
바꿔야 한다.

- [Super Learner, van der Laan et al. (2007)](https://doi.org/10.2202/1544-6115.1309)
- [Rolling forecasting origin 설명, Hyndman](https://robjhyndman.com/hyndsight/tscv/)

적용 결론: 메타모델이 보는 모든 학습 확률은 반드시 그 행보다 이전 시즌만 학습한
teacher에서 나와야 한다. 인샘플 teacher 확률은 금지한다.

### 4.2 확률 보정

Beta calibration은 `log(p)`와 `log(1-p)`를 이용한 모수적 확률 보정이며 identity map을
포함한다. 단순 Platt scaling보다 안전한 후보지만, 이 프로젝트는 시간 이동이 강하므로
2023/2024 양쪽 raw Brier가 개선될 때만 채택해야 한다.

- [Beta calibration, Kull et al. (AISTATS 2017)](https://proceedings.mlr.press/v54/kull17a.html)

적용 결론: 보정은 최종 단계의 작은 후보일 뿐 주 모델이 아니다. 평가/test 평균에
맞추는 보정은 규정 위반이므로 절대 사용하지 않는다.

### 4.3 단조 결합기

실패 성분 확률이 올라갈 때 다른 조건이 같은 성공확률이 올라가는 것은 의미상
부자연스럽다. 교집합 또는 공동 상태 확률을 추정하는 보조모델에 단조 lattice를
사용하면 유연성과 제약을 함께 줄 수 있다.

- [Monotonic Calibrated Interpolated Look-Up Tables, JMLR 2016](https://jmlr.org/papers/v17/15-243.html)

적용 결론:

- `p_m`, `p_r`, `p_o`에 대해 최종 성공확률은 단조 감소
- `p_mr`에 대해서는 단조 증가
- 3~4차원 격자 또는 단조 piecewise-linear 모델이면 충분하며 큰 신경망은 불필요

### 4.4 주변확률이 아니라 공동 상태를 모델링

여러 이진 라벨은 독립적으로 보지 않고 공동분포로 모델링할 수 있다. Conditional
Bernoulli Mixtures는 라벨 의존성을 포착하는 다중라벨 접근이다.

- [Conditional Bernoulli Mixtures for Multi-label Classification, ICML 2016](https://proceedings.mlr.press/v48/lij16.html)

이 데이터는 가능한 상태가 매우 작아 더 단순하게 만들 수 있다.

```text
S       : 성공, 어느 실패도 없음
M       : middle only
R       : reverse only
MR      : middle and reverse
O       : outside
```

5-class softmax 하나를 쓰면 모든 확률이 0~1이고 합이 1이며, 최종 확률은 바로
`P(S)`가 된다. 이는 장기 모델의 가장 깨끗한 구조다.

### 4.5 앙상블 증류

Knowledge distillation은 여러 teacher의 확률 출력을 한 student에 전달하는 방법이다.
way별로 이미 다양화된 ensemble을 단일 공동모델에 이식할 때 적합하다.

- [Distilling the Knowledge in a Neural Network, Hinton et al.](https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/)

적용 결론: student는 실제 라벨만 학습하는 대신 OOF teacher의 부드러운 확률도 같이
맞춘다. 단, teacher soft target 역시 반드시 OOF여야 한다.

### 4.6 TabM

TabM은 MLP 안에서 parameter-efficient ensemble을 구현한다. 논문은 복잡한 attention
또는 retrieval 모델보다 단순 MLP 기반 효율적 앙상블의 성능–효율 균형이 좋았다고
보고한다. 공식 구현은 Apache-2.0이다.

- [TabM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)
- [공식 TabM 저장소](https://github.com/yandex-research/tabm)

적용 결론: 이 데이터처럼 행 수가 많고 tabular인 경우 장기 신경망의 첫 후보로
FT-Transformer보다 TabM/MLP를 먼저 둔다. 외부 사전학습 가중치는 쓰지 않고 공식
데이터로 처음부터 학습한다.

### 4.7 MMoE와 다중과제 손실 균형

현재 실험에서 middle과 reverse/outside의 전처리 선호가 반대였다. 모든 타깃에 하나의
shared trunk를 강제하면 negative transfer가 날 가능성이 높다. MMoE는 공유 expert와
타깃별 gate를 함께 사용해 관계가 다른 과제를 부분 공유한다.

- [Multi-gate Mixture-of-Experts, KDD 2018](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-)
- [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html)
- [PCGrad, NeurIPS 2020](https://arxiv.org/abs/2001.06782)

적용 결론:

- middle 전용 expert, reverse/outside 전용 expert, 공유 expert를 둔다.
- task별 gate가 공유량을 결정하게 한다.
- 손실 크기가 다른 M/R/O/MR/S를 단순 동일 가중하지 않는다.
- 우선 GradNorm으로 손실 균형을 맞추고, 실제 gradient cosine이 자주 음수일 때만
  PCGrad를 추가한다.

## 5. 후보별 우선순위

| 우선순위 | 후보 | 기대 | 위험/비용 | 결론 |
|---:|---|---|---|---|
| 1 | 세 확률 -> bounded overlap -> 합집합 | 세 주변확률에서 빠진 교집합 추정 | overlap 오차가 그대로 전파 | **즉시** |
| 2 | 네 확률의 공동상태 reconciliation | M/R/O/MR을 일관된 5-state 분포로 투영 | OOF 구축 필요 | **즉시 비교** |
| 3 | 5-state CatBoost | 공동분포·논리 일관성 | 새 multiclass 모델 필요 | **높은 우선순위** |
| 4 | 단조 lattice/GAM overlap | 교집합 추정에 비선형성과 의미 제약 | 구현 추가 | 1~3 이후 |
| 5 | OOF residual CatBoost | 합집합 확률의 오차 분석용 | 사건 의미를 흐릴 위험 | 진단 비교군만 |
| 6 | TabM/MMoE 5-state student | ensemble 증류·공동학습 | GPU와 구현 비용 | 장기 핵심 |
| 낮음 | 자유 GBDT/큰 MLP 결합기 | 상호작용 | 기존 fold 전이 실패 재현 위험 | 보류 |
| 낮음 | FT-Transformer/TabPFN류 | 최신성 | 이 문제에 과하고 재현·패키징 부담 | 우선 제외 |

## 6. 조사 결론

가장 합리적인 다음 수는 두 갈래다.

1. **단기:** way별 ensemble OOF를 만들고 `p_mr` 또는 전체 5-state 공동분포를
   추정한 뒤, 포함–배제로 `P(M ∪ R ∪ O)`를 계산한다. 세 입력에서 교집합을
   추정할 때와 별도 `p_mr` teacher를 유지할 때를 정면 비교한다.
2. **장기:** 다섯 상호배타 상태를 직접 예측하는 공동모델을 만든다. 먼저 CatBoost
   multiclass로 구조의 가치를 확인하고, 그 다음 TabM/MMoE student에 way ensemble을
   증류해 end-to-end로 미세조정한다.

이 순서라면 복잡한 딥러닝을 시작하기 전에 "공동분포 모델링 자체가 이득인가"를
낮은 비용으로 검증할 수 있다.
