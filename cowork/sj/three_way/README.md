# 3WAY — 하위 확률별 독립 모델링

> LG Aimers 9기 Phase 2 · KBO 제구 성공 확률
> **1WAY** = 지금까지의 방식, **3WAY** = 이 폴더

---

## 1WAY 와 무엇이 다른가

| | 1WAY (기존) | **3WAY (여기)** |
|---|---|---|
| 하위 타깃 | 5개 (m, r, mr, ob, oz) | **3개 (middle, reverse, ball)** |
| 피처 | **다섯 성분이 111피처를 공유** | **타깃마다 따로 찾는다** |
| 전처리 | 공유 | **타깃마다 따로** |
| 모델 차이 | 트리 파라미터만 기저율에 따라 | 타깃마다 전부 |
| 결합 | 포함–배제 **항등식** | **학습된 결합기** |
| 최종 조정 | 없음 (블렌드 가중치만) | **최종 라벨로 미세조정** |

1WAY 는 "같은 재료를 다섯 조각으로 나눠 본다" 였고,
3WAY 는 **"조각마다 재료를 다시 고른다"** 다.

---

## 왜 해볼 만한가

1WAY 에서 성분 분해가 통한 이유는 정확도가 아니라 **base 와의 상관을 낮춘 것**이었다
(V65: 직접 모델이 단독은 더 좋은데 결합에서 밀림). 그렇다면 하위 확률마다
**서로 다른 피처를 쓰면 상관이 더 내려갈 여지**가 있다. 지금은 다섯 성분이 같은
피처를 봐서 서로 강하게 상관돼 있다.

---

## 설계

```
   [middle 모델]  전처리 A + 피처 A
   [reverse 모델] 전처리 B + 피처 B      →  결합기  →  최종 P(성공)
   [ball 모델]    전처리 C + 피처 C          ↑
                                        최종 라벨로 미세조정
```

### 타깃 정의와 기저율

| 타깃 | 라벨 | 기저율 | null | **BSS 배율** |
|---|---|---:|---:|---:|
| middle | `y_middle` | 0.1496 | 0.1272 | **7.86** |
| reverse | `y_reverse` | 0.2290 | 0.1766 | 5.66 |
| ball | `y_ball` | 0.3695 | 0.2330 | 4.29 |
| outside | `y_outside` | 0.1317 | 0.1144 | 8.74 |
| success | `control_success` | 0.5237 | 0.2494 | 4.01 |

### ★ `ball` 은 항등식을 닫지 못한다

```
실패 = middle OR reverse OR outside      <- 오차 0.000000 으로 성립
```

`ball` 은 실패 유형이 아니라 **교차 속성**이다 — `outside=1` 인 행의 82.4%,
`outside=0` 인 행의 30.1% 가 ball 이다.

지시대로 **M/R/B 로 짓되 `outside` 도 등록**해 두었다.
결합 단계에서 **M/R/B 와 M/R/O 를 실측 비교**한다. 결합기가 학습이라 항등식이
필수는 아니지만, 닫히는 쪽이 나은지는 재봐야 안다.

---

## ★ BSS 기준을 타깃마다 다시 잡는다

`BSS = 100000 × (1 − Brier/null)` 이고 `null = p(1−p)` 다.
배율 `1/null` 이 4.01(success) ~ 8.74(outside) 로 **두 배 넘게 벌어진다.**
같은 Brier 개선이라도 타깃마다 BSS 가 다르게 나온다.

| 지표 | 용도 |
|---|---|
| `bss_raw` | 그 타깃 안에서의 절대 성능 |
| **`bss_centered`** | **순위 판정용.** 평균 정렬로 번 점수를 뺀 순수 신호 |
| **`bss_norm`** | **타깃 간 비교용.** Brier 개선률 × 1000 |

**시드 잡음도 타깃마다 다르다.** 1WAY fold2024 실측 sd 1.37 을 null 배율로 환산한다.

| 타깃 | 잡음 sd 추정 |
|---|---:|
| middle | 2.69 |
| reverse | 1.93 |
| ball | 1.47 |
| success | 1.37 |

**이보다 작은 차이는 같은 값으로 본다.** `harness3.seed_noise()` 가 계산한다.

---

## fold 규칙 (1WAY 에서 승계)

| fold | 쓰는 법 |
|---|---|
| **2024** | **결정 fold** |
| 2023 | 보조. `bss_centered` 로만 본다 (오프셋 교란) |
| **2022** | **쓰지 않는다** — 2022→2024 순위상관이 두 계열 모두 음수 |

`harness3.fold_masks()` 가 2022 를 요청하면 예외를 던진다.

---

## 폴더

```
three_way/
├─ README.md            이 파일
├─ PLAN.md              실험 계획 — 어떤 방식을 어떤 순서로
├─ RESULTS.md           결과
├─ src/
│   ├─ harness3.py        타깃별 라벨 · BSS · fold 규칙
│   ├─ screen_target.py   타깃별 전처리 스크리닝 / 빔 서치
│   └─ combine.py         세 모델 결합 + 최종 라벨 미세조정
└─ outputs/
```

전처리 변환은 **[`../preprocess_lab/transforms/`](../preprocess_lab/README.md) 를 그대로 재사용**한다.
복사하지 않으므로 랩에 변환을 추가하면 3WAY 에서도 바로 쓸 수 있다.

---

## 실행

```bash
# 타깃별 단일 전처리 스크리닝 (GPU, 타깃당 약 12분)
python cowork/sj/three_way/src/screen_target.py --target middle,reverse,ball

# 조합 빔 서치 (GPU, 타깃당 약 40분)
python cowork/sj/three_way/src/screen_target.py --target middle --beam 3 --rounds 3

# 학습 없이 확인
python cowork/sj/three_way/src/screen_target.py --target middle --dry
```

> **GPU 작업은 한 번에 하나.** 겹쳐 돌리면 둘 다 죽는다.
