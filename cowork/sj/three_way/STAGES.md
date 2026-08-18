# 3WAY 단계별 실행 계획

> **한 번에 2~3시간씩 돌리고 결과를 보고 다음을 정한다.**
> 각 Stage 는 독립 실행 가능하고, 중간에 끊겨도 캐시(`.npy`)로 이어진다.
> **GPU 작업은 한 번에 하나.**

실행:
```bash
python cowork/sj/three_way/src/run_stage.py --stage 1        # 그 단계만
python cowork/sj/three_way/src/run_stage.py --stage 1 --dry  # 무엇이 돌지 확인
python cowork/sj/three_way/src/run_stage.py --list           # 전 단계 개요
```

---

## 지금까지 (S0~S5, 완료)

| 단계 | 결과 |
|---|---|
| S1 타깃별 단일 전처리 | **전제 확인** — 타깃 간 선호가 음의 상관 (middle vs reverse −0.215) |
| S2 타깃별 조합 빔 | middle **+104.48** / ball +74.59 / reverse +47.13 |
| S3 타깃별 계층 차감 | 거의 전부 실패. 유일 생존 reverse `p_workload` +7.51 |
| S5 결합 | **항등식이 최선.** 단독 840.23 으로 프로덕션 base(836.51) 를 넘었다 |

**현재 타깃별 최적 전처리**

| 타깃 | 조합 |
|---|---|
| middle | `id_frequency + no_trackman + temporal_cyclic` |
| reverse | `count_multiscale + drop_ids + trackman_quality` |
| ball | `drop_ids + no_trackman + rate_multiscale` |
| outside | (미탐색 — Stage 1 에서) |

---

## Stage 1 — 학습 방식 이식 (약 2시간, 96 fit)

**1WAY 에서 통했는데 3WAY 에 아직 없는 것들.** 전부 학습 방식이라 로짓·배깅과 무관하다.

| arm | 내용 | 1WAY 근거 |
|---|---|---|
| `base` | 현행 (타깃별 S2 최적 조합) | — |
| `fw020` | **F행 학습 가중치 0.20** | +4.07 |
| `short05` | **짧은 등판 가중치 0.5** | Public +4 |
| `treeparam` | **기저율별 트리 파라미터** | +0.91 |
| `interact` | **2차 상호작용 (곱·차·비)** | Public +2 |
| `fw020+short05` | 두 가중치 동시 | — |
| `all` | 넷 다 | — |

**타깃 4개**(middle·reverse·ball·outside) **× arm 7 × fold 2** = 56 fit
+ outside 전처리 스크리닝 16 arm × 1 fold = 16 fit
+ S5 재결합 (CPU)

> **왜 outside 를 넣는가**: S5 에서 최선이 `M+R+O` 항등식이었다.
> outside 는 전처리 최적화를 한 번도 안 했는데 결합에 들어간다.

**보는 것**: 타깃마다 어느 학습 방식이 듣는가. 1WAY 와 같은가 다른가.

---

## Stage 2 — 모델 계열 확장 (약 2.5시간, 110 fit)

**1WAY 는 성분당 XGB 8시드 + CatBoost 8시드 = 16모델이었다. 3WAY 는 CatBoost 1개다.**
계열을 늘리는 것이 가장 큰 미이식 항목이다. (시드 배깅은 Stage 4)

| arm | 내용 |
|---|---|
| `cat` | 현행 CatBoost |
| `xgb` | XGBoost 단독 |
| `xgb_screen` | **XGB 용 전처리 재스크리닝** — 계열마다 최적 전처리가 다를 수 있다 |
| `cat+xgb` | 두 계열 평균 |

전처리 재스크리닝이 핵심이다. 1WAY 에서 xgb 와 cat 이 **서로 다른 전역 offset 을 원했고**
(`all_d100` vs `last3_d075`), 전처리도 갈릴 가능성이 크다.

**타깃 4 × 전처리 16 arm × fold 2024** = 64 fit (XGB 스크리닝)
+ 타깃 4 × 최적조합 × fold 2 × 2계열 = 16 fit
+ 결합 재평가

---

## Stage 3 — 조합 재탐색 + fold 2023 확인 (약 2시간, 90 fit)

Stage 1·2 에서 바뀐 조건 위에서 전처리 조합을 다시 훑는다.
S2 는 학습 방식·모델 계열이 고정된 상태에서 찾은 것이라 최적이 이동했을 수 있다.

- 타깃별 빔 서치 3폭 × 2라운드 (Stage 1·2 최적 설정 위에서)
- **fold 2023 확인** — S2·S3 우승은 fold 2024 단독 판정이었다
- 최종 타깃별 구성 확정

---

## Stage 4 — 시드 배깅 (보류 해제 시, 약 3시간)

1WAY 는 성분당 16모델이었다. 3WAY 는 1개다. **가장 큰 남은 격차일 가능성이 높다.**

- 타깃별 최적 구성 × 8시드 × 2계열 × fold 2
- 배깅 전후 단독 BSS 비교

> 현재 **배깅은 보류 중**(2026-08-19 지시). 해제되면 여기서 한다.

---

## Stage 5 — 앙상블 (모델링 종료 후)

**앙상블은 취소가 아니라 순서의 문제다.** 여기서 조합별로 반드시 한다.

- 1WAY 성분 라인 × 3WAY × 프로덕션 base 조합
- 구간별 결합 가중치 (1WAY 에서 Public +15 를 낸 유일한 결합 구조)
- 팀 결합 (찬우·예나 `val2024_pred.csv` 도착 시)
- **이때 base 와의 상관을 다시 본다** — 모델링 단계에서는 안 본다

---

## 로짓 계열 (별도 보류)

Stage 와 무관하게 보류 중. 해제되면 어느 Stage 뒤에든 붙일 수 있다.

| 항목 | 측정된 천장 |
|---|---|
| F2 `base_margin` | +186.75 (1WAY 잔차 분석) |
| 성분별 외삽 규칙 | m 성분 −0.0188 |
| V87 추세×신뢰 보정 | 1WAY 최악 fold +3.54 |

> 3WAY 는 이미 **Pool baseline 으로 시즌 외삽**을 넣었고 그것만으로 +124.75 를 얻었다.
> 로짓 보정 계열의 남은 여지가 1WAY 보다 작을 수 있다.
