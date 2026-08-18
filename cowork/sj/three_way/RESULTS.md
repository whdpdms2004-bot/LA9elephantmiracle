# 3WAY 결과

> 갱신: 2026-08-18. fold 2024, CatBoost GPU (`iterations=900, lr=0.015, depth=8`).
> 값은 **`bss_centered` 기준 기준선 대비**. 타깃 간 비교는 `bss_norm`.
> **잡음 sd 는 타깃마다 다르다** — middle 2.69 / reverse 1.93 / ball 1.47.

---

## S1 결론 — **3WAY 의 전제가 확인됐다** ★

### 타깃 간 전처리 선호가 음의 상관이다

| 쌍 | Spearman |
|---|---:|
| **middle vs reverse** | **−0.215** |
| **middle vs ball** | **−0.253** |
| reverse vs ball | +0.509 |
| middle vs success(1WAY) | +0.344 |
| reverse vs success(1WAY) | +0.291 |
| ball vs success(1WAY) | +0.194 |

**middle 에 좋은 전처리가 reverse·ball 에는 나쁘다.**
1WAY 는 다섯 성분에 **같은 111피처를 강제**했다. 타깃마다 필요한 것이 반대인데
하나로 묶고 있었다는 뜻이고, 그것이 3WAY 가 이길 여지다.

### 타깃별 최선 3개

| 타깃 | 1위 | 2위 | 3위 |
|---|---|---|---|
| **middle** | `temporal_cyclic` **+32.9** | `no_trackman` +28.6 | `trackman_compact` +17.5 |
| **reverse** | `drop_ids` **+42.4** | `id_frequency` +28.4 | `rate_multiscale` +9.3 |
| **ball** | `drop_ids` **+66.2** | `rate_multiscale` +22.0 | `rate_geometry` +14.2 |

### 가장 극적인 것 — `drop_ids`

| 타깃 | Δ |
|---|---:|
| middle | **−414.1** |
| reverse | **+42.4** |
| ball | **+66.2** |
| success (1WAY) | −57.1 |

**같은 변환 하나가 middle 과 ball 사이에서 480점 스윙한다.**

- **middle(한가운데 실투)은 투수 정체성이 거의 전부다.** ID 를 빼면 무너진다.
  그런데 **TrackMan 은 빼는 게 낫다** — 상위 3개 중 2개가 TrackMan 제거/축소다.
- **reverse·ball 은 정반대.** ID 를 빼는 것이 최선이다.
  투수 개인보다 상황·타자가 결정하는 구조로 보인다.

### 전체 표 (Δ bss_centered)

| 변환 | middle | reverse | ball | success(1WAY) |
|---|---:|---:|---:|---:|
| `temporal_cyclic` | **+32.9** | −3.2 | −1.1 | +4.8 |
| `no_trackman` | **+28.6** | −4.2 | −6.2 | −5.7 |
| `trackman_compact` | +17.5 | +1.4 | −29.5 | −7.0 |
| `trackman_quality` | +11.3 | +5.5 | −2.8 | −3.1 |
| `rate_multiscale` | +7.1 | +9.3 | +22.0 | +0.7 |
| `component_shape` | +6.5 | +6.7 | −0.6 | −1.1 |
| `context_robust` | +6.2 | +7.4 | −21.5 | +0.8 |
| `rate_geometry` | +2.5 | +5.4 | +14.2 | +1.7 |
| `pitcher_reactivity` | +0.9 | +0.5 | −41.2 | −1.4 |
| `count_multiscale` | −0.2 | +2.0 | 0.0 | +1.2 |
| `component_compact` | −3.1 | −6.9 | +9.3 | −8.0 |
| `recent_shape` | −5.8 | +4.4 | −2.2 | −6.3 |
| `ordinal_numeric` | −9.5 | +3.3 | −5.2 | −7.6 |
| `no_component` | −32.8 | −113.5 | −23.1 | −26.0 |
| `id_frequency` | −144.5 | **+28.4** | +1.5 | +7.9 |
| `drop_ids` | **−414.1** | **+42.4** | **+66.2** | −57.1 |

### 타깃 간 비교 (Δ bss_norm)

BSS 배율이 4.01~8.74 로 달라 위 표는 타깃 간 직접 비교가 무효다.
Brier 개선률로 환산하면:

| 변환 | middle | reverse | ball |
|---|---:|---:|---:|
| `no_trackman` | **+0.77** | 0.00 | −0.14 |
| `trackman_compact` | +0.55 | −0.02 | −0.43 |
| `temporal_cyclic` | +0.55 | +0.23 | −0.17 |
| `drop_ids` | **−2.37** | +0.35 | **+0.63** |
| `no_component` | −0.65 | **−1.16** | −0.33 |

**middle 이 전처리에 가장 민감하다.** 변동폭이 다른 두 타깃의 두 배가 넘는다.
1WAY 공유 피처에서 가장 손해 보던 타깃이 middle 일 가능성이 높다.

---

## S2 조합 빔 서치 — 진행 중

3폭 × 3라운드, 타깃 3개. 약 210 fit / 3.5시간.
S1 단일 결과는 캐시 재사용하므로 라운드 1 은 무료다.

> 결과는 완료 후 갱신합니다.

---

## 앞으로

| 단계 | 상태 |
|---|---|
| S0 하네스 · 라벨 검증 | 완료 |
| **S1 타깃별 단일 스크리닝** | **완료 — 전제 확인** |
| S2 타깃별 조합 빔 | 진행 중 |
| S3 타깃별 계층 차감 축 | 대기 |
| S4 타깃별 학습 방식 | 대기 |
| S5 결합 + 최종 라벨 미세조정 | 대기 |
| S6 1WAY 정면 비교 | 대기 |

S3 예상 — S1 이 이미 방향을 시사한다.
middle 은 투수 축이 결정적이니 `EB(투수, ...)` 계열이 크게 들을 것이고,
reverse·ball 은 ID 를 버리는 게 나으므로 **투수 키 계층 차감이 안 들을 수 있다.**
그렇다면 카운트·상황 축을 투수 없이 거는 형태를 시험해야 한다.
