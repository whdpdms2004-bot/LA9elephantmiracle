# ye_hand

팀원 **ye** 의 `model_v3.ipynb`(M2_contam 구조) 를 정직 OOF 규격으로 재현한 등록본.
원본 노트북(`cowork/ye/model_v3.ipynb`)은 `eval_set` 조기종료를 써서(§2 참고) 그 자체
수치(801~804, Public 840)를 결합 판정에 그대로 쓸 수 없다 — 여기 등록된 수치가
정직(leak-free) 재현본이다.

| 항목 | 값 |
|---|---|
| 제출일 | 2026-08-29 |
| zip | 없음 — 팀 결합 기각으로 제출 후보에서 빠짐 (아래 §1·§3) |
| 학습 스크립트 | `models/ye_hand/train_ye_hand.py` |
| val 예측 | `val/ye_hand_2022.csv`, `val/ye_hand_2023.csv`, `val/ye_hand_2024.csv` |
| 기준 모델 | `cw_v17_base` (2024 all 821.0 / 2022 R 754.0) |
| 판정 | **기각** (2024 미상승, 2022 R 하락) |

---

## 1. 성능

| 시즌 | all | R | F | pred_mean | true_mean |
|---|---:|---:|---:|---:|---:|
| 2024 (주) | 798.4 | 797.7 (n=223,497) | 477.4 (n=30,010) | 0.4933 | 0.4861 |
| 2023 | **−1,230.3** | 547.9 (n=219,839) | **−16,825.0** (n=25,686) | 0.5200 | 0.5000 |
| 2022 (비하락 조건) | 2,370.2 | 620.7 (n=217,024) | 31.2 (n=30,448) | 0.5311 | 0.5289 |

기준 모델(`cw_v17_base`) 대비 델타:

| 시즌 | Δall / ΔR | 하락? |
|---|---:|---|
| 2024 all | −22.6 | — (미상승) |
| 2022 R | −133.2 | 예 |

Public: **미측정** — 이 정직 재현본으로는 제출한 적 없음. 원본 `model_v3.ipynb`
노트북(WEIGHTED_DIRECT, eval_set 조기종료 포함 구성)의 Public 840 을 이 등록본의
값으로 인용하면 안 된다 (구성이 다르다 — §2 참고).

### ★ 2023 F 붕괴 — ye_hand 만의 결함이 아니다

2023 은 all·F 가 크게 무너지지만 **R 은 양호(547.9)** 하다. 팀의 다른 모델도
동일하게 무너진다 (`corr.py` 로 오늘 확인): hw_v12 −1,232.0 · yn_fa10c −1,207.4 ·
ye_hand −1,230.3, 전부 비슷한 규모. **2022→2023 game_type 구조단절**(README §0
규칙1 근거) 때문으로 보인다. 규칙 1 은 2023 을 `R` 만 관문으로 쓰므로, all 붕괴
자체가 관문 실패는 아니다.

## 2. 모델 구성

- **타깃/구조**: 1WAY 직접 `control_success`. CatBoost 3-시드 배깅.
- **결합 방식**: 시드 평균만 (팀 결합 이전 개별 멤버 단계).
- **베이스 학습기**: CatBoost `Logloss`/`BrierScore`, depth 6, lr 0.03,
  `l2_leaf_reg` 10.0, `random_strength` 0.5, `border_count` 128,
  **고정 iterations=349**, 시드 11/22/33.
- **피처 세트**: 65개. 원본 as-of 피처 + `reliability_*`/`log1p_*`(투수·타자·구종
  표본신뢰도) + `platoon_split_eb`/`platoon_n_reliability`(투수 전체 대비 상대손
  플래툰 편차, EB 평활 k=200) + `fe_pitcher_futures_share`/`fe_batter_futures_share`/
  `fe_pitcher_prior_n_log`(투수·타자별 과거 game_type=F 비중 + 과거 표본수 로그,
  전부 as-of). 범주형 8개: `game_dayofweek`,`top_bottom`,`game_type`,`base_state`,
  `pitcher_hand`,`batter_hand`,`pitcher_team_id`,`batter_team_id`.
- **표본가중**: `f_recent_strong` — F 게임의 최근 3시즌만 가중치를 1.00/0.60/0.40/
  0.30 로 감쇠(그 외 R 게임·오래된 F 는 그대로/0.30). 노트북 자체 screening 에서
  2024 +28.0(대비 equal) 로 승자.
- **학습 구간**: 폴드별 `season < val_year` 전체. val 예측은 해당 시즌 제외.
- **후처리**: 없음 (raw 확률).

### ★ 원본 노트북과의 차이 — eval_set 조기종료 제거

`model_v3.ipynb` 의 `fit_weighted_direct()` 는
`eval_set=(va_df,...) + early_stopping_rounds=200 + use_best_model=True` 를 쓴다 —
평가 시즌 자신의 라벨을 조기 종료에 쓰는 것으로, hw 가 걸렸던 오염 패턴(팀 규칙 2
위반)과 동일하다. 여기서는 eval_set 없이 **고정 iteration=349** 로 학습한다.
349 는 노트북 자신의 5-seed 실험(`run_direct_5seed`)이 계산해 둔
`median_iteration` 값이자, 노트북의 (미실행) 최종 10-seed 빌드 셀이 쓰려던 값이다
— 임의로 새로 고른 게 아니라 노트북이 이미 지목한 하이퍼파라미터를 세 폴드에
동일하게 고정 적용한 것이다.

### `CONTAM_FEATURES` 이름에 대해

`fe_pitcher_futures_share` 등 3열은 원본 노트북에서 "contamination features" 로
명명돼 있으나, 실제로는 **과거 시즌(`season < S`) 이력만 쓰는 정상적인 as-of
피처**다 (시즌 경계 검증 완료). 이름과 실제 동작이 다르다는 점을 다음 사람이
헷갈리지 않도록 남긴다.

## 3. 앙상블 관점

val 실측 (2024 n=253,507 / 2022 n=247,472):

| 상대 모델 | 확률 상관 | 오차 상관 | 비고 |
|---|---:|---:|---|
| `cw_v17_base` (2024) | 0.9025 | **0.9992** | |
| `cw_v17_base` (2022) | 0.9670 | **0.9992** | |
| `sj_stdmlp`(배포 챔피언, 2024) | 0.9148 | **0.9992** | |
| `sj_stdmlp`(배포 챔피언, 2022) | 0.9684 | **0.9993** | |

정직 프로토콜(2022 적합 → 2024 동결, `sj_stdmlp` 기준):

```
2022 적합:  최적 λ = 0.500 (격자 끝)   2353.1 -> 2432.0  (Δ+78.9)
2024 동결:                              916.0 ->  905.6  (Δ-10.33)  -> 기각
```

2022 에서 직접 λ 를 고르면 좋아 보이지만(착시), 그 λ 를 2024 에 그대로 적용하면
부호가 뒤집힌다 — 팀 규약(적합 fold ≠ 평가 fold)이 정확히 막는 함정.

- 이 모델이 **남과 다르게 맞히는 구간**: 확인 안 됨 (부분군 진단 미실시).
- 결합에서 빼야 하는 조건: **오차 상관이 어느 상대와도 0.999 근방** — 확률 상관은
  cw·sj 대비 상대적으로 낮지만(0.90~0.97), 실제 결합가치를 결정하는 오차 상관은
  거의 구분이 안 된다. `hw_v12`·`yn_fa10c` 도 동일 이유로 팀 결합에서 가중 0 이었다
  (`DECK.md` §5.1).

> 참고 — 구버전(`champion_structural_improvement.ipynb`, 60피처, sj 재현) 은
> 확률 상관이 더 낮다(0.79~0.87, 피처 20개만 cw/sj 와 겹침) 하지만 **오차 상관은
> 0.997~0.998 로 마찬가지로 높다.** 이 데이터셋 자체가 예측 분산 대비 라벨 분산이
> 커서(`DECK.md` §6.3) 어떤 표현을 쓰든 오차 상관이 구조적으로 1 에 가깝게 나오는
> 것으로 보인다 — 확률 상관을 낮추는 것과 오차 상관을 낮추는 것이 이 문제에서는
> 거의 별개다.

## 4. 규정 점검 (cowork/RULES.md 0절)

- [ ] `script.py` 는 추론 전용 — **N/A** (제출 zip 자체가 없음, 결합 기각으로 미제출)
- [x] test 행 간 정보 사용 없음 — `pitcher_id`/`batter_id` 로 한 행씩 조회, 집계 없음
- [x] 외부 데이터·외부 API·추론 시 인터넷 접근 없음
- [x] 투구 이후 시점 정보 없음 — 전부 `season < S` as-of
- [x] `requirements.txt` 만으로 실행 재현됨 — `models/ye_hand/requirements.txt`

## 5. 재현 절차

```bash
# performance_tracking/models/ye_hand/ 안에서 (저장소 루트/data 를 읽는다)
python train_ye_hand.py
```

소요: CPU, 약 5~7분 (3폴드 × 3시드, 1.47M행). 데이터 as-of: `season < val_year`
행만으로 매 폴드 lookup 을 새로 만든다 (그룹핑·평활 전부 폴드별 재계산).

## 6. 남은 것

- 부분군(구간별) 진단 — 이 모델이 어디서 강한지 아직 안 봄
- 2023 F 붕괴의 정확한 기전 규명 (팀 공통 이슈로 보이나 원인 미확정)
- 하이퍼파라미터 튜닝은 미실시 상태였음 — 2026-08-29 재시도 진행 중
  (depth/l2_leaf_reg 그리드 + `sj_stdmlp`/`cw_v17_base` 와 조합 재검증,
  `performance_tracking/sj_run/`, 결과는 이 파일에 갱신 예정)
