# SUBMISSION_LOG — 2026-08-20 · hw · `submit_v10.zip`

**결과: Public 892.1204835291** (직전 최고 857.4008954892 대비 **+34.7196**)

작성 근거: [`AGENTS.md`](../../../AGENTS.md) B1, [`RULES.md`](../../RULES.md) §2

---

## 1. 제출 구성

```text
55피처 = anchor(baseline47) + trend6 + platoon_split + platoon_n  (v9와 동일)
CatBoost 16-seed 단순평균
BASELINE_CATS = [top_bottom, game_type, base_state, pitcher_team_id, batter_team_id]
                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 이번에 신규
season logit offset: 표본수 3구간별(low/mid/high), 이 모델로 재계산
```

v9(857.40, 실LB 확인) 대비 바뀐 것 딱 하나: `pitcher_team_id`/`batter_team_id`를
**숫자가 아니라 CatBoost 네이티브 범주형**으로 학습. 둘 다 고유값 13개뿐(팀 수만큼)이라
저카디널리티, 과적합 위험 낮음.

## 2. 왜 시도했나 — yn님 발견 이식

yn님의 903.37→939.94 여정 문서에서 핵심 원인으로 지목한 것: `pitcher_team_id`/
`batter_team_id`가 지금까지 CatBoost 네이티브 범주형 분기를 못 쓰고 숫자로만 처리되고
있었음. 같은 구조적 허점이 내 파이프라인(`BASELINE_CATS`에 team_id 미포함)에도 있길래
검증함.

**정직한(fit<val_season) 3개년 교차검증** (train.csv만 사용, 리더보드 미참조):

| 연도 | team_id=숫자 | team_id=범주형 | 차이 |
|---|---:|---:|---:|
| 2022 | 2325.33 | 2339.90 | +14.57 |
| 2023 | 21.68 | 24.04 | +2.36 |
| 2024(결정) | 723.85 | 757.77 | **+33.92** |

세 해 전부 양의 방향 → production 반영.

## 3. ★ season logit offset 출처 명시 (필수 문구)

본 제출의 season logit offset(표본수 3구간, `[+0.039570, −0.005819, −0.091600]`)은
**학습 데이터(2019~2024 train.csv)만을 이용해 사전 결정된 상수**이며, 모든 평가 행에
행 자신의 `asof_pitcher_n` 값에 따라 동일한 규칙으로 적용된다.

- target(0.4792)은 v7/v9와 동일 — 2019~2024 시즌 추세 외삽으로 산출, 이번에 재산출하지
  않음(데이터 속성이지 모델 속성이 아님).
- 구간별 pred_mean은 **이 모델(team_id 범주형)의 PHASE1(fit<2024, 정직한 held-out)
  16-seed 앙상블 예측**으로 재계산 — production 체크포인트(2024 포함 학습)를 2024로
  평가하는 in-sample 오염을 피하기 위해 일부러 분리함.
- **리더보드 점수를 참조하거나 역산하여 조정한 값이 아니다.**
- 평가 데이터(`test.csv`)의 값, 분포, 평균, 순위를 일절 사용하지 않았다.
- 구간 경계(`[-1, 200, 2000, inf]`)는 v9_bucketoffset과 동일 — 이미 실LB +8.84로
  검증된 경계를 재사용.

## 4. 행 독립성 확인

`predict(단독 행) == predict(전체 test)[i]`, 5행 샘플로 행별 확인. 최대 차이
`1.11e-16`(float64 머신 엡실론 수준, 반올림 잡음이지 실제 위반 아님) — PASS.

## 5. 결과 및 다음

- **실LB +34.72** — 이번 세션 중 단일 구조변경으로는 최대 성과.
- team_id 범주형 위에 볼카운트 상태(`count_state`) 1개를 더 얹은 v11을 정직검증
  중(+14.43, 결정 fold). 조합탐색(빔서치)에서 다른 후보와 합쳐본 결과는 전부
  count_state 단독보다 못해서(E1+E8 −6.40, E1+futures −2.25) count_state 단독으로
  production 진행.
- 재현 코드: `train_best_model_v10_teamid.py`(같은 폴더), 실행법은 파일 docstring 참고.
