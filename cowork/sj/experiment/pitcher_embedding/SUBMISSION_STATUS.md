# Pitcher embedding submission status

## Current recommendation

- `submit_v1.zip` is an executable **experimental embedding submission**, not the safe baseline.
- The official `baseline_submit.zip` remains the score-reference submission (official reference score: 549.51).
- The embedding model's best temporal holdout Brier Skill Score is 363.23, so it should not replace the official baseline without leaderboard evidence.
- Use the embedding output as a team feature or blending candidate first.

## Validation design

- Fit: 2019–2022
- Model selection and affine-logit calibration: 2023
- Final untouched report: 2024
- The selected architecture was the conditional component model:
  - reverse
  - middle given no reverse
  - far residual given neither
- Final success probability is the product of the three conditional survival probabilities.

## Results

| Experiment | 2024 Brier | 2024 BSS |
|---|---:|---:|
| Direct head, quick, calibrated | 0.249129 | 279.09 |
| Component head, quick, calibrated | 0.248919 | **363.23** |
| Component head, full, calibrated | 0.249022 | 313.58 |
| LightGBM recent-window diagnostic | 0.249566 | 96.60 |

The quick 600,000-row final artifact was retained because its temporal validation pipeline outperformed the full-row pipeline.

## Rule compliance checks

- No aggregation, encoding, calibration, crosswalk, or distribution statistic is computed from test rows.
- Each row uses only its own supplied values and a frozen train+Trackman lookup.
- Whole-batch, reversed-batch, and single-row predictions agree within floating-point tolerance (`< 1e-7`).
- The 245,789-row local GPU benchmark completed model feature creation and inference in 0.676 seconds.
- The zip contains exactly:
  - `model/model.pt`
  - `script.py`
  - `requirements.txt`
- The zip integrity check passed.

## Team feature export

- `outputs/pitcher_season_embedding_oof.parquet`
  - 2,260 unique `(pitcher_id, season)` rows
  - 48 embedding dimensions
  - 2021–2024: season-forward OOF, 100% available
  - 2019–2020: zero fallback with `oof_available=False`
  - enforced condition: `trained_through_season < season`
- `outputs/pitcher_embedding_lookup_2025.parquet`
- 792 known Main pitcher IDs
- 48 dimensions:
  - 16-dimensional supervised pitcher ID representation
  - 24-dimensional prior Trackman tower representation
  - 8-dimensional rookie/experience cohort representation
- The table is trained through 2024 and is intended for 2025 inference only.
- It must not be joined back to the same 2019–2024 training rows. Training-time stacking requires temporal/OOF embeddings.

## Auxiliary-label question for DACON

> **[DACON 답변 요청] train.csv 누적 asof 피처를 이용한 보조 라벨 생성 가능 여부**
>
> 안녕하세요. 동일 투수의 다음 학습 행에 제공된 `asof_pitcher_reverse_rate` 및 `asof_pitcher_middle_rate`의 누적 변화량을 이용하여 현재 학습 행의 reverse/middle 보조 라벨을 복원하고, 이를 학습용 정답으로만 사용하는 것이 허용되는지 문의드립니다. 모델 입력에는 현재 행에서 투구 직전까지 확인 가능한 정보만 사용하며, 테스트 데이터의 다른 행·순서·분포·누적 통계는 일절 사용하지 않습니다. 해당 방식이 허용되지 않는다면 `control_success`만 직접 학습하는 모델을 사용하겠습니다.

Do not submit or post this question automatically. A team member should post it in the competition talk board and retain the official answer.
