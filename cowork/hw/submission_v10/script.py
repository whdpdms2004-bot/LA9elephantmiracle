"""submission_v10 inference script. Self-contained, relative paths only.
Reads ./data/test.csv, writes ./output/submission.csv (row_id, control_success).

v9_bucketoffset(857.40, 실LB 확인됨) 위에 딱 한 가지만 바꾼 버전: pitcher_team_id/
batter_team_id를 숫자가 아니라 CatBoost 네이티브 범주형으로 학습 (yn님의
903.37->939.94 여정 문서에서 착안 -- "team_id가 범주형 분기를 못 쓰고 숫자로만
처리되고 있었다"가 핵심 원인이었음). 나머지(feature set, 하이퍼파라미터, 시드 수,
플래툰 lookup 방식)는 전부 v9와 동일 -- 한 번에 하나만 바꾼다(AGENTS.md 원칙).

정직한(fit<2024, val==2024) 3개년 교차검증에서 2022 +14.57 / 2023 +2.36 /
2024 +33.92 전부 양의 방향 확인 (validate_teamid_categorical.py). 새 팀 fold
기준(2022 제외, 결정 fold=2024)으로 봐도 결정 fold가 세 해 중 가장 큰 이득이라
채택 근거 그대로 유지됨.

Model: CatBoost 16-seed bagging (simple mean), v9와 동일 하이퍼파라미터.
cat_features = [top_bottom, game_type, base_state, pitcher_team_id, batter_team_id]
(마지막 둘이 신규 -- 둘 다 고유값 13개뿐이라 저카디널리티, 과적합 위험 낮음).

Feature set: anchor(baseline47 + trend6) + platoon_split + platoon_n (55 total,
v9와 동일 -- feature set 자체는 안 바뀜, cat_features 등록만 바뀜).
platoon_split/platoon_n은 STATIC LOOKUP TABLE(model/platoon_lookup.csv)에서
가져옴, train.csv(2019-2024) 전체로 fit 이후 생성 -- test 행 간 참조 없음.

★ BUCKET_OFFSETS -- train-data-only로 재산출, 리더보드 미참조 (RULES.md §2).
target=0.4792는 v9와 동일(2019~2024 추세외삽, 데이터 속성이라 모델이 바뀌어도
안 바뀜). 구간별 pred_mean은 이 새 모델(team_id 범주형)의 PHASE1(fit<2024,
정직한 held-out) 16-seed 앙상블 예측으로 재계산함 -- production 체크포인트를
in-sample로 평가하는 함정(Platt scaling 실험에서 걸렸던 것)을 피하기 위해
일부러 PHASE1 모델만 사용:

    bucket          pred_mean(fit<2024 정직 앙상블, val=2024) offset = logit(0.4792)-logit(pred_mean)
    n < 200         0.4693                                    +0.039570
    200 <= n < 2000 0.4807                                    -0.005819
    n >= 2000       0.5021                                    -0.091600

Only train.csv was used at training time (train_best_model_v10_teamid.py). This
script itself only reads test.csv + the precomputed lookup table -- no train.csv
re-read, no test-internal statistics, no rolling/expanding/target-encoding on
test. predict(single row) == predict(full test)[i] holds because every input is
a per-row feature (lookup merge by the row's own pitcher_id/batter_hand only)
computed independently, and every model constant (medians, lookup table, model
weights, offset) was fixed at training time.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "output"
MODEL_DIR = HERE / "model"

ID = "row_id"
TARGET = "control_success"
PREV_PAIRS = (1, 3, 5)
# 구간 경계(오른쪽 미포함, v9_bucketoffset과 동일) + 구간별 오프셋(이 모델용으로 재산출).
BUCKET_EDGES = [-1, 200, 2000, float("inf")]
BUCKET_OFFSETS = [0.03956972950416722, -0.00581874264721248, -0.0915996024128361]  # low, mid, high 순


def _require_file(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"required file not found: {p}")
    return p


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for k in PREV_PAIRS:
        recent_col = f"asof_pitcher_prev{k}_game_success_rate"
        x[f"trend_prev{k}"] = x[recent_col] - x["asof_pitcher_success_rate"]
        x[f"trend_abs_prev{k}"] = x[f"trend_prev{k}"].abs()
    return x


def apply_platoon_lookup(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    x = df.merge(lookup, on=["pitcher_id", "batter_hand"], how="left")
    x["platoon_split"] = x["platoon_split"].fillna(0.0)  # 못 본 (투수,타자손) 조합 -> 보정 없음
    x["platoon_n"] = x["platoon_n"].fillna(0.0)
    return x


def main():
    test_path = _require_file(DATA_DIR / "test.csv")
    feat_path = _require_file(MODEL_DIR / "feature_cols.json")
    medians_path = _require_file(MODEL_DIR / "medians.json")
    lookup_path = _require_file(MODEL_DIR / "platoon_lookup.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(feat_path) as f:
        feat_meta = json.load(f)
    feature_cols = feat_meta["feature_cols"]
    cat_features = feat_meta["cat_features"]
    num_cols = feat_meta["num_cols"]
    cat_files = [m["file"] for m in feat_meta["manifest"]["catboost"]]
    if not cat_files:
        raise RuntimeError("no catboost models listed in feature_cols.json manifest")

    with open(medians_path) as f:
        medians = json.load(f)
    lookup = pd.read_csv(lookup_path)

    test = pd.read_csv(test_path)
    if ID not in test.columns:
        raise KeyError(f"test.csv missing required id column: {ID}")

    x = add_trend(test)
    x = apply_platoon_lookup(x, lookup)

    missing_feats = [c for c in feature_cols if c not in x.columns]
    if missing_feats:
        raise KeyError(f"missing feature columns at inference time: {missing_feats}")

    x_model = x[feature_cols].copy()
    med_series = pd.Series(medians)
    fill_cols = [c for c in num_cols if c in x_model.columns]
    x_model[fill_cols] = x_model[fill_cols].fillna(med_series[fill_cols])
    for c in cat_features:
        if c in x_model.columns:
            x_model[c] = x_model[c].fillna("__NA__").astype(str)

    preds = []
    for fname in cat_files:
        model_path = _require_file(MODEL_DIR / fname)
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        preds.append(model.predict_proba(x_model)[:, 1])

    proba = np.mean(np.vstack(preds), axis=0)  # same-row seed average only, no cross-row stats

    nan_mask = np.isnan(proba)
    if nan_mask.any():
        proba = np.where(nan_mask, 0.5, proba)
    proba = np.clip(proba, 1e-7, 1.0 - 1e-7)

    # 구간별 오프셋: 각 행 자신의 asof_pitcher_n만 보고 구간을 정함 (test 행 간 참조 없음)
    pitcher_n = test["asof_pitcher_n"].fillna(0).to_numpy()
    bucket_idx = np.digitize(pitcher_n, BUCKET_EDGES[1:-1])  # 0,1,2 중 하나
    row_offset = np.array(BUCKET_OFFSETS)[bucket_idx]
    logit = np.log(proba / (1.0 - proba)) + row_offset
    proba = 1.0 / (1.0 + np.exp(-logit))

    proba = np.clip(proba, 0.001, 0.999)

    submission = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: proba})

    assert len(submission) == len(test), "submission row count must match test.csv"
    assert set(submission.columns) == {ID, TARGET}, "submission must have exactly row_id, control_success"
    assert submission[TARGET].between(0, 1).all(), "control_success predictions must be in [0,1]"

    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"wrote {OUT_DIR / 'submission.csv'} rows={len(submission)} n_models={len(cat_files)}")


if __name__ == "__main__":
    main()
