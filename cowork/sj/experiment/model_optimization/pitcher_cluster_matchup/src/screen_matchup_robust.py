from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
MODEL_NAME = "xgboost_insight_insight_success_adjusted"
FEATURES = [
    "match_pair_delta", "match_pair_delta_reliability",
    "match_pair_delta_rate", "match_pair_known",
]


def load_base():
    paths = [
        MODEL_DIR / "insight_feature_ablation_predictions_cluster_base_2022.parquet",
        MODEL_DIR / "insight_feature_ablation_predictions_success_adjusted_2023.parquet",
        MODEL_DIR / "insight_feature_ablation_predictions_success_screen_2024.parquet",
    ]
    pieces = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame["model"].nunique() > 1:
            frame = frame.loc[frame["model"].eq(MODEL_NAME)]
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def transition(frame, train_year, valid_year, alpha):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = (
        frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    ).to_numpy("float64")
    model.fit(frame.loc[train, FEATURES], residual)
    correction = np.clip(model.predict(frame.loc[valid, FEATURES]), -0.05, 0.05)
    y = frame.loc[valid, "control_success"].to_numpy("float64")
    base = frame.loc[valid, "prediction"].to_numpy("float64")
    prediction = np.clip(base + correction, 1e-6, 1 - 1e-6)
    base_brier = float(np.mean((base - y) ** 2))
    brier = float(np.mean((prediction - y) ** 2))
    centered = np.clip(base + correction - correction.mean(), 1e-6, 1 - 1e-6)
    return {
        "base_brier": base_brier,
        "brier": brier,
        "delta_brier": brier - base_brier,
        "correction_mean": float(correction.mean()),
        "correction_std": float(correction.std()),
        "centered_delta_brier_diagnostic": float(np.mean((centered - y) ** 2) - base_brier),
    }


def main():
    registry = pd.read_csv(WORK / "reports" / "matchup_registry.csv")
    base = load_base()
    rows = []
    for _, spec in registry.iterrows():
        config = spec["matchup_config"]
        cache = pd.read_parquet(WORK / "oof" / f"matchup_features_{config}.parquet")
        frame = base.merge(cache, on=["row_id", "season"], validate="one_to_one")
        for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
            f23 = transition(frame, 2022, 2023, alpha)
            f24 = transition(frame, 2023, 2024, alpha)
            robust = (
                0.30 * f23["delta_brier"]
                + 0.70 * f24["delta_brier"]
                + 0.50 * max(f23["delta_brier"], f24["delta_brier"], 0.0)
            )
            rows.append({
                **{key: spec[key] for key in [
                    "matchup_config", "batter_k_left", "batter_k_right",
                    "smoothing", "half_life",
                ]},
                "alpha": alpha,
                "f23_delta_brier": f23["delta_brier"],
                "f24_delta_brier": f24["delta_brier"],
                "f23_correction_mean": f23["correction_mean"],
                "f24_correction_mean": f24["correction_mean"],
                "f23_centered_delta_diagnostic": f23["centered_delta_brier_diagnostic"],
                "f24_centered_delta_diagnostic": f24["centered_delta_brier_diagnostic"],
                "both_improve": f23["delta_brier"] < 0 and f24["delta_brier"] < 0,
                "robust_objective": robust,
            })
        print(json.dumps({"screened": config}, ensure_ascii=False), flush=True)
    result = pd.DataFrame(rows).sort_values("robust_objective")
    result.to_csv(WORK / "reports" / "matchup_robust_screen.csv", index=False)
    print("\nROBUST TOP 15")
    print(result.head(15).to_string(index=False), flush=True)
    print(json.dumps({
        "runs": len(result),
        "both_improve": int(result["both_improve"].sum()),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
