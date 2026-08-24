from __future__ import annotations

import json
import sys
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
sys.path.insert(0, str(MODEL_DIR))
from run_optuna_family import probability_metrics  # noqa: E402


def load_base_predictions():
    p23 = pd.read_parquet(
        MODEL_DIR / "insight_feature_ablation_predictions_success_adjusted_2023.parquet"
    )
    p24 = pd.read_parquet(
        MODEL_DIR / "insight_feature_ablation_predictions_success_screen_2024.parquet"
    )
    model = "xgboost_insight_insight_success_adjusted"
    p23 = p23.loc[p23["model"].eq(model)]
    p24 = p24.loc[p24["model"].eq(model)]
    return pd.concat([p23, p24], ignore_index=True)


def best_performance_blend(frame, probability):
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].rename(columns={"prediction": "performance"})
    joined = frame[["row_id", "control_success"]].copy()
    joined["candidate"] = probability
    joined = joined.merge(performance, on="row_id", validate="one_to_one")
    best = None
    for weight in np.linspace(0.0, 1.0, 201):
        pred = weight * joined["candidate"] + (1.0 - weight) * joined["performance"]
        metrics = probability_metrics(joined["control_success"], pred)
        if best is None or metrics["brier"] < best["brier"]:
            best = {"blend_weight": float(weight), **metrics}
    return best


def main():
    registry = pd.read_csv(WORK / "reports" / "matchup_registry.csv")
    base = load_base_predictions()
    rows = []
    subsets = {
        "pair": [
            "match_pair_delta", "match_pair_delta_reliability",
            "match_pair_delta_rate", "match_pair_known",
        ],
        "hier": [
            "match_pair_delta", "match_pair_delta_reliability",
            "match_pitcher_bhand_delta", "match_pitcher_bhand_delta_reliability",
            "match_phand_batter_delta", "match_phand_batter_delta_reliability",
            "batter_overall_resid", "match_pair_known",
        ],
        "all": None,
    }
    for _, spec in registry.iterrows():
        config = spec["matchup_config"]
        cache = pd.read_parquet(WORK / "oof" / f"matchup_features_{config}.parquet")
        frame = base.merge(cache, on=["row_id", "season"], validate="one_to_one")
        train = frame["season"].eq(2023)
        valid = frame["season"].eq(2024)
        available = [column for column in cache if column not in {"row_id", "season"}]
        for subset, requested in subsets.items():
            features = available if requested is None else [c for c in requested if c in available]
            train_x = frame.loc[train, features]
            valid_x = frame.loc[valid, features]
            train_residual = (
                frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
            ).to_numpy("float64")
            valid_base = frame.loc[valid, "prediction"].to_numpy("float64")
            valid_y = frame.loc[valid, "control_success"].to_numpy("int8")
            for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    Ridge(alpha=alpha, fit_intercept=False),
                )
                model.fit(train_x, train_residual)
                correction = np.clip(model.predict(valid_x), -0.05, 0.05)
                prediction = np.clip(valid_base + correction, 1e-6, 1 - 1e-6)
                metrics = probability_metrics(valid_y, prediction)
                row = {
                    **{key: spec[key] for key in [
                        "matchup_config", "pitcher_config", "batter_algorithm",
                        "batter_k_left", "batter_k_right", "smoothing", "half_life",
                    ]},
                    "subset": subset,
                    "alpha": alpha,
                    "feature_count": len(features),
                    "correction_std": float(np.std(correction)),
                    **metrics,
                }
                rows.append(row)
        print(json.dumps({"screened": config}, ensure_ascii=False), flush=True)
    result = pd.DataFrame(rows)
    top = result.sort_values("brier").groupby("matchup_config", as_index=False).first()
    base24 = base.loc[base["season"].eq(2024)]
    base_metrics = probability_metrics(base24["control_success"], base24["prediction"])
    result["delta_bss_vs_base"] = result["bss"] - base_metrics["bss"]
    result.to_csv(WORK / "reports" / "matchup_ridge_screen_all.csv", index=False)
    top.to_csv(WORK / "reports" / "matchup_ridge_screen_best.csv", index=False)
    print("\nTOP 10")
    print(top.sort_values("brier").head(10).to_string(index=False), flush=True)
    print(json.dumps({"base": base_metrics, "runs": len(result)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
