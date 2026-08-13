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
LARGE_DIR = MODEL_DIR / "large_xgb" / "predictions"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
REPORTS = WORK / "reports"
SUCCESS_CONFIG = "m_com_gmm_kme_b3r4_l1000_h1_d72788b5"
SUCCESS_FEATURES = [
    "match_pair_delta",
    "match_pair_delta_reliability",
    "match_pair_delta_rate",
    "match_pair_known",
]
REVERSE_FEATURES = [
    "reverse_pair_delta",
    "reverse_pair_delta_reliability",
    "reverse_pair_rate",
    "reverse_pair_known",
]
REVERSE_SEEDS = [17, 2026, 4099]
TARGET = "control_success"


def probability_metrics(y, prediction):
    y = np.asarray(y, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    brier = float(np.mean(np.square(p - y)))
    denominator = float(y.mean() * (1.0 - y.mean()))
    return {
        "brier": brier,
        "normalized_brier": brier / denominator,
        "bss": max(0.0, 100000.0 * (1.0 - brier / denominator)),
        "target_mean": float(y.mean()),
        "pred_mean": float(p.mean()),
    }


def read_prediction(candidate, fold):
    path = LARGE_DIR / f"{candidate}_f{fold}_s0.parquet"
    frame = pd.read_parquet(path).sort_values("row_id").reset_index(drop=True)
    return frame[["row_id", "season", TARGET, "prediction"]]


def distribution_parameters(source, reference):
    source = np.asarray(source, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    scale = float(reference.std() / max(source.std(), 1e-8))
    intercept = float(reference.mean() - scale * source.mean())
    return scale, intercept


def probability_mean_std_match(source, scale, intercept):
    source = np.asarray(source, dtype="float64")
    return np.clip(
        scale * source + intercept,
        1e-6,
        1.0 - 1e-6,
    )


def load_large_base():
    raw = {}
    for fold in [2022, 2023, 2024]:
        anchor = read_prediction("anchor_logloss", fold)
        diverse = read_prediction("moderate24_diverse", fold)
        if not anchor[["row_id", "season", TARGET]].equals(
            diverse[["row_id", "season", TARGET]]
        ):
            raise RuntimeError(f"Large prediction alignment failed for {fold}")
        raw[fold] = (anchor, diverse)
    pieces = []
    parameters = {}
    for fold in [2022, 2023, 2024]:
        source_fold = 2022 if fold == 2022 else fold - 1
        source_anchor, source_diverse = raw[source_fold]
        scale, intercept = distribution_parameters(
            source_diverse["prediction"], source_anchor["prediction"]
        )
        parameters[fold] = {
            "source_season": source_fold,
            "scale": scale,
            "intercept": intercept,
        }
        anchor, diverse = raw[fold]
        output = anchor[["row_id", "season", TARGET]].copy()
        output["prediction"] = probability_mean_std_match(
            diverse["prediction"], scale, intercept
        )
        output["anchor_prediction"] = anchor["prediction"].to_numpy("float32")
        output["large_raw_prediction"] = diverse["prediction"].to_numpy("float32")
        pieces.append(output)
    return pd.concat(pieces, ignore_index=True), parameters


def fit_correction(frame, features, alpha, train_year, valid_year):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = frame.loc[train, TARGET] - frame.loc[train, "prediction"]
    model.fit(frame.loc[train, features], residual)
    return np.clip(model.predict(frame.loc[valid, features]), -0.05, 0.05)


def brier_delta(error, correction):
    return float(np.mean(2.0 * error * correction + np.square(correction)))


def robust_objective(f23, f24, denominators):
    n23 = f23 / denominators[2023]
    n24 = f24 / denominators[2024]
    return 0.30 * n23 + 0.70 * n24 + 0.50 * max(n23, n24, 0.0)


def best_pair_weight(y, left, right):
    direction = right - left
    denominator = float(np.dot(direction, direction))
    weight = 0.0 if denominator == 0 else float(
        np.clip(np.dot(y - left, direction) / denominator, 0.0, 1.0)
    )
    return weight, left + weight * direction


def main():
    base, distribution_parameters_by_fold = load_large_base()
    success = pd.read_parquet(
        WORK / "oof" / f"matchup_features_{SUCCESS_CONFIG}.parquet",
        columns=["row_id", "season", *SUCCESS_FEATURES],
    )
    frame = base.merge(success, on=["row_id", "season"], validate="one_to_one")
    seed_frames = {}
    for seed in REVERSE_SEEDS:
        reverse = pd.read_parquet(
            WORK / "oof" / "reverse_batter_seed" / f"seed_{seed}.parquet"
        )
        seed_frames[seed] = base.merge(
            reverse, on=["row_id", "season"], validate="one_to_one"
        )

    folds = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid = frame["season"].eq(valid_year)
        y = frame.loc[valid, TARGET].to_numpy("float64")
        prediction = frame.loc[valid, "prediction"].to_numpy("float64")
        success_correction = fit_correction(
            frame, SUCCESS_FEATURES, 10.0, train_year, valid_year
        )
        reverse_correction = np.mean(
            [
                fit_correction(
                    seed_frames[seed],
                    REVERSE_FEATURES,
                    1000.0,
                    train_year,
                    valid_year,
                )
                for seed in REVERSE_SEEDS
            ],
            axis=0,
        )
        folds[valid_year] = {
            "row_id": frame.loc[valid, "row_id"].to_numpy(),
            "y": y,
            "prediction": prediction,
            "error": prediction - y,
            "success": success_correction,
            "reverse": reverse_correction,
            "denominator": float(y.mean() * (1.0 - y.mean())),
        }

    denominators = {year: fold["denominator"] for year, fold in folds.items()}
    rows = []
    scale_grid = np.round(np.arange(0.0, 0.801, 0.025), 3)
    for success_scale in scale_grid:
        for reverse_scale in scale_grid:
            deltas = {}
            item = {
                "success_scale": float(success_scale),
                "reverse_scale": float(reverse_scale),
            }
            for year, fold in folds.items():
                correction = (
                    success_scale * fold["success"]
                    + reverse_scale * fold["reverse"]
                )
                delta = brier_delta(fold["error"], correction)
                deltas[year] = delta
                item[f"f{str(year)[-2:]}_delta_brier"] = delta
            item["both_improve"] = deltas[2023] < 0 and deltas[2024] < 0
            item["robust_objective"] = robust_objective(
                deltas[2023], deltas[2024], denominators
            )
            rows.append(item)
    scale_results = pd.DataFrame(rows).sort_values("robust_objective")
    selected = scale_results.iloc[0]

    corrected = {}
    corrected_metrics = []
    for year, fold in folds.items():
        prediction = np.clip(
            fold["prediction"]
            + float(selected["success_scale"]) * fold["success"]
            + float(selected["reverse_scale"]) * fold["reverse"],
            1e-6,
            1.0 - 1e-6,
        )
        corrected[year] = prediction
        corrected_metrics.append(
            {"fold": year, **probability_metrics(fold["y"], prediction)}
        )

    ensemble = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    outer_rows = []
    for year, track in [(2023, "robust"), (2024, "performance")]:
        fold = folds[year]
        base_prediction = (
            ensemble[
                ensemble["season"].eq(year) & ensemble["track"].eq(track)
            ]
            .set_index("row_id")["prediction"]
            .reindex(fold["row_id"])
            .to_numpy("float64")
        )
        insight_weight, prediction = best_pair_weight(
            fold["y"], base_prediction, corrected[year]
        )
        outer_rows.append(
            {
                "fold": year,
                "base_track": track,
                "insight_weight": insight_weight,
                **probability_metrics(fold["y"], prediction),
            }
        )

    REPORTS.mkdir(parents=True, exist_ok=True)
    scale_results.to_csv(REPORTS / "large_xgb_correction_grid.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(
        REPORTS / "large_xgb_correction_outer.csv", index=False
    )
    summary = {
        "base": "moderate24_diverse seed0 with fixed prior-season probability mean/std affine mapping",
        "distribution_parameters_by_fold": distribution_parameters_by_fold,
        "correction_training": "rolling residual refit: 2022->2023 and 2023->2024",
        "success_config": SUCCESS_CONFIG,
        "success_alpha": 10.0,
        "reverse_seeds": REVERSE_SEEDS,
        "reverse_alpha": 1000.0,
        "selected": selected.to_dict(),
        "corrected_metrics": corrected_metrics,
        "outer_results": outer_rows,
        "artifacts": {
            "grid": str(REPORTS / "large_xgb_correction_grid.csv"),
            "outer": str(REPORTS / "large_xgb_correction_outer.csv"),
        },
    }
    (REPORTS / "large_xgb_correction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
