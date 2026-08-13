from __future__ import annotations

import itertools
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
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

from deep_pitcher_cluster_search import (  # noqa: E402
    CUTOFFS,
    REVERSE_FEATURES,
    SUCCESS_FEATURES,
    load_base,
    robust_objective,
)
from joint_svd_cluster_search import (  # noqa: E402
    DIRECT_FEATURES,
    pair_features,
    prepare_cutoff,
)
from screen_reverse_batter_clusters import load_main  # noqa: E402


TOP_JOINT = {
    "matrix_lambda": 500.0,
    "svd_dim": 16,
    "pitcher_k": (2, 4),
    "batter_k": (6, 8),
}
SEEDS = [17, 2026, 4099]
ALPHAS = [100.0, 1000.0, 10000.0]
SUCCESS_SCALES = np.round(np.arange(0.10, 0.401, 0.05), 2)
CURRENT_SCALES = np.round(np.arange(0.30, 0.701, 0.05), 2)
JOINT_SCALES = np.round(np.arange(-0.50, 0.501, 0.05), 2)


def correction(frame, features, alpha, train_year, valid_year):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    model.fit(frame.loc[train, features], residual)
    return np.clip(model.predict(frame.loc[valid, features]), -0.05, 0.05)


def quadratic(error, components):
    matrix = np.column_stack(components)
    return (
        np.mean(error[:, None] * matrix, axis=0),
        matrix.T @ matrix / len(matrix),
    )


def delta_brier(linear, gram, scales):
    scales = np.asarray(scales, dtype=float)
    return float(2.0 * np.dot(linear, scales) + scales @ gram @ scales)


def attach_outer_scores(result, folds, best_vectors):
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].set_index("row_id")
    fold = folds[2024]
    perf = performance.loc[fold["row_id"], "prediction"].to_numpy(float)
    perf_error = perf - fold["y"]
    direction0 = fold["base"] - perf
    perf_brier = float(np.mean(perf_error ** 2))
    output = result.copy()
    scale_columns = ["success_scale", "current_scale", "joint_scale"]
    for key, joint_by_year in best_vectors.items():
        feature_name, alpha, current_name = key
        mask = (
            output["joint_features"].eq(feature_name)
            & output["joint_alpha"].eq(alpha)
            & output["current_variant"].eq(current_name)
        )
        components = np.column_stack([
            fold["success"], fold[current_name], joint_by_year[2024]
        ])
        scales = output.loc[mask, scale_columns].to_numpy(float)
        a0 = float(np.mean(perf_error * direction0))
        avec = np.mean(perf_error[:, None] * components, axis=0)
        b0 = float(np.mean(direction0 ** 2))
        bvec = np.mean(direction0[:, None] * components, axis=0)
        gram = components.T @ components / len(components)
        linear = a0 + scales @ avec
        square = b0 + 2 * scales @ bvec + np.einsum(
            "ij,jk,ik->i", scales, gram, scales
        )
        weight = np.clip(-linear / np.maximum(square, 1e-15), 0.50, 0.65)
        brier = perf_brier + 2 * weight * linear + weight * weight * square
        output.loc[mask, "outer_insight_weight"] = weight
        output.loc[mask, "outer_brier_2024"] = brier
        output.loc[mask, "outer_bss_2024"] = np.maximum(
            0.0, 100000 * (1.0 - brier / fold["denominator"])
        )
    return output


def build_joint_features(main):
    pieces = []
    audits = []
    for cutoff in CUTOFFS:
        prepared = prepare_cutoff(
            main, cutoff, TOP_JOINT["matrix_lambda"], TOP_JOINT["svd_dim"]
        )
        feature, audit = pair_features(
            prepared, TOP_JOINT["pitcher_k"], TOP_JOINT["batter_k"]
        )
        pieces.append(feature)
        audits.append({"cutoff": cutoff, **audit})
    return pd.concat(pieces, ignore_index=True), audits


def load_current_feature_sets():
    folder = WORK / "oof" / "reverse_batter_seed"
    return {
        str(seed): pd.read_parquet(folder / f"seed_{seed}.parquet")
        for seed in SEEDS
    }


def main():
    base = load_base()
    main_frame = load_main()
    joint_feature, audits = build_joint_features(main_frame)
    joint_frame = base.merge(
        joint_feature, on=["row_id", "season"], validate="one_to_one"
    )
    current_frames = {
        seed: base.merge(feature, on=["row_id", "season"], validate="one_to_one")
        for seed, feature in load_current_feature_sets().items()
    }

    folds = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid = base["season"].eq(valid_year)
        y = base.loc[valid, "control_success"].to_numpy(float)
        prediction = base.loc[valid, "prediction"].to_numpy(float)
        folds[valid_year] = {
            "row_id": base.loc[valid, "row_id"].to_numpy(),
            "y": y,
            "base": prediction,
            "error": prediction - y,
            "denominator": float(y.mean() * (1.0 - y.mean())),
            "success": correction(
                base, SUCCESS_FEATURES, 10.0, train_year, valid_year
            ),
        }
        seed_corrections = [
            correction(
                frame, REVERSE_FEATURES, 1000.0, train_year, valid_year
            )
            for frame in current_frames.values()
        ]
        folds[valid_year]["current_single17"] = seed_corrections[0]
        folds[valid_year]["current_seedbag"] = np.mean(seed_corrections, axis=0)

    feature_sets = {
        "cluster": REVERSE_FEATURES,
        "direct": DIRECT_FEATURES,
        "cluster_direct": REVERSE_FEATURES + DIRECT_FEATURES,
    }
    rows = []
    best_vectors = {}
    for feature_name, features in feature_sets.items():
        for alpha in ALPHAS:
            for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
                folds[valid_year]["joint"] = correction(
                    joint_frame, features, alpha, train_year, valid_year
                )
            for current_name in ["current_single17", "current_seedbag"]:
                quadratic_terms = {
                    year: quadratic(
                        folds[year]["error"],
                        [
                            folds[year]["success"],
                            folds[year][current_name],
                            folds[year]["joint"],
                        ],
                    )
                    for year in [2023, 2024]
                }
                for scales in itertools.product(
                    SUCCESS_SCALES, CURRENT_SCALES, JOINT_SCALES
                ):
                    delta = {
                        year: delta_brier(*quadratic_terms[year], scales)
                        for year in [2023, 2024]
                    }
                    rows.append({
                        "joint_features": feature_name,
                        "joint_alpha": alpha,
                        "current_variant": current_name,
                        "success_scale": scales[0],
                        "current_scale": scales[1],
                        "joint_scale": scales[2],
                        "f23_delta_brier": delta[2023],
                        "f24_delta_brier": delta[2024],
                        "both_improve": delta[2023] < 0 and delta[2024] < 0,
                        "robust_objective": robust_objective(
                            delta[2023], delta[2024],
                            {year: folds[year]["denominator"] for year in [2023, 2024]},
                        ),
                    })
                best_vectors[(feature_name, alpha, current_name)] = {
                    year: folds[year]["joint"].copy() for year in [2023, 2024]
                }
    result = pd.DataFrame(rows).sort_values("robust_objective")
    result = attach_outer_scores(result, folds, best_vectors)
    reports = WORK / "reports"
    result.to_csv(reports / "joint_svd_residual_ensemble.csv", index=False)
    best = result.iloc[0].to_dict()

    key = (best["joint_features"], best["joint_alpha"], best["current_variant"])
    joint_2024 = best_vectors[key][2024]
    corrected = np.clip(
        folds[2024]["base"]
        + best["success_scale"] * folds[2024]["success"]
        + best["current_scale"] * folds[2024][best["current_variant"]]
        + best["joint_scale"] * joint_2024,
        1e-6, 1 - 1e-6,
    )
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].set_index("row_id")
    perf = performance.loc[folds[2024]["row_id"], "prediction"].to_numpy(float)
    blend = []
    for weight in np.round(np.arange(0.50, 0.651, 0.001), 3):
        prediction = weight * corrected + (1.0 - weight) * perf
        brier = float(np.mean((prediction - folds[2024]["y"]) ** 2))
        bss = max(
            0.0,
            100000.0 * (1.0 - brier / folds[2024]["denominator"]),
        )
        blend.append({"insight_weight": weight, "brier": brier, "bss": bss})
    best_blend = min(blend, key=lambda item: item["brier"])
    stable = result.loc[result["both_improve"]]
    best_outer_by_feature = (
        stable.sort_values("outer_brier_2024")
        .groupby("joint_features", as_index=False)
        .first()
        .sort_values("outer_brier_2024")
        .to_dict("records")
    )
    summary = {
        "joint_config": TOP_JOINT,
        "best": best,
        "blend_2024": best_blend,
        "best_outer_by_feature": best_outer_by_feature,
        "cutoff_audit": audits,
    }
    (reports / "joint_svd_residual_ensemble.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTOP 20")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
