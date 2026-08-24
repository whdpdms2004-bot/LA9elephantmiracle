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
    build_candidate_features,
    load_base,
    robust_objective,
)
from joint_svd_cluster_search import (  # noqa: E402
    DIRECT_FEATURES,
    pair_features,
    prepare_cutoff,
)
from screen_reverse_batter_clusters import load_main  # noqa: E402


MULTIVIEW_CONFIG = "mv_gmm_p16c8w1p0_l2r4_754558ba"
JOINT_CONFIG = {
    "matrix_lambda": 500.0,
    "svd_dim": 16,
    "pitcher_k": (2, 4),
    "batter_k": (6, 8),
}
SEEDS = [17, 2026, 4099]
SUCCESS_SCALES = np.round(np.arange(0.20, 0.451, 0.025), 3)
CURRENT_SCALES = np.round(np.arange(0.55, 0.951, 0.025), 3)
MULTIVIEW_SCALES = np.round(np.arange(0.00, 0.151, 0.025), 3)
JOINT_SCALES = np.round(np.arange(0.05, 0.301, 0.025), 3)


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


def build_joint(main):
    pieces = []
    for cutoff in CUTOFFS:
        prepared = prepare_cutoff(
            main, cutoff, JOINT_CONFIG["matrix_lambda"], JOINT_CONFIG["svd_dim"]
        )
        feature, _ = pair_features(
            prepared, JOINT_CONFIG["pitcher_k"], JOINT_CONFIG["batter_k"]
        )
        pieces.append(feature)
    return pd.concat(pieces, ignore_index=True)


def quadratic(error, components):
    matrix = np.column_stack(components)
    return (
        np.mean(error[:, None] * matrix, axis=0),
        matrix.T @ matrix / len(matrix),
    )


def delta_brier(terms, scales):
    linear, gram = terms
    scales = np.asarray(scales, dtype=float)
    return float(2 * linear @ scales + scales @ gram @ scales)


def outer_blend(fold, corrected):
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].set_index("row_id")
    perf = performance.loc[fold["row_id"], "prediction"].to_numpy(float)
    rows = []
    for weight in np.round(np.arange(0.50, 0.651, 0.0005), 4):
        prediction = weight * corrected + (1.0 - weight) * perf
        brier = float(np.mean((prediction - fold["y"]) ** 2))
        rows.append({
            "insight_weight": float(weight),
            "brier": brier,
            "bss": max(0.0, 100000.0 * (1.0 - brier / fold["denominator"])),
        })
    return min(rows, key=lambda item: item["brier"])


def attach_analytic_outer_scores(result, folds):
    """Optimize the final insight/performance weight for every grid row exactly."""
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].set_index("row_id")
    fold = folds[2024]
    perf = performance.loc[fold["row_id"], "prediction"].to_numpy(float)
    perf_error = perf - fold["y"]
    base_direction = fold["base"] - perf
    perf_brier = float(np.mean(perf_error ** 2))
    output = result.copy()
    scale_columns = [
        "success_scale", "current_scale", "multiview_scale", "joint_scale"
    ]
    for current_name in ["current_single17", "current_seedbag"]:
        for joint_name in ["joint_cluster", "joint_direct", "joint_cluster_direct"]:
            mask = (
                output["current_variant"].eq(current_name)
                & output["joint_variant"].eq(joint_name)
            )
            components = np.column_stack([
                fold["success"], fold[current_name], fold["multiview"], fold[joint_name]
            ])
            linear_error = np.mean(perf_error[:, None] * components, axis=0)
            base_error = float(np.mean(perf_error * base_direction))
            base_square = float(np.mean(base_direction ** 2))
            base_component = np.mean(base_direction[:, None] * components, axis=0)
            gram = components.T @ components / len(components)
            scales = output.loc[mask, scale_columns].to_numpy(float)
            linear = base_error + scales @ linear_error
            square = (
                base_square
                + 2.0 * (scales @ base_component)
                + np.einsum("ij,jk,ik->i", scales, gram, scales)
            )
            weight = np.clip(-linear / np.maximum(square, 1e-15), 0.50, 0.65)
            brier = perf_brier + 2.0 * weight * linear + weight * weight * square
            output.loc[mask, "outer_insight_weight"] = weight
            output.loc[mask, "outer_brier_2024"] = brier
            output.loc[mask, "outer_bss_2024"] = np.maximum(
                0.0, 100000.0 * (1.0 - brier / fold["denominator"])
            )
    return output


def exact_metrics(folds, scales, current_name, joint_name="joint_cluster_direct"):
    output = {}
    for year in [2023, 2024]:
        fold = folds[year]
        prediction = np.clip(
            fold["base"]
            + scales[0] * fold["success"]
            + scales[1] * fold[current_name]
            + scales[2] * fold["multiview"]
            + scales[3] * fold[joint_name],
            1e-6, 1 - 1e-6,
        )
        base_brier = float(np.mean((fold["base"] - fold["y"]) ** 2))
        brier = float(np.mean((prediction - fold["y"]) ** 2))
        output[year] = {
            "prediction": prediction,
            "brier": brier,
            "delta_brier": brier - base_brier,
            "bss": max(0.0, 100000.0 * (1.0 - brier / fold["denominator"])),
        }
    return output


def main():
    main_frame = load_main()
    base = load_base()
    multiview_feature, multiview_audit = build_candidate_features(
        main_frame, MULTIVIEW_CONFIG
    )
    multiview_frame = base.merge(
        multiview_feature, on=["row_id", "season"], validate="one_to_one"
    )
    joint_frame = base.merge(
        build_joint(main_frame), on=["row_id", "season"], validate="one_to_one"
    )
    current_frames = {}
    for seed in SEEDS:
        feature = pd.read_parquet(
            WORK / "oof" / "reverse_batter_seed" / f"seed_{seed}.parquet"
        )
        current_frames[seed] = base.merge(
            feature, on=["row_id", "season"], validate="one_to_one"
        )

    folds = {}
    correlations = []
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid = base["season"].eq(valid_year)
        y = base.loc[valid, "control_success"].to_numpy(float)
        prediction = base.loc[valid, "prediction"].to_numpy(float)
        seed_correction = [
            correction(frame, REVERSE_FEATURES, 1000.0, train_year, valid_year)
            for frame in current_frames.values()
        ]
        fold = {
            "row_id": base.loc[valid, "row_id"].to_numpy(),
            "y": y,
            "base": prediction,
            "error": prediction - y,
            "denominator": float(y.mean() * (1.0 - y.mean())),
            "success": correction(base, SUCCESS_FEATURES, 10.0, train_year, valid_year),
            "current_single17": seed_correction[0],
            "current_seedbag": np.mean(seed_correction, axis=0),
            "multiview": correction(
                multiview_frame, REVERSE_FEATURES, 100.0, train_year, valid_year
            ),
            "joint_cluster": correction(
                joint_frame, REVERSE_FEATURES,
                100.0, train_year, valid_year,
            ),
            "joint_direct": correction(
                joint_frame, DIRECT_FEATURES,
                100.0, train_year, valid_year,
            ),
            "joint_cluster_direct": correction(
                joint_frame, REVERSE_FEATURES + DIRECT_FEATURES,
                100.0, train_year, valid_year,
            ),
        }
        folds[valid_year] = fold
        names = [
            "current_single17", "current_seedbag", "multiview",
            "joint_cluster", "joint_direct", "joint_cluster_direct",
        ]
        matrix = np.column_stack([fold[name] for name in names])
        corr = np.corrcoef(matrix, rowvar=False)
        for i, left in enumerate(names):
            for j in range(i + 1, len(names)):
                correlations.append({
                    "season": valid_year, "left": left, "right": names[j],
                    "correlation": float(corr[i, j]),
                })

    rows = []
    for current_name in ["current_single17", "current_seedbag"]:
        for joint_name in ["joint_cluster", "joint_direct", "joint_cluster_direct"]:
            terms = {
                year: quadratic(
                    folds[year]["error"],
                    [
                        folds[year]["success"], folds[year][current_name],
                        folds[year]["multiview"], folds[year][joint_name],
                    ],
                )
                for year in [2023, 2024]
            }
            for scales in itertools.product(
                SUCCESS_SCALES, CURRENT_SCALES, MULTIVIEW_SCALES, JOINT_SCALES
            ):
                delta = {
                    year: delta_brier(terms[year], scales) for year in [2023, 2024]
                }
                rows.append({
                    "current_variant": current_name,
                    "joint_variant": joint_name,
                    "success_scale": scales[0], "current_scale": scales[1],
                    "multiview_scale": scales[2], "joint_scale": scales[3],
                    "f23_delta_brier": delta[2023],
                    "f24_delta_brier": delta[2024],
                    "both_improve": delta[2023] < 0 and delta[2024] < 0,
                    "robust_objective": robust_objective(
                        delta[2023], delta[2024],
                        {year: folds[year]["denominator"] for year in [2023, 2024]},
                    ),
                })
    result = pd.DataFrame(rows).sort_values("robust_objective")
    result = attach_analytic_outer_scores(result, folds)
    best = result.iloc[0].to_dict()
    scales = [
        best["success_scale"], best["current_scale"],
        best["multiview_scale"], best["joint_scale"],
    ]
    exact = exact_metrics(
        folds, scales, best["current_variant"], best["joint_variant"]
    )
    blend = outer_blend(folds[2024], exact[2024]["prediction"])

    baseline_scales = [0.25, 0.55, 0.0, 0.0]
    baseline_exact = exact_metrics(
        folds, baseline_scales, "current_seedbag", "joint_cluster_direct"
    )
    baseline_blend = outer_blend(folds[2024], baseline_exact[2024]["prediction"])
    stable = result.loc[result["both_improve"]].copy()
    best_outer = stable.sort_values("outer_brier_2024").iloc[0].to_dict()
    outer_scales = [
        best_outer["success_scale"], best_outer["current_scale"],
        best_outer["multiview_scale"], best_outer["joint_scale"],
    ]
    outer_exact = exact_metrics(
        folds, outer_scales, best_outer["current_variant"], best_outer["joint_variant"]
    )
    best_outer["exact_f23_delta_brier"] = outer_exact[2023]["delta_brier"]
    best_outer["exact_f24_delta_brier"] = outer_exact[2024]["delta_brier"]
    summary = {
        "multiview_config": MULTIVIEW_CONFIG,
        "joint_config": JOINT_CONFIG,
        "best": best,
        "exact_f23": {k: v for k, v in exact[2023].items() if k != "prediction"},
        "exact_f24": {k: v for k, v in exact[2024].items() if k != "prediction"},
        "blend_2024": blend,
        "best_outer_with_f23_f24_improvement": best_outer,
        "reconstructed_submit013": {
            "scales": baseline_scales,
            "f23_delta_brier": baseline_exact[2023]["delta_brier"],
            "f24_delta_brier": baseline_exact[2024]["delta_brier"],
            "blend_2024": baseline_blend,
        },
        "correlations": correlations,
        "multiview_audit": multiview_audit,
    }
    reports = WORK / "reports"
    result.to_csv(reports / "deep_embedding_ensemble_grid.csv", index=False)
    pd.DataFrame(correlations).to_csv(
        reports / "deep_embedding_correlation.csv", index=False
    )
    (reports / "deep_embedding_ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTOP 20")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
