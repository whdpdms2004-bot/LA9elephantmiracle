from __future__ import annotations

import argparse
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
    BATTER_K,
    DIRECT_FEATURES,
    MATRIX_LAMBDAS,
    PITCHER_K,
    SVD_DIMS,
    config_id,
    pair_features,
    prepare_cutoff,
)
from screen_reverse_batter_clusters import load_main  # noqa: E402


SEEDS = [17, 2026, 4099]
SUCCESS_SCALES = np.round(np.arange(0.25, 0.451, 0.025), 3)
CURRENT_SCALES = np.round(np.arange(0.65, 0.951, 0.025), 3)
JOINT_SCALES = np.round(np.arange(0.05, 0.301, 0.025), 3)
FEATURE_VARIANTS = {
    "cluster": REVERSE_FEATURES,
    "cluster_direct": REVERSE_FEATURES + DIRECT_FEATURES,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-modes", default="standard")
    parser.add_argument("--output-prefix", default="joint_svd_outer")
    return parser.parse_args()


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


def prepare_fixed_folds(base):
    current_frames = []
    for seed in SEEDS:
        feature = pd.read_parquet(
            WORK / "oof" / "reverse_batter_seed" / f"seed_{seed}.parquet"
        )
        current_frames.append(
            base.merge(feature, on=["row_id", "season"], validate="one_to_one")
        )
    folds = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid = base["season"].eq(valid_year)
        y = base.loc[valid, "control_success"].to_numpy(float)
        prediction = base.loc[valid, "prediction"].to_numpy(float)
        current = np.mean([
            correction(frame, REVERSE_FEATURES, 1000.0, train_year, valid_year)
            for frame in current_frames
        ], axis=0)
        folds[valid_year] = {
            "row_id": base.loc[valid, "row_id"].to_numpy(),
            "y": y, "base": prediction, "error": prediction - y,
            "denominator": float(y.mean() * (1.0 - y.mean())),
            "success": correction(base, SUCCESS_FEATURES, 10.0, train_year, valid_year),
            "current": current,
        }
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[
        performance["track"].eq("performance") & performance["season"].eq(2024),
        ["row_id", "prediction"],
    ].set_index("row_id")
    folds[2024]["performance"] = performance.loc[
        folds[2024]["row_id"], "prediction"
    ].to_numpy(float)
    return folds


def optimize_candidate(folds, joint, metadata):
    scales = np.asarray(list(itertools.product(
        SUCCESS_SCALES, CURRENT_SCALES, JOINT_SCALES
    )), dtype=float)
    deltas = {}
    for year in [2023, 2024]:
        components = np.column_stack([
            folds[year]["success"], folds[year]["current"], joint[year]
        ])
        linear = np.mean(folds[year]["error"][:, None] * components, axis=0)
        gram = components.T @ components / len(components)
        deltas[year] = (
            2 * scales @ linear
            + np.einsum("ij,jk,ik->i", scales, gram, scales)
        )
    normalized23 = deltas[2023] / folds[2023]["denominator"]
    normalized24 = deltas[2024] / folds[2024]["denominator"]
    robust = (
        0.30 * normalized23 + 0.70 * normalized24
        + 0.50 * np.maximum.reduce([normalized23, normalized24, np.zeros(len(scales))])
    )

    fold = folds[2024]
    perf = fold["performance"]
    perf_error = perf - fold["y"]
    direction0 = fold["base"] - perf
    components = np.column_stack([fold["success"], fold["current"], joint[2024]])
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
    perf_brier = float(np.mean(perf_error ** 2))
    outer_brier = perf_brier + 2 * weight * linear + weight * weight * square
    outer_bss = np.maximum(
        0.0, 100000 * (1.0 - outer_brier / fold["denominator"])
    )
    stable = (deltas[2023] < 0) & (deltas[2024] < 0)
    valid_indices = np.flatnonzero(stable)
    if len(valid_indices) == 0:
        return []
    choices = {
        "outer": valid_indices[np.argmin(outer_brier[valid_indices])],
        "robust": valid_indices[np.argmin(robust[valid_indices])],
    }
    rows = []
    for criterion, index in choices.items():
        rows.append({
            **metadata,
            "criterion": criterion,
            "success_scale": scales[index, 0],
            "current_scale": scales[index, 1],
            "joint_scale": scales[index, 2],
            "f23_delta_brier": deltas[2023][index],
            "f24_delta_brier": deltas[2024][index],
            "robust_objective": robust[index],
            "outer_insight_weight": weight[index],
            "outer_brier_2024": outer_brier[index],
            "outer_bss_2024": outer_bss[index],
        })
    return rows


def main():
    args = parse_args()
    cluster_modes = [value for value in args.cluster_modes.split(",") if value]
    main_frame = load_main()
    base = load_base()
    folds = prepare_fixed_folds(base)
    rows = []
    audits = []
    total = (
        len(MATRIX_LAMBDAS) * len(SVD_DIMS) * len(PITCHER_K)
        * len(BATTER_K) * len(cluster_modes)
    )
    completed = 0
    for matrix_lambda in MATRIX_LAMBDAS:
        for dim in SVD_DIMS:
            prepared = {
                cutoff: prepare_cutoff(main_frame, cutoff, matrix_lambda, dim)
                for cutoff in CUTOFFS
            }
            for pitcher_k in PITCHER_K:
                for batter_k in BATTER_K:
                    base_config = config_id(matrix_lambda, dim, pitcher_k, batter_k)
                    for cluster_mode in cluster_modes:
                        config = f"{base_config}_{cluster_mode}"
                        pieces = []
                        for cutoff in CUTOFFS:
                            feature, audit = pair_features(
                                prepared[cutoff], pitcher_k, batter_k, cluster_mode
                            )
                            pieces.append(feature)
                            audits.append({"config": config, "cutoff": cutoff, **audit})
                        frame = base.merge(
                            pd.concat(pieces, ignore_index=True),
                            on=["row_id", "season"], validate="one_to_one",
                        )
                        for variant, features in FEATURE_VARIANTS.items():
                            joint = {
                                valid_year: correction(
                                    frame, features, 100.0, train_year, valid_year
                                )
                                for train_year, valid_year in [(2022, 2023), (2023, 2024)]
                            }
                            metadata = {
                                "config": config, "cluster_mode": cluster_mode,
                                "feature_variant": variant,
                                "matrix_lambda": matrix_lambda, "svd_dim": dim,
                                "pitcher_k_left": pitcher_k[0],
                                "pitcher_k_right": pitcher_k[1],
                                "batter_k_left": batter_k[0],
                                "batter_k_right": batter_k[1],
                            }
                            rows.extend(optimize_candidate(folds, joint, metadata))
                        completed += 1
                        current = pd.DataFrame(rows)
                        best_bss = float(current["outer_bss_2024"].max())
                        print(json.dumps({
                            "completed": completed, "total": total,
                            "config": config, "running_best_outer_bss": best_bss,
                        }, ensure_ascii=False), flush=True)
    result = pd.DataFrame(rows)
    reports = WORK / "reports"
    result.to_csv(reports / f"{args.output_prefix}_search.csv", index=False)
    pd.DataFrame(audits).to_csv(reports / f"{args.output_prefix}_audit.csv", index=False)
    outer = result.loc[result["criterion"].eq("outer")].sort_values(
        "outer_brier_2024"
    )
    robust = result.loc[result["criterion"].eq("robust")].sort_values(
        "robust_objective"
    )
    summary = {
        "best_outer": outer.iloc[0].to_dict(),
        "best_robust": robust.iloc[0].to_dict(),
        "configs": total,
        "feature_variants": list(FEATURE_VARIANTS),
    }
    (reports / f"{args.output_prefix}_search.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTOP OUTER 20")
    print(outer.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
