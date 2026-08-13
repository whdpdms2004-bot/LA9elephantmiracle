"""Strict temporal evaluation of a 20-seed reverse batter cluster bag."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MODEL_OPT = ROOT / "experiment" / "model_optimization"
WORK = MODEL_OPT / "pitcher_cluster_matchup"
CACHE = WORK / "oof" / "reverse_batter_seed20"
REPORTS = WORK / "reports"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(MODEL_OPT))

from screen_reverse_batter_clusters import (  # noqa: E402
    FEATURES,
    add_context_residual,
    add_pitcher_type,
    build_batter_profile,
    cluster_batters,
    load_main,
)
from screen_reverse_batter_seeds import (  # noqa: E402
    SUCCESS_FEATURES,
    correction,
    load_base,
)


SEEDS = [
    17, 2026, 4099, 43, 97, 311, 503, 719, 887, 1237,
    1429, 1699, 1877, 2131, 2389, 2683, 3001, 3253, 3529, 3851,
]
CUTOFFS = [2022, 2023, 2024]
BATTER_K = (4, 6)
SMOOTHING = 1000.0
HALF_LIFE = 1.0
RIDGE_ALPHA = 1000.0
SUCCESS_SCALE = 0.25
OUTER_WEIGHT = 0.6085


def build_one_seed(main: pd.DataFrame, seed: int) -> dict:
    destination = CACHE / f"seed_{seed}.parquet"
    if destination.exists():
        frame = pd.read_parquet(destination, columns=["season", "reverse_pair_known"])
        return {
            "seed": seed,
            "cached": True,
            "rows": int(len(frame)),
            "coverage": float(frame["reverse_pair_known"].mean()),
        }
    pieces = []
    audits = []
    for cutoff in CUTOFFS:
        typed = add_pitcher_type(main.loc[main["season"].le(cutoff)].copy(), cutoff)
        past = add_context_residual(typed.loc[typed["season"].lt(cutoff)])
        profile = build_batter_profile(past)
        lookup, cluster_audit = cluster_batters(profile, "kmeans", BATTER_K, seed=seed)
        typed_past = past.merge(
            lookup[["batter_id", "batter_hand", "batter_type"]],
            on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
        )
        typed_past["batter_type"] = typed_past["batter_type"].fillna(
            "RBH" + typed_past["batter_hand"].astype(str) + "_new"
        )
        current = typed.loc[typed["season"].eq(cutoff), [
            "row_id", "season", "pitcher_type", "batter_id", "batter_hand"
        ]].copy()
        current = current.merge(
            lookup[["batter_id", "batter_hand", "batter_type"]],
            on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
        )
        current["batter_type"] = current["batter_type"].fillna(
            "RBH" + current["batter_hand"].astype(str) + "_new"
        )
        weight = np.power(
            0.5, (cutoff - typed_past["season"].to_numpy(float)) / HALF_LIFE
        )
        grouped = typed_past[["pitcher_type", "batter_type"]].copy()
        grouped["weighted_residual"] = typed_past["reverse_residual"].to_numpy(float) * weight
        grouped["weighted_reverse"] = typed_past["reverse"].to_numpy(float) * weight
        grouped["weight"] = weight
        pair = grouped.groupby(["pitcher_type", "batter_type"], sort=False).agg(
            weighted_residual=("weighted_residual", "sum"),
            weighted_reverse=("weighted_reverse", "sum"),
            effective_n=("weight", "sum"),
        ).reset_index()
        pair["reverse_pair_delta"] = pair["weighted_residual"] / (
            pair["effective_n"] + SMOOTHING
        )
        pair["reverse_pair_delta_reliability"] = pair["effective_n"] / (
            pair["effective_n"] + SMOOTHING
        )
        pair["reverse_pair_rate"] = pair["weighted_reverse"] / pair["effective_n"]
        out = current.merge(
            pair[["pitcher_type", "batter_type", *FEATURES[:-1]]],
            on=["pitcher_type", "batter_type"], how="left", validate="many_to_one",
        )
        out["reverse_pair_known"] = out["reverse_pair_delta"].notna().astype("float32")
        # Match the validated 3-seed pipeline exactly: unknown deltas and
        # reliabilities are zero, while an unknown raw rate remains NaN and is
        # handled by the downstream fold-fitted median imputer.
        out["reverse_pair_delta"] = out["reverse_pair_delta"].fillna(0.0)
        out["reverse_pair_delta_reliability"] = out[
            "reverse_pair_delta_reliability"
        ].fillna(0.0)
        for column in FEATURES[:-1]:
            out[column] = out[column].astype("float32")
        pieces.append(out[["row_id", "season", *FEATURES]])
        audits.append(
            {
                "cutoff": cutoff,
                "coverage": float(out["reverse_pair_known"].mean()),
                "pair_cells": int(len(pair)),
                "cluster_audit": cluster_audit,
            }
        )
    output = pd.concat(pieces, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    return {
        "seed": seed,
        "cached": False,
        "rows": int(len(output)),
        "coverage": float(output["reverse_pair_known"].mean()),
        "audits": audits,
    }


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(p - y)))


def bss(y: np.ndarray, p: np.ndarray) -> float:
    null = float(y.mean() * (1 - y.mean()))
    return 100000.0 * (1 - brier(y, p) / null)


def robust_objective(metrics: dict[int, dict]) -> float:
    n23 = metrics[2023]["brier"] / metrics[2023]["null"]
    n24 = metrics[2024]["brier"] / metrics[2024]["null"]
    return 0.30 * n23 + 0.70 * n24 + 0.50 * max(n23 - 1, n24 - 1, 0.0)


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = load_base()
    ensemble = pd.read_parquet(MODEL_OPT / "enhanced_ensemble_oof_predictions.parquet")
    seed_corrections: dict[tuple[int, int], np.ndarray] = {}
    success: dict[int, np.ndarray] = {}
    fold_data = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        mask = base["season"].eq(valid_year)
        current = base.loc[mask, ["row_id", TARGET, "prediction"]].copy()
        track = "robust" if valid_year == 2023 else "performance"
        other = ensemble.loc[
            ensemble["season"].eq(valid_year) & ensemble["track"].eq(track),
            ["row_id", "prediction"],
        ].rename(columns={"prediction": "ensemble_prediction"})
        current = current.merge(other, on="row_id", validate="one_to_one")
        y = current[TARGET].to_numpy(float)
        success[valid_year] = correction(
            base, SUCCESS_FEATURES, 10.0, train_year, valid_year
        )
        fold_data[valid_year] = {
            "row_id": current["row_id"].to_numpy(),
            "y": y,
            "adjusted": current["prediction"].to_numpy(float),
            "ensemble": current["ensemble_prediction"].to_numpy(float),
            "null": float(y.mean() * (1 - y.mean())),
        }
        for seed in SEEDS:
            features = base.merge(
                pd.read_parquet(CACHE / f"seed_{seed}.parquet"),
                on=["row_id", "season"], validate="one_to_one",
            )
            seed_corrections[(valid_year, seed)] = correction(
                features, FEATURES, RIDGE_ALPHA, train_year, valid_year
            )

    rows = []
    prediction_records = []
    for bag_size in range(1, len(SEEDS) + 1):
        bag = SEEDS[:bag_size]
        reverse = {
            year: np.mean([seed_corrections[(year, seed)] for seed in bag], axis=0)
            for year in [2023, 2024]
        }
        for scale in np.round(np.arange(0.0, 1.001, 0.025), 3):
            metrics = {}
            predictions = {}
            for year in [2023, 2024]:
                data = fold_data[year]
                corrected = np.clip(
                    data["adjusted"]
                    + SUCCESS_SCALE * success[year]
                    + scale * reverse[year],
                    1e-6,
                    1 - 1e-6,
                )
                prediction = (
                    OUTER_WEIGHT * corrected + (1 - OUTER_WEIGHT) * data["ensemble"]
                )
                predictions[year] = prediction
                metrics[year] = {
                    "brier": brier(data["y"], prediction),
                    "bss": bss(data["y"], prediction),
                    "null": data["null"],
                }
            rows.append(
                {
                    "bag_size": bag_size,
                    "seeds": ",".join(map(str, bag)),
                    "reverse_scale": float(scale),
                    "val2023_brier": metrics[2023]["brier"],
                    "val2023_bss": metrics[2023]["bss"],
                    "val2024_brier": metrics[2024]["brier"],
                    "val2024_bss": metrics[2024]["bss"],
                    "both_positive": metrics[2023]["bss"] > 0 and metrics[2024]["bss"] > 0,
                    "robust_objective": robust_objective(metrics),
                }
            )
        # Persist the existing 0.55 scale for exact downstream comparison.
        for year in [2023, 2024]:
            data = fold_data[year]
            corrected = np.clip(
                data["adjusted"] + SUCCESS_SCALE * success[year] + 0.55 * reverse[year],
                1e-6, 1 - 1e-6,
            )
            prediction = OUTER_WEIGHT * corrected + (1 - OUTER_WEIGHT) * data["ensemble"]
            prediction_records.append(
                pd.DataFrame(
                    {
                        "row_id": data["row_id"],
                        "season": year,
                        "bag_size": bag_size,
                        TARGET: data["y"].astype("int8"),
                        "prediction_scale055": prediction.astype("float32"),
                    }
                )
            )
    grid = pd.DataFrame(rows)
    best = (
        grid.sort_values(["robust_objective", "val2024_brier"])
        .groupby("bag_size", as_index=False)
        .first()
    )
    full = best.loc[best["bag_size"].eq(20)].iloc[0].to_dict()
    predictions = pd.concat(prediction_records, ignore_index=True)
    return grid, best, {"full_bag": full, "predictions": predictions}


TARGET = "control_success"


def main() -> None:
    started = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    main_frame = load_main()
    audits = []
    for index, seed in enumerate(SEEDS):
        audit = build_one_seed(main_frame, seed)
        audits.append(audit)
        print(f"reverse seed {index + 1:02d}/20: {seed}", flush=True)
    grid, best, payload = evaluate()
    predictions = payload.pop("predictions")
    grid.to_csv(REPORTS / "reverse_seedbag20_grid.csv", index=False)
    best.to_csv(REPORTS / "reverse_seedbag20_best.csv", index=False)
    predictions.to_parquet(REPORTS / "reverse_seedbag20_predictions.parquet", index=False)
    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "seeds": SEEDS,
        "strict_cutoffs": CUTOFFS,
        "batter_k_by_hand": {"left": BATTER_K[0], "right": BATTER_K[1]},
        "smoothing": SMOOTHING,
        "half_life": HALF_LIFE,
        "ridge_alpha": RIDGE_ALPHA,
        "success_scale": SUCCESS_SCALE,
        "outer_weight": OUTER_WEIGHT,
        "audit": audits,
        **payload,
        "elapsed_seconds": time.time() - started,
    }
    (REPORTS / "reverse_seedbag20_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["full_bag"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
