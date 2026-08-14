from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
SRC = WORK / "src"
sys.path.insert(0, str(SRC))

from screen_reverse_batter_seeds import (  # noqa: E402
    FEATURES,
    SUCCESS_FEATURES,
    correction,
    load_base,
)


SEEDS = [17, 2026, 4099]
CURRENT_SUCCESS_SCALE = 0.25
CURRENT_REVERSE_SCALE = 0.55
CURRENT_INSIGHT_WEIGHT = 0.6085
SUBMIT007_INSIGHT_WEIGHT = 0.5284093304636978


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def bss(y: np.ndarray, p: np.ndarray) -> float:
    denominator = float(y.mean() * (1.0 - y.mean()))
    return 100000.0 * (1.0 - brier(y, p) / denominator)


def load_fold_predictions() -> dict[int, pd.DataFrame]:
    base = load_base()
    seed_frames = {}
    cache_dir = WORK / "oof" / "reverse_batter_seed"
    for seed in SEEDS:
        seed_frames[seed] = base.merge(
            pd.read_parquet(cache_dir / f"seed_{seed}.parquet"),
            on=["row_id", "season"],
            validate="one_to_one",
        )

    game = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=[
            "row_id", "season", "game_type", "game_month", "inning", "top_bottom",
            "balls_before", "strikes_before", "outs_before", "num_runners_on",
            "base_state", "pitcher_hand", "batter_hand",
        ],
    )
    game["count_state"] = (
        game["balls_before"].astype(str) + "-" + game["strikes_before"].astype(str)
    )
    game["hand_matchup"] = (
        game["pitcher_hand"].astype(str) + "-" + game["batter_hand"].astype(str)
    )
    game["inning_bucket"] = pd.cut(
        game["inning"], bins=[0, 3, 6, 9, np.inf],
        labels=["1-3", "4-6", "7-9", "10+"], right=True,
    ).astype(str)
    ensemble = pd.read_parquet(
        MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet"
    )

    output = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid = base["season"].eq(valid_year)
        fold = base.loc[
            valid, ["row_id", "season", "control_success", "prediction"]
        ].copy()
        fold = fold.rename(columns={"prediction": "adjusted_base"})
        fold["success_correction"] = correction(
            base, SUCCESS_FEATURES, 10.0, train_year, valid_year
        )

        reverse = []
        for seed in SEEDS:
            reverse.append(
                correction(
                    seed_frames[seed], FEATURES, 1000.0, train_year, valid_year
                )
            )
        fold["reverse_correction"] = np.mean(reverse, axis=0)
        track = "performance" if valid_year == 2024 else "robust"
        fold_ensemble = ensemble.loc[
            ensemble["track"].eq(track) & ensemble["season"].eq(valid_year),
            ["row_id", "season", "prediction"],
        ].rename(columns={"prediction": "ensemble"})
        fold = fold.merge(fold_ensemble, on=["row_id", "season"], validate="one_to_one")
        fold = fold.merge(game, on=["row_id", "season"], validate="one_to_one")

        fold["corrected_single"] = np.clip(
            fold["adjusted_base"]
            + CURRENT_SUCCESS_SCALE * fold["success_correction"]
            + CURRENT_REVERSE_SCALE * fold["reverse_correction"],
            1e-6,
            1.0 - 1e-6,
        )
        fold["current_blend"] = (
            CURRENT_INSIGHT_WEIGHT * fold["corrected_single"]
            + (1.0 - CURRENT_INSIGHT_WEIGHT) * fold["ensemble"]
        )
        fold["submit007"] = (
            SUBMIT007_INSIGHT_WEIGHT * fold["adjusted_base"]
            + (1.0 - SUBMIT007_INSIGHT_WEIGHT) * fold["ensemble"]
        )
        output[valid_year] = fold
    return output


def distribution_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=["row_id", "season", "game_type", "control_success"],
    )
    labels = pd.read_parquet(
        MODEL_DIR / "failure_component_labels.parquet",
        columns=["row_id", "reverse", "middle", "outside_only"],
    )
    frame = main.merge(labels, on="row_id", validate="one_to_one")
    counts = (
        frame.groupby(["season", "game_type"], observed=True)
        .agg(n=("row_id", "size"), success_rate=("control_success", "mean"))
        .reset_index()
    )
    counts["season_share"] = counts["n"] / counts.groupby("season")["n"].transform("sum")
    failure = (
        frame.groupby(["season", "game_type"], observed=True)
        .agg(
            valid_failure_labels=("reverse", "count"),
            reverse_rate=("reverse", "mean"),
            middle_rate=("middle", "mean"),
            outside_only_rate=("outside_only", "mean"),
        )
        .reset_index()
    )
    return counts, failure


def metric_table(folds: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    model_columns = [
        "adjusted_base", "ensemble", "submit007", "corrected_single", "current_blend"
    ]
    for year, fold in folds.items():
        for group in ["ALL", "R", "F"]:
            part = fold if group == "ALL" else fold.loc[fold["game_type"].eq(group)]
            y = part["control_success"].to_numpy(float)
            for model in model_columns:
                p = part[model].to_numpy(float)
                rows.append(
                    {
                        "season": year,
                        "game_type": group,
                        "model": model,
                        "n": len(part),
                        "share": len(part) / len(fold),
                        "target_mean": float(y.mean()),
                        "pred_mean": float(p.mean()),
                        "mean_gap": float(p.mean() - y.mean()),
                        "brier": brier(y, p),
                        "local_bss": bss(y, p),
                    }
                )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq("adjusted_base"), [
        "season", "game_type", "brier"
    ]].rename(columns={"brier": "adjusted_base_brier"})
    result = result.merge(baseline, on=["season", "game_type"], validate="many_to_one")
    result["delta_brier_vs_adjusted_base"] = result["brier"] - result["adjusted_base_brier"]
    return result.drop(columns="adjusted_base_brier")


def sufficient_statistics(frame: pd.DataFrame, game_type: str) -> dict[str, np.ndarray | float]:
    part = frame.loc[frame["game_type"].eq(game_type)]
    y = part["control_success"].to_numpy(float)
    x = np.column_stack(
        [
            part["success_correction"].to_numpy(float),
            part["reverse_correction"].to_numpy(float),
        ]
    )
    error = part["adjusted_base"].to_numpy(float) - y
    return {
        "base_brier": float(np.mean(error**2)),
        "linear": np.mean(x * error[:, None], axis=0),
        "quadratic": (x.T @ x) / len(x),
        "denominator": float(y.mean() * (1.0 - y.mean())),
    }


def quadratic_brier(stats: dict[str, np.ndarray | float], coef: np.ndarray) -> float:
    linear = np.asarray(stats["linear"])
    quadratic = np.asarray(stats["quadratic"])
    return float(stats["base_brier"] + 2.0 * linear @ coef + coef @ quadratic @ coef)


def tune_r(folds: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, float]]:
    stats = {year: sufficient_statistics(frame, "R") for year, frame in folds.items()}
    rows = []
    for success_scale in np.round(np.arange(0.0, 0.801, 0.05), 2):
        for reverse_scale in np.round(np.arange(0.0, 1.201, 0.05), 2):
                coef = np.array([success_scale, reverse_scale], dtype=float)
                b23 = quadratic_brier(stats[2023], coef)
                b24 = quadratic_brier(stats[2024], coef)
                n23 = b23 / float(stats[2023]["denominator"])
                n24 = b24 / float(stats[2024]["denominator"])
                objective = 0.30 * n23 + 0.70 * n24 + 0.50 * max(
                    n23 - 1.0, n24 - 1.0, 0.0
                )
                rows.append(
                    {
                        "success_scale": success_scale,
                        "reverse_scale": reverse_scale,
                        "r_brier_2023": b23,
                        "r_brier_2024": b24,
                        "robust_objective": objective,
                    }
                )
    result = pd.DataFrame(rows).sort_values("robust_objective").reset_index(drop=True)
    best = result.iloc[0].to_dict()
    fold = folds[2024]
    r = fold["game_type"].eq("R")
    y = fold.loc[r, "control_success"].to_numpy(float)
    corrected = np.clip(
        fold.loc[r, "adjusted_base"].to_numpy(float)
        + best["success_scale"] * fold.loc[r, "success_correction"].to_numpy(float)
        + best["reverse_scale"] * fold.loc[r, "reverse_correction"].to_numpy(float),
        1e-6,
        1.0 - 1e-6,
    )
    ensemble = fold.loc[r, "ensemble"].to_numpy(float)
    weights = np.round(np.arange(0.45, 0.751, 0.001), 3)
    outer = [(brier(y, w * corrected + (1.0 - w) * ensemble), w) for w in weights]
    best["insight_weight"] = float(min(outer)[1])
    best["r_outer_brier_2024"] = float(min(outer)[0])
    return result, best


def add_r_candidate(
    folds: dict[int, pd.DataFrame], best: dict[str, float]
) -> pd.DataFrame:
    rows = []
    for year, fold in folds.items():
        if year == 2024:
            current = fold["current_blend"].to_numpy(float)
        else:
            current = fold["corrected_single"].to_numpy(float)
        candidate = current.copy()
        r_mask = fold["game_type"].eq("R").to_numpy()
        corrected = np.clip(
            fold.loc[r_mask, "adjusted_base"].to_numpy(float)
            + best["success_scale"]
            * fold.loc[r_mask, "success_correction"].to_numpy(float)
            + best["reverse_scale"]
            * fold.loc[r_mask, "reverse_correction"].to_numpy(float),
            1e-6,
            1.0 - 1e-6,
        )
        if year == 2024:
            r_pred = (
                best["insight_weight"] * corrected
                + (1.0 - best["insight_weight"])
                * fold.loc[r_mask, "ensemble"].to_numpy(float)
            )
        else:
            r_pred = corrected
        candidate[r_mask] = np.clip(r_pred, 1e-6, 1.0 - 1e-6)
        y = fold["control_success"].to_numpy(float)
        rows.append(
            {
                "season": year,
                "n": len(fold),
                "r_share": float(r_mask.mean()),
                "current_brier": brier(y, current),
                "candidate_brier": brier(y, candidate),
                "delta_brier": brier(y, candidate) - brier(y, current),
                "current_bss": bss(y, current),
                "candidate_bss": bss(y, candidate),
            }
        )
    return pd.DataFrame(rows)


def incremental_decomposition(fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_all = fold["control_success"].to_numpy(float)
    total_delta = brier(y_all, fold["current_blend"].to_numpy(float)) - brier(
        y_all, fold["submit007"].to_numpy(float)
    )
    for group in ["R", "F"]:
        part = fold.loc[fold["game_type"].eq(group)]
        y = part["control_success"].to_numpy(float)
        old_brier = brier(y, part["submit007"].to_numpy(float))
        new_brier = brier(y, part["current_blend"].to_numpy(float))
        delta = new_brier - old_brier
        share = len(part) / len(fold)
        weighted_delta = share * delta
        rows.append(
            {
                "game_type": group,
                "n": len(part),
                "share": share,
                "submit007_brier": old_brier,
                "submit013_brier": new_brier,
                "group_delta_brier": delta,
                "weighted_delta_brier": weighted_delta,
                "share_of_total_gain": weighted_delta / total_delta,
            }
        )
    return pd.DataFrame(rows)


def tune_r_f24_only(fold: pd.DataFrame) -> dict[str, float]:
    part = fold.loc[fold["game_type"].eq("R")]
    y = part["control_success"].to_numpy(float)
    ensemble = part["ensemble"].to_numpy(float)
    x = np.column_stack(
        [
            part["adjusted_base"].to_numpy(float) - ensemble,
            part["success_correction"].to_numpy(float),
            part["reverse_correction"].to_numpy(float),
        ]
    )
    error = ensemble - y
    stats = {
        "base_brier": float(np.mean(error**2)),
        "linear": np.mean(x * error[:, None], axis=0),
        "quadratic": (x.T @ x) / len(x),
    }
    best = None
    for weight in np.round(np.arange(0.45, 0.751, 0.005), 3):
        for success_scale in np.round(np.arange(0.0, 0.801, 0.05), 2):
            for reverse_scale in np.round(np.arange(0.0, 1.201, 0.05), 2):
                coef = np.array(
                    [weight, weight * success_scale, weight * reverse_scale]
                )
                score = quadratic_brier(stats, coef)
                row = {
                    "insight_weight": weight,
                    "success_scale": success_scale,
                    "reverse_scale": reverse_scale,
                    "r_brier_2024": score,
                    "r_local_bss_2024": 100000.0
                    * (1.0 - score / (y.mean() * (1.0 - y.mean()))),
                }
                if best is None or score < best["r_brier_2024"]:
                    best = row
    return best


def residual_group_audit(fold: pd.DataFrame) -> pd.DataFrame:
    frame = fold.loc[fold["game_type"].eq("R")].copy()
    dimensions = [
        "count_state", "hand_matchup", "inning_bucket", "game_month",
        "outs_before", "num_runners_on", "base_state", "top_bottom",
    ]
    rows = []
    for dimension in dimensions:
        for level, part in frame.groupby(dimension, observed=True, dropna=False):
            if len(part) < 2000:
                continue
            y = part["control_success"].to_numpy(float)
            p7 = part["submit007"].to_numpy(float)
            p13 = part["current_blend"].to_numpy(float)
            rows.append(
                {
                    "dimension": dimension,
                    "level": str(level),
                    "n": len(part),
                    "target_mean": float(y.mean()),
                    "submit013_pred_mean": float(p13.mean()),
                    "submit013_mean_gap": float(p13.mean() - y.mean()),
                    "submit013_brier": brier(y, p13),
                    "delta_brier_013_vs_007": brier(y, p13) - brier(y, p7),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    counts, failures = distribution_tables()
    folds = load_fold_predictions()
    metrics = metric_table(folds)
    tuning, best = tune_r(folds)
    candidate = add_r_candidate(folds, best)
    decomposition = incremental_decomposition(folds[2024])
    f24_oracle = tune_r_f24_only(folds[2024])
    residuals = residual_group_audit(folds[2024])

    counts.to_csv(MODEL_DIR / "r_focus_season_distribution.csv", index=False)
    failures.to_csv(MODEL_DIR / "r_focus_failure_distribution.csv", index=False)
    metrics.to_csv(MODEL_DIR / "r_focus_model_metrics.csv", index=False)
    tuning.head(200).to_csv(MODEL_DIR / "r_focus_tuning_top200.csv", index=False)
    candidate.to_csv(MODEL_DIR / "r_focus_candidate_metrics.csv", index=False)
    decomposition.to_csv(MODEL_DIR / "r_focus_incremental_decomposition.csv", index=False)
    residuals.to_csv(MODEL_DIR / "r_focus_residual_groups.csv", index=False)

    print("\nSEASON DISTRIBUTION")
    print(counts.to_string(index=False))
    print("\nFAILURE DISTRIBUTION")
    print(failures.to_string(index=False))
    print("\nMODEL METRICS")
    print(metrics.to_string(index=False))
    print("\nBEST R TUNING")
    print(best)
    print("\nR-ONLY CANDIDATE")
    print(candidate.to_string(index=False))
    print("\nSUBMIT007 -> SUBMIT013 DECOMPOSITION")
    print(decomposition.to_string(index=False))
    print("\nF24-ONLY R ORACLE GRID")
    print(f24_oracle)
    print("\nR RESIDUAL GROUPS: LARGEST ABSOLUTE MEAN GAP")
    print(
        residuals.assign(abs_gap=residuals["submit013_mean_gap"].abs())
        .sort_values(["abs_gap", "n"], ascending=[False, False])
        .head(20).drop(columns="abs_gap").to_string(index=False)
    )


if __name__ == "__main__":
    main()
