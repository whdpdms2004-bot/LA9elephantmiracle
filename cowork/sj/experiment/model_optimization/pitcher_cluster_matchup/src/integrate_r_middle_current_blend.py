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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(MODEL_DIR))

from analyze_r_focus import CURRENT_INSIGHT_WEIGHT, load_fold_predictions  # noqa: E402
from screen_r_middle_preprocessing import (  # noqa: E402
    FEATURES,
    build_features,
    load_base,
    load_main,
)


CANDIDATES = [
    ("pv2_compact_mi_r5_2fec3702", (2, 3), 500.0),
    ("pv2_compact_mi_r5_2fec3702", (2, 3), 1000.0),
    ("pv2_compact_mi_r5_2fec3702", (2, 3), 2000.0),
    ("pv2_compact_mi_r5_2fec3702", (3, 4), 500.0),
    ("pv2_compact_mi_r5_2fec3702", (4, 6), 500.0),
    ("pv2_physical_mi_r5_e8dd683a", (2, 3), 500.0),
    ("pv2_all_r5_a6fc0d65", (2, 3), 500.0),
]
ALPHAS = [10.0, 100.0, 1000.0, 10000.0]
TARGET_MODES = ["adjusted_residual", "honest_blend_residual", "honest_single_residual"]
BLEND_WEIGHTS = np.unique(np.append(
    np.round(np.arange(0.56, 0.641, 0.002), 3), CURRENT_INSIGHT_WEIGHT
))
MIDDLE_SCALES = np.round(np.arange(0.0, 0.801, 0.025), 3)


def brier(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((prediction - y) ** 2))


def bss(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(y.mean() * (1.0 - y.mean()))
    return 100000.0 * (1.0 - brier(y, prediction) / denominator)


def attach_honest_references(frame: pd.DataFrame, folds: dict[int, pd.DataFrame]) -> pd.DataFrame:
    output = frame.copy()
    output["honest_blend"] = output["prediction"]
    output["honest_single"] = output["prediction"]
    for year in [2023, 2024]:
        mask = output["season"].eq(year)
        lookup = folds[year].set_index("row_id")
        ids = output.loc[mask, "row_id"]
        output.loc[mask, "honest_blend"] = lookup.loc[ids, "current_blend"].to_numpy(float)
        output.loc[mask, "honest_single"] = lookup.loc[ids, "corrected_single"].to_numpy(float)
    return output


def middle_correction(
    frame: pd.DataFrame,
    alpha: float,
    train_year: int,
    valid_year: int,
    target_mode: str,
) -> np.ndarray:
    train = frame["season"].eq(train_year) & frame["game_type"].eq("R")
    valid = frame["season"].eq(valid_year)
    valid_r = valid & frame["game_type"].eq("R")
    reference = {
        "adjusted_residual": "prediction",
        "honest_blend_residual": "honest_blend",
        "honest_single_residual": "honest_single",
    }[target_mode]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    target = frame.loc[train, "control_success"] - frame.loc[train, reference]
    model.fit(frame.loc[train, FEATURES], target)
    output = np.zeros(int(valid.sum()), dtype="float64")
    local_r = frame.loc[valid, "game_type"].eq("R").to_numpy()
    output[local_r] = np.clip(model.predict(frame.loc[valid_r, FEATURES]), -0.05, 0.05)
    return output


def quadratic_stats(error: np.ndarray, components: np.ndarray, mask: np.ndarray) -> dict:
    local_error = error[mask]
    local_components = components[mask]
    return {
        "linear": np.mean(local_components * local_error[:, None], axis=0),
        "quadratic": (local_components.T @ local_components) / len(local_components),
    }


def delta_brier(stats: dict, scale: np.ndarray) -> float:
    return float(
        2.0 * stats["linear"] @ scale
        + scale @ stats["quadratic"] @ scale
    )


def tune(
    frame: pd.DataFrame,
    corrections: dict[int, np.ndarray],
    folds: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series]:
    stats = {}
    denominator = {}
    for year in [2023, 2024]:
        part = frame.loc[frame["season"].eq(year)]
        fold = folds[year].set_index("row_id").loc[part["row_id"]]
        y = part["control_success"].to_numpy(float)
        baseline = fold["current_blend"].to_numpy(float)
        blend_direction = (
            fold["corrected_single"].to_numpy(float)
            - fold["ensemble"].to_numpy(float)
        )
        components = np.column_stack([blend_direction, corrections[year]])
        r_mask = part["game_type"].eq("R").to_numpy()
        stats[year] = {
            "ALL": quadratic_stats(baseline - y, components, np.ones(len(part), dtype=bool)),
            "R": quadratic_stats(baseline - y, components, r_mask),
            "F": quadratic_stats(baseline - y, components, ~r_mask),
        }
        denominator[year] = {
            group: float(y[mask].mean() * (1.0 - y[mask].mean()))
            for group, mask in {
                "ALL": np.ones(len(part), dtype=bool), "R": r_mask, "F": ~r_mask,
            }.items()
        }

    rows = []
    for weight in BLEND_WEIGHTS:
        for middle_scale in MIDDLE_SCALES:
            scale = np.asarray([weight - CURRENT_INSIGHT_WEIGHT, middle_scale])
            deltas = {
                (year, group): delta_brier(stats[year][group], scale)
                for year in [2023, 2024] for group in ["ALL", "R", "F"]
            }
            normalized = {
                key: value / denominator[key[0]][key[1]]
                for key, value in deltas.items()
            }
            n23, n24 = normalized[(2023, "ALL")], normalized[(2024, "ALL")]
            r23, r24 = normalized[(2023, "R")], normalized[(2024, "R")]
            objective = (
                0.30 * n23 + 0.70 * n24 + 0.50 * max(n23, n24, 0.0)
                + 0.25 * (0.30 * r23 + 0.70 * r24) + max(r23, r24, 0.0)
            )
            rows.append({
                "blend_weight": weight,
                "middle_scale": middle_scale,
                "val2023_delta_brier": deltas[(2023, "ALL")],
                "val2024_delta_brier": deltas[(2024, "ALL")],
                "val2023_r_delta_brier": deltas[(2023, "R")],
                "val2024_r_delta_brier": deltas[(2024, "R")],
                "val2023_f_delta_brier": deltas[(2023, "F")],
                "val2024_f_delta_brier": deltas[(2024, "F")],
                "both_improve": n23 < 0.0 and n24 < 0.0,
                "r_both_improve": r23 < 0.0 and r24 < 0.0,
                "objective": objective,
            })
    result = pd.DataFrame(rows)
    eligible = result.loc[result["both_improve"] & result["r_both_improve"]]
    selected = (eligible if len(eligible) else result).sort_values("objective").iloc[0]
    return result, selected


def exact_metrics(
    frame: pd.DataFrame,
    corrections: dict[int, np.ndarray],
    folds: dict[int, pd.DataFrame],
    selected: pd.Series,
) -> dict:
    metrics = {}
    for year in [2023, 2024]:
        part = frame.loc[frame["season"].eq(year)]
        fold = folds[year].set_index("row_id").loc[part["row_id"]]
        prediction = (
            float(selected["blend_weight"]) * fold["corrected_single"].to_numpy(float)
            + (1.0 - float(selected["blend_weight"])) * fold["ensemble"].to_numpy(float)
            + float(selected["middle_scale"]) * corrections[year]
        )
        prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
        y = part["control_success"].to_numpy(float)
        r_mask = part["game_type"].eq("R").to_numpy()
        for group, mask in {"all": np.ones(len(part), bool), "r": r_mask, "f": ~r_mask}.items():
            metrics[f"val{year}_{group}_brier"] = brier(y[mask], prediction[mask])
            metrics[f"val{year}_{group}_bss"] = bss(y[mask], prediction[mask])
    return metrics


def main() -> None:
    main_frame = load_main()
    folds = load_fold_predictions()
    base = load_base().merge(
        main_frame[["row_id", "game_type"]], on="row_id", validate="one_to_one"
    )
    base = attach_honest_references(base, folds)
    all_grid = []
    summaries = []
    for pitcher_config, batter_k, smoothing in CANDIDATES:
        features, audit = build_features(main_frame, pitcher_config, batter_k, smoothing)
        frame = base.merge(features, on=["row_id", "season"], validate="one_to_one")
        for target_mode in TARGET_MODES:
            for alpha in ALPHAS:
                corrections = {
                    year: middle_correction(
                        frame, alpha, train_year, year, target_mode
                    )
                    for train_year, year in [(2022, 2023), (2023, 2024)]
                }
                grid, selected = tune(frame, corrections, folds)
                metadata = {
                    "pitcher_config": pitcher_config,
                    "batter_k_left": batter_k[0],
                    "batter_k_right": batter_k[1],
                    "smoothing": smoothing,
                    "target_mode": target_mode,
                    "alpha": alpha,
                }
                all_grid.append(grid.assign(**metadata))
                summaries.append({
                    **metadata,
                    **selected.to_dict(),
                    **exact_metrics(frame, corrections, folds, selected),
                    "coverage_val2023_r": audit[1]["coverage_r"],
                    "coverage_val2024_r": audit[2]["coverage_r"],
                })
        print(json.dumps({
            "pitcher_config": pitcher_config,
            "batter_k": batter_k,
            "smoothing": smoothing,
            "completed": True,
        }), flush=True)

    summary = pd.DataFrame(summaries).sort_values(
        ["r_both_improve", "both_improve", "objective"],
        ascending=[False, False, True],
    )
    grid = pd.concat(all_grid, ignore_index=True).sort_values("objective")
    reports = WORK / "reports"
    grid.to_csv(reports / "r_middle_current_blend_grid.csv", index=False)
    summary.to_csv(reports / "r_middle_current_blend_summary.csv", index=False)
    best = summary.iloc[0]
    (reports / "r_middle_current_blend_best.json").write_text(
        json.dumps(best.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTOP 20")
    print(summary.head(20).to_string(index=False))
    print(json.dumps({
        "evaluated": len(summary),
        "both_improve": int(summary["both_improve"].sum()),
        "r_both_improve": int(summary["r_both_improve"].sum()),
        "best": best.to_dict(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
