from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(MODEL_DIR))

from analyze_r_focus import load_fold_predictions  # noqa: E402
from integrate_r_middle_current_blend import (  # noqa: E402
    attach_honest_references,
    brier,
    bss,
    middle_correction,
)
from screen_r_middle_preprocessing import build_features, load_base, load_main  # noqa: E402


MIDDLE_CONFIG = "pv2_compact_mi_r5_2fec3702"
MIDDLE_BATTER_K = (2, 3)
MIDDLE_SMOOTHING = 500.0
MIDDLE_ALPHA = 10.0
MIDDLE_TARGET_MODE = "adjusted_residual"

CONTEXTS = {
    "count": ["balls_before", "strikes_before"],
    "count_hands": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
    "count_outs_runners": [
        "balls_before", "strikes_before", "outs_before", "num_runners_on"
    ],
    "count_base": ["balls_before", "strikes_before", "base_state"],
    "count_inning": ["balls_before", "strikes_before", "inning_bucket"],
    "count_month": ["balls_before", "strikes_before", "game_month"],
}
SMOOTHINGS = [100.0, 300.0, 1000.0, 3000.0]
REFERENCE_MODES = ["adjusted", "honest"]
CENTER_MODES = [False, True]
MIDDLE_SCALES = np.round(np.arange(0.0, 0.251, 0.025), 3)
CONTEXT_SCALES = np.round(np.arange(-0.50, 1.001, 0.05), 3)


def load_context() -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=[
            "row_id", "season", "game_type", "game_month", "inning",
            "balls_before", "strikes_before", "outs_before", "num_runners_on",
            "base_state", "pitcher_hand", "batter_hand",
        ],
    )
    frame["inning_bucket"] = pd.cut(
        frame["inning"], bins=[0, 3, 6, 9, np.inf],
        labels=["1-3", "4-6", "7-9", "10+"], right=True,
    ).astype(str)
    return frame


def residual_lookup(
    frame: pd.DataFrame,
    keys: list[str],
    train_year: int,
    valid_year: int,
    smoothing: float,
    reference_mode: str,
    centered: bool,
) -> tuple[np.ndarray, dict]:
    train = frame.loc[
        frame["season"].eq(train_year) & frame["game_type"].eq("R")
    ].copy()
    valid = frame.loc[frame["season"].eq(valid_year)].copy()
    reference = "prediction" if reference_mode == "adjusted" else "honest_blend"
    train["residual"] = train["control_success"] - train[reference]
    center = float(train["residual"].mean()) if centered else 0.0
    train["local_residual"] = train["residual"] - center
    grouped = train.groupby(keys, observed=True, dropna=False).agg(
        residual_sum=("local_residual", "sum"),
        n=("local_residual", "size"),
    ).reset_index()
    grouped["context_correction"] = grouped["residual_sum"] / (grouped["n"] + smoothing)
    joined = valid.merge(
        grouped[keys + ["context_correction", "n"]],
        on=keys, how="left", validate="many_to_one",
    )
    if not joined["row_id"].reset_index(drop=True).equals(
        valid["row_id"].reset_index(drop=True)
    ):
        raise RuntimeError("Context lookup changed validation row order")
    is_r = joined["game_type"].eq("R").to_numpy()
    output = np.zeros(len(joined), dtype="float64")
    output[is_r] = joined.loc[is_r, "context_correction"].fillna(0.0).to_numpy(float)
    audit = {
        "train_year": train_year,
        "valid_year": valid_year,
        "cells": len(grouped),
        "train_residual_mean": float(train["residual"].mean()),
        "coverage_r": float(joined.loc[is_r, "context_correction"].notna().mean()),
        "median_cell_n": float(grouped["n"].median()),
        "min_cell_n": int(grouped["n"].min()),
    }
    return output, audit


def sufficient_stats(
    frame: pd.DataFrame,
    folds: dict[int, pd.DataFrame],
    middle: dict[int, np.ndarray],
    context: dict[int, np.ndarray],
) -> tuple[dict, dict]:
    stats = {}
    denominators = {}
    for year in [2023, 2024]:
        part = frame.loc[frame["season"].eq(year)]
        fold = folds[year].set_index("row_id").loc[part["row_id"]]
        y = part["control_success"].to_numpy(float)
        error = fold["current_blend"].to_numpy(float) - y
        components = np.column_stack([middle[year], context[year]])
        r = part["game_type"].eq("R").to_numpy()
        stats[year] = {}
        denominators[year] = {}
        for group, mask in {"ALL": np.ones(len(part), bool), "R": r, "F": ~r}.items():
            local_e = error[mask]
            local_x = components[mask]
            stats[year][group] = {
                "linear": np.mean(local_x * local_e[:, None], axis=0),
                "quadratic": (local_x.T @ local_x) / len(local_x),
            }
            local_y = y[mask]
            denominators[year][group] = float(local_y.mean() * (1.0 - local_y.mean()))
    return stats, denominators


def delta(stat: dict, scales: np.ndarray) -> float:
    return float(
        2.0 * stat["linear"] @ scales
        + scales @ stat["quadratic"] @ scales
    )


def tune(stats: dict, denominators: dict) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for middle_scale, context_scale in itertools.product(MIDDLE_SCALES, CONTEXT_SCALES):
        scales = np.asarray([middle_scale, context_scale], dtype=float)
        changes = {
            (year, group): delta(stats[year][group], scales)
            for year in [2023, 2024] for group in ["ALL", "R", "F"]
        }
        normalized = {
            key: value / denominators[key[0]][key[1]]
            for key, value in changes.items()
        }
        n23, n24 = normalized[(2023, "ALL")], normalized[(2024, "ALL")]
        r23, r24 = normalized[(2023, "R")], normalized[(2024, "R")]
        objective = (
            0.30 * n23 + 0.70 * n24 + 0.50 * max(n23, n24, 0.0)
            + 0.25 * (0.30 * r23 + 0.70 * r24) + max(r23, r24, 0.0)
        )
        rows.append({
            "middle_scale": middle_scale,
            "context_scale": context_scale,
            "val2023_delta_brier": changes[(2023, "ALL")],
            "val2024_delta_brier": changes[(2024, "ALL")],
            "val2023_r_delta_brier": changes[(2023, "R")],
            "val2024_r_delta_brier": changes[(2024, "R")],
            "val2023_f_delta_brier": changes[(2023, "F")],
            "val2024_f_delta_brier": changes[(2024, "F")],
            "both_improve": n23 < 0.0 and n24 < 0.0,
            "r_both_improve": r23 < 0.0 and r24 < 0.0,
            "objective": objective,
        })
    grid = pd.DataFrame(rows)
    eligible = grid.loc[grid["both_improve"] & grid["r_both_improve"]]
    best = (eligible if len(eligible) else grid).sort_values("objective").iloc[0]
    return grid, best


def exact_metrics(
    frame: pd.DataFrame,
    folds: dict[int, pd.DataFrame],
    middle: dict[int, np.ndarray],
    context: dict[int, np.ndarray],
    best: pd.Series,
) -> dict:
    output = {}
    for year in [2023, 2024]:
        part = frame.loc[frame["season"].eq(year)]
        fold = folds[year].set_index("row_id").loc[part["row_id"]]
        prediction = np.clip(
            fold["current_blend"].to_numpy(float)
            + float(best["middle_scale"]) * middle[year]
            + float(best["context_scale"]) * context[year],
            1e-6, 1.0 - 1e-6,
        )
        y = part["control_success"].to_numpy(float)
        r = part["game_type"].eq("R").to_numpy()
        for group, mask in {"all": np.ones(len(part), bool), "r": r, "f": ~r}.items():
            output[f"val{year}_{group}_brier"] = brier(y[mask], prediction[mask])
            output[f"val{year}_{group}_bss"] = bss(y[mask], prediction[mask])
    return output


def main() -> None:
    main_frame = load_main()
    folds = load_fold_predictions()
    base = load_base().merge(
        main_frame[["row_id", "game_type"]], on="row_id", validate="one_to_one"
    )
    base = attach_honest_references(base, folds)
    context_columns = load_context()
    frame = base.merge(
        context_columns.drop(columns=["game_type"]),
        on=["row_id", "season"], validate="one_to_one",
    )

    middle_features, middle_audit = build_features(
        main_frame, MIDDLE_CONFIG, MIDDLE_BATTER_K, MIDDLE_SMOOTHING
    )
    frame = frame.merge(middle_features, on=["row_id", "season"], validate="one_to_one")
    middle = {
        year: middle_correction(
            frame, MIDDLE_ALPHA, train_year, year, MIDDLE_TARGET_MODE
        )
        for train_year, year in [(2022, 2023), (2023, 2024)]
    }

    all_grid = []
    summaries = []
    audits = []
    for context_name, keys in CONTEXTS.items():
        for smoothing, reference_mode, centered in itertools.product(
            SMOOTHINGS, REFERENCE_MODES, CENTER_MODES
        ):
            corrections = {}
            local_audit = []
            for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
                correction, audit = residual_lookup(
                    frame, keys, train_year, valid_year, smoothing,
                    reference_mode, centered,
                )
                corrections[valid_year] = correction
                local_audit.append(audit)
            stats, denominators = sufficient_stats(frame, folds, middle, corrections)
            grid, best = tune(stats, denominators)
            metadata = {
                "context": context_name,
                "keys": "+".join(keys),
                "smoothing": smoothing,
                "reference_mode": reference_mode,
                "centered": centered,
            }
            all_grid.append(grid.assign(**metadata))
            summaries.append({
                **metadata,
                **best.to_dict(),
                **exact_metrics(frame, folds, middle, corrections, best),
                "coverage_val2023_r": local_audit[0]["coverage_r"],
                "coverage_val2024_r": local_audit[1]["coverage_r"],
            })
            audits.extend({**metadata, **row} for row in local_audit)
        print(json.dumps({"context": context_name, "completed": True}), flush=True)

    summary = pd.DataFrame(summaries).sort_values(
        ["r_both_improve", "both_improve", "objective"],
        ascending=[False, False, True],
    )
    grid = pd.concat(all_grid, ignore_index=True).sort_values("objective")
    reports = WORK / "reports"
    grid.to_csv(reports / "r_context_residual_grid.csv", index=False)
    summary.to_csv(reports / "r_context_residual_summary.csv", index=False)
    pd.DataFrame(audits).to_csv(reports / "r_context_residual_audit.csv", index=False)
    best = summary.iloc[0]
    payload = {
        "middle_audit": middle_audit,
        "evaluated": len(summary),
        "both_improve": int(summary["both_improve"].sum()),
        "r_both_improve": int(summary["r_both_improve"].sum()),
        "best": best.to_dict(),
    }
    (reports / "r_context_residual_best.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTOP 20")
    print(summary.head(20).to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
