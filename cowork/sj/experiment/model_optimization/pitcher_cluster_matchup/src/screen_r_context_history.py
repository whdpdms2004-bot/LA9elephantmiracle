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
from integrate_r_middle_current_blend import attach_honest_references, brier, bss  # noqa: E402
from screen_r_middle_preprocessing import load_base, load_main  # noqa: E402


VARIANTS = {
    "inning_exact": ["balls_before", "strikes_before", "inning"],
    "inning4": ["balls_before", "strikes_before", "inning4"],
    "inning4_runners": [
        "balls_before", "strikes_before", "inning4", "num_runners_on"
    ],
    "inning4_hands": [
        "balls_before", "strikes_before", "inning4",
        "pitcher_hand", "batter_hand",
    ],
}
OLDER_WEIGHTS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
SMOOTHINGS = [500.0, 1000.0, 2000.0, 3000.0, 5000.0]
SCALES = np.round(np.arange(0.0, 1.501, 0.025), 3)


def load_game() -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=[
            "row_id", "season", "game_type", "inning",
            "balls_before", "strikes_before", "num_runners_on",
            "pitcher_hand", "batter_hand",
        ],
    )
    frame["inning4"] = pd.cut(
        frame["inning"], [0, 3, 6, 9, np.inf],
        labels=["1-3", "4-6", "7-9", "10+"], right=True,
    ).astype(str)
    return frame


def history_correction(
    frame: pd.DataFrame,
    keys: list[str],
    valid_year: int,
    smoothing: float,
    older_weight: float,
) -> tuple[np.ndarray, dict]:
    latest_year = valid_year - 1
    train = frame.loc[
        frame["season"].le(latest_year) & frame["game_type"].eq("R")
    ].copy()
    train["weight"] = np.where(
        train["season"].eq(latest_year), 1.0, older_weight
    )
    train = train.loc[train["weight"].gt(0)].copy()
    train["residual"] = train["control_success"] - train["honest_blend"]
    train["weighted_residual"] = train["residual"] * train["weight"]
    grouped = train.groupby(keys, observed=True, dropna=False).agg(
        residual_sum=("weighted_residual", "sum"),
        effective_n=("weight", "sum"),
        raw_n=("weight", "size"),
    ).reset_index()
    grouped["correction"] = grouped["residual_sum"] / (
        grouped["effective_n"] + smoothing
    )
    valid = frame.loc[frame["season"].eq(valid_year)].copy()
    joined = valid.merge(
        grouped[keys + ["correction"]], on=keys,
        how="left", validate="many_to_one",
    )
    if not joined["row_id"].reset_index(drop=True).equals(
        valid["row_id"].reset_index(drop=True)
    ):
        raise RuntimeError("History lookup changed validation row order")
    r = joined["game_type"].eq("R").to_numpy()
    output = np.zeros(len(joined), dtype="float64")
    output[r] = joined.loc[r, "correction"].fillna(0.0).to_numpy(float)
    return output, {
        "valid_year": valid_year,
        "training_seasons": "+".join(map(str, sorted(train["season"].unique()))),
        "cells": len(grouped),
        "coverage_r": float(joined.loc[r, "correction"].notna().mean()),
        "median_effective_n": float(grouped["effective_n"].median()),
    }


def fold_terms(
    frame: pd.DataFrame,
    folds: dict[int, pd.DataFrame],
    correction: dict[int, np.ndarray],
) -> tuple[dict, dict]:
    terms = {}
    denominator = {}
    for year in [2023, 2024]:
        part = frame.loc[frame["season"].eq(year)]
        fold = folds[year].set_index("row_id").loc[part["row_id"]]
        y = part["control_success"].to_numpy(float)
        error = fold["current_blend"].to_numpy(float) - y
        r = part["game_type"].eq("R").to_numpy()
        terms[year] = {}
        denominator[year] = {}
        for group, mask in {"ALL": np.ones(len(part), bool), "R": r}.items():
            local_e = error[mask]
            local_c = correction[year][mask]
            terms[year][group] = {
                "ec": float(np.mean(local_e * local_c)),
                "cc": float(np.mean(local_c ** 2)),
            }
            local_y = y[mask]
            denominator[year][group] = float(local_y.mean() * (1.0 - local_y.mean()))
    return terms, denominator


def tune(terms: dict, denominator: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rows = []
    for scale in SCALES:
        delta = {
            (year, group): 2 * scale * terms[year][group]["ec"]
            + scale * scale * terms[year][group]["cc"]
            for year in [2023, 2024] for group in ["ALL", "R"]
        }
        normalized = {
            key: value / denominator[key[0]][key[1]] for key, value in delta.items()
        }
        n23, n24 = normalized[(2023, "ALL")], normalized[(2024, "ALL")]
        r23, r24 = normalized[(2023, "R")], normalized[(2024, "R")]
        objective = (
            0.30 * n23 + 0.70 * n24 + 0.50 * max(n23, n24, 0.0)
            + 0.25 * (0.30 * r23 + 0.70 * r24) + max(r23, r24, 0.0)
        )
        rows.append({
            "scale": scale,
            "val2023_delta_brier": delta[(2023, "ALL")],
            "val2024_delta_brier": delta[(2024, "ALL")],
            "val2023_r_delta_brier": delta[(2023, "R")],
            "val2024_r_delta_brier": delta[(2024, "R")],
            "both_improve": n23 < 0 and n24 < 0,
            "r_both_improve": r23 < 0 and r24 < 0,
            "objective": objective,
        })
    grid = pd.DataFrame(rows)
    eligible = grid.loc[grid["both_improve"] & grid["r_both_improve"]]
    robust = (eligible if len(eligible) else grid).sort_values("objective").iloc[0]
    recent = (eligible if len(eligible) else grid).sort_values("val2024_delta_brier").iloc[0]
    return grid, robust, recent


def exact_metrics(
    frame: pd.DataFrame,
    folds: dict[int, pd.DataFrame],
    correction: dict[int, np.ndarray],
    scale: float,
) -> dict:
    output = {}
    for year in [2023, 2024]:
        part = frame.loc[frame["season"].eq(year)]
        fold = folds[year].set_index("row_id").loc[part["row_id"]]
        y = part["control_success"].to_numpy(float)
        base_prediction = fold["current_blend"].to_numpy(float)
        prediction = np.clip(
            base_prediction + scale * correction[year], 1e-6, 1.0 - 1e-6
        )
        r = part["game_type"].eq("R").to_numpy()
        for group, mask in {"all": np.ones(len(part), bool), "r": r}.items():
            output[f"val{year}_{group}_brier"] = brier(y[mask], prediction[mask])
            output[f"val{year}_{group}_bss"] = bss(y[mask], prediction[mask])
    return output


def main() -> None:
    folds = load_fold_predictions()
    main_frame = load_main()
    base = load_base().merge(
        main_frame[["row_id", "game_type"]], on="row_id", validate="one_to_one"
    )
    base = attach_honest_references(base, folds)
    frame = base.merge(
        load_game().drop(columns=["game_type"]),
        on=["row_id", "season"], validate="one_to_one",
    )
    grids = []
    summaries = []
    audits = []
    for variant, keys in VARIANTS.items():
        for older_weight, smoothing in itertools.product(OLDER_WEIGHTS, SMOOTHINGS):
            correction = {}
            local_audit = []
            for valid_year in [2023, 2024]:
                correction[valid_year], audit = history_correction(
                    frame, keys, valid_year, smoothing, older_weight
                )
                local_audit.append(audit)
            terms, denominator = fold_terms(frame, folds, correction)
            grid, robust, recent = tune(terms, denominator)
            metadata = {
                "variant": variant,
                "keys": "+".join(keys),
                "older_weight": older_weight,
                "smoothing": smoothing,
            }
            grids.append(grid.assign(**metadata))
            for selection, row in [("robust", robust), ("recent", recent)]:
                summaries.append({
                    **metadata,
                    "selection": selection,
                    **row.to_dict(),
                    **exact_metrics(frame, folds, correction, float(row["scale"])),
                    "coverage_val2023_r": local_audit[0]["coverage_r"],
                    "coverage_val2024_r": local_audit[1]["coverage_r"],
                })
            audits.extend({**metadata, **row} for row in local_audit)
        print(json.dumps({"variant": variant, "completed": True}), flush=True)

    summary = pd.DataFrame(summaries).sort_values(
        ["r_both_improve", "both_improve", "objective"],
        ascending=[False, False, True],
    )
    grid = pd.concat(grids, ignore_index=True).sort_values("objective")
    reports = WORK / "reports"
    summary.to_csv(reports / "r_context_history_summary.csv", index=False)
    grid.to_csv(reports / "r_context_history_grid.csv", index=False)
    pd.DataFrame(audits).to_csv(reports / "r_context_history_audit.csv", index=False)
    payload = {
        "evaluated": len(summary),
        "both_improve": int(summary["both_improve"].sum()),
        "r_both_improve": int(summary["r_both_improve"].sum()),
        "best_robust": summary.loc[summary["selection"].eq("robust")].iloc[0].to_dict(),
        "best_recent": summary.loc[summary["selection"].eq("recent")]
            .sort_values("val2024_delta_brier").iloc[0].to_dict(),
    }
    (reports / "r_context_history_best.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTOP ROBUST")
    print(summary.loc[summary["selection"].eq("robust")].head(15).to_string(index=False))
    print("\nTOP RECENT")
    print(
        summary.loc[summary["selection"].eq("recent")]
        .sort_values("val2024_delta_brier").head(15).to_string(index=False)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
