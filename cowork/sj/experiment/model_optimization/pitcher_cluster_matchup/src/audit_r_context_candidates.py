from __future__ import annotations

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
from screen_r_context_residual import residual_lookup  # noqa: E402
from screen_r_middle_preprocessing import load_base, load_main  # noqa: E402


CANDIDATES = {
    "robust_hands": {
        "keys": [
            "balls_before", "strikes_before", "inning4",
            "pitcher_hand", "batter_hand",
        ],
        "smoothing": 1000.0,
        "scale": 0.60,
    },
    "recent_exact": {
        "keys": ["balls_before", "strikes_before", "inning"],
        "smoothing": 2000.0,
        "scale": 0.70,
    },
    "balanced_runners": {
        "keys": [
            "balls_before", "strikes_before", "inning4", "num_runners_on"
        ],
        "smoothing": 500.0,
        "scale": 0.45,
    },
}


def load_game() -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=[
            "row_id", "season", "game_type", "game_month", "inning",
            "balls_before", "strikes_before", "outs_before", "num_runners_on",
            "pitcher_hand", "batter_hand",
        ],
    )
    frame["inning4"] = pd.cut(
        frame["inning"], [0, 3, 6, 9, np.inf],
        labels=["1-3", "4-6", "7-9", "10+"], right=True,
    ).astype(str)
    frame["count_state"] = (
        frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    )
    frame["season_half"] = np.where(frame["game_month"].le(6), "first", "second")
    return frame


def grouped_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    prediction: np.ndarray,
    group_columns: list[str],
) -> pd.DataFrame:
    work = frame[group_columns].copy()
    work["base_error2"] = (baseline - frame["control_success"].to_numpy(float)) ** 2
    work["new_error2"] = (prediction - frame["control_success"].to_numpy(float)) ** 2
    return work.groupby(group_columns, observed=True, dropna=False).agg(
        n=("base_error2", "size"),
        base_brier=("base_error2", "mean"),
        new_brier=("new_error2", "mean"),
    ).reset_index().assign(
        delta_brier=lambda value: value["new_brier"] - value["base_brier"]
    )


def correction_table(
    frame: pd.DataFrame, keys: list[str], train_year: int, smoothing: float
) -> pd.DataFrame:
    train = frame.loc[
        frame["season"].eq(train_year) & frame["game_type"].eq("R")
    ].copy()
    train["residual"] = train["control_success"] - train["honest_blend"]
    table = train.groupby(keys, observed=True, dropna=False).agg(
        residual_sum=("residual", "sum"), n=("residual", "size")
    ).reset_index()
    table["correction"] = table["residual_sum"] / (table["n"] + smoothing)
    return table


def main() -> None:
    main_frame = load_main()
    folds = load_fold_predictions()
    base = load_base().merge(
        main_frame[["row_id", "game_type"]], on="row_id", validate="one_to_one"
    )
    base = attach_honest_references(base, folds)
    frame = base.merge(
        load_game().drop(columns=["game_type"]),
        on=["row_id", "season"], validate="one_to_one",
    )

    metrics = []
    subgroup_tables = []
    stability = []
    predictions = []
    for name, config in CANDIDATES.items():
        correction = {}
        for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
            correction[valid_year], _ = residual_lookup(
                frame, config["keys"], train_year, valid_year,
                config["smoothing"], "honest", False,
            )
        for year in [2023, 2024]:
            part = frame.loc[frame["season"].eq(year)].copy()
            fold = folds[year].set_index("row_id").loc[part["row_id"]]
            y = part["control_success"].to_numpy(float)
            baseline = fold["current_blend"].to_numpy(float)
            new_prediction = np.clip(
                baseline + config["scale"] * correction[year], 1e-6, 1.0 - 1e-6
            )
            r = part["game_type"].eq("R").to_numpy()
            local_correction = correction[year][r]
            local_error = baseline[r] - y[r]
            optimal_scale = float(
                -np.mean(local_error * local_correction)
                / np.mean(local_correction ** 2)
            )
            for group, mask in {"ALL": np.ones(len(part), bool), "R": r, "F": ~r}.items():
                metrics.append({
                    "candidate": name,
                    "season": year,
                    "group": group,
                    "n": int(mask.sum()),
                    "base_brier": brier(y[mask], baseline[mask]),
                    "new_brier": brier(y[mask], new_prediction[mask]),
                    "delta_brier": brier(y[mask], new_prediction[mask]) - brier(y[mask], baseline[mask]),
                    "base_bss": bss(y[mask], baseline[mask]),
                    "new_bss": bss(y[mask], new_prediction[mask]),
                    "optimal_scale_r": optimal_scale if group == "R" else np.nan,
                })
            r_part = part.loc[r].reset_index(drop=True)
            for dimensions in [
                ["game_month"], ["season_half"], ["count_state"],
                ["inning4"], ["pitcher_hand"], ["batter_hand"],
            ]:
                table = grouped_delta(
                    r_part, baseline[r], new_prediction[r], dimensions
                )
                table["candidate"] = name
                table["season"] = year
                table["dimension"] = "+".join(dimensions)
                table["group_value"] = table[dimensions].astype(str).agg("|".join, axis=1)
                subgroup_tables.append(table[[
                    "candidate", "season", "dimension", "group_value", "n",
                    "base_brier", "new_brier", "delta_brier",
                ]])
            predictions.append(pd.DataFrame({
                "row_id": part["row_id"].to_numpy(),
                "season": year,
                "candidate": name,
                "baseline": baseline,
                "prediction": new_prediction,
                "correction": correction[year],
            }))

        table22 = correction_table(frame, config["keys"], 2022, config["smoothing"])
        table23 = correction_table(frame, config["keys"], 2023, config["smoothing"])
        shared = table22.merge(
            table23, on=config["keys"], suffixes=("_2022", "_2023"),
            validate="one_to_one",
        )
        stable = shared.loc[(shared["n_2022"] >= 100) & (shared["n_2023"] >= 100)]
        stability.append({
            "candidate": name,
            "cells_2022": len(table22),
            "cells_2023": len(table23),
            "shared_cells": len(shared),
            "stable_cells": len(stable),
            "correction_pearson_all": float(shared["correction_2022"].corr(shared["correction_2023"])),
            "correction_pearson_n100": float(stable["correction_2022"].corr(stable["correction_2023"])),
            "sign_agreement_all": float(
                (np.sign(shared["correction_2022"]) == np.sign(shared["correction_2023"])).mean()
            ),
            "sign_agreement_n100": float(
                (np.sign(stable["correction_2022"]) == np.sign(stable["correction_2023"])).mean()
            ),
        })

    metrics_frame = pd.DataFrame(metrics)
    subgroups = pd.concat(subgroup_tables, ignore_index=True)
    stability_frame = pd.DataFrame(stability)
    predictions_frame = pd.concat(predictions, ignore_index=True)
    reports = WORK / "reports"
    metrics_frame.to_csv(reports / "r_context_candidate_metrics.csv", index=False)
    subgroups.to_csv(reports / "r_context_candidate_subgroups.csv", index=False)
    stability_frame.to_csv(reports / "r_context_candidate_stability.csv", index=False)
    predictions_frame.to_parquet(reports / "r_context_candidate_predictions.parquet", index=False)

    month = subgroups.loc[subgroups["dimension"].eq("game_month")]
    payload = {
        "metrics": metrics_frame.to_dict(orient="records"),
        "stability": stability_frame.to_dict(orient="records"),
        "month_improvement_counts": (
            month.assign(improved=month["delta_brier"].lt(0))
            .groupby(["candidate", "season"])["improved"]
            .agg(["sum", "count"]).reset_index().to_dict(orient="records")
        ),
    }
    (reports / "r_context_candidate_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("METRICS")
    print(metrics_frame.to_string(index=False))
    print("\nSTABILITY")
    print(stability_frame.to_string(index=False))
    print("\nMONTH IMPROVEMENT")
    print(pd.DataFrame(payload["month_improvement_counts"]).to_string(index=False))


if __name__ == "__main__":
    main()
