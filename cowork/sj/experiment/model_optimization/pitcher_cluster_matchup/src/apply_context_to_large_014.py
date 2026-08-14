from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
REPORTS = WORK / "reports"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(MODEL_DIR))

from analyze_r_focus import load_fold_predictions  # noqa: E402
from integrate_r_middle_current_blend import attach_honest_references, brier, bss  # noqa: E402
from screen_r_context_history import history_correction  # noqa: E402
from screen_r_middle_preprocessing import load_base, load_main  # noqa: E402
from search_causal_large_xgb import (  # noqa: E402
    attach_features,
    build_variants,
    correction_components,
    read_raw,
    tune_corrections,
)


FINAL_BASE_WEIGHT = 0.3595515607106235
FINAL_CURRENT_WEIGHT = 0.25166569203957306
FINAL_LARGE_WEIGHT = 0.3887827472498035
SCALES = np.round(np.arange(0.0, 1.501, 0.025), 3)


def load_game() -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=[
            "row_id", "season", "game_type", "inning",
            "balls_before", "strikes_before", "pitcher_hand", "batter_hand",
        ],
    )
    frame["inning4"] = pd.cut(
        frame["inning"], [0, 3, 6, 9, np.inf],
        labels=["1-3", "4-6", "7-9", "10+"], right=True,
    ).astype(str)
    return frame


def build_014_predictions() -> tuple[dict[int, pd.DataFrame], dict]:
    raw = read_raw()
    variants, _ = build_variants(raw)
    attached = attach_features(raw)
    components = correction_components(raw, variants, attached)
    _, corrected = tune_corrections(raw, variants, components)
    ensemble = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    output = {}
    for year, track in [(2023, "robust"), (2024, "performance")]:
        current = np.clip(
            raw[year]["anchor"]
            + 0.25 * components["anchor"][year]["success"]
            + 0.55 * components["anchor"][year]["reverse"],
            1e-6, 1.0 - 1e-6,
        )
        base = (
            ensemble.loc[
                ensemble["season"].eq(year) & ensemble["track"].eq(track),
                ["row_id", "prediction"],
            ].set_index("row_id")
            .reindex(raw[year]["row_id"])["prediction"].to_numpy(float)
        )
        large = corrected[("fixed_large_w09", year)]
        prediction = (
            FINAL_BASE_WEIGHT * base
            + FINAL_CURRENT_WEIGHT * current
            + FINAL_LARGE_WEIGHT * large
        )
        output[year] = pd.DataFrame({
            "row_id": raw[year]["row_id"],
            "control_success": raw[year]["y"],
            "prediction_014": prediction,
        })
    return output, raw


def main() -> None:
    predictions, _ = build_014_predictions()
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
    keys = [
        "balls_before", "strikes_before", "inning4", "pitcher_hand", "batter_hand"
    ]
    correction = {
        year: history_correction(frame, keys, year, 5000.0, 1.0)[0]
        for year in [2023, 2024]
    }
    stats = {}
    for year in [2023, 2024]:
        part = frame.loc[frame["season"].eq(year)]
        pred = predictions[year].set_index("row_id").loc[part["row_id"]]
        y = part["control_success"].to_numpy(float)
        baseline = pred["prediction_014"].to_numpy(float)
        r = part["game_type"].eq("R").to_numpy()
        stats[year] = {
            "y": y, "baseline": baseline, "r": r,
            "correction": correction[year],
            "denominator": float(y.mean() * (1.0 - y.mean())),
            "r_denominator": float(y[r].mean() * (1.0 - y[r].mean())),
        }
    rows = []
    for scale in SCALES:
        row = {"scale": scale}
        normalized = {}
        for year in [2023, 2024]:
            local = stats[year]
            new = np.clip(
                local["baseline"] + scale * local["correction"], 1e-6, 1.0 - 1e-6
            )
            delta_all = brier(local["y"], new) - brier(local["y"], local["baseline"])
            r = local["r"]
            delta_r = brier(local["y"][r], new[r]) - brier(
                local["y"][r], local["baseline"][r]
            )
            row[f"val{year}_delta_brier"] = delta_all
            row[f"val{year}_r_delta_brier"] = delta_r
            row[f"val{year}_bss"] = bss(local["y"], new)
            row[f"val{year}_r_bss"] = bss(local["y"][r], new[r])
            normalized[(year, "ALL")] = delta_all / local["denominator"]
            normalized[(year, "R")] = delta_r / local["r_denominator"]
        n23, n24 = normalized[(2023, "ALL")], normalized[(2024, "ALL")]
        r23, r24 = normalized[(2023, "R")], normalized[(2024, "R")]
        row["both_improve"] = n23 < 0 and n24 < 0
        row["r_both_improve"] = r23 < 0 and r24 < 0
        row["objective"] = (
            0.30 * n23 + 0.70 * n24 + 0.50 * max(n23, n24, 0.0)
            + 0.25 * (0.30 * r23 + 0.70 * r24) + max(r23, r24, 0.0)
        )
        rows.append(row)
    grid = pd.DataFrame(rows)
    eligible = grid.loc[grid["both_improve"] & grid["r_both_improve"]]
    robust = eligible.sort_values("objective").iloc[0]
    recent = eligible.sort_values("val2024_delta_brier").iloc[0]
    baseline_metrics = {
        year: {
            "bss": bss(stats[year]["y"], stats[year]["baseline"]),
            "r_bss": bss(
                stats[year]["y"][stats[year]["r"]],
                stats[year]["baseline"][stats[year]["r"]],
            ),
        }
        for year in [2023, 2024]
    }
    grid.to_csv(REPORTS / "r_context_large014_grid.csv", index=False)
    payload = {
        "baseline": baseline_metrics,
        "robust": robust.to_dict(),
        "recent": recent.to_dict(),
        "context": {
            "keys": keys, "history": "equal-weight prior OOF seasons",
            "smoothing": 5000.0,
        },
    }
    (REPORTS / "r_context_large014_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
