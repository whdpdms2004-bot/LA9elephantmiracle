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
    middle_correction,
)
from screen_r_context_residual import (  # noqa: E402
    exact_metrics,
    residual_lookup,
    sufficient_stats,
    tune,
)
from screen_r_middle_preprocessing import build_features, load_base, load_main  # noqa: E402


VARIANTS = {
    "inning4": ["balls_before", "strikes_before", "inning4"],
    "inning3": ["balls_before", "strikes_before", "inning3"],
    "inning2": ["balls_before", "strikes_before", "inning2"],
    "inning_exact": ["balls_before", "strikes_before", "inning"],
    "inning4_half": ["balls_before", "strikes_before", "inning4", "top_bottom"],
    "inning4_hands": [
        "balls_before", "strikes_before", "inning4", "pitcher_hand", "batter_hand"
    ],
    "inning4_outs": ["balls_before", "strikes_before", "inning4", "outs_before"],
    "inning4_runners": [
        "balls_before", "strikes_before", "inning4", "num_runners_on"
    ],
}
SMOOTHINGS = [500.0, 1000.0, 2000.0, 3000.0, 5000.0]
REFERENCE_CENTER = [("honest", False), ("adjusted", True)]


def load_game() -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=[
            "row_id", "season", "game_type", "inning", "top_bottom",
            "balls_before", "strikes_before", "outs_before", "num_runners_on",
            "pitcher_hand", "batter_hand",
        ],
    )
    frame["inning4"] = pd.cut(
        frame["inning"], [0, 3, 6, 9, np.inf],
        labels=["1-3", "4-6", "7-9", "10+"], right=True,
    ).astype(str)
    frame["inning3"] = pd.cut(
        frame["inning"], [0, 3, 6, np.inf],
        labels=["1-3", "4-6", "7+"], right=True,
    ).astype(str)
    frame["inning2"] = pd.cut(
        frame["inning"], [0, 5, np.inf],
        labels=["1-5", "6+"], right=True,
    ).astype(str)
    return frame


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

    features, middle_audit = build_features(
        main_frame, "pv2_compact_mi_r5_2fec3702", (2, 3), 500.0
    )
    frame = frame.merge(features, on=["row_id", "season"], validate="one_to_one")
    middle = {
        year: middle_correction(
            frame, 10.0, train_year, year, "adjusted_residual"
        )
        for train_year, year in [(2022, 2023), (2023, 2024)]
    }

    summaries = []
    grids = []
    audits = []
    for variant, keys in VARIANTS.items():
        for smoothing, (reference, centered) in itertools.product(
            SMOOTHINGS, REFERENCE_CENTER
        ):
            context = {}
            local_audit = []
            for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
                values, audit = residual_lookup(
                    frame, keys, train_year, valid_year, smoothing,
                    reference, centered,
                )
                context[valid_year] = values
                local_audit.append(audit)
            stats, denominators = sufficient_stats(frame, folds, middle, context)
            grid, best = tune(stats, denominators)
            metadata = {
                "variant": variant,
                "keys": "+".join(keys),
                "smoothing": smoothing,
                "reference_mode": reference,
                "centered": centered,
            }
            grids.append(grid.assign(**metadata))
            summaries.append({
                **metadata,
                **best.to_dict(),
                **exact_metrics(frame, folds, middle, context, best),
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
    summary.to_csv(reports / "r_count_inning_fine_summary.csv", index=False)
    grid.to_csv(reports / "r_count_inning_fine_grid.csv", index=False)
    pd.DataFrame(audits).to_csv(reports / "r_count_inning_fine_audit.csv", index=False)
    payload = {
        "middle_audit": middle_audit,
        "evaluated": len(summary),
        "both_improve": int(summary["both_improve"].sum()),
        "r_both_improve": int(summary["r_both_improve"].sum()),
        "best": summary.iloc[0].to_dict(),
    }
    (reports / "r_count_inning_fine_best.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTOP 20")
    print(summary.head(20).to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
