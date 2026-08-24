from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "experiment" / "model_optimization"
TARGET = "control_success"
COMPONENTS = ["success", "reverse", "middle", "ball", "strike"]


def infer_previous_event(frame: pd.DataFrame, component: str, linked: pd.Series):
    rate_column = f"asof_pitcher_{component}_rate"
    current_rate = frame[rate_column]
    next_rate = frame.groupby("pitcher_id", sort=False)[rate_column].shift(-1)
    inferred = (
        next_rate * (frame["asof_pitcher_n"] + 1)
        - current_rate * frame["asof_pitcher_n"]
    )
    valid = linked & current_rate.notna() & next_rate.notna()
    return inferred.round().clip(0, 1).astype("Int8").where(valid), inferred.where(valid)


def main():
    usecols = [
        "row_id",
        "season",
        "pitcher_id",
        "asof_pitcher_n",
        TARGET,
    ] + [f"asof_pitcher_{name}_rate" for name in COMPONENTS]
    frame = pd.read_csv(ROOT / "data" / "train.csv", usecols=usecols)
    next_n = frame.groupby("pitcher_id", sort=False)["asof_pitcher_n"].shift(-1)
    linked = next_n.eq(frame["asof_pitcher_n"] + 1)

    labels = frame[["row_id", "season", TARGET]].copy()
    raw_errors = {}
    for component in COMPONENTS:
        labels[component], raw = infer_previous_event(frame, component, linked)
        binary = labels[component].notna()
        raw_errors[component] = {
            "rows": int(binary.sum()),
            "mean_abs_rounding_error": float(
                (raw[binary] - raw[binary].round()).abs().mean()
            ),
            "max_abs_rounding_error": float(
                (raw[binary] - raw[binary].round()).abs().max()
            ),
        }

    complete = labels[COMPONENTS].notna().all(axis=1)
    success_match = labels.loc[complete, "success"].astype("int8").eq(
        labels.loc[complete, TARGET]
    )
    if not success_match.all():
        raise AssertionError("Inferred success does not match official target")

    failure = 1 - labels[TARGET]
    labels["outside_only"] = (
        failure * (1 - labels["reverse"]) * (1 - labels["middle"])
    ).astype("Int8")

    # Class 0 always means success so predict_proba[:, 0] is directly usable.
    labels["failure_class4_middle_first"] = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    labels["failure_class4_reverse_first"] = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    labels["failure_class5"] = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    labels.loc[complete, "failure_class4_middle_first"] = np.select(
        [
            labels.loc[complete, TARGET].eq(1),
            labels.loc[complete, "middle"].eq(1),
            labels.loc[complete, "reverse"].eq(1),
        ],
        [0, 1, 2],
        default=3,
    ).astype("int8")
    labels.loc[complete, "failure_class4_reverse_first"] = np.select(
        [
            labels.loc[complete, TARGET].eq(1),
            labels.loc[complete, "reverse"].eq(1),
            labels.loc[complete, "middle"].eq(1),
        ],
        [0, 1, 2],
        default=3,
    ).astype("int8")
    labels.loc[complete, "failure_class5"] = np.select(
        [
            labels.loc[complete, TARGET].eq(1),
            labels.loc[complete, "middle"].eq(1)
            & labels.loc[complete, "reverse"].eq(0),
            labels.loc[complete, "reverse"].eq(1)
            & labels.loc[complete, "middle"].eq(0),
            labels.loc[complete, "reverse"].eq(1)
            & labels.loc[complete, "middle"].eq(1),
        ],
        [0, 1, 2, 3],
        default=4,
    ).astype("int8")

    output_path = WORK_DIR / "failure_component_labels.parquet"
    labels.to_parquet(output_path, index=False)
    def json_counts(column: str):
        counts = labels[column].value_counts(dropna=False).sort_index()
        return {str(key): int(value) for key, value in counts.items()}

    audit = {
        "rows": len(labels),
        "complete_component_rows": int(complete.sum()),
        "coverage": float(complete.mean()),
        "inferred_success_match": float(success_match.mean()),
        "component_rates": {
            column: float(labels.loc[complete, column].mean())
            for column in [TARGET, "reverse", "middle", "outside_only", "ball", "strike"]
        },
        "class4_middle_first_counts": json_counts("failure_class4_middle_first"),
        "class4_reverse_first_counts": json_counts("failure_class4_reverse_first"),
        "class5_counts": json_counts("failure_class5"),
        "raw_rounding": raw_errors,
        "leakage_note": (
            "The next training row is used only to reconstruct an auxiliary training label. "
            "No reconstructed current-pitch component is used as an inference feature."
        ),
    }
    (WORK_DIR / "failure_component_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
