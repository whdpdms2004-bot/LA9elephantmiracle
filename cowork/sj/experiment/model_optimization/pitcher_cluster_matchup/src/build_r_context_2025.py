from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
ARTIFACT_DIR = WORK / "artifacts" / "r_context_2025"
sys.path.insert(0, str(MODEL_DIR))

from analyze_r_focus import load_fold_predictions  # noqa: E402


KEYS = [
    "balls_before", "strikes_before", "inning_bucket",
    "pitcher_hand", "batter_hand",
]
CANDIDATES = {
    "robust": {"smoothing": 500.0, "scale": 0.60},
    "recent": {"smoothing": 5000.0, "scale": 1.15},
}


def make_lookup(frame: pd.DataFrame, smoothing: float, scale: float) -> pd.DataFrame:
    work = frame.loc[frame["game_type"].eq("R")].copy()
    work["residual"] = work["control_success"] - work["current_blend"]
    lookup = work.groupby(KEYS, observed=True, dropna=False).agg(
        residual_sum=("residual", "sum"),
        n=("residual", "size"),
        residual_mean=("residual", "mean"),
    ).reset_index()
    lookup["raw_correction"] = lookup["residual_sum"] / (lookup["n"] + smoothing)
    lookup["scaled_correction"] = scale * lookup["raw_correction"]
    return lookup


def apply_lookup(test: pd.DataFrame, lookup: pd.DataFrame) -> np.ndarray:
    frame = test.copy()
    frame["inning_bucket"] = pd.cut(
        frame["inning"], [0, 3, 6, 9, np.inf],
        labels=["1-3", "4-6", "7-9", "10+"], right=True,
    ).astype(str)
    joined = frame.merge(
        lookup[KEYS + ["scaled_correction"]],
        on=KEYS, how="left", validate="many_to_one",
    )
    if not joined["row_id"].reset_index(drop=True).equals(
        frame["row_id"].reset_index(drop=True)
    ):
        raise RuntimeError("2025 lookup changed test row order")
    is_r = joined["game_type"].eq("R")
    correction = np.zeros(len(joined), dtype="float64")
    correction[is_r] = joined.loc[is_r, "scaled_correction"].fillna(0.0).to_numpy(float)
    return correction


def main() -> None:
    folds = load_fold_predictions()
    history = pd.concat(
        [folds[2023], folds[2024]], ignore_index=True
    )
    if set(history["season"].unique()) != {2023, 2024}:
        raise RuntimeError("Final lookup must use exactly the 2023 and 2024 OOF folds")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    sample_test = pd.read_csv(ROOT / "data" / "test.csv")
    audit = {
        "history_seasons": [2023, 2024],
        "history_rows": len(history),
        "history_r_rows": int(history["game_type"].eq("R").sum()),
        "keys": KEYS,
        "candidates": {},
    }
    for name, config in CANDIDATES.items():
        lookup = make_lookup(history, **config)
        path = ARTIFACT_DIR / f"r_context_{name}.csv"
        lookup.to_csv(path, index=False)
        correction = apply_lookup(sample_test, lookup)
        audit["candidates"][name] = {
            **config,
            "lookup_rows": len(lookup),
            "min_cell_n": int(lookup["n"].min()),
            "median_cell_n": float(lookup["n"].median()),
            "max_cell_n": int(lookup["n"].max()),
            "mean_abs_scaled_correction": float(lookup["scaled_correction"].abs().mean()),
            "max_abs_scaled_correction": float(lookup["scaled_correction"].abs().max()),
            "sample_test_nonzero": int(np.count_nonzero(correction)),
            "artifact": str(path.relative_to(ROOT)),
        }
    audit_path = ARTIFACT_DIR / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
