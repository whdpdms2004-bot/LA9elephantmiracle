"""Evaluate R-only TabM OOF predictions against fixed tree anchors.

The report contains only validation seasons 2022/2023.  It deliberately does
not read 2024 predictions.  Blend weights are screened on a small fixed grid,
and cross-fold transfer is reported to expose unstable gains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiment/model_optimization/tabm_context"),
    )
    parser.add_argument(
        "--tabm-runs",
        nargs="+",
        default=["r_t0_k16_d384", "r_t1_k16_d384", "r_t2_k16_d384"],
    )
    parser.add_argument(
        "--tree-models",
        nargs="+",
        default=["cat_robust_t69_seedbag3", "cat_robust_t40_seedbag3"],
    )
    return parser.parse_args()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(np.float64, copy=False)
    p = p.astype(np.float64, copy=False)
    rate = float(y.mean())
    brier = float(np.square(p - y).mean())
    return {
        "n": int(len(y)),
        "target_mean": rate,
        "prediction_mean": float(p.mean()),
        "brier": brier,
        "bss": float(1e5 * (1.0 - brier / (rate * (1.0 - rate)))),
    }


def load_tree_anchor(model: str, folds: list[int]) -> pd.DataFrame:
    root = Path("experiment/model_optimization/enhanced_seed_oof_parts")
    parts = []
    for fold in folds:
        path = root / f"{model}_fold{fold}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=["row_id", "season", "control_success", "prediction"])
        parts.append(part.rename(columns={"prediction": "tree_prediction"}))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    folds = [2022, 2023]
    weights = np.round(np.arange(0.0, 0.51, 0.05), 2)
    rows: list[dict] = []
    transfer_rows: list[dict] = []

    for tabm_name in args.tabm_runs:
        tabm_path = args.root / "outputs" / tabm_name / "oof_all.parquet"
        tabm = pd.read_parquet(tabm_path).rename(columns={"prediction": "tabm_prediction"})
        if set(tabm["season"].unique()) - set(folds):
            raise ValueError(f"Forbidden fold in {tabm_path}")
        if set(tabm["game_type"].astype(str).unique()) != {"R"}:
            raise ValueError(f"Expected R-only OOF in {tabm_path}")

        for tree_name in args.tree_models:
            tree = load_tree_anchor(tree_name, folds)
            joined = tabm.merge(
                tree,
                on=["row_id", "season", "control_success"],
                how="inner",
                validate="one_to_one",
            )
            if len(joined) != len(tabm):
                raise ValueError(f"OOF join loss: tabm={len(tabm)} joined={len(joined)}")

            fold_grid: dict[int, list[dict]] = {}
            for fold in folds:
                part = joined[joined["season"].eq(fold)]
                y = part["control_success"].to_numpy(dtype=np.float64)
                pt = part["tree_prediction"].to_numpy(dtype=np.float64)
                pm = part["tabm_prediction"].to_numpy(dtype=np.float64)
                prediction_corr = float(np.corrcoef(pt, pm)[0, 1])
                error_corr = float(np.corrcoef(pt - y, pm - y)[0, 1])
                fold_grid[fold] = []
                for weight in weights:
                    pred = (1.0 - weight) * pt + weight * pm
                    record = {
                        "tabm": tabm_name,
                        "tree": tree_name,
                        "fold": fold,
                        "tabm_weight": float(weight),
                        "prediction_correlation": prediction_corr,
                        "error_correlation": error_corr,
                        **metrics(y, pred),
                    }
                    rows.append(record)
                    fold_grid[fold].append(record)

            # Honest stability diagnostic: choose the weight on one fold and apply
            # it untouched to the other fold.
            for selection_fold, application_fold in [(2022, 2023), (2023, 2022)]:
                selected = min(fold_grid[selection_fold], key=lambda x: x["brier"])
                applied = next(
                    x
                    for x in fold_grid[application_fold]
                    if x["tabm_weight"] == selected["tabm_weight"]
                )
                anchor = next(x for x in fold_grid[application_fold] if x["tabm_weight"] == 0.0)
                transfer_rows.append(
                    {
                        "tabm": tabm_name,
                        "tree": tree_name,
                        "selection_fold": selection_fold,
                        "application_fold": application_fold,
                        "selected_tabm_weight": selected["tabm_weight"],
                        "selection_bss": selected["bss"],
                        "application_bss": applied["bss"],
                        "anchor_application_bss": anchor["bss"],
                        "application_delta_bss": applied["bss"] - anchor["bss"],
                    }
                )

    grid = pd.DataFrame(rows)
    transfer = pd.DataFrame(transfer_rows)
    output = args.root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output / "tabm_tree_blend_grid.csv", index=False)
    transfer.to_csv(output / "tabm_tree_blend_transfer.csv", index=False)

    mean_grid = (
        grid.groupby(["tabm", "tree", "tabm_weight"], as_index=False)
        .agg(mean_bss=("bss", "mean"), mean_brier=("brier", "mean"))
        .sort_values(["mean_bss"], ascending=False)
    )
    mean_grid.to_csv(output / "tabm_tree_blend_mean.csv", index=False)
    stable = (
        transfer.groupby(["tabm", "tree", "selected_tabm_weight"], as_index=False)
        .agg(
            mean_transfer_delta_bss=("application_delta_bss", "mean"),
            min_transfer_delta_bss=("application_delta_bss", "min"),
        )
        .sort_values(["min_transfer_delta_bss", "mean_transfer_delta_bss"], ascending=False)
    )
    stable.to_csv(output / "tabm_tree_blend_stability.csv", index=False)

    summary = {
        "selection_folds": folds,
        "val2024_read": False,
        "best_mean": mean_grid.head(10).to_dict(orient="records"),
        "cross_fold_transfer": transfer.to_dict(orient="records"),
        "stability": stable.to_dict(orient="records"),
    }
    (output / "tabm_tree_blend_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nBest mean-fold blends")
    print(mean_grid.head(15).to_string(index=False))
    print("\nCross-fold transfer")
    print(transfer.to_string(index=False))


if __name__ == "__main__":
    main()
