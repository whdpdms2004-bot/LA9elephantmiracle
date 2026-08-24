from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_optuna_family import probability_metrics


WORK_DIR = Path(__file__).resolve().parent
RUN_DIR = WORK_DIR / "large_xgb"
PREDICTION_DIR = RUN_DIR / "predictions"
BASE_OOF = WORK_DIR / "enhanced_ensemble_oof_predictions.parquet"
OUTPUT_CSV = RUN_DIR / "large_xgb_calibration.csv"
OUTPUT_JSON = RUN_DIR / "large_xgb_calibration.json"
EPS = 1e-6
FOLD_WEIGHTS = {2023: 0.45, 2024: 0.55}


def logit(values):
    p = np.clip(np.asarray(values, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(values):
    z = np.clip(np.asarray(values, dtype="float64"), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def read_one(candidate, fold, seed=0):
    aliases = [candidate]
    if candidate == "anchor":
        aliases = ["anchor_logloss", "anchor_brier"]
    for alias in aliases:
        path = PREDICTION_DIR / f"{alias}_f{fold}_s{seed}.parquet"
        if path.is_file():
            return pd.read_parquet(path).sort_values("row_id").reset_index(drop=True)
    raise FileNotFoundError((candidate, fold, seed))


def quantile_match(source, reference):
    source = np.asarray(source, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    order = np.argsort(source, kind="mergesort")
    sorted_reference = np.sort(reference)
    output = np.empty_like(source)
    output[order] = sorted_reference
    return output


def transforms(source, reference):
    source = np.asarray(source, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    source_logit = logit(source)
    reference_logit = logit(reference)
    output = {
        "raw": source,
        "prob_mean": source - source.mean() + reference.mean(),
        "prob_mean_std": (
            (source - source.mean())
            * (reference.std() / max(source.std(), EPS))
            + reference.mean()
        ),
        "logit_mean": sigmoid(
            source_logit - source_logit.mean() + reference_logit.mean()
        ),
        "logit_mean_std": sigmoid(
            (source_logit - source_logit.mean())
            * (reference_logit.std() / max(source_logit.std(), EPS))
            + reference_logit.mean()
        ),
        "quantile": quantile_match(source, reference),
    }
    return {name: np.clip(value, EPS, 1.0 - EPS) for name, value in output.items()}


def brier_ratio(y, prediction):
    y = np.asarray(y, dtype="float64")
    brier = float(np.mean(np.square(np.asarray(prediction) - y)))
    return brier / (float(y.mean()) * (1.0 - float(y.mean())))


def best_pair_weight(y, left, right):
    direction = right - left
    denominator = float(np.dot(direction, direction))
    weight = 0.0 if denominator == 0 else float(
        np.clip(np.dot(y - left, direction) / denominator, 0.0, 1.0)
    )
    return weight, left + weight * direction


def main():
    fold_data = {}
    rows = []
    for fold in [2023, 2024]:
        anchor = read_one("anchor", fold)
        diverse = read_one("moderate24_diverse", fold)
        if not anchor["row_id"].equals(diverse["row_id"]):
            raise RuntimeError("Prediction row alignment failed")
        y = anchor["control_success"].to_numpy("int8")
        anchor_prediction = anchor["prediction"].to_numpy("float64")
        transformed = transforms(
            diverse["prediction"].to_numpy("float64"), anchor_prediction
        )
        fold_data[fold] = (anchor["row_id"].to_numpy(), y, anchor_prediction, transformed)
        for name, prediction in transformed.items():
            rows.append(
                {
                    "kind": "transformed_single",
                    "transform": name,
                    "fold": fold,
                    "large_weight": 1.0,
                    **probability_metrics(y, prediction),
                }
            )

    anchor_ratios = {
        fold: brier_ratio(values[1], values[2]) for fold, values in fold_data.items()
    }
    robust_grid = []
    for transform in next(iter(fold_data.values()))[3]:
        for weight in np.linspace(0.0, 1.0, 201):
            ratios = {}
            briers = {}
            for fold, (_, y, anchor, transformed) in fold_data.items():
                prediction = anchor + weight * (transformed[transform] - anchor)
                briers[fold] = float(np.mean(np.square(prediction - y)))
                rate = float(np.mean(y))
                ratios[fold] = briers[fold] / (rate * (1.0 - rate))
            deltas = {fold: ratios[fold] - anchor_ratios[fold] for fold in ratios}
            weighted = sum(FOLD_WEIGHTS[fold] * deltas[fold] for fold in deltas)
            objective = 0.80 * weighted + 0.20 * max(deltas.values())
            robust_grid.append(
                {
                    "transform": transform,
                    "large_weight": float(weight),
                    "objective": float(objective),
                    "f23_delta": deltas[2023],
                    "f24_delta": deltas[2024],
                    "f23_brier": briers[2023],
                    "f24_brier": briers[2024],
                    "f24_bss": max(0.0, 100000.0 * (1.0 - ratios[2024])),
                }
            )
    robust = pd.DataFrame(robust_grid)
    best_by_transform = robust.loc[
        robust.groupby("transform")["objective"].idxmin()
    ].sort_values("objective")

    base = pd.read_parquet(BASE_OOF)
    outer_rows = []
    for record in best_by_transform.to_dict(orient="records"):
        transform = record["transform"]
        large_weight = float(record["large_weight"])
        for fold, track in [(2023, "robust"), (2024, "performance")]:
            row_ids, y, anchor, transformed = fold_data[fold]
            insight = anchor + large_weight * (transformed[transform] - anchor)
            base_prediction = (
                base[base["season"].eq(fold) & base["track"].eq(track)]
                .set_index("row_id")["prediction"]
                .reindex(row_ids)
                .to_numpy("float64")
            )
            insight_weight, prediction = best_pair_weight(
                y, base_prediction, insight
            )
            outer_rows.append(
                {
                    "kind": "base_plus_transformed_large",
                    "transform": transform,
                    "fold": fold,
                    "large_weight": large_weight,
                    "insight_weight": insight_weight,
                    **probability_metrics(y, prediction),
                }
            )

    output = pd.concat(
        [pd.DataFrame(rows), best_by_transform.assign(kind="robust_best"), pd.DataFrame(outer_rows)],
        ignore_index=True,
        sort=False,
    )
    output.to_csv(OUTPUT_CSV, index=False)
    summary = {
        "rule": "All transforms use only the large-model and anchor prediction distributions; no target labels are used in the transform.",
        "best_by_transform": best_by_transform.to_dict(orient="records"),
        "outer_results": pd.DataFrame(outer_rows).to_dict(orient="records"),
        "artifacts": {"results": str(OUTPUT_CSV)},
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
