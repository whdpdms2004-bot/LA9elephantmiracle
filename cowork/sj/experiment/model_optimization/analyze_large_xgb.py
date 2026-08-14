from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_optuna_family import TARGET, probability_metrics


WORK_DIR = Path(__file__).resolve().parent
RUN_DIR = WORK_DIR / "large_xgb"
PREDICTION_DIR = RUN_DIR / "predictions"
OUTPUT_JSON = RUN_DIR / "large_xgb_analysis.json"
OUTPUT_BAGS = RUN_DIR / "large_xgb_seedbags.csv"
OUTPUT_BLENDS = RUN_DIR / "large_xgb_blends.csv"
BASE_OOF = WORK_DIR / "enhanced_ensemble_oof_predictions.parquet"
FOLD_WEIGHTS = {2023: 0.45, 2024: 0.55}


def canonical_candidate(name: str):
    if name in {"anchor_logloss", "anchor_brier"}:
        return "anchor"
    return name


def read_predictions():
    frames = []
    for path in sorted(PREDICTION_DIR.glob("*.parquet")):
        if not path.name.startswith(
            ("anchor_logloss_", "anchor_brier_", "moderate20_", "moderate24_diverse_")
        ):
            continue
        frame = pd.read_parquet(path)
        frame["candidate"] = frame["candidate"].map(canonical_candidate)
        # anchor_logloss seed 0 and anchor_brier seed 0 are identical; keep one.
        frame["source_file"] = path.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    duplicate = (
        data["candidate"].eq("anchor")
        & data["seed_offset"].eq(0)
        & data["source_file"].str.startswith("anchor_brier_")
    )
    return data.loc[~duplicate].drop(columns="source_file")


def metric_row(name, fold, prediction, y, **extra):
    return {"name": name, "fold": fold, **extra, **probability_metrics(y, prediction)}


def brier_metrics(y, prediction):
    target = np.asarray(y, dtype="float64")
    pred = np.asarray(prediction, dtype="float64")
    brier = float(np.mean(np.square(pred - target)))
    rate = float(target.mean())
    normalized = brier / (rate * (1.0 - rate))
    return {
        "brier": brier,
        "normalized_brier": normalized,
        "bss": max(0.0, 100000.0 * (1.0 - normalized)),
        "target_mean": rate,
        "pred_mean": float(pred.mean()),
    }


def build_seedbags(data):
    rows = []
    predictions = {}
    wanted = ["anchor", "moderate20", "moderate24_diverse"]
    for fold in sorted(data["season"].unique()):
        fold_data = data[data["season"].eq(fold)]
        for candidate in wanted:
            part = fold_data[fold_data["candidate"].eq(candidate)]
            if part.empty:
                continue
            pivot = part.pivot(index="row_id", columns="seed_offset", values="prediction")
            target = part.drop_duplicates("row_id").set_index("row_id")[TARGET]
            target = target.reindex(pivot.index).to_numpy("int8")
            seeds = sorted(int(value) for value in pivot.columns)
            for size in range(1, len(seeds) + 1):
                selected = seeds[:size]
                prediction = pivot[selected].mean(axis=1).to_numpy("float64")
                key = (fold, candidate, f"first{size}")
                predictions[key] = (pivot.index.to_numpy(), target, prediction)
                rows.append({
                    "name": candidate,
                    "fold": int(fold),
                    "bag": f"first{size}",
                    "seed_count": size,
                    "seeds": ",".join(map(str, selected)),
                    **brier_metrics(target, prediction),
                })
            prediction = pivot.mean(axis=1).to_numpy("float64")
            predictions[(fold, candidate, "all")] = (
                pivot.index.to_numpy(),
                target,
                prediction,
            )

            # Exhaustive subset audit is cheap for five seeds and exposes unstable seeds.
            for size in range(2, len(seeds) + 1):
                for selected in itertools.combinations(seeds, size):
                    prediction = pivot[list(selected)].mean(axis=1).to_numpy("float64")
                    rows.append({
                        "name": candidate,
                        "fold": int(fold),
                        "bag": "subset",
                        "seed_count": size,
                        "seeds": ",".join(map(str, selected)),
                        **brier_metrics(target, prediction),
                    })
                
    return pd.DataFrame(rows), predictions


def align(items):
    index = pd.Index(items[0][0])
    y = pd.Series(items[0][1], index=index)
    columns = []
    for row_ids, target, prediction in items:
        series = pd.Series(prediction, index=pd.Index(row_ids)).reindex(index)
        if series.isna().any():
            raise RuntimeError("Prediction alignment failed")
        check = pd.Series(target, index=pd.Index(row_ids)).reindex(index)
        if not np.array_equal(check.to_numpy(), y.to_numpy()):
            raise RuntimeError("Target alignment failed")
        columns.append(series.to_numpy("float64"))
    return y.to_numpy("int8"), np.column_stack(columns)


def best_pair_weight(y, left, right):
    direction = right - left
    denominator = float(np.dot(direction, direction))
    if denominator == 0:
        weight = 0.0
    else:
        weight = float(np.clip(np.dot(y - left, direction) / denominator, 0.0, 1.0))
    prediction = left + weight * direction
    return weight, prediction


def robust_delta(fold_metrics, anchor_metrics):
    deltas = {
        fold: fold_metrics[fold]["normalized_brier"]
        - anchor_metrics[fold]["normalized_brier"]
        for fold in fold_metrics
    }
    weighted = sum(FOLD_WEIGHTS[fold] * value for fold, value in deltas.items())
    worst = max(deltas.values())
    return float(0.80 * weighted + 0.20 * worst), deltas


def analyze_structure_blends(bag_predictions, bag_label):
    structures = ["anchor", "moderate20", "moderate24_diverse"]
    fold_arrays = {}
    anchor_metrics = {}
    for fold in FOLD_WEIGHTS:
        items = [bag_predictions[(fold, name, bag_label)] for name in structures]
        y, matrix = align(items)
        fold_arrays[fold] = (y, matrix)
        anchor_metrics[fold] = probability_metrics(y, matrix[:, 0])

    rows = []
    best = None
    grid = np.linspace(0.0, 1.0, 81)
    for w_anchor in grid:
        for w_m20 in grid:
            w_diverse = 1.0 - w_anchor - w_m20
            if w_diverse < -1e-12:
                continue
            weights = np.asarray([w_anchor, w_m20, max(0.0, w_diverse)])
            metrics = {}
            for fold, (y, matrix) in fold_arrays.items():
                metrics[fold] = brier_metrics(y, matrix @ weights)
            objective, deltas = robust_delta(metrics, anchor_metrics)
            row = {
                "kind": "three_structure_grid",
                "bag_label": bag_label,
                "anchor_weight": float(weights[0]),
                "moderate20_weight": float(weights[1]),
                "moderate24_diverse_weight": float(weights[2]),
                "robust_delta_objective": objective,
                "f23_delta_normalized_brier": deltas[2023],
                "f24_delta_normalized_brier": deltas[2024],
                "f23_brier": metrics[2023]["brier"],
                "f24_brier": metrics[2024]["brier"],
                "f24_bss": metrics[2024]["bss"],
            }
            if best is None or row["robust_delta_objective"] < best["robust_delta_objective"]:
                best = row
    rows.append(best)

    equal_weights = np.repeat(1.0 / len(structures), len(structures))
    equal_metrics = {}
    for fold, (y, matrix) in fold_arrays.items():
        equal_metrics[fold] = brier_metrics(y, matrix @ equal_weights)
    objective, deltas = robust_delta(equal_metrics, anchor_metrics)
    rows.append(
        {
            "kind": "three_structure_equal",
            "bag_label": bag_label,
            "anchor_weight": equal_weights[0],
            "moderate20_weight": equal_weights[1],
            "moderate24_diverse_weight": equal_weights[2],
            "robust_delta_objective": objective,
            "f23_delta_normalized_brier": deltas[2023],
            "f24_delta_normalized_brier": deltas[2024],
            "f23_brier": equal_metrics[2023]["brier"],
            "f24_brier": equal_metrics[2024]["brier"],
            "f24_bss": equal_metrics[2024]["bss"],
        }
    )
    return pd.DataFrame(rows), fold_arrays


def analyze_outer_blends(bag_predictions, structure_blend, fold_arrays, bag_label):
    base = pd.read_parquet(BASE_OOF)
    rows = []
    for fold, track in [(2023, "robust"), (2024, "performance")]:
        base_fold = base[base["season"].eq(fold) & base["track"].eq(track)]
        base_series = base_fold.set_index("row_id")["prediction"]
        for candidate in ["anchor", "moderate20", "moderate24_diverse"]:
            row_ids, y, prediction = bag_predictions[(fold, candidate, bag_label)]
            base_prediction = base_series.reindex(row_ids).to_numpy("float64")
            weight, blend = best_pair_weight(y, base_prediction, prediction)
            rows.append(
                metric_row(
                    candidate,
                    fold,
                    blend,
                    y,
                    kind="base_plus_structure",
                    bag_label=bag_label,
                    base_track=track,
                    insight_weight=weight,
                )
            )

        y, matrix = fold_arrays[fold]
        weights = np.asarray(
            [
                structure_blend["anchor_weight"],
                structure_blend["moderate20_weight"],
                structure_blend["moderate24_diverse_weight"],
            ]
        )
        prediction = matrix @ weights
        # fold_arrays use the anchor row order.
        row_ids = bag_predictions[(fold, "anchor", bag_label)][0]
        base_prediction = base_series.reindex(row_ids).to_numpy("float64")
        weight, blend = best_pair_weight(y, base_prediction, prediction)
        rows.append(
            metric_row(
                "large_structure_blend",
                fold,
                blend,
                y,
                kind="base_plus_structure",
                bag_label=bag_label,
                base_track=track,
                insight_weight=weight,
            )
        )
    return pd.DataFrame(rows)


def main():
    data = read_predictions()
    seedbags, bag_predictions = build_seedbags(data)
    blend_frames = []
    outer_frames = []
    best_by_bag = {}
    for bag_label in ["first1", "all"]:
        structure_blends, fold_arrays = analyze_structure_blends(
            bag_predictions, bag_label
        )
        best = structure_blends.sort_values("robust_delta_objective").iloc[0].to_dict()
        best_by_bag[bag_label] = best
        blend_frames.append(structure_blends)
        outer_frames.append(
            analyze_outer_blends(
                bag_predictions, best, fold_arrays, bag_label
            )
        )
    structure_blends = pd.concat(blend_frames, ignore_index=True)
    outer = pd.concat(outer_frames, ignore_index=True)

    result_rows = pd.read_csv(RUN_DIR / "large_xgb_results.csv")
    result_rows["candidate"] = result_rows["candidate"].map(canonical_candidate)
    size_summary = (
        result_rows[result_rows["fold"].eq(2024)]
        .groupby("candidate", as_index=False)
        .agg(
            validation_model_count=("model_size_bytes", "count"),
            validation_model_size_mb=("model_size_mb", "sum"),
            mean_tree_count=("tree_count", "mean"),
            mean_predict_sec=("predict_elapsed_sec", "mean"),
        )
    )
    correlation = {}
    for fold, (_, matrix) in fold_arrays.items():
        correlation[str(fold)] = pd.DataFrame(
            matrix, columns=["anchor", "moderate20", "moderate24_diverse"]
        ).corr().to_dict()

    OUTPUT_BAGS.parent.mkdir(parents=True, exist_ok=True)
    seedbags.to_csv(OUTPUT_BAGS, index=False)
    pd.concat([structure_blends, outer], ignore_index=True).to_csv(OUTPUT_BLENDS, index=False)
    summary = {
        "selection_rule": "Minimize 0.8 * weighted F23/F24 delta normalized Brier + 0.2 * worst-fold delta versus the five-seed anchor.",
        "best_structure_blend": best_by_bag,
        "outer_blends": outer.to_dict(orient="records"),
        "size_summary": size_summary.to_dict(orient="records"),
        "prediction_correlation": correlation,
        "artifacts": {
            "seedbags": str(OUTPUT_BAGS),
            "blends": str(OUTPUT_BLENDS),
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
