"""고정 투수 이력 구간마다 P1 성분 비중과 C1 잔차 비중을 공동 검증한다."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_bucketed_residual import (
    COMPONENT_WEIGHTS, EPS, OUT, base_prediction, load, logit,
    prediction_path, sigmoid,
)
from harness import CACHE, TARGET


COMPONENT_SCALES = np.array([0.50, 0.75, 1.00, 1.25])
RESIDUAL_SCALES = np.arange(0.0, 0.401, 0.025)


def main() -> None:
    df = load()
    train_n = df.loc[df["season"].lt(2023), "asof_pitcher_n"].to_numpy(np.float64)
    quantile_cuts = sorted(set(np.quantile(
        train_n[np.isfinite(train_n)], [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95]
    ).round().astype(int).tolist()))
    schemes = {
        "log8": [50, 100, 250, 500, 1000, 2000, 4000, 8000],
        "train_quantile8": quantile_cuts,
    }
    raw = {}
    for fold in (2023, 2024):
        valid = df["season"].eq(fold).to_numpy()
        y = df.loc[valid, TARGET].to_numpy(np.float64)
        n = df.loc[valid, "asof_pitcher_n"].to_numpy(np.float64)
        base = np.clip(base_prediction(df, fold), EPS, 1 - EPS)
        p0 = np.load(CACHE / f"v75_confirm_P0_{fold}.npy")
        p1 = np.load(CACHE / f"v75_confirm_P1_{fold}.npy")
        production_bucket = np.digitize(n, [100, 500, 2000, 4000])
        weight = COMPONENT_WEIGHTS[production_bucket]
        current = np.clip((1 - weight) * base + weight * p0, EPS, 1 - EPS)
        direction = (
            logit(np.load(prediction_path("C1", fold)))
            - logit(np.load(prediction_path("B0", fold))))
        raw[fold] = {
            "y": y, "n": n, "base": base, "p1": p1, "weight": weight,
            "current": current, "direction": direction,
            "null": y.mean() * (1 - y.mean()),
        }

    reports = []
    combinations = list(itertools.product(
        range(len(COMPONENT_SCALES)), range(len(RESIDUAL_SCALES))))
    for scheme, cuts in schemes.items():
        tables = {}
        for fold, data in raw.items():
            bucket = np.digitize(data["n"], cuts)
            table = np.zeros((len(cuts) + 1, len(combinations)))
            for bucket_id in range(len(cuts) + 1):
                mask = bucket == bucket_id
                ref_sse = np.square(
                    data["current"][mask] - data["y"][mask]).sum()
                for combo_id, (component_id, residual_id) in enumerate(combinations):
                    component_weight = np.clip(
                        COMPONENT_SCALES[component_id] * data["weight"][mask],
                        0.0, 0.95)
                    start = np.clip(
                        (1 - component_weight) * data["base"][mask]
                        + component_weight * data["p1"][mask], EPS, 1 - EPS)
                    pred = sigmoid(
                        logit(start)
                        + RESIDUAL_SCALES[residual_id]
                        * data["direction"][mask])
                    sse = np.square(pred - data["y"][mask]).sum()
                    table[bucket_id, combo_id] = (
                        100000.0 * (ref_sse - sse)
                        / (len(data["y"]) * data["null"]))
            tables[fold] = table

        selections = {}
        for method in ("select2023", "select2024", "robust_local"):
            ids = []
            for bucket_id in range(len(cuts) + 1):
                def key(combo_id: int):
                    a = tables[2023][bucket_id, combo_id]
                    b = tables[2024][bucket_id, combo_id]
                    if method == "select2023":
                        return a, b
                    if method == "select2024":
                        return b, a
                    return min(a, b), a + b
                ids.append(max(range(len(combinations)), key=key))
            deltas = {
                fold: float(sum(tables[fold][bucket_id, ids[bucket_id]]
                                for bucket_id in range(len(ids))))
                for fold in (2023, 2024)
            }
            selections[method] = {
                "component_scales": [
                    float(COMPONENT_SCALES[combinations[i][0]]) for i in ids],
                "residual_scales": [
                    float(RESIDUAL_SCALES[combinations[i][1]]) for i in ids],
                "deltas": deltas,
                "worst": min(deltas.values()),
                "sum": sum(deltas.values()),
            }
        reports.append({"scheme": scheme, "cuts": cuts, "selections": selections})

    reports.sort(
        key=lambda item: (
            item["selections"]["select2024"]["worst"],
            item["selections"]["select2024"]["sum"]),
        reverse=True)
    path = OUT / "joint_bucket_weights.json"
    path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
