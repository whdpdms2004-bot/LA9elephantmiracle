"""C1 잔차의 투수 이력 구간 경계를 바꿔 cross-season 전이를 비교한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_bucketed_residual import (
    COMPONENT_WEIGHTS, EPS, OUT, SINGLE, base_prediction, load, logit,
    metrics, prediction_path, sigmoid,
)
from harness import CACHE, TARGET


SCALES = np.arange(0.0, 0.401, 0.025)


def main() -> None:
    df = load()
    train_n = df.loc[df["season"].lt(2023), "asof_pitcher_n"].to_numpy(np.float64)
    quantile_cuts = sorted(set(np.quantile(
        train_n[np.isfinite(train_n)], [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95]
    ).round().astype(int).tolist()))
    schemes = {
        "production5": [100, 500, 2000, 4000],
        "log8": [50, 100, 250, 500, 1000, 2000, 4000, 8000],
        "log7": [50, 200, 500, 1000, 2000, 4000, 8000],
        "volume5": [250, 1000, 3000, 6000],
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
        base_bucket = np.digitize(n, [100, 500, 2000, 4000])
        weight = COMPONENT_WEIGHTS[base_bucket]
        current = np.clip((1 - weight) * base + weight * p0, EPS, 1 - EPS)
        p1_start = np.clip((1 - weight) * base + weight * p1, EPS, 1 - EPS)
        direction = (
            logit(np.load(prediction_path("C1", fold)))
            - logit(np.load(prediction_path("B0", fold))))
        raw[fold] = {
            "y": y, "n": n, "current": current, "p1_start": p1_start,
            "direction": direction, "null": y.mean() * (1 - y.mean()),
        }

    reports = []
    for name, cuts in schemes.items():
        tables = {}
        buckets = {}
        for fold, data in raw.items():
            bucket = np.digitize(data["n"], cuts)
            buckets[fold] = bucket
            table = np.zeros((len(cuts) + 1, len(SCALES)))
            for bucket_id in range(len(cuts) + 1):
                mask = bucket == bucket_id
                ref_sse = np.square(
                    data["current"][mask] - data["y"][mask]).sum()
                for scale_id, scale in enumerate(SCALES):
                    pred = sigmoid(
                        logit(data["p1_start"][mask])
                        + scale * data["direction"][mask])
                    sse = np.square(pred - data["y"][mask]).sum()
                    table[bucket_id, scale_id] = (
                        100000.0 * (ref_sse - sse)
                        / (len(data["y"]) * data["null"]))
            tables[fold] = table

        selections = {}
        for method in ("select2023", "select2024", "robust_local"):
            ids = []
            for bucket_id in range(len(cuts) + 1):
                candidates = []
                for scale_id in range(len(SCALES)):
                    a = tables[2023][bucket_id, scale_id]
                    b = tables[2024][bucket_id, scale_id]
                    if method == "select2023":
                        key = (a, b)
                    elif method == "select2024":
                        key = (b, a)
                    else:
                        key = (min(a, b), a + b)
                    candidates.append(key)
                ids.append(max(range(len(SCALES)), key=lambda i: candidates[i]))
            deltas = {
                fold: float(sum(tables[fold][bucket_id, ids[bucket_id]]
                                for bucket_id in range(len(ids))))
                for fold in (2023, 2024)
            }
            selections[method] = {
                "scales": [float(SCALES[i]) for i in ids],
                "deltas": deltas,
                "worst": min(deltas.values()),
                "sum": sum(deltas.values()),
            }
        reports.append({
            "scheme": name,
            "cuts": cuts,
            "bucket_rows": {
                str(fold): np.bincount(bucket, minlength=len(cuts) + 1).tolist()
                for fold, bucket in buckets.items()
            },
            "selections": selections,
        })

    ranked = sorted(
        reports,
        key=lambda item: (
            item["selections"]["select2024"]["worst"],
            item["selections"]["select2024"]["sum"]),
        reverse=True,
    )
    path = OUT / "bucket_scheme_comparison.json"
    path.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([
        {
            "scheme": item["scheme"],
            "cuts": item["cuts"],
            "select2024": item["selections"]["select2024"],
            "robust_local": item["selections"]["robust_local"],
        }
        for item in ranked
    ], ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
