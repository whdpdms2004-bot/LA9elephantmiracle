"""2022에서만 C1 잔차 구간 가중치를 정해 2023/2024로 순방향 검증한다."""
from __future__ import annotations

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


SCALES = np.arange(0.0, 0.401, 0.025)


def main() -> None:
    df = load()
    train_n = df.loc[df["season"].lt(2022), "asof_pitcher_n"].to_numpy(np.float64)
    quantile_cuts = sorted(set(np.quantile(
        train_n[np.isfinite(train_n)], [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95]
    ).round().astype(int).tolist()))
    schemes = {
        "production5": [100, 500, 2000, 4000],
        "log8": [50, 100, 250, 500, 1000, 2000, 4000, 8000],
        "train2021_quantile8": quantile_cuts,
    }
    raw = {}
    for fold in (2022, 2023, 2024):
        valid = df["season"].eq(fold).to_numpy()
        y = df.loc[valid, TARGET].to_numpy(np.float64)
        n = df.loc[valid, "asof_pitcher_n"].to_numpy(np.float64)
        base = np.clip(base_prediction(df, fold), EPS, 1 - EPS)
        p0 = np.load(CACHE / f"v75_confirm_P0_{fold}.npy")
        p1 = np.load(CACHE / f"v75_confirm_P1_{fold}.npy")
        production_bucket = np.digitize(n, [100, 500, 2000, 4000])
        weight = COMPONENT_WEIGHTS[production_bucket]
        current = np.clip((1 - weight) * base + weight * p0, EPS, 1 - EPS)
        future_start = np.clip((1 - weight) * base + weight * p1, EPS, 1 - EPS)
        direction = (
            logit(np.load(prediction_path("C1", fold)))
            - logit(np.load(prediction_path("B0", fold))))
        raw[fold] = {
            "y": y, "n": n, "current": current, "future_start": future_start,
            "direction": direction, "null": y.mean() * (1 - y.mean()),
        }

    reports = []
    for scheme, cuts in schemes.items():
        selected_ids = []
        bucket_2022 = np.digitize(raw[2022]["n"], cuts)
        for bucket_id in range(len(cuts) + 1):
            mask = bucket_2022 == bucket_id
            ref_sse = np.square(
                raw[2022]["current"][mask] - raw[2022]["y"][mask]).sum()
            gains = []
            for scale in SCALES:
                pred = sigmoid(
                    logit(raw[2022]["current"][mask])
                    + scale * raw[2022]["direction"][mask])
                gains.append(ref_sse - np.square(pred - raw[2022]["y"][mask]).sum())
            selected_ids.append(int(np.argmax(gains)))
        scales = np.asarray([SCALES[i] for i in selected_ids])

        deltas = {}
        detail = {}
        for fold, data in raw.items():
            bucket = np.digitize(data["n"], cuts)
            start = data["current"] if fold == 2022 else data["future_start"]
            pred = sigmoid(logit(start) + scales[bucket] * data["direction"])
            ref_sse = np.square(data["current"] - data["y"]).sum()
            sse = np.square(pred - data["y"]).sum()
            delta = 100000.0 * (ref_sse - sse) / (len(data["y"]) * data["null"])
            deltas[fold] = float(delta)
            detail[fold] = {
                "rows": np.bincount(bucket, minlength=len(cuts) + 1).tolist(),
                "pred_mean": float(pred.mean()),
            }
        reports.append({
            "scheme": scheme,
            "cuts": cuts,
            "selected_on": 2022,
            "residual_scales": scales.tolist(),
            "deltas_vs_current_p0": deltas,
            "future_worst": min(deltas[2023], deltas[2024]),
            "future_sum": deltas[2023] + deltas[2024],
            "detail": detail,
        })

    reports.sort(key=lambda item: (item["future_worst"], item["future_sum"]),
                 reverse=True)
    path = OUT / "forward_bucket_selection.json"
    path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
