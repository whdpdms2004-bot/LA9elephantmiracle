"""구간별 C1 후보 위에 여러 모델/피처 잔차가 더해지는지 비교한다."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_bucketed_residual import (
    COMPONENT_WEIGHTS, CUTS, EPS, SINGLE, base_prediction, load, logit,
    metrics, prediction_path, sigmoid,
)

from harness import CACHE, TARGET


GPU_PREFIX = "confirm_xgboost_v2r200_tm500_robust_cuda_efull"
GPU_SEEDS = (20260818, 20260819, 20260820)
CAT = HERE / "outputs" / "single_catboost"
OUT = HERE / "outputs" / "combined"
SECONDARY_SCALES = np.arange(-0.20, 0.201, 0.025)


def cpu_direction(arm: str, fold: int) -> np.ndarray:
    return (logit(np.load(prediction_path(arm, fold)))
            - logit(np.load(prediction_path("B0", fold))))


def gpu_direction(arm: str, fold: int) -> np.ndarray:
    directions = []
    for seed in GPU_SEEDS:
        directions.append(
            logit(np.load(SINGLE / f"{GPU_PREFIX}_s{seed}_{arm}_{fold}.npy"))
            - logit(np.load(SINGLE / f"{GPU_PREFIX}_s{seed}_B0_{fold}.npy")))
    return np.mean(directions, axis=0)


def cat_direction(arm: str, fold: int) -> np.ndarray:
    return (logit(np.load(CAT / f"{arm}_{fold}.npy"))
            - logit(np.load(CAT / f"B0_{fold}.npy")))


def main() -> None:
    bucket_report = json.loads(
        (OUT / "bucketed_c1_residual.json").read_text(encoding="utf-8"))
    selected = bucket_report["best"]
    df = load()
    fold_data = {}
    for fold in (2023, 2024):
        valid = df["season"].eq(fold).to_numpy()
        y = df.loc[valid, TARGET].to_numpy(np.float64)
        bucket = np.digitize(
            df.loc[valid, "asof_pitcher_n"].to_numpy(np.float64), CUTS)
        base = np.clip(base_prediction(df, fold), EPS, 1 - EPS)
        p0 = np.load(CACHE / f"v75_confirm_P0_{fold}.npy")
        p1 = np.load(CACHE / f"v75_confirm_P1_{fold}.npy")
        weight = COMPONENT_WEIGHTS[bucket]
        current = np.clip((1 - weight) * base + weight * p0, EPS, 1 - EPS)
        component_weight = np.clip(
            float(selected["component_scale"]) * weight, 0.0, 0.95)
        p1_start = np.clip(
            (1 - component_weight) * base + component_weight * p1, EPS, 1 - EPS)
        c1 = cpu_direction("C1", fold)
        row_scale = np.asarray(selected["residual_scales"])[bucket]
        primary = sigmoid(logit(p1_start) + row_scale * c1)
        fold_data[fold] = {
            "y": y,
            "current": current,
            "primary": primary,
            "directions": {
                "cpu_c0": cpu_direction("C0", fold),
                "cpu_c2": cpu_direction("C2", fold),
                "cpu_c3": cpu_direction("C3", fold),
                "cpu_c4": cpu_direction("C4", fold),
                "cpu_k6": cpu_direction("K6", fold),
                "cpu_h6": cpu_direction("H6", fold),
                "gpu3_c1": gpu_direction("C1", fold),
                "gpu3_k6": gpu_direction("K6", fold),
                "gpu3_h6": gpu_direction("H6", fold),
                "cat_c0": cat_direction("C0", fold),
                "cat_c1": cat_direction("C1", fold),
            },
        }

    def deltas(option_scales: dict[str, float]) -> dict[int, float]:
        result = {}
        for fold, data in fold_data.items():
            z = logit(data["primary"]).copy()
            for option, scale in option_scales.items():
                z += scale * data["directions"][option]
            pred = sigmoid(z)
            result[fold] = float(
                metrics(data["y"], pred)["bss_raw"]
                - metrics(data["y"], data["current"])["bss_raw"])
        return result

    primary_deltas = deltas({})
    single = []
    options = list(fold_data[2023]["directions"])
    for option in options:
        best = None
        for scale in SECONDARY_SCALES:
            ds = deltas({option: float(scale)})
            score = (min(ds.values()), sum(ds.values()))
            if best is None or score > best["score"]:
                best = {"option": option, "scale": float(scale),
                        "deltas": ds, "score": score}
        single.append(best)
    single.sort(key=lambda item: item["score"], reverse=True)

    pair = []
    pair_scales = np.arange(-0.10, 0.151, 0.025)
    for first, second in itertools.combinations(
            [item["option"] for item in single[:5]], 2):
        best = None
        for first_scale, second_scale in itertools.product(pair_scales, repeat=2):
            ds = deltas({first: float(first_scale), second: float(second_scale)})
            score = (min(ds.values()), sum(ds.values()))
            if best is None or score > best["score"]:
                best = {
                    "options": {first: float(first_scale), second: float(second_scale)},
                    "deltas": ds, "score": score,
                }
        pair.append(best)
    pair.sort(key=lambda item: item["score"], reverse=True)

    report = {
        "primary_deltas": primary_deltas,
        "best_single": single,
        "best_pairs": pair,
    }
    path = OUT / "secondary_residuals.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "primary": primary_deltas,
        "top_single": single[:5],
        "top_pairs": pair[:5],
    }, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
