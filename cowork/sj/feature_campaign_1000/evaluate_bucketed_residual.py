"""투수 과거 투구 수 구간별 C1 잔차 강도를 고정해 두 forward fold에서 검증한다.

구간과 가중치는 검증 데이터에서만 고르고 최종 추론에서는 한 행의
asof_pitcher_n만 참조한다. 따라서 test 행 간 집계나 분포 의존성이 없다.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SJ = HERE.parent
SRC = SJ / "claude" / "src"
sys.path.insert(0, str(SRC))
from harness import CACHE, TARGET, load, metrics

MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SINGLE = HERE / "outputs" / "single_xgb"
OUT = HERE / "outputs" / "combined"
PREFIX = "confirm_xgboost_v2r200_tm500_robust_cpu_efull"
CUTS = np.array([100, 500, 2000, 4000])
COMPONENT_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
SCALES = np.arange(0.0, 0.401, 0.05)
EPS = 1e-7


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, EPS, 1 - EPS)
    return np.log(values / (1 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def base_prediction(df: pd.DataFrame, fold: int) -> np.ndarray:
    ids = df.loc[df["season"].eq(fold), "row_id"].to_numpy()
    if fold == 2024:
        return (pd.read_parquet(PROD).set_index("row_id").reindex(ids)
                ["submit021_reverse20_s040_tabm"].to_numpy(np.float64))
    names = sorted({
        re.match(r"(.+)_fold\d{4}\.parquet", path.name).group(1)
        for path in OOF_DIR.glob("*.parquet")
    })
    values = [
        pd.read_parquet(OOF_DIR / f"{name}_fold{fold}.parquet")
        .set_index("row_id").reindex(ids)["prediction"].to_numpy(np.float64)
        for name in names if (OOF_DIR / f"{name}_fold{fold}.parquet").exists()
    ]
    return np.mean(values, axis=0)


def prediction_path(arm: str, fold: int) -> Path:
    for path in (
        SINGLE / f"{PREFIX}_scampaign_{arm}_{fold}.npy",
        SINGLE / f"{PREFIX}_{arm}_{fold}.npy",
    ):
        if path.exists():
            return path
    raise FileNotFoundError(arm, fold)


def main() -> None:
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
        b0 = np.load(prediction_path("B0", fold))
        c1 = np.load(prediction_path("C1", fold))
        direction = logit(c1) - logit(b0)
        weight = COMPONENT_WEIGHTS[bucket]
        current = np.clip((1 - weight) * base + weight * p0, EPS, 1 - EPS)
        fold_data[fold] = {
            "y": y, "bucket": bucket, "base": base, "p1": p1,
            "direction": direction, "weight": weight, "current": current,
            "null": y.mean() * (1 - y.mean()),
        }

    best = None
    transfer = {}
    for component_scale in (0.50, 0.75, 1.00, 1.25):
        gains = {}
        for fold, data in fold_data.items():
            component_weight = np.clip(
                component_scale * data["weight"], 0.0, 0.95)
            start = np.clip(
                (1 - component_weight) * data["base"]
                + component_weight * data["p1"], EPS, 1 - EPS)
            table = np.zeros((5, len(SCALES)), dtype=np.float64)
            for bucket_id in range(5):
                mask = data["bucket"] == bucket_id
                reference_sse = np.square(
                    data["current"][mask] - data["y"][mask]).sum()
                for scale_id, scale in enumerate(SCALES):
                    pred = sigmoid(
                        logit(start[mask]) + scale * data["direction"][mask])
                    candidate_sse = np.square(pred - data["y"][mask]).sum()
                    table[bucket_id, scale_id] = (
                        100000.0 * (reference_sse - candidate_sse)
                        / (len(data["y"]) * data["null"]))
            gains[fold] = table

        component_best = {2023: None, 2024: None}
        for ids in itertools.product(range(len(SCALES)), repeat=5):
            deltas = {
                fold: float(sum(gains[fold][b, ids[b]] for b in range(5)))
                for fold in (2023, 2024)
            }
            score = (min(deltas.values()), sum(deltas.values()))
            item = {
                "component_scale": component_scale,
                "scale_ids": ids,
                "scales": [float(SCALES[i]) for i in ids],
                "deltas": deltas,
                "score": score,
            }
            if best is None or score > best["score"]:
                best = item
            for selection_fold in (2023, 2024):
                other = 2024 if selection_fold == 2023 else 2023
                transfer_score = (deltas[selection_fold], deltas[other])
                old = component_best[selection_fold]
                if old is None or transfer_score > old["transfer_score"]:
                    component_best[selection_fold] = {
                        **item, "transfer_score": transfer_score}
        transfer[component_scale] = component_best

    assert best is not None
    detail = {}
    for fold, data in fold_data.items():
        component_weight = np.clip(
            best["component_scale"] * data["weight"], 0.0, 0.95)
        start = np.clip(
            (1 - component_weight) * data["base"]
            + component_weight * data["p1"], EPS, 1 - EPS)
        row_scale = np.asarray(best["scales"])[data["bucket"]]
        pred = sigmoid(logit(start) + row_scale * data["direction"])
        score = metrics(data["y"], pred)
        current_score = metrics(data["y"], data["current"])
        detail[fold] = {
            "delta_bss": float(score["bss_raw"] - current_score["bss_raw"]),
            "bss_raw": float(score["bss_raw"]),
            "brier": float(score["brier"]),
            "pred_mean": float(score["pred_mean"]),
            "bucket_rows": np.bincount(data["bucket"], minlength=5).tolist(),
        }

    report = {
        "cuts": CUTS.tolist(),
        "selection": "maximize minimum delta BSS across Val2023/Val2024",
        "best": {
            "component_scale": best["component_scale"],
            "residual_scales": best["scales"],
            "deltas": best["deltas"],
        },
        "detail": detail,
        "cross_season_transfer": {
            str(component_scale): {
                str(fold): {
                    "selected_scales": item["scales"],
                    "selected_delta": item["deltas"][fold],
                    "other_fold_delta": item["deltas"][2024 if fold == 2023 else 2023],
                }
                for fold, item in values.items()
            }
            for component_scale, values in transfer.items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bucketed_c1_residual.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["best"], ensure_ascii=False, indent=2))
    print(json.dumps(detail, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
