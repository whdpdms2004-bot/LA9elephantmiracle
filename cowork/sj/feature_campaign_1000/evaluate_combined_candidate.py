"""현재 component 기준과 C1 base residual + P1 후보를 동일 OOF에서 비교한다."""
from __future__ import annotations

import json
import argparse
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
GPU_PREFIX = "confirm_xgboost_v2r200_tm500_robust_cuda_efull"
GPU_SEEDS = (20260818, 20260819, 20260820)
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
RESIDUAL_SCALE = 0.50
EPS = 1e-7


def load_xgb_prediction(arm: str, fold: int) -> np.ndarray:
    """캐시 태그 도입 전/후의 동일 campaign seed 예측을 모두 지원한다."""
    candidates = [
        SINGLE / f"{PREFIX}_scampaign_{arm}_{fold}.npy",
        SINGLE / f"{PREFIX}_{arm}_{fold}.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(candidates)


def base_prediction(df: pd.DataFrame, fold: int) -> np.ndarray:
    ids = df.loc[df["season"].eq(fold), "row_id"].to_numpy()
    if fold == 2024:
        return (pd.read_parquet(PROD).set_index("row_id").reindex(ids)
                ["submit021_reverse20_s040_tabm"].to_numpy(np.float64))
    names = sorted({
        re.match(r"(.+)_fold\d{4}\.parquet", path.name).group(1)
        for path in OOF_DIR.glob("*.parquet")
    })
    values = []
    for name in names:
        path = OOF_DIR / f"{name}_fold{fold}.parquet"
        if path.exists():
            values.append(pd.read_parquet(path).set_index("row_id").reindex(ids)
                          ["prediction"].to_numpy(np.float64))
    return np.mean(values, axis=0)


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, EPS, 1 - EPS)
    return np.log(values / (1 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        choices=[
            "C0", "C1", "C2", "C3", "C4", "K1", "K6", "K12",
            "H6", "P6", "CT6", "IN6"],
        default="C1")
    parser.add_argument("--family", choices=["xgboost", "xgboost_gpu3", "catboost"],
                        default="xgboost")
    return parser.parse_args()


def main():
    args = parse_args()
    candidate_tag = args.candidate.lower()
    df = load()
    rows = []
    predictions = {}
    for fold in (2023, 2024):
        valid = df["season"].eq(fold).to_numpy()
        y = df.loc[valid, TARGET].to_numpy(np.float64)
        game_type = df.loc[valid, "game_type"].astype(str).to_numpy()
        month = df.loc[valid, "game_month"].to_numpy()
        bucket = np.digitize(
            df.loc[valid, "asof_pitcher_n"].to_numpy(np.float64), CUTS)
        weight = BW[bucket]
        base = np.clip(base_prediction(df, fold), EPS, 1 - EPS)
        if args.family == "xgboost":
            model_b0 = load_xgb_prediction("B0", fold)
            model_candidate = load_xgb_prediction(args.candidate, fold)
            residual_direction = logit(model_candidate) - logit(model_b0)
        elif args.family == "xgboost_gpu3":
            directions = []
            for seed in GPU_SEEDS:
                seed_b0 = np.load(
                    SINGLE / f"{GPU_PREFIX}_s{seed}_B0_{fold}.npy")
                seed_candidate = np.load(
                    SINGLE / f"{GPU_PREFIX}_s{seed}_{args.candidate}_{fold}.npy")
                directions.append(logit(seed_candidate) - logit(seed_b0))
            residual_direction = np.mean(directions, axis=0)
            model_b0 = np.mean([
                np.load(SINGLE / f"{GPU_PREFIX}_s{seed}_B0_{fold}.npy")
                for seed in GPU_SEEDS], axis=0)
            model_candidate = sigmoid(logit(model_b0) + residual_direction)
        else:
            cat_dir = HERE / "outputs" / "single_catboost"
            model_b0 = np.load(cat_dir / f"B0_{fold}.npy")
            model_candidate = np.load(cat_dir / f"{args.candidate}_{fold}.npy")
            residual_direction = logit(model_candidate) - logit(model_b0)
        component_p0 = np.load(CACHE / f"v75_confirm_P0_{fold}.npy")
        component_p1 = np.load(CACHE / f"v75_confirm_P1_{fold}.npy")
        base_c1 = sigmoid(logit(base) + RESIDUAL_SCALE * residual_direction)
        current_p0 = np.clip((1 - weight) * base + weight * component_p0,
                             EPS, 1 - EPS)
        p1_only = np.clip((1 - weight) * base + weight * component_p1,
                          EPS, 1 - EPS)
        variants = {
            "current_p0": current_p0,
            "p1_only": p1_only,
            f"{candidate_tag}_residual_p0": np.clip(
                (1 - weight) * base_c1 + weight * component_p0, EPS, 1 - EPS),
            f"{candidate_tag}_residual_p1": np.clip(
                (1 - weight) * base_c1 + weight * component_p1, EPS, 1 - EPS),
        }
        for scale in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
            tag = f"{int(round(100 * scale)):02d}"
            variants[f"p0_then_{candidate_tag}_s{tag}"] = sigmoid(
                logit(current_p0) + scale * residual_direction)
            variants[f"p1_then_{candidate_tag}_s{tag}"] = sigmoid(
                logit(p1_only) + scale * residual_direction)
        for component_scale in (0.50, 0.75, 1.00, 1.25):
            component_weight = np.clip(component_scale * weight, 0.0, 0.95)
            component_pred = np.clip(
                (1 - component_weight) * base + component_weight * component_p1,
                EPS, 1 - EPS)
            for residual_scale in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
                ctag = f"{int(round(100 * component_scale)):03d}"
                rtag = f"{int(round(100 * residual_scale)):02d}"
                variants[f"joint_c{ctag}_r{rtag}"] = sigmoid(
                    logit(component_pred) + residual_scale * residual_direction)
        if args.family == "xgboost" and args.candidate == "C1":
            cat_dir = HERE / "outputs" / "single_catboost"
            cat_direction = (
                logit(np.load(cat_dir / f"C1_{fold}.npy"))
                - logit(np.load(cat_dir / f"B0_{fold}.npy")))
            for component_scale in (0.75, 1.00):
                component_weight = np.clip(component_scale * weight, 0.0, 0.95)
                component_pred = np.clip(
                    (1 - component_weight) * base + component_weight * component_p1,
                    EPS, 1 - EPS)
                for xgb_scale in (0.05, 0.10, 0.15, 0.20):
                    for cat_scale in (0.05, 0.10, 0.15, 0.20):
                        ctag = f"{int(round(100 * component_scale)):03d}"
                        xtag = f"{int(round(100 * xgb_scale)):02d}"
                        atag = f"{int(round(100 * cat_scale)):02d}"
                        variants[f"dual_c{ctag}_x{xtag}_a{atag}"] = sigmoid(
                            logit(component_pred)
                            + xgb_scale * residual_direction
                            + cat_scale * cat_direction)
            k6_directions = []
            for seed in GPU_SEEDS:
                k6_directions.append(
                    logit(np.load(
                        SINGLE / f"{GPU_PREFIX}_s{seed}_K6_{fold}.npy"))
                    - logit(np.load(
                        SINGLE / f"{GPU_PREFIX}_s{seed}_B0_{fold}.npy")))
            k6_direction = np.mean(k6_directions, axis=0)
            for component_scale in (0.75, 1.00):
                component_weight = np.clip(component_scale * weight, 0.0, 0.95)
                component_pred = np.clip(
                    (1 - component_weight) * base + component_weight * component_p1,
                    EPS, 1 - EPS)
                for xgb_scale in (0.05, 0.10, 0.15, 0.20):
                    for k6_scale in (0.05, 0.10, 0.15, 0.20, 0.25):
                        ctag = f"{int(round(100 * component_scale)):03d}"
                        xtag = f"{int(round(100 * xgb_scale)):02d}"
                        ktag = f"{int(round(100 * k6_scale)):02d}"
                        variants[f"c1k6_c{ctag}_x{xtag}_k{ktag}"] = sigmoid(
                            logit(component_pred)
                            + xgb_scale * residual_direction
                            + k6_scale * k6_direction)
        ref = metrics(y, variants["current_p0"])["bss_raw"]
        for name, pred in variants.items():
            score = metrics(y, pred)
            row_gain = ((variants["current_p0"] - y) ** 2
                        - (pred - y) ** 2)
            null = y.mean() * (1 - y.mean())
            se = 100000 * float(row_gain.std(ddof=1) / np.sqrt(len(y))) / null
            item = {
                "fold": fold,
                "variant": name,
                "bss": score["bss_raw"],
                "dbss_vs_current": score["bss_raw"] - ref,
                "t_row": ((score["bss_raw"] - ref) / se if se > 0 else 0.0),
                "brier": score["brier"],
                "pred_mean": float(pred.mean()),
                "r_brier": float(np.mean((pred[game_type == "R"] - y[game_type == "R"]) ** 2)),
                "f_brier": float(np.mean((pred[game_type == "F"] - y[game_type == "F"]) ** 2)),
            }
            for value in sorted(np.unique(month)):
                mask = month == value
                item[f"month_{int(value):02d}_brier"] = float(
                    np.mean((pred[mask] - y[mask]) ** 2))
            rows.append(item)
        predictions[fold] = variants
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{args.family}_{candidate_tag}"
    result.to_csv(OUT / f"{stem}_candidate_metrics.csv", index=False)
    pivot = result.pivot_table(
        index="variant", columns="fold", values="dbss_vs_current")
    pivot["worst"] = pivot.min(axis=1)
    pivot["sum"] = pivot[[2023, 2024]].sum(axis=1)
    pivot = pivot.sort_values(["worst", "sum"], ascending=False)
    print(pivot.round(3).to_string())
    best = pivot.reset_index().iloc[0].to_dict()
    (OUT / f"{stem}_summary.json").write_text(json.dumps({
        "candidate": args.candidate,
        "family": args.family,
        "base_residual": f"scaled * (logit({args.candidate}) - logit(B0))",
        "component": "P1 confirmed second-order",
        "comparison": "current P0 component on production base",
        "best": best,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
