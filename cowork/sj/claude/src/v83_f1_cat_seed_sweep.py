"""strict F1 CatBoost의 3-fold seed 안정성을 GPU로 측정한다."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


HERE = Path(__file__).resolve().parent
SJ = HERE.parent
MODEL_OPT = SJ / "experiment" / "model_optimization"
sys.path.insert(0, str(MODEL_OPT))
sys.path.insert(0, str(HERE))

from evaluate_bucketed_residual import logit, sigmoid
from evaluate_train_only_season_offsets import forecast_offset
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    TARGET,
    probability_metrics,
    recency_weights,
)
from v77_single_xgb_screen import (
    build_component_unique,
    build_component_unique_forward,
)
from v80_single_catboost import make_features, raw_bss


PARAMS = MODEL_OPT / "catboost_v2r200_tm500_robust_best.json"
OUTPUT = HERE / "outputs" / "f1_cat_seed_sweep"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--seeds", default="20262843,20262844,20262845")
    parser.add_argument("--max-iterations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = [int(value) for value in args.folds.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    frame, base_features = load_enhanced_frame()
    rates = frame.groupby("season")[TARGET].mean()
    params = json.loads(PARAMS.read_text(encoding="utf-8"))["best_params"]
    half_life = float(params.pop("half_life"))
    if args.max_iterations is not None:
        params["iterations"] = args.max_iterations
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold in folds:
        train_mask = frame["season"].lt(fold)
        valid_mask = frame["season"].eq(fold)
        valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
        weights = recency_weights(
            frame.loc[train_mask, "season"], fold, half_life)
        static = build_component_unique(frame, base_features, fold)
        forward = build_component_unique_forward(
            frame, base_features, fold, cache={fold: static})
        work, features = make_features(
            frame, base_features, fold, "F1", forward)
        categorical = [
            column for column in CATEGORICAL_COLUMNS if column in features]
        for column in categorical:
            work[column] = work[column].fillna("__MISSING__").astype(str)
        train_pool = Pool(
            work.loc[train_mask, features],
            label=frame.loc[train_mask, TARGET],
            cat_features=categorical,
            weight=weights,
        )
        valid_pool = Pool(
            work.loc[valid_mask, features],
            label=valid_y,
            cat_features=categorical,
        )
        offset = forecast_offset(rates, fold, window=None, damping=0.25)
        print(
            f"fold={fold} train={int(train_mask.sum())} valid={int(valid_mask.sum())} "
            f"features={len(features)} offset={offset:+.8f}", flush=True)

        for seed in seeds:
            started = time.time()
            model = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                eval_metric="Logloss",
                task_type="GPU",
                devices="0",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(
                train_pool,
                eval_set=valid_pool,
                use_best_model=True,
                early_stopping_rounds=220,
            )
            prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
            adjusted = sigmoid(logit(prediction) + offset)
            raw_score = probability_metrics(valid_y, prediction)
            adjusted_score = probability_metrics(valid_y, adjusted)
            elapsed = time.time() - started
            np.save(OUTPUT / f"F1_{fold}_seed{seed}.npy", prediction)
            rows.append({
                "fold": fold,
                "seed": seed,
                "n_features": len(features),
                "best_iteration": int(model.get_best_iteration()),
                "elapsed_sec": elapsed,
                "season_logit_offset": offset,
                "bss_raw": raw_bss(raw_score),
                "bss_adjusted": raw_bss(adjusted_score),
                "brier_raw": raw_score["brier"],
                "brier_adjusted": adjusted_score["brier"],
                "pred_mean_raw": raw_score["pred_mean"],
                "pred_mean_adjusted": adjusted_score["pred_mean"],
            })
            print(
                f"  seed={seed} raw={rows[-1]['bss_raw']:.3f} "
                f"adjusted={rows[-1]['bss_adjusted']:.3f} "
                f"iter={rows[-1]['best_iteration']} t={elapsed:.1f}s",
                flush=True,
            )
            del model, prediction, adjusted
            gc.collect()

        del train_pool, valid_pool, work, forward, static, weights
        gc.collect()

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "metrics.csv", index=False)
    summary = result.groupby("seed").agg(
        mean_bss=("bss_adjusted", "mean"),
        worst_bss=("bss_adjusted", "min"),
        std_bss=("bss_adjusted", "std"),
        mean_raw=("bss_raw", "mean"),
        mean_iteration=("best_iteration", "mean"),
        elapsed_sec=("elapsed_sec", "sum"),
    )
    pivot = result.pivot(index="seed", columns="fold", values="bss_adjusted")
    recency_weights = {2022: 1.0 / 7.0, 2023: 2.0 / 7.0, 2024: 4.0 / 7.0}
    if all(fold in pivot for fold in recency_weights):
        summary["recent_weighted_bss"] = sum(
            pivot[fold] * weight for fold, weight in recency_weights.items())
        summary["bss_2024"] = pivot[2024]
        summary = summary.sort_values(
            ["recent_weighted_bss", "worst_bss"], ascending=False)
    else:
        summary = summary.sort_values(
            ["worst_bss", "mean_bss"], ascending=False)
    summary.to_csv(OUTPUT / "summary.csv")
    print("\n" + summary.round(4).to_string())
    print(f"saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
