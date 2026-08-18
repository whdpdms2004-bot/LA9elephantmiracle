"""기존 robust CatBoost에서 component hierarchy C0/C1을 두 fold 비교한다."""
from __future__ import annotations

import argparse
import json
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


HERE = Path(__file__).resolve().parent
SJ = HERE.parents[1]
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = SJ / "feature_campaign_1000"
MO = SJ / "experiment" / "model_optimization"
sys.path.insert(0, str(MO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAMPAIGN))
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import CATEGORICAL_COLUMNS, TARGET, probability_metrics, recency_weights
from v77_single_xgb_screen import (
    add_direct_products, build_component_unique, build_component_unique_forward,
)

PARAMS = MO / "catboost_v2r200_tm500_robust_best.json"
OUT = CAMPAIGN / "outputs" / "single_catboost"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="2023,2024")
    parser.add_argument("--arms", default="B0,C0,C1,C3,C4")
    parser.add_argument("--random-seed", type=int, default=None)
    return parser.parse_args()


def raw_bss(score: dict) -> float:
    return 100000.0 * (1.0 - score["normalized_brier"])


def make_features(frame: pd.DataFrame, base: list[str], fold: int, arm: str,
                  hierarchy: pd.DataFrame):
    work = frame[base].copy()
    features = list(base)
    if arm in ("C0", "C1", "C3", "C4", "F0", "F1"):
        selected = hierarchy
        if arm in ("C3", "C4"):
            suffixes = (
                "platoon_split", "platoon_rel", "platoon_split_w",
                "bat_platoon_split", "bat_platoon_rel", "bat_platoon_split_w",
                "count_platoon_split", "count_platoon_rel", "count_platoon_w",
                "inning_platoon_split", "inning_platoon_rel", "inning_platoon_w",
            )
            selected = hierarchy[
                [column for column in hierarchy
                 if column.endswith(suffixes) or "_bat_pl_" in column]
            ]
        work = pd.concat([work, selected.reset_index(drop=True)], axis=1)
        features.extend(selected.columns.tolist())
    if arm in ("C1", "C4", "F1"):
        direct_source = frame.copy(deep=False)
        direct = add_direct_products(direct_source)
        work = pd.concat([work, direct_source[direct].reset_index(drop=True)], axis=1)
        features.extend(direct)
    return work, features


def main():
    args = parse_args()
    folds = [int(value) for value in args.folds.split(",") if value]
    arms = [value for value in args.arms.split(",") if value]
    allowed = {"B0", "C0", "C1", "C3", "C4", "F0", "F1"}
    unknown = [arm for arm in arms if arm not in allowed]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    frame, base_features = load_enhanced_frame()
    params = json.loads(PARAMS.read_text(encoding="utf-8"))["best_params"]
    half_life = float(params.pop("half_life"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in folds:
        train_mask = frame["season"].lt(fold)
        valid_mask = frame["season"].eq(fold)
        valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
        weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
        hierarchy = build_component_unique(frame, base_features, fold)
        forward_hierarchy = None
        if any(arm in ("F0", "F1") for arm in arms):
            forward_hierarchy = build_component_unique_forward(
                frame, base_features, fold, cache={fold: hierarchy})
        print(f"fold={fold} rows={valid_mask.sum()} hierarchy={hierarchy.shape[1]}", flush=True)
        for arm in arms:
            started = time.time()
            selected_hierarchy = (
                forward_hierarchy if arm in ("F0", "F1") else hierarchy)
            work, features = make_features(
                frame, base_features, fold, arm, selected_hierarchy)
            categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
            for column in categorical:
                work[column] = work[column].fillna("__MISSING__").astype(str)
            train_pool = Pool(work.loc[train_mask], label=frame.loc[train_mask, TARGET],
                              cat_features=categorical, weight=weights)
            valid_pool = Pool(work.loc[valid_mask], label=valid_y,
                              cat_features=categorical)
            model = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                eval_metric="Logloss",
                task_type="GPU",
                devices="0",
                random_seed=(args.random_seed if args.random_seed is not None
                             else 20260818 + fold),
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True,
                      early_stopping_rounds=220)
            pred = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
            score = probability_metrics(valid_y, pred)
            bss = raw_bss(score)
            np.save(OUT / f"{arm}_{fold}.npy", pred)
            rows.append({
                "fold": fold,
                "arm": arm,
                "n_features": len(features),
                "best_iteration": int(model.get_best_iteration()),
                "elapsed_sec": time.time() - started,
                "bss_raw": bss,
                **score,
            })
            print(f"  {arm} f={len(features)} BSSraw={bss:.2f} "
                  f"mean={score['pred_mean']:.5f} iter={model.get_best_iteration()} "
                  f"t={time.time()-started:.1f}s", flush=True)
            del model, train_pool, valid_pool, work, pred
            gc.collect()
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "metrics.csv", index=False)
    pivot = result.pivot_table(index="arm", columns="fold", values="bss_raw")
    if "B0" in pivot.index:
        pivot = pivot.subtract(pivot.loc["B0"], axis=1)
        label = "BSS minus B0"
    else:
        label = "BSS raw (B0 not requested)"
    print(f"\n{label}\n{pivot.round(3).to_string()}")


if __name__ == "__main__":
    main()
