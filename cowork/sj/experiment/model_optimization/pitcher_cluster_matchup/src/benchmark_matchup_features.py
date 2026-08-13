from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_cluster_features import best_blend, run_model  # noqa: E402
from benchmark_insight_features import (  # noqa: E402
    add_calibration_features,
    build_past_only_lookups,
    load_local_trial,
)
from run_optuna_enhanced import load_enhanced_frame  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--folds", default="2024")
    parser.add_argument("--modes", default="pair,hier,all")
    return parser.parse_args()


def select_features(cache, mode):
    available = [column for column in cache if column not in {"row_id", "season"}]
    if mode == "pair":
        requested = [
            "match_pair_delta", "match_pair_delta_reliability",
            "match_pair_delta_rate", "match_pair_known",
        ]
    elif mode == "hier":
        requested = [
            "match_pair_delta", "match_pair_delta_reliability",
            "match_pair_delta_rate", "match_pair_known",
            "match_pitcher_bhand_delta", "match_pitcher_bhand_delta_reliability",
            "match_phand_batter_delta", "match_phand_batter_delta_reliability",
            "batter_overall_resid", "match_batter_known",
        ]
    elif mode == "all":
        return available
    else:
        raise ValueError(mode)
    return [column for column in requested if column in available]


def main():
    args = parse_args()
    configs = [value for value in args.configs.split(",") if value]
    folds = [int(value) for value in args.folds.split(",") if value]
    modes = [value for value in args.modes.split(",") if value]
    frame, base_features = load_enhanced_frame()
    labels = pd.read_parquet(MODEL_DIR / "failure_component_labels.parquet")
    lookups, _ = build_past_only_lookups(frame, labels)
    frame, _, prior_columns = add_calibration_features(frame, lookups)
    adjusted = [
        column for column in prior_columns
        if column.startswith(("pitcher_success_", "batter_success_"))
        and "_adjusted_smoothed_" in column
    ]
    base_features = list(dict.fromkeys(base_features + adjusted))
    trial = load_local_trial()
    performance = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    performance = performance.loc[performance["track"].eq("performance")]
    results = []
    predictions = []
    for config in configs:
        cache = pd.read_parquet(WORK / "oof" / f"matchup_features_{config}.parquet")
        if not frame["row_id"].equals(cache["row_id"]):
            raise RuntimeError(f"Cache misaligned: {config}")
        for mode in modes:
            matchup_features = select_features(cache, mode)
            combined = pd.concat([frame, cache[matchup_features]], axis=1)
            features = list(dict.fromkeys(base_features + matchup_features))
            for fold in folds:
                experiment = f"matchup_{config}_{mode}"
                row, pred = run_model(combined, features, fold, trial, experiment)
                row.update({
                    "matchup_config": config,
                    "mode": mode,
                    "matchup_feature_count": len(matchup_features),
                })
                blend = best_blend(pred, performance.loc[performance["season"].eq(fold)])
                if blend:
                    row.update({f"blend_{key}": value for key, value in blend.items()})
                results.append(row)
                predictions.append(pred)
            del combined
            gc.collect()
        del cache
        gc.collect()
    pd.DataFrame(results).to_csv(
        WORK / "reports" / "matchup_xgb_validation.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_parquet(
        WORK / "oof" / "matchup_xgb_predictions.parquet", index=False
    )
    print(json.dumps({"runs": len(results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
