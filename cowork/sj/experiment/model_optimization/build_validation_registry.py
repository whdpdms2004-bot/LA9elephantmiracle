from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "experiment" / "model_optimization"
STUDIES = {
    "xgboost_v1": {
        "name": "xgboost_v1_full_2023_2024",
        "family": "xgboost",
        "feature_version": "V1",
        "trackman": False,
    },
    "catboost_v1": {
        "name": "catboost_v1_full_2023_2024",
        "family": "catboost",
        "feature_version": "V1",
        "trackman": False,
    },
    "xgboost_enhanced_robust": {
        "name": "xgboost_v2r200_tm500_robust",
        "family": "xgboost",
        "feature_version": "V2R200_TM500_ALL",
        "trackman": True,
    },
    "xgboost_enhanced_recent": {
        "name": "xgboost_v2r200_tm500_2024",
        "family": "xgboost",
        "feature_version": "V2R200_TM500_ALL",
        "trackman": True,
    },
    "xgboost_enhanced_local": {
        "name": "xgboost_v2r200_tm500_local_2024",
        "family": "xgboost",
        "feature_version": "V2R200_TM500_ALL",
        "trackman": True,
    },
    "catboost_enhanced_robust": {
        "name": "catboost_v2r200_tm500_robust",
        "family": "catboost",
        "feature_version": "V2R200_TM500_ALL",
        "trackman": True,
    },
    "lightgbm_enhanced_robust": {
        "name": "lightgbm_v2r200_tm500_robust",
        "family": "lightgbm",
        "feature_version": "V2R200_TM500_ALL",
        "trackman": True,
    },
    "xgboost_insight_success_local": {
        "name": "xgboost_insight_success_local_2024",
        "family": "xgboost",
        "feature_version": "INSIGHT_PRIOR_SUCCESS",
        "trackman": True,
    },
}
FIXED_RESULTS = {
    "xgb_v1_fixed": WORK_DIR / "xgb_v1_fixed_2024.json",
    "cat_v1_fixed": WORK_DIR / "cat_v1_fixed_2024.json",
    "lgb_v1_fixed": WORK_DIR / "lgb_v1_fixed_2024.json",
    "xgb_v3_embedding_fixed": WORK_DIR / "xgb_v3_fixed_2024.json",
}


def trial_rows():
    rows = []
    for experiment, config in STUDIES.items():
        study_name = config["name"]
        database = WORK_DIR / f"{study_name}.db"
        if not database.is_file():
            continue
        study = optuna.load_study(
            study_name=study_name, storage=f"sqlite:///{database.as_posix()}"
        )
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            family = config["family"]
            for fold in [2023, 2024]:
                metrics = trial.user_attrs.get(f"fold_{fold}")
                if not metrics:
                    continue
                rows.append(
                    {
                        "experiment": experiment,
                        "feature_version": config["feature_version"],
                        "family": family,
                        "trial": int(trial.number),
                        "fold": int(fold),
                        "train_through": int(fold - 1),
                        "trackman": config["trackman"],
                        "trackman_cutoff": fold if config["trackman"] else None,
                        "min_trackman_season_pitches": (
                            trial.user_attrs.get("min_trackman_season_pitches", 500)
                            if config["trackman"]
                            else None
                        ),
                        "brier": metrics["brier"],
                        "normalized_brier": metrics["normalized_brier"],
                        "bss": metrics["bss"],
                        "auc": metrics["auc"],
                        "logloss": metrics["logloss"],
                        "target_mean": metrics["target_mean"],
                        "pred_mean": metrics["pred_mean"],
                        "mean_gap": metrics["mean_gap"],
                        "best_iteration": trial.user_attrs.get(
                            f"best_iteration_{fold}"
                        ),
                        "objective": trial.value,
                        "source": str(database.relative_to(ROOT)),
                    }
                )
    return rows


def fixed_rows():
    rows = []
    for experiment, path in FIXED_RESULTS.items():
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        trackman = "embedding" in experiment
        rows.append(
            {
                "experiment": experiment,
                "feature_version": "V3" if trackman else "V1",
                "family": experiment.split("_")[0],
                "trial": None,
                "fold": 2024,
                "train_through": 2023,
                "trackman": trackman,
                "trackman_cutoff": 2024 if trackman else None,
                "min_trackman_season_pitches": 100 if trackman else None,
                "brier": result.get("brier"),
                "normalized_brier": (
                    1.0 - result.get("bss", 0.0) / 100000.0
                ),
                "bss": result.get("bss"),
                "auc": result.get("auc"),
                "logloss": result.get("logloss"),
                "target_mean": result.get("target_mean"),
                "pred_mean": result.get("pred_mean"),
                "mean_gap": (
                    result.get("pred_mean") - result.get("target_mean")
                    if result.get("pred_mean") is not None
                    and result.get("target_mean") is not None
                    else None
                ),
                "best_iteration": result.get("best_iteration"),
                "objective": None,
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def oof_rows():
    path = WORK_DIR / "top_models_individual_metrics.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        family = str(item["model_name"]).split("_", 1)[0]
        rows.append(
            {
                "experiment": item["model_name"],
                "feature_version": "V1_OOF_RETRAIN",
                "family": family,
                "trial": None,
                "fold": int(item["season"]),
                "train_through": int(item["season"] - 1),
                "trackman": False,
                "trackman_cutoff": None,
                "min_trackman_season_pitches": None,
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": None,
                "objective": None,
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def ensemble_rows():
    path = WORK_DIR / "ensemble_selection.json"
    if not path.is_file():
        return []
    selection = json.loads(path.read_text(encoding="utf-8"))
    best = selection["best"]
    rows = []
    for transition, metrics in best["transition_parameters"].items():
        _, valid_year = transition.split("_to_")
        rows.append(
            {
                "experiment": "v1_oof_ensemble_best",
                "feature_version": "V1_OOF_ENSEMBLE",
                "family": "ensemble",
                "trial": None,
                "fold": int(valid_year),
                "train_through": int(valid_year) - 1,
                "trackman": False,
                "trackman_cutoff": None,
                "min_trackman_season_pitches": None,
                "brier": metrics.get("brier"),
                "normalized_brier": metrics.get("normalized_brier"),
                "bss": metrics.get("bss"),
                "auc": metrics.get("auc"),
                "logloss": metrics.get("logloss"),
                "target_mean": metrics.get("target_mean"),
                "pred_mean": metrics.get("pred_mean"),
                "mean_gap": metrics.get("mean_gap"),
                "best_iteration": None,
                "objective": best.get("objective"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def v2_fixed_rows():
    path = WORK_DIR / "v2_fixed_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": item["experiment"],
                "feature_version": item.get("feature_version", "V2_TEMPORAL"),
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": bool(item.get("trackman", False)),
                "trackman_cutoff": None,
                "min_trackman_season_pitches": None,
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": item.get("best_iteration"),
                "objective": None,
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def v2_ablation_rows():
    path = WORK_DIR / "v2_ablation_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": item["experiment"],
                "feature_version": item["feature_version"],
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": bool(item.get("trackman", False)),
                "trackman_cutoff": None,
                "min_trackman_season_pitches": None,
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": item.get("best_iteration"),
                "objective": None,
                "feature_count": item.get("feature_count"),
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def failure_model_rows():
    path = WORK_DIR / "failure_multiclass_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": item["experiment"],
                "feature_version": f"{item['feature_version']}+{item['auxiliary_target']}",
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": False,
                "trackman_cutoff": None,
                "min_trackman_season_pitches": None,
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": item.get("best_iteration"),
                "objective": None,
                "feature_count": item.get("feature_count"),
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def failure_blend_rows():
    path = WORK_DIR / "failure_blend_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": f"{item['experiment']}::{item['multiclass_model']}",
                "feature_version": "V2_FAILURE_BLEND",
                "family": "ensemble",
                "trial": None,
                "fold": int(item["fold"]),
                "train_through": int(item["fold"] - 1),
                "selection_fold": int(item["selection_fold"]),
                "trackman": False,
                "trackman_cutoff": None,
                "min_trackman_season_pitches": None,
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": None,
                "objective": None,
                "binary_weight": item.get("binary_weight"),
                "prediction_correlation": item.get("prediction_correlation"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def trackman500_rows():
    path = WORK_DIR / "trackman500_fixed_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": item["experiment"],
                "feature_version": item["feature_version"],
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": True,
                "trackman_cutoff": int(item["trackman_cutoff"]),
                "min_trackman_season_pitches": int(
                    item["min_trackman_season_pitches"]
                ),
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": item.get("best_iteration"),
                "objective": None,
                "feature_count": item.get("feature_count"),
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def cat_enhanced_rows():
    path = WORK_DIR / "cat_enhanced_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        uses_tm = bool(item.get("trackman", False))
        rows.append(
            {
                "experiment": item["experiment"],
                "feature_version": item["feature_version"],
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": uses_tm,
                "trackman_cutoff": int(item["fold"]) if uses_tm else None,
                "min_trackman_season_pitches": 500 if uses_tm else None,
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": item.get("best_iteration"),
                "objective": None,
                "feature_count": item.get("feature_count"),
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def strict_embedding_rows():
    directory = (
        ROOT
        / "experiment"
        / "pitcher_embedding"
        / "outputs"
        / "trackman500_multitask"
    )
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("multitask_dim*_validation.csv")):
        frame = pd.read_csv(path)
        for item in frame.to_dict("records"):
            rows.append(
                {
                    "experiment": f"strict_tm500_multitask_dim{int(item['embedding_dim'])}",
                    "feature_version": f"TM500_MULTITASK_DIM{int(item['embedding_dim'])}",
                    "family": "pytorch",
                    "trial": None,
                    "fold": int(item["fold"]),
                    "train_through": int(item["train_through"]),
                    "trackman": True,
                    "trackman_cutoff": int(item["trackman_cutoff"]),
                    "min_trackman_season_pitches": int(
                        item["min_trackman_season_pitches"]
                    ),
                    "brier": item.get("brier"),
                    "normalized_brier": item.get("normalized_brier"),
                    "bss": item.get("bss"),
                    "auc": item.get("auc"),
                    "logloss": item.get("logloss"),
                    "target_mean": item.get("target_mean"),
                    "pred_mean": item.get("pred_mean"),
                    "mean_gap": item.get("mean_gap"),
                    "best_iteration": item.get("epochs"),
                    "objective": None,
                    "elapsed_sec": item.get("elapsed_sec"),
                    "source": str(path.relative_to(ROOT)),
                }
            )
    return rows


def trackman_pitchgroup_rows():
    path = WORK_DIR / "trackman_pitchgroup_fixed_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": item["experiment"],
                "feature_version": item["feature_version"],
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": True,
                "trackman_cutoff": int(item["trackman_cutoff"]),
                "min_trackman_season_pitches": int(
                    item["min_trackman_season_pitches"]
                ),
                "min_trackman_group_pitches": int(
                    item["min_trackman_group_pitches"]
                ),
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": item.get("best_iteration"),
                "objective": None,
                "feature_count": item.get("feature_count"),
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def trackman_gated_rows():
    path = WORK_DIR / "trackman_gated_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": item["experiment"],
                "feature_version": item["feature_version"],
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": True,
                "trackman_cutoff": int(item["trackman_cutoff"]),
                "min_trackman_season_pitches": int(
                    item["min_trackman_season_pitches"]
                ),
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": None,
                "objective": None,
                "available_best_iteration": item.get("available_best_iteration"),
                "unavailable_best_iteration": item.get("unavailable_best_iteration"),
                "available_bss": item.get("available_bss"),
                "unavailable_bss": item.get("unavailable_bss"),
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def smoothing_grid_rows():
    path = WORK_DIR / "smoothing_grid_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": item["experiment"],
                "feature_version": item["feature_version"],
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": True,
                "trackman_cutoff": int(item["trackman_cutoff"]),
                "min_trackman_season_pitches": int(
                    item["min_trackman_season_pitches"]
                ),
                "smoothing_strength": int(item["smoothing_strength"]),
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": item.get("best_iteration"),
                "objective": None,
                "feature_count": item.get("feature_count"),
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def enhanced_seed_oof_rows():
    path = WORK_DIR / "enhanced_seed_oof_metrics.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": item["model_name"],
                "feature_version": item.get("feature_version", "V2R200_TM500_ALL"),
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": True,
                "trackman_cutoff": int(item["fold"]),
                "min_trackman_season_pitches": 500,
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": None,
                "objective": item.get("objective"),
                "feature_count": item.get("feature_count"),
                "seed_count": 3,
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def direct_brier_rows():
    path = WORK_DIR / "direct_brier_results.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            {
                "experiment": item["experiment"],
                "feature_version": item["feature_version"],
                "family": item["family"],
                "trial": item.get("trial"),
                "fold": int(item["fold"]),
                "train_through": int(item["train_through"]),
                "trackman": True,
                "trackman_cutoff": int(item["trackman_cutoff"]),
                "min_trackman_season_pitches": 500,
                "brier": item.get("brier"),
                "normalized_brier": item.get("normalized_brier"),
                "bss": item.get("bss"),
                "auc": item.get("auc"),
                "logloss": item.get("logloss"),
                "target_mean": item.get("target_mean"),
                "pred_mean": item.get("pred_mean"),
                "mean_gap": item.get("mean_gap"),
                "best_iteration": item.get("best_iteration"),
                "objective": None,
                "feature_count": item.get("feature_count"),
                "clipped_fraction": item.get("clipped_fraction"),
                "elapsed_sec": item.get("elapsed_sec"),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def enhanced_ensemble_rows():
    path = WORK_DIR / "enhanced_ensemble_selection.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for track, candidate in data.get("best", {}).items():
        for transition, metrics in candidate.get("transition_parameters", {}).items():
            calibration_year, validation_year = map(int, transition.split("_to_"))
            rows.append(
                {
                    "experiment": f"enhanced_oof_ensemble_{track}",
                    "feature_version": f"ENHANCED_OOF_{track.upper()}",
                    "family": "ensemble",
                    "trial": None,
                    "fold": validation_year,
                    "train_through": calibration_year,
                    "trackman": True,
                    "trackman_cutoff": validation_year,
                    "min_trackman_season_pitches": 500,
                    "brier": metrics.get("brier"),
                    "normalized_brier": metrics.get("normalized_brier"),
                    "bss": metrics.get("bss"),
                    "auc": metrics.get("auc"),
                    "logloss": metrics.get("logloss"),
                    "target_mean": metrics.get("target_mean"),
                    "pred_mean": metrics.get("pred_mean"),
                    "mean_gap": metrics.get("mean_gap"),
                    "best_iteration": None,
                    "objective": candidate.get("objective"),
                    "model_count": len(metrics.get("selected_models", [])),
                    "blend_space": candidate.get("space"),
                    "calibration": candidate.get("calibration"),
                    "source": str(path.relative_to(ROOT)),
                }
            )
    return rows


def insight_feature_rows():
    paths = [
        WORK_DIR / "insight_feature_ablation_results_exact_recheck.csv",
        WORK_DIR / "insight_feature_ablation_results_exact_recheck_2022.csv",
        WORK_DIR / "insight_feature_ablation_results_exact_recheck_2023.csv",
        WORK_DIR / "insight_feature_ablation_results_component_screen_2024.csv",
        WORK_DIR / "insight_feature_ablation_results_component_top_2023.csv",
        WORK_DIR / "insight_feature_ablation_results_success_screen_2024.csv",
        WORK_DIR / "insight_feature_ablation_results_success_adjusted_2023.csv",
        WORK_DIR / "insight_catboost_results_trial39_2024.csv",
    ]
    rows = []
    seen = set()
    for path in paths:
        if not path.is_file():
            continue
        for item in pd.read_csv(path).to_dict("records"):
            key = (item.get("experiment"), item.get("feature_version"), item.get("fold"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "experiment": item.get("experiment"),
                    "feature_version": item.get("feature_version"),
                    "family": item.get("family"),
                    "trial": item.get("trial"),
                    "fold": int(item["fold"]),
                    "train_through": int(item["train_through"]),
                    "trackman": bool(item.get("trackman", True)),
                    "trackman_cutoff": int(item["trackman_cutoff"]),
                    "min_trackman_season_pitches": int(
                        item.get("min_trackman_season_pitches", 500)
                    ),
                    "brier": item.get("brier"),
                    "normalized_brier": item.get("normalized_brier"),
                    "bss": item.get("bss"),
                    "auc": item.get("auc"),
                    "logloss": item.get("logloss"),
                    "target_mean": item.get("target_mean"),
                    "pred_mean": item.get("pred_mean"),
                    "mean_gap": item.get("mean_gap"),
                    "best_iteration": item.get("best_iteration"),
                    "objective": None,
                    "feature_count": item.get("feature_count"),
                    "elapsed_sec": item.get("elapsed_sec"),
                    "source": str(path.relative_to(ROOT)),
                }
            )
    return rows


def write_markdown(registry: pd.DataFrame):
    optuna_rows = registry[registry["trial"].notna()].copy()
    recent = optuna_rows[optuna_rows["fold"].eq(2024)]
    top_recent = recent.sort_values("normalized_brier").groupby("family").head(10)
    robust = (
        optuna_rows.sort_values("objective")
        .drop_duplicates(["family", "trial"])
        .groupby("family")
        .head(10)
    )
    fixed = registry[registry["trial"].isna()].sort_values(
        ["fold", "normalized_brier"]
    )

    lines = [
        "# 검증 결과 레지스트리",
        "",
        "이 문서는 `build_validation_registry.py`로 재생성한다. 모델 선택은 clipped BSS가 아니라 normalized Brier를 우선한다.",
        "",
        "## 검증 원칙",
        "",
        "- fold 2023: 2019~2022 학습 → 2023 검증",
        "- fold 2024: 2019~2023 학습 → 2024 검증",
        "- 최종 제출: 선택된 설정으로 2019~2024 전체 재학습",
        "- Trackman strict 실험: target fold보다 이전 시즌, 시즌당 500구 이상만 허용",
        "",
        f"전체 기록 행: {len(registry):,}",
        "",
        "## 2024 단일 모델 상위",
        "",
        top_recent[
            [
                "family",
                "trial",
                "brier",
                "normalized_brier",
                "bss",
                "auc",
                "target_mean",
                "pred_mean",
                "best_iteration",
            ]
        ].to_markdown(index=False, floatfmt=".8f"),
        "",
        "## 다중 fold 목적값 상위",
        "",
        robust[
            ["family", "trial", "objective", "fold", "bss", "auc", "mean_gap"]
        ].to_markdown(index=False, floatfmt=".8f"),
        "",
        "## 고정 ablation 및 OOF 재학습",
        "",
        fixed[
            [
                "experiment",
                "feature_version",
                "fold",
                "trackman",
                "brier",
                "bss",
                "auc",
                "pred_mean",
            ]
        ].to_markdown(index=False, floatfmt=".8f"),
        "",
        "## 파일",
        "",
        "- 전체 행: `validation_registry.csv`",
        "- 기존 Optuna 전체: 각 study DB와 leaderboard CSV",
        "- OOF 앙상블 완료 후 이 스크립트를 다시 실행해 결과를 합친다.",
    ]
    (WORK_DIR / "VALIDATION_LOG.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_korean_markdown(registry: pd.DataFrame):
    single_2024 = registry[
        registry["fold"].eq(2024)
        & registry["family"].isin(["xgboost", "catboost", "xgb", "cat"])
    ].sort_values("normalized_brier")
    top_single = single_2024.head(15)
    ablations = registry[
        registry["feature_version"].astype(str).str.contains(
            "V2|V3|ENSEMBLE|INSIGHT", regex=True
        )
    ].sort_values(["fold", "normalized_brier"])
    lines = [
        "# 검증 결과 레지스트리",
        "",
        "`build_validation_registry.py`가 모든 완료 실험을 모아 재생성한다. "
        "모델 선택은 0으로 잘리는 BSS보다 Brier와 normalized Brier를 우선한다.",
        "",
        "## 검증 원칙",
        "",
        "- fold 2023: 2019~2022 학습 → 2023 검증",
        "- fold 2024: 2019~2023 학습 → 2024 검증",
        "- 최종 제출: 선택된 설정으로 2019~2024 전체 재학습",
        "- 2024 검증에는 2024 Trackman 정보를 어떤 형태로도 사용하지 않음",
        "- Trackman 실험은 검증연도 이전 시즌이며 시즌당 500구 이상인 투수-시즌만 허용",
        "",
        f"전체 검증 기록: {len(registry):,}개",
        "",
        "## 2024 단일 모델 상위",
        "",
        top_single[
            [
                "experiment",
                "feature_version",
                "family",
                "trial",
                "brier",
                "bss",
                "auc",
                "target_mean",
                "pred_mean",
                "mean_gap",
            ]
        ].to_markdown(index=False, floatfmt=".8f"),
        "",
        "## 주요 ablation 및 앙상블",
        "",
        ablations[
            [
                "experiment",
                "feature_version",
                "fold",
                "trackman",
                "brier",
                "bss",
                "auc",
                "pred_mean",
                "best_iteration",
            ]
        ].to_markdown(index=False, floatfmt=".8f"),
        "",
        "## 원본 기록",
        "",
        "- 전체 행 단위 결과: `validation_registry.csv`",
        "- Optuna 원본: 각 study DB와 leaderboard CSV",
        "- 실패하거나 성능이 하락한 실험도 삭제하지 않고 함께 보존",
    ]
    (WORK_DIR / "VALIDATION_LOG.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    registry = pd.DataFrame(
        trial_rows()
        + fixed_rows()
        + oof_rows()
        + ensemble_rows()
        + v2_fixed_rows()
        + v2_ablation_rows()
        + failure_model_rows()
        + failure_blend_rows()
        + trackman500_rows()
        + cat_enhanced_rows()
        + strict_embedding_rows()
        + trackman_pitchgroup_rows()
        + trackman_gated_rows()
        + smoothing_grid_rows()
        + direct_brier_rows()
        + enhanced_seed_oof_rows()
        + enhanced_ensemble_rows()
        + insight_feature_rows()
    )
    registry = registry.sort_values(
        ["fold", "normalized_brier", "family", "trial"], na_position="last"
    ).reset_index(drop=True)
    registry.to_csv(WORK_DIR / "validation_registry.csv", index=False)
    write_korean_markdown(registry)
    print(f"rows={len(registry)}")
    print(WORK_DIR / "validation_registry.csv")
    print(WORK_DIR / "VALIDATION_LOG.md")


if __name__ == "__main__":
    main()
