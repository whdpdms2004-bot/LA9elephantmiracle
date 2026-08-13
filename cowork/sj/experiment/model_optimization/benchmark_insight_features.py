from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

from benchmark_v2_ablation import encode_fold
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import ROOT, SEED, TARGET, probability_metrics, recency_weights
from v2_temporal_features import ROW_RATE_SPECS


WORK_DIR = ROOT / "experiment" / "model_optimization"
STUDY_NAME = "xgboost_v2r200_tm500_local_2024"
TRIAL_NUMBER = 93
EPS = 1e-5

COMPONENT_TARGET = {
    "pitcher_success": "control_success",
    "pitcher_reverse": "reverse",
    "pitcher_middle": "middle",
    "pitcher_ball": "ball",
    "pitcher_strike": "strike",
    "batter_success": "control_success",
    "batter_middle": "middle",
}


def logit(values):
    p = np.clip(np.asarray(values, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(values):
    z = np.clip(np.asarray(values, dtype="float64"), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def weighted_prior(season_rates: dict[int, float], target_season: int, half_life: float, fallback: float):
    years = np.asarray([year for year in season_rates if year < target_season], dtype="int16")
    if len(years) == 0:
        return float(fallback)
    values = np.asarray([season_rates[int(year)] for year in years], dtype="float64")
    ages = target_season - years
    weights = np.power(0.5, ages / half_life)
    return float(np.dot(values, weights / weights.sum()))


def build_past_only_lookups(frame: pd.DataFrame, labels: pd.DataFrame):
    if not frame["row_id"].equals(labels["row_id"]):
        raise RuntimeError("Failure component labels are not aligned with train rows")
    seasons = sorted(int(value) for value in frame["season"].unique())
    component_rates = {}
    for target_name in sorted(set(COMPONENT_TARGET.values())):
        source = frame[TARGET] if target_name == TARGET else labels[target_name]
        rate = pd.DataFrame({"season": frame["season"], "value": source}).groupby("season")["value"].mean()
        component_rates[target_name] = {int(k): float(v) for k, v in rate.items()}

    lookups = {}
    audit = []
    for prefix, (rate_column, _, fixed_prior) in ROW_RATE_SPECS.items():
        if prefix not in COMPONENT_TARGET:
            continue
        target_name = COMPONENT_TARGET[prefix]
        actual = component_rates[target_name]
        mean_asof_series = frame.groupby("season")[rate_column].mean()
        mean_asof = {int(k): float(v) for k, v in mean_asof_series.items()}
        prefix_lookup = {}
        for season in seasons:
            source_season = season - 1
            if source_season in actual:
                prior_last = actual[source_season]
                gap_logit = float(logit([prior_last])[0] - logit([mean_asof[source_season]])[0])
            else:
                prior_last = float(fixed_prior)
                gap_logit = 0.0
                source_season = None
            record = {
                "source_season": source_season,
                "prior_last": float(prior_last),
                "prior_ewm1": weighted_prior(actual, season, 1.0, fixed_prior),
                "prior_ewm2": weighted_prior(actual, season, 2.0, fixed_prior),
                "gap_logit_last": float(np.clip(gap_logit, -0.50, 0.50)),
            }
            prefix_lookup[season] = record
            audit.append({"prefix": prefix, "target_season": season, **record})
        lookups[prefix] = prefix_lookup
    return lookups, audit


def add_calibration_features(frame: pd.DataFrame, lookups):
    output = frame.copy()
    gap_columns = []
    prior_columns = []
    for prefix, (rate_column, count_column, fixed_prior) in ROW_RATE_SPECS.items():
        if prefix not in lookups:
            continue
        lookup = lookups[prefix]
        season = output["season"].astype(int)
        prior_last = season.map({key: value["prior_last"] for key, value in lookup.items()}).astype("float32")
        prior_ewm1 = season.map({key: value["prior_ewm1"] for key, value in lookup.items()}).astype("float32")
        prior_ewm2 = season.map({key: value["prior_ewm2"] for key, value in lookup.items()}).astype("float32")
        gap = season.map({key: value["gap_logit_last"] for key, value in lookup.items()}).astype("float32")
        raw = output[rate_column].astype("float64")
        count = output[count_column].clip(lower=0).astype("float64")
        raw_fill = raw.fillna(prior_last.astype("float64"))
        adjusted = sigmoid(logit(raw_fill) + gap.to_numpy("float64"))

        output[f"{prefix}_gap_logit_last"] = gap
        output[f"{prefix}_adjusted_last"] = adjusted.astype("float32")
        gap_columns.extend([f"{prefix}_gap_logit_last", f"{prefix}_adjusted_last"])

        for name, prior in [
            ("last", prior_last),
            ("ewm1", prior_ewm1),
            ("ewm2", prior_ewm2),
        ]:
            column = f"{prefix}_dynamic_{name}_smoothed_200"
            output[column] = ((raw_fill * count + prior.astype("float64") * 200.0) / (count + 200.0)).astype("float32")
            prior_columns.append(column)
        for strength in [150.0, 250.0]:
            column = f"{prefix}_dynamic_last_smoothed_{int(strength)}"
            output[column] = ((raw_fill * count + prior_last.astype("float64") * strength) / (count + strength)).astype("float32")
            prior_columns.append(column)
        column = f"{prefix}_adjusted_smoothed_200"
        output[column] = ((adjusted * count + prior_last.astype("float64") * 200.0) / (count + 200.0)).astype("float32")
        prior_columns.append(column)
    return output, gap_columns, prior_columns


def add_momentum_features(frame: pd.DataFrame):
    output = frame.copy()
    columns = []
    reliability = (output["asof_pitcher_n"].clip(lower=0) / (output["asof_pitcher_n"].clip(lower=0) + 200.0)).astype("float32")
    for component in ["success", "middle"]:
        long_column = f"asof_pitcher_{component}_rate"
        deltas = []
        for window in [1, 3, 5]:
            recent_column = f"asof_pitcher_prev{window}_game_{component}_rate"
            delta = (output[recent_column] - output[long_column]).astype("float32")
            base = f"{component}_momentum_prev{window}"
            output[f"{base}_pos"] = delta.clip(lower=0)
            output[f"{base}_neg"] = (-delta).clip(lower=0)
            output[f"{base}_shrunk"] = delta * reliability
            columns.extend([f"{base}_pos", f"{base}_neg", f"{base}_shrunk"])
            deltas.append(delta)
        matrix = pd.concat(deltas, axis=1)
        output[f"{component}_momentum_median"] = matrix.median(axis=1).astype("float32")
        output[f"{component}_momentum_sign_sum"] = np.sign(matrix).sum(axis=1).astype("float32")
        output[f"{component}_prev1_minus_prev5"] = (
            output[f"asof_pitcher_prev1_game_{component}_rate"]
            - output[f"asof_pitcher_prev5_game_{component}_rate"]
        ).astype("float32")
        columns.extend([
            f"{component}_momentum_median",
            f"{component}_momentum_sign_sum",
            f"{component}_prev1_minus_prev5",
        ])
    return output, columns


def add_context_interactions(frame: pd.DataFrame):
    output = frame.copy()
    is_f = output["game_type"].astype(str).eq("F").astype("float32")
    full_count = (output["balls_before"].eq(3) & output["strikes_before"].eq(2)).astype("float32")
    columns = [
        "is_f_adjusted_reverse",
        "full_count_adjusted_success",
        "late_adjusted_middle",
    ]
    output[columns[0]] = is_f * output["pitcher_reverse_adjusted_last"]
    output[columns[1]] = full_count * output["pitcher_success_adjusted_last"]
    output[columns[2]] = output["inning"].ge(7).astype("float32") * output["pitcher_middle_adjusted_last"]
    return output, columns


def load_local_trial():
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(WORK_DIR / f'{STUDY_NAME}.db').as_posix()}",
    )
    return next(trial for trial in study.trials if trial.number == TRIAL_NUMBER)


def run_one(frame, features, version, fold, trial):
    started = time.time()
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    train_mask, valid_mask, train_x, valid_x = encode_fold(frame, features, fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    model = XGBClassifier(
        **params,
        grow_policy="lossguide",
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED + fold,
        n_jobs=6,
        early_stopping_rounds=220,
    )
    model.fit(
        train_x,
        train_y,
        sample_weight=weights,
        eval_set=[(valid_x, valid_y)],
        verbose=False,
    )
    prediction = model.predict_proba(valid_x)[:, 1]
    row = {
        "experiment": f"xgboost_insight_{version.lower()}",
        "family": "xgboost",
        "feature_version": version,
        "trial": TRIAL_NUMBER,
        "fold": fold,
        "train_through": fold - 1,
        "trackman": True,
        "trackman_cutoff": fold,
        "min_trackman_season_pitches": 500,
        "feature_count": len(features),
        "best_iteration": int(model.best_iteration),
        "elapsed_sec": time.time() - started,
        **probability_metrics(valid_y, prediction),
    }
    pred = pd.DataFrame(
        {
            "row_id": frame.loc[valid_mask, "row_id"].to_numpy(),
            "season": fold,
            TARGET: valid_y,
            "model": row["experiment"],
            "prediction": prediction.astype("float32"),
        }
    )
    print(json.dumps(row, ensure_ascii=False), flush=True)
    del model, train_x, valid_x, train_y, valid_y, weights, prediction
    gc.collect()
    return row, pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        default="INSIGHT_BASE,INSIGHT_GAP,INSIGHT_PRIOR,INSIGHT_MOMENTUM,INSIGHT_CALIB_MOM,INSIGHT_ALL",
    )
    parser.add_argument("--folds", default="2024,2023")
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    frame, base_features = load_enhanced_frame()
    labels = pd.read_parquet(WORK_DIR / "failure_component_labels.parquet")
    lookups, audit = build_past_only_lookups(frame, labels)
    frame, gap_columns, prior_columns = add_calibration_features(frame, lookups)
    frame, momentum_columns = add_momentum_features(frame)
    frame, context_columns = add_context_interactions(frame)

    variants = {
        "INSIGHT_BASE": base_features,
        "INSIGHT_GAP": list(dict.fromkeys(base_features + gap_columns)),
        "INSIGHT_PRIOR": list(dict.fromkeys(base_features + prior_columns)),
        "INSIGHT_MOMENTUM": list(dict.fromkeys(base_features + momentum_columns)),
        "INSIGHT_CALIB_MOM": list(dict.fromkeys(base_features + gap_columns + prior_columns + momentum_columns)),
        "INSIGHT_ALL": list(dict.fromkeys(base_features + gap_columns + prior_columns + momentum_columns + context_columns)),
    }
    prefix_groups = {
        "INSIGHT_PRIOR_SUCCESS": ("pitcher_success_", "batter_success_"),
        "INSIGHT_PRIOR_FAILURE": (
            "pitcher_reverse_",
            "pitcher_middle_",
            "pitcher_ball_",
            "pitcher_strike_",
        ),
        "INSIGHT_PRIOR_REVERSE": ("pitcher_reverse_",),
        "INSIGHT_PRIOR_MIDDLE": ("pitcher_middle_",),
        "INSIGHT_PRIOR_PITCHER": (
            "pitcher_success_",
            "pitcher_reverse_",
            "pitcher_middle_",
            "pitcher_ball_",
            "pitcher_strike_",
        ),
        "INSIGHT_PRIOR_NO_REVERSE": (
            "pitcher_success_",
            "pitcher_middle_",
            "pitcher_ball_",
            "pitcher_strike_",
            "batter_success_",
        ),
    }
    for name, prefixes in prefix_groups.items():
        selected = [column for column in prior_columns if column.startswith(prefixes)]
        variants[name] = list(dict.fromkeys(base_features + selected))
    variants["INSIGHT_PRIOR_LAST"] = list(
        dict.fromkeys(
            base_features
            +
            [column for column in prior_columns if "_dynamic_last_" in column]
        )
    )
    variants["INSIGHT_PRIOR_EWM"] = list(
        dict.fromkeys(
            base_features
            +
            [column for column in prior_columns if "_dynamic_ewm" in column]
        )
    )
    variants["INSIGHT_PRIOR_CORE"] = list(
        dict.fromkeys(
            base_features
            +
            [
                column
                for column in prior_columns
                if column.endswith("dynamic_last_smoothed_200")
                or column.endswith("dynamic_ewm1_smoothed_200")
                or column.endswith("dynamic_ewm2_smoothed_200")
            ]
        )
    )
    success_columns = [
        column
        for column in prior_columns
        if column.startswith(("pitcher_success_", "batter_success_"))
    ]
    success_subsets = {
        "INSIGHT_SUCCESS_PITCHER": [
            column for column in success_columns if column.startswith("pitcher_success_")
        ],
        "INSIGHT_SUCCESS_BATTER": [
            column for column in success_columns if column.startswith("batter_success_")
        ],
        "INSIGHT_SUCCESS_LAST": [
            column for column in success_columns if "_dynamic_last_" in column
        ],
        "INSIGHT_SUCCESS_EWM": [
            column for column in success_columns if "_dynamic_ewm" in column
        ],
        "INSIGHT_SUCCESS_ADJUSTED": [
            column for column in success_columns if "_adjusted_smoothed_" in column
        ],
        "INSIGHT_SUCCESS_CORE": [
            column
            for column in success_columns
            if column.endswith("dynamic_last_smoothed_200")
            or column.endswith("dynamic_ewm1_smoothed_200")
            or column.endswith("dynamic_ewm2_smoothed_200")
        ],
    }
    for name, selected in success_subsets.items():
        variants[name] = list(dict.fromkeys(base_features + selected))
    requested_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(requested_variants) - set(variants))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    trial = load_local_trial()
    results = []
    predictions = []
    screen = []
    for fold in folds:
        for version in requested_variants:
            row, pred = run_one(frame, variants[version], version, fold, trial)
            results.append(row)
            predictions.append(pred)

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    result_path = WORK_DIR / f"insight_feature_ablation_results{suffix}.csv"
    prediction_path = WORK_DIR / f"insight_feature_ablation_predictions{suffix}.parquet"
    summary_path = WORK_DIR / f"insight_feature_ablation_summary{suffix}.json"
    audit_path = WORK_DIR / f"insight_feature_leakage_audit{suffix}.json"
    pd.DataFrame(results).to_csv(result_path, index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(prediction_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "base_feature_count": len(base_features),
                "trial": TRIAL_NUMBER,
                "study": STUDY_NAME,
                "variant_feature_counts": {key: len(value) for key, value in variants.items()},
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "rule": "Every target-season row uses component priors and calibration gaps from strictly earlier seasons only.",
                "max_source_before_target": all(
                    item["source_season"] is None or item["source_season"] < item["target_season"]
                    for item in audit
                ),
                "records": audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"results": str(result_path), "summary": str(summary_path), "audit": str(audit_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
