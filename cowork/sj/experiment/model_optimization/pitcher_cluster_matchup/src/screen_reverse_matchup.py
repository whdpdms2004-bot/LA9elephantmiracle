from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
PITCHER_CONFIG = "com_gmm_p8_l2r4_7de7125e"
BATTER_CONFIG = "m_com_gmm_kme_b3r4_l1000_h1_d72788b5"
MODEL_NAME = "xgboost_insight_insight_success_adjusted"
CUTOFFS = [2022, 2023, 2024]
CENTER_MODES = ["season", "context", "asof_pitcher"]
SMOOTHINGS = [500.0, 1000.0, 2000.0]
HALF_LIVES = [1.0, 2.0]
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
SUCCESS_FEATURES = [
    "match_pair_delta",
    "match_pair_delta_reliability",
    "match_pair_delta_rate",
    "match_pair_known",
]
REVERSE_FEATURES = [
    "reverse_pair_delta",
    "reverse_pair_delta_reliability",
    "reverse_pair_rate",
    "reverse_pair_known",
]


def config_name(center_mode: str, smoothing: float, half_life: float) -> str:
    raw = f"{center_mode}|{smoothing}|{half_life}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"reverse_{center_mode}_l{int(smoothing)}_h{int(half_life)}_{digest}"


def load_main() -> pd.DataFrame:
    usecols = [
        "row_id", "season", "pitcher_id", "pitcher_hand", "batter_id",
        "batter_hand", "balls_before", "strikes_before",
        "asof_pitcher_reverse_rate",
    ]
    main = pd.read_csv(ROOT / "data" / "train.csv", usecols=usecols)
    labels = pd.read_parquet(
        MODEL_DIR / "failure_component_labels.parquet",
        columns=["row_id", "reverse"],
    )
    if not main["row_id"].equals(labels["row_id"]):
        raise RuntimeError("Failure labels are not row-aligned")
    main["reverse"] = labels["reverse"].astype("float32")
    return main


def attach_types(frame: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    pitcher = pd.read_parquet(
        WORK / "clusters" / PITCHER_CONFIG / f"pitcher_lookup_{cutoff}.parquet",
        columns=["pitcher_id", "cluster_code"],
    ).rename(columns={"cluster_code": "pitcher_type"})
    batter = pd.read_parquet(
        WORK / "clusters" / "batter" / BATTER_CONFIG / f"batter_lookup_{cutoff}.parquet",
        columns=["batter_id", "batter_hand", "batter_type"],
    )
    output = frame.merge(pitcher, on="pitcher_id", how="left", validate="many_to_one")
    output["pitcher_type"] = output["pitcher_type"].fillna(
        "H" + output["pitcher_hand"].astype(str) + "_new"
    )
    output = output.merge(
        batter,
        on=["batter_id", "batter_hand"],
        how="left",
        validate="many_to_one",
    )
    output["batter_type"] = output["batter_type"].fillna(
        "BH" + output["batter_hand"].astype(str) + "_new"
    )
    return output


def add_reverse_residual(past: pd.DataFrame, mode: str) -> pd.DataFrame:
    output = past.loc[past["reverse"].notna()].copy()
    season_rate = output.groupby("season")["reverse"].transform("mean")
    if mode == "season":
        expected = season_rate
    elif mode == "context":
        keys = [
            "season", "pitcher_hand", "batter_hand", "balls_before", "strikes_before"
        ]
        expected = output.groupby(keys)["reverse"].transform("mean").fillna(season_rate)
    elif mode == "asof_pitcher":
        context = output.groupby(
            ["season", "pitcher_hand", "batter_hand", "balls_before", "strikes_before"]
        )["reverse"].transform("mean")
        expected = output["asof_pitcher_reverse_rate"].fillna(context).fillna(season_rate)
    else:
        raise ValueError(mode)
    output["reverse_residual"] = output["reverse"] - expected.clip(0.01, 0.99)
    return output


def aggregate_pair(past: pd.DataFrame, cutoff: int, half_life: float):
    weight = np.power(
        0.5, (cutoff - past["season"].to_numpy("float64")) / half_life
    )
    work = past[["pitcher_type", "batter_type"]].copy()
    work["weighted_residual"] = past["reverse_residual"].to_numpy("float64") * weight
    work["weighted_reverse"] = past["reverse"].to_numpy("float64") * weight
    work["recency_weight"] = weight
    return work.groupby(["pitcher_type", "batter_type"], sort=False).agg(
        weighted_residual=("weighted_residual", "sum"),
        weighted_reverse=("weighted_reverse", "sum"),
        effective_n=("recency_weight", "sum"),
        raw_n=("recency_weight", "size"),
    ).reset_index()


def build_features(main: pd.DataFrame):
    by_config = {
        config_name(mode, smoothing, half_life): []
        for mode in CENTER_MODES
        for smoothing in SMOOTHINGS
        for half_life in HALF_LIVES
    }
    audits = []
    for cutoff in CUTOFFS:
        typed = attach_types(main.loc[main["season"].le(cutoff)].copy(), cutoff)
        current = typed.loc[typed["season"].eq(cutoff), [
            "row_id", "season", "pitcher_type", "batter_type"
        ]].copy()
        for mode in CENTER_MODES:
            past = add_reverse_residual(typed.loc[typed["season"].lt(cutoff)], mode)
            for half_life in HALF_LIVES:
                grouped = aggregate_pair(past, cutoff, half_life)
                for smoothing in SMOOTHINGS:
                    config = config_name(mode, smoothing, half_life)
                    pair = grouped[[
                        "pitcher_type", "batter_type", "weighted_residual",
                        "weighted_reverse", "effective_n",
                    ]].copy()
                    pair["reverse_pair_delta"] = pair["weighted_residual"] / (
                        pair["effective_n"] + smoothing
                    )
                    pair["reverse_pair_delta_reliability"] = pair["effective_n"] / (
                        pair["effective_n"] + smoothing
                    )
                    pair["reverse_pair_rate"] = pair["weighted_reverse"] / pair["effective_n"]
                    out = current.merge(
                        pair[["pitcher_type", "batter_type", *REVERSE_FEATURES[:-1]]],
                        on=["pitcher_type", "batter_type"],
                        how="left",
                        validate="many_to_one",
                    )
                    out["reverse_pair_known"] = out["reverse_pair_delta"].notna().astype("float32")
                    out["reverse_pair_delta"] = out["reverse_pair_delta"].fillna(0.0)
                    out["reverse_pair_delta_reliability"] = out[
                        "reverse_pair_delta_reliability"
                    ].fillna(0.0)
                    for column in REVERSE_FEATURES:
                        out[column] = out[column].astype("float32")
                    by_config[config].append(out[["row_id", "season", *REVERSE_FEATURES]])
                    audits.append({
                        "config": config,
                        "cutoff": cutoff,
                        "center_mode": mode,
                        "smoothing": smoothing,
                        "half_life": half_life,
                        "pair_cells": int(len(pair)),
                        "coverage": float(out["reverse_pair_known"].mean()),
                        "residual_mean": float(past["reverse_residual"].mean()),
                    })
        print(json.dumps({"built_cutoff": cutoff}, ensure_ascii=False), flush=True)

    cache_dir = WORK / "oof" / "reverse"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for config, pieces in by_config.items():
        pd.concat(pieces, ignore_index=True).to_parquet(
            cache_dir / f"{config}.parquet", index=False
        )
    pd.DataFrame(audits).to_csv(
        WORK / "reports" / "reverse_matchup_feature_audit.csv", index=False
    )
    return list(by_config)


def load_base() -> pd.DataFrame:
    paths = [
        MODEL_DIR / "insight_feature_ablation_predictions_cluster_base_2022.parquet",
        MODEL_DIR / "insight_feature_ablation_predictions_success_adjusted_2023.parquet",
        MODEL_DIR / "insight_feature_ablation_predictions_success_screen_2024.parquet",
    ]
    pieces = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame["model"].nunique() > 1:
            frame = frame.loc[frame["model"].eq(MODEL_NAME)]
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def fit_fold(frame: pd.DataFrame, features, alpha: float, train_year: int, valid_year: int):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = (
        frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    ).to_numpy("float64")
    model.fit(frame.loc[train, features], residual)
    correction = np.clip(model.predict(frame.loc[valid, features]), -0.05, 0.05)
    y = frame.loc[valid, "control_success"].to_numpy("float64")
    base = frame.loc[valid, "prediction"].to_numpy("float64")
    pred = np.clip(base + correction, 1e-6, 1 - 1e-6)
    base_brier = float(np.mean((base - y) ** 2))
    candidate_brier = float(np.mean((pred - y) ** 2))
    denominator = float(y.mean() * (1.0 - y.mean()))
    return {
        "delta_brier": candidate_brier - base_brier,
        "normalized_delta": (candidate_brier - base_brier) / denominator,
        "correction_std": float(correction.std()),
    }


def screen(configs):
    base = load_base()
    success = pd.read_parquet(
        WORK / "oof" / f"matchup_features_{BATTER_CONFIG}.parquet",
        columns=["row_id", "season", *SUCCESS_FEATURES],
    )
    base = base.merge(success, on=["row_id", "season"], validate="one_to_one")
    rows = []
    for config in configs:
        reverse = pd.read_parquet(WORK / "oof" / "reverse" / f"{config}.parquet")
        frame = base.merge(reverse, on=["row_id", "season"], validate="one_to_one")
        parts = config.split("_")
        for feature_mode, features in [
            ("reverse_only", REVERSE_FEATURES),
            ("success_plus_reverse", SUCCESS_FEATURES + REVERSE_FEATURES),
        ]:
            for alpha in ALPHAS:
                f23 = fit_fold(frame, features, alpha, 2022, 2023)
                f24 = fit_fold(frame, features, alpha, 2023, 2024)
                objective = (
                    0.30 * f23["normalized_delta"]
                    + 0.70 * f24["normalized_delta"]
                    + 0.50 * max(f23["normalized_delta"], f24["normalized_delta"], 0.0)
                )
                rows.append({
                    "config": config,
                    "center_mode": "_".join(parts[1:-3]),
                    "smoothing": float(parts[-3][1:]),
                    "half_life": float(parts[-2][1:]),
                    "feature_mode": feature_mode,
                    "alpha": alpha,
                    "f23_delta_brier": f23["delta_brier"],
                    "f24_delta_brier": f24["delta_brier"],
                    "f23_correction_std": f23["correction_std"],
                    "f24_correction_std": f24["correction_std"],
                    "both_improve": f23["delta_brier"] < 0 and f24["delta_brier"] < 0,
                    "robust_objective": objective,
                })
        print(json.dumps({"screened": config}, ensure_ascii=False), flush=True)
    result = pd.DataFrame(rows).sort_values("robust_objective")
    result.to_csv(WORK / "reports" / "reverse_matchup_screen.csv", index=False)
    print("\nTOP 20")
    print(result.head(20).to_string(index=False))
    print(json.dumps({
        "runs": len(result),
        "both_improve": int(result["both_improve"].sum()),
        "best": result.iloc[0].to_dict(),
    }, ensure_ascii=False, indent=2))


def main():
    main_frame = load_main()
    configs = build_features(main_frame)
    screen(configs)


if __name__ == "__main__":
    main()
