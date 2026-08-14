from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(MODEL_DIR))

from analyze_r_focus import load_fold_predictions  # noqa: E402


CLUSTER_DIR = WORK / "clusters_preprocess_v2"
REGISTRY = WORK / "reports" / "cluster_registry_preprocess_v2.csv"
CUTOFFS = [2022, 2023, 2024]
BATTER_K = [(2, 3), (3, 4), (4, 6)]
SMOOTHINGS = [500.0, 1000.0, 2000.0]
ALPHAS = [100.0, 1000.0, 10000.0]
PROFILE_LAMBDA = 200.0
HALF_LIFE = 1.0
FEATURES = [
    "middle_pair_delta", "middle_pair_delta_reliability",
    "middle_pair_rate", "middle_pair_known",
]
SUCCESS_SCALES = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
REVERSE_SCALES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70]
MIDDLE_SCALES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]


def load_main() -> pd.DataFrame:
    main = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=[
            "row_id", "season", "game_type", "pitcher_id", "pitcher_hand",
            "batter_id", "batter_hand", "balls_before", "strikes_before",
        ],
    )
    labels = pd.read_parquet(
        MODEL_DIR / "failure_component_labels.parquet",
        columns=["row_id", "middle"],
    )
    if not main["row_id"].equals(labels["row_id"]):
        raise RuntimeError("Failure labels are not row-aligned")
    main["middle"] = labels["middle"].astype("float32")
    return main


def selected_pitcher_configs() -> list[str]:
    registry = pd.read_csv(REGISTRY)
    available = set(registry["config_id"])
    # Use one representative per feature contract.  The registry deliberately
    # contains both compact/no-indicator and compact/missing-indicator variants,
    # so representation alone is not a unique selector.
    selected = [
        "pv2_all_r5_a6fc0d65",
        "pv2_compact_mi_r5_2fec3702",
        "pv2_physical_mi_r5_e8dd683a",
    ]
    selected = [config for config in selected if config in available]
    if len(selected) != 3:
        raise RuntimeError(f"Expected 3 pitcher configs, got {selected}")
    return selected


def attach_pitcher_type(frame: pd.DataFrame, cutoff: int, config: str) -> pd.DataFrame:
    lookup = pd.read_parquet(
        CLUSTER_DIR / config / f"pitcher_lookup_{cutoff}.parquet",
        columns=["pitcher_id", "cluster_code"],
    ).rename(columns={"cluster_code": "pitcher_type"})
    output = frame.merge(lookup, on="pitcher_id", how="left", validate="many_to_one")
    output["pitcher_type"] = output["pitcher_type"].fillna(
        "PV2H" + output["pitcher_hand"].astype(str) + "_new"
    )
    return output


def add_middle_residual(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[
        frame["game_type"].eq("R") & frame["middle"].notna()
    ].copy()
    keys = ["season", "pitcher_hand", "batter_hand", "balls_before", "strikes_before"]
    expected = output.groupby(keys, observed=True)["middle"].transform("mean")
    season_rate = output.groupby("season")["middle"].transform("mean")
    output["middle_residual"] = output["middle"] - expected.fillna(season_rate)
    return output


def batter_profile(past: pd.DataFrame) -> pd.DataFrame:
    base = past.groupby(["batter_id", "batter_hand"], sort=False).agg(
        batter_n=("middle", "size"), resid_sum=("middle_residual", "sum")
    ).reset_index()
    base["batter_middle_resid"] = base["resid_sum"] / (
        base["batter_n"] + PROFILE_LAMBDA
    )
    for key, prefix in [("pitcher_type", "vs_pt"), ("pitcher_hand", "vs_ph")]:
        grouped = past.groupby(["batter_id", "batter_hand", key], sort=False).agg(
            n=("middle", "size"), resid_sum=("middle_residual", "sum")
        ).reset_index()
        grouped["value"] = grouped["resid_sum"] / (grouped["n"] + PROFILE_LAMBDA)
        value = grouped.pivot(
            index=["batter_id", "batter_hand"], columns=key, values="value"
        )
        value.columns = [f"midprof_{prefix}_{column}_resid" for column in value.columns]
        count = grouped.pivot(
            index=["batter_id", "batter_hand"], columns=key, values="n"
        )
        count.columns = [f"midprof_{prefix}_{column}_logn" for column in count.columns]
        base = base.merge(value, left_on=["batter_id", "batter_hand"], right_index=True, how="left")
        base = base.merge(np.log1p(count), left_on=["batter_id", "batter_hand"], right_index=True, how="left")
    return base


def cluster_batters(profile: pd.DataFrame, k_pair: tuple[int, int], seed: int = 2026):
    columns = [
        column for column in profile
        if column.startswith("batter_middle_") or column.startswith("midprof_")
    ]
    pieces = []
    audit = []
    for hand, k in [(1, k_pair[0]), (2, k_pair[1])]:
        part = profile.loc[profile["batter_hand"].eq(hand)].copy()
        raw = part[columns].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
        value = SimpleImputer(strategy="median").fit_transform(raw)
        value = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(value)
        value = np.clip(value, -5.0, 5.0)
        dim = max(1, min(8, value.shape[1], len(part) - 1))
        embedding = PCA(n_components=dim, random_state=seed).fit_transform(value)
        labels = KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(embedding)
        order = sorted(
            range(k),
            key=lambda label: float(part.loc[labels == label, "batter_middle_resid"].median()),
        )
        mapping = {old: new for new, old in enumerate(order)}
        labels = np.asarray([mapping[int(label)] for label in labels], dtype="int16")
        result = part[["batter_id", "batter_hand"]].copy()
        result["batter_type"] = [f"MBH{hand}_C{label:02d}" for label in labels]
        pieces.append(result)
        counts = np.bincount(labels, minlength=k)
        audit.append({
            "hand": hand, "k": k, "batters": len(part),
            "min_cluster_size": int(counts.min()),
        })
    return pd.concat(pieces, ignore_index=True), audit


def build_features(main: pd.DataFrame, pitcher_config: str, k_pair: tuple[int, int], smoothing: float):
    pieces = []
    audits = []
    for cutoff in CUTOFFS:
        typed = attach_pitcher_type(
            main.loc[main["season"].le(cutoff)].copy(), cutoff, pitcher_config
        )
        past = add_middle_residual(typed.loc[typed["season"].lt(cutoff)])
        lookup, audit = cluster_batters(batter_profile(past), k_pair)
        typed_past = past.merge(
            lookup, on=["batter_id", "batter_hand"], how="left", validate="many_to_one"
        )
        typed_past["batter_type"] = typed_past["batter_type"].fillna(
            "MBH" + typed_past["batter_hand"].astype(str) + "_new"
        )
        weight = np.power(
            0.5, (cutoff - typed_past["season"].to_numpy("float64")) / HALF_LIFE
        )
        work = typed_past[["pitcher_type", "batter_type"]].copy()
        work["weighted_residual"] = typed_past["middle_residual"].to_numpy(float) * weight
        work["weighted_middle"] = typed_past["middle"].to_numpy(float) * weight
        work["weight"] = weight
        pair = work.groupby(["pitcher_type", "batter_type"], sort=False).agg(
            weighted_residual=("weighted_residual", "sum"),
            weighted_middle=("weighted_middle", "sum"),
            effective_n=("weight", "sum"),
        ).reset_index()
        pair["middle_pair_delta"] = pair["weighted_residual"] / (
            pair["effective_n"] + smoothing
        )
        pair["middle_pair_delta_reliability"] = pair["effective_n"] / (
            pair["effective_n"] + smoothing
        )
        pair["middle_pair_rate"] = pair["weighted_middle"] / pair["effective_n"]

        current = typed.loc[typed["season"].eq(cutoff), [
            "row_id", "season", "game_type", "pitcher_type", "batter_id", "batter_hand"
        ]].merge(
            lookup, on=["batter_id", "batter_hand"], how="left", validate="many_to_one"
        )
        current["batter_type"] = current["batter_type"].fillna(
            "MBH" + current["batter_hand"].astype(str) + "_new"
        )
        current = current.merge(
            pair[["pitcher_type", "batter_type", *FEATURES[:-1]]],
            on=["pitcher_type", "batter_type"], how="left", validate="many_to_one",
        )
        is_r = current["game_type"].eq("R")
        current["middle_pair_known"] = (
            is_r & current["middle_pair_delta"].notna()
        ).astype("float32")
        for column in FEATURES[:-1]:
            current.loc[~is_r, column] = np.nan
        current["middle_pair_delta"] = current["middle_pair_delta"].fillna(0.0)
        current["middle_pair_delta_reliability"] = current[
            "middle_pair_delta_reliability"
        ].fillna(0.0)
        for column in FEATURES:
            current[column] = current[column].astype("float32")
        pieces.append(current[["row_id", "season", *FEATURES]])
        audits.append({
            "cutoff": cutoff, "pair_cells": len(pair),
            "coverage_r": float(current.loc[is_r, "middle_pair_known"].mean()),
            "batter_audit": json.dumps(audit),
        })
    return pd.concat(pieces, ignore_index=True), audits


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
            frame = frame.loc[frame["model"].eq("xgboost_insight_insight_success_adjusted")]
        pieces.append(frame[["row_id", "season", "control_success", "prediction"]])
    return pd.concat(pieces, ignore_index=True)


def middle_correction(frame: pd.DataFrame, alpha: float, train_year: int, valid_year: int):
    train = frame["season"].eq(train_year) & frame["game_type"].eq("R")
    valid = frame["season"].eq(valid_year)
    valid_r = valid & frame["game_type"].eq("R")
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    model.fit(frame.loc[train, FEATURES], residual)
    output = np.zeros(int(valid.sum()), dtype="float64")
    local_r = frame.loc[valid, "game_type"].eq("R").to_numpy()
    output[local_r] = np.clip(
        model.predict(frame.loc[valid_r, FEATURES]), -0.05, 0.05
    )
    return output


def stats(error: np.ndarray, corrections: np.ndarray, mask: np.ndarray):
    e = error[mask]
    x = corrections[mask]
    return {
        "linear": np.mean(x * e[:, None], axis=0),
        "quadratic": (x.T @ x) / len(x),
    }


def delta(stat: dict, scales: np.ndarray) -> float:
    return float(2.0 * stat["linear"] @ scales + scales @ stat["quadratic"] @ scales)


def tune(frame: pd.DataFrame, middle: dict[int, np.ndarray], existing: dict[int, pd.DataFrame]):
    fold_stats = {}
    denominators = {}
    r_denominators = {}
    for year in [2023, 2024]:
        valid = frame["season"].eq(year)
        part = frame.loc[valid]
        existing_part = existing[year].set_index("row_id").loc[part["row_id"]]
        y = part["control_success"].to_numpy(float)
        error = part["prediction"].to_numpy(float) - y
        corrections = np.column_stack([
            existing_part["success_correction"].to_numpy(float),
            existing_part["reverse_correction"].to_numpy(float),
            middle[year],
        ])
        r = part["game_type"].eq("R").to_numpy()
        fold_stats[year] = {
            "all": stats(error, corrections, np.ones(len(part), dtype=bool)),
            "r": stats(error, corrections, r),
        }
        denominators[year] = float(y.mean() * (1.0 - y.mean()))
        yr = y[r]
        r_denominators[year] = float(yr.mean() * (1.0 - yr.mean()))

    rows = []
    for scales in itertools.product(SUCCESS_SCALES, REVERSE_SCALES, MIDDLE_SCALES):
        scale = np.asarray(scales, dtype=float)
        all_delta = {year: delta(fold_stats[year]["all"], scale) for year in [2023, 2024]}
        r_delta = {year: delta(fold_stats[year]["r"], scale) for year in [2023, 2024]}
        overall = (
            0.30 * all_delta[2023] / denominators[2023]
            + 0.70 * all_delta[2024] / denominators[2024]
            + 0.50 * max(
                all_delta[2023] / denominators[2023],
                all_delta[2024] / denominators[2024], 0.0,
            )
        )
        nr23 = r_delta[2023] / r_denominators[2023]
        nr24 = r_delta[2024] / r_denominators[2024]
        objective = overall + 0.25 * (0.30 * nr23 + 0.70 * nr24) + max(nr23, nr24, 0.0)
        rows.append({
            "success_scale": scales[0], "reverse_scale": scales[1],
            "middle_scale": scales[2],
            "val2023_delta_brier": all_delta[2023],
            "val2024_delta_brier": all_delta[2024],
            "val2023_r_delta_brier": r_delta[2023],
            "val2024_r_delta_brier": r_delta[2024],
            "both_improve": all_delta[2023] < 0 and all_delta[2024] < 0,
            "r_both_improve": r_delta[2023] < 0 and r_delta[2024] < 0,
            "objective": objective,
        })
    result = pd.DataFrame(rows)
    eligible = result.loc[result["both_improve"] & result["r_both_improve"]]
    if len(eligible):
        best = eligible.sort_values("objective").iloc[0]
    else:
        best = result.sort_values("objective").iloc[0]
    return result, best


def outer_bss(frame: pd.DataFrame, middle_2024: np.ndarray, existing_2024: pd.DataFrame, best):
    part = frame.loc[frame["season"].eq(2024)]
    existing_part = existing_2024.set_index("row_id").loc[part["row_id"]]
    corrected = np.clip(
        part["prediction"].to_numpy(float)
        + float(best["success_scale"]) * existing_part["success_correction"].to_numpy(float)
        + float(best["reverse_scale"]) * existing_part["reverse_correction"].to_numpy(float)
        + float(best["middle_scale"]) * middle_2024,
        1e-6, 1.0 - 1e-6,
    )
    perf = existing_part["ensemble"].to_numpy(float)
    y = part["control_success"].to_numpy(float)
    denominator = y.mean() * (1.0 - y.mean())
    candidates = []
    for weight in np.round(np.arange(0.45, 0.701, 0.001), 3):
        prediction = weight * corrected + (1.0 - weight) * perf
        brier = float(np.mean((prediction - y) ** 2))
        candidates.append((brier, weight, 100000.0 * (1.0 - brier / denominator)))
    return min(candidates)


def main() -> None:
    main_frame = load_main()
    base = load_base().merge(
        main_frame[["row_id", "game_type"]], on="row_id", validate="one_to_one"
    )
    existing = load_fold_predictions()
    all_rows = []
    best_rows = []
    audit_rows = []
    for pitcher_config in selected_pitcher_configs():
        for k_pair in BATTER_K:
            for smoothing in SMOOTHINGS:
                features, audits = build_features(main_frame, pitcher_config, k_pair, smoothing)
                frame = base.merge(features, on=["row_id", "season"], validate="one_to_one")
                for alpha in ALPHAS:
                    middle = {
                        year: middle_correction(frame, alpha, train_year, year)
                        for train_year, year in [(2022, 2023), (2023, 2024)]
                    }
                    grid, best = tune(frame, middle, existing)
                    brier, weight, bss = outer_bss(frame, middle[2024], existing[2024], best)
                    metadata = {
                        "pitcher_config": pitcher_config,
                        "batter_k_left": k_pair[0], "batter_k_right": k_pair[1],
                        "smoothing": smoothing, "alpha": alpha,
                    }
                    grid = grid.assign(**metadata)
                    all_rows.append(grid)
                    row = {**metadata, **best.to_dict(),
                           "outer_weight": weight, "outer_brier_2024": brier,
                           "outer_bss_2024": bss}
                    best_rows.append(row)
                audit_rows.extend({
                    "pitcher_config": pitcher_config,
                    "batter_k": str(k_pair), "smoothing": smoothing, **audit,
                } for audit in audits)
                print(json.dumps({
                    "pitcher_config": pitcher_config, "batter_k": k_pair,
                    "smoothing": smoothing, "completed": True,
                }, ensure_ascii=False), flush=True)
    full = pd.concat(all_rows, ignore_index=True).sort_values("objective")
    best = pd.DataFrame(best_rows).sort_values(
        ["r_both_improve", "both_improve", "objective"], ascending=[False, False, True]
    )
    full.to_csv(WORK / "reports" / "r_middle_preprocess_grid.csv", index=False)
    best.to_csv(WORK / "reports" / "r_middle_preprocess_best.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(
        WORK / "reports" / "r_middle_preprocess_audit.csv", index=False
    )
    print("\nTOP 20")
    print(best.head(20).to_string(index=False))
    print(json.dumps({
        "configs": len(best),
        "r_both_improve": int(best["r_both_improve"].sum()),
        "best": best.iloc[0].to_dict(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
