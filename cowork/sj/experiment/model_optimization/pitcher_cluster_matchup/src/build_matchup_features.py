from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "experiment" / "model_optimization" / "pitcher_cluster_matchup"
SEED = 2026


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitcher-configs", required=True)
    parser.add_argument("--batter-k-pairs", default="3-4,4-6,6-8,8-12")
    parser.add_argument("--batter-algorithms", default="kmeans,gmm_diag")
    parser.add_argument("--lambdas", default="100,500,1000,2000")
    parser.add_argument("--half-lives", default="1,2,99")
    parser.add_argument("--profile-lambda", type=float, default=200.0)
    parser.add_argument("--pca-dim", type=int, default=8)
    return parser.parse_args()


def matchup_id(pitcher_config, batter_algorithm, k_pair, smoothing, half_life, pca_dim):
    raw = f"{pitcher_config}|{batter_algorithm}|{k_pair}|{smoothing}|{half_life}|{pca_dim}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"m_{pitcher_config[:7]}_{batter_algorithm[:3]}_b{k_pair[0]}r{k_pair[1]}_l{int(smoothing)}_h{half_life:g}_{digest}"


def pitcher_lookup(config, cutoff):
    path = WORK / "clusters" / config / f"pitcher_lookup_{cutoff}.parquet"
    lookup = pd.read_parquet(path)[
        ["pitcher_id", "pitcher_hand", "cluster_code", "cluster_index", "cohort"]
    ].copy()
    lookup["pitcher_type"] = lookup["cluster_code"].astype(str)
    return lookup


def build_batter_profile(past, profile_lambda):
    global_by_season = past.groupby("season")["control_success"].mean()
    past = past.copy()
    past["success_residual"] = (
        past["control_success"] - past["season"].map(global_by_season)
    ).astype("float32")
    base = (
        past.groupby(["batter_id", "batter_hand"], sort=False)
        .agg(batter_n=("control_success", "size"), batter_resid_sum=("success_residual", "sum"))
        .reset_index()
    )
    base["batter_overall_resid"] = base["batter_resid_sum"] / (
        base["batter_n"] + profile_lambda
    )
    for key, prefix in [("pitcher_type", "vs_pt"), ("pitcher_hand", "vs_ph")]:
        grouped = (
            past.groupby(["batter_id", "batter_hand", key], sort=False)
            .agg(n=("control_success", "size"), resid_sum=("success_residual", "sum"))
            .reset_index()
        )
        grouped["value"] = grouped["resid_sum"] / (grouped["n"] + profile_lambda)
        pivot_value = grouped.pivot(
            index=["batter_id", "batter_hand"], columns=key, values="value"
        )
        pivot_value.columns = [f"bprof_{prefix}_{str(value)}_resid" for value in pivot_value.columns]
        pivot_n = grouped.pivot(
            index=["batter_id", "batter_hand"], columns=key, values="n"
        )
        pivot_n.columns = [f"bprof_{prefix}_{str(value)}_logn" for value in pivot_n.columns]
        pivot_n = np.log1p(pivot_n)
        base = base.merge(
            pivot_value, left_on=["batter_id", "batter_hand"], right_index=True, how="left"
        )
        base = base.merge(
            pivot_n, left_on=["batter_id", "batter_hand"], right_index=True, how="left"
        )
    return base, past


def cluster_batters(profile, algorithm, k_pair, pca_dim):
    feature_columns = [
        column for column in profile
        if column.startswith("batter_overall_") or column.startswith("bprof_")
    ]
    pieces = []
    audit = []
    for hand, k in [(1, k_pair[0]), (2, k_pair[1])]:
        part = profile.loc[profile["batter_hand"].eq(hand)].copy()
        raw = part[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy("float64")
        values = SimpleImputer(strategy="median").fit_transform(raw)
        values = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(values)
        values = np.clip(values, -5.0, 5.0)
        dim = max(1, min(pca_dim, values.shape[1], len(part) - 1))
        embedding = PCA(n_components=dim, random_state=SEED).fit_transform(values)
        if algorithm == "kmeans":
            model = KMeans(n_clusters=k, n_init=30, random_state=SEED)
            labels = model.fit_predict(embedding)
        elif algorithm == "gmm_diag":
            model = GaussianMixture(
                n_components=k, covariance_type="diag", reg_covar=1e-4,
                n_init=5, max_iter=500, random_state=SEED,
            )
            labels = model.fit_predict(embedding)
        else:
            raise ValueError(algorithm)
        # Stable numbering: batters with low pitcher-control residual first.
        overall = part["batter_overall_resid"].to_numpy("float64")
        ordering = sorted(
            range(k),
            key=lambda value: float(np.nanmedian(overall[labels == value])),
        )
        old_to_new = {old: new for new, old in enumerate(ordering)}
        labels = np.asarray([old_to_new[int(value)] for value in labels], dtype="int16")
        result = part[["batter_id", "batter_hand", "batter_n", "batter_overall_resid"]].copy()
        result["batter_cluster"] = labels
        result["batter_type"] = [f"BH{hand}_C{int(value):02d}" for value in labels]
        pieces.append(result)
        counts = np.bincount(labels, minlength=k)
        audit.append({
            "hand": hand,
            "batters": int(len(part)),
            "k": k,
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
        })
    return pd.concat(pieces, ignore_index=True), audit


def shrink_table(frame, keys, residual_column, smoothing, weight_column):
    grouped = frame.groupby(keys, sort=False).agg(
        weighted_resid=("weighted_residual", "sum"),
        effective_n=(weight_column, "sum"),
        raw_n=("control_success", "size"),
        weighted_success=("weighted_success", "sum"),
    ).reset_index()
    grouped[residual_column] = grouped["weighted_resid"] / (
        grouped["effective_n"] + smoothing
    )
    grouped[f"{residual_column}_reliability"] = grouped["effective_n"] / (
        grouped["effective_n"] + smoothing
    )
    grouped[f"{residual_column}_rate"] = grouped["weighted_success"] / grouped["effective_n"]
    return grouped.drop(columns=["weighted_resid", "weighted_success"])


def build_cutoff_features(main, cutoff, pitcher_config, batter_algorithm, k_pair,
                          smoothing, half_life, profile_lambda, pca_dim):
    p_lookup = pitcher_lookup(pitcher_config, cutoff)
    past = main.loc[main["season"].lt(cutoff)].copy()
    past = past.merge(
        p_lookup[["pitcher_id", "pitcher_type"]],
        on="pitcher_id", how="left", validate="many_to_one",
    )
    past["pitcher_type"] = past["pitcher_type"].fillna(
        "H" + past["pitcher_hand"].astype(str) + "_new"
    )
    batter_profile, past = build_batter_profile(past, profile_lambda)
    b_lookup, batter_audit = cluster_batters(
        batter_profile, batter_algorithm, k_pair, pca_dim
    )
    past = past.merge(
        b_lookup[["batter_id", "batter_hand", "batter_type"]],
        on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
    )
    past["batter_type"] = past["batter_type"].fillna(
        "BH" + past["batter_hand"].astype(str) + "_new"
    )
    if half_life >= 90:
        past["recency_weight"] = 1.0
    else:
        past["recency_weight"] = np.power(
            0.5, (cutoff - past["season"].to_numpy("float64")) / half_life
        )
    past["weighted_residual"] = past["success_residual"] * past["recency_weight"]
    past["weighted_success"] = past["control_success"] * past["recency_weight"]
    pair = shrink_table(
        past, ["pitcher_type", "batter_type"], "match_pair_delta",
        smoothing, "recency_weight",
    )
    pitcher_bhand = shrink_table(
        past, ["pitcher_type", "batter_hand"], "match_pitcher_bhand_delta",
        smoothing, "recency_weight",
    )
    phand_batter = shrink_table(
        past, ["pitcher_hand", "batter_type"], "match_phand_batter_delta",
        smoothing, "recency_weight",
    )

    current = main.loc[main["season"].eq(cutoff), [
        "row_id", "season", "pitcher_id", "pitcher_hand", "batter_id", "batter_hand",
    ]].copy()
    current = current.merge(
        p_lookup[["pitcher_id", "pitcher_type"]],
        on="pitcher_id", how="left", validate="many_to_one",
    )
    current["pitcher_type"] = current["pitcher_type"].fillna(
        "H" + current["pitcher_hand"].astype(str) + "_new"
    )
    current = current.merge(
        b_lookup[[
            "batter_id", "batter_hand", "batter_type", "batter_n", "batter_overall_resid"
        ]],
        on=["batter_id", "batter_hand"], how="left", validate="many_to_one",
    )
    current["batter_type"] = current["batter_type"].fillna(
        "BH" + current["batter_hand"].astype(str) + "_new"
    )
    current = current.merge(pair, on=["pitcher_type", "batter_type"], how="left")
    current = current.merge(
        pitcher_bhand, on=["pitcher_type", "batter_hand"], how="left",
        suffixes=("", "_pbh"),
    )
    current = current.merge(
        phand_batter, on=["pitcher_hand", "batter_type"], how="left",
        suffixes=("", "_phb"),
    )
    current["match_batter_known"] = current["batter_n"].notna().astype("int8")
    current["match_pair_known"] = current["match_pair_delta"].notna().astype("int8")
    feature_columns = [
        "batter_n", "batter_overall_resid", "match_batter_known", "match_pair_known",
        "match_pair_delta", "match_pair_delta_reliability", "match_pair_delta_rate",
        "effective_n", "raw_n",
        "match_pitcher_bhand_delta", "match_pitcher_bhand_delta_reliability",
        "match_pitcher_bhand_delta_rate", "effective_n_pbh", "raw_n_pbh",
        "match_phand_batter_delta", "match_phand_batter_delta_reliability",
        "match_phand_batter_delta_rate", "effective_n_phb", "raw_n_phb",
    ]
    output = current[["row_id", "season", *feature_columns]].copy()
    for column in feature_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
        if column.endswith(("delta", "resid")):
            output[column] = output[column].fillna(0.0)
        elif column.endswith(("known", "reliability")):
            output[column] = output[column].fillna(0.0)
        elif column.endswith(("_n", "raw_n", "effective_n", "_pbh", "_phb")):
            output[column] = output[column].fillna(0.0)
    output["batter_n"] = np.log1p(output["batter_n"].fillna(0.0))
    for column in ["effective_n", "raw_n", "effective_n_pbh", "raw_n_pbh", "effective_n_phb", "raw_n_phb"]:
        output[column] = np.log1p(output[column].fillna(0.0))
    for column in feature_columns:
        output[column] = output[column].astype("float32")
    audit = {
        "cutoff": cutoff,
        "max_evidence_season": int(past["season"].max()),
        "pitcher_types": int(past["pitcher_type"].nunique()),
        "batter_types": int(b_lookup["batter_type"].nunique()),
        "pair_cells": int(len(pair)),
        "current_rows": int(len(output)),
        "pair_coverage": float(output["match_pair_known"].mean()),
        "batter_audit": batter_audit,
    }
    return output, b_lookup, audit


def build_config(main, pitcher_config, batter_algorithm, k_pair, smoothing,
                 half_life, profile_lambda, pca_dim):
    config = matchup_id(
        pitcher_config, batter_algorithm, k_pair, smoothing, half_life, pca_dim
    )
    pieces = []
    audits = []
    batter_dir = WORK / "clusters" / "batter" / config
    batter_dir.mkdir(parents=True, exist_ok=True)
    for cutoff in sorted(int(value) for value in main["season"].unique()):
        if cutoff == int(main["season"].min()):
            part = main.loc[main["season"].eq(cutoff), ["row_id", "season"]].copy()
            for column in [
                "batter_n", "batter_overall_resid", "match_batter_known", "match_pair_known",
                "match_pair_delta", "match_pair_delta_reliability", "match_pair_delta_rate",
                "effective_n", "raw_n", "match_pitcher_bhand_delta",
                "match_pitcher_bhand_delta_reliability", "match_pitcher_bhand_delta_rate",
                "effective_n_pbh", "raw_n_pbh", "match_phand_batter_delta",
                "match_phand_batter_delta_reliability", "match_phand_batter_delta_rate",
                "effective_n_phb", "raw_n_phb",
            ]:
                part[column] = np.float32(0.0)
            pieces.append(part)
            continue
        part, b_lookup, audit = build_cutoff_features(
            main, cutoff, pitcher_config, batter_algorithm, k_pair,
            smoothing, half_life, profile_lambda, pca_dim,
        )
        b_lookup.to_parquet(batter_dir / f"batter_lookup_{cutoff}.parquet", index=False)
        pieces.append(part)
        audits.append(audit)
    output = pd.concat(pieces, ignore_index=True)
    output = output.sort_values("row_id").reset_index(drop=True)
    reference = main[["row_id", "season"]].sort_values("row_id").reset_index(drop=True)
    if not output["row_id"].equals(reference["row_id"]):
        raise RuntimeError(f"Row mismatch: {config}")
    path = WORK / "oof" / f"matchup_features_{config}.parquet"
    output.to_parquet(path, index=False)
    record = {
        "matchup_config": config,
        "pitcher_config": pitcher_config,
        "batter_algorithm": batter_algorithm,
        "batter_k_left": k_pair[0],
        "batter_k_right": k_pair[1],
        "smoothing": smoothing,
        "half_life": half_life,
        "pca_dim": pca_dim,
        "rows": int(len(output)),
        "features": int(len(output.columns) - 2),
        "pair_coverage_2024": next(
            (item["pair_coverage"] for item in audits if item["cutoff"] == 2024), np.nan
        ),
        "audit": json.dumps(audits, ensure_ascii=False),
        "path": str(path.relative_to(ROOT)),
    }
    print(json.dumps({key: value for key, value in record.items() if key != "audit"}, ensure_ascii=False), flush=True)
    return record


def main():
    args = parse_args()
    pitcher_configs = [value for value in args.pitcher_configs.split(",") if value]
    k_pairs = [tuple(map(int, value.split("-"))) for value in args.batter_k_pairs.split(",") if value]
    algorithms = [value for value in args.batter_algorithms.split(",") if value]
    smoothings = [float(value) for value in args.lambdas.split(",") if value]
    half_lives = [float(value) for value in args.half_lives.split(",") if value]
    main = pd.read_csv(ROOT / "data" / "train.csv", usecols=[
        "row_id", "season", "pitcher_id", "pitcher_hand", "batter_id", "batter_hand",
        "control_success",
    ])
    records = []
    for pitcher_config in pitcher_configs:
        for algorithm in algorithms:
            for k_pair in k_pairs:
                for smoothing in smoothings:
                    for half_life in half_lives:
                        records.append(build_config(
                            main, pitcher_config, algorithm, k_pair, smoothing,
                            half_life, args.profile_lambda, args.pca_dim,
                        ))
    registry_path = WORK / "reports" / "matchup_registry.csv"
    pd.DataFrame(records).to_csv(registry_path, index=False)
    print(json.dumps({"registry": str(registry_path), "configs": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
