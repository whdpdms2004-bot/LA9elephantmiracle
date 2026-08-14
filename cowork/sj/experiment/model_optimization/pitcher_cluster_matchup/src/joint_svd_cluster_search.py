from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

from deep_pitcher_cluster_search import (  # noqa: E402
    CUTOFFS,
    REVERSE_FEATURES,
    SUCCESS_CONFIG,
    SUCCESS_FEATURES,
    brier_delta_from_terms,
    brier_terms,
    load_base,
    robust_objective,
)
from screen_reverse_batter_clusters import add_context_residual, load_main  # noqa: E402


SEED = 2026
MATRIX_LAMBDAS = [100.0, 500.0]
SVD_DIMS = [4, 8, 16]
PITCHER_K = [(2, 4), (3, 6), (4, 8)]
BATTER_K = [(3, 4), (4, 6), (6, 8)]
SMOOTHING = 1000.0
HALF_LIFE = 1.0
RIDGE_ALPHAS = [100.0, 1000.0, 10000.0]
SUCCESS_SCALES = [0.10, 0.20, 0.25, 0.30, 0.40]
REVERSE_SCALES = [0.25, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]
MIN_PITCHER_N = 100
MIN_BATTER_N = 50
DIRECT_FEATURES = [
    "jsvd_pair_score", "jsvd_pitcher_norm", "jsvd_batter_norm", "jsvd_pair_known",
]


def config_id(matrix_lambda, dim, pitcher_k, batter_k):
    raw = f"{matrix_lambda}|{dim}|{pitcher_k}|{batter_k}|{SEED}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return (
        f"jsvd_l{int(matrix_lambda)}_d{dim}_"
        f"p{pitcher_k[0]}r{pitcher_k[1]}_"
        f"b{batter_k[0]}r{batter_k[1]}_{digest}"
    )


def build_joint_embedding(past, matrix_lambda, dim, seed=SEED):
    """Fit a leakage-safe pitcher-by-batter residual SVD using seasons before cutoff."""
    weight = np.power(
        0.5,
        (past["cutoff"].iloc[0] - past["season"].to_numpy("float64")) / HALF_LIFE,
    )
    work = past[["pitcher_id", "batter_id"]].copy()
    work["weighted_residual"] = past["reverse_residual"].to_numpy(float) * weight
    work["effective_n"] = weight
    cells = work.groupby(["pitcher_id", "batter_id"], sort=False).agg(
        weighted_residual=("weighted_residual", "sum"),
        effective_n=("effective_n", "sum"),
    ).reset_index()

    pitcher_n = work.groupby("pitcher_id")["effective_n"].sum()
    batter_n = work.groupby("batter_id")["effective_n"].sum()
    pitchers = pitcher_n.index[pitcher_n.ge(MIN_PITCHER_N)]
    batters = batter_n.index[batter_n.ge(MIN_BATTER_N)]
    cells = cells.loc[
        cells["pitcher_id"].isin(pitchers) & cells["batter_id"].isin(batters)
    ].copy()
    pitcher_index = pd.Index(pitchers).sort_values()
    batter_index = pd.Index(batters).sort_values()
    row = pitcher_index.get_indexer(cells["pitcher_id"])
    col = batter_index.get_indexer(cells["batter_id"])
    reliability = cells["effective_n"].to_numpy(float) / (
        cells["effective_n"].to_numpy(float) + matrix_lambda
    )
    value = cells["weighted_residual"].to_numpy(float) / (
        cells["effective_n"].to_numpy(float) + matrix_lambda
    )
    # Reliability weighting prevents a handful of low-count matchups dominating SVD.
    value *= np.sqrt(reliability)
    matrix = sparse.csr_matrix(
        (value, (row, col)), shape=(len(pitcher_index), len(batter_index))
    )
    n_components = max(1, min(dim, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=seed)
    pitcher_embedding = svd.fit_transform(matrix)
    batter_basis = svd.components_.T
    batter_embedding = batter_basis * svd.singular_values_[None, :]
    return (
        pd.DataFrame(pitcher_embedding, index=pitcher_index),
        pd.DataFrame(batter_embedding, index=batter_index),
        pd.DataFrame(batter_basis, index=batter_index),
        {
            "pitchers_embedded": int(len(pitcher_index)),
            "batters_embedded": int(len(batter_index)),
            "matrix_cells": int(len(cells)),
            "explained_variance": float(svd.explained_variance_ratio_.sum()),
        },
    )


def hand_cluster(
    embedding, hand_lookup, k_pair, prefix, cluster_mode="standard", seed=SEED
):
    rows = []
    audit = []
    for hand, k in [(1, k_pair[0]), (2, k_pair[1])]:
        ids = hand_lookup.index[hand_lookup.eq(hand)].intersection(embedding.index)
        values = embedding.loc[ids].to_numpy("float64")
        if len(ids) < k:
            raise RuntimeError(f"Not enough {prefix} hand={hand}: {len(ids)} < {k}")
        values = StandardScaler().fit_transform(values)
        if cluster_mode == "unit":
            values = normalize(values, norm="l2")
        elif cluster_mode == "quantile":
            values = QuantileTransformer(
                n_quantiles=min(100, len(values)), output_distribution="normal",
                random_state=seed,
            ).fit_transform(values)
        elif cluster_mode != "standard":
            raise ValueError(f"Unknown cluster_mode={cluster_mode}")
        labels = KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(values)
        frame = pd.DataFrame({f"{prefix}_id": ids, f"{prefix}_hand": hand})
        frame[f"{prefix}_cluster"] = labels.astype("int16")
        frame[f"{prefix}_type"] = [f"JS{prefix[0].upper()}H{hand}_C{x:02d}" for x in labels]
        rows.append(frame)
        counts = np.bincount(labels, minlength=k)
        audit.append({
            "hand": int(hand), "k": int(k), "members": int(len(ids)),
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
        })
    return pd.concat(rows, ignore_index=True), audit


def prepare_cutoff(main, cutoff, matrix_lambda, dim, seed=SEED):
    scoped = main.loc[main["season"].le(cutoff)].copy()
    past = add_context_residual(scoped.loc[scoped["season"].lt(cutoff)])
    past["cutoff"] = cutoff
    pitcher_embedding, batter_embedding, batter_basis, audit = build_joint_embedding(
        past, matrix_lambda, dim, seed
    )
    pitcher_hand = past.groupby("pitcher_id")["pitcher_hand"].last()
    batter_hand = past.groupby("batter_id")["batter_hand"].last()
    current = scoped.loc[scoped["season"].eq(cutoff)].copy()
    return {
        "past": past,
        "current": current,
        "pitcher_embedding": pitcher_embedding,
        "batter_embedding": batter_embedding,
        "batter_basis": batter_basis,
        "pitcher_hand": pitcher_hand,
        "batter_hand": batter_hand,
        "audit": audit,
    }


def pair_features(
    prepared, pitcher_k, batter_k, cluster_mode="standard", seed=SEED
):
    past = prepared["past"]
    current = prepared["current"]
    pitcher_lookup, pitcher_audit = hand_cluster(
        prepared["pitcher_embedding"], prepared["pitcher_hand"], pitcher_k, "pitcher",
        cluster_mode, seed,
    )
    batter_lookup, batter_audit = hand_cluster(
        prepared["batter_embedding"], prepared["batter_hand"], batter_k, "batter",
        cluster_mode, seed,
    )
    typed_past = past.merge(
        pitcher_lookup[["pitcher_id", "pitcher_type"]], on="pitcher_id", how="left"
    ).merge(
        batter_lookup[["batter_id", "batter_type"]], on="batter_id", how="left"
    )
    typed_past["pitcher_type"] = typed_past["pitcher_type"].fillna(
        "JSPH" + typed_past["pitcher_hand"].astype(str) + "_new"
    )
    typed_past["batter_type"] = typed_past["batter_type"].fillna(
        "JSBH" + typed_past["batter_hand"].astype(str) + "_new"
    )
    cutoff = int(current["season"].iloc[0])
    weight = np.power(
        0.5, (cutoff - typed_past["season"].to_numpy("float64")) / HALF_LIFE
    )
    work = typed_past[["pitcher_type", "batter_type"]].copy()
    work["weighted_residual"] = typed_past["reverse_residual"].to_numpy(float) * weight
    work["weighted_reverse"] = typed_past["reverse"].to_numpy(float) * weight
    work["effective_n"] = weight
    pair = work.groupby(["pitcher_type", "batter_type"], sort=False).agg(
        weighted_residual=("weighted_residual", "sum"),
        weighted_reverse=("weighted_reverse", "sum"),
        effective_n=("effective_n", "sum"),
    ).reset_index()
    pair["reverse_pair_delta"] = pair["weighted_residual"] / (
        pair["effective_n"] + SMOOTHING
    )
    pair["reverse_pair_delta_reliability"] = pair["effective_n"] / (
        pair["effective_n"] + SMOOTHING
    )
    pair["reverse_pair_rate"] = pair["weighted_reverse"] / pair["effective_n"]

    out = current[[
        "row_id", "season", "pitcher_id", "pitcher_hand", "batter_id", "batter_hand"
    ]].merge(
        pitcher_lookup[["pitcher_id", "pitcher_type"]], on="pitcher_id", how="left"
    ).merge(
        batter_lookup[["batter_id", "batter_type"]], on="batter_id", how="left"
    )
    out["pitcher_type"] = out["pitcher_type"].fillna(
        "JSPH" + out["pitcher_hand"].astype(str) + "_new"
    )
    out["batter_type"] = out["batter_type"].fillna(
        "JSBH" + out["batter_hand"].astype(str) + "_new"
    )
    pitcher_position = prepared["pitcher_embedding"]
    batter_basis = prepared["batter_basis"]
    pitcher_vector = pitcher_position.reindex(out["pitcher_id"]).to_numpy(float)
    batter_vector = batter_basis.reindex(out["batter_id"]).to_numpy(float)
    known = np.isfinite(pitcher_vector).all(axis=1) & np.isfinite(batter_vector).all(axis=1)
    out["jsvd_pair_score"] = np.where(
        known, np.nansum(pitcher_vector * batter_vector, axis=1), 0.0
    )
    out["jsvd_pitcher_norm"] = np.where(
        known, np.linalg.norm(np.nan_to_num(pitcher_vector), axis=1), 0.0
    )
    out["jsvd_batter_norm"] = np.where(
        known, np.linalg.norm(np.nan_to_num(batter_vector), axis=1), 0.0
    )
    out["jsvd_pair_known"] = known.astype("float32")
    out = out.merge(
        pair[["pitcher_type", "batter_type", *REVERSE_FEATURES[:-1]]],
        on=["pitcher_type", "batter_type"], how="left", validate="many_to_one",
    )
    out["reverse_pair_known"] = out["reverse_pair_delta"].notna().astype("float32")
    out["reverse_pair_delta"] = out["reverse_pair_delta"].fillna(0.0)
    out["reverse_pair_delta_reliability"] = out[
        "reverse_pair_delta_reliability"
    ].fillna(0.0)
    for column in REVERSE_FEATURES:
        out[column] = out[column].astype("float32")
    for column in DIRECT_FEATURES:
        out[column] = out[column].astype("float32")
    audit = {
        **prepared["audit"],
        "cluster_mode": cluster_mode,
        "pitcher_cluster_audit": json.dumps(pitcher_audit),
        "batter_cluster_audit": json.dumps(batter_audit),
        "pair_cells": int(len(pair)),
        "coverage": float(out["reverse_pair_known"].mean()),
    }
    return out[["row_id", "season", *REVERSE_FEATURES, *DIRECT_FEATURES]], audit


def fit_correction(frame, features, alpha, train_year, valid_year):
    train = frame["season"].eq(train_year)
    valid = frame["season"].eq(valid_year)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = frame.loc[train, "control_success"] - frame.loc[train, "prediction"]
    model.fit(frame.loc[train, features], residual)
    return np.clip(model.predict(frame.loc[valid, features]), -0.05, 0.05)


def evaluate(base, features, metadata):
    frame = base.merge(features, on=["row_id", "season"], validate="one_to_one")
    fold = {}
    success = {}
    for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
        valid = frame["season"].eq(valid_year)
        y = frame.loc[valid, "control_success"].to_numpy(float)
        prediction = frame.loc[valid, "prediction"].to_numpy(float)
        fold[valid_year] = {
            "error": prediction - y,
            "denominator": float(y.mean() * (1.0 - y.mean())),
        }
        success[valid_year] = fit_correction(
            frame, SUCCESS_FEATURES, 10.0, train_year, valid_year
        )
    rows = []
    for alpha in RIDGE_ALPHAS:
        reverse = {
            valid_year: fit_correction(
                frame, REVERSE_FEATURES, alpha, train_year, valid_year
            )
            for train_year, valid_year in [(2022, 2023), (2023, 2024)]
        }
        terms = {
            year: brier_terms(fold[year]["error"], success[year], reverse[year])
            for year in [2023, 2024]
        }
        for success_scale in SUCCESS_SCALES:
            for reverse_scale in REVERSE_SCALES:
                delta = {
                    year: brier_delta_from_terms(
                        terms[year], success_scale, reverse_scale
                    )
                    for year in [2023, 2024]
                }
                rows.append({
                    **metadata,
                    "alpha": alpha,
                    "success_scale": success_scale,
                    "reverse_scale": reverse_scale,
                    "f23_delta_brier": delta[2023],
                    "f24_delta_brier": delta[2024],
                    "both_improve": delta[2023] < 0 and delta[2024] < 0,
                    "robust_objective": robust_objective(
                        delta[2023], delta[2024],
                        {year: fold[year]["denominator"] for year in [2023, 2024]},
                    ),
                })
    return pd.DataFrame(rows).sort_values("robust_objective")


def main():
    main_frame = load_main()
    base = load_base()
    all_rows = []
    audit_rows = []
    total = len(MATRIX_LAMBDAS) * len(SVD_DIMS) * len(PITCHER_K) * len(BATTER_K)
    completed = 0
    for matrix_lambda in MATRIX_LAMBDAS:
        for dim in SVD_DIMS:
            prepared = {
                cutoff: prepare_cutoff(main_frame, cutoff, matrix_lambda, dim)
                for cutoff in CUTOFFS
            }
            for pitcher_k in PITCHER_K:
                for batter_k in BATTER_K:
                    config = config_id(matrix_lambda, dim, pitcher_k, batter_k)
                    pieces = []
                    for cutoff in CUTOFFS:
                        feature, audit = pair_features(
                            prepared[cutoff], pitcher_k, batter_k
                        )
                        pieces.append(feature)
                        audit_rows.append({
                            "config": config, "cutoff": cutoff,
                            "matrix_lambda": matrix_lambda, "svd_dim": dim,
                            "pitcher_k_left": pitcher_k[0],
                            "pitcher_k_right": pitcher_k[1],
                            "batter_k_left": batter_k[0],
                            "batter_k_right": batter_k[1], **audit,
                        })
                    metadata = {
                        "config": config, "matrix_lambda": matrix_lambda,
                        "svd_dim": dim, "pitcher_k_left": pitcher_k[0],
                        "pitcher_k_right": pitcher_k[1],
                        "batter_k_left": batter_k[0], "batter_k_right": batter_k[1],
                    }
                    result = evaluate(base, pd.concat(pieces, ignore_index=True), metadata)
                    all_rows.append(result)
                    completed += 1
                    best = result.iloc[0]
                    print(json.dumps({
                        "completed": completed, "total": total, "config": config,
                        "robust_objective": best["robust_objective"],
                        "f23_delta_brier": best["f23_delta_brier"],
                        "f24_delta_brier": best["f24_delta_brier"],
                    }, ensure_ascii=False), flush=True)
    reports = WORK / "reports"
    grid = pd.concat(all_rows, ignore_index=True).sort_values("robust_objective")
    grid.to_csv(reports / "joint_svd_cluster_grid.csv", index=False)
    best = grid.groupby("config", as_index=False).first().sort_values("robust_objective")
    best.to_csv(reports / "joint_svd_cluster_best.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(reports / "joint_svd_cluster_audit.csv", index=False)
    print("\nTOP 20")
    print(best.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
