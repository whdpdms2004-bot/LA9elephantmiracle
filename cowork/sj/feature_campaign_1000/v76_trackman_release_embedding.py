"""V76: strict-as-of TrackMan release-consistency pitcher representation.

기존 production TrackMan 피처는 물리량별 mean/std와 시즌 변화가 중심이다.
이 스크립트는 그와 겹치지 않는 다음 표현을 만든다.

1. rel_height/rel_side 공분산 타원의 trace, area, eigen ratio
2. 구종군별 release consistency와 구종군 평균 릴리스 간 separation
3. 구속-릴리스, 구속-무브먼트 correlation
4. 위 target-free 통계 벡터의 fold-fit PCA 좌표

예측 cutoff S마다 TrackMan season < S만 사용하며, 기존 strict-as-of crosswalk도
동일 cutoff 파일을 사용한다. 결과는 pitcher_id 단건 lookup이므로 test 행 독립이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


SJ = Path(__file__).resolve().parents[1]
TRACKMAN = SJ / "data" / "trackman_history.csv"
TM500 = SJ / "experiment" / "pitcher_embedding" / "outputs" / "trackman500"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "trackman_release"
GROUPS = ["all", "fastball", "breaking", "offspeed"]
BASE = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
]
USECOLS = ["season", "pitcher_trackman_id", "pitch_type_group", *BASE]
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cutoffs", default="2023,2024,2025")
    p.add_argument("--pca-dim", type=int, default=12)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def add_moments(frame: pd.DataFrame) -> pd.DataFrame:
    """공분산/상관을 groupby mean으로 계산하기 위한 교차곱을 추가한다."""
    out = frame.copy()
    pairs = [
        ("rel_height", "rel_side"),
        ("rel_speed", "rel_height"),
        ("rel_speed", "rel_side"),
        ("rel_speed", "extension"),
        ("rel_speed", "induced_vert_break"),
        ("rel_speed", "horz_break"),
    ]
    for col in BASE:
        out[f"__sq__{col}"] = out[col].astype("float64") ** 2
    for left, right in pairs:
        out[f"__cross__{left}__{right}"] = (
            out[left].astype("float64") * out[right].astype("float64"))
    return out


PAIRS = [
    ("rel_height", "rel_side"),
    ("rel_speed", "rel_height"),
    ("rel_speed", "rel_side"),
    ("rel_speed", "extension"),
    ("rel_speed", "induced_vert_break"),
    ("rel_speed", "horz_break"),
]


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    work = add_moments(frame)
    value_cols = [*BASE]
    value_cols += [f"__sq__{col}" for col in BASE]
    value_cols += [f"__cross__{a}__{b}" for a, b in PAIRS]
    mean = work.groupby(keys, sort=False)[value_cols].mean()
    size = work.groupby(keys, sort=False).size().rename("n")
    out = mean.join(size).reset_index()
    for col in BASE:
        var = out[f"__sq__{col}"] - out[col] ** 2
        out[f"{col}_std"] = np.sqrt(np.clip(var, 0.0, None))
    for left, right in PAIRS:
        cov = out[f"__cross__{left}__{right}"] - out[left] * out[right]
        denom = out[f"{left}_std"] * out[f"{right}_std"]
        out[f"corr_{left}_{right}"] = cov / np.clip(denom, EPS, None)
        if (left, right) == ("rel_height", "rel_side"):
            vh = out["rel_height_std"] ** 2
            vs = out["rel_side_std"] ** 2
            trace = vh + vs
            det = np.clip(vh * vs - cov ** 2, 0.0, None)
            disc = np.sqrt(np.clip(trace ** 2 - 4.0 * det, 0.0, None))
            eig_hi = 0.5 * (trace + disc)
            eig_lo = 0.5 * (trace - disc)
            out["release_trace"] = trace
            out["release_area"] = np.sqrt(det)
            out["release_eigen_ratio"] = eig_lo / np.clip(eig_hi, EPS, None)
    drop = [c for c in out.columns if c.startswith("__")]
    return out.drop(columns=drop)


def build_season_profiles(trackman: pd.DataFrame) -> pd.DataFrame:
    overall = aggregate(trackman, ["pitcher_trackman_id", "season"])
    overall["pitch_group"] = "all"
    known = trackman[trackman["pitch_type_group"].isin(GROUPS[1:])]
    grouped = aggregate(
        known, ["pitcher_trackman_id", "season", "pitch_type_group"]
    ).rename(columns={"pitch_type_group": "pitch_group"})
    long = pd.concat([overall, grouped], ignore_index=True, sort=False)
    values = [c for c in long.columns
              if c not in ("pitcher_trackman_id", "season", "pitch_group")]
    wide = long.pivot_table(
        index=["pitcher_trackman_id", "season"],
        columns="pitch_group", values=values, aggfunc="first"
    )
    wide.columns = [f"{group}_{value}" for value, group in wide.columns]
    wide = wide.reset_index()

    # 구종군 평균 릴리스 간 거리. 구종별 목표가 다른 효과를 전체 분산과 분리한다.
    for g1, g2 in [("fastball", "breaking"), ("fastball", "offspeed"),
                   ("breaking", "offspeed")]:
        needed = [f"{g}_{c}" for g in (g1, g2)
                  for c in ("rel_height", "rel_side", "extension")]
        if all(c in wide for c in needed):
            dist2 = sum((wide[f"{g1}_{c}"] - wide[f"{g2}_{c}"]) ** 2
                        for c in ("rel_height", "rel_side", "extension"))
            wide[f"release_sep_{g1}_{g2}"] = np.sqrt(dist2)
    return wide


def recency_summary(season_profiles: pd.DataFrame, eligible: pd.DataFrame,
                    cutoff: int, half_life: float = 2.0) -> pd.DataFrame:
    valid = eligible.loc[eligible["season"].lt(cutoff),
                         ["pitcher_trackman_id", "season", "tm_season_n"]]
    part = season_profiles.merge(
        valid, on=["pitcher_trackman_id", "season"], how="inner",
        validate="one_to_one")
    if len(part) and part["season"].max() >= cutoff:
        raise AssertionError("future TrackMan season entered V76 summary")
    features = [c for c in part.columns if c not in (
        "pitcher_trackman_id", "season", "tm_season_n")]
    records = []
    for pitcher, block in part.groupby("pitcher_trackman_id", sort=False):
        block = block.sort_values("season")
        latest = block.iloc[-1]
        age = cutoff - block["season"].to_numpy(np.float64)
        weights = np.power(0.5, age / half_life) * np.sqrt(
            block["tm_season_n"].to_numpy(np.float64))
        record = {
            "pitcher_trackman_id": int(pitcher),
            "tr_eligible_seasons": int(len(block)),
            "tr_total_pitches": int(block["tm_season_n"].sum()),
            "tr_last_season": int(latest["season"]),
            "tr_season_gap": int(cutoff - latest["season"]),
        }
        for col in features:
            values = block[col].to_numpy(np.float64)
            finite = np.isfinite(values)
            record[f"tr_latest_{col}"] = float(latest[col])
            if finite.any():
                w = weights[finite] / weights[finite].sum()
                record[f"tr_recent_{col}"] = float(np.dot(values[finite], w))
                record[f"tr_between_{col}_std"] = float(np.std(values[finite]))
            else:
                record[f"tr_recent_{col}"] = np.nan
                record[f"tr_between_{col}_std"] = np.nan
        records.append(record)
    return pd.DataFrame(records)


def robust_pca(summary: pd.DataFrame, dim: int):
    id_cols = ["pitcher_trackman_id", "tr_eligible_seasons",
               "tr_total_pitches", "tr_last_season", "tr_season_gap"]
    features = [c for c in summary.columns if c not in id_cols]
    matrix = summary[features].replace([np.inf, -np.inf], np.nan)
    median = matrix.median(axis=0)
    q25, q75 = matrix.quantile(0.25), matrix.quantile(0.75)
    scale = (q75 - q25).where((q75 - q25).abs() > 1e-8, 1.0)
    x = ((matrix.fillna(median) - median) / scale).clip(-10, 10)
    keep = x.std(axis=0) > 1e-8
    x = x.loc[:, keep]
    n_dim = min(dim, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_dim, svd_solver="full")
    z = pca.fit_transform(x.to_numpy(np.float64))
    embedding = pd.DataFrame(
        z, columns=[f"tr_pca_{i:02d}" for i in range(n_dim)])
    embedding.insert(0, "pitcher_trackman_id",
                     summary["pitcher_trackman_id"].to_numpy())
    spec = {
        "input_columns": x.columns.tolist(),
        "median": {c: float(median[c]) for c in x.columns},
        "scale_iqr": {c: float(scale[c]) for c in x.columns},
        "components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_components": int(n_dim),
    }
    return embedding, spec


def _robust_matrix(frame: pd.DataFrame, columns: list[str]):
    matrix = frame[columns].replace([np.inf, -np.inf], np.nan)
    median = matrix.median(axis=0)
    q25, q75 = matrix.quantile(0.25), matrix.quantile(0.75)
    scale = (q75 - q25).where((q75 - q25).abs() > 1e-8, 1.0)
    x = ((matrix.fillna(median) - median) / scale).clip(-10, 10)
    keep = x.std(axis=0) > 1e-8
    return x.loc[:, keep], median.loc[keep], scale.loc[keep]


def residual_pca(summary: pd.DataFrame, old_stats: pd.DataFrame, dim: int,
                 alpha: float = 100.0):
    """기존 tm500 mean/std로 설명되는 부분을 제거한 target-free PCA."""
    merged = summary.merge(old_stats, on="pitcher_trackman_id", how="inner",
                           suffixes=("", "_old"), validate="one_to_one")
    new_meta = {"pitcher_trackman_id", "tr_eligible_seasons", "tr_total_pitches",
                "tr_last_season", "tr_season_gap"}
    old_meta = {"pitcher_trackman_id", "tm500_eligible_seasons",
                "tm500_total_pitches", "tm500_last_season", "tm500_season_gap",
                "tm500_last_season_n", "tm500_cutoff",
                "tm500_trained_through_season", "tm500_min_season_pitches"}
    new_cols = [c for c in summary.columns if c not in new_meta]
    old_cols = [c for c in old_stats.columns if c not in old_meta]
    new_x, new_med, new_scale = _robust_matrix(merged, new_cols)
    old_x, old_med, old_scale = _robust_matrix(merged, old_cols)
    ridge = Ridge(alpha=alpha).fit(old_x.to_numpy(), new_x.to_numpy())
    residual = new_x.to_numpy() - ridge.predict(old_x.to_numpy())
    n_dim = min(dim, residual.shape[0], residual.shape[1])
    pca = PCA(n_components=n_dim, svd_solver="full")
    z = pca.fit_transform(residual)
    embedding = pd.DataFrame(
        z, columns=[f"tr_resid_pca_{i:02d}" for i in range(n_dim)])
    embedding.insert(0, "pitcher_trackman_id",
                     merged["pitcher_trackman_id"].to_numpy())
    spec = {
        "ridge_alpha": alpha,
        "new_columns": new_x.columns.tolist(),
        "old_columns": old_x.columns.tolist(),
        "new_median": {c: float(new_med[c]) for c in new_x.columns},
        "new_scale_iqr": {c: float(new_scale[c]) for c in new_x.columns},
        "old_median": {c: float(old_med[c]) for c in old_x.columns},
        "old_scale_iqr": {c: float(old_scale[c]) for c in old_x.columns},
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": ridge.intercept_.tolist(),
        "pca_components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_components": int(n_dim),
    }
    return embedding, spec


def main():
    args = parse_args()
    cutoffs = sorted({int(x) for x in args.cutoffs.split(",") if x.strip()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {TRACKMAN}", flush=True)
    tm = pd.read_csv(TRACKMAN, usecols=USECOLS)
    for col in BASE:
        tm[col] = pd.to_numeric(tm[col], errors="coerce").astype("float32")
    tm["season"] = tm["season"].astype("int16")
    tm["pitcher_trackman_id"] = tm["pitcher_trackman_id"].astype("int32")
    print(f"loaded {tm.shape}", flush=True)
    season_profiles = build_season_profiles(tm)
    season_profiles.to_parquet(args.output_dir / "season_profiles.parquet", index=False)
    manifest = []
    for cutoff in cutoffs:
        cutoff_dir = args.output_dir / f"cutoff_{cutoff}"
        cutoff_dir.mkdir(parents=True, exist_ok=True)
        src = TM500 / f"cutoff_{cutoff}"
        eligible = pd.read_parquet(src / "eligible_pitcher_seasons.parquet")
        crosswalk = pd.read_parquet(src / "crosswalk.parquet")
        old_stats = pd.read_parquet(src / "trackman500_stats.parquet")
        if len(eligible) and eligible["season"].max() >= cutoff:
            raise AssertionError("future season in eligible source")
        summary = recency_summary(season_profiles, eligible, cutoff)
        embedding, pca_spec = robust_pca(summary, args.pca_dim)
        residual_embedding, residual_spec = residual_pca(
            summary, old_stats, args.pca_dim)
        main_lookup = (crosswalk.merge(summary, on="pitcher_trackman_id", how="left",
                                       validate="one_to_one")
                       .merge(embedding, on="pitcher_trackman_id", how="left",
                              validate="one_to_one")
                       .merge(residual_embedding, on="pitcher_trackman_id", how="left",
                              validate="one_to_one"))
        if main_lookup["pitcher_id"].duplicated().any():
            raise AssertionError("duplicate main pitcher in V76 lookup")
        summary.to_parquet(cutoff_dir / "trackman_release_summary.parquet", index=False)
        embedding.to_parquet(cutoff_dir / "trackman_release_pca.parquet", index=False)
        residual_embedding.to_parquet(
            cutoff_dir / "trackman_release_residual_pca.parquet", index=False)
        main_lookup.to_parquet(cutoff_dir / "main_pitcher_release.parquet", index=False)
        (cutoff_dir / "pca_spec.json").write_text(
            json.dumps(pca_spec, indent=1, sort_keys=True), encoding="utf-8")
        (cutoff_dir / "residual_pca_spec.json").write_text(
            json.dumps(residual_spec, indent=1, sort_keys=True), encoding="utf-8")
        report = {
            "cutoff": cutoff,
            "trained_through_season": cutoff - 1,
            "max_trackman_season": int(eligible["season"].max()),
            "trackman_pitchers": int(len(summary)),
            "mapped_main_pitchers": int(len(main_lookup)),
            "summary_features": int(summary.shape[1] - 1),
            "pca_features": int(embedding.shape[1] - 1),
            "pca_explained_variance": float(sum(pca_spec["explained_variance_ratio"])),
            "residual_pca_features": int(residual_embedding.shape[1] - 1),
            "residual_pca_explained_variance": float(
                sum(residual_spec["explained_variance_ratio"])),
        }
        manifest.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
