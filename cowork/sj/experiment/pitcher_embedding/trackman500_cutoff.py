from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "experiment" / "pitcher_embedding" / "outputs" / "trackman500"
)
MIN_SEASON_PITCHES = 500
TM_METRICS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
MAIN_FINGERPRINT_COLUMNS = [
    "season",
    "pitcher_id",
    "pitcher_hand",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "batter_hand",
]
TM_FINGERPRINT_COLUMNS = [
    "season",
    "pitcher_trackman_id",
    "pitcher_hand",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "batter_hand",
]
TM_LOAD_COLUMNS = sorted(
    set(TM_FINGERPRINT_COLUMNS + TM_METRICS + ["pitch_type_group"])
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoffs", default="2022,2023,2024,2025")
    parser.add_argument("--min-season-pitches", type=int, default=500)
    parser.add_argument("--similarity", type=float, default=0.80)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def make_state_code(frame: pd.DataFrame, is_main: bool) -> np.ndarray:
    month = frame["game_month"].to_numpy(np.int64) - 1
    day = frame["game_dayofweek"].to_numpy(np.int64)
    inning = np.minimum(frame["inning"].to_numpy(np.int64), 20)
    if is_main:
        bottom = frame["top_bottom"].astype(str).eq("B").to_numpy(np.int64)
        batter_right = frame["batter_hand"].to_numpy(np.int64) == 2
    else:
        bottom = frame["top_bottom"].astype(str).eq("Bottom").to_numpy(np.int64)
        batter_right = frame["batter_hand"].astype(str).eq("Right").to_numpy(np.int64)

    code = month.copy()
    for values, base in [
        (day, 7),
        (inning, 21),
        (bottom, 2),
        (frame["balls_before"].to_numpy(np.int64), 4),
        (frame["strikes_before"].to_numpy(np.int64), 3),
        (frame["outs_before"].to_numpy(np.int64), 3),
        (batter_right, 2),
    ]:
        code = code * base + values
    return code


def season_counts(trackman: pd.DataFrame) -> pd.DataFrame:
    return (
        trackman.groupby(["pitcher_trackman_id", "season"], sort=False)
        .size()
        .rename("tm_season_n")
        .reset_index()
    )


def eligible_pitcher_seasons(
    trackman: pd.DataFrame, cutoff: int, min_season_pitches: int
) -> pd.DataFrame:
    counts = season_counts(trackman)
    eligible = counts[
        counts["season"].lt(cutoff)
        & counts["tm_season_n"].ge(min_season_pitches)
    ].copy()
    if not eligible.empty:
        if eligible["season"].max() >= cutoff:
            raise AssertionError("Future Trackman season entered eligibility")
        if eligible["tm_season_n"].min() < min_season_pitches:
            raise AssertionError("Low-volume Trackman season entered eligibility")
    return eligible


def eligible_trackman_rows(
    trackman: pd.DataFrame, cutoff: int, min_season_pitches: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = eligible_pitcher_seasons(trackman, cutoff, min_season_pitches)
    rows = trackman.merge(
        eligible,
        on=["pitcher_trackman_id", "season"],
        how="inner",
        validate="many_to_one",
    )
    if len(rows) != int(eligible["tm_season_n"].sum()):
        raise AssertionError("Eligible Trackman row count mismatch")
    if len(rows) and rows["season"].max() >= cutoff:
        raise AssertionError("Future Trackman row entered cutoff artifact")
    return rows, eligible


def _normalize_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norm = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1.clip(1e-9)
    return sparse.diags(1.0 / norm) @ matrix


def build_cutoff_crosswalk(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    cutoff: int,
    min_season_pitches: int = MIN_SEASON_PITCHES,
    similarity_threshold: float = 0.80,
    margin_threshold: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_past = main.loc[main["season"].lt(cutoff), MAIN_FINGERPRINT_COLUMNS].copy()
    tm_eligible, _ = eligible_trackman_rows(trackman, cutoff, min_season_pitches)
    tm_past = tm_eligible[TM_FINGERPRINT_COLUMNS].copy()
    if len(main_past) and main_past["season"].max() >= cutoff:
        raise AssertionError("Future main season entered crosswalk")
    if len(tm_past) and tm_past["season"].max() >= cutoff:
        raise AssertionError("Future Trackman season entered crosswalk")

    main_past["state"] = make_state_code(main_past, is_main=True)
    tm_past["state"] = make_state_code(tm_past, is_main=False)
    records = []
    common_seasons = np.intersect1d(
        main_past["season"].unique(), tm_past["season"].unique()
    )
    for season in common_seasons:
        a = main_past[main_past["season"].eq(season)]
        b = tm_past[tm_past["season"].eq(season)]
        main_ids = np.sort(a["pitcher_id"].unique())
        tm_ids = np.sort(b["pitcher_trackman_id"].unique())
        if len(main_ids) == 0 or len(tm_ids) < 2:
            continue
        main_index = {value: index for index, value in enumerate(main_ids)}
        tm_index = {value: index for index, value in enumerate(tm_ids)}
        states = np.union1d(a["state"].unique(), b["state"].unique())
        state_index = {value: index for index, value in enumerate(states)}

        ag = a.groupby(["pitcher_id", "state"], sort=False).size().reset_index(name="n")
        bg = (
            b.groupby(["pitcher_trackman_id", "state"], sort=False)
            .size()
            .reset_index(name="n")
        )
        matrix_a = sparse.csr_matrix(
            (
                ag["n"],
                (ag["pitcher_id"].map(main_index), ag["state"].map(state_index)),
            ),
            shape=(len(main_ids), len(states)),
            dtype=np.float32,
        )
        matrix_b = sparse.csr_matrix(
            (
                bg["n"],
                (
                    bg["pitcher_trackman_id"].map(tm_index),
                    bg["state"].map(state_index),
                ),
            ),
            shape=(len(tm_ids), len(states)),
            dtype=np.float32,
        )
        similarity = (_normalize_rows(matrix_a) @ _normalize_rows(matrix_b).T).toarray()

        main_hand = (
            a.groupby("pitcher_id")["pitcher_hand"].first().reindex(main_ids).to_numpy()
        )
        tm_hand_raw = (
            b.groupby("pitcher_trackman_id")["pitcher_hand"]
            .first()
            .reindex(tm_ids)
            .to_numpy()
        )
        tm_hand = np.where(tm_hand_raw == "Left", 1, 2)
        similarity[main_hand[:, None] != tm_hand[None, :]] = -1.0

        order = np.argsort(similarity, axis=1)
        best = order[:, -1]
        second = order[:, -2]
        main_n = a.groupby("pitcher_id").size().reindex(main_ids).to_numpy()
        tm_n = b.groupby("pitcher_trackman_id").size().reindex(tm_ids).to_numpy()
        for row_index, pitcher_id in enumerate(main_ids):
            best_index = best[row_index]
            second_index = second[row_index]
            records.append(
                {
                    "evidence_season": int(season),
                    "pitcher_id": int(pitcher_id),
                    "pitcher_trackman_id": int(tm_ids[best_index]),
                    "similarity": float(similarity[row_index, best_index]),
                    "second_similarity": float(similarity[row_index, second_index]),
                    "margin": float(
                        similarity[row_index, best_index]
                        - similarity[row_index, second_index]
                    ),
                    "main_n": int(main_n[row_index]),
                    "trackman_n": int(tm_n[best_index]),
                }
            )

    diagnostics = pd.DataFrame(records)
    if diagnostics.empty:
        columns = [
            "pitcher_id",
            "pitcher_trackman_id",
            "cw_match_seasons",
            "cw_mean_sim",
            "cw_min_margin",
            "cw_total_main_n",
            "cw_total_trackman_n",
            "evidence_max_season",
            "cutoff",
            "trained_through_season",
            "min_trackman_season_pitches",
        ]
        return pd.DataFrame(columns=columns), diagnostics

    accepted = diagnostics[
        diagnostics["similarity"].ge(similarity_threshold)
        & diagnostics["margin"].ge(margin_threshold)
    ].copy()
    vote_table = (
        accepted.groupby(["pitcher_id", "pitcher_trackman_id"], as_index=False)
        .agg(
            cw_match_seasons=("evidence_season", "nunique"),
            cw_mean_sim=("similarity", "mean"),
            cw_min_margin=("margin", "min"),
            cw_total_main_n=("main_n", "sum"),
            cw_total_trackman_n=("trackman_n", "sum"),
            evidence_max_season=("evidence_season", "max"),
        )
        .sort_values(
            ["pitcher_id", "cw_match_seasons", "cw_mean_sim", "cw_total_main_n"],
            ascending=[True, False, False, False],
        )
    )
    crosswalk = vote_table.drop_duplicates("pitcher_id", keep="first")
    crosswalk = (
        crosswalk.sort_values(
            [
                "pitcher_trackman_id",
                "cw_match_seasons",
                "cw_mean_sim",
                "cw_total_main_n",
            ],
            ascending=[True, False, False, False],
        )
        .drop_duplicates("pitcher_trackman_id", keep="first")
        .sort_values("pitcher_id")
        .reset_index(drop=True)
    )
    crosswalk["cutoff"] = int(cutoff)
    crosswalk["trained_through_season"] = int(cutoff - 1)
    crosswalk["min_trackman_season_pitches"] = int(min_season_pitches)
    if len(crosswalk) and crosswalk["evidence_max_season"].max() >= cutoff:
        raise AssertionError("Future season entered crosswalk vote")
    if crosswalk["pitcher_id"].duplicated().any():
        raise AssertionError("Duplicate main pitcher in crosswalk")
    if crosswalk["pitcher_trackman_id"].duplicated().any():
        raise AssertionError("Duplicate Trackman pitcher in crosswalk")
    return crosswalk, diagnostics


def build_cutoff_trackman_stats(
    trackman: pd.DataFrame,
    cutoff: int,
    min_season_pitches: int = MIN_SEASON_PITCHES,
    recency_half_life: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, eligible = eligible_trackman_rows(trackman, cutoff, min_season_pitches)
    if rows.empty:
        return pd.DataFrame(), eligible

    season_metric = rows.groupby(
        ["pitcher_trackman_id", "season"], sort=False
    )[TM_METRICS].agg(["mean", "std"])
    season_metric.columns = [
        f"{metric}_{stat}" for metric, stat in season_metric.columns
    ]
    season_metric = season_metric.reset_index()
    season_metric = season_metric.merge(
        eligible,
        on=["pitcher_trackman_id", "season"],
        how="left",
        validate="one_to_one",
    )

    mix = pd.crosstab(
        [rows["pitcher_trackman_id"], rows["season"]],
        rows["pitch_type_group"],
        normalize="index",
    )
    for group_name in ["fastball", "breaking", "offspeed", "other"]:
        if group_name not in mix.columns:
            mix[group_name] = 0.0
    mix = mix[["fastball", "breaking", "offspeed", "other"]]
    mix.columns = [f"pitch_group_{column}_rate" for column in mix.columns]
    mix = mix.reset_index()
    season_table = season_metric.merge(
        mix,
        on=["pitcher_trackman_id", "season"],
        how="left",
        validate="one_to_one",
    )

    feature_columns = [
        column
        for column in season_table.columns
        if column not in {"pitcher_trackman_id", "season", "tm_season_n"}
    ]
    ages = cutoff - season_table["season"].to_numpy(float)
    season_table["_recency_weight"] = np.power(0.5, ages / recency_half_life)

    records = []
    for pitcher_id, part in season_table.groupby("pitcher_trackman_id", sort=False):
        part = part.sort_values("season")
        latest = part.iloc[-1]
        weights = part["_recency_weight"].to_numpy(float)
        weights /= weights.sum()
        record = {
            "pitcher_trackman_id": int(pitcher_id),
            "tm500_eligible_seasons": int(len(part)),
            "tm500_total_pitches": int(part["tm_season_n"].sum()),
            "tm500_last_season": int(latest["season"]),
            "tm500_season_gap": int(cutoff - latest["season"]),
            "tm500_last_season_n": int(latest["tm_season_n"]),
            "tm500_cutoff": int(cutoff),
            "tm500_trained_through_season": int(cutoff - 1),
            "tm500_min_season_pitches": int(min_season_pitches),
        }
        for column in feature_columns:
            values = part[column].to_numpy(float)
            finite = np.isfinite(values)
            record[f"tm500_latest_{column}"] = float(latest[column])
            if finite.any():
                normalized = weights[finite] / weights[finite].sum()
                record[f"tm500_recent_{column}"] = float(
                    np.dot(values[finite], normalized)
                )
                record[f"tm500_between_{column}_std"] = float(
                    np.nanstd(values, ddof=0)
                )
            else:
                record[f"tm500_recent_{column}"] = np.nan
                record[f"tm500_between_{column}_std"] = np.nan
        records.append(record)
    stats = pd.DataFrame(records)
    if len(stats):
        if stats["tm500_last_season"].max() >= cutoff:
            raise AssertionError("Future season entered Trackman stats")
        if stats["tm500_last_season_n"].min() < min_season_pitches:
            raise AssertionError("Low-volume latest season entered Trackman stats")
    return stats, eligible


def build_and_save_cutoff(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    cutoff: int,
    output_dir: Path,
    min_season_pitches: int,
    similarity_threshold: float,
    margin_threshold: float,
) -> dict:
    cutoff_dir = output_dir / f"cutoff_{cutoff}"
    cutoff_dir.mkdir(parents=True, exist_ok=True)
    crosswalk, diagnostics = build_cutoff_crosswalk(
        main,
        trackman,
        cutoff,
        min_season_pitches,
        similarity_threshold,
        margin_threshold,
    )
    stats, eligible = build_cutoff_trackman_stats(
        trackman, cutoff, min_season_pitches
    )
    crosswalk.to_parquet(cutoff_dir / "crosswalk.parquet", index=False)
    diagnostics.to_parquet(cutoff_dir / "crosswalk_diagnostics.parquet", index=False)
    stats.to_parquet(cutoff_dir / "trackman500_stats.parquet", index=False)
    eligible.to_parquet(cutoff_dir / "eligible_pitcher_seasons.parquet", index=False)

    matched_stats = crosswalk.merge(
        stats, on="pitcher_trackman_id", how="left", validate="one_to_one"
    )
    matched_stats.to_parquet(cutoff_dir / "main_pitcher_trackman500.parquet", index=False)
    past_rows = main["season"].lt(cutoff)
    matched_main_rows = main.loc[past_rows, "pitcher_id"].isin(
        crosswalk["pitcher_id"]
    )
    report = {
        "cutoff": int(cutoff),
        "trained_through_season": int(cutoff - 1),
        "min_season_pitches": int(min_season_pitches),
        "eligible_pitcher_seasons": int(len(eligible)),
        "eligible_trackman_pitchers": int(eligible["pitcher_trackman_id"].nunique()),
        "eligible_trackman_rows": int(eligible["tm_season_n"].sum()),
        "crosswalk_pitchers": int(len(crosswalk)),
        "crosswalk_main_past_row_coverage": float(matched_main_rows.mean()),
        "crosswalk_evidence_max_season": (
            int(crosswalk["evidence_max_season"].max()) if len(crosswalk) else None
        ),
        "stats_pitchers": int(len(stats)),
    }
    (cutoff_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    args = parse_args()
    cutoffs = sorted({int(value) for value in args.cutoffs.split(",") if value})
    main_frame = pd.read_csv(ROOT / "data" / "train.csv", usecols=MAIN_FINGERPRINT_COLUMNS)
    trackman = pd.read_csv(ROOT / "data" / "trackman_history.csv", usecols=TM_LOAD_COLUMNS)
    reports = []
    for cutoff in cutoffs:
        print(f"building cutoff={cutoff}", flush=True)
        report = build_and_save_cutoff(
            main_frame,
            trackman,
            cutoff,
            args.output_dir,
            args.min_season_pitches,
            args.similarity,
            args.margin,
        )
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
