from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trackman500_cutoff import ROOT, TM_METRICS, eligible_trackman_rows


BASE = ROOT / "experiment" / "pitcher_embedding" / "outputs" / "trackman500"
MODEL_DIR = ROOT / "experiment" / "model_optimization"
GROUPS = ["fastball", "breaking", "offspeed", "other"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoffs", default="2020,2021,2022,2023,2024,2025")
    parser.add_argument("--min-season-pitches", type=int, default=500)
    parser.add_argument("--min-group-pitches", type=int, default=30)
    parser.add_argument("--half-life", type=float, default=2.0)
    return parser.parse_args()


def weighted_mean(values: np.ndarray, weights: np.ndarray):
    valid = np.isfinite(values)
    if not valid.any():
        return np.nan
    weight = weights[valid]
    weight = weight / weight.sum()
    return float(np.dot(values[valid], weight))


def build_group_stats(trackman, cutoff, min_season_pitches, min_group_pitches, half_life):
    rows, eligible = eligible_trackman_rows(trackman, cutoff, min_season_pitches)
    if rows.empty:
        return pd.DataFrame(), {"eligible_rows": 0, "feature_count": 0}
    rows = rows.copy()
    rows["pitch_type_group"] = (
        rows["pitch_type_group"].fillna("other").where(
            rows["pitch_type_group"].isin(GROUPS), "other"
        )
    )
    grouped = rows.groupby(
        ["pitcher_trackman_id", "season", "pitch_type_group"], sort=False
    )
    season_group = grouped[TM_METRICS].agg(["mean", "std"])
    season_group.columns = [f"{metric}_{stat}" for metric, stat in season_group.columns]
    season_group = season_group.reset_index()
    group_n = grouped.size().rename("group_n").reset_index()
    season_group = season_group.merge(
        group_n,
        on=["pitcher_trackman_id", "season", "pitch_type_group"],
        validate="one_to_one",
    ).merge(
        eligible,
        on=["pitcher_trackman_id", "season"],
        how="left",
        validate="many_to_one",
    )
    season_group["group_rate"] = season_group["group_n"] / season_group["tm_season_n"]
    reliable = season_group[season_group["group_n"].ge(min_group_pitches)].copy()
    value_columns = [
        column
        for column in reliable.columns
        if column
        not in {
            "pitcher_trackman_id",
            "season",
            "pitch_type_group",
            "group_n",
            "tm_season_n",
        }
    ]
    records = []
    for (pitcher_id, pitch_group), part in reliable.groupby(
        ["pitcher_trackman_id", "pitch_type_group"], sort=False
    ):
        part = part.sort_values("season")
        latest = part.iloc[-1]
        weight = np.power(0.5, (cutoff - part["season"].to_numpy(float)) / half_life)
        prefix = f"tmg500_{pitch_group}"
        record = {
            "pitcher_trackman_id": int(pitcher_id),
            f"{prefix}_available": 1,
            f"{prefix}_eligible_seasons": int(len(part)),
            f"{prefix}_total_pitches": int(part["group_n"].sum()),
            f"{prefix}_last_season": int(latest["season"]),
            f"{prefix}_season_gap": int(cutoff - latest["season"]),
            f"{prefix}_last_n": int(latest["group_n"]),
        }
        for column in value_columns:
            values = part[column].to_numpy(float)
            latest_value = float(latest[column])
            recent_value = weighted_mean(values, weight)
            record[f"{prefix}_latest_{column}"] = latest_value
            record[f"{prefix}_recent_{column}"] = recent_value
            record[f"{prefix}_latest_minus_recent_{column}"] = (
                latest_value - recent_value
                if np.isfinite(latest_value) and np.isfinite(recent_value)
                else np.nan
            )
        records.append(record)
    group_frame = pd.DataFrame(records)
    if group_frame.empty:
        wide = pd.DataFrame(columns=["pitcher_trackman_id"])
    else:
        pieces = []
        for group in GROUPS:
            columns = [
                column
                for column in group_frame
                if column == "pitcher_trackman_id" or column.startswith(f"tmg500_{group}_")
            ]
            piece = group_frame.loc[
                group_frame[f"tmg500_{group}_available"].eq(1) if f"tmg500_{group}_available" in group_frame else [],
                columns,
            ]
            if len(piece):
                pieces.append(piece)
        wide = pieces[0]
        for piece in pieces[1:]:
            wide = wide.merge(piece, on="pitcher_trackman_id", how="outer", validate="one_to_one")

    # Mix entropy/HHI uses every pitch from an eligible 500+ season; the
    # 30-pitch threshold only controls noisy group-specific physical moments.
    mix_count = (
        rows.groupby(["pitcher_trackman_id", "season", "pitch_type_group"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    for group in GROUPS:
        if group not in mix_count:
            mix_count[group] = 0
    mix_count = mix_count[GROUPS]
    mix_rate = mix_count.div(mix_count.sum(axis=1), axis=0).reset_index()
    mix_records = []
    for pitcher_id, part in mix_rate.groupby("pitcher_trackman_id", sort=False):
        part = part.sort_values("season")
        latest = part.iloc[-1]
        weight = np.power(0.5, (cutoff - part["season"].to_numpy(float)) / half_life)
        weight /= weight.sum()
        record = {"pitcher_trackman_id": int(pitcher_id)}
        latest_rates = []
        recent_rates = []
        for group in GROUPS:
            latest_value = float(latest[group])
            recent_value = float(np.dot(part[group].to_numpy(float), weight))
            record[f"tmg500_mix_latest_{group}_rate"] = latest_value
            record[f"tmg500_mix_recent_{group}_rate"] = recent_value
            latest_rates.append(latest_value)
            recent_rates.append(recent_value)
        for name, rates in [("latest", latest_rates), ("recent", recent_rates)]:
            probability = np.asarray(rates, dtype=float)
            positive = probability[probability > 0]
            record[f"tmg500_mix_{name}_entropy"] = float(-np.sum(positive * np.log(positive)))
            record[f"tmg500_mix_{name}_hhi"] = float(np.sum(probability**2))
            record[f"tmg500_mix_{name}_max_rate"] = float(probability.max())
        mix_records.append(record)
    mix_frame = pd.DataFrame(mix_records)
    wide = mix_frame if wide.empty else wide.merge(
        mix_frame, on="pitcher_trackman_id", how="outer", validate="one_to_one"
    )

    for group in GROUPS:
        column = f"tmg500_{group}_available"
        if column not in wide:
            wide[column] = 0
        wide[column] = wide[column].fillna(0).astype("int8")
    for comparison in [("fastball", "breaking"), ("fastball", "offspeed")]:
        left, right = comparison
        for horizon in ["latest", "recent"]:
            for metric in ["rel_speed_mean", "spin_rate_mean", "induced_vert_break_mean", "horz_break_mean"]:
                a = f"tmg500_{left}_{horizon}_{metric}"
                b = f"tmg500_{right}_{horizon}_{metric}"
                if a in wide and b in wide:
                    wide[f"tmg500_{left}_minus_{right}_{horizon}_{metric}"] = wide[a] - wide[b]
    audit = {
        "cutoff": cutoff,
        "max_evidence_season": int(rows["season"].max()),
        "min_season_pitches": min_season_pitches,
        "min_group_pitches": min_group_pitches,
        "eligible_rows": len(rows),
        "pitchers": int(wide["pitcher_trackman_id"].nunique()),
        "feature_count": len(wide.columns) - 1,
    }
    if audit["max_evidence_season"] >= cutoff:
        raise AssertionError("Future season entered pitch-group Trackman features")
    return wide, audit


def main():
    args = parse_args()
    cutoffs = sorted({int(value) for value in args.cutoffs.split(",") if value})
    usecols = ["pitcher_trackman_id", "season", "pitch_type_group", *TM_METRICS]
    trackman = pd.read_csv(ROOT / "data" / "trackman_history.csv", usecols=usecols)
    audits = []
    for cutoff in cutoffs:
        print(f"building pitch-group cutoff={cutoff}", flush=True)
        stats, audit = build_group_stats(
            trackman,
            cutoff,
            args.min_season_pitches,
            args.min_group_pitches,
            args.half_life,
        )
        cutoff_dir = BASE / f"cutoff_{cutoff}"
        crosswalk = pd.read_parquet(cutoff_dir / "crosswalk.parquet")
        matched = crosswalk[["pitcher_id", "pitcher_trackman_id"]].merge(
            stats, on="pitcher_trackman_id", how="left", validate="one_to_one"
        )
        stats.to_parquet(cutoff_dir / "trackman500_pitchgroup_stats.parquet", index=False)
        matched.to_parquet(
            cutoff_dir / "main_pitcher_trackman500_pitchgroup.parquet", index=False
        )
        (cutoff_dir / "pitchgroup_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        audits.append(audit)
        print(json.dumps(audit, ensure_ascii=False), flush=True)

    train = pd.read_csv(
        ROOT / "data" / "train.csv", usecols=["row_id", "season", "pitcher_id"]
    )
    reference = pd.read_parquet(
        BASE / "cutoff_2025" / "main_pitcher_trackman500_pitchgroup.parquet"
    )
    feature_columns = [
        column
        for column in reference
        if column not in {"pitcher_id", "pitcher_trackman_id"}
    ]
    parts = []
    season_audit = []
    for season, part in train.groupby("season", sort=True):
        base = part[["row_id", "season", "pitcher_id"]].copy()
        if int(season) == 2019:
            for column in feature_columns:
                base[column] = np.nan
        else:
            lookup = pd.read_parquet(
                BASE / f"cutoff_{int(season)}" / "main_pitcher_trackman500_pitchgroup.parquet"
            )
            base = base.merge(
                lookup[["pitcher_id", *feature_columns]],
                on="pitcher_id",
                how="left",
                validate="many_to_one",
            )
        availability_columns = [
            column for column in feature_columns if column.endswith("_available")
        ]
        base[availability_columns] = base[availability_columns].fillna(0).astype("int8")
        base["tmg500_any_available"] = base[availability_columns].max(axis=1).astype("int8")
        season_audit.append(
            {
                "season": int(season),
                "rows": len(base),
                "coverage": float(base["tmg500_any_available"].mean()),
                "evidence_max_season": int(season) - 1 if int(season) > 2019 else None,
            }
        )
        parts.append(base.drop(columns="pitcher_id"))
    cache = pd.concat(parts, ignore_index=True)
    cache = train[["row_id"]].merge(cache, on="row_id", validate="one_to_one")
    cache.to_parquet(MODEL_DIR / "trackman500_pitchgroup_asof_train.parquet", index=False)
    final_lookup = reference.drop(columns="pitcher_trackman_id")
    availability_columns = [
        column for column in feature_columns if column.endswith("_available")
    ]
    final_lookup[availability_columns] = final_lookup[availability_columns].fillna(0).astype("int8")
    final_lookup["tmg500_any_available"] = final_lookup[availability_columns].max(axis=1).astype("int8")
    final_lookup.to_parquet(
        MODEL_DIR / "trackman500_pitchgroup_lookup_2025.parquet", index=False
    )
    manifest = {
        "cutoff_audits": audits,
        "season_audit": season_audit,
        "feature_count": len(feature_columns) + 1,
        "feature_columns": feature_columns + ["tmg500_any_available"],
        "strict_rule": "season S uses only eligible 500+ Trackman seasons < S",
    }
    (MODEL_DIR / "trackman500_pitchgroup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
