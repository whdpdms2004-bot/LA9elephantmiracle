from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "experiment" / "model_optimization" / "pitcher_cluster_matchup"
TM_BASE = ROOT / "experiment" / "pitcher_embedding" / "outputs" / "trackman500"
LABEL_PATH = ROOT / "experiment" / "model_optimization" / "failure_component_labels.parquet"
TARGETS = ["control_success", "reverse", "middle", "outside_only"]
SEED = 2026
PROFILE_SCHEMA_VERSION = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoffs", default="2020,2021,2022,2023,2024,2025")
    parser.add_argument("--half-life", type=float, default=2.0)
    parser.add_argument("--profile-dir", default="profiles")
    parser.add_argument("--audit-name", default="profile_audit.json")
    return parser.parse_args()


def weighted_mean(values, weights):
    values = np.asarray(values, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.dot(values[valid], weights[valid]) / weights[valid].sum())


def aggregate_rate_profile(season_table, cutoff, half_life):
    records = []
    for pitcher_id, part in season_table.groupby("pitcher_id", sort=False):
        part = part.sort_values("season")
        latest = part.iloc[-1]
        ages = cutoff - part["season"].to_numpy("float64")
        recency = np.power(0.5, ages / half_life)
        evidence = np.minimum(part["n"].to_numpy("float64"), 1000.0)
        weights = recency * evidence
        record = {
            "pitcher_id": int(pitcher_id),
            "pitcher_hand": int(latest["pitcher_hand"]),
            "ctl_total_n": int(part["n"].sum()),
            "ctl_last_n": int(latest["n"]),
            "ctl_last_season": int(latest["season"]),
            "ctl_season_gap": int(cutoff - latest["season"]),
            "ctl_history_seasons": int(len(part)),
            "ctl_rookie": int(latest["n"] <= 100),
        }
        for target in TARGETS:
            rate = part[f"{target}_rate"].to_numpy("float64")
            residual = part[f"{target}_resid"].to_numpy("float64")
            record[f"ctl_{target}_recent_rate"] = weighted_mean(rate, weights)
            record[f"ctl_{target}_recent_resid"] = weighted_mean(residual, weights)
            record[f"ctl_{target}_latest_rate"] = float(latest[f"{target}_rate"])
            record[f"ctl_{target}_latest_resid"] = float(latest[f"{target}_resid"])
            finite = residual[np.isfinite(residual)]
            record[f"ctl_{target}_between_std"] = (
                float(np.std(finite)) if len(finite) > 1 else 0.0
            )
        records.append(record)
    return pd.DataFrame(records)


def add_split_profiles(profile, past):
    output = profile.copy()
    definitions = {
        "batter_hand1": past["batter_hand"].eq(1),
        "batter_hand2": past["batter_hand"].eq(2),
        "full_count": past["balls_before"].eq(3) & past["strikes_before"].eq(2),
        "two_strike": past["strikes_before"].eq(2),
        "high_li": past["li"].ge(2.0),
    }
    global_rate = float(past["control_success"].mean())
    for name, mask in definitions.items():
        selected = past.loc[mask, ["pitcher_id", "control_success"]]
        grouped = selected.groupby("pitcher_id")["control_success"].agg(["size", "mean"])
        grouped[f"ctl_split_{name}_n"] = grouped["size"].astype("int32")
        grouped[f"ctl_split_{name}_resid"] = grouped["mean"] - global_rate
        grouped = grouped[[f"ctl_split_{name}_n", f"ctl_split_{name}_resid"]]
        output = output.merge(grouped, left_on="pitcher_id", right_index=True, how="left")
    if {"ctl_split_batter_hand1_resid", "ctl_split_batter_hand2_resid"}.issubset(output):
        output["ctl_batter_hand_resid_gap"] = (
            output["ctl_split_batter_hand1_resid"]
            - output["ctl_split_batter_hand2_resid"]
        )
    return output


def physical_columns(frame):
    keep = []
    for column in frame.columns:
        if column.startswith("tm500_"):
            if any(token in column for token in [
                "recent_", "between_", "eligible_seasons", "total_pitches",
                "season_gap", "last_season_n",
            ]):
                keep.append(column)
        elif column.startswith("tmg500_"):
            if (
                "_recent_" in column
                or column.endswith("_available")
                or "_minus_" in column and "_recent_" in column
            ):
                keep.append(column)
    excluded = {
        "tm500_cutoff", "tm500_trained_through_season",
        "tm500_min_season_pitches",
    }
    return [column for column in keep if column not in excluded]


def attach_trackman(profile, cutoff):
    cutoff_dir = TM_BASE / f"cutoff_{cutoff}"
    general = pd.read_parquet(cutoff_dir / "main_pitcher_trackman500.parquet")
    pitchgroup = pd.read_parquet(cutoff_dir / "main_pitcher_trackman500_pitchgroup.parquet")
    general_cols = ["pitcher_id", "pitcher_trackman_id", "cw_mean_sim", "cw_min_margin"]
    general_cols += physical_columns(general)
    pitch_cols = ["pitcher_id"] + physical_columns(pitchgroup)
    lookup = general[general_cols].merge(
        pitchgroup[pitch_cols], on="pitcher_id", how="left", validate="one_to_one"
    )
    output = profile.merge(lookup, on="pitcher_id", how="left", validate="one_to_one")
    output["tm_available"] = output["pitcher_trackman_id"].notna().astype("int8")
    output["cohort"] = np.select(
        [
            output["ctl_rookie"].eq(1),
            output["tm_available"].eq(1),
        ],
        ["rookie", "tm_eligible"],
        default="control_only",
    )
    sign = np.where(output["pitcher_hand"].eq(1), -1.0, 1.0)
    for column in list(output.columns):
        if column.startswith(("tm500_", "tmg500_")) and any(
            token in column for token in ["horz_break", "rel_side"]
        ):
            output[f"{column}_arm"] = output[column].astype("float64") * sign
    return output


def build_one(main, labels, cutoff, half_life):
    if cutoff <= int(main["season"].min()):
        return pd.DataFrame()
    past = main.loc[main["season"].lt(cutoff)].copy()
    if labels is not None:
        label_cols = ["reverse", "middle", "outside_only"]
        past[label_cols] = labels.loc[past.index, label_cols].astype("float32")
    else:
        for column in ["reverse", "middle", "outside_only"]:
            past[column] = np.nan

    season_table = (
        past.groupby(["pitcher_id", "season", "pitcher_hand"], sort=False)
        .agg(
            n=("control_success", "size"),
            control_success_rate=("control_success", "mean"),
            reverse_rate=("reverse", "mean"),
            middle_rate=("middle", "mean"),
            outside_only_rate=("outside_only", "mean"),
        )
        .reset_index()
    )
    high_volume = season_table["n"].ge(100)
    component_rates = ["reverse_rate", "middle_rate", "outside_only_rate"]
    invalid_high_volume = high_volume & season_table[component_rates].isna().any(axis=1)
    if invalid_high_volume.any():
        bad = season_table.loc[
            invalid_high_volume,
            ["pitcher_id", "season", "n", *component_rates],
        ].head(20)
        raise RuntimeError(
            "High-volume pitcher-season has missing failure components; "
            "failure-label cache is stale or misaligned.\n" + bad.to_string(index=False)
        )
    for target in TARGETS:
        rate_column = f"{target}_rate"
        numerator = (season_table[rate_column] * season_table["n"]).groupby(
            season_table["season"]
        ).sum()
        denominator = season_table["n"].groupby(season_table["season"]).sum()
        season_global = numerator / denominator
        season_table[f"{target}_resid"] = (
            season_table[rate_column] - season_table["season"].map(season_global)
        )
    profile = aggregate_rate_profile(season_table, cutoff, half_life)
    profile = add_split_profiles(profile, past)
    profile = attach_trackman(profile, cutoff)
    profile.insert(1, "cutoff", int(cutoff))
    return profile


def audit_hands(main):
    crosswalk = pd.read_parquet(TM_BASE / "cutoff_2025" / "crosswalk.parquet")
    tm = pd.read_csv(
        ROOT / "data" / "trackman_history.csv",
        usecols=["pitcher_trackman_id", "pitcher_hand"],
    ).drop_duplicates("pitcher_trackman_id")
    hand = main[["pitcher_id", "pitcher_hand"]].drop_duplicates("pitcher_id")
    joined = (
        crosswalk[["pitcher_id", "pitcher_trackman_id"]]
        .merge(hand, on="pitcher_id")
        .merge(tm, on="pitcher_trackman_id", suffixes=("_main", "_trackman"))
    )
    table = pd.crosstab(joined["pitcher_hand_main"], joined["pitcher_hand_trackman"])
    expected = joined["pitcher_hand_main"].map({1: "Left", 2: "Right"})
    mismatch = joined.loc[expected.ne(joined["pitcher_hand_trackman"])]
    return {
        "mapping": {"1": "Left", "2": "Right"},
        "matched_pitchers": int(len(joined)),
        "mismatch_count": int(len(mismatch)),
        "crosstab": {
            str(index): {str(column): int(table.loc[index, column]) for column in table.columns}
            for index in table.index
        },
        "mismatches": mismatch.to_dict("records"),
    }


def main():
    args = parse_args()
    cutoffs = sorted({int(value) for value in args.cutoffs.split(",") if value})
    usecols = [
        "row_id", "season", "pitcher_id", "pitcher_hand", "batter_hand",
        "balls_before", "strikes_before", "li", "control_success",
    ]
    main = pd.read_csv(ROOT / "data" / "train.csv", usecols=usecols)
    labels = pd.read_parquet(LABEL_PATH)
    if not main["row_id"].equals(labels["row_id"]):
        raise RuntimeError("Failure labels are not row-aligned")
    component_coverage = labels.groupby("season")[[
        "reverse", "middle", "outside_only"
    ]].apply(lambda part: part.notna().mean()).reset_index()
    if component_coverage.drop(columns="season").min().min() < 0.995:
        raise RuntimeError(
            "Failure-component label coverage is unexpectedly low.\n"
            + component_coverage.to_string(index=False)
        )

    profile_dir = WORK / args.profile_dir
    profile_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "rule": "Every cutoff S profile uses main and TrackMan seasons strictly before S.",
        "min_trackman_season_pitches": 500,
        "half_life": args.half_life,
        "hand_audit": audit_hands(main),
        "failure_component_coverage_by_season": component_coverage.to_dict("records"),
        "cutoffs": [],
    }
    for cutoff in cutoffs:
        profile = build_one(main, labels, cutoff, args.half_life)
        path = profile_dir / f"pitcher_profile_cutoff_{cutoff}.parquet"
        profile.to_parquet(path, index=False)
        record = {
            "cutoff": cutoff,
            "max_main_season": cutoff - 1,
            "pitchers": int(profile["pitcher_id"].nunique()),
            "tm_available": int(profile["tm_available"].sum()),
            "rookies": int(profile["ctl_rookie"].sum()),
            "columns": int(len(profile.columns)),
            "high_volume_missing_latest_components": int(
                (
                    profile["ctl_last_n"].ge(100)
                    & profile[[
                        "ctl_reverse_latest_resid", "ctl_middle_latest_resid",
                        "ctl_outside_only_latest_resid",
                    ]].isna().any(axis=1)
                ).sum()
            ),
            "path": str(path.relative_to(ROOT)),
        }
        audit["cutoffs"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    audit_path = WORK / "reports" / args.audit_name
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audit": str(audit_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
