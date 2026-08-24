from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiment" / "model_optimization"

TM_METRICS = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
]
RATE_COLUMNS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def numeric_audit(frame: pd.DataFrame, columns: list[str], source: str) -> pd.DataFrame:
    rows = []
    quantiles = [0.0, 0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999, 1.0]
    for column in columns:
        value = pd.to_numeric(frame[column], errors="coerce")
        finite = value[np.isfinite(value)]
        q = finite.quantile(quantiles)
        q1, q3 = q.loc[0.25], q.loc[0.75]
        iqr = q3 - q1
        low = q1 - 3.0 * iqr
        high = q3 + 3.0 * iqr
        rows.append({
            "source": source,
            "column": column,
            "n": len(value),
            "missing_rate": float(value.isna().mean()),
            "nonfinite_rate": float((~np.isfinite(value) & value.notna()).mean()),
            "zero_rate": float(value.eq(0).mean()),
            "negative_rate": float(value.lt(0).mean()),
            "q0000": float(q.loc[0.0]),
            "q0001": float(q.loc[0.001]),
            "q0010": float(q.loc[0.01]),
            "q0250": float(q1),
            "q0500": float(q.loc[0.5]),
            "q0750": float(q3),
            "q0990": float(q.loc[0.99]),
            "q0999": float(q.loc[0.999]),
            "q1000": float(q.loc[1.0]),
            "outside_3iqr_rate": float(((finite < low) | (finite > high)).mean()),
        })
    return pd.DataFrame(rows)


def main_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    numeric = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric.remove("control_success")
    audit = numeric_audit(train, numeric, "train")

    checks = []
    ranges = {
        "game_month": (1, 12), "game_dayofweek": (0, 6), "inning": (1, None),
        "balls_before": (0, 3), "strikes_before": (0, 2), "outs_before": (0, 2),
        "runner_on_1b": (0, 1), "runner_on_2b": (0, 1), "runner_on_3b": (0, 1),
        "num_runners_on": (0, 3), "home_win_expectancy": (0, 100),
        "away_win_expectancy": (0, 100), "li": (0, None),
    }
    for column, (lower, upper) in ranges.items():
        invalid = train[column].lt(lower)
        if upper is not None:
            invalid |= train[column].gt(upper)
        checks.append({"check": f"range:{column}", "violations": int(invalid.sum())})
    for column in RATE_COLUMNS:
        invalid = train[column].notna() & ~train[column].between(0, 1)
        checks.append({"check": f"rate_0_1:{column}", "violations": int(invalid.sum())})
    checks.extend([
        {
            "check": "run_total=sum_top_bottom",
            "violations": int(train["run_total_before"].ne(
                train["run_top_before"] + train["run_bot_before"]
            ).sum()),
        },
        {
            "check": "num_runners=sum_flags",
            "violations": int(train["num_runners_on"].ne(
                train[["runner_on_1b", "runner_on_2b", "runner_on_3b"]].sum(axis=1)
            ).sum()),
        },
        {
            "check": "pitchmix_n=pitcher_n",
            "violations": int(train["asof_pitcher_pitchmix_n"].ne(
                train["asof_pitcher_n"]
            ).sum()),
        },
    ])
    checks = pd.DataFrame(checks)
    return audit, checks


def trackman_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usecols = ["season", "pitcher_trackman_id", "pitch_type_group", *TM_METRICS]
    tm = pd.read_csv(ROOT / "data" / "trackman_history.csv", usecols=usecols)
    audit = numeric_audit(tm, TM_METRICS, "trackman")

    season_rows = []
    for (season, metric), group in (
        tm.melt(
            id_vars=["season"], value_vars=TM_METRICS,
            var_name="metric", value_name="value",
        ).groupby(["season", "metric"], observed=True)
    ):
        value = group["value"].dropna()
        q = value.quantile([0.01, 0.5, 0.99])
        season_rows.append({
            "season": int(season), "metric": metric, "n": len(value),
            "missing_rate": 1.0 - len(value) / len(group),
            "q01": float(q.loc[0.01]), "median": float(q.loc[0.5]),
            "q99": float(q.loc[0.99]),
        })
    season = pd.DataFrame(season_rows)

    # Strict temporal clipping audit: thresholds learned only from seasons before validation.
    transfer_rows = []
    for valid_year in [2023, 2024]:
        past = tm.loc[tm["season"].lt(valid_year)]
        valid = tm.loc[tm["season"].eq(valid_year)]
        for metric in TM_METRICS:
            lower, upper = past[metric].quantile([0.001, 0.999])
            value = valid[metric].dropna()
            transfer_rows.append({
                "valid_year": valid_year, "metric": metric,
                "past_q001": float(lower), "past_q999": float(upper),
                "valid_below_rate": float(value.lt(lower).mean()),
                "valid_above_rate": float(value.gt(upper).mean()),
                "valid_clipped_rate": float((value.lt(lower) | value.gt(upper)).mean()),
            })
    transfer = pd.DataFrame(transfer_rows)
    return audit, season, transfer


def profile_audit() -> pd.DataFrame:
    path = OUT / "pitcher_cluster_matchup" / "profiles" / "pitcher_profile_cutoff_2024.parquet"
    profile = pd.read_parquet(path)
    numeric_cols = [
        c for c in profile.columns
        if c.startswith(("tm500_", "tmg500_", "ctl_"))
        and pd.api.types.is_numeric_dtype(profile[c])
    ]
    return numeric_audit(profile, numeric_cols, "pitcher_profile_cutoff_2024")


def main() -> None:
    train_audit, logical = main_audit()
    tm_audit, tm_season, tm_transfer = trackman_audit()
    profiles = profile_audit()
    train_audit.to_csv(OUT / "preprocess_train_numeric_audit.csv", index=False)
    logical.to_csv(OUT / "preprocess_train_logical_checks.csv", index=False)
    tm_audit.to_csv(OUT / "preprocess_trackman_numeric_audit.csv", index=False)
    tm_season.to_csv(OUT / "preprocess_trackman_season_audit.csv", index=False)
    tm_transfer.to_csv(OUT / "preprocess_trackman_transfer_clip.csv", index=False)
    profiles.to_csv(OUT / "preprocess_profile_numeric_audit.csv", index=False)
    print("TRAIN LOGICAL")
    print(logical.to_string(index=False))
    print("\nTRAIN EXTREMES")
    print(train_audit.sort_values("outside_3iqr_rate", ascending=False).head(15).to_string(index=False))
    print("\nTRACKMAN")
    print(tm_audit.to_string(index=False))
    print("\nTRACKMAN TEMPORAL CLIP")
    print(tm_transfer.to_string(index=False))
    print("\nPROFILE EXTREMES")
    print(profiles.sort_values("outside_3iqr_rate", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()

