from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "control_success"
RATE_COLUMNS = [
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(name: str, payload) -> None:
    with (OUT / name).open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=json_default)


def wilson_bounds(successes: pd.Series, counts: pd.Series, z: float = 1.96):
    n = counts.astype(float)
    p = successes.astype(float) / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return center - half, center + half


def group_target(df: pd.DataFrame, cols: list[str], global_rate: float) -> pd.DataFrame:
    grouped = (
        df.groupby(cols, dropna=False, observed=True)[TARGET]
        .agg(n="size", successes="sum", success_rate="mean")
        .reset_index()
    )
    grouped["lift_pp"] = (grouped["success_rate"] - global_rate) * 100.0
    grouped["ci95_low"], grouped["ci95_high"] = wilson_bounds(
        grouped["successes"], grouped["n"]
    )
    return grouped.sort_values(cols, kind="stable")


def schema_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        nonnull = s.dropna()
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "rows": len(s),
            "missing_n": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100.0),
            "nunique": int(s.nunique(dropna=True)),
            "sample_values": " | ".join(map(str, nonnull.head(3).tolist())),
        }
        if pd.api.types.is_numeric_dtype(s) and len(nonnull):
            q = nonnull.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            row.update(
                {
                    "min": float(nonnull.min()),
                    "p01": float(q.loc[0.01]),
                    "p25": float(q.loc[0.25]),
                    "median": float(q.loc[0.5]),
                    "p75": float(q.loc[0.75]),
                    "p99": float(q.loc[0.99]),
                    "max": float(nonnull.max()),
                    "mean": float(nonnull.mean()),
                    "std": float(nonnull.std()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_check(checks: list[dict], name: str, mask, note: str, severity="error"):
    if np.isscalar(mask):
        violations = int(mask)
        denominator = 1
    else:
        values = pd.Series(mask).fillna(False)
        violations = int(values.sum())
        denominator = int(len(values))
    checks.append(
        {
            "check": name,
            "violations": violations,
            "denominator": denominator,
            "violation_pct": 100.0 * violations / max(denominator, 1),
            "severity": severity,
            "note": note,
        }
    )


def clipped_metrics(y: pd.Series, pred: pd.Series) -> dict:
    valid = y.notna() & pred.notna()
    yy = y.loc[valid].astype(int).to_numpy()
    pp = np.clip(pred.loc[valid].astype(float).to_numpy(), 1e-6, 1.0 - 1e-6)
    return {
        "n": int(valid.sum()),
        "coverage_pct": float(valid.mean() * 100.0),
        "mean_prediction": float(pp.mean()),
        "actual_rate": float(yy.mean()),
        "brier": float(brier_score_loss(yy, pp)),
        "logloss": float(log_loss(yy, pp, labels=[0, 1])),
        "auc": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None,
    }


def js_divergence(a: pd.Series, b: pd.Series) -> float:
    keys = a.index.union(b.index)
    p = a.reindex(keys, fill_value=0).astype(float).to_numpy()
    q = b.reindex(keys, fill_value=0).astype(float).to_numpy()
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()


def save_plot(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


print("[1/7] Loading train/test data...")
train = pd.read_csv(DATA / "train.csv", low_memory=False)
test = pd.read_csv(DATA / "test.csv", low_memory=False)
submission = pd.read_csv(DATA / "sample_submission.csv", low_memory=False)

train["_row_seq"] = pd.to_numeric(
    train["row_id"].astype(str).str.extract(r"(\d+)$", expand=False), errors="coerce"
)
test["_row_seq"] = pd.to_numeric(
    test["row_id"].astype(str).str.extract(r"(\d+)$", expand=False), errors="coerce"
)
global_rate = float(train[TARGET].mean())

overview = {
    "train_shape": list(train.drop(columns="_row_seq").shape),
    "test_sample_shape": list(test.drop(columns="_row_seq").shape),
    "submission_sample_shape": list(submission.shape),
    "train_memory_mb": float(train.memory_usage(deep=True).sum() / 1024**2),
    "test_memory_mb": float(test.memory_usage(deep=True).sum() / 1024**2),
    "target_success_n": int(train[TARGET].sum()),
    "target_failure_n": int((train[TARGET] == 0).sum()),
    "target_success_rate": global_rate,
    "seasons": sorted(map(int, train["season"].dropna().unique())),
    "row_id_unique": bool(train["row_id"].is_unique),
    "train_input_columns_equal_test_columns": [
        c for c in train.columns if c not in {TARGET, "_row_seq"}
    ]
    == [c for c in test.columns if c != "_row_seq"],
    "train_only_columns": sorted(
        set(train.columns) - {TARGET, "_row_seq"} - set(test.columns)
    ),
    "test_only_columns": sorted(set(test.columns) - {"_row_seq"} - set(train.columns)),
    "test_sample_seasons": sorted(map(int, test["season"].dropna().unique())),
    "submission_row_ids_match_test_sample": bool(
        submission["row_id"].astype(str).tolist() == test["row_id"].astype(str).tolist()
    ),
}
write_json("main_overview.json", overview)
schema_profile(train.drop(columns="_row_seq")).to_csv(
    OUT / "train_schema_profile.csv", index=False, encoding="utf-8-sig"
)
schema_profile(test.drop(columns="_row_seq")).to_csv(
    OUT / "test_sample_schema_profile.csv", index=False, encoding="utf-8-sig"
)

print("[2/7] Running integrity and redundancy checks...")
checks: list[dict] = []
add_check(checks, "row_id_null", train["row_id"].isna(), "row_id must be present")
add_check(
    checks,
    "row_id_duplicate",
    train["row_id"].duplicated(keep=False),
    "row_id must be unique",
)
add_check(
    checks,
    "row_id_prefix",
    ~train["row_id"].astype(str).str.match(r"^TRAIN_\d+$"),
    "Expected TRAIN_<integer>",
)
add_check(
    checks,
    "target_not_binary",
    ~train[TARGET].isin([0, 1]),
    "control_success must be 0/1",
)
add_check(
    checks,
    "season_outside_2019_2024",
    ~train["season"].between(2019, 2024),
    "Documented training horizon is 2019-2024",
)
for col, low, high in [
    ("balls_before", 0, 3),
    ("strikes_before", 0, 2),
    ("outs_before", 0, 2),
    ("game_month", 1, 12),
    ("game_dayofweek", 0, 6),
    ("inning", 1, 30),
]:
    add_check(
        checks,
        f"{col}_outside_range",
        ~train[col].between(low, high),
        f"Expected range [{low}, {high}]",
    )
add_check(
    checks,
    "run_total_mismatch",
    train["run_total_before"] != train["run_top_before"] + train["run_bot_before"],
    "run_total_before should equal top + bottom scores",
)
add_check(
    checks,
    "home_score_diff_mismatch",
    train["score_diff_home"] != train["run_bot_before"] - train["run_top_before"],
    "score_diff_home should equal home(bottom) - away(top)",
)
expected_pitcher_diff = np.where(
    train["top_bottom"].eq("T"), train["score_diff_home"], -train["score_diff_home"]
)
add_check(
    checks,
    "pitcher_team_score_diff_mismatch",
    train["score_diff_pitcher_team"].to_numpy() != expected_pitcher_diff,
    "Pitcher is home in top half and away in bottom half",
)
runner_sum = train[["runner_on_1b", "runner_on_2b", "runner_on_3b"]].sum(axis=1)
add_check(
    checks,
    "runner_count_mismatch",
    train["num_runners_on"] != runner_sum,
    "num_runners_on should equal the three runner flags",
)
expected_base = (
    train["runner_on_1b"].map({0: "_", 1: "1"})
    + train["runner_on_2b"].map({0: "_", 1: "2"})
    + train["runner_on_3b"].map({0: "_", 1: "3"})
)
add_check(
    checks,
    "base_state_mismatch",
    train["base_state"] != expected_base,
    "base_state should encode the three runner flags",
)
win_expectancy_deviation = (
    train["home_win_expectancy"] + train["away_win_expectancy"] - 100.0
)
add_check(
    checks,
    "win_expectancy_not_exactly_100",
    ~np.isclose(win_expectancy_deviation, 0.0, atol=1e-6),
    "Informational: documented complements can differ by 0.1 due to one-decimal rounding",
    severity="info",
)
add_check(
    checks,
    "win_expectancy_deviation_gt_0_1",
    win_expectancy_deviation.abs() > 0.100001,
    "Home and away win expectancy should sum to 100 within displayed rounding",
)
write_json(
    "win_expectancy_rounding_profile.json",
    {
        "deviation_value_counts": {
            str(k): int(v)
            for k, v in win_expectancy_deviation.round(6).value_counts().sort_index().items()
        },
        "max_absolute_deviation": float(win_expectancy_deviation.abs().max()),
    },
)
for col in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
    add_check(
        checks,
        f"{col}_negative",
        train[col] < 0,
        "Historical sample size cannot be negative",
    )
for col in RATE_COLUMNS:
    add_check(
        checks,
        f"{col}_outside_0_1",
        train[col].notna() & ~train[col].between(0, 1),
        "Rate features should lie in [0, 1]",
    )
pitchmix_sum = train[
    [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
].sum(axis=1, min_count=3)
add_check(
    checks,
    "pitchmix_sum_not_1",
    pitchmix_sum.notna() & ~np.isclose(pitchmix_sum, 1.0, atol=2e-5),
    "Three documented pitch-mix shares should sum to 1 when all are present",
    severity="warning",
)
add_check(
    checks,
    "pitcher_n_pitchmix_n_mismatch",
    train["asof_pitcher_n"] != train["asof_pitcher_pitchmix_n"],
    "Checks whether two sample-size columns are exact duplicates",
    severity="info",
)
pd.DataFrame(checks).to_csv(
    OUT / "integrity_checks.csv", index=False, encoding="utf-8-sig"
)

redundancy = []
exact_pairs = [
    ("run_total_before", "run_top_before + run_bot_before"),
    ("score_diff_home", "run_bot_before - run_top_before"),
    ("num_runners_on", "runner flags sum"),
    ("base_state", "runner flags encoding"),
    ("away_win_expectancy", "100 - home_win_expectancy"),
    ("asof_pitcher_pitchmix_n", "asof_pitcher_n"),
]
for col, derived_from in exact_pairs:
    redundancy.append({"column": col, "derived_from": derived_from})
pd.DataFrame(redundancy).to_csv(
    OUT / "known_redundant_features.csv", index=False, encoding="utf-8-sig"
)

feature_cols = [c for c in train.columns if c not in {TARGET, "_row_seq", "row_id"}]
feature_hash = pd.util.hash_pandas_object(train[feature_cols], index=False)
duplicate_mask = feature_hash.duplicated(keep=False)
duplicate_groups = (
    pd.DataFrame(
        {
            "feature_hash": feature_hash.loc[duplicate_mask].to_numpy(),
            TARGET: train.loc[duplicate_mask, TARGET].to_numpy(),
        }
    )
    .groupby("feature_hash", observed=True)[TARGET]
    .agg(n="size", target_nunique="nunique", success_rate="mean")
    .reset_index()
)
duplicate_summary = {
    "duplicate_feature_rows": int(duplicate_mask.sum()),
    "duplicate_feature_row_pct": float(duplicate_mask.mean() * 100.0),
    "duplicate_groups": int(len(duplicate_groups)),
    "conflicting_target_groups": int((duplicate_groups["target_nunique"] > 1).sum()),
}
write_json("duplicate_feature_summary.json", duplicate_summary)

print("[3/7] Profiling target, missingness, entities, and time structure...")
train["count_state"] = (
    train["balls_before"].astype(str) + "-" + train["strikes_before"].astype(str)
)
train["hand_matchup"] = (
    train["pitcher_hand"].astype(str) + "v" + train["batter_hand"].astype(str)
)
train["inning_bucket"] = pd.cut(
    train["inning"], [0, 3, 6, 9, np.inf], labels=["1-3", "4-6", "7-9", "10+"]
)
train["score_bucket"] = pd.cut(
    train["score_diff_pitcher_team"],
    [-np.inf, -5.5, -3.5, -1.5, -0.5, 0.5, 1.5, 3.5, 5.5, np.inf],
    labels=["<=-6", "-5~-4", "-3~-2", "-1", "0", "+1", "+2~+3", "+4~+5", ">=+6"],
)
train["leverage_bucket"] = pd.cut(
    train["li"],
    [-np.inf, 0.5, 1.0, 2.0, 3.0, np.inf],
    labels=["<=0.5", "0.5-1", "1-2", "2-3", ">3"],
)
train["pitcher_history_bucket"] = pd.cut(
    train["asof_pitcher_n"],
    [-1, 0, 10, 50, 200, 1000, np.inf],
    labels=["0", "1-10", "11-50", "51-200", "201-1000", ">1000"],
)
train["batter_history_bucket"] = pd.cut(
    train["asof_batter_n"],
    [-1, 0, 10, 50, 200, 1000, np.inf],
    labels=["0", "1-10", "11-50", "51-200", "201-1000", ">1000"],
)

group_specs = {
    "season": ["season"],
    "season_game_type": ["season", "game_type"],
    "season_month": ["season", "game_month"],
    "season_hand_matchup": ["season", "hand_matchup"],
    "month": ["game_month"],
    "dayofweek": ["game_dayofweek"],
    "count_state": ["balls_before", "strikes_before", "count_state"],
    "outs": ["outs_before"],
    "inning": ["inning"],
    "inning_bucket": ["inning_bucket"],
    "top_bottom": ["top_bottom"],
    "game_type": ["game_type"],
    "base_state": ["base_state"],
    "num_runners": ["num_runners_on"],
    "score_bucket": ["score_bucket"],
    "leverage_bucket": ["leverage_bucket"],
    "pitcher_hand": ["pitcher_hand"],
    "batter_hand": ["batter_hand"],
    "hand_matchup": ["hand_matchup"],
    "pitcher_team": ["pitcher_team_id"],
    "batter_team": ["batter_team_id"],
    "pitcher_history_bucket": ["pitcher_history_bucket"],
    "batter_history_bucket": ["batter_history_bucket"],
}
group_outputs = {}
for name, cols in group_specs.items():
    table = group_target(train, cols, global_rate)
    table.to_csv(OUT / f"target_by_{name}.csv", index=False, encoding="utf-8-sig")
    group_outputs[name] = int(len(table))

missing_rows = []
for col in train.columns:
    if col.startswith("_") or not train[col].isna().any():
        continue
    miss = train[col].isna()
    missing_rows.append(
        {
            "column": col,
            "missing_n": int(miss.sum()),
            "missing_pct": float(miss.mean() * 100.0),
            "success_rate_when_missing": float(train.loc[miss, TARGET].mean()),
            "success_rate_when_present": float(train.loc[~miss, TARGET].mean()),
            "rate_gap_pp": float(
                (train.loc[miss, TARGET].mean() - train.loc[~miss, TARGET].mean()) * 100.0
            ),
        }
    )
missing_effects = pd.DataFrame(missing_rows).sort_values("missing_pct", ascending=False)
missing_effects.to_csv(OUT / "missingness_effects.csv", index=False, encoding="utf-8-sig")

asof_cols = [c for c in train.columns if c.startswith("asof_")]
missing_matrix = train[asof_cols].isna().to_numpy(dtype=np.uint32)
missing_codes = np.zeros(len(train), dtype=np.uint32)
for bit in range(len(asof_cols)):
    missing_codes |= missing_matrix[:, bit] << bit
missing_patterns = (
    pd.Series(missing_codes)
    .value_counts()
    .head(25)
    .rename_axis("missing_code")
    .reset_index(name="n")
)
missing_patterns["missing_columns"] = missing_patterns["missing_code"].map(
    lambda code: " | ".join(
        col for bit, col in enumerate(asof_cols) if int(code) & (1 << bit)
    )
    or "<none>"
)
missing_patterns["pct"] = missing_patterns["n"] / len(train) * 100.0
missing_patterns.to_csv(OUT / "asof_missing_patterns_top25.csv", index=False, encoding="utf-8-sig")

pitcher_stats = (
    train.groupby("pitcher_id", observed=True)
    .agg(
        n=(TARGET, "size"),
        successes=(TARGET, "sum"),
        success_rate=(TARGET, "mean"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        n_seasons=("season", "nunique"),
        pitcher_hand=("pitcher_hand", lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
    )
    .reset_index()
)
pitcher_stats["ci95_low"], pitcher_stats["ci95_high"] = wilson_bounds(
    pitcher_stats["successes"], pitcher_stats["n"]
)
pitcher_stats.sort_values("n", ascending=False).to_csv(
    OUT / "pitcher_target_profile.csv", index=False, encoding="utf-8-sig"
)
batter_stats = (
    train.groupby("batter_id", observed=True)
    .agg(
        n=(TARGET, "size"),
        successes=(TARGET, "sum"),
        success_rate=(TARGET, "mean"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        n_seasons=("season", "nunique"),
    )
    .reset_index()
)
batter_stats.sort_values("n", ascending=False).to_csv(
    OUT / "batter_target_profile.csv", index=False, encoding="utf-8-sig"
)

entity_summary = {
    "pitchers": int(train["pitcher_id"].nunique()),
    "batters": int(train["batter_id"].nunique()),
    "pitcher_teams": int(train["pitcher_team_id"].nunique()),
    "batter_teams": int(train["batter_team_id"].nunique()),
    "top_10_pitcher_volume_share_pct": float(
        pitcher_stats.nlargest(10, "n")["n"].sum() / len(train) * 100.0
    ),
    "top_50_pitcher_volume_share_pct": float(
        pitcher_stats.nlargest(50, "n")["n"].sum() / len(train) * 100.0
    ),
    "pitchers_n_ge_100": int((pitcher_stats["n"] >= 100).sum()),
    "pitchers_n_ge_500": int((pitcher_stats["n"] >= 500).sum()),
    "pitchers_n_ge_1000": int((pitcher_stats["n"] >= 1000).sum()),
    "stable_pitcher_rate_quantiles_n_ge_500": {
        str(k): float(v)
        for k, v in pitcher_stats.loc[pitcher_stats["n"] >= 500, "success_rate"]
        .quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
        .items()
    },
}
write_json("entity_summary.json", entity_summary)

entity_by_season = (
    train.groupby("season")
    .agg(
        rows=(TARGET, "size"),
        pitchers=("pitcher_id", "nunique"),
        batters=("batter_id", "nunique"),
        pitcher_teams=("pitcher_team_id", "nunique"),
        batter_teams=("batter_team_id", "nunique"),
        success_rate=(TARGET, "mean"),
    )
    .reset_index()
)
entity_by_season.to_csv(OUT / "entity_by_season.csv", index=False, encoding="utf-8-sig")

season_players = {
    int(season): set(group["pitcher_id"].unique()) for season, group in train.groupby("season")
}
overlap_rows = []
seasons = sorted(season_players)
for prev, curr in zip(seasons[:-1], seasons[1:]):
    a, b = season_players[prev], season_players[curr]
    overlap_rows.append(
        {
            "from_season": prev,
            "to_season": curr,
            "from_pitchers": len(a),
            "to_pitchers": len(b),
            "overlap": len(a & b),
            "retained_from_pct": len(a & b) / len(a) * 100.0,
            "new_in_to_pct": len(b - a) / len(b) * 100.0,
            "jaccard": len(a & b) / len(a | b),
        }
    )
pd.DataFrame(overlap_rows).to_csv(
    OUT / "pitcher_overlap_consecutive_seasons.csv", index=False, encoding="utf-8-sig"
)

pitcher_season_stats = (
    train.groupby(["season", "pitcher_id"], observed=True)[TARGET]
    .agg(n="size", success_rate="mean")
    .reset_index()
)
pitcher_drift_rows = []
for prev, curr in zip(seasons[:-1], seasons[1:]):
    prev_stats = pitcher_season_stats[pitcher_season_stats["season"] == prev][
        ["pitcher_id", "n", "success_rate"]
    ].rename(columns={"n": "n_prev", "success_rate": "rate_prev"})
    curr_stats = pitcher_season_stats[pitcher_season_stats["season"] == curr][
        ["pitcher_id", "n", "success_rate"]
    ].rename(columns={"n": "n_curr", "success_rate": "rate_curr"})
    paired = prev_stats.merge(curr_stats, on="pitcher_id", how="inner")
    stable = paired[(paired["n_prev"] >= 100) & (paired["n_curr"] >= 100)].copy()
    delta = stable["rate_curr"] - stable["rate_prev"]
    pitcher_drift_rows.append(
        {
            "from_season": prev,
            "to_season": curr,
            "overlap_pitchers": int(len(paired)),
            "pitchers_n_ge_100_both": int(len(stable)),
            "median_within_pitcher_delta_pp": float(delta.median() * 100.0),
            "current_volume_weighted_delta_pp": float(
                np.average(delta, weights=stable["n_curr"]) * 100.0
            ),
            "share_of_stable_pitchers_declining_pct": float((delta < 0).mean() * 100.0),
        }
    )
pd.DataFrame(pitcher_drift_rows).to_csv(
    OUT / "within_pitcher_drift_consecutive_seasons.csv",
    index=False,
    encoding="utf-8-sig",
)

# row_id and as-of progression are inspected explicitly because using future rows is forbidden.
ordered = train.sort_values(["pitcher_id", "_row_seq"], kind="stable").copy()
ordered["pitcher_cumcount"] = ordered.groupby("pitcher_id").cumcount()
ordered["n_diff"] = ordered.groupby("pitcher_id")["asof_pitcher_n"].diff()
ordered["next_n"] = ordered.groupby("pitcher_id")["asof_pitcher_n"].shift(-1)
ordered["next_success_rate"] = ordered.groupby("pitcher_id")[
    "asof_pitcher_success_rate"
].shift(-1)
prev_successes = (
    ordered["asof_pitcher_success_rate"].fillna(0.0) * ordered["asof_pitcher_n"]
)
next_successes = ordered["next_success_rate"] * ordered["next_n"]
ordered["future_implied_target"] = next_successes - prev_successes
recoverable = ordered["next_n"].eq(ordered["asof_pitcher_n"] + 1) & ordered[
    "next_success_rate"
].notna()
recovery_accuracy = np.isclose(
    ordered.loc[recoverable, "future_implied_target"],
    ordered.loc[recoverable, TARGET],
    atol=2e-4,
)
rounded_recovery_accuracy = np.rint(
    ordered.loc[recoverable, "future_implied_target"]
).eq(ordered.loc[recoverable, TARGET])
row_order_summary = {
    "row_seq_missing": int(train["_row_seq"].isna().sum()),
    "row_seq_unique": bool(train["_row_seq"].is_unique),
    "row_seq_monotonic_increasing": bool(train["_row_seq"].is_monotonic_increasing),
    "season_monotonic_by_row_seq": bool(
        train.sort_values("_row_seq")["season"].is_monotonic_increasing
    ),
    "row_seq_season_rank_corr": float(
        train["_row_seq"].rank().corr(train["season"].rank())
    ),
    "asof_n_equals_dataset_pitcher_cumcount_pct": float(
        (ordered["asof_pitcher_n"] == ordered["pitcher_cumcount"]).mean() * 100.0
    ),
    "within_pitcher_n_diff_eq_1_pct": float(ordered["n_diff"].eq(1).mean() * 100.0),
    "within_pitcher_n_diff_lt_0_n": int(ordered["n_diff"].lt(0).sum()),
    "within_pitcher_n_diff_gt_1_n": int(ordered["n_diff"].gt(1).sum()),
    "future_row_target_recovery_coverage_pct": float(recoverable.mean() * 100.0),
    "future_row_target_recovery_raw_tolerance_accuracy_pct": float(
        recovery_accuracy.mean() * 100.0
    ),
    "future_row_target_recovery_after_rounding_accuracy_pct": float(
        rounded_recovery_accuracy.mean() * 100.0
    ),
    "future_implied_target_max_rounding_error": float(
        (
            ordered.loc[recoverable, "future_implied_target"]
            - np.rint(ordered.loc[recoverable, "future_implied_target"])
        )
        .abs()
        .max()
    ),
    "warning": "The recovery statistic is a leakage diagnostic only. Test rows are independent and may not be used to construct features.",
}
write_json("row_order_and_future_leakage_diagnostic.json", row_order_summary)

train["row_decile_within_season"] = train.groupby("season")["_row_seq"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop")
)
group_target(train, ["season", "row_decile_within_season"], global_rate).to_csv(
    OUT / "target_by_row_decile_within_season.csv", index=False, encoding="utf-8-sig"
)
group_target(
    train, ["season", "row_decile_within_season", "game_type"], global_rate
).to_csv(
    OUT / "target_by_row_decile_season_game_type.csv",
    index=False,
    encoding="utf-8-sig",
)
row_game_structure = (
    train.groupby(["season", "game_type"], observed=True)
    .agg(
        n=("_row_seq", "size"),
        row_seq_min=("_row_seq", "min"),
        row_seq_max=("_row_seq", "max"),
        success_rate=(TARGET, "mean"),
    )
    .reset_index()
)
gap_counts = (
    train.sort_values("_row_seq")
    .groupby(["season", "game_type"], observed=True)["_row_seq"]
    .apply(lambda s: int(s.diff().gt(1).sum()))
    .rename("internal_sequence_gaps")
    .reset_index()
)
row_game_structure.merge(gap_counts, on=["season", "game_type"]).to_csv(
    OUT / "row_id_game_type_structure.csv", index=False, encoding="utf-8-sig"
)

print("[4/7] Measuring univariate signal and calibration...")
sample = train.sample(n=min(300_000, len(train)), random_state=20260805)
numeric_candidates = [
    c
    for c in train.columns
    if pd.api.types.is_numeric_dtype(train[c])
    and c not in {TARGET, "_row_seq", "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
]
signal_rows = []
for col in numeric_candidates:
    valid = sample[col].notna()
    x = sample.loc[valid, col].astype(float)
    y = sample.loc[valid, TARGET].astype(int)
    if x.nunique() < 2 or y.nunique() < 2:
        auc = None
        corr = None
    else:
        auc = float(roc_auc_score(y, x))
        corr = float(x.corr(y))
    x0 = x[y.eq(0)]
    x1 = x[y.eq(1)]
    pooled = math.sqrt((x0.var() + x1.var()) / 2.0) if len(x0) and len(x1) else np.nan
    signal_rows.append(
        {
            "feature": col,
            "sample_n": int(valid.sum()),
            "missing_pct": float((~valid).mean() * 100.0),
            "mean_target_0": float(x0.mean()) if len(x0) else None,
            "mean_target_1": float(x1.mean()) if len(x1) else None,
            "standardized_mean_difference": float((x1.mean() - x0.mean()) / pooled)
            if pooled and not np.isnan(pooled)
            else None,
            "pearson_corr": corr,
            "auc_raw_direction": auc,
            "auc_strength": abs(auc - 0.5) if auc is not None else None,
        }
    )
numeric_signal = pd.DataFrame(signal_rows).sort_values("auc_strength", ascending=False)
numeric_signal.to_csv(OUT / "numeric_univariate_signal.csv", index=False, encoding="utf-8-sig")

categorical_candidates = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning_bucket",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "hand_matchup",
    "pitcher_team_id",
    "batter_team_id",
]
cat_signal_rows = []
for col in categorical_candidates:
    stat = train.groupby(col, dropna=False, observed=True)[TARGET].agg(["size", "mean"])
    valid_stat = stat[stat["size"] >= 100]
    weighted_var = float(
        np.average((stat["mean"] - global_rate) ** 2, weights=stat["size"])
    )
    cat_signal_rows.append(
        {
            "feature": col,
            "n_categories": int(len(stat)),
            "categories_n_ge_100": int(len(valid_stat)),
            "weighted_target_rate_sd_pp": math.sqrt(weighted_var) * 100.0,
            "min_rate_n_ge_100": float(valid_stat["mean"].min()) if len(valid_stat) else None,
            "max_rate_n_ge_100": float(valid_stat["mean"].max()) if len(valid_stat) else None,
            "range_pp_n_ge_100": float(
                (valid_stat["mean"].max() - valid_stat["mean"].min()) * 100.0
            )
            if len(valid_stat)
            else None,
        }
    )
pd.DataFrame(cat_signal_rows).sort_values(
    "weighted_target_rate_sd_pp", ascending=False
).to_csv(OUT / "categorical_target_association.csv", index=False, encoding="utf-8-sig")

overall_constant = pd.Series(global_rate, index=train.index)
baseline_metrics = {
    "global_constant_all_rows": clipped_metrics(train[TARGET], overall_constant),
    "asof_pitcher_success_rate_nonmissing": clipped_metrics(
        train[TARGET], train["asof_pitcher_success_rate"]
    ),
    "asof_pitcher_success_rate_global_fill": clipped_metrics(
        train[TARGET], train["asof_pitcher_success_rate"].fillna(global_rate)
    ),
    "asof_batter_success_rate_global_fill": clipped_metrics(
        train[TARGET], train["asof_batter_success_rate"].fillna(global_rate)
    ),
}
prior_mask = train["season"] <= 2023
valid_2024 = train["season"] == 2024
prior_rate = float(train.loc[prior_mask, TARGET].mean())
baseline_metrics["2024_prior_seasons_constant"] = clipped_metrics(
    train.loc[valid_2024, TARGET], pd.Series(prior_rate, index=train.index[valid_2024])
)
baseline_metrics["2024_asof_pitcher_success_rate_prior_fill"] = clipped_metrics(
    train.loc[valid_2024, TARGET],
    train.loc[valid_2024, "asof_pitcher_success_rate"].fillna(prior_rate),
)
write_json("descriptive_baseline_metrics.json", baseline_metrics)

season_calibration_rows = []
for season, season_df in train.groupby("season", observed=True):
    historical = season_df["asof_pitcher_success_rate"]
    season_metrics = clipped_metrics(season_df[TARGET], historical.fillna(global_rate))
    season_calibration_rows.append(
        {
            "season": int(season),
            "n": int(len(season_df)),
            "actual_rate": float(season_df[TARGET].mean()),
            "mean_asof_pitcher_rate": float(historical.mean()),
            "calibration_gap_pp": float(
                (season_df[TARGET].mean() - historical.mean()) * 100.0
            ),
            "brier_global_fill": season_metrics["brier"],
            "logloss_global_fill": season_metrics["logloss"],
            "auc": season_metrics["auc"],
        }
    )
pd.DataFrame(season_calibration_rows).to_csv(
    OUT / "asof_pitcher_rate_calibration_by_season.csv",
    index=False,
    encoding="utf-8-sig",
)

cal = train[[TARGET, "asof_pitcher_success_rate", "season"]].copy()
cal["rate_bin"] = pd.cut(
    cal["asof_pitcher_success_rate"],
    np.linspace(0, 1, 21),
    include_lowest=True,
)
calibration = (
    cal.dropna(subset=["asof_pitcher_success_rate"])
    .groupby("rate_bin", observed=True)
    .agg(
        n=(TARGET, "size"),
        mean_prediction=("asof_pitcher_success_rate", "mean"),
        actual_rate=(TARGET, "mean"),
    )
    .reset_index()
)
calibration["calibration_gap_pp"] = (
    calibration["actual_rate"] - calibration["mean_prediction"]
) * 100.0
calibration.to_csv(
    OUT / "asof_pitcher_rate_calibration.csv", index=False, encoding="utf-8-sig"
)

numeric_corr_cols = [
    c
    for c in numeric_candidates
    if sample[c].nunique(dropna=True) > 1 and c not in {"season"}
]
corr = sample[numeric_corr_cols + [TARGET]].corr()
corr.to_csv(OUT / "numeric_correlation_matrix_sample.csv", encoding="utf-8-sig")

print("[5/7] Loading and profiling Trackman history...")
track = pd.read_csv(DATA / "trackman_history.csv", low_memory=False)
track["game_date_parsed"] = pd.to_datetime(track["game_date"], format="mixed", errors="coerce")
track_overview = {
    "shape": list(track.drop(columns="game_date_parsed").shape),
    "memory_mb": float(track.memory_usage(deep=True).sum() / 1024**2),
    "seasons": sorted(map(int, track["season"].dropna().unique())),
    "date_min": track["game_date_parsed"].min(),
    "date_max": track["game_date_parsed"].max(),
    "pitchers": int(track["pitcher_trackman_id"].nunique()),
    "batters": int(track["batter_trackman_id"].nunique()),
    "pitcher_teams": int(track["pitcher_team"].nunique()),
    "batter_teams": int(track["batter_team"].nunique()),
    "trackman_id_unique": bool(track["trackman_id"].is_unique),
}
write_json("trackman_overview.json", track_overview)
schema_profile(track.drop(columns="game_date_parsed")).to_csv(
    OUT / "trackman_schema_profile.csv", index=False, encoding="utf-8-sig"
)

track_checks: list[dict] = []
add_check(
    track_checks,
    "trackman_id_duplicate",
    track["trackman_id"].duplicated(keep=False),
    "trackman_id should be unique",
)
add_check(
    track_checks,
    "game_date_parse_failure",
    track["game_date_parsed"].isna(),
    "game_date should parse as MM/DD/YYYY",
)
add_check(
    track_checks,
    "game_date_season_mismatch",
    track["game_date_parsed"].notna()
    & track["season"].ne(track["game_date_parsed"].dt.year),
    "season should equal game_date year",
)
add_check(
    track_checks,
    "game_pitch_no_duplicate",
    track.duplicated(["trackman_game_id", "pitch_no"], keep=False),
    "A game and pitch number should usually identify one pitch",
    severity="warning",
)
for col, low, high in [
    ("balls_before", 0, 3),
    ("strikes_before", 0, 2),
    ("outs_before", 0, 2),
    ("pitch_of_pa", 1, 30),
    ("inning", 1, 30),
]:
    add_check(
        track_checks,
        f"{col}_outside_range",
        track[col].notna() & ~track[col].between(low, high),
        f"Expected range [{low}, {high}]",
    )
pd.DataFrame(track_checks).to_csv(
    OUT / "trackman_integrity_checks.csv", index=False, encoding="utf-8-sig"
)

track_by_season = (
    track.groupby("season")
    .agg(
        rows=("trackman_id", "size"),
        games=("trackman_game_id", "nunique"),
        pitchers=("pitcher_trackman_id", "nunique"),
        batters=("batter_trackman_id", "nunique"),
        pitcher_teams=("pitcher_team", "nunique"),
        date_min=("game_date_parsed", "min"),
        date_max=("game_date_parsed", "max"),
    )
    .reset_index()
)
track_by_season.to_csv(OUT / "trackman_by_season.csv", index=False, encoding="utf-8-sig")

pitch_group = (
    track.groupby(["season", "pitch_type_group"], dropna=False, observed=True)
    .size()
    .rename("n")
    .reset_index()
)
pitch_group["share"] = pitch_group["n"] / pitch_group.groupby("season")["n"].transform("sum")
pitch_group.to_csv(
    OUT / "trackman_pitch_group_by_season.csv", index=False, encoding="utf-8-sig"
)

type_concordance = {
    "both_present_n": int((track["tagged_pitch_type"].notna() & track["auto_pitch_type"].notna()).sum()),
    "exact_match_pct_both_present": float(
        (
            track.loc[
                track["tagged_pitch_type"].notna() & track["auto_pitch_type"].notna(),
                "tagged_pitch_type",
            ]
            == track.loc[
                track["tagged_pitch_type"].notna() & track["auto_pitch_type"].notna(),
                "auto_pitch_type",
            ]
        ).mean()
        * 100.0
    ),
    "tagged_missing_pct": float(track["tagged_pitch_type"].isna().mean() * 100.0),
    "auto_missing_pct": float(track["auto_pitch_type"].isna().mean() * 100.0),
    "group_missing_pct": float(track["pitch_type_group"].isna().mean() * 100.0),
}
write_json("trackman_pitch_type_concordance.json", type_concordance)

physical_cols = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
physical_rows = []
for col in physical_cols:
    s = track[col].dropna().astype(float)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    physical_rows.append(
        {
            "feature": col,
            "n": int(s.size),
            "missing_pct": float(track[col].isna().mean() * 100.0),
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p25": float(q1),
            "median": float(s.median()),
            "p75": float(q3),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "three_iqr_lower": float(lower),
            "three_iqr_upper": float(upper),
            "outside_three_iqr_n": int(((s < lower) | (s > upper)).sum()),
            "outside_three_iqr_pct": float(((s < lower) | (s > upper)).mean() * 100.0),
        }
    )
pd.DataFrame(physical_rows).to_csv(
    OUT / "trackman_physical_profile.csv", index=False, encoding="utf-8-sig"
)

physical_by_group = (
    track.groupby("pitch_type_group", dropna=False, observed=True)[physical_cols]
    .agg(["count", "mean", "std", "median"])
)
physical_by_group.columns = [f"{a}_{b}" for a, b in physical_by_group.columns]
physical_by_group.reset_index().to_csv(
    OUT / "trackman_physical_by_pitch_group.csv", index=False, encoding="utf-8-sig"
)

track_pitcher = (
    track.groupby("pitcher_trackman_id", observed=True)
    .agg(
        n=("trackman_id", "size"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        n_seasons=("season", "nunique"),
        n_teams=("pitcher_team", "nunique"),
        n_hands=("pitcher_hand", "nunique"),
        rel_speed_mean=("rel_speed", "mean"),
        spin_rate_mean=("spin_rate", "mean"),
    )
    .reset_index()
)
track_pitcher.sort_values("n", ascending=False).to_csv(
    OUT / "trackman_pitcher_profile.csv", index=False, encoding="utf-8-sig"
)

print("[6/7] Assessing main-to-Trackman compatibility and linkage feasibility...")
main_pitcher_ids = set(train["pitcher_id"].dropna().astype(str))
track_pitcher_ids = set(track["pitcher_trackman_id"].dropna().astype(str))
main_batter_ids = set(train["batter_id"].dropna().astype(str))
track_batter_ids = set(track["batter_trackman_id"].dropna().astype(str))
id_linkage = {
    "pitcher_direct_id_overlap_n": len(main_pitcher_ids & track_pitcher_ids),
    "batter_direct_id_overlap_n": len(main_batter_ids & track_batter_ids),
    "main_pitcher_id_example": sorted(main_pitcher_ids)[:5],
    "trackman_pitcher_id_example": sorted(track_pitcher_ids)[:5],
    "main_pitcher_id_count": len(main_pitcher_ids),
    "trackman_pitcher_id_count": len(track_pitcher_ids),
    "conclusion": "No player-level join is assumed unless an explicit or independently validated crosswalk is built.",
}
write_json("main_trackman_id_linkage.json", id_linkage)

compare_rows = []
track_common = track.assign(
    top_bottom=track["top_bottom"].map({"Top": "T", "Bottom": "B"})
)
for season in sorted(set(train["season"]) & set(track["season"])):
    m = train[train["season"] == season]
    t = track_common[track_common["season"] == season]
    for col in [
        "game_month",
        "game_dayofweek",
        "inning",
        "top_bottom",
        "balls_before",
        "strikes_before",
        "outs_before",
    ]:
        compare_rows.append(
            {
                "season": int(season),
                "feature": col,
                "main_n": int(m[col].notna().sum()),
                "trackman_n": int(t[col].notna().sum()),
                "js_divergence_bits": js_divergence(
                    m[col].value_counts(dropna=False), t[col].value_counts(dropna=False)
                ),
            }
        )
pd.DataFrame(compare_rows).to_csv(
    OUT / "main_trackman_marginal_divergence.csv", index=False, encoding="utf-8-sig"
)

coverage_compare = entity_by_season.merge(
    track_by_season,
    on="season",
    how="outer",
    suffixes=("_main", "_trackman"),
)
coverage_compare["trackman_to_main_row_ratio"] = (
    coverage_compare["rows_trackman"] / coverage_compare["rows_main"]
)
coverage_compare.to_csv(
    OUT / "main_trackman_season_coverage.csv", index=False, encoding="utf-8-sig"
)

# Heuristic team candidates use only common pre-pitch distribution fingerprints.
# They are diagnostic hypotheses, not a validated join key.
def team_fingerprint(df: pd.DataFrame, team_col: str, top_col: str) -> pd.DataFrame:
    work = df.copy()
    work["count_state_fp"] = work["balls_before"].astype(str) + "-" + work[
        "strikes_before"
    ].astype(str)
    work["inning_bucket_fp"] = pd.cut(
        work["inning"], [0, 3, 6, 9, np.inf], labels=["1-3", "4-6", "7-9", "10+"]
    ).astype(str)
    pieces = []
    for feature in ["season", "game_month", "count_state_fp", "inning_bucket_fp", top_col]:
        tab = pd.crosstab(work[team_col], work[feature], normalize="index")
        tab.columns = [f"{feature}={v}" for v in tab.columns]
        pieces.append(tab)
    volume = work.groupby(team_col).size().rename("total_n").to_frame()
    result = volume.join(pieces, how="left").fillna(0.0)
    return result


main_fp = team_fingerprint(train, "pitcher_team_id", "top_bottom")
track_fp = team_fingerprint(track_common, "pitcher_team", "top_bottom")
common_fp_cols = sorted((set(main_fp.columns) & set(track_fp.columns)) - {"total_n"})
main_matrix = main_fp[common_fp_cols].to_numpy(float)
track_matrix = track_fp[common_fp_cols].to_numpy(float)
main_norm = np.linalg.norm(main_matrix, axis=1, keepdims=True)
track_norm = np.linalg.norm(track_matrix, axis=1, keepdims=True)
similarity = (main_matrix @ track_matrix.T) / np.maximum(main_norm @ track_norm.T, 1e-12)
candidate_rows = []
for i, main_id in enumerate(main_fp.index):
    order_idx = np.argsort(-similarity[i])[:3]
    for rank, j in enumerate(order_idx, start=1):
        candidate_rows.append(
            {
                "main_pitcher_team_id": main_id,
                "candidate_rank": rank,
                "trackman_pitcher_team": track_fp.index[j],
                "cosine_similarity": float(similarity[i, j]),
                "main_total_n": int(main_fp.iloc[i]["total_n"]),
                "trackman_total_n": int(track_fp.iloc[j]["total_n"]),
                "volume_ratio_trackman_to_main": float(
                    track_fp.iloc[j]["total_n"] / main_fp.iloc[i]["total_n"]
                ),
                "status": "heuristic_only_not_validated",
            }
        )
pd.DataFrame(candidate_rows).to_csv(
    OUT / "trackman_team_crosswalk_candidates.csv", index=False, encoding="utf-8-sig"
)

print("[7/7] Rendering key charts and final summary...")
sns.set_theme(style="whitegrid", context="notebook")

season_rate = pd.read_csv(OUT / "target_by_season.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=season_rate, x="season", y="success_rate", color="#2563EB", ax=ax)
ax.axhline(global_rate, color="#DC2626", linestyle="--", linewidth=1.5, label="Overall")
ax.set_ylim(max(0, season_rate["success_rate"].min() - 0.04), season_rate["success_rate"].max() + 0.04)
ax.set_title("Control-success rate by season")
ax.set_ylabel("Success rate")
ax.legend()
save_plot(fig, "01_target_rate_by_season.png")

count_rate = pd.read_csv(OUT / "target_by_count_state.csv")
pivot = count_rate.pivot(index="balls_before", columns="strikes_before", values="success_rate")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlBu", center=global_rate, ax=ax)
ax.set_title("Control-success rate by pre-pitch count")
save_plot(fig, "02_target_rate_count_heatmap.png")

fig, ax = plt.subplots(figsize=(9, 5.5))
miss_plot = missing_effects.head(20).sort_values("missing_pct")
sns.barplot(data=miss_plot, x="missing_pct", y="column", color="#F59E0B", ax=ax)
ax.set_title("Top missingness rates in train")
ax.set_xlabel("Missing (%)")
ax.set_ylabel("")
save_plot(fig, "03_missingness_top20.png")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    calibration["mean_prediction"],
    calibration["actual_rate"],
    marker="o",
    color="#2563EB",
    label="Observed",
)
ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="Perfect calibration")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Calibration of as-of pitcher success rate")
ax.set_xlabel("Mean historical rate")
ax.set_ylabel("Current-pitch success rate")
ax.legend()
save_plot(fig, "04_asof_pitcher_rate_calibration.png")

fig, ax = plt.subplots(figsize=(8, 5))
plot_pitchers = pitcher_stats[pitcher_stats["n"] >= 100].copy()
ax.scatter(
    plot_pitchers["n"],
    plot_pitchers["success_rate"],
    s=12,
    alpha=0.4,
    color="#0F766E",
)
ax.axhline(global_rate, color="#DC2626", linestyle="--", linewidth=1.2)
ax.set_xscale("log")
ax.set_title("Pitcher volume vs observed control-success rate (n>=100)")
ax.set_xlabel("Pitch count (log scale)")
ax.set_ylabel("Success rate")
save_plot(fig, "05_pitcher_volume_vs_rate.png")

signal_plot = numeric_signal.dropna(subset=["auc_strength"]).head(15).sort_values("auc_strength")
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.barplot(data=signal_plot, x="auc_strength", y="feature", color="#7C3AED", ax=ax)
ax.set_title("Top univariate numeric signal (|AUC - 0.5|, sampled)")
ax.set_xlabel("AUC distance from random")
ax.set_ylabel("")
save_plot(fig, "06_numeric_univariate_signal.png")

fig, ax = plt.subplots(figsize=(9, 7))
top_corr_cols = (
    numeric_signal.dropna(subset=["auc_strength"]).head(18)["feature"].tolist() + [TARGET]
)
sns.heatmap(
    sample[top_corr_cols].corr(),
    cmap="vlag",
    center=0,
    vmin=-1,
    vmax=1,
    square=False,
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Correlation among high-signal/redundant numeric features")
save_plot(fig, "07_numeric_correlation_heatmap.png")

pitch_group_plot = pitch_group.pivot(index="season", columns="pitch_type_group", values="share").fillna(0)
fig, ax = plt.subplots(figsize=(9, 5))
pitch_group_plot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c")
ax.set_title("Trackman pitch-type-group composition by season")
ax.set_ylabel("Share")
ax.legend(title="Pitch group", bbox_to_anchor=(1.02, 1), loc="upper left")
save_plot(fig, "08_trackman_pitch_group_by_season.png")

physical_sample = track.loc[track["pitch_type_group"].notna()].sample(
    n=min(150_000, track["pitch_type_group"].notna().sum()), random_state=20260805
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(
    data=physical_sample,
    x="pitch_type_group",
    y="rel_speed",
    showfliers=False,
    ax=axes[0],
)
axes[0].set_title("Release speed by pitch group")
axes[0].tick_params(axis="x", rotation=20)
sns.boxplot(
    data=physical_sample,
    x="pitch_type_group",
    y="spin_rate",
    showfliers=False,
    ax=axes[1],
)
axes[1].set_title("Spin rate by pitch group")
axes[1].tick_params(axis="x", rotation=20)
save_plot(fig, "09_trackman_physical_by_pitch_group.png")

final_summary = {
    "overview": overview,
    "entity_summary": entity_summary,
    "duplicate_summary": duplicate_summary,
    "row_order_summary": row_order_summary,
    "baseline_metrics": baseline_metrics,
    "trackman_overview": track_overview,
    "id_linkage": id_linkage,
    "generated_csv_count": len(list(OUT.glob("*.csv"))),
    "generated_png_count": len(list(OUT.glob("*.png"))),
}
write_json("eda_summary.json", final_summary)
print(f"EDA complete. Outputs: {OUT}")
