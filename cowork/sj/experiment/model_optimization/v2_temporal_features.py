from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_optuna_family import ROOT, SEED, TARGET, add_v1_features


WORK_DIR = ROOT / "experiment" / "model_optimization"
GROUP_SPECS = {
    "pitcher": ["pitcher_id"],
    "batter": ["batter_id"],
    "pitcher_team": ["pitcher_team_id"],
    "batter_team": ["batter_team_id"],
    "game_type": ["game_type"],
    "count": ["balls_before", "strikes_before"],
    "hand_matchup": ["pitcher_hand", "batter_hand"],
    "pitcher_batter_hand": ["pitcher_id", "batter_hand"],
    "pitcher_count": ["pitcher_id", "balls_before", "strikes_before"],
    "pitcher_game_type": ["pitcher_id", "game_type"],
    "batter_pitcher_hand": ["batter_id", "pitcher_hand"],
    "pitcher_team_game_type": ["pitcher_team_id", "game_type"],
}
ROW_RATE_SPECS = {
    "pitcher_success": ("asof_pitcher_success_rate", "asof_pitcher_n", 0.50),
    "pitcher_reverse": ("asof_pitcher_reverse_rate", "asof_pitcher_n", 0.23),
    "pitcher_middle": ("asof_pitcher_middle_rate", "asof_pitcher_n", 0.15),
    "pitcher_ball": ("asof_pitcher_ball_rate", "asof_pitcher_n", 0.50),
    "pitcher_strike": ("asof_pitcher_strike_rate", "asof_pitcher_n", 0.50),
    "batter_success": ("asof_batter_success_rate", "asof_batter_n", 0.50),
    "batter_middle": ("asof_batter_middle_rate", "asof_batter_n", 0.15),
    "pitcher_fastball": (
        "asof_pitcher_fastball_rate",
        "asof_pitcher_pitchmix_n",
        0.50,
    ),
    "pitcher_breaking": (
        "asof_pitcher_breaking_rate",
        "asof_pitcher_pitchmix_n",
        0.30,
    ),
    "pitcher_offspeed": (
        "asof_pitcher_offspeed_rate",
        "asof_pitcher_pitchmix_n",
        0.20,
    ),
}


def add_v2_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = add_v1_features(frame)
    for prefix, (rate_column, count_column, prior) in ROW_RATE_SPECS.items():
        rate = output[rate_column].astype(float)
        count = output[count_column].clip(lower=0).astype(float)
        output[f"{prefix}_is_missing"] = rate.isna().astype("int8")
        for strength in [10.0, 50.0, 200.0, 500.0]:
            suffix = int(strength)
            output[f"{prefix}_smoothed_{suffix}"] = (
                (rate.fillna(prior) * count + prior * strength)
                / (count + strength)
            ).astype("float32")
            output[f"{prefix}_reliability_{suffix}"] = (
                count / (count + strength)
            ).astype("float32")

    pitcher_n = output["asof_pitcher_n"]
    batter_n = output["asof_batter_n"]
    for threshold in [0, 25, 100, 500, 1000]:
        operator = "eq" if threshold == 0 else "le"
        output[f"pitcher_n_{operator}_{threshold}"] = getattr(pitcher_n, operator)(
            threshold
        ).astype("int8")
        output[f"batter_n_{operator}_{threshold}"] = getattr(batter_n, operator)(
            threshold
        ).astype("int8")

    success_recent = output[
        [
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
    ]
    middle_recent = output[
        [
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
        ]
    ]
    output["pitcher_recent_success_mean"] = success_recent.mean(axis=1).astype("float32")
    output["pitcher_recent_success_std"] = success_recent.std(axis=1).astype("float32")
    output["pitcher_recent_success_range"] = (
        success_recent.max(axis=1) - success_recent.min(axis=1)
    ).astype("float32")
    output["pitcher_recent_middle_mean"] = middle_recent.mean(axis=1).astype("float32")
    output["pitcher_recent_middle_std"] = middle_recent.std(axis=1).astype("float32")
    output["pitcher_recent_middle_range"] = (
        middle_recent.max(axis=1) - middle_recent.min(axis=1)
    ).astype("float32")
    output["pitcher_failure_rate_sum"] = (
        output["asof_pitcher_reverse_rate"]
        + output["asof_pitcher_middle_rate"]
        + 1.0
        - output["asof_pitcher_success_rate"]
    ).astype("float32")
    output["pitcher_control_component_gap"] = (
        output["asof_pitcher_success_rate"]
        + output["asof_pitcher_reverse_rate"]
        + output["asof_pitcher_middle_rate"]
        - 1.0
    ).astype("float32")
    return output


def make_group_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(index=frame.index, dtype="object")
    if len(columns) == 1:
        return frame[columns[0]].fillna("__MISSING__").astype(str)
    values = frame[columns].fillna("__MISSING__").astype(str)
    return values.agg("\x1f".join, axis=1)


def global_priors(source: pd.DataFrame, target_season: int) -> dict:
    if source.empty:
        return {"last": 0.5, "ewm": 0.5, "trend": 0.5}
    season_rate = source.groupby("season")[TARGET].mean().sort_index()
    last = float(season_rate.iloc[-1])
    age = target_season - season_rate.index.to_numpy(float)
    weight = np.power(0.5, age / 1.5)
    ewm = float(np.dot(season_rate.to_numpy(float), weight / weight.sum()))
    if len(season_rate) >= 2:
        coefficients = np.polyfit(
            season_rate.index.to_numpy(float), season_rate.to_numpy(float), deg=1, w=weight
        )
        trend = float(np.polyval(coefficients, target_season))
    else:
        trend = last
    return {
        "last": last,
        "ewm": ewm,
        "trend": float(np.clip(trend, 0.35, 0.65)),
    }


def weighted_group_lookup(
    source: pd.DataFrame,
    columns: list[str],
    target_season: int,
    prior: float,
    smoothing: float,
    half_life: float,
) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(columns=["key", "value", "effective_n"])
    key = make_group_key(source, columns)
    age = target_season - source["season"].to_numpy(float)
    weight = np.power(0.5, age / half_life)
    work = pd.DataFrame(
        {
            "key": key.to_numpy(),
            "weighted_y": weight * source[TARGET].to_numpy(float),
            "weight": weight,
        }
    )
    grouped = work.groupby("key", sort=False).agg(
        weighted_y=("weighted_y", "sum"), effective_n=("weight", "sum")
    )
    grouped["value"] = (
        grouped["weighted_y"] + smoothing * prior
    ) / (grouped["effective_n"] + smoothing)
    return grouped[["value", "effective_n"]].reset_index()


def last_season_lookup(
    source: pd.DataFrame,
    columns: list[str],
    target_season: int,
    prior: float,
    smoothing: float,
) -> pd.DataFrame:
    last = source[source["season"].eq(target_season - 1)]
    if last.empty:
        return pd.DataFrame(columns=["key", "value", "n"])
    work = pd.DataFrame(
        {"key": make_group_key(last, columns).to_numpy(), "target": last[TARGET].to_numpy()}
    )
    grouped = work.groupby("key", sort=False)["target"].agg(["sum", "count"])
    grouped["value"] = (grouped["sum"] + smoothing * prior) / (
        grouped["count"] + smoothing
    )
    return grouped[["value", "count"]].rename(columns={"count": "n"}).reset_index()


def build_target_season_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    target_season: int,
    all_smoothing: float = 80.0,
    last_smoothing: float = 40.0,
    half_life: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    priors = global_priors(history, target_season)
    output = pd.DataFrame(index=target.index)
    output["te_global_last"] = np.float32(priors["last"])
    output["te_global_ewm"] = np.float32(priors["ewm"])
    output["te_global_trend"] = np.float32(priors["trend"])
    serialized = {"target_season": target_season, "priors": priors, "groups": {}}

    for name, columns in GROUP_SPECS.items():
        target_key = make_group_key(target, columns)
        all_lookup = weighted_group_lookup(
            history,
            columns,
            target_season,
            priors["ewm"],
            all_smoothing,
            half_life,
        )
        last_lookup = last_season_lookup(
            history,
            columns,
            target_season,
            priors["last"],
            last_smoothing,
        )
        all_value = all_lookup.set_index("key")["value"] if len(all_lookup) else pd.Series(dtype=float)
        all_count = (
            all_lookup.set_index("key")["effective_n"]
            if len(all_lookup)
            else pd.Series(dtype=float)
        )
        last_value = (
            last_lookup.set_index("key")["value"]
            if len(last_lookup)
            else pd.Series(dtype=float)
        )
        last_count = (
            last_lookup.set_index("key")["n"]
            if len(last_lookup)
            else pd.Series(dtype=float)
        )
        all_feature = target_key.map(all_value).fillna(priors["ewm"]).astype("float32")
        last_feature = target_key.map(last_value).fillna(priors["last"]).astype("float32")
        output[f"te_{name}_all"] = all_feature.to_numpy()
        output[f"te_{name}_last"] = last_feature.to_numpy()
        output[f"te_{name}_delta"] = (last_feature - all_feature).to_numpy("float32")
        output[f"te_{name}_log_all_n"] = np.log1p(
            target_key.map(all_count).fillna(0).to_numpy(float)
        ).astype("float32")
        output[f"te_{name}_log_last_n"] = np.log1p(
            target_key.map(last_count).fillna(0).to_numpy(float)
        ).astype("float32")
        serialized["groups"][name] = {
            "columns": columns,
            "all_value": dict(zip(all_lookup.get("key", []), all_lookup.get("value", []))),
            "all_count": dict(
                zip(all_lookup.get("key", []), all_lookup.get("effective_n", []))
            ),
            "last_value": dict(
                zip(last_lookup.get("key", []), last_lookup.get("value", []))
            ),
            "last_count": dict(zip(last_lookup.get("key", []), last_lookup.get("n", []))),
        }
    return output, serialized


def build_temporal_target_features(
    frame: pd.DataFrame,
    all_smoothing: float = 80.0,
    last_smoothing: float = 40.0,
    half_life: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    if TARGET not in frame:
        raise ValueError(f"{TARGET} is required")
    result = pd.DataFrame(index=frame.index)
    final_lookup = None
    seasons = sorted(frame["season"].unique())
    for season in seasons:
        target_mask = frame["season"].eq(season)
        history = frame[frame["season"].lt(season)]
        part, _ = build_target_season_features(
            history,
            frame.loc[target_mask],
            int(season),
            all_smoothing,
            last_smoothing,
            half_life,
        )
        result.loc[target_mask, part.columns] = part.to_numpy()

    _, final_lookup = build_target_season_features(
        frame,
        frame.iloc[:0].copy(),
        int(max(seasons) + 1),
        all_smoothing,
        last_smoothing,
        half_life,
    )
    result = result.astype("float32")
    if not np.isfinite(result.to_numpy()).all():
        raise AssertionError("Non-finite temporal target feature")
    return result, final_lookup


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    row_features = add_v2_row_features(train)
    temporal, final_lookup = build_temporal_target_features(train)
    new_row_columns = [column for column in row_features if column not in train.columns]
    cache = pd.concat(
        [train[["row_id", "season"]], row_features[new_row_columns], temporal], axis=1
    )
    cache_path = WORK_DIR / "v2_temporal_train.parquet"
    cache.to_parquet(cache_path, index=False)
    lookup_path = WORK_DIR / "v2_temporal_lookup_2025.json"
    lookup_path.write_text(
        json.dumps(final_lookup, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "rows": len(cache),
        "columns": len(cache.columns),
        "row_feature_columns": new_row_columns,
        "temporal_feature_columns": temporal.columns.tolist(),
        "group_specs": GROUP_SPECS,
        "all_smoothing": 80.0,
        "last_smoothing": 40.0,
        "half_life": 2.0,
        "seed": SEED,
    }
    (WORK_DIR / "v2_temporal_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved={cache_path} shape={cache.shape}")
    print(f"saved={lookup_path}")


if __name__ == "__main__":
    main()
