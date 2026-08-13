from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "missed_bunt", "foul_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CALLED_PITCH_DESCRIPTIONS = {"called_strike", "ball", "blocked_ball", "pitchout"}


def prepare_pitcher_eda(raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare regular-season pitch rows for descriptive, not predictive, analysis."""
    required = {"game_pk", "game_date", "at_bat_number", "pitch_number", "pitcher", "type"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError(f"Missing EDA columns: {missing}")

    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if "game_type" in df:
        df = df[df["game_type"].eq("R")]
    df = df[df["type"].notna() & df["pitcher"].notna()].copy()
    key = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    df = (df.sort_values(key, kind="stable", na_position="last")
            .drop_duplicates(key, keep="last").reset_index(drop=True))

    description = df.get("description", pd.Series(index=df.index, dtype="string")).astype("string")
    df["is_strike"] = df["type"].eq("S").astype("int8")
    df["is_swing"] = description.isin(SWING_DESCRIPTIONS).astype("int8")
    df["is_whiff"] = description.isin(WHIFF_DESCRIPTIONS).astype("int8")
    df["is_called_pitch"] = description.isin(CALLED_PITCH_DESCRIPTIONS).astype("int8")
    df["is_called_strike"] = description.eq("called_strike").astype("int8")
    df["pfx_x_in"] = pd.to_numeric(df.get("pfx_x"), errors="coerce") * 12
    df["pfx_z_in"] = pd.to_numeric(df.get("pfx_z"), errors="coerce") * 12
    df["month"] = df["game_date"].dt.to_period("M").astype("string")
    return df


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def pitcher_overview(df: pd.DataFrame, min_pitches: int = 100) -> pd.DataFrame:
    named = "player_name" in df and df["player_name"].notna().any()
    agg = {
        "pitches": ("pitch_number", "size"),
        "games": ("game_pk", "nunique"),
        "strike_rate": ("is_strike", "mean"),
        "swings": ("is_swing", "sum"),
        "whiffs": ("is_whiff", "sum"),
        "called_pitches": ("is_called_pitch", "sum"),
        "called_strikes": ("is_called_strike", "sum"),
        "avg_velocity": ("release_speed", "mean"),
        "avg_spin": ("release_spin_rate", "mean"),
        "first_date": ("game_date", "min"),
        "last_date": ("game_date", "max"),
    }
    if named:
        agg["pitcher_name"] = ("player_name", "first")
    out = df.groupby("pitcher", dropna=False).agg(**agg).reset_index()
    out["whiff_rate"] = _safe_ratio(out["whiffs"], out["swings"])
    out["called_strike_rate"] = _safe_ratio(out["called_strikes"], out["called_pitches"])
    if not named:
        out["pitcher_name"] = out["pitcher"].astype("Int64").astype(str)
    return out[out["pitches"].ge(min_pitches)].sort_values("pitches", ascending=False).reset_index(drop=True)


def pitch_type_profile(pitcher_df: pd.DataFrame, min_pitches: int = 5) -> pd.DataFrame:
    out = (pitcher_df.groupby(["pitch_type", "pitch_name"], dropna=False)
           .agg(pitches=("pitch_number", "size"), strikes=("is_strike", "sum"),
                swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"),
                avg_velocity=("release_speed", "mean"), avg_spin=("release_spin_rate", "mean"),
                pfx_x_in=("pfx_x_in", "mean"), pfx_z_in=("pfx_z_in", "mean"),
                release_x=("release_pos_x", "mean"), release_z=("release_pos_z", "mean"))
           .reset_index())
    out = out[out["pitches"].ge(min_pitches)].copy()
    out["usage_rate"] = out["pitches"] / len(pitcher_df)
    out["strike_rate"] = _safe_ratio(out["strikes"], out["pitches"])
    out["whiff_rate"] = _safe_ratio(out["whiffs"], out["swings"])
    return out.sort_values("pitches", ascending=False).reset_index(drop=True)


def count_profile(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    return (pitcher_df.groupby(["balls", "strikes"], dropna=False)
            .agg(pitches=("pitch_number", "size"), strike_rate=("is_strike", "mean"),
                 swing_rate=("is_swing", "mean"))
            .reset_index())


def handedness_profile(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    out = (pitcher_df.groupby("stand", dropna=False)
           .agg(pitches=("pitch_number", "size"), strike_rate=("is_strike", "mean"),
                swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"),
                avg_velocity=("release_speed", "mean"))
           .reset_index())
    out["whiff_rate"] = _safe_ratio(out["whiffs"], out["swings"])
    return out


def monthly_profile(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    out = (pitcher_df.groupby("month", dropna=False)
           .agg(pitches=("pitch_number", "size"), strike_rate=("is_strike", "mean"),
                swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"),
                avg_velocity=("release_speed", "mean"), avg_spin=("release_spin_rate", "mean"))
           .reset_index())
    out["whiff_rate"] = _safe_ratio(out["whiffs"], out["swings"])
    return out.sort_values("month")


def select_pitcher(df: pd.DataFrame, pitcher_id: int | None = None) -> tuple[int, str, pd.DataFrame]:
    overview = pitcher_overview(df, min_pitches=1)
    if overview.empty:
        raise ValueError("No pitcher rows are available.")
    selected = int(overview.iloc[0]["pitcher"]) if pitcher_id is None else int(pitcher_id)
    pitcher_df = df[df["pitcher"].eq(selected)].copy()
    if pitcher_df.empty:
        raise KeyError(f"Pitcher ID {selected} was not found.")
    name_values = pitcher_df.get("player_name", pd.Series(dtype="string")).dropna()
    name = str(name_values.iloc[0]) if len(name_values) else str(selected)
    return selected, name, pitcher_df
