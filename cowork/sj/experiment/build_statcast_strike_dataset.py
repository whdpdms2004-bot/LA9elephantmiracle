from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEQUENCE_COLUMNS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]

RAW_COLUMNS = [
    "game_pk", "game_date", "game_year", "game_type", "home_team", "away_team",
    "at_bat_number", "pitch_number", "pitcher", "player_name", "batter", "fielder_2",
    "type", "description", "pitch_type", "pitch_name", "balls", "strikes",
    "outs_when_up", "inning", "inning_topbot", "on_1b", "on_2b", "on_3b",
    "home_score", "away_score", "bat_score", "fld_score", "stand", "p_throws",
    "sz_top", "sz_bot", "if_fielding_alignment", "of_fielding_alignment",
    "release_speed", "effective_speed", "release_spin_rate", "spin_axis", "pfx_x",
    "pfx_z", "plate_x", "plate_z", "zone", "release_pos_x", "release_pos_y",
    "release_pos_z", "release_extension", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "events", "bb_type", "launch_speed", "launch_angle", "hit_distance_sc",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle", "woba_value",
    "babip_value", "iso_value", "home_win_exp", "bat_win_exp", "delta_home_win_exp",
    "delta_run_exp", "post_home_score", "post_away_score", "post_bat_score",
    "post_fld_score",
]

CURRENT_PITCH_LEAKAGE = {
    "type", "description", "zone", "plate_x", "plate_z", "release_speed",
    "effective_speed", "release_spin_rate", "spin_axis", "pfx_x", "pfx_z", "vx0",
    "vy0", "vz0", "ax", "ay", "az", "release_pos_x", "release_pos_y",
    "release_pos_z", "release_extension", "events", "bb_type", "launch_speed",
    "launch_angle", "hit_distance_sc", "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle", "woba_value", "babip_value", "iso_value",
    "post_home_score", "post_away_score", "post_bat_score", "post_fld_score",
    "delta_home_win_exp", "delta_run_exp", "home_win_exp", "bat_win_exp",
}


@dataclass
class BuildConfig:
    seasons: dict[int, tuple[str, str]] = field(default_factory=lambda: {
        2017: ("2017-04-02", "2017-10-01"),
        2018: ("2018-03-29", "2018-10-01"),
        2019: ("2019-03-20", "2019-09-29"),
    })
    raw_dir: Path = Path("data/statcast_raw_sequence")
    processed_dir: Path = Path("data/processed")
    train_years: tuple[int, ...] = (2017, 2018)
    test_years: tuple[int, ...] = (2019,)
    rolling_window: int = 20
    rolling_min_periods: int = 5
    li_prior_strength: float = 50.0
    include_current_pitch_type: bool = True


def iter_month_ranges(start_date: str, end_date: str) -> Iterable[tuple[str, str]]:
    current, final = pd.Timestamp(start_date), pd.Timestamp(end_date)
    while current <= final:
        chunk_end = min(current + pd.offsets.MonthEnd(0), final)
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + pd.Timedelta(days=1)


def collect_raw(config: BuildConfig, max_retries: int = 3, overwrite: bool = False) -> pd.DataFrame:
    """Download monthly raw Statcast files. Requires pybaseball and network access."""
    try:
        from pybaseball import cache, statcast
    except ImportError as exc:
        raise ImportError("Install dependencies first: pip install pybaseball pyarrow") from exc

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    cache.enable()
    manifest: list[dict] = []

    for season, (season_start, season_end) in config.seasons.items():
        season_dir = config.raw_dir / str(season)
        season_dir.mkdir(parents=True, exist_ok=True)
        for order, (start_date, end_date) in enumerate(iter_month_ranges(season_start, season_end), 1):
            path = season_dir / f"statcast_{start_date}_{end_date}.parquet"
            meta_path = path.with_suffix(".json")
            if overwrite or not (path.exists() and meta_path.exists()):
                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"[download] {start_date} ~ {end_date}")
                        raw = statcast(start_dt=start_date, end_dt=end_date, verbose=True, parallel=True)
                        break
                    except Exception:
                        if attempt == max_retries:
                            raise
                        time.sleep(10 * attempt)
                if raw.empty:
                    print(f"[empty] {start_date} ~ {end_date}")
                    continue
                raw = raw.copy()
                raw.insert(0, "_source_row", np.arange(len(raw), dtype=np.int64))
                sort_cols = [c for c in SEQUENCE_COLUMNS if c in raw]
                raw = raw.sort_values(sort_cols, kind="stable", na_position="last").reset_index(drop=True)
                raw.insert(0, "_sequence_in_chunk", np.arange(len(raw), dtype=np.int64))
                raw.to_parquet(path, index=False, compression="zstd")
                meta = {
                    "start_date": start_date, "end_date": end_date,
                    "row_count": int(len(raw)), "column_count": int(raw.shape[1]),
                    "columns": list(raw.columns), "sort_columns_used": sort_cols,
                    "data_processing": ["no filtering", "no deduplication", "stable sorting only"],
                }
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                manifest.append({
                    "season": season, "chunk_order": order,
                    "start_date": meta["start_date"], "end_date": meta["end_date"],
                    "row_count": meta["row_count"], "column_count": meta["column_count"],
                    "file_path": str(path),
                })

    manifest_df = pd.DataFrame(manifest).sort_values(["season", "chunk_order"])
    manifest_df.to_csv(config.raw_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    return manifest_df


def load_raw(config: BuildConfig, columns: list[str] | None = None) -> pd.DataFrame:
    """Load raw chunks while tolerating columns absent from individual seasons."""
    import pyarrow.parquet as pq

    paths = sorted(config.raw_dir.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found below {config.raw_dir.resolve()}")
    requested = columns or RAW_COLUMNS
    frames = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema.names)
        use = [c for c in requested if c in available]
        frames.append(pd.read_parquet(path, columns=use))
    return pd.concat(frames, ignore_index=True, sort=False)


def audit_raw(df: pd.DataFrame) -> pd.DataFrame:
    key = [c for c in SEQUENCE_COLUMNS if c in df]
    rows = [
        ("rows", len(df)), ("columns", df.shape[1]),
        ("games", df["game_pk"].nunique() if "game_pk" in df else np.nan),
        ("duplicate_sequence_keys", df.duplicated(key).sum() if key else np.nan),
        ("missing_type", df["type"].isna().sum() if "type" in df else np.nan),
        ("missing_pitch_type", df["pitch_type"].isna().sum() if "pitch_type" in df else np.nan),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


def integrity_report(
    df: pd.DataFrame,
    config: BuildConfig,
    *,
    model_dataset: bool = True,
) -> pd.DataFrame:
    """Return machine-readable PASS/WARN/FAIL checks for raw or model data."""
    results: list[dict[str, object]] = []

    def add(check: str, status: str, value: object, detail: str) -> None:
        results.append({"check": check, "status": status, "value": value, "detail": detail})

    add("non_empty", "PASS" if len(df) else "FAIL", len(df), "행 수가 1개 이상이어야 합니다.")

    required = {"game_date", "game_pk", "at_bat_number", "pitch_number"}
    if model_dataset:
        required |= {"game_year", "game_type", "is_strike", "split"}
    missing = sorted(required - set(df.columns))
    add("required_columns", "PASS" if not missing else "FAIL", len(missing),
        "누락: " + ", ".join(missing) if missing else "필수 열이 모두 존재합니다.")

    sequence_key = [c for c in SEQUENCE_COLUMNS if c in df.columns]
    if len(sequence_key) == len(SEQUENCE_COLUMNS):
        null_keys = int(df[sequence_key].isna().any(axis=1).sum())
        duplicates = int(df.duplicated(sequence_key, keep=False).sum())
        add("null_sequence_key", "PASS" if null_keys == 0 else "FAIL", null_keys,
            "투구 식별 키에 결측값이 없어야 합니다.")
        add("duplicate_sequence_key", "PASS" if duplicates == 0 else "FAIL", duplicates,
            "중복에 참여한 전체 행 수입니다.")

    if "game_date" in df:
        parsed = pd.to_datetime(df["game_date"], errors="coerce")
        bad_dates = int(parsed.isna().sum())
        add("valid_game_date", "PASS" if bad_dates == 0 else "FAIL", bad_dates,
            "파싱할 수 없는 날짜 수입니다.")

    if "game_type" in df:
        non_regular = int((~df["game_type"].eq("R")).sum())
        expected = model_dataset
        status = "PASS" if (not expected or non_regular == 0) else "FAIL"
        add("regular_season_only", status, non_regular,
            "모델 데이터에는 game_type='R'만 허용합니다.")

    for col, lower, upper in [
        ("balls", 0, 3), ("strikes", 0, 2), ("outs_when_up", 0, 2),
        ("base_state", 0, 7), ("runner_count", 0, 3),
    ]:
        if col in df:
            numeric = pd.to_numeric(df[col], errors="coerce")
            invalid = int((numeric.notna() & ~numeric.between(lower, upper)).sum())
            add(f"range_{col}", "PASS" if invalid == 0 else "FAIL", invalid,
                f"허용 범위는 {lower}~{upper}입니다.")

    if model_dataset and "is_strike" in df:
        invalid_target = int((df["is_strike"].isna() | ~df["is_strike"].isin([0, 1])).sum())
        add("binary_target", "PASS" if invalid_target == 0 else "FAIL", invalid_target,
            "is_strike는 결측 없이 0 또는 1이어야 합니다.")
        rate = pd.to_numeric(df["is_strike"], errors="coerce").mean()
        status = "PASS" if pd.notna(rate) and 0.05 < rate < 0.95 else "WARN"
        add("target_rate", status, round(float(rate), 6) if pd.notna(rate) else np.nan,
            "극단적인 타깃 비율은 수집/라벨 오류 가능성을 점검해야 합니다.")

    if model_dataset:
        leaked = sorted(CURRENT_PITCH_LEAKAGE.intersection(df.columns))
        add("current_pitch_leakage", "PASS" if not leaked else "FAIL", len(leaked),
            "잔존 누수 열: " + ", ".join(leaked) if leaked else "현재 투구 누수 열이 없습니다.")

        if {"game_year", "split"}.issubset(df.columns):
            expected_split = np.select(
                [df["game_year"].isin(config.train_years), df["game_year"].isin(config.test_years)],
                ["train", "test"], default="unused",
            )
            split_errors = int((df["split"].astype(str).to_numpy() != expected_split).sum())
            add("split_matches_year", "PASS" if split_errors == 0 else "FAIL", split_errors,
                "연도별 train/test 지정과 불일치한 행 수입니다.")

        if {"game_date", "split"}.issubset(df.columns):
            dates = pd.to_datetime(df["game_date"], errors="coerce")
            train_dates = dates[df["split"].eq("train")]
            test_dates = dates[df["split"].eq("test")]
            if len(train_dates) and len(test_dates):
                chronological = bool(train_dates.max() < test_dates.min())
                add("chronological_split", "PASS" if chronological else "FAIL",
                    f"{train_dates.max().date()} < {test_dates.min().date()}",
                    "마지막 학습일은 첫 테스트일보다 빨라야 합니다.")
            else:
                add("chronological_split", "WARN", "not_evaluable",
                    "빠른 테스트 등 한쪽 분할이 비어 있어 평가하지 못했습니다.")

    report = pd.DataFrame(results)
    order = pd.Categorical(report["status"], ["FAIL", "WARN", "PASS"], ordered=True)
    return report.assign(_order=order).sort_values(["_order", "check"]).drop(columns="_order").reset_index(drop=True)


def assert_integrity(report: pd.DataFrame) -> None:
    """Raise with all failed checks, while allowing explicit WARN results."""
    failed = report[report["status"].eq("FAIL")]
    if not failed.empty:
        details = "; ".join(f"{row.check}={row.value}" for row in failed.itertuples())
        raise AssertionError(f"Dataset integrity checks failed: {details}")


def _rolling_prior(series: pd.Series, window: int, min_periods: int, stat: str = "mean") -> pd.Series:
    rolled = series.shift(1).rolling(window, min_periods=min_periods)
    return getattr(rolled, stat)()


def _add_trajectory_angles(df: pd.DataFrame) -> None:
    """Approximate plate-crossing VAA/HAA, retained only through lagged features."""
    needed = {"vx0", "vy0", "vz0", "ax", "ay", "az"}
    if not needed.issubset(df.columns):
        df["_vaa_current"] = np.nan
        df["_haa_current"] = np.nan
        return
    y0, y_plate = 50.0, 17.0 / 12.0
    a, b, c = 0.5 * df["ay"].astype(float), df["vy0"].astype(float), y0 - y_plate
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        t_quad = (-b - np.sqrt(disc.clip(lower=0))) / (2 * a)
        t_linear = -c / b
    t = t_quad.where(a.abs() > 1e-9, t_linear).where(lambda x: x.gt(0))
    vx = df["vx0"] + df["ax"] * t
    vy = df["vy0"] + df["ay"] * t
    vz = df["vz0"] + df["az"] * t
    df["_vaa_current"] = np.degrees(np.arctan2(vz, vy.abs()))
    df["_haa_current"] = np.degrees(np.arctan2(vx, vy.abs()))


def _add_leak_safe_li(df: pd.DataFrame, config: BuildConfig) -> None:
    """Prequential LI for train rows and a frozen train-only lookup for test rows."""
    state = ["inning", "inning_topbot", "outs_when_up", "base_state", "score_diff_home"]
    valid_state = [c for c in state if c in df]
    delta = pd.to_numeric(df.get("delta_home_win_exp"), errors="coerce").abs()
    train = df["game_year"].isin(config.train_years)
    train_delta = delta.where(train)
    global_train = train_delta.mean()
    if not np.isfinite(global_train) or global_train == 0:
        df["li_pa"] = 1.0
        df["li_pa_shrunk"] = 1.0
        df["leverage_class"] = "medium"
        return

    train_part = df.loc[train, valid_state].copy()
    train_part["_d"] = train_delta.loc[train].to_numpy()
    group = train_part.groupby(valid_state, dropna=False, sort=False)["_d"]
    prior_sum = group.cumsum() - train_part["_d"].fillna(0)
    observed = train_part["_d"].notna().astype("int32")
    prior_n = (
        observed.groupby([train_part[c] for c in valid_state], dropna=False, sort=False).cumsum()
        - observed
    )
    global_prior = train_part["_d"].expanding().mean().shift(1).fillna(global_train)
    shrunk = (prior_sum.fillna(0) + config.li_prior_strength * global_prior) / (
        prior_n + config.li_prior_strength
    )
    df["li_pa_shrunk"] = np.nan
    df.loc[train, "li_pa_shrunk"] = (shrunk / global_prior).to_numpy()

    lookup = train_part.groupby(valid_state, dropna=False)["_d"].agg(["sum", "count"])
    lookup["li"] = ((lookup["sum"] + config.li_prior_strength * global_train) /
                    (lookup["count"] + config.li_prior_strength) / global_train)
    test_index = df.index[~train]
    if len(test_index):
        test_keys = pd.MultiIndex.from_frame(df.loc[test_index, valid_state])
        df.loc[test_index, "li_pa_shrunk"] = lookup["li"].reindex(test_keys).fillna(1.0).to_numpy()
    df["li_pa_shrunk"] = df["li_pa_shrunk"].fillna(1.0).clip(0, 10).astype("float32")
    df["li_pa"] = df["li_pa_shrunk"]
    df["leverage_class"] = pd.cut(
        df["li_pa_shrunk"], [-np.inf, 0.85, 2.0, np.inf], labels=["low", "medium", "high"]
    )


def build_features(raw: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """Create one row per regular-season pitch with leak-safe pre-pitch features."""
    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_type"].eq("R") & df["type"].notna()].copy()
    df = df.sort_values(SEQUENCE_COLUMNS, kind="stable", na_position="last")
    df = df.drop_duplicates(SEQUENCE_COLUMNS, keep="last").reset_index(drop=True)

    for c in ["bat_score", "fld_score", "home_score", "away_score", "balls", "strikes", "inning"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_strike"] = df["type"].eq("S").astype("int8")
    df["score_diff_bat"] = df["bat_score"] - df["fld_score"]
    df["score_diff_home"] = df["home_score"] - df["away_score"]
    i1, i2, i3 = (df[c].notna().astype("int8") for c in ["on_1b", "on_2b", "on_3b"])
    df["base_state"] = i1 + 2 * i2 + 4 * i3
    df["runner_count"] = i1 + i2 + i3
    df["risp"] = ((i2 == 1) | (i3 == 1)).astype("int8")
    df["bases_loaded"] = ((i1 == 1) & (i2 == 1) & (i3 == 1)).astype("int8")
    df["count_state"] = df["balls"].astype("Int64").astype(str) + "-" + df["strikes"].astype("Int64").astype(str)
    df["two_strike"] = df["strikes"].eq(2).astype("int8")
    df["full_count"] = (df["balls"].eq(3) & df["strikes"].eq(2)).astype("int8")
    df["pitcher_ahead"] = ((df["strikes"] > df["balls"]) & ~df["full_count"].astype(bool)).astype("int8")
    df["batter_ahead"] = (df["balls"] > df["strikes"]).astype("int8")
    df["matchup"] = df["p_throws"].astype("string") + "_" + df["stand"].astype("string")
    df["same_handed_matchup"] = df["p_throws"].eq(df["stand"]).astype("int8")
    df["late_inning"] = df["inning"].ge(7).astype("int8")
    df["extra_inning"] = df["inning"].ge(10).astype("int8")
    df["close_game"] = df["score_diff_bat"].abs().le(2).astype("int8")

    # Within-game history.
    game_pitcher = df.groupby(["game_pk", "pitcher"], sort=False, group_keys=False)
    df["pitcher_pitch_count_before"] = game_pitcher.cumcount().astype("int32")
    for lag in (1, 2, 3):
        df[f"prev_pitch_type_{lag}"] = game_pitcher["pitch_type"].shift(lag)
    df["prev_description"] = game_pitcher["description"].shift(1)

    _add_trajectory_angles(df)
    lag_cols = [
        "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x", "plate_z",
        "release_pos_x", "release_pos_z", "_vaa_current", "_haa_current",
    ]
    for col in lag_cols:
        if col in df:
            name = {"_vaa_current": "prev_vaa_approx", "_haa_current": "prev_haa_approx"}.get(col, f"prev_{col}")
            df[name] = game_pitcher[col].shift(1)

    # Number of earlier PA encounters for this pitcher-batter pair in the game.
    pa_first = ~df.duplicated(["game_pk", "pitcher", "batter", "at_bat_number"])
    encounter = pa_first.astype("int16").groupby([df["game_pk"], df["pitcher"], df["batter"]]).cumsum() - 1
    df["times_faced_in_game"] = encounter.groupby(
        [df["game_pk"], df["pitcher"], df["batter"], df["at_bat_number"]]
    ).transform("max").astype("int16")

    # Cross-game pitcher history; every rolling window starts with shift(1).
    pitcher_group = df.groupby("pitcher", sort=False, group_keys=False)
    w, mp = config.rolling_window, config.rolling_min_periods
    for col in ["release_speed", "release_spin_rate", "pfx_x", "pfx_z", "release_pos_x", "release_pos_z"]:
        if col in df:
            df[f"{col}_last{w}_mean"] = pitcher_group[col].transform(
                lambda s: _rolling_prior(s, w, mp)
            ).astype("float32")
    for code in ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "KC", "ST"]:
        indicator = df["pitch_type"].eq(code).astype("float32")
        df[f"{code.lower()}_usage_last_{w}"] = indicator.groupby(df["pitcher"], sort=False).transform(
            lambda s: _rolling_prior(s, w, mp)
        ).astype("float32")
    df["pitcher_strike_rate_before"] = df["is_strike"].groupby(df["pitcher"], sort=False).transform(
        lambda s: s.shift(1).expanding(min_periods=10).mean()
    ).astype("float32")
    df[f"pitcher_strike_rate_last{w}"] = df["is_strike"].groupby(df["pitcher"], sort=False).transform(
        lambda s: _rolling_prior(s, w, mp)
    ).astype("float32")
    sx = pitcher_group["release_pos_x"].transform(lambda s: _rolling_prior(s, w, mp, "std"))
    sz = pitcher_group["release_pos_z"].transform(lambda s: _rolling_prior(s, w, mp, "std"))
    df[f"release_dispersion_last{w}"] = np.sqrt(sx * sx + sz * sz).astype("float32")

    # Previous game workload and rest: summarize games first, then shift by pitcher.
    games = (df.groupby(["pitcher", "game_pk"], sort=False)
               .agg(game_date=("game_date", "min"), game_pitch_count=("pitch_number", "size"))
               .reset_index().sort_values(["pitcher", "game_date", "game_pk"]))
    games["prev_game_pitch_count"] = games.groupby("pitcher")["game_pitch_count"].shift(1)
    games["rest_days"] = games.groupby("pitcher")["game_date"].diff().dt.days - 1
    df = df.merge(games[["pitcher", "game_pk", "prev_game_pitch_count", "rest_days"]],
                  on=["pitcher", "game_pk"], how="left", validate="many_to_one")

    _add_leak_safe_li(df, config)
    df["split"] = np.select(
        [df["game_year"].isin(config.train_years), df["game_year"].isin(config.test_years)],
        ["train", "test"], default="unused",
    )
    return df


def make_model_dataset(enriched: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """Drop current-pitch outcomes/trajectory values while retaining identifiers and target."""
    drop = CURRENT_PITCH_LEAKAGE.intersection(enriched.columns)
    if not config.include_current_pitch_type:
        drop = drop | {"pitch_type", "pitch_name"}
    model = enriched.drop(columns=sorted(drop), errors="ignore").copy()
    model = model.drop(columns=["_vaa_current", "_haa_current"], errors="ignore")
    assert "is_strike" in model and not (CURRENT_PITCH_LEAKAGE - {"type"}).intersection(model.columns)
    return model


def save_dataset(model: pd.DataFrame, config: BuildConfig) -> dict[str, Path]:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all": config.processed_dir / "strike_prediction_dataset.parquet",
        "train": config.processed_dir / "strike_prediction_train_2017_2018.parquet",
        "test": config.processed_dir / "strike_prediction_test_2019.parquet",
        "schema": config.processed_dir / "strike_prediction_schema.json",
        "integrity": config.processed_dir / "strike_prediction_integrity_report.csv",
    }
    model.to_parquet(outputs["all"], index=False, compression="zstd")
    model[model["split"].eq("train")].to_parquet(outputs["train"], index=False, compression="zstd")
    model[model["split"].eq("test")].to_parquet(outputs["test"], index=False, compression="zstd")
    schema = {
        "target": "is_strike", "target_definition": "1 when Statcast type == 'S', else 0",
        "prediction_time": "immediately before the current pitch",
        "train_years": list(config.train_years), "test_years": list(config.test_years),
        "include_current_pitch_type": config.include_current_pitch_type,
        "row_count": int(len(model)), "column_count": int(model.shape[1]),
        "columns": {c: str(t) for c, t in model.dtypes.items()},
        "leakage_rule": "Current pitch result, location, trajectory, physical log, WPA and batted-ball fields removed",
    }
    outputs["schema"].write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    integrity_report(model, config, model_dataset=True).to_csv(
        outputs["integrity"], index=False, encoding="utf-8-sig"
    )
    return outputs


def run_pipeline(config: BuildConfig, download: bool = False) -> tuple[pd.DataFrame, dict[str, Path]]:
    if download:
        collect_raw(config)
    raw = load_raw(config)
    print(audit_raw(raw).to_string(index=False))
    enriched = build_features(raw, config)
    model = make_model_dataset(enriched, config)
    report = integrity_report(model, config, model_dataset=True)
    print(report.to_string(index=False))
    assert_integrity(report)
    outputs = save_dataset(model, config)
    return model, outputs


if __name__ == "__main__":
    model_df, saved = run_pipeline(BuildConfig(), download=False)
    print(model_df.groupby("split")["is_strike"].agg(["size", "mean"]))
    print({k: str(v.resolve()) for k, v in saved.items()})
