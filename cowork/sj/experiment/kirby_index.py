"""Kirby Index: release-angle 기반 커맨드 정량화.

FanGraphs "Introducing the Kirby Index" (Michael Rosen, 2024-05-03) 재현/적용 모듈.
https://blogs.fangraphs.com/introducing-the-kirby-index-a-new-way-to-quantify-command/

핵심 아이디어
1. Statcast 9-파라미터 운동학(vx0, vy0, vz0, ax, ay, az는 y=50ft 평면 기준)을
   릴리스 지점까지 역전파하여 수직/수평 릴리스 각도(VRA/HRA)를 계산한다.
2. 포심 기준 {VRA, HRA, release_pos_x, release_pos_z} 4개 변수만으로
   플레이트 위치의 대부분이 설명된다(원문: 수직 R^2≈0.92, 수평 R^2≈0.85).
3. 투수별로 4개 변수의 표준편차를 구하고, 낮을수록 좋게 백분위화한 뒤
   위치 모델의 변수 중요도로 가중평균 → Kirby Index (0~1, 높을수록 커맨드 반복성 우수).

주의: 이 지표의 원자료(각도)는 투구가 릴리스된 후에야 알 수 있으므로,
스트라이크 예측 모델에는 add_command_features()의 과거-전용(rolling+shift) 피처만 사용한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

Y_MEASUREMENT_PLANE = 50.0          # Statcast 운동학 파라미터 기준 평면 (ft)
Y_RELEASE_FALLBACK = 54.5           # release_pos_y/extension 둘 다 없을 때
MOUND_TO_PLATE = 60.5

ANGLE_INPUTS = ["vra_deg", "hra_deg", "release_pos_x", "release_pos_z"]
KINEMATIC_COLUMNS = ["vx0", "vy0", "vz0", "ax", "ay", "az"]


# ---------------------------------------------------------------------------
# 1. 릴리스 각도 계산
# ---------------------------------------------------------------------------

def _time_shift_to_plane(vy0: pd.Series, ay: pd.Series, dy: pd.Series) -> pd.Series:
    """y=50 평면에서 y=50+dy 평면까지의 시간 이동량 t를 반환(릴리스 방향은 t<0).

    0.5*ay*t^2 + vy0*t - dy = 0 의 물리적 근을 취한다.
    """
    a = 0.5 * ay.astype(float)
    b = vy0.astype(float)
    c = -dy.astype(float)
    disc = (b * b - 4 * a * c).clip(lower=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        quad = (-b - np.sqrt(disc)) / (2 * a)
        linear = -c / b
    t = quad.where(a.abs() > 1e-9, linear)
    return t


def add_release_angles(df: pd.DataFrame, *, degrees: bool = True) -> pd.DataFrame:
    """vra_deg, hra_deg 열 추가. 부호 규약은 VAA/HAA와 동일(-atan(v/vy))."""
    out = df.copy()
    missing = [c for c in KINEMATIC_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"Missing kinematic columns: {missing}")

    if "release_pos_y" in out.columns:
        y_release = pd.to_numeric(out["release_pos_y"], errors="coerce")
    else:
        y_release = pd.Series(np.nan, index=out.index, dtype=float)
    if "release_extension" in out.columns:
        ext_based = MOUND_TO_PLATE - pd.to_numeric(out["release_extension"], errors="coerce")
        y_release = y_release.fillna(ext_based)
    y_release = y_release.fillna(Y_RELEASE_FALLBACK)

    vy0 = pd.to_numeric(out["vy0"], errors="coerce")
    dy = y_release - Y_MEASUREMENT_PLANE
    t_rel = _time_shift_to_plane(vy0, pd.to_numeric(out["ay"], errors="coerce"), dy)

    vx_r = pd.to_numeric(out["vx0"], errors="coerce") + pd.to_numeric(out["ax"], errors="coerce") * t_rel
    vy_r = vy0 + pd.to_numeric(out["ay"], errors="coerce") * t_rel
    vz_r = pd.to_numeric(out["vz0"], errors="coerce") + pd.to_numeric(out["az"], errors="coerce") * t_rel

    # vy_r < 0 (홈플레이트 방향) 전제. atan2(v, -vy)는 -atan(v/vy)와 동일.
    # 상승/우측 = 양수, 하강/좌측 = 음수 (VAA/HAA 부호 규약과 일치)
    vra = np.arctan2(vz_r, -vy_r)
    hra = np.arctan2(vx_r, -vy_r)
    if degrees:
        vra = np.degrees(vra)
        hra = np.degrees(hra)
    out["vra_deg"] = vra.astype("float32")
    out["hra_deg"] = hra.astype("float32")
    return out


# ---------------------------------------------------------------------------
# 2. 위치 모델 검증 (원문 R^2 재현)
# ---------------------------------------------------------------------------

def location_model_report(
    df: pd.DataFrame,
    *,
    pitch_type: str | None = "FF",
    n_estimators: int = 200,
    max_samples: int = 150_000,
    random_state: int = 0,
) -> dict:
    """{VRA,HRA,릴리스 좌표} → plate_x/plate_z 예측 R^2와 변수 중요도 반환."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    data = df
    if pitch_type is not None and "pitch_type" in df.columns:
        data = df[df["pitch_type"].eq(pitch_type)]
    cols = ANGLE_INPUTS + ["plate_x", "plate_z"]
    data = data[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) > max_samples:
        data = data.sample(max_samples, random_state=random_state)
    if len(data) < 500:
        raise ValueError(f"Not enough rows for validation: {len(data)}")

    X = data[ANGLE_INPUTS].to_numpy()
    result: dict = {"n_pitches": int(len(data)), "pitch_type": pitch_type}
    importances = {}
    for target in ["plate_z", "plate_x"]:
        y = data[target].to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=random_state)
        model = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=-1, random_state=random_state,
            min_samples_leaf=5,
        )
        model.fit(X_tr, y_tr)
        result[f"r2_{target}"] = float(r2_score(y_te, model.predict(X_te)))
        importances[target] = model.feature_importances_
    weight = (importances["plate_z"] + importances["plate_x"]) / 2.0
    result["weights"] = {c: float(w) for c, w in zip(ANGLE_INPUTS, weight / weight.sum())}
    return result


# ---------------------------------------------------------------------------
# 3. Kirby Index 산출
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {  # 위치 모델 중요도가 없을 때의 근사 가중치
    "vra_deg": 0.4, "hra_deg": 0.4, "release_pos_x": 0.1, "release_pos_z": 0.1,
}


def kirby_index_table(
    df: pd.DataFrame,
    *,
    pitch_type: str | None = "FF",
    group_cols: tuple[str, ...] = ("game_year", "pitcher"),
    min_pitches: int = 125,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """그룹(기본: 시즌×투수)별 SD → 백분위 → 가중평균 Kirby Index."""
    data = df
    if pitch_type is not None and "pitch_type" in df.columns:
        data = df[df["pitch_type"].eq(pitch_type)]
    needed = list(group_cols) + ANGLE_INPUTS
    data = data[needed].dropna()

    grouped = data.groupby(list(group_cols))
    table = grouped[ANGLE_INPUTS].std().add_suffix("_sd")
    table["n_pitches"] = grouped.size()
    table = table[table["n_pitches"] >= min_pitches].reset_index()
    if table.empty:
        return table

    w = weights or DEFAULT_WEIGHTS
    total = sum(w[c] for c in ANGLE_INPUTS)
    index = np.zeros(len(table))
    year_col = group_cols[0] if group_cols[0] in table.columns else None
    for col in ANGLE_INPUTS:
        sd = table[f"{col}_sd"]
        # 시즌 내부에서 백분위화(낮은 SD = 높은 점수)
        if year_col is not None:
            pct = sd.groupby(table[year_col]).rank(pct=True, ascending=False)
        else:
            pct = sd.rank(pct=True, ascending=False)
        table[f"{col}_pctile"] = pct
        index += (w[col] / total) * pct.to_numpy()
    table["kirby_index"] = index
    if "player_name" in df.columns and "pitcher" in table.columns:
        names = df.dropna(subset=["player_name"]).groupby("pitcher")["player_name"].last()
        table.insert(1, "player_name", table["pitcher"].map(names))
    return table.sort_values("kirby_index", ascending=False).reset_index(drop=True)


def year_to_year_stickiness(table: pd.DataFrame, *, year_col: str = "game_year") -> pd.DataFrame:
    """인접 시즌 간 Kirby Index 상관(원문: R^2≈0.5 수준이면 성공적 재현)."""
    rows = []
    years = sorted(table[year_col].unique())
    for y1, y2 in zip(years[:-1], years[1:]):
        a = table[table[year_col].eq(y1)].set_index("pitcher")["kirby_index"]
        b = table[table[year_col].eq(y2)].set_index("pitcher")["kirby_index"]
        common = a.index.intersection(b.index)
        if len(common) >= 10:
            r = float(np.corrcoef(a.loc[common], b.loc[common])[0, 1])
            rows.append({"year_from": y1, "year_to": y2, "n_pitchers": len(common),
                         "pearson_r": r, "r_squared": r * r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. 스트라이크 예측 모델용 누수 방지 피처
# ---------------------------------------------------------------------------

def add_command_features(
    df: pd.DataFrame,
    *,
    window: int = 40,
    min_periods: int = 10,
    sort: bool = True,
) -> pd.DataFrame:
    """과거 투구만 사용하는 rolling 릴리스 각도 SD 피처를 추가.

    현재 투구의 각도는 shift(1)로 항상 제외되므로 스트라이크 예측 입력으로 안전하다.
    그룹은 (pitcher, pitch_type): 커맨드 반복성은 구종 단위 개념이기 때문.
    """
    out = add_release_angles(df) if "vra_deg" not in df.columns else df.copy()
    if sort:
        key = [c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if c in out.columns]
        out = out.sort_values(key, kind="stable", na_position="last")

    group = out.groupby(["pitcher", "pitch_type"], sort=False, group_keys=False)
    for col, name in [("vra_deg", "vra"), ("hra_deg", "hra")]:
        out[f"cmd_{name}_sd_last{window}"] = group[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).std()
        ).astype("float32")
    out[f"cmd_angle_dispersion_last{window}"] = np.sqrt(
        out[f"cmd_vra_sd_last{window}"] ** 2 + out[f"cmd_hra_sd_last{window}"] ** 2
    ).astype("float32")
    # 모델 데이터셋에 합류시킬 때 현재 투구 각도 원본은 반드시 제거한다.
    return out


COMMAND_FEATURE_LEAKAGE = {"vra_deg", "hra_deg"}


def add_prepitch_ff_command_features(
    df: pd.DataFrame,
    *,
    window: int = 40,
    min_periods: int = 10,
    sort: bool = True,
) -> pd.DataFrame:
    """엄격한 투구 전 시점용 커맨드 피처.

    add_command_features와 달리 (pitcher, pitch_type) 그룹을 쓰지 않는다.
    그룹 기준에 현재 구종이 들어가면 '현재 투구가 무슨 구종인지'가 피처에
    간접 유입되므로, 구종 결정 전 예측에서는 누수다.

    대신 '투수의 최근 포심 릴리스 각도 반복성'을 상태 변수로 계산해
    이후 모든 투구(구종 무관)에 브로드캐스트한다. 각 FF 투구에서 shift(1)
    적용 후 계산하므로 항상 과거 투구만 사용한다.
    """
    out = add_release_angles(df) if "vra_deg" not in df.columns else df.copy()
    if sort:
        key = [c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number"] if c in out.columns]
        out = out.sort_values(key, kind="stable", na_position="last")

    is_ff = out["pitch_type"].eq("FF")
    made = []
    for col, name in [("vra_deg", "vra"), ("hra_deg", "hra")]:
        ff = out.loc[is_ff, ["pitcher", col]]
        rolled = ff.groupby("pitcher", sort=False)[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).std()
        )
        tmp = pd.Series(np.nan, index=out.index, dtype="float64")
        tmp.loc[rolled.index] = rolled
        feature = f"cmd_ff_{name}_sd_last{window}"
        out[feature] = tmp.groupby(out["pitcher"]).ffill().astype("float32")
        made.append(feature)
    out[f"cmd_ff_angle_dispersion_last{window}"] = np.sqrt(
        out[made[0]] ** 2 + out[made[1]] ** 2
    ).astype("float32")
    return out


# ---------------------------------------------------------------------------
# 5. 실데이터 실행 진입점
# ---------------------------------------------------------------------------

def run_on_raw(raw_dir: Path = Path("data/statcast_raw_sequence"),
               report_dir: Path = Path("reports/kirby_index")) -> None:
    """원본 parquet 존재 시: 검증 R^2 → 시즌별 Kirby Index → 스티키니스 저장."""
    from build_statcast_strike_dataset import BuildConfig, load_raw

    config = BuildConfig(raw_dir=raw_dir)
    columns = ["game_year", "game_type", "game_date", "game_pk", "at_bat_number",
               "pitch_number", "pitcher", "player_name", "pitch_type",
               "plate_x", "plate_z", "release_pos_x", "release_pos_y", "release_pos_z",
               "release_extension"] + KINEMATIC_COLUMNS
    raw = load_raw(config, columns=columns)
    raw = raw[raw["game_type"].eq("R")].copy()
    raw = add_release_angles(raw)

    report_dir.mkdir(parents=True, exist_ok=True)
    report = location_model_report(raw)
    print("[검증] 위치 모델:", {k: v for k, v in report.items() if k != "weights"})
    print("[검증] 가중치:", report["weights"])

    table = kirby_index_table(raw, weights=report["weights"])
    table.to_csv(report_dir / "kirby_index_by_season.csv", index=False, encoding="utf-8-sig")
    sticky = year_to_year_stickiness(table)
    sticky.to_csv(report_dir / "stickiness.csv", index=False, encoding="utf-8-sig")
    pd.Series(report["weights"]).to_json(report_dir / "location_model_weights.json")
    print(f"[저장] {report_dir.resolve()}")
    print(sticky.to_string(index=False))
    print(table.head(15).to_string(index=False))


if __name__ == "__main__":
    if Path("data/statcast_raw_sequence").exists():
        run_on_raw()
    else:
        print("data/statcast_raw_sequence not found. Collect raw data first:")
        print('  python -c "from build_statcast_strike_dataset import *; collect_raw(BuildConfig())"')
