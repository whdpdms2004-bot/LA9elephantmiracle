"""행 단위 2차 상호작용 후보의 시간 전이성을 저비용으로 감사한다.

각 fold 이전 시즌에서만 bin 경계와 Target lookup을 만들고 다음 시즌에 적용한다.
joint BSS와 두 단변량 중 좋은 값의 차이를 synergy로 기록한다. 이 lookup 자체를
최종 피처로 쓰기 위한 실험이 아니라, 어떤 명시적 상호작용을 GBDT에 줄지 고르는
진단이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "claude" / "src"
sys.path.insert(0, str(SRC))
from harness import TARGET, forecast_base_rate, load, metrics

OUT = Path(__file__).resolve().parent / "outputs" / "audit"
K_SINGLE = 200.0
K_PAIR = 500.0
N_BINS = 10

PAIRS = [
    ("asof_pitcher_ball_rate", "balls_before"),
    ("asof_pitcher_ball_rate", "strikes_before"),
    ("asof_pitcher_reverse_rate", "balls_before"),
    ("asof_pitcher_reverse_rate", "strikes_before"),
    ("asof_pitcher_success_rate", "balls_before"),
    ("asof_pitcher_success_rate", "strikes_before"),
    ("asof_pitcher_prev3_game_success_rate", "balls_before"),
    ("asof_pitcher_prev5_game_success_rate", "balls_before"),
    ("asof_pitcher_prev5_game_success_rate", "strikes_before"),
    ("asof_pitcher_fastball_rate", "balls_before"),
    ("asof_pitcher_breaking_rate", "strikes_before"),
    ("asof_pitcher_offspeed_rate", "strikes_before"),
    ("home_win_expectancy", "inning"),
    ("li", "inning"),
    ("li", "balls_before"),
    ("score_diff_home", "inning"),
    ("pitcher_hand", "batter_hand"),
    ("game_type", "balls_before"),
    ("game_type", "asof_pitcher_success_rate"),
]


def grouped_values(train: pd.Series, valid: pd.Series):
    numeric = pd.to_numeric(train, errors="coerce")
    if pd.api.types.is_numeric_dtype(train) and numeric.nunique(dropna=True) > N_BINS:
        edges = np.unique(np.nanquantile(
            numeric.dropna().to_numpy(np.float64), np.linspace(0, 1, N_BINS + 1)))
        if len(edges) >= 3:
            edges[0], edges[-1] = -np.inf, np.inf
            return (
                pd.cut(numeric, edges, labels=False, include_lowest=True)
                .fillna(-1).astype("int16").to_numpy(),
                pd.cut(pd.to_numeric(valid, errors="coerce"), edges, labels=False,
                       include_lowest=True).fillna(-1).astype("int16").to_numpy(),
            )
    return (train.astype(str).fillna("__NA__").to_numpy(),
            valid.astype(str).fillna("__NA__").to_numpy())


def lookup_prediction(train_y: np.ndarray, train_groups: list[np.ndarray],
                      valid_groups: list[np.ndarray], global_train: float,
                      base_rate: float, strength: float) -> np.ndarray:
    names = [f"g{i}" for i in range(len(train_groups))]
    fit = pd.DataFrame({name: values for name, values in zip(names, train_groups)})
    fit["y"] = train_y
    grouped = fit.groupby(names, dropna=False)["y"].agg(["sum", "size"])
    residual = (grouped["sum"] + strength * global_train) / (
        grouped["size"] + strength) - global_train
    keys = pd.MultiIndex.from_arrays(valid_groups, names=names)
    mapped = residual.reindex(keys).fillna(0.0).to_numpy(np.float64)
    return np.clip(base_rate + mapped, 1e-6, 1 - 1e-6)


def main():
    df = load()
    rows = []
    for fold in (2023, 2024):
        train = df[df["season"] < fold]
        valid = df[df["season"] == fold]
        train_y = train[TARGET].to_numpy(np.float64)
        valid_y = valid[TARGET].to_numpy(np.float64)
        base_rate = forecast_base_rate(df, df["season"].to_numpy() < fold, fold)
        base_bss = metrics(valid_y, np.full(len(valid), base_rate))["bss_raw"]
        global_train = float(train_y.mean())
        grouped = {}
        single_bss = {}
        for column in sorted({value for pair in PAIRS for value in pair}):
            grouped[column] = grouped_values(train[column], valid[column])
            pred = lookup_prediction(
                train_y, [grouped[column][0]], [grouped[column][1]],
                global_train, base_rate, K_SINGLE)
            single_bss[column] = metrics(valid_y, pred)["bss_raw"]
        for left, right in PAIRS:
            pred = lookup_prediction(
                train_y,
                [grouped[left][0], grouped[right][0]],
                [grouped[left][1], grouped[right][1]],
                global_train, base_rate, K_PAIR)
            pair_bss = metrics(valid_y, pred)["bss_raw"]
            best_single = max(single_bss[left], single_bss[right])
            rows.append({
                "fold": fold,
                "left": left,
                "right": right,
                "left_bss": single_bss[left],
                "right_bss": single_bss[right],
                "pair_bss": pair_bss,
                "pair_delta_vs_constant": pair_bss - base_bss,
                "synergy_vs_best_single": pair_bss - best_single,
            })
    result = pd.DataFrame(rows)
    summary = (result.pivot_table(
        index=["left", "right"], columns="fold",
        values="synergy_vs_best_single").reset_index())
    summary.columns = ["left", "right", "synergy_2023", "synergy_2024"]
    summary["min_synergy"] = summary[["synergy_2023", "synergy_2024"]].min(axis=1)
    summary["mean_synergy"] = summary[["synergy_2023", "synergy_2024"]].mean(axis=1)
    summary = summary.sort_values("min_synergy", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "pair_interaction_folds.csv", index=False)
    summary.to_csv(OUT / "pair_interaction_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:+.2f}"))
    print(json.dumps({
        "method": "fold-train-only binned EB lookup diagnostic",
        "single_strength": K_SINGLE,
        "pair_strength": K_PAIR,
        "bins": N_BINS,
        "positive_both": int((summary["min_synergy"] > 0).sum()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
