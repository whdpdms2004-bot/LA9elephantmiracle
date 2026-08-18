"""hw x sj Val2024 결합 재분석 -- 21_TEAM_SUMMARY.md SS9-3의 후속.

전체 결합은 이득이 거의 없지만(상관 0.91), 구간별로 보면 이득이 몰려있는 곳이
있다 (투수 저표본, game_type=F). 상세: hw_sj_segmented_blend_analysis.md

데이터: cowork/hw/val2024_pred.csv, cowork/sj/val2024_pred.csv, data/train.csv 만
사용. 리더보드 미참조 (RULES.md SS2 -- w* = M^-1 A를 홀드아웃 2024 실제 라벨로 계산).

저장소 루트 기준 상대경로만 사용 (AGENTS.md A6 -- 절대경로 하드코딩 금지).
저장소 어디에서 clone해도 그대로 동작.

실행 (저장소 루트에서):
    py cowork/hw/hw_sj_blend_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "control_success"
REPO = Path(__file__).resolve().parents[2]  # cowork/hw -> cowork -> 저장소 루트
HW_PRED_PATH = REPO / "cowork" / "hw" / "val2024_pred.csv"
SJ_PRED_PATH = REPO / "cowork" / "sj" / "val2024_pred.csv"
TRAIN_PATH = REPO / "data" / "train.csv"


def analyze(sub, label):
    yy = sub[TARGET].to_numpy()
    rr = yy.mean()
    UU = rr * (1 - rr) if 0 < rr < 1 else 1e-9
    p_hw, p_sj = sub["p_hw"].to_numpy(), sub["p_sj"].to_numpy()
    d_hw, d_sj = p_hw - rr, p_sj - rr
    V_hw, V_sj = np.mean(d_hw**2), np.mean(d_sj**2)
    C = np.mean(d_hw * d_sj)
    corr = C / np.sqrt(V_hw * V_sj) if V_hw > 0 and V_sj > 0 else float("nan")
    A_hw = np.mean(d_hw * (yy - rr))
    A_sj = np.mean(d_sj * (yy - rr))
    bss_hw = (2 * A_hw - V_hw) / UU * 100000
    bss_sj = (2 * A_sj - V_sj) / UU * 100000
    M = np.array([[V_hw, C], [C, V_sj]])
    Avec = np.array([A_hw, A_sj])
    try:
        w = np.linalg.solve(M, Avec)
        V_blend = w @ M @ w
        A_blend = w @ Avec
        bss_blend = (2 * A_blend - V_blend) / UU * 100000
    except np.linalg.LinAlgError:
        w = [float("nan"), float("nan")]
        bss_blend = float("nan")
    gain = bss_blend - max(bss_hw, bss_sj)
    print(f"{label:28s} n={len(sub):7d} corr={corr:.4f}  hw={bss_hw:8.1f} sj={bss_sj:8.1f} "
          f"blend={bss_blend:8.1f}  w=[{w[0]:+.3f},{w[1]:+.3f}]  이득={gain:+7.2f}")


def main():
    raw = pd.read_csv(TRAIN_PATH)
    val = raw[raw.season == 2024][
        ["row_id", "pitcher_id", "batter_id", "asof_pitcher_n", "asof_batter_n", "game_type", TARGET]
    ].copy()

    hw = pd.read_csv(HW_PRED_PATH).rename(columns={TARGET: "p_hw"})
    sj = pd.read_csv(SJ_PRED_PATH).rename(columns={TARGET: "p_sj"})
    df = val.merge(hw, on="row_id", how="inner").merge(sj, on="row_id", how="inner")
    print(f"병합된 행수: {len(df)} / val2024 전체 {len(val)}\n")

    analyze(df, "전체")

    print("\n=== asof_pitcher_n 구간별 ===")
    df["n_bucket"] = pd.cut(df["asof_pitcher_n"].fillna(0), [-1, 100, 500, 2000, 4000, 1e9],
                             labels=["<100", "100-500", "500-2000", "2000-4000", "4000+"])
    for b in df["n_bucket"].cat.categories:
        analyze(df[df["n_bucket"] == b], f"n={b}")

    print("\n=== game_type 별 ===")
    for gt in df["game_type"].unique():
        analyze(df[df["game_type"] == gt], f"game_type={gt}")

    print("\n=== asof_batter_n 구간별 ===")
    df["bn_bucket"] = pd.cut(df["asof_batter_n"].fillna(0), [-1, 100, 500, 2000, 4000, 1e9],
                              labels=["<100", "100-500", "500-2000", "2000-4000", "4000+"])
    for b in df["bn_bucket"].cat.categories:
        analyze(df[df["bn_bucket"] == b], f"batter_n={b}")


if __name__ == "__main__":
    main()
