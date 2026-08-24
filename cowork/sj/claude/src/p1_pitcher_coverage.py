"""P1-0: 투수별 모델링 임계값 결정을 위한 커버리지 EDA.

질문: 직전 시즌(또는 누적)에 T구 이상 던진 투수 집합이 다음 시즌 투구의 몇 %를 차지하나.
투수별 모델의 적용 가능 범위 = 이득의 상한이므로 임계값 선택의 1차 근거가 된다.

출력: outputs/p1_pitcher_coverage.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # cowork/sj/claude
DATA = ROOT.parent / "data" / "train.csv"           # cowork/sj/data/train.csv
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TARGET = "control_success"

THRESHOLDS = [1, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]

df = pd.read_csv(DATA, usecols=["season", "pitcher_id", TARGET],
                 dtype={"season": "int16", "pitcher_id": "int32", TARGET: "int8"})
print(f"loaded {len(df):,} rows  seasons={sorted(df['season'].unique())}", flush=True)


def counts(mask):
    return df.loc[mask].groupby("pitcher_id").size()


def coverage(base_seasons, target_season, label):
    base_cnt = counts(df["season"].isin(base_seasons))
    tgt = df[df["season"] == target_season]
    tgt_cnt = tgt.groupby("pitcher_id").size()
    tgt_total = len(tgt)

    rows = []
    for t in THRESHOLDS:
        qual = set(base_cnt[base_cnt >= t].index)
        sub = tgt[tgt["pitcher_id"].isin(qual)]
        rows.append({
            "threshold": t,
            "n_pitchers_base": len(qual),
            "base_pitch_share": float(base_cnt[base_cnt >= t].sum() / base_cnt.sum()),
            "n_pitchers_in_target": int(tgt_cnt.index.isin(list(qual)).sum()),
            "target_pitch_n": len(sub),
            "target_pitch_share": len(sub) / tgt_total,
            "target_success_rate": float(sub[TARGET].mean()) if len(sub) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.insert(0, "spec", label)
    return out


def within_season(season):
    cnt = counts(df["season"] == season)
    total = cnt.sum()
    rows = []
    for t in THRESHOLDS:
        q = cnt[cnt >= t]
        rows.append({
            "threshold": t,
            "n_pitchers_base": len(q),
            "base_pitch_share": np.nan,
            "n_pitchers_in_target": len(q),
            "target_pitch_n": int(q.sum()),
            "target_pitch_share": float(q.sum() / total),
            "target_success_rate": np.nan,
        })
    out = pd.DataFrame(rows)
    out.insert(0, "spec", f"within_{season}")
    return out


def show(frame):
    d = frame.copy()
    for c in ["base_pitch_share", "target_pitch_share", "target_success_rate"]:
        d[c] = (d[c] * 100).round(2)
    print(d.to_string(index=False), flush=True)


specs = [
    within_season(2024),
    coverage([2023], 2024, "2023_only->2024"),
    coverage([2019, 2020, 2021, 2022, 2023], 2024, "2019-2023->2024"),
    coverage([2022], 2023, "2022_only->2023"),
    coverage([2019, 2020, 2021, 2022], 2023, "2019-2022->2023"),
    coverage([2021], 2022, "2021_only->2022"),
    # 2025 추론에 실제로 적용될 구성
    coverage([2024], 2025, "2024_only->2025") if 2025 in set(df["season"]) else None,
]
specs = [s for s in specs if s is not None]
for s in specs:
    print("\n" + "=" * 78)
    print(s["spec"].iloc[0])
    print("=" * 78)
    show(s)

# cold start 구조
c23 = counts(df["season"] == 2023)
c1923 = counts(df["season"] <= 2023)
t24 = df[df["season"] == 2024]
tot24 = len(t24)
no23 = t24[~t24["pitcher_id"].isin(c23.index)]
no_hist = t24[~t24["pitcher_id"].isin(c1923.index)]
print("\n" + "=" * 78)
print("2024 cold start")
print("=" * 78)
print(f"total {tot24:,} pitches / {t24['pitcher_id'].nunique()} pitchers")
print(f"no 2023 appearance : {len(no23):,} ({len(no23)/tot24*100:.2f}%) "
      f"/ {no23['pitcher_id'].nunique()} pitchers / rate {no23[TARGET].mean():.6f}")
print(f"no 2019-2023 hist  : {len(no_hist):,} ({len(no_hist)/tot24*100:.2f}%) "
      f"/ {no_hist['pitcher_id'].nunique()} pitchers / rate {no_hist[TARGET].mean():.6f}")

# 2023 투구수 구간별 2024 분량/성공률
bins = [-1, 0, 24, 99, 199, 499, 999, 10 ** 9]
labels = ["none", "1-24", "25-99", "100-199", "200-499", "500-999", "1000+"]
m = t24[["pitcher_id", TARGET]].copy()
m["c2023"] = m["pitcher_id"].map(c23).fillna(0).astype(int)
m["bucket"] = pd.cut(m["c2023"], bins=bins, labels=labels)
g = m.groupby("bucket", observed=True).agg(
    pitchers=("pitcher_id", "nunique"),
    pitch_n=(TARGET, "size"),
    success_rate=(TARGET, "mean"))
g["pitch_share_pct"] = (g["pitch_n"] / tot24 * 100).round(2)
print("\n" + "=" * 78)
print("2024 rows bucketed by the pitcher's 2023 pitch count")
print("=" * 78)
print(g.round(6).to_string())

pd.concat(specs, ignore_index=True).to_csv(OUT / "p1_pitcher_coverage.csv", index=False)
g.reset_index().to_csv(OUT / "p1_pitcher_bucket_2024.csv", index=False)
print(f"\nsaved -> {OUT / 'p1_pitcher_coverage.csv'}")
