# -*- coding: utf-8 -*-
"""항목 M — 투수 x 카운트국면 조건부 성공률 피처.

## 왜 이 피처인가

`performance_tracking/group_by_perform/RESULTS.md` §7: **55개 모델 전부**가
2스트라이크 카운트(특히 `0-2`)에서 축 평균 대비 −222~−600 으로 진다. 예외 0/55.
CatBoost·LGB·XGB·FT·MLP·증류가 같은 자리에서 같은 만큼 지므로 모델 계열의 결함이
아니라 **입력에 그 축이 없다**는 뜻이다.

실제로 없다. 공식 `asof_pitcher_success_rate` 는 **카운트를 전부 섞은 평균 하나**다.
"이 투수가 0-2 에서 어떤가" 는 입력 어디에도 없다.

## 시즌 동결 — 왜 확장(expanding) 이 아닌가

2025 평가에서는 그 시즌 이력을 볼 수 없다 (행 독립성). 그러니 test 에서는
**≤2024 로 만든 LUT** 를 쓸 수밖에 없다. 학습 때만 시즌 내 확장 as-of 를 쓰면
학습과 추론의 피처 분포가 달라지고, 그건 전이를 깎는다 (피처 전이율 0.06 이
이미 낮다).

그래서 **시즌 S 의 행은 시즌 < S 만으로 만든 LUT** 를 본다. 정의상 as-of 안전하고
2025 에서 하게 될 일과 정확히 같다. fold2022(val 2022, 학습 ≤2021) 도
fold2024(val 2024, 학습 ≤2023) 도 같은 표를 쓴다 — 시즌 동결이라 폴드와 무관하게
누수가 없다.

## 수축

    prior_lg   = 리그 그 국면 평균 (시즌 < S)
    prior_self = (투수 전체 성공 + M2*prior_lg) / (투수 전체 표본 + M2)
    rate       = (국면 성공 + M1*prior_self) / (국면 표본 + M1)

두 단계로 좁힌다. 국면 표본이 적은 투수는 자기 전체 평균으로, 이력이 없는 투수는
리그 국면 평균으로 흘러간다.

    python cowork/sj/sj_final/src/build_countphase.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[4]
TRAIN = ROOT / "data" / "train.csv"
OUT = ROOT / "cowork" / "sj" / "last_week" / "outputs"
FOLDS = (2022, 2024)
M1, M2 = 50.0, 200.0

COLS = ["cph_rate", "cph_n", "cph_dev_self", "cph_dev_lg", "cph_2s_rate", "cph_2s_dev"]


def phase(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """RESULTS.md §3.1 의 7국면. 겹치지 않게 우선순위를 준다."""
    return np.select(
        [(b == 3) & (s == 2), b == 3, s == 2,
         (b == 0) & (s == 0), (b == 0) & (s == 1), (b == 1) & (s == 1)],
        ["full", "b3", "s2", "p00", "p01", "p11"], default="ahead_bat")


def shrunk(num, den, prior, m):
    return (num + m * prior) / (den + m)


def main() -> None:
    use = ["row_id", "season", "balls_before", "strikes_before",
           "pitcher_id", "control_success"]
    d = pd.read_csv(TRAIN, usecols=lambda c: c.strip("﻿") in use)
    d.columns = [c.strip("﻿") for c in d.columns]
    d["row_id"] = d["row_id"].astype(str)
    d["ph"] = phase(d.balls_before.to_numpy(), d.strikes_before.to_numpy())
    d["s2"] = np.where(d.strikes_before.to_numpy() == 2, "S2", "N")
    d["y"] = d["control_success"].astype(float)
    lab = d[d.y.notna()]
    print(f"train {len(d):,}행 · 라벨 {len(lab):,}행 · 시즌 {sorted(d.season.unique())}")

    out = np.full((len(d), len(COLS)), np.nan, np.float32)
    ci = {c: i for i, c in enumerate(COLS)}

    for S in sorted(d.season.unique()):
        h = lab[lab.season < S]
        rows = (d.season == S).to_numpy()
        if h.empty:
            print(f"  season {S}: 이전 시즌 없음 — 전부 결측 ({rows.sum():,}행)")
            continue

        lg = h.groupby("ph").y.agg(["sum", "size"])
        lg_rate = (lg["sum"] / lg["size"]).rename("prior_lg")
        lg2 = h.groupby("s2").y.agg(["sum", "size"])
        lg2_rate = (lg2["sum"] / lg2["size"])

        allp = h.groupby("pitcher_id").y.agg(["sum", "size"])
        gl = float(h.y.mean())
        prior_self = shrunk(allp["sum"], allp["size"], gl, M2).rename("prior_self")

        g = h.groupby(["pitcher_id", "ph"]).y.agg(["sum", "size"]).reset_index()
        g = g.merge(prior_self, on="pitcher_id").merge(lg_rate, on="ph")
        g["cph_rate"] = shrunk(g["sum"], g["size"], g["prior_self"], M1)
        g["cph_n"] = np.log1p(g["size"])
        g["cph_dev_self"] = g["cph_rate"] - g["prior_self"]
        g["cph_dev_lg"] = g["cph_rate"] - g["prior_lg"]

        g2 = h.groupby(["pitcher_id", "s2"]).y.agg(["sum", "size"]).reset_index()
        g2 = g2.merge(prior_self, on="pitcher_id")
        g2["lg2"] = g2["s2"].map(lg2_rate)
        g2["cph_2s_rate"] = shrunk(g2["sum"], g2["size"], g2["prior_self"], M1)
        g2["cph_2s_dev"] = g2["cph_2s_rate"] - g2["prior_self"]

        cur = d.loc[rows, ["pitcher_id", "ph", "s2"]]
        a = cur.merge(g[["pitcher_id", "ph", "cph_rate", "cph_n",
                         "cph_dev_self", "cph_dev_lg"]],
                      on=["pitcher_id", "ph"], how="left")
        b = cur.merge(g2[["pitcher_id", "s2", "cph_2s_rate", "cph_2s_dev"]],
                      on=["pitcher_id", "s2"], how="left")
        # 이력 없는 투수는 리그 국면 평균으로 (수축의 마지막 단계)
        a["cph_rate"] = a["cph_rate"].fillna(a["ph"].map(lg_rate))
        a["cph_n"] = a["cph_n"].fillna(0.0)
        b["cph_2s_rate"] = b["cph_2s_rate"].fillna(b["s2"].map(lg2_rate))

        for c in ("cph_rate", "cph_n", "cph_dev_self", "cph_dev_lg"):
            out[rows, ci[c]] = a[c].to_numpy(np.float32)
        for c in ("cph_2s_rate", "cph_2s_dev"):
            out[rows, ci[c]] = b[c].to_numpy(np.float32)
        print(f"  season {S}: 이력 {len(h):,}행 · 투수x국면 {len(g):,}쌍 · "
              f"결측 {np.isnan(out[rows]).any(1).sum():,}행")

    res = pd.DataFrame(out, columns=COLS)
    res.insert(0, "row_id", d["row_id"].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    for f in FOLDS:
        p = OUT / f"countphase_features_fold{f}.parquet"
        res.to_parquet(p, index=False)
        print(f"저장 {p.name}  {res.shape}")

    print("\n=== 2스트라이크 국면에서 신호가 있나 (상관, 라벨 있는 행) ===")
    m = lab.index
    chk = res.loc[m].assign(y=lab.y.to_numpy(), ph=lab.ph.to_numpy(), season=lab.season.to_numpy())
    chk = chk[chk.season >= 2020]
    for ph in ("s2", "full", "p00", "ahead_bat"):
        t = chk[chk.ph == ph]
        print(f"  {ph:<10} n={len(t):>9,}  corr(cph_rate,y)={t.cph_rate.corr(t.y):+.4f}  "
              f"corr(cph_dev_self,y)={t.cph_dev_self.corr(t.y):+.4f}")


if __name__ == "__main__":
    main()
