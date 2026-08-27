# -*- coding: utf-8 -*-
"""구간별 결합 가중. **모델마다 잘하는 구간이 다르면** 한 벌의 w 로는 못 담는다.

## 왜 볼 만한가

오늘 잰 것 중 가장 큰 구조적 사실은 **R 은 포화, F 가 전장**이다 —
멤버 간 BSS 폭이 R 에서 1.9, F 에서 68. 한 벌의 w 는 88.2% 를 차지하는 R 에
끌려가므로 F 에서 최적이 아닐 수 있다.

챔피언 작성자는 **투구표본수** 축으로 시험하고 이득이 작아 접었다. 여기서는
`game_type` · 볼카운트 · 표본수 · 이닝 · 점수차를 다 본다.

## 자유도 함정

구간을 k 개로 쪼개면 자유 모수가 k 배가 된다. 자체적합은 반드시 좋아 보이고
전이는 무너진다. 그래서 **월3~6 에서 구간별 w 를 적합해 월7~10 에서만 평가**한다.
이 하네스는 이미 리더보드로 검증됐다 — 제출3(합 1.143)의 Δval −6.1 을
실제 −6.7 로 맞혔다.

합=1 제약을 건다. 자유 적합은 같은 분할에서 −24.8 로 무너진다.

    python bucket_blend.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(r"C:\Users\isj67\Desktop\LA9elephantmiracle")
PT = ROOT / "performance_tracking"
WORK = ROOT / "cowork" / "sj" / "sj_final" / "work"
FIT_M = (3, 4, 5, 6)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def best_w1(A, B, y):
    """합=1 제약 아래 격자 탐색. w_a 만 자유롭다."""
    r = y.mean()
    g = np.arange(0.20, 0.86, 0.01)
    s = [bss(np.clip(r + wa * (A - r) + (1 - wa) * (B - r), 1e-6, 1 - 1e-6), y)
         for wa in g]
    i = int(np.argmax(s))
    return float(g[i]), float(s[i])


def main():
    L = pd.read_csv(PT / ".cache" / "labels_2024.csv")
    d = L[["row_id", "y", "game_month", "game_type"]].copy()
    for n, f in [("cw", "sj_cb_ft_fonly_2024.csv"), ("sj", "sj3way_2024.csv")]:
        d = d.merge(pd.read_csv(PT / "val" / f)[["row_id", "pred"]]
                    .rename(columns={"pred": n}), on="row_id")

    # 구간 축을 피처 행렬에서 가져온다
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_arm import load_base
    X, y_, season, row_id = load_base()
    names = json.load(open(WORK / "meta.json", encoding="utf-8"))["names"]
    keep = np.asarray(season) == 2024
    axcols = ["balls_before", "strikes_before", "asof_pitcher_n", "inning",
              "abs_score_diff", "outs_before", "num_runners_on"]
    ax = pd.DataFrame({"row_id": np.asarray(row_id)[keep]})
    for c in axcols:
        if c in names:
            ax[c] = np.asarray(X)[keep][:, names.index(c)]
    d = d.merge(ax, on="row_id", how="inner")

    y = d["y"].to_numpy(float)
    m = d["game_month"].to_numpy()
    fit = np.isin(m, FIT_M)
    ev = ~fit
    A = d["cw"].to_numpy(float)
    B = d["sj"].to_numpy(float)
    re_ = y[ev].mean()

    print("=" * 92)
    print("구간별 결합 가중 — 적합 월3~6 %s행 -> 평가 월7~10 %s행 · 합=1 제약"
          % (f"{fit.sum():,}", f"{ev.sum():,}"))
    print("=" * 92)

    wa0, _ = best_w1(A[fit], B[fit], y[fit])
    base = bss(np.clip(re_ + wa0 * (A[ev] - re_) + (1 - wa0) * (B[ev] - re_),
                       1e-6, 1 - 1e-6), y[ev])
    print("\n단일 가중 기준선  w_cw %.2f  ->  eval %.1f" % (wa0, base))

    # 구간 정의
    bs = d["balls_before"].to_numpy() if "balls_before" in d else None
    st = d["strikes_before"].to_numpy() if "strikes_before" in d else None
    pn = d["asof_pitcher_n"].to_numpy() if "asof_pitcher_n" in d else None
    axes = {}
    axes["game_type R/F"] = d["game_type"].to_numpy().astype(str)
    if bs is not None:
        axes["balls 0/1/2/3"] = np.clip(bs, 0, 3).astype(int).astype(str)
        axes["two_strike"] = np.where(st >= 2, "2S", "lt2S")
        axes["count 12칸"] = np.char.add(np.clip(bs, 0, 3).astype(int).astype(str),
                                         np.clip(st, 0, 2).astype(int).astype(str))
        axes["압박 count"] = np.where((bs >= 3) | (st >= 2), "pressure", "normal")
    if pn is not None:
        for thr in (100, 300, 500, 1000):
            axes["pitcher_n >= %d" % thr] = np.where(pn >= thr, "hi", "lo")
        axes["pitcher_n 4분위"] = pd.qcut(pn, 4, labels=["q1", "q2", "q3", "q4"],
                                          duplicates="drop").astype(str)
    if "inning" in d:
        axes["inning <=3/4-6/7+"] = np.select(
            [d["inning"].to_numpy() <= 3, d["inning"].to_numpy() <= 6],
            ["early", "mid"], "late")
    if "abs_score_diff" in d:
        axes["점수차 <=2"] = np.where(d["abs_score_diff"].to_numpy() <= 2, "close", "blow")

    print("\n%-22s %6s %10s %10s   %s" % ("구간 축", "칸수", "eval", "Δ 단일", "칸별 w_cw (적합월)"))
    print("-" * 92)
    out = []
    for nm, g in axes.items():
        g = np.asarray(g)
        labs = sorted(set(g.tolist()))
        p = np.empty(ev.sum())
        ws = {}
        gv, Ae, Be, ye = g[ev], A[ev], B[ev], y[ev]
        small = False
        for lb in labs:
            fm = fit & (g == lb)
            if fm.sum() < 3000:
                small = True
            wa, _ = best_w1(A[fm], B[fm], y[fm]) if fm.sum() >= 500 else (wa0, 0)
            ws[lb] = wa
            sel = gv == lb
            p[sel] = re_ + wa * (Ae[sel] - re_) + (1 - wa) * (Be[sel] - re_)
        s = bss(np.clip(p, 1e-6, 1 - 1e-6), ye)
        out.append((s - base, nm, len(labs), s, ws, small))
    out.sort(reverse=True)
    for dlt, nm, k, s, ws, small in out:
        tag = " ★소표본칸" if small else ""
        wtxt = " ".join("%s=%.2f" % (a, b) for a, b in sorted(ws.items())[:6])
        print("%-22s %6d %10.1f %+10.1f   %s%s" % (nm, k, s, dlt, wtxt, tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
