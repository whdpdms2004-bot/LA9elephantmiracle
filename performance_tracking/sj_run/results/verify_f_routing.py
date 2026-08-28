# -*- coding: utf-8 -*-
"""hw 2순위 F행 라우팅 — 독립 재검증 (val 예측 파일만, 재학습 없음)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"C:\Users\isj67\Desktop\LA9elephantmiracle")
sys.path.insert(0, str(ROOT / "performance_tracking" / "tools"))
from common import load_labels
VAL = ROOT / "performance_tracking/val"

def bss(p, y):
    r = y.mean(); return 100000.0*(1.0-((p-y)**2).mean()/(r*(1.0-r)))

def get(name, season, L):
    d = pd.read_csv(VAL / f"{name}_{season}.csv")
    m = L[["row_id"]].merge(d, on="row_id", how="left")
    assert m["pred"].notna().all(), f"{name}_{season} row_id 불일치"
    return m["pred"].to_numpy(np.float64)

for SEASON, SJ3 in ((2024, "sj3way"), (2022, "sj3way_nv")):
    L = load_labels(SEASON)
    y = L["y"].to_numpy(np.float64); r = y.mean()
    gt = L["game_type"].to_numpy(); mo = L["game_month"].to_numpy()
    isF = (gt == "F"); isR = (gt == "R")
    std = get("sj_stdmlp", SEASON, L); s3 = get(SJ3, SEASON, L)
    hw  = get("hw_v12_honest", SEASON, L)
    champ = np.clip(r + 0.6*(std-r) + 0.4*(s3-r), 1e-6, 1-1e-6)
    print("="*78)
    print("val%d   (%s, F %d행 %.1f%% · R %d행)" % (SEASON, SJ3, isF.sum(), 100*isF.mean(), isR.sum()))
    print("="*78)
    print("  챔피언 전체 %8.1f   F %9.1f   R %8.1f" % (bss(champ,y), bss(champ[isF],y[isF]) if isF.any() else float('nan'), bss(champ[isR],y[isR])))
    print("  hw_honest  전체 %8.1f   F %9.1f   R %8.1f" % (bss(hw,y), bss(hw[isF],y[isF]) if isF.any() else float('nan'), bss(hw[isR],y[isR])))
    print()
    rng = np.random.default_rng(20260829)
    n = len(y); B = 400
    idx = rng.integers(0, n, size=(B, n))
    segs = [("전체", np.ones(n, bool))]
    if SEASON == 2024:
        segs += [("월7-10(정방향)", np.isin(mo,[7,8,9,10])), ("월3-6(역방향)", np.isin(mo,[3,4,5,6]))]
    segs += [("R만(관문)", isR), ("F만", isF)]
    for Lam in (0.10, 0.20):
        p = champ.copy(); p[isF] = (1-Lam)*champ[isF] + Lam*hw[isF]
        print("  L=%.2f" % Lam)
        for nm, m in segs:
            if m.sum() == 0: continue
            d = bss(p[m], y[m]) - bss(champ[m], y[m])
            if nm == "전체":
                bs = np.array([bss(p[i],y[i])-bss(champ[i],y[i]) for i in idx])
                lo, hi = np.percentile(bs,[2.5,97.5])
                print("    %-14s Δ %+8.2f   부트 95%% CI [%+.2f, %+.2f]  %s"
                      % (nm, d, lo, hi, "유의" if lo>0 else "0 포함"))
            else:
                print("    %-14s Δ %+8.2f" % (nm, d))
        print()
