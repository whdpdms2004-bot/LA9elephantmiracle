# -*- coding: utf-8 -*-
"""cw 내부 **3번째 멤버**를 원시 예측으로 판정한다. `mlp` 는 죽어 있다.

## 왜 이 실험인가

원시 예측·월전방분할로 재보니 **`mlp` 가중이 depth 5~8 전부에서 0.00** 이다.
보정된 계기에서는 0.05~0.08 이 붙어 살아 있는 것처럼 보였는데, 그건
`calib` 가 평가 라벨로 스케일을 맞춰준 덕이었다.

그러면 그 자리를 무엇으로 채울지가 문제다. 오늘 두 계기가 엇갈렸던
`cb_f_only` 를 **오염되지 않은 계기로** 다시 판정한다.

## 후보

    mlp          현행 (죽어 있다)
    cb_f_only    F 행만으로 학습한 CatBoost — 오늘 val 채택안
    cb_f_d8      F 행 + depth 8
    cb_last1     직전 시즌만으로 학습
    없음         cb + ft 두 멤버로만

`run_family` 의 `ROW_FILTERS` 와 같은 정의를 쓰되 **calib 를 거치지 않는다.**

    python third_raw.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
sys.path.insert(0, str(ROOT / "performance_tracking" / "tools"))
sys.path.insert(0, str(FINAL / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FIT_M = (3, 4, 5, 6)
STEP = 0.05
SEEDS = 3

# (이름, depth, 행필터)  행필터 None 이면 전체
CAND = [("cb_f_only", 5, "F"), ("cb_f_d8", 8, "F"), ("cb_last1", 5, "last1")]


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def simplex(k, step=STEP):
    if k == 1:
        return [np.array([1.0])]
    g = np.arange(0.0, 1.0 + 1e-9, step)
    out = []
    for c in itertools.product(g, repeat=k - 1):
        s = sum(c)
        if s <= 1.0 + 1e-9:
            out.append(np.array(list(c) + [1.0 - s]))
    return out


def main():
    from common import load_labels
    from run_arm import CB_P, load_base
    import atoms as A
    from catboost import CatBoostRegressor, Pool

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]
    gt_all = np.asarray(np.load(FINAL / "work" / "game_type.npy", allow_pickle=True), str)

    got = {}
    for fold in (2024, 2022):
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        Xall = np.concatenate([np.asarray(X), E], axis=1)
        Xt = np.ascontiguousarray(Xall[tri])
        Xv = np.ascontiguousarray(Xall[va])
        yt = y_all[tri]
        st = season[tri]
        gt = gt_all[tri]
        del E, Xall
        for nm, d, rows in CAND:
            fp = FINAL / "preds" / f"RAW_{nm}_{fold}.npy"
            if fp.exists():
                got[(fold, nm)] = np.load(fp)
                continue
            if rows == "F":
                mk = gt == "F"
            elif rows == "last1":
                mk = st == st.max()
            else:
                mk = np.ones(len(yt), bool)
            xt, yy = np.ascontiguousarray(Xt[mk]), yt[mk]
            acc = np.zeros(len(va))
            t0 = time.time()
            for sd in range(SEEDS):
                m = CatBoostRegressor(**{**CB_P, "depth": d}, random_seed=sd,
                                      task_type="GPU", devices="0", border_count=128)
                m.fit(Pool(xt, yy.astype(np.float64)))
                acc += np.clip(m.predict(Pool(Xv)), 1e-6, 1 - 1e-6)
                del m
            got[(fold, nm)] = acc / SEEDS
            np.save(fp, got[(fold, nm)])
            print("  fold%d %-12s %s행 %.0f초" % (fold, nm, f"{len(xt):,}", time.time() - t0),
                  flush=True)
        del Xt, Xv

    LB = {f: load_labels(f) for f in (2024, 2022)}
    ml = {f: np.load(FINAL / "preds" / f"S1_base__mlp_{f}.npy") for f in (2024, 2022)}
    ft = {f: np.load(FINAL / "preds" / f"S1_base__ft_{f}.npy") for f in (2024, 2022)}
    cb = {f: np.load(FINAL / "preds" / f"RAW_d6_{f}.npy") for f in (2024, 2022)}

    print("\n" + "=" * 96)
    print("cw 내부 3번째 멤버 — 원시 예측 · 월전방분할 (합=1·비음수)")
    print("=" * 96)
    print("%-12s %10s %12s %12s %12s   %s"
          % ("3번째", "단독2024", "2024 정방향", "2024 역방향", "2022 정방향", "w (cb/ft/3rd)"))
    print("-" * 96)

    opts = [("없음(2멤버)", None), ("mlp", "mlp")] + [(n, n) for n, _, _ in CAND]
    base = None
    for label, key in opts:
        cols = {}
        for f in (2024, 2022):
            if key is None:
                cols[f] = np.column_stack([cb[f], ft[f]])
            elif key == "mlp":
                cols[f] = np.column_stack([cb[f], ft[f], ml[f]])
            else:
                cols[f] = np.column_stack([cb[f], ft[f], got[(f, key)]])
        vals = []
        wshow = None
        for f in (2024, 2022):
            y = LB[f]["y"].to_numpy(np.float64)
            mth = LB[f]["game_month"].to_numpy()
            f1 = np.isin(mth, FIT_M)
            cand = simplex(cols[f].shape[1])
            dirs = ((f1, ~f1), (~f1, f1)) if f == 2024 else ((f1, ~f1),)
            for fitm, evm in dirs:
                rf, re_ = y[fitm].mean(), y[evm].mean()
                sc = [bss(np.clip(rf + (cols[f][fitm] - rf) @ w, 1e-6, 1 - 1e-6), y[fitm])
                      for w in cand]
                w = cand[int(np.argmax(sc))]
                vals.append(bss(np.clip(re_ + (cols[f][evm] - re_) @ w, 1e-6, 1 - 1e-6),
                                y[evm]))
                if wshow is None:
                    wshow = w
        solo = (bss(got[(2024, key)], LB[2024]["y"].to_numpy(np.float64))
                if key not in (None, "mlp")
                else (bss(ml[2024], LB[2024]["y"].to_numpy(np.float64))
                      if key == "mlp" else float("nan")))
        if base is None:
            base = vals
        d = [v - b for v, b in zip(vals, base)]
        print("%-12s %10.1f %12.1f %12.1f %12.1f   %s   Δ %+.1f/%+.1f/%+.1f"
              % (label, solo, vals[0], vals[1], vals[2],
                 np.array2string(wshow, precision=2), d[0], d[1], d[2]))
    print("\n(기준 = 없음(2멤버). Δ 는 2024정/2024역/2022정 순)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
