"""구간가중 이득의 표본오차 — 이득이 폴드 잡음과 구별되는가.

예측 파일이 고정이라 재실행 잡음(±2 BSS, CatBoost GPU 비결정성)은 여기 없다.
남는 것은 **평가 폴드 자체의 표본 잡음**이다. 행 부트스트랩으로 그걸 잰다.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import gbp_common as G
from bucket_blend import MIN_FIT, blend, bss
from common import load_labels, load_pred

B = 400
RNG = np.random.default_rng(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--member", action="append", default=None)
    ap.add_argument("--fit", type=int, default=2022)
    ap.add_argument("--eval", dest="ev", type=int, default=2024)
    ap.add_argument("--scope", default="all")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--tag", default="strong")
    a = ap.parse_args()
    members = a.member or ["sj_stdmlp", "sj_grid_w060", "cw_v17_base", "sj_cb_ft_fonly"]

    labf, labe = load_labels(a.fit), load_labels(a.ev)
    mf = np.ones(len(labf), bool) if a.scope == "all" else (labf.game_type == "R").to_numpy()
    me = np.ones(len(labe), bool) if a.scope == "all" else (labe.game_type == "R").to_numpy()
    yf, ye = labf.y.to_numpy()[mf], labe.y.to_numpy()[me]
    Pf = np.column_stack([load_pred(m, a.fit, labf)[mf] for m in members])
    Pe = np.column_stack([load_pred(m, a.ev, labe)[me] for m in members])
    axf = G.load_axes(a.fit)[mf].reset_index(drop=True)
    axe = G.load_axes(a.ev)[me].reset_index(drop=True)

    w_glob = G.fit_weights(Pf, yf)
    p_glob = blend(Pe, w_glob)

    idx = RNG.integers(0, len(ye), size=(B, len(ye)))
    rows = []
    for axis, (title, order) in G.AXES.items():
        vf, ve = axf[axis].to_numpy(), axe[axis].to_numpy()
        p = p_glob.copy()
        for b in (order or sorted(pd.unique(ve))):
            fb, eb = vf == b, ve == b
            if not eb.any():
                continue
            w = G.fit_weights(Pf[fb], yf[fb], w0=w_glob) if fb.sum() >= MIN_FIT else w_glob
            p[eb] = blend(Pe[eb], a.lam * w + (1 - a.lam) * w_glob)
        d = bss(ye, p) - bss(ye, p_glob)
        # 부트스트랩: 같은 리샘플에서 두 예측을 같이 재야 상관이 살아남는다
        bs = np.array([bss(ye[i], p[i]) - bss(ye[i], p_glob[i]) for i in idx])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"axis_title": title, "delta": d, "se": bs.std(),
                     "lo95": lo, "hi95": hi, "0포함": lo <= 0 <= hi})
        print(f"  {title:<34} {d:+7.2f}  se {bs.std():5.2f}  "
              f"[{lo:+6.2f}, {hi:+6.2f}]  {'잡음과 구별 안 됨' if lo <= 0 <= hi else '★유의'}")

    df = pd.DataFrame(rows)
    df.to_csv(G.OUT / f"bucket_ci_{a.tag}_{a.fit}to{a.ev}_{a.scope}_l{a.lam}.csv", index=False)


if __name__ == "__main__":
    main()
