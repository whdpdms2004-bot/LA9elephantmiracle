"""구간마다 앙상블 가중을 다르게 주면 오르는가 — 정직한 판정.

PLAN.md §4 규약: 가중은 한 시즌에서 적합해 **동결**하고 다른 시즌에서 평가한다.
같은 시즌에서 고르고 같은 시즌에서 재면 구간 가중은 반드시 좋아 보인다 - 그
`self` 값도 같이 내서 얼마나 부풀려지는지 보인다.

수축 lambda: w_bin(l) = l*w_bin + (1-l)*w_global.  l=0 이 전역 가중 그대로다.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

import gbp_common as G
from common import load_labels, load_pred

LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_FIT = 2000          # 적합 시즌 구간이 이보다 작으면 전역 가중으로 되돌린다


def bss(y, p):
    r = float(y.mean())
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / (r * (1 - r)))


def blend(P, w):
    return np.clip(P @ w, G.EPS, 1 - G.EPS)


def run(members: list[str], fit_s: int, ev_s: int, scope: str, out: list) -> None:
    labf, labe = load_labels(fit_s), load_labels(ev_s)
    axf, axe = G.load_axes(fit_s), G.load_axes(ev_s)
    mf = np.ones(len(labf), bool) if scope == "all" else (labf.game_type == "R").to_numpy()
    me = np.ones(len(labe), bool) if scope == "all" else (labe.game_type == "R").to_numpy()

    yf, ye = labf.y.to_numpy()[mf], labe.y.to_numpy()[me]
    Pf = np.column_stack([load_pred(m, fit_s, labf)[mf] for m in members])
    Pe = np.column_stack([load_pred(m, ev_s, labe)[me] for m in members])
    axf, axe = axf[mf].reset_index(drop=True), axe[me].reset_index(drop=True)

    # ★ 진짜 기준선은 전역 블렌드가 아니라 **최고 단독 멤버**다.
    # 블렌드가 최고 단독보다 낮으면 구간가중으로 블렌드를 올려도 제출에는 못 쓴다.
    solo = {m: bss(ye, Pe[:, i]) for i, m in enumerate(members)}
    solo_fit = {m: bss(yf, Pf[:, i]) for i, m in enumerate(members)}
    best_solo_name = max(solo_fit, key=solo_fit.get)      # 적합 시즌으로 고른다(정직)
    best_solo = solo[best_solo_name]

    w_glob = G.fit_weights(Pf, yf)
    base = bss(ye, blend(Pe, w_glob))
    print(f"    최고단독({best_solo_name}, 적합시즌 기준 선택) {best_solo:,.1f}"
          f"  · 전역블렌드 {base:,.1f}  · 차 {base - best_solo:+.1f}")
    # self 상한: 평가 시즌에서 직접 적합한 전역 가중
    base_self = bss(ye, blend(Pe, G.fit_weights(Pe, ye)))
    out.append({"axis": "(전역)", "axis_title": "(전역 가중 하나)", "lam": 0.0,
                "fit": fit_s, "eval": ev_s, "scope": scope, "n_bins": 1,
                "bss": base, "delta": 0.0, "bss_self": base_self,
                "best_solo": best_solo, "best_solo_name": best_solo_name,
                "vs_solo": base - best_solo,
                "w": " ".join(f"{x:.3f}" for x in w_glob)})

    for axis, (title, order) in G.AXES.items():
        vf, ve = axf[axis].to_numpy(), axe[axis].to_numpy()
        bins = [b for b in (order or sorted(pd.unique(ve))) if (ve == b).any()]
        W, W_self, used = {}, {}, 0
        for b in bins:
            fb, eb = vf == b, ve == b
            if fb.sum() < MIN_FIT:
                W[b] = w_glob
            else:
                W[b] = G.fit_weights(Pf[fb], yf[fb], w0=w_glob)
                used += 1
            W_self[b] = G.fit_weights(Pe[eb], ye[eb], w0=w_glob) if eb.sum() >= MIN_FIT else w_glob

        for lam in LAMBDAS:
            p = np.empty(len(ye))
            ps = np.empty(len(ye))
            for b in bins:
                eb = ve == b
                p[eb] = blend(Pe[eb], lam * W[b] + (1 - lam) * w_glob)
                ps[eb] = blend(Pe[eb], lam * W_self[b] + (1 - lam) * w_glob)
            out.append({"axis": axis, "axis_title": title, "lam": lam,
                        "fit": fit_s, "eval": ev_s, "scope": scope,
                        "n_bins": used, "bss": bss(ye, p), "delta": bss(ye, p) - base,
                        "bss_self": bss(ye, ps), "best_solo": best_solo,
                        "best_solo_name": best_solo_name,
                        "vs_solo": bss(ye, p) - best_solo, "w": ""})
        print(f"    {title:<34} 적합구간 {used}/{len(bins)}  "
              f"l=1 델타 {out[-1]['delta']:+.1f}  (self {out[-1]['bss_self'] - base_self:+.1f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--member", action="append", default=None)
    ap.add_argument("--tag", default="core")
    a = ap.parse_args()
    members = a.member or ["sj_stdmlp", "cw_v17_base", "hw_v12_honest", "ye_hand"]
    print(f"멤버 {len(members)}개: {', '.join(members)}")

    rows: list[dict] = []
    for fit_s, ev_s in ((2022, 2024), (2024, 2022)):
        for scope in ("all", "R"):
            print(f"\n[fit {fit_s} -> eval {ev_s} · {scope}]")
            run(members, fit_s, ev_s, scope, rows)

    df = pd.DataFrame(rows)
    df.to_csv(G.OUT / f"bucket_{a.tag}.csv", index=False)
    print(f"\nout/bucket_{a.tag}.csv 저장")

    # 판정: 2024 all(주 판정) 과 2022 R(관문) 에서 동시에 양수인 (축, lam) 만 살아남는다
    d = df[df.axis != "(전역)"]
    k = d.pivot_table(index=["axis_title", "lam"], columns=["eval", "scope"], values="delta")
    dec, guard = (2024, "all"), (2022, "R")
    if dec in k.columns and guard in k.columns:
        k["둘다양수"] = (k[dec] > 0) & (k[guard] > 0)
        print("\n=== 델타 (전역 가중 대비) ===")
        print(k.round(2).to_string())
        win = k[k["둘다양수"]]
        print(f"\n두 조건 다 양수: {len(win)}건" + (f"\n{win.round(2).to_string()}" if len(win) else ""))


if __name__ == "__main__":
    main()
