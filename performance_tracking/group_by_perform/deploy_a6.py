"""항목 P — 배포 가능한 A6 구간별 결합가중을 만든다.

## 기준선을 챔피언의 실제 가중으로 고정하는 이유

전역 가중을 다시 적합하면 그 자체가 하나의 변경이고, 그 종류의 전이율은
**0.0 으로 실측돼 있다** (PLAN_next.md §0). 그걸 섞으면 이득이 버킷 때문인지
재적합 때문인지 갈리지 않는다. 그래서

    기준선     w = {cw 0.6, sj 0.4}   <- sj_grid_w060.zip 이 Public 1080.425 로 실측한 값
    후보       w_bin(l) = l*w_bin + (1-l)*{0.6, 0.4}

로 두고 **버킷 구조 하나만** 바꾼다. l=0 이 정확히 현행 챔피언이다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import gbp_common as G
from bucket_blend import MIN_FIT, bss
from common import load_labels, load_pred

MEMBERS = {"cw": "cw_v17_base", "sj": "sj_cb_ft_fonly"}
W_CHAMP = np.array([0.6, 0.4])          # sj_grid_w060.zip 실측 구성
AXIS = "a6_psucc"
COL = "asof_pitcher_success_rate"       # 배포 시 버킷을 가르는 test.csv 컬럼


def combine(P, w, r):
    """배포 script.py 와 같은 식: p = r + (P - r) @ w."""
    return np.clip(r + (P - r) @ w, G.EPS, 1 - G.EPS)


def main() -> None:
    edges = json.loads((G.OUT / "bins.json").read_text(encoding="utf-8"))["a6_psucc_edges"]
    labels = G.AXES[AXIS][1]
    rows = []
    fits = {}

    for fit_s, ev_s in ((2022, 2024), (2024, 2022)):
        labf, labe = load_labels(fit_s), load_labels(ev_s)
        axf, axe = G.load_axes(fit_s), G.load_axes(ev_s)
        yf, ye = labf.y.to_numpy(), labe.y.to_numpy()
        Pf = np.column_stack([load_pred(v, fit_s, labf) for v in MEMBERS.values()])
        Pe = np.column_stack([load_pred(v, ev_s, labe) for v in MEMBERS.values()])
        rf, re_ = float(yf.mean()), float(ye.mean())

        W = {}
        for b in labels:
            fb = (axf[AXIS] == b).to_numpy()
            W[b] = (G.fit_weights(Pf[fb], yf[fb], w0=W_CHAMP)
                    if fb.sum() >= MIN_FIT else W_CHAMP.copy())
        fits[fit_s] = W

        isR = (labe.game_type == "R").to_numpy()
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            p = np.empty(len(ye))
            for b in labels:
                eb = (axe[AXIS] == b).to_numpy()
                if eb.any():
                    p[eb] = combine(Pe[eb], lam * W[b] + (1 - lam) * W_CHAMP, re_)
            base = combine(Pe, W_CHAMP, re_)
            rows.append({"fit": fit_s, "eval": ev_s, "lam": lam,
                         "all": bss(ye, p), "d_all": bss(ye, p) - bss(ye, base),
                         "R": bss(ye[isR], p[isR]),
                         "d_R": bss(ye[isR], p[isR]) - bss(ye[isR], base[isR])})

    df = pd.DataFrame(rows)
    df.to_csv(G.OUT / "deploy_a6.csv", index=False)
    print("=== 챔피언 w={cw 0.6, sj 0.4} 대비 (버킷 구조만 바꿈) ===")
    print(df.pivot_table(index="lam", columns="eval", values=["d_all", "d_R"]).round(2).to_string())
    print("\n=== 절대 BSS ===")
    print(df.pivot_table(index="lam", columns="eval", values=["all", "R"]).round(1).to_string())

    print("\n=== 구간별 가중 (적합 시즌별) ===")
    for s, W in fits.items():
        print(f"  fit{s}: " + "  ".join(f"{b}=({W[b][0]:.3f},{W[b][1]:.3f})" for b in labels))

    # 배포용: 두 시즌 가중의 평균을 쓴다 (한 시즌 자체적합을 피한다)
    lam = 0.75
    out = {"members": list(MEMBERS), "bucket_col": COL,
           "bucket_edges": [None] + [round(e, 6) for e in edges] + [None],
           "bucket_labels": labels[:-1] + ["nohist"], "buckets": {}}
    lab2024 = load_labels(2024)
    ax2024 = G.load_axes(2024)
    y24 = lab2024.y.to_numpy()
    for b in labels:
        w = np.mean([fits[s][b] for s in fits], axis=0)
        w = lam * w + (1 - lam) * W_CHAMP
        m = (ax2024[AXIS] == b).to_numpy()
        key = "nohist" if b == "이력없음" else b
        out["buckets"][key] = {"n": int(m.sum()), "r": float(y24.mean()),
                               "w": {"cw": float(w[0]), "sj": float(w[1])},
                               "sum": float(w.sum())}
    (G.OUT / "blend_weights_a6.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nout/blend_weights_a6.json (lambda={lam}, 두 시즌 평균)")
    for k, v in out["buckets"].items():
        print(f"  {k:<8} n={v['n']:>7,}  cw={v['w']['cw']:.4f} sj={v['w']['sj']:.4f} 합={v['sum']:.4f}")


if __name__ == "__main__":
    main()
