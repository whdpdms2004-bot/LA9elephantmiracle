"""[재도전] 구간별 결합가중 — **편차만** 옮긴다.

## §9 가 기각한 것과 기각하지 않은 것

§9 는 `w_bin(l) = l*w_bin(적합) + (1-l)*w_champ` 를 기각했다. 이 추정량은
`w_bin(적합)` 이 **적합 시즌 고유의 수준 오차를 통째로 싣고 온다.** 구간 간 변동
(σ 0.10~0.24) 보다 시즌 간 변동(0.6+)이 3~6배라, l 을 키우면 틀린 수준이 그대로
들어가 단조로 진다. 실제로 그랬다 (네 칸 전부 -0.8 ~ -13.4).

**그런데 모양의 상관은 rho +0.48 로 살아 있었다.** 수준과 모양을 분리하면 다르다.

    delta_bin = w_bin(적합) - w_global(적합)     <- 같은 시즌에서 빼면 수준이 소거된다
    w_bin     = w_champ + l * delta_bin          <- 수준은 검증값에서, 모양만 적합에서

`w_global` 을 **같은 적합 시즌에서** 빼는 것이 핵심이다. 다른 시즌 값이나 챔피언
값을 빼면 수준 차가 delta 에 남는다.

## 제약

1. share 가중 합이 0 이 되게 delta 를 중심화한다 — 전역 평균 가중을 정확히 보존.
2. 각 구간에서 합=1 로 재정규화 (set_blend.py 규약, 합 1.143 이 Public -6.7).
3. 적합 구간이 2,000행 미만이면 delta=0 (= 챔피언 가중 그대로).

## 판정

기준선은 **챔피언 실측 가중** `w = {cw 0.6, sj 0.4}` (Public 1080.425).
l=0 이 정확히 현행 챔피언이다. 두 방향 다 낸다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gbp_common as G
from bucket_blend import MIN_FIT, bss
from common import load_labels, load_pred

MEMBERS = {"cw": "cw_v17_base", "sj": "sj_cb_ft_fonly"}
W_CHAMP = np.array([0.6, 0.4])
LAMS = (0.0, 0.5, 1.0, 1.5, 2.0)


def comb(P, w, r):
    return np.clip(r + (P - r) @ w, G.EPS, 1 - G.EPS)


def main() -> None:
    rows, shapes = [], {}
    for fit_s, ev_s in ((2022, 2024), (2024, 2022)):
        labf, labe = load_labels(fit_s), load_labels(ev_s)
        axf, axe = G.load_axes(fit_s), G.load_axes(ev_s)
        yf, ye = labf.y.to_numpy(), labe.y.to_numpy()
        Pf = np.column_stack([load_pred(v, fit_s, labf) for v in MEMBERS.values()])
        Pe = np.column_stack([load_pred(v, ev_s, labe) for v in MEMBERS.values()])
        rf, re_ = float(yf.mean()), float(ye.mean())
        isR = (labe.game_type == "R").to_numpy()

        w_glob_f = G.fit_weights(Pf, yf)            # ★ 같은 시즌의 전역 — 수준 소거용
        base = comb(Pe, W_CHAMP, re_)

        for axis, (title, order) in G.AXES.items():
            vf, ve = axf[axis].to_numpy(), axe[axis].to_numpy()
            bins = [b for b in (order or sorted(pd.unique(ve))) if (ve == b).any()]
            d = {}
            for b in bins:
                fb = vf == b
                d[b] = (G.fit_weights(Pf[fb], yf[fb], w0=w_glob_f) - w_glob_f
                        if fb.sum() >= MIN_FIT else np.zeros(2))
            # share 가중 중심화 — 전역 평균 가중을 정확히 보존한다
            sh = np.array([(vf == b).mean() for b in bins])
            sh = sh / sh.sum()
            mu = sum(s * d[b] for s, b in zip(sh, bins))
            for b in bins:
                d[b] = d[b] - mu
            shapes[(fit_s, axis)] = {b: d[b].copy() for b in bins}

            for lam in LAMS:
                p = base.copy()
                for b in bins:
                    eb = ve == b
                    w = W_CHAMP + lam * d[b]
                    w = np.clip(w, 0, None)
                    w = w / w.sum() if w.sum() > 0 else W_CHAMP
                    p[eb] = comb(Pe[eb], w, re_)
                rows.append({"axis": axis, "axis_title": title, "lam": lam,
                             "fit": fit_s, "eval": ev_s,
                             "d_all": bss(ye, p) - bss(ye, base),
                             "d_R": bss(ye[isR], p[isR]) - bss(ye[isR], base[isR])})

    df = pd.DataFrame(rows)
    df.to_csv(G.OUT / "bucket_delta.csv", index=False)
    k = df.pivot_table(index=["axis_title", "lam"], columns="eval", values=["d_all", "d_R"])
    dec, gu = ("d_all", 2024), ("d_R", 2022)
    k["최소"] = k.min(1)
    print("=== 편차 전달 — 챔피언 w={cw .6, sj .4} 대비 ===")
    good = k[(k[dec] > 0) & (k[gu] > 0)]
    print(good.round(2).to_string() if len(good) else "  두 관문 다 양수인 (축,lam) 없음")
    print("\n=== 주 판정 칸(2024 all) 상위 8 ===")
    print(k.sort_values(dec, ascending=False).head(8).round(2).to_string())

    print("\n=== 모양(delta)이 시즌을 건너 재현되는가 — cw 성분 상관 ===")
    for axis in ("a6_psucc", "a11_typemo", "a1b_phase", "a12_role", "a5_asofn"):
        a, b = shapes.get((2022, axis)), shapes.get((2024, axis))
        if not a or not b:
            continue
        ks = [x for x in a if x in b]
        u = np.array([a[x][0] for x in ks]); v = np.array([b[x][0] for x in ks])
        if len(ks) > 2 and u.std() > 0 and v.std() > 0:
            print(f"  {G.AXES[axis][0]:<26} rho {np.corrcoef(u, v)[0,1]:+.3f}   "
                  f"|delta| 2022 {np.abs(u).mean():.3f} / 2024 {np.abs(v).mean():.3f}")


if __name__ == "__main__":
    main()
