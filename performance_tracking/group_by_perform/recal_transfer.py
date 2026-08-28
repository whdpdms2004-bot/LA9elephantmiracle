"""구간별 보정이 시즌을 건너 전이되는지 판정한다 — PLAN.md §4.

같은 시즌에서 고르고 같은 시즌에서 재면 반드시 좋아 보인다. 그래서 적합은 한
시즌에서, 판정은 다른 시즌에서 한다. 두 방향 다 낸다.

두 가지 처방을 잰다.
  shift  : 구간마다 상수 c 하나. Brier 감소 = -2c*bias_eval - c^2 (항등식)
  logit  : 구간마다 a + b*logit(p). 기울기 오보정까지 잡는 상한

그리고 반드시 전역 대조군(구간을 나누지 않은 같은 처방)과 비교한다. 구간별
이득의 대부분이 전역 치우침이면 축을 나눈 의미가 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gbp_common as G
from common import load_labels, load_pred

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_logit(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """IRLS 2모수 로지스틱. sklearn 정규화 기본값에 휘둘리지 않게 직접 푼다."""
    X = np.column_stack([np.ones(len(p)), logit(p)])
    w = np.array([0.0, 1.0])
    for _ in range(50):
        z = X @ w
        mu = 1 / (1 + np.exp(-z))
        s = np.clip(mu * (1 - mu), 1e-9, None)
        H = X.T @ (X * s[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - mu))
        w = w + step
        if np.abs(step).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def gains(y_e, p_e, y_f, p_f, nu_e, n_e) -> dict:
    """적합 시즌(f)에서 상수·로짓 보정을 뽑아 평가 시즌(e)에서 회수량을 잰다."""
    if len(y_f) < 200 or len(y_e) < 200:
        return {"shift_xfer": np.nan, "logit_xfer": np.nan,
                "bias_fit": np.nan, "bias_eval": np.nan, "c": np.nan, "b": np.nan}
    b_f = float(p_f.mean() - y_f.mean())
    b_e = float(p_e.mean() - y_e.mean())
    c = -b_f                                             # 적합 시즌의 치우침을 되돌린다
    d_shift = -(2 * c * b_e + c * c)                     # Brier 감소량 (항등식)
    a, s = fit_logit(y_f, p_f)
    p_new = 1 / (1 + np.exp(-(a + s * logit(p_e))))
    d_logit = float(np.mean((p_e - y_e) ** 2) - np.mean((p_new - y_e) ** 2))
    k = 100000.0 * (len(y_e) / n_e) / nu_e
    return {"shift_xfer": k * d_shift, "logit_xfer": k * d_logit,
            "bias_fit": b_f, "bias_eval": b_e, "c": c, "b": s}


def main() -> None:
    lab = {s: load_labels(s) for s in G.SEASONS}
    ax = {s: G.load_axes(s) for s in G.SEASONS}
    # 2022 의 F 는 기저율이 0.69~0.74 라 all 로 재면 축 결론이 전부 F 로 끌려간다.
    # 보정 전이는 R 에서만 판정한다 (PLAN.md §4).
    isR = {s: (lab[s]["game_type"].to_numpy() == "R") for s in G.SEASONS}
    names = [m for m in G.MAIN]
    rows, glob = [], []

    for name in names:
        try:
            pr = {s: load_pred(name, s, lab[s]) for s in G.SEASONS}
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        for ev in G.SEASONS:
            ft = 2022 if ev == 2024 else 2024
            re_, rf_ = isR[ev], isR[ft]
            y_e, y_f = lab[ev]["y"].to_numpy()[re_], lab[ft]["y"].to_numpy()[rf_]
            pe, pf = pr[ev][re_], pr[ft][rf_]
            axe, axf = ax[ev][re_], ax[ft][rf_]
            n_e = len(y_e)
            nu_e = float(y_e.mean() * (1 - y_e.mean()))
            g = gains(y_e, pe, y_f, pf, nu_e, n_e)
            glob.append({"model": name, "eval": ev, "fit": ft, "axis": "(전역)",
                         "bin": "(전체)", "n_eval": n_e, **g})
            for axis, (title, order) in G.AXES.items():
                for b in (order or sorted(pd.unique(ax[ev][axis]))):
                    me = (axe[axis] == b).to_numpy()
                    mf = (axf[axis] == b).to_numpy()
                    if me.sum() < 200 or mf.sum() < 200:
                        continue
                    g = gains(y_e[me], pe[me], y_f[mf], pf[mf], nu_e, n_e)
                    rows.append({"model": name, "eval": ev, "fit": ft, "axis": axis,
                                 "axis_title": title, "bin": b, "n_eval": int(me.sum()), **g})
        print(f"  {name} 완료")

    df = pd.DataFrame(rows)
    gv = pd.DataFrame(glob)
    df.to_csv(G.OUT / "recal_bin.csv", index=False)
    gv.to_csv(G.OUT / "recal_global.csv", index=False)

    # 축 단위 합계 vs 전역 대조군
    agg = (df.groupby(["model", "eval", "axis", "axis_title"])
             [["shift_xfer", "logit_xfer"]].sum().reset_index()
             .merge(gv[["model", "eval", "shift_xfer", "logit_xfer"]],
                    on=["model", "eval"], suffixes=("", "_glob")))
    agg["shift_net"] = agg.shift_xfer - agg.shift_xfer_glob
    agg["logit_net"] = agg.logit_xfer - agg.logit_xfer_glob
    agg.to_csv(G.OUT / "recal_axis.csv", index=False)

    print(f"\nout/recal_bin.csv {len(df):,}행 · recal_axis.csv {len(agg):,}행")
    e = agg[agg["eval"] == 2024]
    print("\n=== eval2024 (fit2022) · 축별 순이득 = 구간별 - 전역 ===")
    print(e.pivot_table(index="axis_title", columns="model", values="shift_net")
           .round(1).to_string())


if __name__ == "__main__":
    main()
