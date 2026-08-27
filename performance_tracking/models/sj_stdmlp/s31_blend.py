# -*- coding: utf-8 -*-
"""[S7 / §31] 결합 가중을 **계획서가 지정한 규약**으로 판정한다.

## 계획서가 요구한 것 (§31)

> **가중치는 전역 단일값이 아니라 구종별로 다르게 탐색한다.**
> 적합은 `fit(val2022) -> 동결 -> eval(val2024)` 로만 한다.
> 결합층 상수는 같은 폴드 자체적합 금지 (§22).

지금까지 나는 **월 전방분할**(같은 시즌 안에서 월3~6 -> 월7~10)로 판정해왔다.
그건 시즌 **내부** 전이다. §31 이 지정한 것은 **시즌 전이**
(fit 2022 -> eval 2024) 이고, 배포 구조(2019~2024 학습 -> 2025 예측)와 더 가깝다.
두 계기가 다른 답을 낼 수 있으므로 지정된 쪽으로 다시 잰다.

## 구종 축의 조작화

원본 데이터에 **실제 던진 구종 열이 없다.** 있는 것은 투수의 as-of 성향뿐이다.

    asof_pitcher_fastball_rate · breaking_rate · offspeed_rate

그래서 "구종별" 을 **투수 주무기**(세 성향의 argmax)로 조작화한다.
as-of 값이라 그 행 시점까지의 정보만 쓰고, 행 독립적이다.

## arm

    global        전역 단일 가중 (기준)
    dom3          주무기 3칸 (fastball / breaking / offspeed)
    dom3_shrink   위를 전역 가중으로 lam 수축 — 볼카운트·R/F 때 이게 살렸다

## 판정

`fit(2022) -> 동결 -> eval(2024)`. 역방향(fit 2024 -> eval 2022)도 함께 낸다 —
한 방향만 좋으면 시간에 안 걸린 우연이다 (볼카운트에서 이미 겪었다).

    python s31_blend.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
PT = ROOT / "performance_tracking"
sys.path.insert(0, str(PT / "tools"))
sys.path.insert(0, str(FINAL / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

RIDGE = 0.02
TEND = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate"]
LAB = ["fastball", "breaking", "offspeed"]


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def fitw(M, y):
    r = y.mean()
    D = M - r
    A = D.T @ (y - r) / len(y)
    Q = D.T @ D / len(y)
    Q = Q + RIDGE * np.trace(Q) / len(Q) * np.eye(len(Q))
    return np.linalg.solve(Q, A)


def cal(p, q):
    eps = 1e-6
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    lg = np.log(p / (1 - p))
    o = 1.0 / (1.0 + np.exp(-(q["logit_scale"] * (lg - q["logit_center_C0"])
                              + q["logit_target_C1"])))
    return np.clip(o, max(eps, q["target_rate"] - q["cap"]),
                   min(1 - eps, q["target_rate"] + q["cap"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(FINAL / "submit" / "submit_sj_stdmlp.zip"))
    a = ap.parse_args()

    from common import load_labels
    from run_arm import load_base
    P = FINAL / "preds"
    par = json.loads(zipfile.ZipFile(a.zip).read("model/cw/model/params.json"))

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]
    ti = [names.index(c) for c in TEND]

    D = {}
    for f in (2024, 2022):
        va = np.where(season == f)[0]
        y = load_labels(f)["y"].to_numpy(np.float64)
        M = np.column_stack([
            cal(np.load(P / ("E2_var_cb2_a0.15_%d.npy" % f)), par["model_cb"]),
            cal(np.load(P / ("FTX_q64_168_%d.npy" % f)), par["model_ft"]),
            cal(np.load(P / ("PREP_std_176_%d.npy" % f)), par["model_mlp"])])
        T = np.asarray(X)[va][:, ti].astype(np.float64)
        T = np.nan_to_num(T, nan=-1.0)
        dom = np.argmax(T, axis=1)
        dom[np.all(T <= 0, axis=1)] = 0          # 성향 정보가 없으면 fastball 로
        D[f] = (M, y, dom)

    print("=" * 88)
    print("[S7 / §31] 결합 가중 — 계획서 규약 fit(2022) -> 동결 -> eval(2024)")
    print("=" * 88)
    for f in (2024, 2022):
        M, y, dom = D[f]
        print("\nfold%d 주무기 분포" % f)
        for k, lb in enumerate(LAB):
            m = dom == k
            if m.sum() == 0:
                continue
            print("  %-9s %9s행 (%4.1f%%)  기준선 BSS %8.1f"
                  % (lb, format(int(m.sum()), ","), 100 * m.mean(),
                     bss(M[m] @ np.array([par["blend_w_cb"], par["blend_w_ft"],
                                          par["blend_w_mlp"]])
                         + y[m].mean() * (1 - par["blend_w_cb"] - par["blend_w_ft"]
                                          - par["blend_w_mlp"]), y[m])))

    def run(fit_f, ev_f, mode, lam=1.0):
        Mf, yf, df = D[fit_f]
        Me, ye, de = D[ev_f]
        re_ = ye.mean()
        wg = fitw(Mf, yf)
        if mode == "global":
            return bss(np.clip(re_ + (Me - re_) @ wg, 1e-6, 1 - 1e-6), ye), wg
        out = np.empty(len(ye))
        ws = {}
        for k in range(3):
            mf = df == k
            w = fitw(Mf[mf], yf[mf]) if mf.sum() >= 5000 else wg
            w = lam * w + (1 - lam) * wg
            ws[k] = w
            sel = de == k
            out[sel] = re_ + (Me[sel] - re_) @ w
        return bss(np.clip(out, 1e-6, 1 - 1e-6), ye), ws

    print("\n" + "-" * 88)
    print("%-26s %12s %12s   %s" % ("가중 규칙", "eval2024", "eval2022(역)", "판정"))
    print("-" * 88)
    b24, wg = run(2022, 2024, "global")
    b22, _ = run(2024, 2022, "global")
    print("%-26s %12.1f %12.1f   기준  w %.3f/%.3f/%.3f"
          % ("전역 단일", b24, b22, wg[0], wg[1], wg[2]))
    for lam in (0.3, 0.5, 0.7, 1.0):
        s24, ws = run(2022, 2024, "dom3", lam)
        s22, _ = run(2024, 2022, "dom3", lam)
        ok = "양방향 이득" if s24 > b24 and s22 > b22 else ""
        print("%-26s %+12.1f %+12.1f   %s"
              % ("구종별 lam=%.1f" % lam, s24 - b24, s22 - b22, ok))
    _, ws = run(2022, 2024, "dom3", 1.0)
    print("\n칸별 가중 (fit 2022, 수축 없음)")
    for k, lb in enumerate(LAB):
        print("  %-9s cb %.3f · ft %.3f · mlp %.3f" % (lb, ws[k][0], ws[k][1], ws[k][2]))
    print("\n(§31 은 fit(2022)->eval(2024) 만 요구하지만, 한 방향만 좋으면")
    print(" 시간에 안 걸린 우연이므로 역방향도 함께 본다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
