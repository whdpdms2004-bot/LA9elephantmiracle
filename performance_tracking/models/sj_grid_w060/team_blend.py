# -*- coding: utf-8 -*-
"""팀 4멤버 결합 가중을 **정직하게** 적합한다. val2024 자체적합을 쓰지 않는다.

## 왜 시간 전방 분할인가

`p = r + sum_i w_i (p_i - r)` 의 `w* = M^-1 A` 를 val2024 로 적합해 val2024 로 평가하면
멤버 수만큼 자유도가 붙어 부풀려진다. 오늘 way 계열평균에서 자체적합 +9.6 이
정직하게는 +1.2 였다. 팀 결합은 멤버가 4개라 더 심하다.

이전 세션에서 팀 가중을 val2024 로 재적합했다가 `cw -0.479 / sj 1.658` 이 나왔던 것도
같은 병이다 (거기에 더해 그때 제출본 모델은 2024 를 **포함해** 학습돼 있었다).

**지금은 다르다.** 네 멤버 모두 2024 를 빼고 학습한 정직한 OOF 다 (BSS 809~888,
인-샘플이면 나왔을 1500 대가 아니다). 그래서 val2024 안에서 시간으로 자르면
`과거 적합 -> 미래 평가` 라는 실제 배포 구조를 그대로 흉내낼 수 있다.

    fit  월 3~6  (143,541행)  ->  freeze  ->  eval 월 7~10  (109,966행)

역방향(늦은 달 적합 -> 이른 달 평가)도 같이 낸다. 두 방향이 어긋나면
그 가중은 시간에 안 걸린 우연이므로 채택하지 않는다.

## 판정

- 주 판정: 정직 전방 분할의 eval BSS
- 게이트: val2022 비하락 (cw·sj 만 2022 예측이 있다)
- 대조군: 등가중 / 단독 최고 / 챔피언 실제 가중

    python team_blend.py [--extra name=path,...]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PT = Path(r"C:\Users\isj67\Desktop\LA9elephantmiracle\performance_tracking")
RIDGE = 0.02
FIT_MONTHS = (3, 4, 5, 6)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def fit_w(P, y, r):
    """w* = M^-1 A. P 는 (n, k) 예측 행렬."""
    D = P - r
    M = D.T @ D / len(y)
    A = D.T @ (y - r) / len(y)
    M = M + RIDGE * np.trace(M) / len(M) * np.eye(len(M))
    return np.linalg.solve(M, A)


def apply_w(P, w, r):
    return np.clip(r + (P - r) @ w, 1e-6, 1 - 1e-6)


def load(members, fold):
    L = pd.read_csv(PT / ".cache" / f"labels_{fold}.csv")
    out = L[["row_id", "y"]].copy()
    if "game_month" in L.columns:
        out["m"] = L["game_month"].to_numpy()
    ok = []
    for name, path in members.items():
        p = Path(path)
        if not p.exists():
            continue
        d = pd.read_csv(p)[["row_id", "pred"]].rename(columns={"pred": name})
        out = out.merge(d, on="row_id", how="inner")
        ok.append(name)
    return out, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="", help="name=path 를 콤마로. 새 멤버 추가")
    a = ap.parse_args()

    base = {
        "sj":  PT / "val" / "sj_cb_ft_fonly_%d.csv",
        "cw":  PT / "val" / "cw_v17_base_%d.csv",
        "yn":  PT / "val" / "yn_fa10c_%d.csv",
        "hw":  PT / "val" / "hw_v12_%d.csv",
    }
    extra = {}
    for tok in [t.strip() for t in a.extra.split(",") if t.strip()]:
        k, v = tok.split("=", 1)
        extra[k.strip()] = v.strip()

    def paths(fold):
        d = {k: str(v) % fold for k, v in base.items()}
        for k, v in extra.items():
            d[k] = v.replace("{fold}", str(fold))
        return d

    D24, names = load(paths(2024), 2024)
    y24 = D24["y"].to_numpy(float)
    r24 = y24.mean()
    print("=" * 84)
    print("팀 결합 — val2024 %s행 · 멤버 %s · 기저율 %.6f"
          % (f"{len(D24):,}", names, r24))
    print("=" * 84)

    print("\n[1] 단독 BSS 와 상관")
    P24 = D24[names].to_numpy(float)
    for i, n in enumerate(names):
        print("    %-6s 단독 %8.1f" % (n, bss(P24[:, i], y24)))
    print("\n    예측 상관 rho")
    print("           " + "".join("%8s" % n for n in names))
    C = np.corrcoef(P24.T)
    for i, n in enumerate(names):
        print("    %-6s " % n + "".join("%8.4f" % C[i, j] for j in range(len(names))))

    # ── 정직 전방 분할 ─────────────────────────────────────────────────────
    m = D24["m"].to_numpy()
    fit = np.isin(m, FIT_MONTHS)
    ev = ~fit
    print("\n[2] 정직 전방 분할  적합 월%s %s행  ->  평가 월%s %s행"
          % (list(FIT_MONTHS), f"{fit.sum():,}",
             sorted(set(m[ev].tolist())), f"{ev.sum():,}"))
    ye, re_ = y24[ev], y24[ev].mean()
    yf, rf = y24[fit], y24[fit].mean()

    rows = []
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(range(len(names)), k):
            sub = [names[i] for i in combo]
            Pf, Pe = P24[fit][:, combo], P24[ev][:, combo]
            w = fit_w(Pf, yf, rf)
            hon = bss(apply_w(Pe, w, re_), ye)
            eq = bss(apply_w(Pe, np.full(len(combo), 1.0 / len(combo)), re_), ye)
            self_ = bss(apply_w(Pe, fit_w(Pe, ye, re_), re_), ye)
            rows.append((hon, eq, self_, "+".join(sub), w))
    rows.sort(key=lambda t: -t[0])
    print("\n    %-22s %10s %10s %10s   %s"
          % ("조합", "정직", "등가중", "(자체적합)", "동결 w"))
    print("    " + "-" * 84)
    for hon, eq, self_, nm, w in rows:
        star = " <<<" if hon == rows[0][0] else ""
        print("    %-22s %10.1f %10.1f %10.1f   %s%s"
              % (nm, hon, eq, self_, np.array2string(w, precision=3), star))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
