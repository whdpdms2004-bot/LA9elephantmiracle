# -*- coding: utf-8 -*-
"""cw 내부 3멤버(cb·ft·mlp) 가중을 **새 cb 로** 다시 적합한다.

## 왜 다시 적합하나

제출 2번이 이걸 해서 Public 1074.19 -> 1076.72 (+2.53) 를 얻었다. 바뀐 것은
`cb .599->.710 / ft .290->.236 / mlp .121->.069` 뿐이다. `id_freq` 로 cb 가
강해졌으면 가중도 cb 쪽으로 옮겨야 한다는 것이 리더보드로 확인됐다.

격자 재탐색으로 cb 가 또 강해졌으므로(단독 900.3 -> 908.9) 같은 이유로 다시 적합한다.

## 방법

`p = r + sum w_i (p_i - r)`, `w* = M^-1 A`, ridge 0.02. 두 폴드(2024·2022)에서
각각 적합해 **평균**한다 — 제출 2번이 쓴 방법 그대로다. 한 폴드에만 맞추면
그 폴드의 우연에 붙는다.

    python refit_cw_weights.py --cb GRID_idfreq__g_d6_l3k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
PREDS = FINAL / "preds"
RIDGE = 0.02
FOLDS = (2024, 2022)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def fit(P, y):
    r = y.mean()
    D = P - r
    M = D.T @ D / len(y)
    A = D.T @ (y - r) / len(y)
    M = M + RIDGE * np.trace(M) / len(M) * np.eye(len(M))
    return np.linalg.solve(M, A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cb", required=True, help="새 cb 예측 파일의 접두 (폴드·확장자 제외)")
    ap.add_argument("--ref", default="", help="비교용 기존 cb 접두")
    a = ap.parse_args()

    sys.path.insert(0, str(FINAL.parents[2] / "performance_tracking" / "tools"))
    sys.path.insert(0, str(HERE))
    from common import load_labels

    ws = []
    print("=" * 76)
    print("cw 내부 가중 재적합 — cb = %s" % a.cb)
    print("=" * 76)
    for f in FOLDS:
        y = load_labels(f)["y"].to_numpy(np.float64)
        cb = np.load(PREDS / f"{a.cb}_{f}.npy")
        ft = np.load(PREDS / f"S1_base__ft_{f}.npy")
        ml = np.load(PREDS / f"S1_base__mlp_{f}.npy")
        P = np.column_stack([cb, ft, ml])
        w = fit(P, y)
        r = y.mean()
        b = bss(np.clip(r + (P - r) @ w, 1e-6, 1 - 1e-6), y)
        print("  val%d  단독 cb %8.1f · ft %8.1f · mlp %8.1f  ->  블렌드 %8.1f"
              % (f, bss(cb, y), bss(ft, y), bss(ml, y), b))
        print("        w = cb %.4f · ft %.4f · mlp %.4f  (합 %.4f)"
              % (w[0], w[1], w[2], w.sum()))
        ws.append(w)
    W = np.mean(ws, axis=0)
    print("\n  두 폴드 평균  w = cb %.4f · ft %.4f · mlp %.4f  (합 %.4f)"
          % (W[0], W[1], W[2], W.sum()))

    # 동결 가중을 그대로 썼을 때와 비교한다 — 재적합이 실제로 필요한지 본다
    OLD = np.array([0.7103, 0.2356, 0.0693])
    print("\n  %-10s %10s %10s   %s" % ("폴드", "새 w", "제출2 w", "Δ"))
    for f in FOLDS:
        y = load_labels(f)["y"].to_numpy(np.float64)
        P = np.column_stack([np.load(PREDS / f"{a.cb}_{f}.npy"),
                             np.load(PREDS / f"S1_base__ft_{f}.npy"),
                             np.load(PREDS / f"S1_base__mlp_{f}.npy")])
        r = y.mean()
        n = bss(np.clip(r + (P - r) @ W, 1e-6, 1 - 1e-6), y)
        o = bss(np.clip(r + (P - r) @ OLD, 1e-6, 1 - 1e-6), y)
        print("  val%-6d %10.1f %10.1f   %+.1f" % (f, n, o, n - o))
    np.save(FINAL / "work" / "cw_w_new.npy", W)
    print("\n  -> work/cw_w_new.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
