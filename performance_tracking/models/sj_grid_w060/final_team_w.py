# -*- coding: utf-8 -*-
"""팀 결합 가중을 **실제 cw 모듈 예측**으로 정한다. 대용물이 아니다.

앞서 `sj_cb_ft_fonly`(887.7)를 cw 모듈 대용으로 썼지만, 제출될 cw 모듈은
`w*(새 cb, ft, mlp)` 블렌드라 더 강하다. cw 가 강해지면 최적 w_cw 도 올라가므로
대용물로 고른 값은 과소가 된다. 여기서는 실제 조합을 만들어 잰다.

## 판정 규약

`p = r + w_cw(p_cw - r) + w_sj(p_sj - r)`, **합 = 1 제약**.
합을 풀면 같은 분할에서 -24.8 로 무너진다. 합이 1 이면 center_shift 도 필요 없다.

월3~6 적합 -> 월7~10 평가, 그리고 역방향. **두 방향 모두**에서 좋아야 채택한다.

이 하네스는 리더보드로 검증됐다 — 제출3(합 1.143)의 Δval -6.1 을 실제 -6.7 로 맞혔다.

    python final_team_w.py --cb GRID_idfreq__g_d6_l3k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ROOT = FINAL.parents[2]
PREDS = FINAL / "preds"
PT = ROOT / "performance_tracking"
FIT_M = (3, 4, 5, 6)
GRID = np.arange(0.20, 0.901, 0.01)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cb", required=True)
    ap.add_argument("--w", default="", help="cb,ft,mlp 가중. 비우면 work/cw_w_new.npy")
    a = ap.parse_args()

    W = (np.array([float(x) for x in a.w.split(",")]) if a.w
         else np.load(FINAL / "work" / "cw_w_new.npy"))

    sys.path.insert(0, str(PT / "tools"))
    sys.path.insert(0, str(HERE))
    from common import load_labels

    y = load_labels(2024)["y"].to_numpy(np.float64)
    P = np.column_stack([np.load(PREDS / f"{a.cb}_2024.npy"),
                         np.load(PREDS / "S1_base__ft_2024.npy"),
                         np.load(PREDS / "S1_base__mlp_2024.npy")])
    r_all = y.mean()
    p_cw = np.clip(r_all + (P - r_all) @ W, 1e-6, 1 - 1e-6)

    # row_id 정렬 — preds 는 fold 순서, sj3way 는 row_id 기준이다
    L = pd.read_csv(PT / ".cache" / "labels_2024.csv")
    assert len(L) == len(y), "라벨 행수 불일치"
    sj = (pd.read_csv(PT / "val" / "sj3way_2024.csv")
          .set_index("row_id").loc[L["row_id"]]["pred"].to_numpy(float))
    m = L["game_month"].to_numpy()

    print("=" * 80)
    print("팀 결합 가중 — cw 모듈 = w*(%s, ft, mlp)" % a.cb)
    print("  w_cw내부 = cb %.4f · ft %.4f · mlp %.4f" % tuple(W))
    print("=" * 80)
    print("\n  단독  cw모듈 %8.1f   sj3way %8.1f   rho %.4f"
          % (bss(p_cw, y), bss(sj, y), np.corrcoef(p_cw, sj)[0, 1]))

    def best(fitm, evm, tag):
        rf, re_ = y[fitm].mean(), y[evm].mean()
        sf = [bss(np.clip(rf + w * (p_cw[fitm] - rf) + (1 - w) * (sj[fitm] - rf),
                          1e-6, 1 - 1e-6), y[fitm]) for w in GRID]
        w0 = float(GRID[int(np.argmax(sf))])
        ev = lambda w: bss(np.clip(re_ + w * (p_cw[evm] - re_)
                                   + (1 - w) * (sj[evm] - re_), 1e-6, 1 - 1e-6), y[evm])
        # 배포 중인 제출2 구성 (합 1.098 + shift)
        dep = bss(np.clip(re_ + 0.4546 * (p_cw[evm] - re_)
                          + 0.6433 * (sj[evm] - re_) - 0.003223, 1e-6, 1 - 1e-6), y[evm])
        print("\n  %s" % tag)
        print("     적합월 최적 w_cw %.2f  ->  eval %.1f" % (w0, ev(w0)))
        print("     제출2 배포값(합1.098+shift)  eval %.1f   Δ %+.1f" % (dep, ev(w0) - dep))
        print("     %-6s" % "w_cw" + "".join("%8.2f" % w for w in
                                             (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)))
        print("     %-6s" % "eval" + "".join("%8.1f" % ev(w) for w in
                                             (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)))
        return w0

    f1 = np.isin(m, FIT_M)
    wa = best(f1, ~f1, "정방향  월3~6 적합 -> 월7~10 평가")
    wb = best(~f1, f1, "역방향  월7~10 적합 -> 월3~6 평가")
    print("\n  두 방향 최적 w_cw = %.2f · %.2f  ->  채택 %.2f (중점)"
          % (wa, wb, round((wa + wb) / 2, 2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
