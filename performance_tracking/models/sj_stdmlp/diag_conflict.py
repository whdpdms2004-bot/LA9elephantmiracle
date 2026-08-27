# -*- coding: utf-8 -*-
"""`cb_f_only` 판정이 계기마다 다르다. **어느 쪽이 틀렸는지 먼저 밝힌다.**

두 측정이 같은 구성(`d6_l10k` + `id_freq`)에서 다른 값을 냈다.

    refit_cw_weights (GRID 저장 예측 사용)   블렌드 val2024  906.9
    seed_budget      (cb 를 새로 학습)        블렌드 val2024  877.0

**cb 절대값이 30 어긋난다.** 이 불일치는 `mlp` vs `cb_f_only` 뿐 아니라 오늘 내린
다른 결론들도 오염시킬 수 있으므로 먼저 잡는다.

## 의심 후보

1. **행 정렬** — `load_base()` 의 `season==fold` 순서와 `load_labels(fold)` 순서가
   다르면 예측과 라벨이 어긋난다. 저장된 `.npy` 는 `va` 순서다.
2. **원자 입력** — `A.build` 에 넘기는 `train_mask` 가 다르면 빈도표가 달라진다.
3. **시드 정렬** — 누적 스냅샷과 단순 평균의 차이.

## 방법

같은 시드·같은 파라미터로 cb 를 **새로 학습**해 저장 예측과 **행 단위로** 비교한다.
정렬이 원인이면 `row_id` 로 맞췄을 때 일치할 것이다.

    python diag_conflict.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
sys.path.insert(0, str(ROOT / "performance_tracking" / "tools"))
sys.path.insert(0, str(FINAL / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def main():
    from common import load_labels
    from run_arm import CB_P, load_base
    import atoms as A
    from catboost import CatBoostRegressor, Pool

    fold = 2024
    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]
    tri = np.where(season <= fold - 1)[0]
    va = np.where(season == fold)[0]

    L = load_labels(fold)
    rid_va = np.asarray(row_id)[va]
    print("=" * 78)
    print("행 정렬 검사")
    print("=" * 78)
    print("  load_base  season==%d  %s행   첫 3개 %s" % (fold, f"{len(va):,}", list(rid_va[:3])))
    print("  labels_%d              %s행   첫 3개 %s"
          % (fold, f"{len(L):,}", list(L['row_id'][:3])))
    same = bool(np.array_equal(rid_va.astype(str), L["row_id"].to_numpy().astype(str)))
    print("  두 순서가 동일한가: %s" % ("예" if same else "★아니오 — 이것이 원인이다"))

    y_lab = L["y"].to_numpy(np.float64)
    y_base = y_all[va].astype(np.float64)
    print("  y 두 경로 최대절대차: %.3e" % float(np.abs(y_lab - y_base).max()))

    # 저장된 GRID 예측
    saved = np.load(FINAL / "preds" / ("GRID_idfreq__g_d6_l10k_%d.npy" % fold))
    print("\n  저장 예측 GRID_idfreq__g_d6_l10k  단독 BSS %.1f (labels 기준) / %.1f (load_base 기준)"
          % (bss(saved, y_lab), bss(saved, y_base)))

    # 새로 학습 — seed_budget 과 같은 경로
    E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
    Xall = np.concatenate([np.asarray(X), E], axis=1)
    Xt = np.ascontiguousarray(Xall[tri])
    Xv = np.ascontiguousarray(Xall[va])
    yt = y_all[tri]
    del E, Xall
    print("\n  새 학습 %d피처 · %s행" % (Xt.shape[1], f"{len(Xt):,}"))
    acc = np.zeros(len(va))
    for sd in range(3):
        m = CatBoostRegressor(**{**CB_P, "depth": 6}, random_seed=sd,
                              task_type="GPU", devices="0", border_count=128)
        m.fit(Pool(Xt, yt.astype(np.float64)))
        p = np.clip(m.predict(Pool(Xv)), 1e-6, 1 - 1e-6)
        acc += p
        print("    seed%d 단독 %.1f  (누적평균 %.1f)"
              % (sd, bss(p, y_lab), bss(acc / (sd + 1), y_lab)))
        del m
    fresh = acc / 3

    print("\n" + "=" * 78)
    print("=" * 78)
    print("  저장 %.1f  vs  새학습 %.1f   차 %+.1f"
          % (bss(saved, y_lab), bss(fresh, y_lab), bss(fresh, y_lab) - bss(saved, y_lab)))
    print("  행단위 최대절대차 %.4e · 상관 %.6f"
          % (float(np.abs(saved - fresh).max()), float(np.corrcoef(saved, fresh)[0, 1])))

    # 블렌드 비교
    ft = np.load(FINAL / "preds" / ("S1_base__ft_%d.npy" % fold))
    ml = np.load(FINAL / "preds" / ("S1_base__mlp_%d.npy" % fold))
    RIDGE = 0.02

    def blend(cb):
        M = np.column_stack([cb, ft, ml])
        r = y_lab.mean()
        D = M - r
        Q = D.T @ D / len(y_lab)
        Aa = D.T @ (y_lab - r) / len(y_lab)
        Q = Q + RIDGE * np.trace(Q) / len(Q) * np.eye(len(Q))
        w = np.linalg.solve(Q, Aa)
        return bss(np.clip(r + (M - r) @ w, 1e-6, 1 - 1e-6), y_lab), w

    for nm, cb in (("저장", saved), ("새학습", fresh)):
        b, w = blend(cb)
        print("  %-6s 블렌드 %.1f   w = %s" % (nm, b, np.array2string(w, precision=3)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
