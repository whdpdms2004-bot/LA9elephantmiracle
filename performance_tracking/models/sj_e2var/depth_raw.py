# -*- coding: utf-8 -*-
"""깊이 순위를 **오염되지 않은 계기**로 다시 매긴다.

## 왜 다시 재나

`run_family` 가 저장하는 예측은 `calib(acc/seeds, yv)` 의 출력이다
(`run_family.py:343`). `run_arm.calib` 는 **평가 폴드 라벨 `yv` 로**

1. 로짓을 평가 폴드 기저율로 재중심화 (`c1 = log(r/(1-r))`, `r = yv.mean()`)
2. 스케일 `k` 를 BSS 최대화하도록 선택

**배포 시에는 테스트 기저율을 모른다.** 실측 격차가 크다 — 같은 `d6_l10k` 가
저장 904.1 / 원시 851.3 (ρ 0.999996, 최대차 0.017). 예측 분산이 라벨 분산의
0.86% 뿐이라 미세한 재중심화가 BSS 를 크게 움직인다.

모든 arm 이 같은 처리를 받았으므로 **상대 순위는 살아남을 가능성이 높고**
리더보드도 d6>d5 를 확인해줬다 (예측 +4.2 -> 실측 +3.68). 그래도
`d7`·`d8` 기각이 계기 탓인지 아닌지는 원시로 확인해야 한다.

## 이 계기

- cb 를 **원시로** 학습한다 (calib 없음)
- 결합은 **월전방분할**로만 적합한다 (월3~6 -> 월7~10, 역방향도)
- 스케일은 blend `w` 가 흡수한다 — 별도 보정을 하지 않는다
- val2022 는 비하락 관문으로만 쓴다

    python depth_raw.py --depths 5,6,7,8
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
sys.path.insert(0, str(ROOT / "performance_tracking" / "tools"))
sys.path.insert(0, str(FINAL / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FIT_M = (3, 4, 5, 6)
STEP = 0.02


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def simplex(k, step=STEP):
    if k == 1:
        return [np.array([1.0])]
    g = np.arange(0.0, 1.0 + 1e-9, step)
    out = []
    for c in itertools.product(g, repeat=k - 1):
        s = sum(c)
        if s <= 1.0 + 1e-9:
            out.append(np.array(list(c) + [1.0 - s]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="5,6,7,8")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]

    from common import load_labels
    from run_arm import CB_P, load_base
    import atoms as A
    from catboost import CatBoostRegressor, Pool

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]

    raw = {}
    for fold in (2024, 2022):
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        Xall = np.concatenate([np.asarray(X), E], axis=1)
        Xt = np.ascontiguousarray(Xall[tri])
        Xv = np.ascontiguousarray(Xall[va])
        yt = y_all[tri]
        del E, Xall
        for d in depths:
            acc = np.zeros(len(va))
            t0 = time.time()
            for sd in range(a.seeds):
                m = CatBoostRegressor(**{**CB_P, "depth": d}, random_seed=sd,
                                      task_type="GPU", devices="0", border_count=128)
                m.fit(Pool(Xt, yt.astype(np.float64)))
                acc += np.clip(m.predict(Pool(Xv)), 1e-6, 1 - 1e-6)
                del m
            raw[(fold, d)] = acc / a.seeds
            np.save(FINAL / "preds" / f"RAW_d{d}_{fold}.npy", raw[(fold, d)])
            print("  fold%d d%d  %.0f초" % (fold, d, time.time() - t0), flush=True)
        del Xt, Xv

    ft = {f: np.load(FINAL / "preds" / f"S1_base__ft_{f}.npy") for f in (2024, 2022)}
    ml = {f: np.load(FINAL / "preds" / f"S1_base__mlp_{f}.npy") for f in (2024, 2022)}
    LB = {f: load_labels(f) for f in (2024, 2022)}

    print("\n" + "=" * 92)
    print("원시 예측 · 월전방분할 결합 (합=1·비음수) — calib 없음")
    print("=" * 92)
    print("%-8s %10s %12s %12s   %12s   %s"
          % ("depth", "cb단독", "2024 정방향", "2024 역방향", "2022 정방향", "w (cb/ft/mlp)"))
    print("-" * 92)

    cand = simplex(3)
    base = {}
    for d in depths:
        row = []
        for f in (2024, 2022):
            y = LB[f]["y"].to_numpy(np.float64)
            m_ = LB[f]["game_month"].to_numpy()
            M = np.column_stack([raw[(f, d)], ft[f], ml[f]])
            f1 = np.isin(m_, FIT_M)
            for fitm, evm in ((f1, ~f1), (~f1, f1)):
                rf, re_ = y[fitm].mean(), y[evm].mean()
                sc = [bss(np.clip(rf + (M[fitm] - rf) @ w, 1e-6, 1 - 1e-6), y[fitm])
                      for w in cand]
                w = cand[int(np.argmax(sc))]
                row.append((bss(np.clip(re_ + (M[evm] - re_) @ w, 1e-6, 1 - 1e-6),
                                y[evm]), w))
        solo = bss(raw[(2024, d)], LB[2024]["y"].to_numpy(np.float64))
        if not base:
            base = {"a": row[0][0], "b": row[1][0], "c": row[2][0]}
        print("%-8d %10.1f %12.1f %12.1f   %12.1f   %s"
              % (d, solo, row[0][0], row[1][0], row[2][0],
                 np.array2string(row[0][1], precision=2)))
    print("\n(2022 는 비하락 관문. 2024 정·역 양방향이 주판정)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
