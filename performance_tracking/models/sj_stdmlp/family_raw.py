# -*- coding: utf-8 -*-
"""[축7 재검증] `단독 ~ rho(cb)` 선을 **원시 계기 + 다른 표현**으로 다시 본다.

## 왜 다시 하나

이번 캠페인 내내 판정 근거로 쓴 명제가 있다.

> 단독 성능 ~ rho(cb) 상관 = +0.994 (21멤버).
> "정확하면서 탈상관"은 선택지에 없다.

이 명제에 문제가 둘 있다.

1. **오염된 계기로 쟀다.** `run_arm.calib` 가 평가 폴드 라벨로 로짓 스케일을
   맞춘 예측을 썼다 (같은 d6 가 저장 904.1 / 원시 851.3).
2. **21멤버가 전부 같은 표현 위에 있었다.** 원시값(트리) 아니면 분위수 순위(NN).
   [축2] 가 반례를 만들었다 — `std` 표현의 MLP 는 단독이 낮은데도 가중 0.15 를
   받고 블렌드를 +12.3 올린다.

즉 그 선은 **모델 계열의 법칙이 아니라 표현의 법칙**일 수 있다.
같은 표현 안에서는 성립하고, 표현을 바꾸면 벗어난다.

## arm — 표현 × 계열 교차

    cb_raw176     원시값 + 트리        (기준, 이미 있음)
    lgb_raw176    원시값 + leaf-wise 트리   — 계열만 다르고 표현은 같다
    mlp_std176    z 점수 + NN          (이미 있음 — 선을 벗어난 것)
    mlp_q64_176   분위수순위 + NN       (이미 있음 — 선 위에 있는 것)
    lin_std176    z 점수 + **선형**      — 표현은 같고 계열이 극단적으로 다르다

`lin_std176` 이 핵심이다. 선형 모델은 단독 성능이 낮을 수밖에 없는데,
**크기를 보존한 표현 위의 선형** 이 트리와 근본적으로 다른 실수를 한다면
"단독이 낮아도 가중을 받는" 두 번째 사례가 된다.

## 판정

원시 예측만 쓴다. 각 멤버의 **단독 BSS** 와 **rho(cb)** 를 찍어 선 위에 있는지
보고, 배포 순서 결합에서 **가중을 받는지**로 판정한다.

    python family_raw.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
PT = ROOT / "performance_tracking"
sys.path.insert(0, str(ROOT / "cowork" / "cw" / "v17" / "src"))
sys.path.insert(0, str(FINAL / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FIT_M = (3, 4, 5, 6)
SEEDS = 3


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def fit_lgb(Xt, yt, Xv, seed, threads=10):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=31,
                          min_child_samples=200, reg_lambda=1000.0,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=seed, n_jobs=threads, verbose=-1)
    m.fit(Xt, yt)
    return np.clip(m.predict(Xv), 1e-6, 1 - 1e-6)


def fit_linear(Zt, yt, Zv, alpha=100.0):
    """z 점수 위의 능형 회귀. 시드 무관(닫힌 해)이라 1회면 된다."""
    n, d = Zt.shape
    Xc = np.hstack([Zt, np.ones((n, 1), np.float32)]).astype(np.float64)
    A = Xc.T @ Xc
    A[np.arange(d + 1), np.arange(d + 1)] += alpha
    w = np.linalg.solve(A, Xc.T @ yt.astype(np.float64))
    Zv2 = np.hstack([Zv, np.ones((len(Zv), 1), np.float32)]).astype(np.float64)
    return np.clip(Zv2 @ w, 1e-6, 1 - 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()

    import importlib.util as iu
    sp = iu.spec_from_file_location("pt_common", PT / "tools" / "common.py")
    mod = iu.module_from_spec(sp)
    sp.loader.exec_module(mod)
    load_labels = mod.load_labels

    from run_arm import load_base
    from prep_mlp import apply_prep, make_prep
    import atoms as A

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]

    for fold in (2024, 2022):
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        y = load_labels(fold)["y"].to_numpy(np.float64)
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        X176 = np.concatenate([np.asarray(X), E], axis=1)
        del E
        Xt = np.ascontiguousarray(X176[tri])
        Xv = np.ascontiguousarray(X176[va])
        del X176

        fp = FINAL / "preds" / ("FAMR_lgb_raw176_%d.npy" % fold)
        if not fp.exists():
            t0 = time.time()
            acc = np.zeros(len(va))
            for sd in range(a.seeds):
                acc += fit_lgb(Xt, y_all[tri], Xv, sd)
            np.save(fp, acc / a.seeds)
            print("  fold%d lgb_raw176  %.0f초  단독 %.1f"
                  % (fold, time.time() - t0, bss(acc / a.seeds, y)), flush=True)

        fp = FINAL / "preds" / ("FAMR_lin_std176_%d.npy" % fold)
        if not fp.exists():
            t0 = time.time()
            prep = make_prep(Xt, "std")
            Zt = apply_prep(Xt, prep, True)
            Zv = apply_prep(Xv, prep, True)
            p = fit_linear(Zt, y_all[tri], Zv)
            np.save(fp, p)
            print("  fold%d lin_std176  %.0f초  단독 %.1f"
                  % (fold, time.time() - t0, bss(p, y)), flush=True)
            del Zt, Zv
        del Xt, Xv

    # ── 판정 ────────────────────────────────────────────────────────────────
    P = FINAL / "preds"
    y24 = load_labels(2024)["y"].to_numpy(np.float64)
    cb = np.load(P / "E2_var_cb2_a0.15_2024.npy")
    members = [("cb_raw176 (트리·원시값)", "E2_var_cb2_a0.15"),
               ("lgb_raw176 (트리·원시값)", "FAMR_lgb_raw176"),
               ("mlp_q64_176 (NN·순위)", "PREP_q64_176"),
               ("mlp_std176 (NN·z점수)", "PREP_std_176"),
               ("lin_std176 (선형·z점수)", "FAMR_lin_std176")]
    print("\n" + "=" * 84)
    print("[축7 재검증] 표현 x 계열 — 원시 예측")
    print("=" * 84)
    print("%-28s %10s %10s   %s" % ("멤버", "단독2024", "rho(cb)", "표현"))
    print("-" * 84)
    for nm, f in members:
        fp = P / ("%s_2024.npy" % f)
        if not fp.exists():
            print("%-28s   (없음)" % nm)
            continue
        p = np.load(fp)
        print("%-28s %10.1f %10.4f" % (nm, bss(p, y24), np.corrcoef(p, cb)[0, 1]))
    print("\n(선 위에 있으면 단독이 높을수록 rho 도 높다.")
    print(" 표현이 다른 멤버가 그 선을 벗어나면 '선은 표현의 법칙' 이라는 뜻이다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
