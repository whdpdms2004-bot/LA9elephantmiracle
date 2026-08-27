# -*- coding: utf-8 -*-
"""[축2] prep 변환 축 — **168 을 NN 이 어떻게 보느냐**가 통째로 미탐색이다 (§30.2).

## 왜 지금 다시 보나

원래 이 축을 쳐낸 이유는 "`mlp` 를 블렌드에서 빼는 게 이득(-7.2)이라 대상 소멸"
이었다. 그런데 두 가지가 바뀌었다.

1. **원시 계기로 재보니 `mlp` 가중이 depth 5~8 전부에서 0.00 이다.**
   보정된 계기에서 0.05~0.08 로 살아 보인 것은 `run_arm.calib` 가 평가 라벨로
   스케일을 맞춰준 덕이었다. 지금 `mlp` 는 **완전히 죽어 있다.**
2. **`id_freq` 8열을 `mlp` 는 아직 못 본다.** cb 에만 줬다 (176열).
   cb 단독을 크게 올린 피처인데 NN 쪽은 168 그대로다.

그래서 이 축의 질문은 **"전처리나 입력을 바꾸면 mlp 가 살아나는가"** 다.
살아나지 않으면 `mlp` 를 빼는 편이 낫고, 그것도 확정적인 답이다.

## 축 두 개의 교차

**전처리** (`dl.make_prep` 은 열당 분위수 64경계 -> 순위를 [-1,1] 로)

    q64      기준 (현행)
    q256     분해능 4배 — 순위 변환의 계단이 곱다
    gauss    순위를 **역정규 CDF** 로 — NN 입력을 정규분포로 만드는 표준 변환
    std      분위수 대신 **로버스트 z 점수** (중앙값/IQR, ±4 클립)

**입력**

    168      기준 (+ 결측마스크 -> 336)
    176      `id_freq` 8열 포함 (+ 마스크 -> 352)

## 판정

**원시 예측만 쓴다** (`calib` 금지). 월전방분할 양방향 + val2022 관문.
기준은 `cb(d6) + ft` 2멤버이고, `mlp` 후보가 **가중을 받아 이득을 내야** 채택이다.

    python prep_mlp.py --arms q64:168,q64:176,q256:176,gauss:176,std:176
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
sys.path.insert(0, str(ROOT / "cowork" / "cw" / "v17" / "src"))
sys.path.insert(0, str(FINAL / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FIT_M = (3, 4, 5, 6)
STEP = 0.05
SEEDS = 3


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


def honest(cols, y, mth, both=True):
    f1 = np.isin(mth, FIT_M)
    cand = simplex(cols.shape[1])
    out, w0 = [], None
    dirs = ((f1, ~f1), (~f1, f1)) if both else ((f1, ~f1),)
    for fitm, evm in dirs:
        rf, re_ = y[fitm].mean(), y[evm].mean()
        sc = [bss(np.clip(rf + (cols[fitm] - rf) @ w, 1e-6, 1 - 1e-6), y[fitm])
              for w in cand]
        w = cand[int(np.argmax(sc))]
        if w0 is None:
            w0 = w
        out.append(bss(np.clip(re_ + (cols[evm] - re_) @ w, 1e-6, 1 - 1e-6), y[evm]))
    return out, w0


# ── 전처리 변환 ─────────────────────────────────────────────────────────────
def make_prep(Xtr, kind):
    """학습 데이터에서만 상수를 뽑는다. 행 독립성·시간 인과에 영향 없음."""
    d = Xtr.shape[1]
    if kind == "std":
        med, iqr = np.empty(d), np.empty(d)
        for j in range(d):
            col = Xtr[:, j]
            ok = np.isfinite(col)
            v = col[ok] if ok.sum() > 100 else np.array([0.0])
            med[j] = np.median(v)
            q1, q3 = np.percentile(v, [25, 75])
            iqr[j] = max(q3 - q1, 1e-6)
        return {"kind": kind, "med": med, "iqr": iqr}
    n_q = 256 if kind == "q256" else 64
    qs = np.linspace(0, 1, n_q + 1)[1:-1]
    bnds = []
    for j in range(d):
        col = Xtr[:, j]
        ok = np.isfinite(col)
        bnds.append(np.unique(np.quantile(col[ok], qs)) if ok.sum() > 100
                    else np.array([0.0]))
    return {"kind": kind, "bnds": bnds}


def apply_prep(X, prep, with_mask=True):
    n, d = X.shape
    Z = np.empty((n, d * 2 if with_mask else d), np.float32)
    if prep["kind"] == "std":
        for j in range(d):
            col = X[:, j]
            ok = np.isfinite(col)
            z = (np.where(ok, col, prep["med"][j]) - prep["med"][j]) / prep["iqr"][j]
            Z[:, j] = np.clip(z, -4.0, 4.0).astype(np.float32)
            if with_mask:
                Z[:, d + j] = ok.astype(np.float32)
        return Z
    for j in range(d):
        col = X[:, j]
        ok = np.isfinite(col)
        b = prep["bnds"][j]
        r = np.searchsorted(b, np.where(ok, col, 0.0)).astype(np.float32) / max(len(b), 1)
        if prep["kind"] == "gauss":
            # 순위를 역정규 CDF 로. 극단값이 무한대로 가지 않게 안쪽으로 민다.
            u = np.clip(r, 1e-4, 1 - 1e-4)
            from scipy.special import ndtri
            Z[:, j] = np.where(ok, ndtri(u), 0.0).astype(np.float32)
        else:
            Z[:, j] = np.where(ok, r * 2.0 - 1.0, 0.0)
        if with_mask:
            Z[:, d + j] = ok.astype(np.float32)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="q64:168,q64:176,q256:176,gauss:176,std:176")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    arms = []
    for tok in a.arms.split(","):
        k, cols = tok.split(":")
        arms.append((k.strip(), int(cols)))

    # ★ `cw/v17/src/common.py` 가 `performance_tracking/tools/common.py` 를 가린다
    # (둘 다 sys.path 에 있고 cw 쪽이 앞선다). 파일 경로로 명시해 불러온다.
    import importlib.util as _iu
    _sp = _iu.spec_from_file_location(
        "pt_common", ROOT / "performance_tracking" / "tools" / "common.py")
    _m = _iu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    load_labels = _m.load_labels
    from run_arm import fit_torch, load_base
    import atoms as A

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]

    got = {}
    for fold in (2024, 2022):
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        X176 = np.concatenate([np.asarray(X), E], axis=1)
        del E
        for kind, ncol in arms:
            fp = FINAL / "preds" / ("PREP_%s_%d_%d.npy" % (kind, ncol, fold))
            if fp.exists():
                got[(fold, kind, ncol)] = np.load(fp)
                continue
            src = X176 if ncol == 176 else np.asarray(X)
            Xt_raw = np.ascontiguousarray(src[tri])
            Xv_raw = np.ascontiguousarray(src[va])
            t0 = time.time()
            prep = make_prep(Xt_raw, kind)
            Xt = apply_prep(Xt_raw, prep, True)
            Xv = apply_prep(Xv_raw, prep, True)
            del Xt_raw, Xv_raw
            acc = np.zeros(len(va))
            for sd in range(a.seeds):
                p = fit_torch("mlp", Xt, y_all[tri], Xv, sd)
                acc += np.clip(p, 1e-6, 1 - 1e-6)
            got[(fold, kind, ncol)] = acc / a.seeds
            np.save(fp, got[(fold, kind, ncol)])
            print("  fold%d %-6s %d열 -> 입력 %d  %.0f초  단독 %.1f"
                  % (fold, kind, ncol, Xt.shape[1], time.time() - t0,
                     bss(got[(fold, kind, ncol)],
                         load_labels(fold)["y"].to_numpy(np.float64))), flush=True)
            del Xt, Xv
        del X176

    # ── 판정 ────────────────────────────────────────────────────────────────
    LB = {f: load_labels(f) for f in (2024, 2022)}
    ft = {f: np.load(FINAL / "preds" / ("S1_base__ft_%d.npy" % f)) for f in (2024, 2022)}
    ml = {f: np.load(FINAL / "preds" / ("S1_base__mlp_%d.npy" % f)) for f in (2024, 2022)}
    cb = {f: np.load(FINAL / "preds" / ("E2_var_cb2_a0.15_%d.npy" % f))
          for f in (2024, 2022)}

    y24 = LB[2024]["y"].to_numpy(np.float64)
    m24 = LB[2024]["game_month"].to_numpy()
    y22 = LB[2022]["y"].to_numpy(np.float64)
    m22 = LB[2022]["game_month"].to_numpy()

    print("")
    print("=" * 100)
    print("[축2] prep 변환 x id_freq — 원시 예측 · 월전방분할 (합=1 / 비음수)")
    print("  기준 멤버 = cb2(E2/var a=0.15) + ft.  mlp 후보가 가중을 받아야 의미가 있다")
    print("=" * 100)
    print("%-18s %10s %11s %11s %11s   %s"
          % ("mlp 구성", "단독2024", "2024정", "2024역", "2022정", "w (cb/ft/mlp)"))
    print("-" * 100)

    b24, _ = honest(np.column_stack([cb[2024], ft[2024]]), y24, m24)
    b22, _ = honest(np.column_stack([cb[2022], ft[2022]]), y22, m22, both=False)
    print("%-18s %10s %11.1f %11.1f %11.1f   %s"
          % ("없음 (2멤버)", "-", b24[0], b24[1], b22[0], "[0.7 0.3]"))

    cands = [("현행 mlp", ml)] + [("%s:%d" % (k, c), None) for k, c in arms]
    for label, src in cands:
        if src is None:
            k, c = label.split(":")
            c = int(c)
            p24, p22 = got[(2024, k, c)], got[(2022, k, c)]
        else:
            p24, p22 = src[2024], src[2022]
        v24, w_ = honest(np.column_stack([cb[2024], ft[2024], p24]), y24, m24)
        v22, _ = honest(np.column_stack([cb[2022], ft[2022], p22]), y22, m22, both=False)
        print("%-18s %10.1f %11.1f %11.1f %11.1f   %s   d %+.1f/%+.1f/%+.1f"
              % (label, bss(p24, y24), v24[0], v24[1], v22[0],
                 np.array2string(w_, precision=2),
                 v24[0] - b24[0], v24[1] - b24[1], v22[0] - b22[0]))
    print("")
    print("(mlp 가중이 0 이면 그 구성은 아무것도 더하지 않는다는 뜻이다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
