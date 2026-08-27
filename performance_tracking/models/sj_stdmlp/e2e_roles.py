# -*- coding: utf-8 -*-
"""[항목 B / E2] 앙상블 결과를 되먹여 멤버가 역할을 나누게 한다.

## 문제 의식

지금은 멤버를 따로 학습해 놓고 나중에 섞는다. 각 멤버는 자기 혼자 잘하려고
학습됐지 앙상블 안에서 무슨 역할을 맡을지 모른 채 학습된다. 그 결과 21멤버가
전부 `단독 성능 ~ rho(cb)` 선(r=+0.994)에 붙었고, 그 선을 벗어난 유일한 방법이
다른 학습 데이터(F행만/최근시즌만) 였다.

표본가중은 학습 데이터를 바꾸는 가장 부드러운 방법이다. 앙상블이 못 맞히는 행에
가중을 실어 재학습하면 멤버가 스스로 상보적인 역할을 맡는다.

## 걸림돌과 해법

앙상블이 못 맞히는 곳을 알려면 **학습행에 대한 out-of-fold 예측**이 필요하다.
검증행 예측만으로는 학습행 가중을 만들 수 없다.

그래서 학습 구간(<= fold-1)의 마지막 K 시즌을 차례로 held-out 삼아 cb 의 OOF 를
만든다. 각 조각은 **그 이전 시즌으로만** 학습하므로 미래를 보지 않는다.
cb 는 시드당 20~60초라 감당된다 (ft 는 시드당 600~700초라 제외).

    1. 학습행 cb OOF 를 시간 K-fold 로 만든다
    2. 행별 오차 e = (y - p_oof)^2 로 표본가중을 만든다
    3. cb2 를 그 가중으로 학습              <- cb 가 못하는 곳을 맡아라
    4. cb + ft + mlp + cb2 를 월전방분할로 정직 판정

## 판정

**원시 예측만 쓴다.** `run_arm.calib` 는 평가 폴드 라벨로 로짓 스케일을 맞추므로
금지한다 (같은 d6 가 저장 904.1 / 원시 851.3 으로 갈렸다).

합=1 / 비음수. cb2 에 가중이 붙고 2024 양방향 + 2022 이 모두 비하락이어야 채택.
가중이 0 이면 역할 분담이 안 된다는 결론이고, 그것도 확정적이다.

    python e2e_roles.py --alpha 0.5,1.0,2.0
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
STEP = 0.05
SEEDS = 3
DEPTH = 6


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def simplex(k, step=STEP):
    """합=1 / 비음수 격자. 자유 적합은 정직 분할에서 -24.8 로 무너진다."""
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
    """월전방분할. 적합 월에서만 w 를 고르고 평가 월에서 잰다."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0.5,1.0,2.0",
                    help="가중 세기. w = clip(1 + alpha*(s/mean(s) - 1), 0.1, 10)")
    ap.add_argument("--kfold", type=int, default=3, help="OOF 로 쓸 마지막 시즌 수")
    # ── 가중 신호 ───────────────────────────────────────────────────────────
    # `err` 는 **잡음을 좇는다.** (y - p)^2 가 큰 행은 줄일 수 있는 오차가 큰 곳이
    # 아니라 **라벨이 동전던지기인 곳**이다 — 예측 분산이 라벨 분산의 0.86% 뿐이라
    # 오차의 거의 전부가 환원 불가능한 성분이다. 실측으로 cb2 단독이 -214 ~ -14238
    # 까지 무너졌다 (rho 는 0.40 ~ -0.33 으로 뚫렸지만 신호가 남지 않았다).
    #
    # `var` 는 **시드 간 분산**을 쓴다. 같은 데이터·다른 시드에서 예측이 흔들리는
    # 행은 라벨이 애매한 곳이 아니라 **모델이 결정을 못 내리는 곳**이다
    # (인식적 불확실성). 라벨을 보지 않으므로 잡음을 좇을 수 없다.
    ap.add_argument("--mode", default="err", choices=("err", "var"),
                    help="err = OOF 제곱오차 · var = 시드 간 예측 분산")
    ap.add_argument("--oof-seeds", type=int, default=3, help="var 모드의 OOF 시드 수")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alpha.split(",")]

    from common import load_labels
    from run_arm import CB_P, load_base
    import atoms as A
    from catboost import CatBoostRegressor, Pool

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]

    res = {}
    for fold in (2024, 2022):
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        Xall = np.concatenate([np.asarray(X), E], axis=1)
        Xt = np.ascontiguousarray(Xall[tri])
        Xv = np.ascontiguousarray(Xall[va])
        yt = y_all[tri].astype(np.float64)
        st = season[tri]
        del E, Xall

        # ── 1. 학습행 OOF ────────────────────────────────────────────────────
        # `var` 모드는 시드별 예측을 다 남겨야 분산을 잴 수 있다.
        nsd = a.oof_seeds if a.mode == "var" else 1
        oof_p = FINAL / "preds" / ("OOF_cb_%s%d.npy" % ("v%d_" % nsd if nsd > 1 else "", fold))
        if oof_p.exists():
            stack = np.load(oof_p)
        else:
            seas = sorted(set(st.tolist()))
            stack = np.full((nsd, len(yt)), np.nan)
            for s_ in seas[-a.kfold:]:
                trm = st < s_
                vam = st == s_
                if trm.sum() < 50000:
                    continue
                t0 = time.time()
                for sd in range(nsd):
                    m = CatBoostRegressor(**dict(CB_P, depth=DEPTH), random_seed=sd,
                                          task_type="GPU", devices="0", border_count=128)
                    m.fit(Pool(np.ascontiguousarray(Xt[trm]), yt[trm]))
                    stack[sd, vam] = np.clip(
                        m.predict(Pool(np.ascontiguousarray(Xt[vam]))), 1e-6, 1 - 1e-6)
                    del m
                print("  fold%d OOF season%d  학습 %s행 · %d시드  %.0f초"
                      % (fold, s_, format(int(trm.sum()), ","), nsd, time.time() - t0),
                      flush=True)
            np.save(oof_p, stack)
        oof = np.nanmean(stack, axis=0)
        have = ~np.isnan(oof)
        sig = np.full(len(yt), np.nan)
        if a.mode == "err":
            sig[have] = (yt[have] - oof[have]) ** 2
        else:
            # 시드 간 표준편차. 라벨을 보지 않는다.
            sig[have] = stack[:, have].std(axis=0)
        me = float(np.nanmean(sig))
        print("  fold%d OOF 보유 %s행 (%.1f%%) · OOF BSS %.1f · 신호(%s) 평균 %.3e 최대 %.3e"
              % (fold, format(int(have.sum()), ","), 100 * have.mean(),
                 bss(oof[have], yt[have]), a.mode, me, float(np.nanmax(sig))), flush=True)

        # ── 2·3. 상보 가중 재학습 ────────────────────────────────────────────
        for al in alphas:
            fp = FINAL / "preds" / ("E2_%s_cb2_a%s_%d.npy" % (a.mode, al, fold))
            if fp.exists():
                res[(fold, al)] = np.load(fp)
                continue
            w = np.ones(len(yt))
            w[have] = np.clip(1.0 + al * (sig[have] / me - 1.0), 0.1, 10.0)
            acc = np.zeros(len(va))
            t0 = time.time()
            for sd in range(SEEDS):
                m = CatBoostRegressor(**dict(CB_P, depth=DEPTH), random_seed=sd,
                                      task_type="GPU", devices="0", border_count=128)
                m.fit(Pool(Xt, yt, weight=w))
                acc += np.clip(m.predict(Pool(Xv)), 1e-6, 1 - 1e-6)
                del m
            res[(fold, al)] = acc / SEEDS
            np.save(fp, res[(fold, al)])
            print("  fold%d cb2 alpha=%.1f  가중 [%.2f, %.2f]  %.0f초"
                  % (fold, al, w.min(), w.max(), time.time() - t0), flush=True)
        del Xt, Xv

    # ── 4. 판정 ─────────────────────────────────────────────────────────────
    LB = {f: load_labels(f) for f in (2024, 2022)}
    ft = {f: np.load(FINAL / "preds" / ("S1_base__ft_%d.npy" % f)) for f in (2024, 2022)}
    ml = {f: np.load(FINAL / "preds" / ("S1_base__mlp_%d.npy" % f)) for f in (2024, 2022)}
    cb = {f: np.load(FINAL / "preds" / ("RAW_d6_%d.npy" % f)) for f in (2024, 2022)}

    print("")
    print("=" * 98)
    print("[E2/%s] 상보 표본가중 재학습 — 원시 예측 · 월전방분할 (합=1 / 비음수)" % a.mode)
    print("=" * 98)
    print("%-22s %10s %11s %11s %11s   %s"
          % ("구성", "cb2단독", "2024정", "2024역", "2022정", "w"))
    print("-" * 98)

    y24 = LB[2024]["y"].to_numpy(np.float64)
    m24 = LB[2024]["game_month"].to_numpy()
    y22 = LB[2022]["y"].to_numpy(np.float64)
    m22 = LB[2022]["game_month"].to_numpy()

    b24, w24 = honest(np.column_stack([cb[2024], ft[2024], ml[2024]]), y24, m24)
    b22, _ = honest(np.column_stack([cb[2022], ft[2022], ml[2022]]), y22, m22, both=False)
    print("%-22s %10s %11.1f %11.1f %11.1f   %s"
          % ("현행 cb+ft+mlp", "-", b24[0], b24[1], b22[0],
             np.array2string(w24, precision=2)))
    for al in alphas:
        c24 = np.column_stack([cb[2024], ft[2024], ml[2024], res[(2024, al)]])
        c22 = np.column_stack([cb[2022], ft[2022], ml[2022], res[(2022, al)]])
        v24, w_ = honest(c24, y24, m24)
        v22, _ = honest(c22, y22, m22, both=False)
        rho = float(np.corrcoef(res[(2024, al)], cb[2024])[0, 1])
        print("%-22s %10.1f %11.1f %11.1f %11.1f   %s   d %+.1f/%+.1f/%+.1f  rho(cb) %.4f"
              % ("+cb2 alpha=%.1f" % al, bss(res[(2024, al)], y24),
                 v24[0], v24[1], v22[0], np.array2string(w_, precision=2),
                 v24[0] - b24[0], v24[1] - b24[1], v22[0] - b22[0], rho))
    print("")
    print("(cb2 에 가중이 붙고 2024 양방향 + 2022 이 모두 비하락이어야 채택)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
