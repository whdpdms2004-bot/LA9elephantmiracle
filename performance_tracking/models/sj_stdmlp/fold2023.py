# -*- coding: utf-8 -*-
"""[중재] fold 2023 예측을 만들어 **두 계기의 불일치를 판정**한다.

## 무엇이 갈렸나

같은 팀 멤버 후보를 두 계기가 정반대로 판정한다.

    월전방분할 (시즌 **내부**: 월3~6 <-> 월7~10)   hw 가중 0.00, 이득 없음
    §31 규약   (시즌 **간**: fit2022 -> eval2024)   hw 가중 0.05~0.10, +3.5

배포는 시즌 간 전이(2019~2024 학습 -> 2025 예측)이므로 §31 쪽이 구조적으로 가깝다.
그러나 §31 의 fit 폴드인 2022 는 `game_type` **구조 단절 이전**이다
(F 성공률 0.7087 vs 2024 의 0.4593). 그래서 그 답도 못 믿는다.

## 중재자 — fold 2023

`run_arm.py` 가 이미 적어뒀다.

> fold 2023 은 §28.2 규칙3 에 따라 **판정 수치로 싣지 않는다.** 다만 결합 가중치를
> 적합하는 폴드로는 쓴다 — 단절 **이후**라 2024 와 같은 레짐이고 인접 시즌이라
> 배포(최근 과거로 적합 -> 다음 시즌에 적용)와 **동형**이다.

즉 `fit(2023) -> eval(2024)` 가 배포와 가장 닮은 전이다. 이걸로 중재한다.

    fit(2022) -> eval(2024)   단절 이전으로 적합. §31 이 지정한 것
    fit(2023) -> eval(2024)   단절 이후 · 인접 시즌. **배포와 동형**  <- 중재자

## 만드는 것

    cb   E2_var_cb2_a0.15_2023   depth6 · id_freq · 시드분산 상보가중 (배포 구성)
    ft   FTX_q64_168_2023        원시 (calib 없음)
    mlp  PREP_std_176_2023       로버스트 z 점수 + id_freq

팀원 멤버(sj3way_nv · hw · ye)는 각자 러너로 따로 만든다.

    python fold2023.py
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

FOLD = 2023
DEPTH = 6
WVAR = 0.15
KFOLD = 3


def log(m):
    print(m, flush=True)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ft-seeds", type=int, default=2)
    a = ap.parse_args()

    import importlib.util as iu
    sp = iu.spec_from_file_location("pt_common", PT / "tools" / "common.py")
    mod = iu.module_from_spec(sp)
    sp.loader.exec_module(mod)
    load_labels = mod.load_labels

    from run_arm import CB_P, fit_torch, load_base
    from prep_mlp import apply_prep, make_prep
    from catboost import CatBoostRegressor, Pool
    import atoms as A

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]
    y = load_labels(FOLD)["y"].to_numpy(np.float64)

    tri = np.where(season <= FOLD - 1)[0]
    va = np.where(season == FOLD)[0]
    E, en = A.build(X, names, season <= FOLD - 1, FOLD, ["id_freq"])
    X176 = np.concatenate([np.asarray(X), E], axis=1)
    del E
    Xt = np.ascontiguousarray(X176[tri])
    Xv = np.ascontiguousarray(X176[va])
    yt = y_all[tri].astype(np.float64)
    st = season[tri]
    log("=" * 76)
    log("[중재] fold%d — 학습 %s행 · 검증 %s행 · %d피처"
        % (FOLD, f"{len(tri):,}", f"{len(va):,}", X176.shape[1]))
    log("=" * 76)

    # ── cb — 배포 구성 그대로 (시드분산 상보 표본가중) ─────────────────────
    fp = FINAL / "preds" / ("E2_var_cb2_a0.15_%d.npy" % FOLD)
    if not fp.exists():
        t0 = time.time()
        stack = np.full((3, len(yt)), np.nan)
        for s_ in sorted(set(st.tolist()))[-KFOLD:]:
            trm, vam = st < s_, st == s_
            if trm.sum() < 50000:
                continue
            for sd in range(3):
                m = CatBoostRegressor(**dict(CB_P, depth=DEPTH), random_seed=sd,
                                      task_type="GPU", devices="0", border_count=128)
                m.fit(Pool(np.ascontiguousarray(Xt[trm]), yt[trm]))
                stack[sd, vam] = np.clip(
                    m.predict(Pool(np.ascontiguousarray(Xt[vam]))), 1e-6, 1 - 1e-6)
                del m
        have = ~np.isnan(stack[0])
        sdv = stack[:, have].std(axis=0)
        w = np.ones(len(yt))
        w[have] = np.clip(1.0 + WVAR * (sdv / sdv.mean() - 1.0), 0.1, 10.0)
        acc = np.zeros(len(va))
        for sd in range(a.seeds):
            m = CatBoostRegressor(**dict(CB_P, depth=DEPTH), random_seed=sd,
                                  task_type="GPU", devices="0", border_count=128)
            m.fit(Pool(Xt, yt, weight=w))
            acc += np.clip(m.predict(Pool(Xv)), 1e-6, 1 - 1e-6)
            del m
        np.save(fp, acc / a.seeds)
        log("  cb   %.0f초  단독 %.1f" % (time.time() - t0, bss(acc / a.seeds, y)))
    else:
        log("  cb   (이미 있음, 단독 %.1f)" % bss(np.load(fp), y))

    # ── mlp — 로버스트 z 점수 + id_freq ────────────────────────────────────
    fp = FINAL / "preds" / ("PREP_std_176_%d.npy" % FOLD)
    if not fp.exists():
        t0 = time.time()
        pr = make_prep(Xt, "std")
        Zt, Zv = apply_prep(Xt, pr, True), apply_prep(Xv, pr, True)
        acc = np.zeros(len(va))
        for sd in range(a.seeds):
            acc += np.clip(fit_torch("mlp", Zt, y_all[tri], Zv, sd), 1e-6, 1 - 1e-6)
        np.save(fp, acc / a.seeds)
        log("  mlp  %.0f초  단독 %.1f" % (time.time() - t0, bss(acc / a.seeds, y)))
        del Zt, Zv
    else:
        log("  mlp  (이미 있음, 단독 %.1f)" % bss(np.load(fp), y))

    # ── ft — 원시 (calib 없음), 168열 분위수 순위 ──────────────────────────
    fp = FINAL / "preds" / ("FTX_q64_168_%d.npy" % FOLD)
    if not fp.exists():
        t0 = time.time()
        src = np.asarray(X)
        Ft = np.ascontiguousarray(src[tri])
        Fv = np.ascontiguousarray(src[va])
        pr = make_prep(Ft, "q64")
        Zt, Zv = apply_prep(Ft, pr, False), apply_prep(Fv, pr, False)
        del Ft, Fv
        acc = np.zeros(len(va))
        for sd in range(a.ft_seeds):
            ts = time.time()
            acc += np.clip(fit_torch("ft", Zt, y_all[tri], Zv, sd), 1e-6, 1 - 1e-6)
            log("    ft seed%d %.0f초" % (sd, time.time() - ts))
        np.save(fp, acc / a.ft_seeds)
        log("  ft   %.0f초  단독 %.1f" % (time.time() - t0, bss(acc / a.ft_seeds, y)))
        del Zt, Zv
    else:
        log("  ft   (이미 있음, 단독 %.1f)" % bss(np.load(fp), y))

    log("\n완료 — fold2023 cb·ft·mlp 준비됨")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
