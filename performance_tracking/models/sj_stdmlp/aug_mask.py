# -*- coding: utf-8 -*-
"""[마스킹 증강] 학습행에 **이력을 지운 사본**을 덧붙여 학습한다.

## 출처와 동기

`cowork/sj/three_way/NEXT_PLAN.md` 가 way 분해 작업 중에 남긴 실측:

> **원본을 덮어쓰면 손해, 원본 + 마스킹 사본을 함께 학습하면 이득.**
> reverse 에서 clip995 증강이 f23 853.0 → 868.7

way 분해 자체는 S3 게이트에서 닫았지만 **이 관찰은 way 와 무관하다.**
학습 데이터를 바꾸는 축이고, 이번 캠페인이 "다양성을 만드는 유일한 레버는
다른 학습 데이터" 라고 확인한 바로 그 축이다.

## 왜 지금 이게 맞는 자리인가

`id_freq` 가 이 캠페인 최초의 큰 이득이었고, 그 실체는
**val2024 행의 19.86% 가 학습에 없던 투수**라는 것이었다 (새 ID 81명).
그런데 `id_freq` 는 그 상황을 **표시**만 한다 — 미출현 플래그를 세울 뿐,
"이력이 없을 때 어떻게 예측할지" 를 학습시키지는 않는다.

마스킹 증강이 정확히 그 자리다. `asof_pitcher_*` 16열을 지운 사본을 학습에 넣으면
모델이 **이력 없는 투수 체제**를 직접 배운다. 배포 시점에 그 체제가 20% 다.

## arm

    mask_p20    학습행의 20% 를 골라 asof_pitcher_* 를 NaN 으로 만든 사본을 덧붙임
    mask_p40    40%
    mask_pb20   asof_pitcher_* + asof_batter_* 둘 다, 20%

원본은 **그대로 둔다** (덮어쓰면 손해라는 것이 위 실측이다). 행 수가 1.2배·1.4배가 된다.

## 행 독립성

증강은 **학습에만** 쓰인다. 추론 코드는 한 글자도 안 바뀌므로
`predict(단독 행) == predict(전체)[i]` 가 그대로 유지된다.
`id_freq` 룩업도 원본 학습행에서만 만든다 (사본은 빈도표에 기여하지 않는다).

## 판정

원시 예측 · 배포 순서 재현. 현행 배포 cb(id_freq · d6 · 시드분산가중)가 기준이고
2024 주판정 + 2022 비하락.

    python aug_mask.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
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

RIDGE = 0.02
SEEDS = 3
DEPTH = 6
WVAR = 0.15
KFOLD = 3

# (이름, 마스킹할 접두, 사본 비율)
ARMS = [("mask_p20", ("asof_pitcher",), 0.20),
        ("mask_p40", ("asof_pitcher",), 0.40),
        ("mask_pb20", ("asof_pitcher", "asof_batter"), 0.20)]


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
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--zip", default=str(FINAL / "submit" / "submit_sj_stdmlp.zip"))
    a = ap.parse_args()

    import importlib.util as iu
    sp = iu.spec_from_file_location("pt_common", PT / "tools" / "common.py")
    mod = iu.module_from_spec(sp)
    sp.loader.exec_module(mod)
    load_labels = mod.load_labels

    from run_arm import CB_P, load_base
    from catboost import CatBoostRegressor, Pool
    import atoms as A

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]

    for fold in (2024, 2022):
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        y = load_labels(fold)["y"].to_numpy(np.float64)
        # `id_freq` 룩업은 **원본 학습행에서만** 만든다 — 사본은 빈도에 기여하지 않는다
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        Xall = np.concatenate([np.asarray(X), E], axis=1)
        del E
        Xt = np.ascontiguousarray(Xall[tri])
        Xv = np.ascontiguousarray(Xall[va])
        del Xall
        yt = y_all[tri].astype(np.float64)
        st = season[tri]

        # 시드분산 상보 표본가중 — 현행 배포 구성과 같게 (원본 행 기준으로 만든다)
        oofp = FINAL / "preds" / ("OOF_cb_v3_%d.npy" % fold)
        if oofp.exists():
            stack = np.load(oofp)
        else:
            stack = np.full((3, len(yt)), np.nan)
            for s_ in sorted(set(st.tolist()))[-KFOLD:]:
                trm = st < s_
                vam = st == s_
                if trm.sum() < 50000:
                    continue
                for sd in range(3):
                    m = CatBoostRegressor(**dict(CB_P, depth=DEPTH), random_seed=sd,
                                          task_type="GPU", devices="0", border_count=128)
                    m.fit(Pool(np.ascontiguousarray(Xt[trm]), yt[trm]))
                    stack[sd, vam] = np.clip(
                        m.predict(Pool(np.ascontiguousarray(Xt[vam]))), 1e-6, 1 - 1e-6)
                    del m
            np.save(oofp, stack)
        have = ~np.isnan(stack[0])
        sdv = stack[:, have].std(axis=0)
        w0 = np.ones(len(yt))
        w0[have] = np.clip(1.0 + WVAR * (sdv / sdv.mean() - 1.0), 0.1, 10.0)

        for nm, prefixes, frac in ARMS:
            fp = FINAL / "preds" / ("AUG_%s_%d.npy" % (nm, fold))
            if fp.exists():
                print("  fold%d %-10s (이미 있음, 단독 %.1f)"
                      % (fold, nm, bss(np.load(fp), y)), flush=True)
                continue
            t0 = time.time()
            cols = [i for i, c in enumerate(names)
                    if any(c.startswith(px) for px in prefixes)]
            rng = np.random.default_rng(0)
            pick = rng.choice(len(yt), int(frac * len(yt)), replace=False)
            Xc = Xt[pick].copy()
            Xc[:, cols] = np.nan                 # 이력을 지운다
            Xa = np.ascontiguousarray(np.vstack([Xt, Xc]))
            ya = np.concatenate([yt, yt[pick]])
            wa = np.concatenate([w0, w0[pick]])
            del Xc
            acc = np.zeros(len(va))
            for sd in range(a.seeds):
                m = CatBoostRegressor(**dict(CB_P, depth=DEPTH), random_seed=sd,
                                      task_type="GPU", devices="0", border_count=128)
                m.fit(Pool(Xa, ya, weight=wa))
                acc += np.clip(m.predict(Pool(Xv)), 1e-6, 1 - 1e-6)
                del m
            p = acc / a.seeds
            np.save(fp, p)
            print("  fold%d %-10s %d열 마스킹 · 학습 %s행 (원본 %s) %5.0f초  단독 %.1f"
                  % (fold, nm, len(cols), format(len(ya), ","), format(len(yt), ","),
                     time.time() - t0, bss(p, y)), flush=True)
            del Xa, ya, wa
        del Xt, Xv

    # ── 판정 ────────────────────────────────────────────────────────────────
    P = FINAL / "preds"
    par = json.loads(zipfile.ZipFile(a.zip).read("model/cw/model/params.json"))
    Y = {f: load_labels(f)["y"].to_numpy(np.float64) for f in (2024, 2022)}

    def sc(cbfile):
        ms = []
        for f in (2024, 2022):
            ms.append((np.column_stack([
                cal(np.load(P / ("%s_%d.npy" % (cbfile, f))), par["model_cb"]),
                cal(np.load(P / ("FTX_q64_168_%d.npy" % f)), par["model_ft"]),
                cal(np.load(P / ("PREP_std_176_%d.npy" % f)), par["model_mlp"])]), Y[f]))
        w = np.mean([fitw(M, y) for M, y in ms], axis=0)
        return [bss(np.clip(y.mean() + (M - y.mean()) @ w, 1e-6, 1 - 1e-6), y)
                for M, y in ms], w

    RATE = 3.706 / 4.4
    print("\n" + "=" * 96)
    print("[마스킹 증강] 원시 예측 · 배포 순서 재현")
    print("  기준 = 현행 배포 cb (증강 없음). ft·mlp 고정")
    print("=" * 96)
    print("%-16s %10s %10s %10s   %8s %8s   %s"
          % ("증강", "cb단독24", "val2024", "val2022", "Δ24", "Δ22", "Public 예상"))
    print("-" * 96)
    base, _ = sc("E2_var_cb2_a0.15")
    y24 = Y[2024]
    print("%-16s %10.1f %10.1f %10.1f   %8s %8s   1080.4"
          % ("없음 (기준)", bss(np.load(P / "E2_var_cb2_a0.15_2024.npy"), y24),
             base[0], base[1], "기준", "기준"))
    for nm, _, _ in ARMS:
        if not (P / ("AUG_%s_2024.npy" % nm)).exists():
            continue
        s, w = sc("AUG_%s" % nm)
        print("%-16s %10.1f %10.1f %10.1f   %+8.1f %+8.1f   %.1f"
              % (nm, bss(np.load(P / ("AUG_%s_2024.npy" % nm)), y24), s[0], s[1],
                 s[0] - base[0], s[1] - base[1],
                 1080.425 + (s[0] - base[0]) * RATE))
    print("\n(2024 주판정 + 2022 비하락. 증강은 학습에만 쓰이므로 추론 비용은 0 이다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
