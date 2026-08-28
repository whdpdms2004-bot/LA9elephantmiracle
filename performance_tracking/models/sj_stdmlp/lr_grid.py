# -*- coding: utf-8 -*-
"""[하이퍼 재탐색 2] `lr x iterations` — 동결값이 아직 X168 기준이다.

## 왜 이것이 남았나

제출 5회 실측으로 개정된 법칙:

    CB 하이퍼 (depth 5->6)   val +4.0 -> Public +3.71   전이율 **0.93**
    피처 (id_freq)           val +21.0 -> +1.33         0.06
    표현 (std_mlp)           val +17   -> +1.68         0.10
    결합가중                  잃을 수는 있어도 벌 수 없다

**하이퍼파라미터만 전이된다.** 그런데 §29.2 의 조건부 재탐색에서 나는
`depth x l2` 만 봤고 `lr x iterations` 는 동결값(0.02 x 3000)을 그대로 뒀다.
그 동결도 **X168 입력 기준**이고, `id_freq` 로 176 이 되면서 조건이 발동한 상태다.
`depth` 가 5→6 으로 움직였으니 `lr`·`iterations` 도 움직였을 수 있다.

## 추론 예산이 축을 제한한다

`iterations` 는 추론 시간에 **선형**이다. 현재 서버 399초/600초이고 cb 가 그 중
약 33%(132초)다. `iterations` 를 2배로 하면 총 531초 — 여유가 급격히 준다.

그래서 격자를 **비용 고정 축**과 **비용 절감 축**으로 나눈다.

    비용 동일 (it=3000)   lr 0.010 / 0.015 / 0.020(현행) / 0.030 / 0.040
    비용 절감             lr 0.030 x it 2000 (0.67x) · lr 0.050 x it 1500 (0.50x)

용량(lr x it)이 커지는 쪽만 보는 게 아니라 **같은 값을 더 싸게 내는 조합**도 본다 —
싸지면 그 여유로 시드를 늘릴 수 있다 ([축5] 재검증에서 8시드가 +0.6 인데
예산 때문에 못 썼다).

## 계기

원시 예측 · 배포 순서 재현(모델별 `apply_calibration` 후 확률결합) · 가중 고정.
cb 는 배포 구성 그대로 (depth 6 · l2 10000 · id_freq · 시드분산 상보가중 0.15).
판정은 val2024 주판정 + val2022 비하락.

    python lr_grid.py
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

DEPTH = 6
WVAR = 0.15
KFOLD = 3

# (이름, lr, iterations)
ARMS = [("lr010_it3000", 0.010, 3000),
        ("lr015_it3000", 0.015, 3000),
        ("lr020_it3000", 0.020, 3000),      # 현행 동결값
        ("lr030_it3000", 0.030, 3000),
        ("lr040_it3000", 0.040, 3000),
        ("lr030_it2000", 0.030, 2000),
        ("lr050_it1500", 0.050, 1500)]


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


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
    ap.add_argument("--seeds", type=int, default=3)
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
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        Xall = np.concatenate([np.asarray(X), E], axis=1)
        del E
        Xt = np.ascontiguousarray(Xall[tri])
        Xv = np.ascontiguousarray(Xall[va])
        del Xall
        yt = y_all[tri].astype(np.float64)
        st = season[tri]

        # 시드분산 상보 표본가중 — 배포 구성과 같게. 폴드당 한 번만 만든다
        wp = FINAL / "preds" / ("OOF_cb_v3_%d.npy" % fold)
        if wp.exists():
            stack = np.load(wp)
        else:
            stack = np.full((3, len(yt)), np.nan)
            for s_ in sorted(set(st.tolist()))[-KFOLD:]:
                trm, vam = st < s_, st == s_
                if trm.sum() < 50000:
                    continue
                for sd in range(3):
                    m = CatBoostRegressor(**dict(CB_P, depth=DEPTH), random_seed=sd,
                                          task_type="GPU", devices="0",
                                          border_count=128)
                    m.fit(Pool(np.ascontiguousarray(Xt[trm]), yt[trm]))
                    stack[sd, vam] = np.clip(
                        m.predict(Pool(np.ascontiguousarray(Xt[vam]))),
                        1e-6, 1 - 1e-6)
                    del m
            np.save(wp, stack)
        have = ~np.isnan(stack[0])
        sdv = stack[:, have].std(axis=0)
        w = np.ones(len(yt))
        w[have] = np.clip(1.0 + WVAR * (sdv / sdv.mean() - 1.0), 0.1, 10.0)

        for nm, lr, it in ARMS:
            fp = FINAL / "preds" / ("LRG_%s_%d.npy" % (nm, fold))
            if fp.exists():
                print("  fold%d %-14s (이미 있음, 단독 %.1f)"
                      % (fold, nm, bss(np.load(fp), y)), flush=True)
                continue
            t0 = time.time()
            acc = np.zeros(len(va))
            for sd in range(a.seeds):
                m = CatBoostRegressor(**dict(CB_P, depth=DEPTH, learning_rate=lr,
                                             iterations=it),
                                      random_seed=sd, task_type="GPU",
                                      devices="0", border_count=128)
                m.fit(Pool(Xt, yt, weight=w))
                acc += np.clip(m.predict(Pool(Xv)), 1e-6, 1 - 1e-6)
                del m
            np.save(fp, acc / a.seeds)
            print("  fold%d %-14s lr%.3f it%d  %4.0f초  단독 %.1f"
                  % (fold, nm, lr, it, time.time() - t0, bss(acc / a.seeds, y)),
                  flush=True)
        del Xt, Xv

    # ── 판정 ────────────────────────────────────────────────────────────────
    P = FINAL / "preds"
    par = json.loads(zipfile.ZipFile(a.zip).read("model/cw/model/params.json"))
    W = np.array([par["blend_w_cb"], par["blend_w_ft"], par["blend_w_mlp"]])
    RATE = 0.93          # 하이퍼파라미터 전이율 (제출 5회 실측)
    print("\n" + "=" * 96)
    print("[하이퍼 재탐색 2] lr x iterations — 원시 예측 · 배포 순서 · 가중 고정")
    print("=" * 96)
    print("%-14s %10s %10s %10s   %8s %8s %8s   %s"
          % ("구성", "cb단독24", "val2024", "val2022", "Δ24", "Δ22", "추론배수",
             "Public 예상"))
    print("-" * 96)

    def sc(tag):
        out = []
        for fold in (2024, 2022):
            y = load_labels(fold)["y"].to_numpy(np.float64)
            M = np.column_stack([
                cal(np.load(P / ("LRG_%s_%d.npy" % (tag, fold))), par["model_cb"]),
                cal(np.load(P / ("FTX_q64_168_%d.npy" % fold)), par["model_ft"]),
                cal(np.load(P / ("PREP_std_176_%d.npy" % fold)), par["model_mlp"])])
            r = y.mean()
            out.append(bss(np.clip(r + (M - r) @ W, 1e-6, 1 - 1e-6), y))
        return out

    base = sc("lr020_it3000")
    y24 = load_labels(2024)["y"].to_numpy(np.float64)
    for nm, lr, it in ARMS:
        if not (P / ("LRG_%s_2024.npy" % nm)).exists():
            continue
        s = sc(nm)
        solo = bss(np.load(P / ("LRG_%s_2024.npy" % nm)), y24)
        mark = " <- 현행" if nm == "lr020_it3000" else ""
        gate = "" if s[1] >= base[1] - 0.5 else "  ★관문"
        print("%-14s %10.1f %10.1f %10.1f   %+8.1f %+8.1f %8.2fx   %.1f%s%s"
              % (nm, solo, s[0], s[1], s[0] - base[0], s[1] - base[1],
                 it / 3000.0, 1082.106 + (s[0] - base[0]) * RATE, mark, gate))
    print("\n(전이율 0.93 은 제출 5회 실측값이다. iterations 가 추론 시간에 선형이므로")
    print(" 배수 1.0 초과는 서버 399초/600초 예산을 갉는다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
