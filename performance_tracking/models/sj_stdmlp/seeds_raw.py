# -*- coding: utf-8 -*-
"""[축5 재검증] 시드 수를 **원시 예측**으로 다시 판정한다.

## 왜 다시 하나

이번 캠페인에서 판정 근거로 쓴 것 중 아직 재검증하지 않은 것이 남았다.

> [축5] 시드 수 3 / 5 / 8 — 기각. 3시드로 충분하다.
> Δ(8−3) 이 val2024 −0.6 · val2022 −0.9 로 **두 폴드 모두 음수**다.

그 판정도 `run_arm.calib` 을 거친 예측으로 했다. 그 계기가 이번에 두 결론을
뒤집었다 — `cb_f_only`("3.42시그마 채택안" → val2022 −64.6)와 [축7]의 ρ 0.994 선.

**시드 평균은 분산을 줄이는 조작이고, `calib` 은 그 분산을 평가 라벨로 다시
맞춰준다.** 즉 보정이 시드 평균의 이득을 미리 먹어버렸을 수 있다.
원시로 재면 시드 수의 효과가 달라질 수 있다.

## 계기

- 원시 예측만 쓴다 (`calib` 금지)
- cb 는 **배포 구성** 그대로 (depth 6 · l2 10000 · id_freq · 시드분산 상보가중 0.15)
- 판정은 **배포 순서 재현** — 모델별 `apply_calibration` 후 확률결합,
  가중은 제출본 고정값
- 시드 누적 스냅샷(1·2·3·5·8)을 한 번의 학습으로 전부 낸다

## 비용도 같이 본다

시드를 늘리면 추론 시간이 선형으로 는다. 제출1 이 서버 388초/600초였고
현재 구성이 로컬 환산 409초다. 8시드면 cb 구간이 2.67배가 된다 —
이득이 있어도 예산을 넘으면 못 쓴다.

    python seeds_raw.py --max-seeds 8
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
SNAPS = (1, 2, 3, 5, 8)


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
    ap.add_argument("--max-seeds", type=int, default=8)
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
    snaps = {}
    for fold in (2024, 2022):
        fp = FINAL / "preds" / ("SEEDS_cb_%d.npy" % fold)
        if fp.exists():
            snaps[fold] = np.load(fp)
            print("  fold%d (이미 있음)" % fold, flush=True)
            continue
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        Xall = np.concatenate([np.asarray(X), E], axis=1)
        del E
        Xt = np.ascontiguousarray(Xall[tri])
        Xv = np.ascontiguousarray(Xall[va])
        del Xall
        yt = y_all[tri].astype(np.float64)
        st = season[tri]

        # 배포 구성 그대로 — 시드분산 상보 표본가중
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
        out = np.zeros((len(SNAPS), len(va)))
        for sd in range(a.max_seeds):
            t0 = time.time()
            m = CatBoostRegressor(**dict(CB_P, depth=DEPTH), random_seed=sd,
                                  task_type="GPU", devices="0", border_count=128)
            m.fit(Pool(Xt, yt, weight=w))
            acc += np.clip(m.predict(Pool(Xv)), 1e-6, 1 - 1e-6)
            del m
            if (sd + 1) in SNAPS:
                out[SNAPS.index(sd + 1)] = acc / (sd + 1)
            print("    fold%d seed%d %.0f초" % (fold, sd, time.time() - t0), flush=True)
        np.save(fp, out)
        snaps[fold] = out
        del Xt, Xv

    # ── 판정 ────────────────────────────────────────────────────────────────
    P = FINAL / "preds"
    par = json.loads(zipfile.ZipFile(a.zip).read("model/cw/model/params.json"))
    W = np.array([par["blend_w_cb"], par["blend_w_ft"], par["blend_w_mlp"]])
    print("\n" + "=" * 84)
    print("[축5 재검증] 시드 수 — 원시 예측 · 배포 순서 재현 · 가중 고정")
    print("=" * 84)
    print("%-8s %11s %11s %11s %11s   %s"
          % ("시드", "cb단독24", "val2024", "cb단독22", "val2022", "cb 추론 배수"))
    print("-" * 84)
    base = None
    for i, k in enumerate(SNAPS):
        row = []
        for fold in (2024, 2022):
            y = load_labels(fold)["y"].to_numpy(np.float64)
            M = np.column_stack([
                cal(snaps[fold][i], par["model_cb"]),
                cal(np.load(P / ("FTX_q64_168_%d.npy" % fold)), par["model_ft"]),
                cal(np.load(P / ("PREP_std_176_%d.npy" % fold)), par["model_mlp"])])
            r = y.mean()
            row += [bss(snaps[fold][i], y),
                    bss(np.clip(r + (M - r) @ W, 1e-6, 1 - 1e-6), y)]
        if k == 3:
            base = row
        tag = " <- 현행" if k == 3 else ""
        d = "" if base is None else ("   Δ %+.1f / %+.1f" % (row[1] - base[1],
                                                             row[3] - base[3]))
        print("%-8d %11.1f %11.1f %11.1f %11.1f   %.2fx%s%s"
              % (k, row[0], row[1], row[2], row[3], k / 3.0, tag, d))
    print("\n(현행 3시드 기준. Δ 는 val2024 / val2022 순)")
    print("추론 예산: 제출1 이 서버 388초/600초, 현재 구성이 로컬 환산 409초다.")
    print("cb 구간이 배수만큼 늘어난다 — 이득이 있어도 예산을 넘으면 못 쓴다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
