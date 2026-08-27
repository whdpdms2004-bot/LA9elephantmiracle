# -*- coding: utf-8 -*-
"""[축1 재판정] 전처리 원자 **조합**을 깨끗한 계기로 다시 잰다.

## 왜 다시 하나

`cowork/sj/preprocess_lab/RESULTS.md` 가 직접 말한다.

> **1위 `id_frequency + temporal_cyclic + trackman_quality` = +16.52** (295 피처)
> 최고 단일 `id_frequency` = +7.94. **조합이 두 배를 넘겼다**
> **단독 성능으로 전처리를 거르면 안 된다** — `trackman_quality` 는 단독 **−3.11**
> 인데 최상위 조합 전부에 들어 있다

**그런데 지금 배포본에는 `id_freq` 하나만 들어가 있다.** 조합 판정을 `run_family`
경로로 했고, 그 경로가 저장하는 예측은 `calib(p, yv)` — **평가 폴드 라벨로**
로짓을 재중심화하고 스케일을 고른 것이다. 그 오염이 이번 캠페인에서 두 결론을
뒤집었다 (`cb_f_only`, [축7]의 ρ 선). 조합 판정도 같은 의심을 받는다.

랩의 "단독으로 거르면 안 된다" 는 이번 캠페인의 법칙 3
("다른 표현이면서 단독 신호도 있어야 한다") 과 같은 구조의 관찰이다 —
단독과 결합 기여가 다르다는 것.

## 이번 계기

- 원시 예측만 쓴다 (`calib` 금지)
- cb 는 **현행 배포 구성**으로 학습한다 (depth 6 · l2 10000 · 시드분산 가중 alpha 0.15)
- 판정은 **배포 순서 재현** — 모델별 `apply_calibration` 후 확률결합,
  가중은 조합마다 두 폴드 평균으로 재적합
- `ft`·`mlp` 는 고정한다. cb 의 입력 원자만 바꾼다

## arm

    id_freq                                현행 배포본 (기준)
    id_freq+temporal                       랩 7위 (+12.93)
    id_freq+tm_quality                     랩 9위 (+12.85)
    id_freq+temporal+tm_quality            ★랩 1위 (+16.52)
    id_freq+temporal+tm_quality+context    랩 3위 (+15.47)
    id_freq+rate_geom+tm_quality           랩 5위 (+14.33)

    python atoms_raw.py
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

ARMS = ["id_freq",
        "id_freq+temporal",
        "id_freq+tm_quality",
        "id_freq+temporal+tm_quality",
        "id_freq+temporal+tm_quality+context",
        "id_freq+rate_geom+tm_quality"]


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
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--zip", default=str(FINAL / "submit" / "submit_sj_stdmlp.zip"))
    a = ap.parse_args()
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]

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
        yt = y_all[tri].astype(np.float64)
        st = season[tri]
        for arm in arms:
            tag = arm.replace("+", "-")
            fp = FINAL / "preds" / ("ATOMR_%s_%d.npy" % (tag, fold))
            if fp.exists():
                print("  fold%d %-40s (이미 있음, 단독 %.1f)"
                      % (fold, arm, bss(np.load(fp), y)), flush=True)
                continue
            t0 = time.time()
            E, en = A.build(X, names, season <= fold - 1, fold, arm.split("+"))
            Xall = np.concatenate([np.asarray(X), E], axis=1)
            del E
            Xt = np.ascontiguousarray(Xall[tri])
            Xv = np.ascontiguousarray(Xall[va])
            del Xall

            # 현행 배포 구성과 같게 — 시드분산 상보 표본가중을 건다.
            # 원자 조합을 바꿔도 이 절차는 그대로여야 비교가 공정하다.
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
            p = acc / a.seeds
            np.save(fp, p)
            print("  fold%d %-40s %3d열 %5.0f초  단독 %.1f"
                  % (fold, arm, Xt.shape[1], time.time() - t0, bss(p, y)), flush=True)
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
    print("\n" + "=" * 104)
    print("[축1 재판정] 원자 조합 — 원시 예측 · 배포 순서 재현")
    print("  기준 = 현행 배포본의 cb (id_freq 만). ft·mlp 고정, 가중은 조합마다 재적합")
    print("=" * 104)
    print("%-38s %10s %10s %10s   %8s %8s   %s"
          % ("원자 조합", "cb단독24", "val2024", "val2022", "Δ24", "Δ22", "Public 예상"))
    print("-" * 104)
    base, _ = sc("ATOMR_id_freq")
    y24 = Y[2024]
    for arm in arms:
        tag = arm.replace("+", "-")
        if not (P / ("ATOMR_%s_2024.npy" % tag)).exists():
            continue
        s, w = sc("ATOMR_%s" % tag)
        solo = bss(np.load(P / ("ATOMR_%s_2024.npy" % tag)), y24)
        mark = " <- 기준" if arm == "id_freq" else ""
        print("%-38s %10.1f %10.1f %10.1f   %+8.1f %+8.1f   %.1f%s"
              % (arm, solo, s[0], s[1], s[0] - base[0], s[1] - base[1],
                 1080.425 + (s[0] - base[0]) * RATE, mark))
    print("\n(랩은 fold2024 · depth8 · 900트리에서 1위 조합이 단독 +16.52 였다.")
    print(" 여기는 현행 배포 구성(depth6 · 3000트리 · 시드분산가중)에서 결합 기여로 판정한다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
