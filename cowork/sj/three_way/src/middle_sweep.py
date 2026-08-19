"""middle 조합 x 구조 스윕 — 프레임을 한 번만 만들고 안에서 돈다.

왜 새로 쓰나
    middle_next.py 는 --combo 하나마다 프로세스를 새로 띄운다.
    프레임 구성(component_unique + forward + make_features)이 조합당 5분이라
    20회 스윕이 2시간이 된다. 실제 학습은 조합당 80초뿐이다.
    여기서는 fold 당 프레임을 한 번 만들고 조합만 갈아끼운다. 20회가 35분이 된다.

무엇을 재는가
    재훑기에서 나온 것: split_ball + rate_multiscale 이 f24 854.4 로 단독 최고였고,
    거기에 id_frequency 를 더하면 671.9 로 떨어졌다. 오프셋도 -0.0116 vs -0.0200.
    ID 빈도 인코딩이 학습 분포로 끌어당긴다 — 기저율이 튄 2024 에서 손해다.
    그래서 **id 계열을 뺀 조합** 을 넓게 훑는다.

구조 arm
    split_ball    y_mb + y_mz 가법 분할 (재훑기 최고 구조)
    single        분할 없음 — 조합 효과가 구조와 무관한지 가른다

판정
    fold 2024 목표 / fold 2023 강건성 관문 (sd 2.69).
    두 fold 를 한 실행에서 돌리되 fold 마다 프레임을 다시 만든다 (fold 종속).

사용
    python middle_sweep.py --folds 2024,2023
    python middle_sweep.py --folds 2024 --combos rate_multiscale,drop_ids
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guards import assert_features_clean, save_prediction, train_season_trend
from harness3 import (LAB, OUT, SUCCESS, bss, load_labeled, seed_noise)

SEED = 20262844
EPS = 1e-7

# id_frequency 를 뺀 조합들. rate_multiscale 단독이 현재 최고이므로 그 주변을 훑는다.
COMBOS = [
    "rate_multiscale",
    "rate_multiscale+no_trackman",
    "rate_multiscale+drop_ids",
    "rate_multiscale+temporal_cyclic",
    "rate_multiscale+count_multiscale",
    "rate_multiscale+context_robust",
    "rate_multiscale+rate_geometry",
    "rate_multiscale+component_compact",
    "rate_multiscale+trackman_compact",
    "rate_multiscale+no_trackman+drop_ids",
    "drop_ids",
    "no_trackman",
]


def cat_params(base: dict, rate: float) -> dict:
    p = dict(base)
    p.update({"bootstrap_type": "Bayesian", "bagging_temperature": 1.0})
    p.pop("subsample", None)
    if rate < 0.06:
        p.update({"depth": 6, "l2_leaf_reg": 10.0})
    return p


def slug(combo: str) -> str:
    return combo.replace("+", "_")[:44]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="2024,2023")
    ap.add_argument("--combos", default=",".join(COMBOS))
    ap.add_argument("--arms", default="split_ball")
    ap.add_argument("--iterations", type=int, default=900)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    from catboost import CatBoostClassifier, Pool
    from run_optuna_enhanced import load_enhanced_frame
    from run_optuna_family import CATEGORICAL_COLUMNS, recency_weights
    from v77_single_xgb_screen import (build_component_unique,
                                       build_component_unique_forward)
    from v80_single_catboost import make_features
    import v85_preprocess_screen as M
    sys.path.insert(0, str(LAB))
    import transforms as T
    T.load_all()

    frame, enhanced = load_enhanced_frame()
    labeled = load_labeled()
    season = frame["season"].to_numpy()
    ymid = pd.to_numeric(labeled["y_middle"], errors="coerce").to_numpy(np.float64)
    yball = pd.to_numeric(labeled["y_ball"], errors="coerce").to_numpy(np.float64)
    ok = (labeled["label_ok"].to_numpy() == 1) & ~np.isnan(ymid) & ~np.isnan(yball)
    ymb = ((ymid == 1) & (yball == 1)).astype(np.float64)
    ymz = ((ymid == 1) & (yball == 0)).astype(np.float64)
    assert np.abs((ymb + ymz)[ok] - ymid[ok]).max() < 1e-9, "분할이 middle 을 안 덮는다"
    sd = seed_noise("middle")

    P0 = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(P0.pop("half_life"))
    P0.update({"iterations": args.iterations, "learning_rate": 0.015, "depth": 8,
               "random_seed": SEED, "task_type": "GPU", "devices": "0", "verbose": 0})

    combos = [c.strip() for c in args.combos.split(",") if c.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_rows = []

    for fold in [int(f) for f in args.folds.split(",") if f.strip()]:
        tr, va = (season < fold) & ok, (season == fold) & ok
        y_va = ymid[va].astype("int8")
        t0 = time.time()
        static = build_component_unique(frame, enhanced, fold)
        forward = build_component_unique_forward(frame, enhanced, fold,
                                                 cache={fold: static})
        base_fr, f1_features = make_features(frame, enhanced, fold, "F1", forward)
        for c in (SUCCESS, "season"):
            if c not in base_fr.columns:
                base_fr[c] = frame[c].to_numpy()
        cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
        w = np.asarray(recency_weights(frame.loc[tr, "season"], fold, half_life),
                       np.float64)

        print(f"{chr(10)}{'=' * 100}")
        print(f"fold {fold}  학습 {tr.sum():,}  검증 {va.sum():,}  "
              f"검증 기저율 {ymid[va].mean():.4f}  프레임 {time.time() - t0:.0f}s")
        print(f"  잡음 sd {sd:.2f}   기준: split_ball+rate_multiscale "
              f"f24 854.4 / f23 466.9")
        print("=" * 100)
        print(f"  {'조합':<40}{'arm':<12}{'bss_raw':>10}{'centered':>10}"
              f"{'오프셋':>10}{'평균':>9}  출처", flush=True)

        for combo in combos:
            ctup = tuple(x for x in combo.split("+") if x)
            try:
                fr, feats, cat_cols = T.build(base_fr, f1_features, cats0, ctup,
                                              pd.Series(tr, index=frame.index), fold)
            except Exception as exc:                              # noqa: BLE001
                print(f"  {combo[:40]:<40}{'':12}  전처리 실패: "
                      f"{type(exc).__name__}: {str(exc)[:44]}", flush=True)
                continue
            assert_features_clean(feats, "middle")

            def fit(y):
                pr = train_season_trend(y[tr], season[tr], fold)
                bl = float(np.log(pr / (1 - pr)))
                p_tr = Pool(fr.loc[tr, feats], y[tr].astype("int8"),
                            cat_features=cat_cols, weight=w,
                            baseline=np.full(int(tr.sum()), bl))
                p_va = Pool(fr.loc[va, feats], cat_features=cat_cols,
                            baseline=np.full(int(va.sum()), bl))
                m = CatBoostClassifier(**cat_params(P0, float(y[tr].mean())))
                m.fit(p_tr)
                out = m.predict_proba(p_va)[:, 1]
                del p_tr, p_va, m
                gc.collect()
                return out

            for arm in arms:
                npy = OUT / f"ms_middle__{arm}_{slug(combo)}__{fold}.npy"
                if npy.exists():
                    pred = np.load(npy)
                    m, src = bss(y_va, pred), "cache"
                else:
                    t1 = time.time()
                    try:
                        pred = (fit(ymb) + fit(ymz)) if arm == "split_ball" else fit(ymid)
                    except Exception as exc:                      # noqa: BLE001
                        print(f"  {combo[:40]:<40}{arm:<12}  실패: "
                              f"{type(exc).__name__}: {str(exc)[:40]}", flush=True)
                        continue
                    pred = np.clip(pred, EPS, 1 - EPS)
                    save_prediction(npy, pred, y_va, where=f"middle/{arm}/{combo}")
                    m, src = bss(y_va, pred), f"fit {time.time() - t1:.0f}s"
                print(f"  {combo[:40]:<40}{arm:<12}{m['bss_raw']:>10.1f}"
                      f"{m['bss_centered']:>10.1f}{m['offset']:>+10.4f}"
                      f"{float(pred.mean()):>9.4f}  {src}", flush=True)
                all_rows.append({"target": "middle", "combo": combo, "arm": arm,
                                 "fold": fold, **m})
                pd.DataFrame(all_rows).to_csv(OUT / "middle_sweep.csv", index=False)
            del fr
            gc.collect()
        del base_fr, static, forward
        gc.collect()

    if all_rows:
        t = pd.DataFrame(all_rows)
        p = OUT / "middle_sweep.csv"
        if p.exists():
            old = pd.read_csv(p)
            t = pd.concat([old, t], ignore_index=True)
        t = t.drop_duplicates(["target", "combo", "arm", "fold"], keep="last")
        t.to_csv(p, index=False)
        print(f"{chr(10)}saved -> {p}")
        w24 = t[t.fold == 2024].sort_values("bss_raw", ascending=False)
        if len(w24):
            b = w24.iloc[0]
            print(f"  fold 2024 최고 {b.bss_raw:.1f}  ({b.combo} / {b.arm})   "
                  f"목표 1300 까지 {1300 - b.bss_raw:+.0f}")


if __name__ == "__main__":
    main()
