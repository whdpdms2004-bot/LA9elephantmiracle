"""middle 전용 하이퍼파라미터 탐색 — 3WAY 에서 아직 안 건드린 축.

왜 이게 남았나
    3WAY 의 모든 실행은 1WAY 에서 Optuna 로 맞춘 파라미터를 그대로 쓴다.
    그 값은 control_success(기저율 약 0.55)를 목적함수로 맞춘 것이다.
    middle 은 기저율 0.15, split_ball 의 하위 타깃은 0.027 / 0.122 다.
    깊이·정규화·경계수가 그 기저율에 맞을 이유가 없다.

무엇을 목적함수로 두나
    fold 2023 의 **bss_centered** 를 최대화한다. raw 가 아니다.
    실측: middle 은 f23 raw 순위가 오프셋 탓에 f24 와 반대로 간다.
    fold 간에 전이되는 것은 판별력이고 그게 centered 다.
    centered 는 guards.centered_bss 를 거치므로 float 만 나온다 —
    이동된 배열은 어디에도 저장되지 않는다 (조항 2).

    탐색은 fold 2023 에서만 한다. fold 2024 는 상위 후보 재적합에만 쓴다.

구조는 고정
    split_ball (y_mb + y_mz), 전처리 rate_multiscale+no_trackman.
    재탐색에서 두 fold 통과한 최고 구성이다 (f24 903.9 / f23 467.1).

사용
    python middle_tune.py --trials 24                # fold 2023 탐색
    python middle_tune.py --refit 4                  # 상위 4개를 fold 2024 재적합
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
from harness3 import LAB, OUT, SUCCESS, bss, load_labeled, seed_noise

SEED = 20262844
EPS = 1e-7
COMBO = "rate_multiscale+no_trackman"
STUDY = OUT / "middle_tune.csv"


def sample(P0: dict, rng: np.random.Generator, i: int) -> dict:
    """P0(1WAY Optuna 결과)를 상속하고 탐색 키만 덮어쓴다.

    처음에 P0 를 안 상속하고 처음부터 썼더니 t0 이 f23 225.6 으로 나왔다.
    실제 현행은 467.1 이다 — P0 의 l2_leaf_reg 124.9, random_strength 0.0004,
    border_count 254 가 빠져 있었다. 탐색은 그 값 주위에서 해야 한다.

    t0  현행 3WAY 설정 (P0 + depth8/lr0.015/iter900 덮어쓰기)  = 기준점
    t1  P0 그대로 (depth9/lr0.0078/iter4487)  — 덮어쓰기가 손해인지 본다
    """
    base = {k: v for k, v in P0.items() if k != "half_life"}
    if i == 0:
        base.update({"depth": 8, "learning_rate": 0.015, "iterations": 900,
                     "bagging_temperature": 1.0})
        return base
    if i == 1:
        return base                                   # P0 원본
    base.update({
        "depth": int(rng.choice([6, 7, 8, 9, 10])),
        "learning_rate": float(rng.choice([0.0078, 0.012, 0.015, 0.02, 0.03])),
        "iterations": int(rng.choice([900, 1600, 2600, 4487])),
        "l2_leaf_reg": float(rng.choice([25.0, 60.0, 124.9, 250.0, 500.0])),
        "bagging_temperature": float(rng.choice([0.5, 1.0, 2.74, 5.0])),
        "random_strength": float(rng.choice([0.0004, 0.01, 0.2, 1.0])),
        "border_count": int(rng.choice([128, 254])),
        "one_hot_max_size": int(rng.choice([2, 8, 32])),
    })
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--fold", type=int, default=2023)
    ap.add_argument("--refit", type=int, default=0,
                    help=">0 이면 탐색 대신 상위 N개를 이 fold 에서 재적합")
    ap.add_argument("--combo", default=COMBO)
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

    fold = args.fold
    tr, va = (season < fold) & ok, (season == fold) & ok
    y_va = ymid[va].astype("int8")
    sd = seed_noise("middle")

    t0 = time.time()
    static = build_component_unique(frame, enhanced, fold)
    forward = build_component_unique_forward(frame, enhanced, fold,
                                             cache={fold: static})
    base_fr, f1_features = make_features(frame, enhanced, fold, "F1", forward)
    for c in (SUCCESS, "season"):
        if c not in base_fr.columns:
            base_fr[c] = frame[c].to_numpy()
    cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
    fr, feats, cat_cols = T.build(base_fr, f1_features, cats0,
                                  tuple(x for x in args.combo.split("+") if x),
                                  pd.Series(tr, index=frame.index), fold)
    assert_features_clean(feats, "middle")

    P0 = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(P0["half_life"])
    w = np.asarray(recency_weights(frame.loc[tr, "season"], fold, half_life),
                   np.float64)

    print(f"middle 파라미터 탐색  fold {fold}  전처리 {args.combo}")
    print(f"  학습 {tr.sum():,}  검증 {va.sum():,}  프레임 {time.time() - t0:.0f}s")
    print(f"  목적함수 fold {fold} bss_centered (판별력). 잡음 sd {sd:.2f}")
    print(f"  기준 현행설정: f24 903.9 / f23 467.1 (cen 504.3)", flush=True)

    def fit_one(y, hp):
        p = dict(hp)
        p.update({"random_seed": SEED, "task_type": "GPU", "devices": "0",
                  "verbose": 0, "bootstrap_type": "Bayesian",
                  "loss_function": "Logloss"})
        p.pop("subsample", None)
        if float(y[tr].mean()) < 0.06:          # 얇은 하위 타깃은 한 단계 보수적으로
            p["depth"] = max(4, int(p["depth"]) - 2)
        pr = train_season_trend(y[tr], season[tr], fold)
        bl = float(np.log(pr / (1 - pr)))
        p_tr = Pool(fr.loc[tr, feats], y[tr].astype("int8"), cat_features=cat_cols,
                    weight=w, baseline=np.full(int(tr.sum()), bl))
        p_va = Pool(fr.loc[va, feats], cat_features=cat_cols,
                    baseline=np.full(int(va.sum()), bl))
        m = CatBoostClassifier(**p)
        m.fit(p_tr)
        out = m.predict_proba(p_va)[:, 1]
        del p_tr, p_va, m
        gc.collect()
        return out

    rows = []
    if args.refit > 0:
        if not STUDY.exists():
            print("탐색 결과가 없다. 먼저 --trials 로 돌려라.")
            return
        prev = pd.read_csv(STUDY)
        prev = prev[prev.fold != fold].sort_values("bss_centered", ascending=False)
        cand = prev.head(args.refit)
        print(f"{chr(10)}상위 {len(cand)}개를 fold {fold} 에서 재적합")
        space = [(int(r.trial), json.loads(r.hp)) for r in cand.itertuples()]
    else:
        rng = np.random.default_rng(SEED)
        space = [(i, sample(P0, rng, i)) for i in range(args.trials)]

    print(f"{chr(10)}  {'trial':<6}{'bss_raw':>10}{'centered':>10}{'오프셋':>10}"
          f"  depth lr      iter  l2     bagT  rs      bord ohe   시간")
    best = -1e18
    for i, hp in space:
        npy = OUT / f"mt_middle__t{i:03d}__{fold}.npy"
        if npy.exists():
            pred, src = np.load(npy), 0.0
        else:
            t1 = time.time()
            try:
                pred = np.clip(fit_one(ymb, hp) + fit_one(ymz, hp), EPS, 1 - EPS)
            except Exception as exc:                              # noqa: BLE001
                print(f"  t{i:<5}  실패: {type(exc).__name__}: {str(exc)[:60]}",
                      flush=True)
                continue
            save_prediction(npy, pred, y_va, where=f"middle/tune/t{i}")
            src = time.time() - t1
        m = bss(y_va, pred)
        mark = "  <-- 최고" if m["bss_centered"] > best else ""
        best = max(best, m["bss_centered"])
        print(f"  t{i:<5}{m['bss_raw']:>10.1f}{m['bss_centered']:>10.1f}"
              f"{m['offset']:>+10.4f}  {hp['depth']:<6}{hp['learning_rate']:<8}"
              f"{hp['iterations']:<6}{hp['l2_leaf_reg']:<7.1f}{hp['bagging_temperature']:<6}"
              f"{hp['random_strength']:<8}{hp['border_count']:<5}"
              f"{hp['one_hot_max_size']:<5}{src:>5.0f}s{mark}", flush=True)
        rows.append({"trial": i, "fold": fold, "combo": args.combo,
                     "hp": json.dumps(hp), **m})
        t = pd.DataFrame(rows)
        if STUDY.exists():
            t = pd.concat([pd.read_csv(STUDY), t], ignore_index=True)
        t.drop_duplicates(["trial", "fold"], keep="last").to_csv(STUDY, index=False)

    if rows:
        t = pd.DataFrame(rows).sort_values("bss_centered", ascending=False)
        b = t.iloc[0]
        print(f"{chr(10)}  fold {fold} 최고 centered {b.bss_centered:.1f} "
              f"(raw {b.bss_raw:.1f}, trial {int(b.trial)})")
        print(f"  현행 대비 centered {b.bss_centered - 504.3:+.1f}  (잡음 sd {sd:.2f})")
        print(f"saved -> {STUDY}")


if __name__ == "__main__":
    main()
