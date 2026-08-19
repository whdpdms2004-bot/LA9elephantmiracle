"""middle 구조 변경 — 모델 계열 확장 + 하위 분할.

Stage M 에서 배운 것
    성향 피처는 전부 실패 (-0.1 ~ -5.0). 원본 as-of 비율이 이미 있어
    엔트로피·차·비는 트리가 만들 수 있는 형태였다.
    투수 키 아웃 축도 fold 2023 에서 뒤집혔다 (ph_outs -19.1).
    유일 생존은 outs_count (+1.9/+12.5) — 투수 없이 아웃x카운트.

    -> 파라미터·피처 조정은 한계다. 구조를 바꾼다.

A. 모델 계열 (1WAY 최대 미이식 항목)
    1WAY 는 성분당 XGB 8시드 + CatBoost 8시드 = 16모델이었다. 3WAY 는 CatBoost 1개.
    시드 배깅은 보류 중이므로 **계열만** 늘린다.
    xgb        XGBoost 단독
    cat_xgb    두 계열 평균 (배깅 아님 — 서로 다른 알고리즘)

B. 하위 분할 (1WAY V35 P2_msplit 의 3WAY 판)
    middle 을 ball 로 쪼개 각각 예측하고 더한다.
        y_mb = middle & ball        y_mz = middle & !ball
        p_middle = p_mb + p_mz
    1WAY 에서 P2_msplit 이 한때 살아남았다가 세 fold 확인에서 탈락했다.
    3WAY 는 타깃별 결과가 다르므로 재시도 가치가 있다.

    기저율: middle 0.1496 중 ball 18.3% -> mb 약 0.027, mz 약 0.122
    mb 가 얇으므로 트리 파라미터를 기저율에 맞춘다.

판정
    fold 2024 목표 / fold 2023 강건성 관문 (sd 2.69).

사용
    python middle_next.py --fold 2024
    python middle_next.py --fold 2024 --dry
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
from guards import assert_features_clean, save_prediction
from harness3 import (DECISION_FOLD, LAB, OUT, SUCCESS, TARGETS, bss,
                      load_labeled, seed_noise)

SEED = 20262844
TARGET = "middle"
BASE_COMBO = ("id_frequency", "rate_multiscale", "temporal_cyclic")
EPS = 1e-7


def cat_params(base: dict, rate: float) -> dict:
    p = dict(base)
    p.update({"bootstrap_type": "Bayesian", "bagging_temperature": 1.0})
    p.pop("subsample", None)
    if rate < 0.06:                      # mb 처럼 얇은 타깃
        p.update({"depth": 6, "l2_leaf_reg": 10.0})
    return p


def xgb_params(rate: float, base_score: float) -> dict:
    return {"max_depth": 0, "grow_policy": "lossguide",
            "max_leaves": 18 if rate > 0.10 else 10,
            "min_child_weight": 64 if rate > 0.10 else 192,
            "reg_lambda": 3.0 if rate > 0.10 else 8.0,
            "eta": 0.03, "subsample": 0.85, "colsample_bytree": 0.8,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "device": "cuda", "seed": SEED,
            "base_score": float(np.clip(base_score, 1e-4, 1 - 1e-4))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=DECISION_FOLD)
    ap.add_argument("--arms", default="cat,xgb,cat_xgb,split_ball,split_ball_xgb")
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--combo", default="", help="전처리 조합. 비우면 BASE_COMBO")
    ap.add_argument("--tag", default="", help="캐시 구분용 접미사")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    import xgboost as xgb
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
    tr_mask, va_mask = (season < args.fold) & ok, (season == args.fold) & ok
    y_va = ymid[va_mask].astype("int8")
    sd = seed_noise(TARGET)

    # 하위 분할 라벨
    ymb = ((ymid == 1) & (yball == 1)).astype(np.float64)
    ymz = ((ymid == 1) & (yball == 0)).astype(np.float64)
    assert np.abs((ymb + ymz)[ok] - ymid[ok]).max() < 1e-9, "분할이 middle 을 안 덮는다"

    print(f"타깃 {TARGET}  fold {args.fold}  학습 {tr_mask.sum():,}  검증 {va_mask.sum():,}")
    for nm, v in (("middle", ymid), ("mb(=m&ball)", ymb), ("mz(=m&!ball)", ymz)):
        print(f"  {nm:<14} 학습 기저율 {v[tr_mask].mean():.4f}")
    print(f"  전처리 {'+'.join(BASE_COMBO)}   잡음 sd {sd:.2f}", flush=True)

    static = build_component_unique(frame, enhanced, args.fold)
    forward = build_component_unique_forward(frame, enhanced, args.fold,
                                             cache={args.fold: static})
    base_fr, f1_features = make_features(frame, enhanced, args.fold, "F1", forward)
    for c in (SUCCESS, "season"):
        if c not in base_fr.columns:
            base_fr[c] = frame[c].to_numpy()
    cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
    _combo = tuple(c for c in (args.combo.split("+") if args.combo else BASE_COMBO) if c)
    print(f"  전처리 실제: {'+'.join(_combo) or 'baseline'}", flush=True)
    fr, feats, cat_cols = T.build(base_fr, f1_features, cats0, _combo,
                                  pd.Series(tr_mask, index=frame.index), args.fold)
    assert_features_clean(feats, TARGET)

    P0 = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(P0.pop("half_life"))
    P0.update({"iterations": args.iterations, "learning_rate": 0.015, "depth": 8,
               "random_seed": SEED, "task_type": "GPU", "devices": "0", "verbose": 0})
    w = np.asarray(recency_weights(frame.loc[tr_mask, "season"], args.fold,
                                   half_life), np.float64)

    def prior_of(y):
        s = pd.Series(y[tr_mask]).groupby(
            pd.Series(season[tr_mask])).mean().sort_index()
        span = float(s.index[-1]) - float(s.index[0])
        step = (float(s.iloc[-1]) - float(s.iloc[0])) / span if span > 0 else 0.0
        return float(np.clip(float(s.iloc[-1]) + step, 0.005, 0.995))

    # XGB 용 인코딩 (범주형은 코드화)
    X = fr[feats].copy()
    for c in cat_cols:
        vals = X.loc[tr_mask, c].fillna("__M__").astype(str)
        mp = {v: i for i, v in enumerate(pd.unique(vals))}
        X[c] = X[c].fillna("__M__").astype(str).map(mp).fillna(-1)
    X = X.apply(pd.to_numeric, errors="coerce").astype("float32")

    def fit_cat(y):
        pr = prior_of(y)
        bl = float(np.log(pr / (1 - pr)))
        p_tr = Pool(fr.loc[tr_mask, feats], y[tr_mask].astype("int8"),
                    cat_features=cat_cols, weight=w,
                    baseline=np.full(int(tr_mask.sum()), bl))
        p_va = Pool(fr.loc[va_mask, feats], ymid[va_mask].astype("int8"),
                    cat_features=cat_cols,
                    baseline=np.full(int(va_mask.sum()), bl))
        m = CatBoostClassifier(**cat_params(P0, float(y[tr_mask].mean())))
        m.fit(p_tr)
        out = m.predict_proba(p_va)[:, 1]
        del p_tr, p_va, m
        gc.collect()
        return out

    def fit_xgb(y):
        pr = prior_of(y)
        d_tr = xgb.DMatrix(X.loc[tr_mask], label=y[tr_mask], weight=w, missing=np.nan)
        d_va = xgb.DMatrix(X.loc[va_mask], missing=np.nan)
        m = xgb.train(xgb_params(float(y[tr_mask].mean()), pr), d_tr,
                      num_boost_round=args.iterations)
        out = m.predict(d_va)
        del d_tr, d_va, m
        gc.collect()
        return out

    rows, b0 = [], None
    print(f"{chr(10)}  {'arm':<16}{'bss_raw':>11}{'Δ':>9}{'centered':>11}"
          f"{'오프셋':>10}  출처")
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        npy = OUT / f"mn_{TARGET}__{arm}{args.tag}__{args.fold}.npy"
        if npy.exists():
            m, src = bss(y_va, np.load(npy)), "cache"
        elif args.dry:
            print(f"  {arm:<16}{'':>11}{'':>9}{'':>11}{'':>10}  dry", flush=True)
            continue
        else:
            t0 = time.time()
            try:
                if arm == "cat":
                    pred = fit_cat(ymid)
                elif arm == "xgb":
                    pred = fit_xgb(ymid)
                elif arm == "cat_xgb":
                    pred = 0.5 * fit_cat(ymid) + 0.5 * fit_xgb(ymid)
                elif arm == "split_ball":
                    pred = fit_cat(ymb) + fit_cat(ymz)
                elif arm == "split_ball_xgb":
                    pred = fit_xgb(ymb) + fit_xgb(ymz)
                else:
                    print(f"  {arm:<16}  모르는 arm"); continue
            except Exception as exc:                              # noqa: BLE001
                print(f"  {arm:<16}  실패: {type(exc).__name__}: {str(exc)[:60]}",
                      flush=True)
                continue
            pred = np.clip(pred, EPS, 1 - EPS)
            save_prediction(npy, pred, y_va, where=f"{TARGET}/{arm}")
            m, src = bss(y_va, pred), f"fit {time.time() - t0:.0f}s"
        if b0 is None:
            b0 = m["bss_raw"]
        d = m["bss_raw"] - b0
        mark = "" if abs(d) > sd else "  (잡음)"
        print(f"  {arm:<16}{m['bss_raw']:>11.1f}{d:>+9.1f}{m['bss_centered']:>11.1f}"
              f"{m['offset']:>+10.4f}  {src}{mark}", flush=True)
        rows.append({"target": TARGET, "arm": arm + args.tag,
                     "combo": "+".join(_combo), "fold": args.fold, "d": d, **m})
        pd.DataFrame(rows).to_csv(OUT / f"middle_next{args.tag}_{args.fold}.csv", index=False)

    if rows:
        t = pd.DataFrame(rows).sort_values("bss_raw", ascending=False)
        t.to_csv(OUT / f"middle_next{args.tag}_{args.fold}.csv", index=False)
        print(f"{chr(10)}  최고 {t.bss_raw.max():.1f} ({t.iloc[0].arm})   "
              f"목표 1300 까지 {1300 - t.bss_raw.max():+.0f}")


if __name__ == "__main__":
    main()
