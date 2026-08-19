"""Stage W1b/W3: way 별 base 구조 + 모델 계열 다양화. 타깃 무관.

middle 에서 split_ball 이 유일하게 두 fold 양수였다 (f23 +30.9 / f24 +5.7).
같은 구조를 다른 way 에도 건다.

분할 근거 — ball 이 way 마다 다르게 걸린다
    middle  중 ball 18.3%   ->  mb 0.027 / mz 0.122
    reverse 중 ball 36.1%   ->  rb 0.083 / rz 0.146
    outside 중 ball 82.4%   ->  ob 0.108 / oz 0.023
    outside 는 1WAY 에서 이 분할(ob/oz)로 +2.64 를 얻은 전례가 있다.

arm
    single        분할 없이 CatBoost 하나
    split_ball    ball 로 쪼개 각각 예측 후 합산
    split_count   카운트군(4)으로 쪼개 합산 — 얇아지므로 파라미터를 낮춘다
    lgbm          LightGBM 단독 (미시도 계열)
    cat_lgbm      CatBoost + LightGBM 평균

주의
    XGBoost 는 middle 에서 f24 -97.2 로 확실히 열등했다. 여기서는 뺀다.
    시드 배깅은 보류 중이다 — 여기 arm 은 전부 서로 다른 구성이다.

판정
    fold 2024 목표 / fold 2023 강건성 관문.

사용
    python way_base.py --target reverse
    python way_base.py --target reverse,outside --fold 2023
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
from harness3 import (DECISION_FOLD, LAB, OUT, SUCCESS, TARGETS, bss,
                      load_labeled, seed_noise)

SEED = 20262844
EPS = 1e-7
BEST_COMBO = {
    "middle": "id_frequency+rate_multiscale+temporal_cyclic",
    "reverse": "drop_ids",
    "outside": "drop_ids+no_trackman+rate_multiscale",
    "mr": "id_frequency+rate_multiscale",
}
ARMS = ["single", "split_ball", "split_count", "lgbm", "cat_lgbm"]


def cat_params(base: dict, rate: float) -> dict:
    p = dict(base)
    p.update({"bootstrap_type": "Bayesian", "bagging_temperature": 1.0})
    p.pop("subsample", None)
    if rate < 0.06:
        p.update({"depth": 6, "l2_leaf_reg": 10.0})
    return p


def lgb_params(rate: float) -> dict:
    return {"objective": "binary", "metric": "binary_logloss",
            "num_leaves": 64 if rate > 0.10 else 24,
            "min_data_in_leaf": 128 if rate > 0.10 else 400,
            "lambda_l2": 5.0 if rate > 0.10 else 12.0,
            "learning_rate": 0.03, "feature_fraction": 0.8,
            "bagging_fraction": 0.85, "bagging_freq": 1,
            "verbosity": -1, "seed": SEED, "num_threads": 8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="reverse,outside")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--fold", type=int, default=DECISION_FOLD)
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--rule", default="linear_all",
                    help="시즌 외삽 규칙: linear_all/last/linear_3/median_diff/ewm")
    ap.add_argument("--dry", action="store_true")
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
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
        print("! lightgbm 미설치 — lgbm arm 은 건너뛴다")

    frame, enhanced = load_enhanced_frame()
    labeled = load_labeled()
    season = frame["season"].to_numpy()
    yball = pd.to_numeric(labeled["y_ball"], errors="coerce").to_numpy(np.float64)
    num = lambda c: pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    cnt = np.digitize(num("balls_before") * 3 + num("strikes_before"), [3, 6, 9])

    static = build_component_unique(frame, enhanced, args.fold)
    forward = build_component_unique_forward(frame, enhanced, args.fold,
                                             cache={args.fold: static})
    base_fr, f1_features = make_features(frame, enhanced, args.fold, "F1", forward)
    for c in (SUCCESS, "season"):
        if c not in base_fr.columns:
            base_fr[c] = frame[c].to_numpy()
    cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]

    P0 = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(P0.pop("half_life"))
    P0.update({"iterations": args.iterations, "learning_rate": 0.015, "depth": 8,
               "random_seed": SEED, "task_type": "GPU", "devices": "0", "verbose": 0})

    all_rows = []
    for tg in [t.strip() for t in args.target.split(",") if t.strip()]:
        yv = pd.to_numeric(labeled[TARGETS[tg]], errors="coerce").to_numpy(np.float64)
        ok = (labeled["label_ok"].to_numpy() == 1) & ~np.isnan(yv) & ~np.isnan(yball)
        tr_mask, va_mask = (season < args.fold) & ok, (season == args.fold) & ok
        y_va = yv[va_mask].astype("int8")
        sd = seed_noise(tg)
        combo = tuple(c for c in BEST_COMBO.get(tg, "").split("+") if c)

        print(f"{chr(10)}{'=' * 96}")
        print(f"타깃 {tg}  fold {args.fold}  전처리 {'+'.join(combo) or 'baseline'}")
        print(f"  기저율 {yv[tr_mask].mean():.4f}  잡음 sd {sd:.2f}")
        print("=" * 96, flush=True)

        fr, feats, cat_cols = T.build(base_fr, f1_features, cats0, combo,
                                      pd.Series(tr_mask, index=frame.index), args.fold)
        assert_features_clean(feats, tg)
        w = np.asarray(recency_weights(frame.loc[tr_mask, "season"],
                                       args.fold, half_life), np.float64)

        # 분할 라벨
        parts = {
            "split_ball": [("b1", ((yv == 1) & (yball == 1)).astype(float)),
                           ("b0", ((yv == 1) & (yball == 0)).astype(float))],
            "split_count": [(f"c{k}", ((yv == 1) & (cnt == k)).astype(float))
                            for k in np.unique(cnt)],
        }
        for nm, ps in parts.items():
            tot = sum(v for _, v in ps)
            assert np.abs(tot[ok] - yv[ok]).max() < 1e-9, f"{nm} 분할이 안 덮는다"
            print(f"  {nm}: " + " ".join(f"{k}={v[tr_mask].mean():.4f}" for k, v in ps))

        Xl = None

        def prior_of(y):
            # 관문이 허용하는 학습 시즌 추세. fold 이상 시즌이 섞이면 예외를 던진다.
            return train_season_trend(y[tr_mask], season[tr_mask], args.fold,
                                      rule=args.rule)

        def fit_cat(y):
            pr = prior_of(y)
            bl = float(np.log(pr / (1 - pr)))
            p_tr = Pool(fr.loc[tr_mask, feats], y[tr_mask].astype("int8"),
                        cat_features=cat_cols, weight=w,
                        baseline=np.full(int(tr_mask.sum()), bl))
            p_va = Pool(fr.loc[va_mask, feats], y_va, cat_features=cat_cols,
                        baseline=np.full(int(va_mask.sum()), bl))
            m = CatBoostClassifier(**cat_params(P0, float(y[tr_mask].mean())))
            m.fit(p_tr)
            out = m.predict_proba(p_va)[:, 1]
            del p_tr, p_va, m
            gc.collect()
            return out

        def fit_lgb(y):
            nonlocal Xl
            if lgb is None:
                raise RuntimeError("lightgbm 없음")
            if Xl is None:
                Xl = fr[feats].copy()
                for c in cat_cols:
                    vals = Xl.loc[tr_mask, c].fillna("__M__").astype(str)
                    mp = {v: i for i, v in enumerate(pd.unique(vals))}
                    Xl[c] = Xl[c].fillna("__M__").astype(str).map(mp).fillna(-1)
                Xl = Xl.apply(pd.to_numeric, errors="coerce").astype("float32")
            d = lgb.Dataset(Xl.loc[tr_mask], label=y[tr_mask], weight=w,
                            free_raw_data=False)
            m = lgb.train(lgb_params(float(y[tr_mask].mean())), d,
                          num_boost_round=args.iterations)
            out = m.predict(Xl.loc[va_mask])
            # 시즌 외삽 사전확률로 로짓 이동 (CatBoost baseline 과 같은 역할)
            pr, trm = prior_of(y), float(y[tr_mask].mean())
            z = np.log(np.clip(out, EPS, 1 - EPS) / (1 - np.clip(out, EPS, 1 - EPS)))
            z += np.log(pr / (1 - pr)) - np.log(trm / (1 - trm))
            del d, m
            gc.collect()
            return 1.0 / (1.0 + np.exp(-z))

        rows, b0 = [], None
        print(f"{chr(10)}  {'arm':<14}{'bss_raw':>11}{'Δ':>9}{'centered':>11}  출처")
        for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
            _rt = "" if args.rule == "linear_all" else f"_{args.rule}"
            npy = OUT / f"wb_{tg}__{arm}{_rt}__{args.fold}.npy"
            if npy.exists():
                m, src = bss(y_va, np.load(npy)), "cache"
            elif args.dry:
                print(f"  {arm:<14}{'':>11}{'':>9}{'':>11}  dry", flush=True)
                continue
            else:
                t0 = time.time()
                try:
                    if arm == "single":
                        pred = fit_cat(yv)
                    elif arm in ("split_ball", "split_count"):
                        pred = sum(fit_cat(v) for _, v in parts[arm])
                    elif arm == "lgbm":
                        pred = fit_lgb(yv)
                    elif arm == "cat_lgbm":
                        pred = 0.5 * fit_cat(yv) + 0.5 * fit_lgb(yv)
                    else:
                        print(f"  {arm:<14}  모르는 arm"); continue
                except Exception as exc:                          # noqa: BLE001
                    print(f"  {arm:<14}  실패: {type(exc).__name__}: "
                          f"{str(exc)[:56]}", flush=True)
                    continue
                pred = np.clip(pred, EPS, 1 - EPS)
                save_prediction(npy, pred, y_va, where=f"{tg}/{arm}")
                m, src = bss(y_va, pred), f"fit {time.time() - t0:.0f}s"
            if b0 is None:
                b0 = m["bss_raw"]
            d = m["bss_raw"] - b0
            mark = "" if abs(d) > sd else "  (잡음)"
            print(f"  {arm:<14}{m['bss_raw']:>11.1f}{d:>+9.1f}"
                  f"{m['bss_centered']:>11.1f}  {src}{mark}", flush=True)
            rows.append({"target": tg, "arm": arm + _rt, "rule": args.rule,
                         "fold": args.fold, "d": d, **m})
        all_rows += rows
        del fr, Xl
        gc.collect()

    if all_rows:
        t = pd.DataFrame(all_rows)
        p = OUT / f"way_base_{args.fold}.csv"
        if p.exists():
            t = pd.concat([pd.read_csv(p), t], ignore_index=True)
        t = t.drop_duplicates(["target", "arm", "fold"], keep="last")
        t.to_csv(p, index=False)
        print(f"{chr(10)}saved -> {p}")


if __name__ == "__main__":
    main()
