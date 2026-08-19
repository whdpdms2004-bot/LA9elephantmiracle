"""3WAY: 하위 확률마다 전처리를 따로 스크리닝한다.

1WAY 는 다섯 성분이 같은 111피처를 공유하고 트리 파라미터만 달랐다.
여기서는 타깃마다 어떤 전처리가 맞는지를 따로 찾는다.

preprocess_lab 의 변환 레지스트리를 그대로 재사용한다 (복사하지 않음).
그래서 랩에 새 변환을 추가하면 여기서도 바로 쓸 수 있다.

사용
    # 타깃 하나, 단일 전처리 전부
    python screen_target.py --target middle

    # 세 타깃 순차 (GPU 하나씩)
    python screen_target.py --target middle,reverse,ball

    # 조합 빔 서치
    python screen_target.py --target middle --beam 3 --rounds 3

    # 학습 없이 확인
    python screen_target.py --target middle --dry
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
from guards import (assert_features_clean,
                    save_prediction, train_season_trend)
from harness3 import (AUX_FOLD, CAMPAIGN, DECISION_FOLD, LAB, OUT, SUCCESS,
                      TARGETS, bss, load_labeled, seed_noise, verdict)

SEED = 20262844


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="middle,reverse,outside",
                    help="middle,reverse,ball,outside,success 중 쉼표 구분")
    ap.add_argument("--fold", type=int, default=DECISION_FOLD)
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--combos", default="")
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--learning-rate", type=float, default=0.015)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--arm", default="base",
                    help="학습 방식 arm (train_arms.ARM_SPECS). 기본 base")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    targets = [t.strip() for t in args.target.split(",") if t.strip()]
    for t in targets:
        if t not in TARGETS:
            raise ValueError(f"모르는 타깃: {t}. 가능: {sorted(TARGETS)}")

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
    from train_arms import ARM_SPECS, tree_params_for
    _spec = ARM_SPECS.get(args.arm)
    if _spec is None:
        raise SystemExit(f"모르는 arm: {args.arm}. 가능: {sorted(ARM_SPECS)}")
    print(f"학습 방식 arm: {args.arm}  {_spec}")

    reg = T.load_all()
    print(f"등록된 변환 {len(reg)}개 (preprocess_lab 재사용)")

    frame, enhanced = load_enhanced_frame()
    labeled = load_labeled()
    assert (frame["row_id"].to_numpy() == labeled["row_id"].to_numpy()).all(), \
        "행 순서 불일치 — 라벨 병합 전제가 깨졌다"

    season = frame["season"].to_numpy()
    train_all = season < args.fold
    valid_all = season == args.fold

    static = build_component_unique(frame, enhanced, args.fold)
    forward = build_component_unique_forward(frame, enhanced, args.fold,
                                             cache={args.fold: static})
    base, f1_features = make_features(frame, enhanced, args.fold, "F1", forward)
    for col in (SUCCESS, "season"):
        if col not in base.columns:
            base[col] = frame[col].to_numpy()

    params = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(params.pop("half_life"))
    params.update({"iterations": args.iterations, "learning_rate": args.learning_rate,
                   "depth": args.depth, "random_seed": SEED, "task_type": "GPU",
                   "devices": "0", "verbose": 0})

    all_rows = []
    for target in targets:
        col = TARGETS[target]
        yv = pd.to_numeric(labeled[col], errors="coerce").to_numpy(np.float64)
        ok = ((labeled["label_ok"].to_numpy() == 1) if target != "success"
              else np.ones(len(labeled), bool)) & ~np.isnan(yv)
        tr_mask = train_all & ok
        va_mask = valid_all & ok
        y_va = yv[va_mask].astype("int8")
        sd = seed_noise(target)
        print(f"{chr(10)}{'=' * 96}")
        print(f"타깃 {target}  ({col})   fold {args.fold}")
        print(f"  학습 {int(tr_mask.sum()):,}행  검증 {int(va_mask.sum()):,}행  "
              f"기저율 {y_va.mean():.4f}  null {y_va.mean()*(1-y_va.mean()):.4f}  "
              f"잡음 sd 추정 {sd:.2f}")
        print("=" * 96, flush=True)

        _hl = float(_spec.get("half_life", half_life))
        weights = np.asarray(recency_weights(
            frame.loc[tr_mask, "season"], args.fold, _hl), np.float64).copy()
        if "w_f" in _spec:
            weights[(frame["game_type"].astype(str).to_numpy() == "F")[tr_mask]] *= _spec["w_f"]
        # 시즌 외삽 사전확률. 1WAY 는 성분마다 base_score 를 이렇게 외삽했는데
        # 3WAY 하위 모델은 기본값을 써서 오프셋이 +0.011~+0.025 났다 (페널티 47~250).
        _s = pd.Series(yv[tr_mask]).groupby(
            pd.Series(frame.loc[tr_mask, "season"].to_numpy())).mean().sort_index()
        _span = float(_s.index[-1]) - float(_s.index[0])
        _prior = float(np.clip(
            float(_s.iloc[-1]) + ((float(_s.iloc[-1]) - float(_s.iloc[0])) / _span
                                  if _span > 0 else 0.0), 0.005, 0.995))
        print(f"  시즌 외삽 사전확률 {_prior:.4f} "
              f"(직전 시즌 {float(_s.iloc[-1]):.4f}, 검증 실제 {y_va.mean():.4f})",
              flush=True)
        tr_series = pd.Series(tr_mask, index=frame.index)
        scored: dict[tuple, dict] = {}

        def score(names):
            key = tuple(sorted(names))
            if key in scored:
                return scored[key]
            tag = "+".join(key) if key else "baseline"
            npy = OUT / (f"{target}__{tag}__{args.fold}.npy" if args.arm == "base"
                         else f"{target}__{tag}__{args.arm}__{args.fold}.npy")
            if npy.exists():
                m = bss(y_va, np.load(npy))
                scored[key] = {"target": target, "combo": tag, "n_atoms": len(key),
                               "n_features": -1, "src": "cache", **m}
                return scored[key]
            cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
            fr, feats, cats = T.build(base, f1_features, cats0, key,
                                      tr_series, args.fold)
            if args.dry:
                scored[key] = {"target": target, "combo": tag, "n_atoms": len(key),
                               "n_features": len(feats), "src": "dry",
                               "bss_raw": np.nan, "bss_centered": np.nan,
                               "bss_norm": np.nan, "offset": np.nan}
                del fr
                gc.collect()
                return scored[key]
            t0 = time.time()
            _b = np.log(_prior / (1 - _prior))
            assert_features_clean(feats, target)
            pool_tr = Pool(fr.loc[tr_mask, feats], yv[tr_mask].astype("int8"),
                           cat_features=cats, weight=weights,
                           baseline=np.full(int(tr_mask.sum()), _b, np.float64))
            pool_va = Pool(fr.loc[va_mask, feats], y_va, cat_features=cats,
                           baseline=np.full(int(va_mask.sum()), _b, np.float64))
            _p = dict(params)
            if _spec.get("treeparam"):
                _p.update(tree_params_for(float(yv[tr_mask].mean())))
            _p.update(_spec.get("p", {}))
            if _p.get("grow_policy") == "Lossguide":
                _p.pop("depth", None)
            elif "max_leaves" in _p:
                _p.pop("max_leaves", None)
            if _p.get("bootstrap_type") == "Bayesian":
                _p.pop("subsample", None)
            model = CatBoostClassifier(**_p)
            model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
            pred = model.predict_proba(pool_va)[:, 1]
            save_prediction(npy, pred, y_va, where=f"{target}")
            m = bss(y_va, pred)
            scored[key] = {"target": target, "combo": tag, "n_atoms": len(key),
                           "n_features": len(feats),
                           "src": f"fit {time.time() - t0:.0f}s", **m}
            del fr, pool_tr, pool_va, model
            gc.collect()
            return scored[key]

        b0 = score(())
        print(f"  {'조합':<44}{'bss_ctr':>10}{'Δ':>9}{'bss_norm':>10}"
              f"{'오프셋':>9}  출처")
        print(f"  {'baseline':<44}{b0['bss_centered']:>10.2f}{0.0:>+9.2f}"
              f"{b0['bss_norm']:>10.2f}{b0['offset']:>+9.4f}  {b0['src']}", flush=True)

        def show(r):
            d = r["bss_centered"] - b0["bss_centered"]
            mark = "" if abs(d) > sd else "  (잡음)"
            print(f"  {r['combo']:<44}{r['bss_centered']:>10.2f}{d:>+9.2f}"
                  f"{r['bss_norm']:>10.2f}{r['offset']:>+9.4f}  {r['src']}{mark}",
                  flush=True)

        atoms = sorted(reg)
        if args.combos:
            for spec in [c.strip() for c in args.combos.split(",") if c.strip()]:
                key = () if spec == "baseline" else tuple(spec.split("+"))
                if T.compatible(key):
                    show(score(key))
        else:
            for a in atoms:
                show(score((a,)))
                pd.DataFrame(scored.values()).to_csv(
                    OUT / (f"screen_{target}_{args.fold}.csv" if args.arm == "base"
                       else f"screen_{target}_{args.arm}_{args.fold}.csv"), index=False)

        if args.beam:
            beam = [c for c, _ in sorted(
                [(k, v) for k, v in scored.items() if len(k) == 1],
                key=lambda t: -t[1]["bss_centered"])[:args.beam]]
            for rd in range(2, args.rounds + 1):
                print(f"  --- 라운드 {rd} ---", flush=True)
                cands = {tuple(sorted(c + (a,))) for c in beam for a in atoms
                         if a not in c and T.compatible(c + (a,))}
                res = []
                for c in sorted(cands):
                    res.append((c, score(c)))
                    show(res[-1][1])
                    pd.DataFrame(scored.values()).to_csv(
                        OUT / (f"screen_{target}_{args.fold}.csv" if args.arm == "base"
                       else f"screen_{target}_{args.arm}_{args.fold}.csv"), index=False)
                prev = max(scored[c]["bss_centered"] for c in beam)
                beam = [c for c, _ in sorted(
                    res, key=lambda t: -t[1]["bss_centered"])[:args.beam]]
                if not beam or max(scored[c]["bss_centered"] for c in beam) <= prev:
                    print("  개선 없음 - 종료", flush=True)
                    break

        t = pd.DataFrame(scored.values()).sort_values("bss_centered", ascending=False)
        t["d_centered"] = t["bss_centered"] - b0["bss_centered"]
        t["verdict"] = [verdict({DECISION_FOLD: d}, target) for d in t["d_centered"]]
        t.to_csv(OUT / (f"screen_{target}_{args.fold}.csv" if args.arm == "base"
                       else f"screen_{target}_{args.arm}_{args.fold}.csv"), index=False)
        all_rows.append(t)
        print(f"{chr(10)}  [{target}] 상위 6")
        print(t.head(6)[["combo", "bss_centered", "d_centered", "bss_norm",
                         "verdict"]].round(2).to_string(index=False))

    if all_rows:
        pd.concat(all_rows).to_csv(OUT / f"screen_all_{args.fold}.csv", index=False)
        print(f"{chr(10)}saved -> {OUT / f'screen_all_{args.fold}.csv'}")


if __name__ == "__main__":
    main()
