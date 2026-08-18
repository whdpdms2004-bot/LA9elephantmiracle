"""Stage 1: 1WAY 에서 통했던 학습 방식을 3WAY 타깃별로 이식한다.

1WAY 에서 채택됐는데 3WAY 에 아직 없는 것들 (전부 학습 방식, 로짓·배깅과 무관)

    fw020      F행(포스트시즌) 학습 가중치 0.20        1WAY +4.07
    short05    짧은 등판 학습 가중치 0.5               1WAY Public +4
    treeparam  기저율별 트리 파라미터                  1WAY +0.91
    interact   2차 상호작용 (곱·차·비, 제곱 제외)       1WAY Public +2

핵심 질문
    타깃마다 어느 학습 방식이 듣는가. 1WAY(success 타깃) 와 같은가 다른가.
    S1 에서 전처리 선호가 타깃 간 음의 상관이었으므로 학습 방식도 갈릴 수 있다.

주의
    2차 상호작용은 1WAY 에서 **곱과 차·비만 두 fold 양수, 제곱은 무의미** 였다.
    그대로 따른다.

사용
    python train_arms.py --target middle,reverse,ball,outside --fold 2024
    python train_arms.py --arms base,fw020 --dry
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
from harness3 import (CAMPAIGN, DECISION_FOLD, LAB, OUT, SUCCESS, TARGETS, bss,
                      load_labeled, seed_noise)

SEED = 20262844
BEST_COMBO = {
    "middle": "id_frequency+no_trackman+temporal_cyclic",
    "reverse": "count_multiscale+drop_ids+trackman_quality",
    "ball": "drop_ids+no_trackman+rate_multiscale",
    "outside": "drop_ids+no_trackman+rate_multiscale",
    "mr": "id_frequency+no_trackman+temporal_cyclic",
}
ARMS = ["base", "fw020", "short05", "treeparam", "interact",
        "fw020+short05", "all"]


def short_outing_mask(frame: pd.DataFrame) -> np.ndarray:
    """평소보다 크게 짧은 등판의 행. 1WAY V63 의 정의를 따른다.

    등판 구분은 asof_pitcher_prev1_game_success_rate 가 일정한 구간(런)으로 잡는다.
    학습 행에서만 쓰므로 test 행간 집계에 해당하지 않는다.
    """
    pid = frame["pitcher_id"].to_numpy()
    prev1 = pd.to_numeric(frame["asof_pitcher_prev1_game_success_rate"],
                          errors="coerce").to_numpy()
    n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy()
    key = pd.DataFrame({"p": pid, "v": np.nan_to_num(prev1, nan=-1.0)})
    grp = (key != key.shift()).any(axis=1).cumsum()
    size = grp.map(grp.value_counts())
    med = pd.Series(size).groupby(pd.Series(pid)).transform("median")
    return (size.to_numpy() < 0.5 * np.maximum(med.to_numpy(), 1)).astype(bool)


def interactions(frame: pd.DataFrame, feats: list[str]) -> dict:
    """2차 상호작용. 곱과 차·비만 (1WAY: 제곱은 무의미)."""
    cand = [c for c in ("asof_pitcher_success_rate", "asof_batter_success_rate",
                        "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
                        "asof_pitcher_ball_rate", "li") if c in frame.columns]
    num = lambda c: pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    out = {}
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = num(cand[i]), num(cand[j])
            s = f"{cand[i][:14]}_{cand[j][:14]}"
            out[f"ix_mul_{s}"] = (a * b).astype(np.float32)
            out[f"ix_dif_{s}"] = (a - b).astype(np.float32)
            out[f"ix_rat_{s}"] = (a / np.clip(np.abs(b), 1e-3, None)).astype(np.float32)
    return out


def tree_params_for(rate: float) -> dict:
    """기저율 구간별 트리 파라미터. 1WAY 의 params_for 를 CatBoost 로 옮겼다."""
    if rate < 0.06:
        return {"depth": 5, "l2_leaf_reg": 12.0, "min_data_in_leaf": 256}
    if rate < 0.15:
        return {"depth": 6, "l2_leaf_reg": 8.0, "min_data_in_leaf": 128}
    return {"depth": 8, "l2_leaf_reg": 3.0, "min_data_in_leaf": 64}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="middle,reverse,ball,outside")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--fold", type=int, default=DECISION_FOLD)
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--learning-rate", type=float, default=0.015)
    ap.add_argument("--depth", type=int, default=8)
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

    frame, enhanced = load_enhanced_frame()
    labeled = load_labeled()
    season = frame["season"].to_numpy()
    train_all, valid_all = season < args.fold, season == args.fold
    is_f = frame["game_type"].astype(str).to_numpy() == "F"
    short = short_outing_mask(frame)
    print(f"F행 {is_f.sum():,}  짧은 등판 {short.sum():,} / {len(frame):,}", flush=True)

    static = build_component_unique(frame, enhanced, args.fold)
    forward = build_component_unique_forward(frame, enhanced, args.fold,
                                             cache={args.fold: static})
    base_fr, f1_features = make_features(frame, enhanced, args.fold, "F1", forward)
    for c in (SUCCESS, "season"):
        if c not in base_fr.columns:
            base_fr[c] = frame[c].to_numpy()

    P0 = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(P0.pop("half_life"))
    P0.update({"iterations": args.iterations, "learning_rate": args.learning_rate,
               "depth": args.depth, "random_seed": SEED, "task_type": "GPU",
               "devices": "0", "verbose": 0})

    rows = []
    for target in [t.strip() for t in args.target.split(",") if t.strip()]:
        col = TARGETS[target]
        yv = pd.to_numeric(labeled[col], errors="coerce").to_numpy(np.float64)
        ok = ((labeled["label_ok"].to_numpy() == 1) if target != "success"
              else np.ones(len(labeled), bool)) & ~np.isnan(yv)
        tr_mask, va_mask = train_all & ok, valid_all & ok
        y_va = yv[va_mask].astype("int8")
        league_tr = float(yv[tr_mask].mean())
        sd = seed_noise(target)
        combo = tuple(sorted(c for c in BEST_COMBO[target].split("+") if c))

        s_ = pd.Series(yv[tr_mask]).groupby(
            pd.Series(frame.loc[tr_mask, "season"].to_numpy())).mean().sort_index()
        span = float(s_.index[-1]) - float(s_.index[0])
        prior = float(np.clip(float(s_.iloc[-1]) +
                              ((float(s_.iloc[-1]) - float(s_.iloc[0])) / span
                               if span > 0 else 0.0), 0.005, 0.995))
        bl = float(np.log(prior / (1 - prior)))

        print(f"{chr(10)}{'=' * 104}")
        print(f"타깃 {target} ({col})  fold {args.fold}  기저율 {league_tr:.4f}  "
              f"외삽 {prior:.4f}  잡음 sd {sd:.2f}")
        print(f"  전처리: {'+'.join(combo)}")
        print("=" * 104, flush=True)

        cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
        pre_fr, pre_feats, pre_cats = T.build(base_fr, f1_features, cats0, combo,
                                              pd.Series(tr_mask, index=frame.index),
                                              args.fold)
        ix = interactions(frame, pre_feats)

        print(f"  {'arm':<16}{'피처':>6}{'학습행':>10}{'bss_ctr':>10}{'Δ':>9}"
              f"{'bss_norm':>10}  출처")
        b0 = None
        for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
            npy = OUT / f"tr_{target}__{arm}__{args.fold}.npy"
            parts = set(arm.split("+")) if arm != "all" else {
                "fw020", "short05", "treeparam", "interact"}
            if npy.exists():
                m, src, nf, ntr = bss(y_va, np.load(npy)), "cache", -1, -1
            else:
                w = recency_weights(frame.loc[tr_mask, "season"], args.fold, half_life)
                w = np.asarray(w, np.float64).copy()
                if "fw020" in parts:
                    w[is_f[tr_mask]] *= 0.20
                if "short05" in parts:
                    w[short[tr_mask]] *= 0.50
                params = dict(P0)
                if "treeparam" in parts:
                    params.update(tree_params_for(league_tr))
                fr = M.add_columns(pre_fr, ix) if "interact" in parts else pre_fr
                feats = (list(dict.fromkeys(pre_feats + list(ix)))
                         if "interact" in parts else pre_feats)
                nf, ntr = len(feats), int(tr_mask.sum())
                if args.dry:
                    print(f"  {arm:<16}{nf:>6}{ntr:>10,}{'':>10}{'':>9}{'':>10}  dry",
                          flush=True)
                    continue
                t0 = time.time()
                p_tr = Pool(fr.loc[tr_mask, feats], yv[tr_mask].astype("int8"),
                            cat_features=pre_cats, weight=w,
                            baseline=np.full(ntr, bl))
                p_va = Pool(fr.loc[va_mask, feats], y_va, cat_features=pre_cats,
                            baseline=np.full(int(va_mask.sum()), bl))
                mdl = CatBoostClassifier(**params)
                mdl.fit(p_tr, eval_set=p_va, use_best_model=True)
                pred = mdl.predict_proba(p_va)[:, 1]
                np.save(npy, pred)
                m, src = bss(y_va, pred), f"fit {time.time() - t0:.0f}s"
                del fr, p_tr, p_va, mdl
                gc.collect()
            if b0 is None:
                b0 = m["bss_centered"]
            d = m["bss_centered"] - b0
            mark = "" if abs(d) > sd else "  (잡음)"
            print(f"  {arm:<16}{nf:>6}{ntr:>10,}{m['bss_centered']:>10.2f}{d:>+9.2f}"
                  f"{m['bss_norm']:>10.2f}  {src}{mark}", flush=True)
            rows.append({"target": target, "arm": arm, "fold": args.fold,
                         "n_features": nf, "n_train": ntr, "d": d, **m})
            pd.DataFrame(rows).to_csv(OUT / f"train_arms_{args.fold}.csv", index=False)

    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(OUT / f"train_arms_{args.fold}.csv", index=False)
        print(f"{chr(10)}{'=' * 104}{chr(10)}타깃별 최선 arm{chr(10)}{'=' * 104}")
        for tg in t["target"].unique():
            s = t[(t.target == tg) & (t.arm != "base")].nlargest(3, "d")
            print(f"  {tg:<9}" + " | ".join(f"{r.arm} {r.d:+.1f}" for r in s.itertuples()))
        print(f"{chr(10)}saved -> {OUT / f'train_arms_{args.fold}.csv'}")


if __name__ == "__main__":
    main()
