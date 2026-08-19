"""S3: 타깃별 계층 차감 축.

1WAY 와의 결정적 차이
    1WAY 는 계층 차감 테이블을 **전부 control_success 라벨로** 만들었다.
    (component_features.make_platoon_table 등)
    3WAY 는 **그 타깃의 라벨로** 만든다.
        middle 모델의 플래툰 테이블은 y_middle 로,
        reverse 모델의 것은 y_reverse 로.
    "이 투수가 좌타 상대로 한가운데를 더 던지는가" 와
    "이 투수가 좌타 상대로 성공률이 낮은가" 는 다른 질문이다.

구조 (1WAY 최대 발견을 그대로)
    split = EB(키 + 축) − EB(키)
    주효과를 빼야 "남들보다 얼마나 다른가" 만 남는다. 1겹 +0.16 vs 2겹 +8.44.

S1·S2 가 시사한 것
    middle  투수 정체성이 거의 전부 (drop_ids −414) -> 투수 키 축이 들 것
    reverse ID 를 빼는 게 최선 (drop_ids +42) -> 투수 키 축이 안 들 수 있다
    ball    같음 (drop_ids +66)
    그래서 **투수 키 축과 투수 없는 축을 둘 다** 시험한다.

축 목록
    p_hand        EB(투수, 타자손) − EB(투수)                1WAY 최대 기여 축
    p_count       EB(투수, 타자손, 카운트) − EB(투수, 타자손)   1WAY 2위 축
    p_inning      EB(투수, 타자손, 이닝) − EB(투수, 타자손)
    p_workload    EB(투수, 부하5군) − EB(투수)                 미탐색
    nop_count     EB(타자손, 카운트) − EB(타자손)              투수 없이
    nop_situation EB(카운트, 이닝군, 주자상태) − 리그평균         투수 없이
    b_hand        EB(타자, 투수손) − EB(타자)                  타자 키

셀 크기 관문
    한 행 기여 1/(n+K) 가 1% 넘으면 경고를 찍는다 (자기 라벨 누수, METHOD.md 7).

사용
    python s3_layered.py --target middle,reverse,ball
    python s3_layered.py --target middle --base "id_frequency+no_trackman+temporal_cyclic"
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
from harness3 import DECISION_FOLD, LAB, OUT, SUCCESS, TARGETS, bss, load_labeled, seed_noise

SEED = 20262844
K = 300.0

# S2 에서 나온 타깃별 최고 조합 (outputs/screen_*_2024.csv 1위)
S2_BEST = {
    "middle": "id_frequency+no_trackman+temporal_cyclic",
    "reverse": "count_multiscale+drop_ids+trackman_quality",
    "ball": "drop_ids+no_trackman+rate_multiscale",
    "outside": "",
    "success": "",
}


def eb(num, den, league):
    return (num + K * league) / (den + K)


def layered(keys_outer, keys_inner, y, tr, league, name, warn=True):
    """split = EB(inner) − EB(outer). inner 가 outer 를 포함해야 한다.

    keys_* 는 (이름, 배열) 리스트. 학습 행에서만 테이블을 만들고 전 행에 조인한다.
    """
    n = len(y)
    out_cols = [k for k, _ in keys_outer]
    in_cols = [k for k, _ in keys_inner]
    full = pd.DataFrame({k: v for k, v in keys_inner})
    d = full.loc[tr].copy()
    d["_y"] = y[tr]
    g_in = d.groupby(in_cols)["_y"].agg(["sum", "size"])
    g_out = d.groupby(out_cols)["_y"].agg(["sum", "size"])
    e_in = eb(g_in["sum"], g_in["size"], league).rename("e_in")
    e_out = eb(g_out["sum"], g_out["size"], league).rename("e_out")

    med = float(g_in["size"].median())
    contrib = 1.0 / (med + K) * 100
    if warn and contrib > 1.0:
        print(f"      ! {name} 셀 기여 {contrib:.2f}% (>1%) — 자기 라벨 누수 주의",
              flush=True)

    tbl = e_in.reset_index().merge(e_out.reset_index(), on=out_cols, how="left")
    tbl["split"] = tbl["e_in"] - tbl["e_out"]
    tbl["rel"] = (g_in["size"].reindex(
        pd.MultiIndex.from_frame(tbl[in_cols]) if len(in_cols) > 1
        else tbl[in_cols[0]]).to_numpy())
    tbl["rel"] = tbl["rel"] / (tbl["rel"] + K)
    j = full.merge(tbl[in_cols + ["split", "rel"]], on=in_cols, how="left")
    s = np.nan_to_num(j["split"].to_numpy(np.float64), nan=0.0)
    r = np.nan_to_num(j["rel"].to_numpy(np.float64), nan=0.0)
    return {f"lay_{name}_split": s.astype(np.float32),
            f"lay_{name}_rel": r.astype(np.float32),
            f"lay_{name}_split_w": (s * r).astype(np.float32)}, med, contrib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="middle,reverse,outside")
    ap.add_argument("--base", default="", help="비우면 S2_BEST 사용")
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

    static = build_component_unique(frame, enhanced, args.fold)
    forward = build_component_unique_forward(frame, enhanced, args.fold,
                                             cache={args.fold: static})
    base_fr, f1_features = make_features(frame, enhanced, args.fold, "F1", forward)
    for c in (SUCCESS, "season"):
        if c not in base_fr.columns:
            base_fr[c] = frame[c].to_numpy()

    params = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(params.pop("half_life"))
    params.update({"iterations": args.iterations, "learning_rate": args.learning_rate,
                   "depth": args.depth, "random_seed": SEED, "task_type": "GPU",
                   "devices": "0", "verbose": 0})

    num = lambda c: pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    pid = frame["pitcher_id"].to_numpy()
    bid = frame["batter_id"].to_numpy()
    bh = frame["batter_hand"].astype(str).to_numpy()
    ph = frame["pitcher_hand"].astype(str).to_numpy()
    cnt = np.digitize(num("balls_before") * 3 + num("strikes_before"), [3, 6, 9])
    inn = np.digitize(num("inning"), [4, 7, 10])
    wl = np.digitize(num("asof_pitcher_n"), [100, 500, 2000, 4000])
    bs = frame["base_state"].astype(str).to_numpy()

    AXES = {
        "p_hand":        ([("p", pid)], [("p", pid), ("h", bh)]),
        "p_count":       ([("p", pid), ("h", bh)], [("p", pid), ("h", bh), ("c", cnt)]),
        "p_inning":      ([("p", pid), ("h", bh)], [("p", pid), ("h", bh), ("i", inn)]),
        "p_workload":    ([("p", pid)], [("p", pid), ("w", wl)]),
        "nop_count":     ([("h", bh)], [("h", bh), ("c", cnt)]),
        "nop_situation": ([("c", cnt)], [("c", cnt), ("i", inn), ("b", bs)]),
        "b_hand":        ([("b", bid)], [("b", bid), ("ph", ph)]),
    }

    rows = []
    for target in [t.strip() for t in args.target.split(",") if t.strip()]:
        col = TARGETS[target]
        yv = pd.to_numeric(labeled[col], errors="coerce").to_numpy(np.float64)
        ok = ((labeled["label_ok"].to_numpy() == 1) if target != "success"
              else np.ones(len(labeled), bool)) & ~np.isnan(yv)
        tr_mask, va_mask = train_all & ok, valid_all & ok
        y_va = yv[va_mask].astype("int8")
        league = float(yv[tr_mask].mean())
        sd = seed_noise(target)
        combo = tuple(sorted((args.base or S2_BEST.get(target, "")).split("+"))) \
            if (args.base or S2_BEST.get(target)) else ()
        combo = tuple(c for c in combo if c)

        print(f"{chr(10)}{'=' * 100}")
        print(f"타깃 {target}  라벨 {col}  기저율 {league:.4f}  잡음 sd {sd:.2f}")
        print(f"  기준 조합 (S2 최고): {'+'.join(combo) or 'baseline'}")
        print(f"  테이블도 {col} 라벨로 만든다 — 1WAY 는 전부 control_success 였다")
        print("=" * 100, flush=True)

        weights = recency_weights(frame.loc[tr_mask, "season"], args.fold, half_life)
        tr_series = pd.Series(tr_mask, index=frame.index)
        cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
        pre_fr, pre_feats, pre_cats = T.build(base_fr, f1_features, cats0, combo,
                                              tr_series, args.fold)

        def run(tag, extras):
            npy = OUT / f"s3_{target}__{tag}__{args.fold}.npy"
            if npy.exists():
                return bss(y_va, np.load(npy)), "cache", -1
            fr = pre_fr if not extras else M.add_columns(pre_fr, extras)
            feats = list(dict.fromkeys(list(pre_feats) + list(extras)))
            missing = [c for c in feats if c not in fr.columns]
            assert not missing, f"{tag}: 프레임에 없는 열 {missing[:3]}"
            assert len(feats) == len(pre_feats) + len(extras), (
                f"{tag}: 피처 수 불일치 {len(feats)} != {len(pre_feats)}+{len(extras)}")
            if args.dry:
                return {"bss_centered": np.nan, "bss_norm": np.nan,
                        "offset": np.nan}, "dry", len(feats)
            t0 = time.time()
            assert_features_clean(feats, target)
            p_tr = Pool(fr.loc[tr_mask, feats], yv[tr_mask].astype("int8"),
                        cat_features=pre_cats, weight=weights)
            p_va = Pool(fr.loc[va_mask, feats], y_va, cat_features=pre_cats)
            m = CatBoostClassifier(**params)
            m.fit(p_tr, eval_set=p_va, use_best_model=True)
            pred = m.predict_proba(p_va)[:, 1]
            save_prediction(npy, pred, y_va, where=f"{target}")
            r = bss(y_va, pred)
            del fr, p_tr, p_va, m
            gc.collect()
            return r, f"fit {time.time() - t0:.0f}s", len(feats)

        b0, src, nf = run("base", {})
        print(f"  {'축':<16}{'셀중앙':>8}{'기여%':>8}{'bss_ctr':>10}{'Δ':>9}"
              f"{'피처':>6}  출처")
        print(f"  {'(기준)':<16}{'':>8}{'':>8}{b0['bss_centered']:>10.2f}"
              f"{0.0:>+9.2f}{nf:>6}  {src}", flush=True)
        rows.append({"target": target, "axis": "(기준)", "combo": "+".join(combo),
                     **b0, "d": 0.0})

        for name, (ko, ki) in AXES.items():
            extras, med, contrib = layered(ko, ki, yv, tr_mask, league, name)
            r, src, nf = run(name, extras)
            d = r["bss_centered"] - b0["bss_centered"]
            mark = "" if abs(d) > sd else "  (잡음)"
            print(f"  {name:<16}{int(med):>8,}{contrib:>8.3f}"
                  f"{r['bss_centered']:>10.2f}{d:>+9.2f}{nf:>6}  {src}{mark}",
                  flush=True)
            rows.append({"target": target, "axis": name, "combo": "+".join(combo),
                         "cell_median": med, "cell_contrib": contrib, **r, "d": d})
            pd.DataFrame(rows).to_csv(OUT / f"s3_layered_{args.fold}.csv", index=False)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"s3_layered_{args.fold}.csv", index=False)
    print(f"{chr(10)}{'=' * 100}{chr(10)}타깃별 최선 축{chr(10)}{'=' * 100}")
    for tg in t["target"].unique():
        s = t[(t.target == tg) & (t.axis != "(기준)")].nlargest(3, "d")
        print(f"  {tg:<9}" + " | ".join(f"{r.axis} {r.d:+.1f}" for r in s.itertuples()))
    print(f"{chr(10)}saved -> {OUT / f's3_layered_{args.fold}.csv'}")


if __name__ == "__main__":
    main()
