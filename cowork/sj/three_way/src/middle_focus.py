"""Stage M: middle 집중 — 아웃카운트 축 + 투수·타자 성향.

왜 middle 인가
    drop_ids(ID 제거) -414  -> 투수 정체성이 거의 전부다
    no_trackman      +28.6  -> TrackMan 은 오히려 방해
    전처리 조합 이득  +104   -> 세 타깃 중 조합 민감도 최고
    middle 은 "한가운데 실투" 다. 투수가 누구이고 어떤 상황에서 몰리는가가 핵심.

M1 아웃카운트 축 — S3 에서 outs 를 단독 축으로 건 적이 없다
    p_outs      EB(투수, 아웃) - EB(투수)
    ph_outs     EB(투수, 타자손, 아웃) - EB(투수, 타자손)
    outs_count  아웃 x 카운트 교차 (투수 없이)

M2 투수·타자 성향 — 지금은 원본 as-of 비율을 그대로 넣고 있다
    mix         구질 성향: 엔트로피, 주구질 편중, 속구-변화구 로그비
    mtend       실투 성향: asof_pitcher_middle_rate 와 prev1/3/5 이탈
    btend       타자 성향: asof_batter_middle_rate 와 투수 성향의 상성
    prior       asof_pitcher_middle_rate 를 로짓 사전확률로 (base_margin 형태)

판정
    fold 2024 목표 / fold 2023 강건성 관문.
    2023 에서 잡음(sd 2.69)만큼 나빠지면 기각 — lossguide 가 그렇게 걸렸다.

사용
    python middle_focus.py --fold 2024
    python middle_focus.py --fold 2024 --dry
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
from harness3 import DECISION_FOLD, LAB, OUT, SUCCESS, TARGETS, bss, load_labeled, seed_noise
from s3_layered import layered

SEED = 20262844
TARGET = "middle"
# 2026-08-19 재탐색(bayesian 위, 두 fold): no_trackman 은 f23 -11.2 로 기각됐다.
# 강건성 관문을 통과한 최고는 rate_multiscale (+0.9 / +81.8).
BASE_COMBO = ("id_frequency", "rate_multiscale", "temporal_cyclic")
BASE_ARM = {"p": {"bootstrap_type": "Bayesian", "bagging_temperature": 1.0}}


def mix_features(frame: pd.DataFrame) -> dict:
    """구질 성향. 행 단위 변환이라 행 독립."""
    n = lambda c: pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    f, b, o = n("asof_pitcher_fastball_rate"), n("asof_pitcher_breaking_rate"), \
        n("asof_pitcher_offspeed_rate")
    m = np.vstack([f, b, o])
    s = np.nansum(m, axis=0)
    p = np.where(s > 0, m / np.maximum(s, 1e-6), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=0)
    return {
        "mx_entropy": np.nan_to_num(ent).astype(np.float32),
        "mx_top": np.nan_to_num(np.nanmax(p, axis=0)).astype(np.float32),
        "mx_hhi": np.nan_to_num(np.nansum(p ** 2, axis=0)).astype(np.float32),
        "mx_fb_br": np.nan_to_num(np.log(np.clip(f, 1e-3, None)
                                         / np.clip(b, 1e-3, None))).astype(np.float32),
        "mx_fb_off": np.nan_to_num(np.log(np.clip(f, 1e-3, None)
                                          / np.clip(o, 1e-3, None))).astype(np.float32),
    }


def tendency_features(frame: pd.DataFrame) -> dict:
    """실투 성향과 타자 성향, 그리고 상성."""
    n = lambda c: pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    pm = n("asof_pitcher_middle_rate")
    bm = n("asof_batter_middle_rate")
    prev = np.vstack([n(f"asof_pitcher_prev{k}_game_middle_rate") for k in (1, 3, 5)])
    with np.errstate(invalid="ignore"):
        recent = np.nanmean(prev, axis=0)
    out = {
        "td_p_middle": np.nan_to_num(pm).astype(np.float32),
        "td_b_middle": np.nan_to_num(bm).astype(np.float32),
        "td_recent_gap": np.nan_to_num(recent - pm).astype(np.float32),
        "td_recent_sd": np.nan_to_num(np.nanstd(prev, axis=0)).astype(np.float32),
        # 상성 — 투수 실투 성향 x 타자 유도 성향
        "td_mul": np.nan_to_num(pm * bm).astype(np.float32),
        "td_dif": np.nan_to_num(pm - bm).astype(np.float32),
        "td_rat": np.nan_to_num(pm / np.clip(bm, 1e-3, None)).astype(np.float32),
    }
    for k, v in zip((1, 3, 5), prev):
        out[f"td_prev{k}_gap"] = np.nan_to_num(v - pm).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=DECISION_FOLD)
    ap.add_argument("--arms", default="base,p_outs,ph_outs,outs_count,mix,mtend,"
                                      "tend_all,mix+tend_all,all")
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
    yv = pd.to_numeric(labeled[TARGETS[TARGET]], errors="coerce").to_numpy(np.float64)
    ok = (labeled["label_ok"].to_numpy() == 1) & ~np.isnan(yv)
    tr_mask, va_mask = (season < args.fold) & ok, (season == args.fold) & ok
    y_va = yv[va_mask].astype("int8")
    league = float(yv[tr_mask].mean())
    sd = seed_noise(TARGET)

    s_ = pd.Series(yv[tr_mask]).groupby(
        pd.Series(season[tr_mask])).mean().sort_index()
    span = float(s_.index[-1]) - float(s_.index[0])
    prior = float(np.clip(float(s_.iloc[-1]) +
                          ((float(s_.iloc[-1]) - float(s_.iloc[0])) / span
                           if span > 0 else 0.0), 0.005, 0.995))
    bl = float(np.log(prior / (1 - prior)))

    print(f"타깃 {TARGET}  fold {args.fold}  학습 {tr_mask.sum():,}  검증 {va_mask.sum():,}")
    print(f"  기저율 {league:.4f}  외삽 {prior:.4f}  실제 {y_va.mean():.4f}  잡음 sd {sd:.2f}")
    print(f"  기준: 전처리 {'+'.join(BASE_COMBO)}  학습 bayesian", flush=True)

    static = build_component_unique(frame, enhanced, args.fold)
    forward = build_component_unique_forward(frame, enhanced, args.fold,
                                             cache={args.fold: static})
    base_fr, f1_features = make_features(frame, enhanced, args.fold, "F1", forward)
    for c in (SUCCESS, "season"):
        if c not in base_fr.columns:
            base_fr[c] = frame[c].to_numpy()
    cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
    pre_fr, pre_feats, pre_cats = T.build(base_fr, f1_features, cats0, BASE_COMBO,
                                          pd.Series(tr_mask, index=frame.index),
                                          args.fold)

    num = lambda c: pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    pid = frame["pitcher_id"].to_numpy()
    bh = frame["batter_hand"].astype(str).to_numpy()
    outs = num("outs_before").astype(int).astype(str)
    cnt = np.digitize(num("balls_before") * 3 + num("strikes_before"), [3, 6, 9])

    # M1 축 — 타깃(y_middle) 라벨로 만든다. 학습 행만.
    AX = {
        "p_outs":     ([("p", pid)], [("p", pid), ("o", outs)]),
        "ph_outs":    ([("p", pid), ("h", bh)], [("p", pid), ("h", bh), ("o", outs)]),
        "outs_count": ([("o", outs)], [("o", outs), ("c", cnt)]),
    }
    EXTRAS = {}
    for name, (ko, ki) in AX.items():
        e, med, contrib = layered(ko, ki, yv, tr_mask, league, name)
        EXTRAS[name] = e
        print(f"  축 {name:<12} 셀중앙 {int(med):>7,}  기여 {contrib:.3f}%"
              f"{'  ! 1% 초과' if contrib > 1 else ''}", flush=True)
    EXTRAS["mix"] = mix_features(frame)
    EXTRAS["mtend"] = tendency_features(frame)
    EXTRAS["tend_all"] = {**EXTRAS["mix"], **EXTRAS["mtend"]}

    P0 = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(P0.pop("half_life"))
    P0.update({"iterations": args.iterations, "learning_rate": args.learning_rate,
               "depth": args.depth, "random_seed": SEED, "task_type": "GPU",
               "devices": "0", "verbose": 0, **BASE_ARM["p"]})
    P0.pop("subsample", None)
    w = recency_weights(frame.loc[tr_mask, "season"], args.fold, half_life)

    rows, b0 = [], None
    print(f"{chr(10)}  {'arm':<16}{'피처':>6}{'bss_raw':>11}{'Δ':>9}"
          f"{'centered':>11}  출처")
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        parts = ([] if arm == "base" else
                 (["p_outs", "ph_outs", "outs_count", "tend_all"] if arm == "all"
                  else arm.split("+")))
        ex = {}
        for p_ in parts:
            if p_ not in EXTRAS:
                print(f"  {arm:<16}  모르는 구성 {p_}"); ex = None; break
            ex.update(EXTRAS[p_])
        if ex is None:
            continue
        npy = OUT / f"mf_{TARGET}__{arm}__{args.fold}.npy"
        if npy.exists():
            m, src, nf = bss(y_va, np.load(npy)), "cache", -1
        else:
            fr = M.add_columns(pre_fr, ex) if ex else pre_fr
            feats = list(dict.fromkeys(pre_feats + list(ex)))
            nf = len(feats)
            if args.dry:
                print(f"  {arm:<16}{nf:>6}{'':>11}{'':>9}{'':>11}  dry", flush=True)
                del fr; gc.collect(); continue
            assert_features_clean(feats, f"{TARGET}/{arm}")
            t0 = time.time()
            p_tr = Pool(fr.loc[tr_mask, feats], yv[tr_mask].astype("int8"),
                        cat_features=pre_cats, weight=w,
                        baseline=np.full(int(tr_mask.sum()), bl))
            p_va = Pool(fr.loc[va_mask, feats], y_va, cat_features=pre_cats,
                        baseline=np.full(int(va_mask.sum()), bl))
            mdl = CatBoostClassifier(**P0)
            mdl.fit(p_tr, eval_set=p_va, use_best_model=True)
            pred = mdl.predict_proba(p_va)[:, 1]
            save_prediction(npy, pred, y_va, where=f"{TARGET}/{arm}")
            m, src = bss(y_va, pred), f"fit {time.time() - t0:.0f}s"
            del fr, p_tr, p_va, mdl
            gc.collect()
        if b0 is None:
            b0 = m["bss_raw"]
        d = m["bss_raw"] - b0
        mark = "" if abs(d) > sd else "  (잡음)"
        print(f"  {arm:<16}{nf:>6}{m['bss_raw']:>11.1f}{d:>+9.1f}"
              f"{m['bss_centered']:>11.1f}  {src}{mark}", flush=True)
        rows.append({"target": TARGET, "arm": arm, "fold": args.fold,
                     "n_features": nf, "d": d, **m})
        pd.DataFrame(rows).to_csv(OUT / f"middle_focus_{args.fold}.csv", index=False)

    if rows:
        t = pd.DataFrame(rows).sort_values("bss_raw", ascending=False)
        t.to_csv(OUT / f"middle_focus_{args.fold}.csv", index=False)
        print(f"{chr(10)}  상위 3: " + " | ".join(
            f"{r.arm} {r.bss_raw:.0f}" for r in t.head(3).itertuples()))
        print(f"  목표 1300 까지 {1300 - t.bss_raw.max():+.0f}")


if __name__ == "__main__":
    main()
