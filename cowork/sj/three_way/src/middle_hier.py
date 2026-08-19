"""middle 조건부 분해 — 가법 분할의 다음 단계.

가법 분할(split_ball, 실측 f24 854.4)이 왜 먹혔나
    y_mb = m&ball 를 **전 행** 에서 학습한다. ball 이 아닌 행은 전부 음성이므로
    모델이 "이 행이 ball 인가" 와 "ball 이면 middle 인가" 를 한 모델에서 섞어 푼다.

조건부 분해는 그 둘을 분리한다
    p(m) = p(b) * p(m|b) + (1-p(b)) * p(m|~b)
    p(m|b)   ball 행만으로 학습
    p(m|~b)  비ball 행만으로 학습
    p(b)     ball 모델. 3WAY 보조 way 로 이미 정의돼 있다.

    각 조건부 모델은 자기 부분집합만 보므로 기저율이 얇아지지 않는다.
    가법 분할의 y_mb 기저율 0.027 대비 훨씬 진한 신호로 학습한다.

arm
    hier          p(b) 를 **이미 학습해둔 ball way 예측** 으로 결합
    hier_fitb     p(b) 를 이 스크립트에서 새로 적합 (진단용 — 아래 참고)
    hier_prior_b  p(b) 를 학습 시즌 추세 상수로 (ball 모델 없이)
    hier_true_b   p(b) 대신 실제 ball 을 넣은 상한선 — 분해 가치 측정용

왜 ball 예측을 새로 적합하지 않나 (실측)
    이 스크립트에서 기본 설정으로 적합한 ball 모델은 평균이 무너졌다.
        fold 2024  예측 평균 0.2335 (실제 0.3657)
        fold 2023  예측 평균 0.2547 (실제 0.3684)
    그 결과 hier 가 f24 912.1 / f23 141.1 로 fold 간에 갈렸다 —
    f24 의 좋은 점수는 ball 저평가가 middle 저평가를 상쇄한 우연이었다.
    반면 ball way 에서 이미 만든 예측은 f24 BSS 1705~1726, 평균 0.3692 로
    사실상 완벽히 보정돼 있다. 새로 적합할 이유가 없다.

주의
    hier_true_b 는 검증 라벨(y_ball)을 쓰므로 **절대 제출 경로에 못 넣는다.**
    분해 자체에 가치가 있는지 / p(b) 오차가 얼마나 깎아먹는지 가르는 진단이다.
    그래서 저장하지 않는다.

판정
    fold 2024 목표 / fold 2023 강건성 관문 (sd 2.69).

사용
    python middle_hier.py --fold 2024
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
from harness3 import (DECISION_FOLD, LAB, OUT, SUCCESS, bss, load_labeled,
                      seed_noise)

SEED = 20262844
EPS = 1e-7


def cat_params(base: dict, rate: float) -> dict:
    p = dict(base)
    p.update({"bootstrap_type": "Bayesian", "bagging_temperature": 1.0})
    p.pop("subsample", None)
    if rate < 0.06:
        p.update({"depth": 6, "l2_leaf_reg": 10.0})
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=DECISION_FOLD)
    ap.add_argument("--arms", default="hier,hier_fitb,hier_prior_b,hier_true_b")
    ap.add_argument("--ball-pred", default="ball__drop_ids+no_trackman+rate_multiscale",
                    help="p(b) 로 쓸 저장된 ball 예측 (두 fold 모두 있는 것)")
    ap.add_argument("--combo", default="rate_multiscale",
                    help="재훑기 최고 조합. id_frequency 를 뺀 것이 middle 에 유리했다")
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--tag", default="")
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
    tr, va = (season < args.fold) & ok, (season == args.fold) & ok
    y_va = ymid[va].astype("int8")
    sd = seed_noise("middle")

    b1, b0 = yball == 1, yball == 0
    print(f"middle 조건부 분해  fold {args.fold}  학습 {tr.sum():,}  검증 {va.sum():,}")
    print(f"  p(ball)     학습 {yball[tr].mean():.4f}")
    print(f"  p(m|ball)   학습 {ymid[tr & b1].mean():.4f}  ({int((tr & b1).sum()):,}행)")
    print(f"  p(m|~ball)  학습 {ymid[tr & b0].mean():.4f}  ({int((tr & b0).sum()):,}행)")
    print(f"  참고: 가법분할 y_mb 기저율 {((ymid == 1) & b1)[tr].mean():.4f} — 신호가 훨씬 얇다")
    print(f"  잡음 sd {sd:.2f}", flush=True)

    static = build_component_unique(frame, enhanced, args.fold)
    forward = build_component_unique_forward(frame, enhanced, args.fold,
                                             cache={args.fold: static})
    base_fr, f1_features = make_features(frame, enhanced, args.fold, "F1", forward)
    for c in (SUCCESS, "season"):
        if c not in base_fr.columns:
            base_fr[c] = frame[c].to_numpy()
    cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
    combo = tuple(c for c in args.combo.split("+") if c)
    print(f"  전처리 {'+'.join(combo) or 'baseline'}", flush=True)
    fr, feats, cat_cols = T.build(base_fr, f1_features, cats0, combo,
                                  pd.Series(tr, index=frame.index), args.fold)
    assert_features_clean(feats, "middle")

    P0 = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(P0.pop("half_life"))
    P0.update({"iterations": args.iterations, "learning_rate": 0.015, "depth": 8,
               "random_seed": SEED, "task_type": "GPU", "devices": "0", "verbose": 0})

    def fit(y, tr_m, va_m):
        """tr_m 행으로 학습해 va_m 행을 예측한다. 사전확률은 학습 시즌 추세."""
        pr = train_season_trend(y[tr_m], season[tr_m], args.fold)
        bl = float(np.log(pr / (1 - pr)))
        w = np.asarray(recency_weights(frame.loc[tr_m, "season"], args.fold,
                                       half_life), np.float64)
        p_tr = Pool(fr.loc[tr_m, feats], y[tr_m].astype("int8"),
                    cat_features=cat_cols, weight=w,
                    baseline=np.full(int(tr_m.sum()), bl))
        p_va = Pool(fr.loc[va_m, feats], cat_features=cat_cols,
                    baseline=np.full(int(va_m.sum()), bl))
        m = CatBoostClassifier(**cat_params(P0, float(y[tr_m].mean())))
        m.fit(p_tr)
        out = m.predict_proba(p_va)[:, 1]
        del p_tr, p_va, m
        gc.collect()
        return out

    cache = {}

    def cond_parts():
        """조건부 두 모델을 검증 전 행에 대해 예측한다."""
        if "cond" not in cache:
            t0 = time.time()
            pb = fit(ymid, tr & b1, va)
            pz = fit(ymid, tr & b0, va)
            cache["cond"] = (pb, pz)
            print(f"    조건부 두 모델 학습 {time.time() - t0:.0f}s  "
                  f"p(m|b) 평균 {pb.mean():.4f}  p(m|~b) 평균 {pz.mean():.4f}",
                  flush=True)
        return cache["cond"]

    def p_ball_saved():
        """ball way 에서 이미 만든 예측을 쓴다. 같은 fold 의 검증 행 순서와 맞는다."""
        if "pbs" not in cache:
            f = OUT / f"{args.ball_pred}__{args.fold}.npy"
            if not f.exists():
                raise FileNotFoundError(f"ball 예측이 없다: {f}")
            v = np.load(f).astype(np.float64)
            if len(v) != int(va.sum()):
                raise ValueError(f"ball 예측 길이 불일치 {len(v)} vs {int(va.sum())}")
            cache["pbs"] = v
            print(f"    ball 예측 {args.ball_pred}  평균 {v.mean():.4f} "
                  f"(실제 {yball[va].mean():.4f})  BSS {bss(yball[va].astype(int), v)['bss_raw']:.1f}",
                  flush=True)
        return cache["pbs"]

    def p_ball_model():
        if "pb" not in cache:
            t0 = time.time()
            cache["pb"] = fit(yball, tr, va)
            print(f"    ball 새 적합 {time.time() - t0:.0f}s  평균 "
                  f"{cache['pb'].mean():.4f} (실제 {yball[va].mean():.4f})", flush=True)
        return cache["pb"]

    rows = []
    print(f"{chr(10)}  {'arm':<16}{'bss_raw':>11}{'centered':>11}{'오프셋':>10}  출처")
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        npy = OUT / f"mh_middle__{arm}{args.tag}__{args.fold}.npy"
        if npy.exists() and arm != "hier_true_b":
            m, src = bss(y_va, np.load(npy)), "cache"
        else:
            t0 = time.time()
            pmb, pmz = cond_parts()
            if arm == "hier":
                pb = p_ball_saved()
            elif arm == "hier_fitb":
                pb = p_ball_model()
            elif arm == "hier_prior_b":
                pb = np.full(int(va.sum()),
                             train_season_trend(yball[tr], season[tr], args.fold))
            elif arm == "hier_true_b":
                pb = yball[va].astype(np.float64)     # 진단 전용. 저장하지 않는다.
            else:
                print(f"  {arm:<16}  모르는 arm")
                continue
            pred = np.clip(pb * pmb + (1 - pb) * pmz, EPS, 1 - EPS)
            if arm != "hier_true_b":
                save_prediction(npy, pred, y_va, where=f"middle/{arm}")
            m, src = bss(y_va, pred), f"fit {time.time() - t0:.0f}s"
        note = "  (진단전용/제출불가)" if arm == "hier_true_b" else ""
        print(f"  {arm:<16}{m['bss_raw']:>11.1f}{m['bss_centered']:>11.1f}"
              f"{m['offset']:>+10.4f}  {src}{note}", flush=True)
        rows.append({"target": "middle", "arm": arm + args.tag, "combo": args.combo,
                     "fold": args.fold, **m})

    if rows:
        t = pd.DataFrame(rows)
        p = OUT / f"middle_hier_{args.fold}.csv"
        if p.exists():
            t = pd.concat([pd.read_csv(p), t], ignore_index=True)
        t.drop_duplicates(["target", "arm", "fold"], keep="last").to_csv(p, index=False)
        print(f"{chr(10)}saved -> {p}")
        print("  비교: 가법분할 split_ball+rate_multiscale  f24 854.4 / f23 466.9")


if __name__ == "__main__":
    main()
