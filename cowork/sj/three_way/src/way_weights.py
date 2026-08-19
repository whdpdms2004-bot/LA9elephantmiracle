"""학습 시즌 가중치 스윕 — fold 2023 의 낮은 판별력이 2022 탓인지 가른다.

문제
    fold 2023 은 raw 뿐 아니라 centered 도 낮다.
        reverse  f23 centered 423.9  vs  f24 centered 1516.9   (3.6배)
        middle   f23 centered 540.0  vs  f24 centered  975.0
    오프셋 문제가 아니다. 그 fold 에서 신호 자체가 약하다.

가설
    fold 2023 의 가장 최근 학습 시즌은 2022 다. 그런데 2022 는 이례적인 해다 —
    1WAY 에서 2022 -> 2024 구성 순위상관이 -0.537(xgb) / -0.645(cat) 였고
    그래서 2022 는 판정 fold 에서 배제돼 있다.
    현행 recency 가중치(half_life 1.67)는 **그 2022 에 가장 큰 가중치를 준다.**
    이례적인 해를 가장 무겁게 학습하니 fold 2023 이 무너지는 것일 수 있다.

스킴
    recency     현행. half_life 1.67 로 최근 시즌 우대
    flat        전 시즌 동일 가중
    drop_2022   2022 행 가중치 0
    half_2022   2022 가중치를 절반으로
    hl_long     half_life 4.0 — 완만한 최근성
    anti        오래된 시즌을 우대 (대조군. 나빠야 정상이다)

    drop_2022 / half_2022 는 fold 2024 에서도 의미가 있다 (2022 가 학습에 들어간다).
    두 fold 모두 돌려 fold 특정 요행이 아닌지 본다.

규정
    가중치는 학습에만 쓰이고 추론 경로에 없다. 시즌 라벨은 학습 행의 메타데이터다.
    검증/평가 라벨은 보지 않는다.

사용
    python way_weights.py --target reverse --folds 2023,2024
"""
from __future__ import annotations

import argparse
import gc
import re
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guards import assert_features_clean, save_prediction, train_season_trend
from harness3 import LAB, OUT, SUCCESS, TARGETS, bss, load_labeled, seed_noise

SEED = 20262844
EPS = 1e-7
BEST_COMBO = {
    "middle": "rate_multiscale+no_trackman+no_component",
    "reverse": "drop_ids",
    "outside": "drop_ids+no_trackman+rate_multiscale",
    "success": "drop_ids+no_trackman+rate_multiscale",
}
SCHEMES = ["recency", "flat", "drop_2022", "half_2022", "hl_long", "anti"]

# 실측으로 드러난 것: fold 2023 의 지배적 지렛대는 전처리도 모델도 아닌 학습 행 가중치다.
#   drop_2022  reverse f23  -1223.7 -> -562.2  (centered +593, 판별력 회복)
#   fw020      reverse f23    157.2 ->  410.8  (포스트시즌 행 가중치 0.20)
# 둘은 서로 다른 행을 건드리므로 곱해서 같이 쓸 수 있다.


def weights(scheme: str, seasons: np.ndarray, fold: int, half_life: float,
            recency_fn, is_f: np.ndarray | None = None) -> np.ndarray:
    """스킴 이름은 '+' 로 조합할 수 있다. 예: drop_2022+fw020"""
    if "+" in scheme:
        w = np.ones(len(seasons), np.float64)
        for part in scheme.split("+"):
            w = w * weights(part, seasons, fold, half_life, recency_fn, is_f)
        # 각 조각이 recency 를 곱해 중복되므로 한 번 나눠 되돌린다
        n = scheme.count("+")
        base = np.asarray(recency_fn(pd.Series(seasons), fold, half_life), np.float64)
        return w / np.maximum(base, 1e-12) ** n
    base = np.asarray(recency_fn(pd.Series(seasons), fold, half_life), np.float64)
    if scheme == "fw020":                       # 포스트시즌 행 가중치 0.20
        if is_f is None:
            raise ValueError("fw020 에는 포스트시즌 표식이 필요하다")
        return np.where(is_f, base * 0.20, base)
    if scheme == "fw050":
        if is_f is None:
            raise ValueError("fw050 에는 포스트시즌 표식이 필요하다")
        return np.where(is_f, base * 0.50, base)
    if scheme == "recency":
        return base
    if scheme == "flat":
        return np.ones_like(base)
    m = re.fullmatch(r"drop_(\d{4})", scheme)      # 임의 시즌 제거
    if m:
        return np.where(seasons == int(m.group(1)), 0.0, base)
    m = re.fullmatch(r"half_(\d{4})", scheme)
    if m:
        return np.where(seasons == int(m.group(1)), base * 0.5, base)
    if scheme == "hl_long":
        return np.asarray(recency_fn(pd.Series(seasons), fold, 4.0), np.float64)
    if scheme == "anti":
        return np.asarray(recency_fn(pd.Series(seasons), fold, -half_life), np.float64) \
            if half_life else base
    raise ValueError(scheme)


def cat_params(base: dict, rate: float, keep_bagging: bool = False) -> dict:
    """keep_bagging 이면 P0 의 bagging_temperature(2.736)를 그대로 둔다.

    train_arms 는 P0 값을 유지하는데 여기서는 1.0 으로 덮어쓰고 있었다.
    reverse fold 2023 에서 tr:fw020 이 +410.8 인데 같은 스킴/조합으로
    여기서 돌리면 -835.3 이 나온 차이가 이것이다.
    """
    p = dict(base)
    p["bootstrap_type"] = "Bayesian"
    if not keep_bagging:
        p["bagging_temperature"] = 1.0
    p.pop("subsample", None)
    if rate < 0.06:
        # 얇은 하위 타깃은 한 단계 보수적으로. **--depth 를 무시하면 안 된다.**
        # 예전에는 depth 를 6 으로 고정해서 split_bc(8분할, 전 조각이 0.06 미만)
        # 의 depth 9/11/13 실험이 전부 depth 6 으로 돌아 같은 값이 나왔다.
        p["depth"] = max(4, int(p.get("depth", 8)) - 2)
        p["l2_leaf_reg"] = 10.0
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="reverse")
    ap.add_argument("--folds", default="2023,2024")
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--arm", default="split_ball",
                    choices=("single", "split_ball", "split_count", "split_bc"))
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--combo", default="")
    ap.add_argument("--interact", action="store_true",
                    help="2차 상호작용 피처를 추가한다 (train_arms.interactions)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="시드 배깅용. 결과는 시드별로 따로 저장된다")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--l2", type=float, default=0.0,
                    help=">0 이면 l2_leaf_reg 를 이 값으로 (P0 기본 124.9)")
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--inner-es", action="store_true",
                    help="학습 데이터 내부에서 반복 횟수를 정한다 (채점 fold 라벨 미사용)")
    ap.add_argument("--keep-bagging", action="store_true",
                    help="P0 의 bagging_temperature 를 유지 (train_arms 와 동일)")
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
    yball = pd.to_numeric(labeled["y_ball"], errors="coerce").to_numpy(np.float64)
    _n = lambda c: pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    cnt = np.digitize(_n("balls_before") * 3 + _n("strikes_before"), [3, 6, 9])

    P0 = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(P0.pop("half_life"))
    P0.update({"iterations": args.iterations, "learning_rate": args.lr,
               "depth": args.depth, "random_seed": args.seed, "task_type": "GPU",
               "devices": "0", "verbose": 0})
    if args.l2 > 0:
        P0["l2_leaf_reg"] = args.l2

    rows = []
    for tg in [t.strip() for t in args.target.split(",") if t.strip()]:
        yv = pd.to_numeric(labeled[TARGETS[tg]], errors="coerce").to_numpy(np.float64)
        ok = (labeled["label_ok"].to_numpy() == 1) & ~np.isnan(yv) & ~np.isnan(yball)
        combo = args.combo or BEST_COMBO.get(tg, "")
        ctup = tuple(x for x in combo.split("+") if x)
        sd = seed_noise(tg)

        for fold in [int(f) for f in args.folds.split(",") if f.strip()]:
            tr, va = (season < fold) & ok, (season == fold) & ok
            y_va = yv[va].astype("int8")
            t0 = time.time()
            static = build_component_unique(frame, enhanced, fold)
            forward = build_component_unique_forward(frame, enhanced, fold,
                                                     cache={fold: static})
            base_fr, f1_features = make_features(frame, enhanced, fold, "F1", forward)
            for c in (SUCCESS, "season"):
                if c not in base_fr.columns:
                    base_fr[c] = frame[c].to_numpy()
            cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
            fr, feats, cat_cols = T.build(base_fr, f1_features, cats0, ctup,
                                          pd.Series(tr, index=frame.index), fold)
            if args.interact:
                from train_arms import interactions
                import v85_preprocess_screen as _M
                ix = interactions(frame, feats)
                fr = _M.add_columns(fr, ix)
                feats = list(dict.fromkeys(list(feats) + list(ix)))
                print(f"  상호작용 {len(ix)}열 추가 -> 피처 {len(feats)}개", flush=True)
            assert_features_clean(feats, tg)
            s_tr = season[tr]
            is_f_tr = (frame["game_type"].astype(str).to_numpy() == "F")[tr]

            if args.arm == "split_ball":
                parts = [("b1", ((yv == 1) & (yball == 1)).astype(float)),
                         ("b0", ((yv == 1) & (yball == 0)).astype(float))]
            elif args.arm == "split_count":
                parts = [(f"c{k}", ((yv == 1) & (cnt == k)).astype(float))
                         for k in np.unique(cnt)]
            elif args.arm == "split_bc":          # ball x 카운트군
                parts = [(f"b{b}c{k}",
                          ((yv == 1) & (yball == b) & (cnt == k)).astype(float))
                         for b in (0, 1) for k in np.unique(cnt)]
            else:
                parts = [("all", yv)]
            if args.arm != "single":
                tot = sum(v for _, v in parts)
                assert np.abs(tot[ok] - yv[ok]).max() < 1e-9, "분할이 타깃을 안 덮는다"

            print(f"{chr(10)}{'=' * 100}")
            print(f"{tg}  fold {fold}  arm {args.arm}  전처리 {combo or 'baseline'}")
            print(f"  학습 {tr.sum():,}  검증 {va.sum():,}  프레임 {time.time() - t0:.0f}s"
                  f"  잡음 sd {sd:.2f}")
            print(f"  학습 시즌 구성: " + "  ".join(
                f"{y}:{int((s_tr == y).sum()):,}" for y in sorted(set(s_tr.tolist()))))
            print("=" * 100)
            print(f"  {'스킴':<12}{'bss_raw':>10}{'centered':>10}{'오프셋':>10}"
                  f"{'2022비중':>10}  출처", flush=True)

            for sch in [x.strip() for x in args.schemes.split(",") if x.strip()]:
                # 파일명에 조합이 없으면 --combo 를 바꿔도 옛 예측을 재사용해버린다.
                # 기본 조합일 때만 옛 이름을 유지해 기존 캐시를 살린다.
                _cs = "" if combo == BEST_COMBO.get(tg, "") else                     "_" + combo.replace("+", "-")[:36]
                # 설정을 결정하는 인자는 전부 키에 넣는다.
                # 조합과 반복 횟수를 빼먹어 두 번 옛 캐시를 재사용한 적이 있다.
                _cs += "_kb" if args.keep_bagging else ""
                _cs += "_ies" if args.inner_es else ""
                _cs += f"_it{args.iterations}" if args.iterations != 900 else ""
                _cs += f"_d{args.depth}" if args.depth != 8 else ""
                _cs += f"_l2{args.l2:g}" if args.l2 > 0 else ""
                _cs += f"_lr{args.lr:g}" if args.lr != 0.015 else ""
                _cs += f"_s{args.seed}" if args.seed != SEED else ""
                _cs += "_ix" if args.interact else ""
                npy = OUT / f"ww_{tg}__{args.arm}_{sch}{_cs}__{fold}.npy"
                try:
                    w = weights(sch, s_tr, fold, half_life, recency_weights,
                                is_f_tr)
                except Exception as exc:                          # noqa: BLE001
                    print(f"  {sch:<12}  가중치 실패: {str(exc)[:50]}"); continue
                share = float(w[s_tr == 2022].sum() / w.sum()) if (s_tr == 2022).any() else 0.0
                if npy.exists():
                    pred, src = np.load(npy), "cache"
                else:
                    t1 = time.time()
                    try:
                        pred = np.zeros(int(va.sum()))
                        for _, yp in parts:
                            pr = train_season_trend(yp[tr], season[tr], fold)
                            bl = float(np.log(pr / (1 - pr)))
                            n_it = None
                            if args.inner_es:
                                # 학습 데이터 안에서만 반복 횟수를 정한다.
                                # 내부 검증 = 학습 시즌 중 가장 최근(가중치>0)인 해.
                                # 채점 fold(va) 의 라벨은 어디에도 쓰지 않는다.
                                live = np.unique(s_tr[w > 0])
                                iv_season = int(live.max())
                                i_va = (s_tr == iv_season)
                                i_tr = (~i_va) & (w > 0)
                                if i_tr.sum() > 20000 and i_va.sum() > 5000:
                                    idx = np.flatnonzero(tr)
                                    q_tr = Pool(fr.iloc[idx[i_tr]][feats],
                                                yp[tr][i_tr].astype("int8"),
                                                cat_features=cat_cols, weight=w[i_tr],
                                                baseline=np.full(int(i_tr.sum()), bl))
                                    q_va = Pool(fr.iloc[idx[i_va]][feats],
                                                yp[tr][i_va].astype("int8"),
                                                cat_features=cat_cols,
                                                baseline=np.full(int(i_va.sum()), bl))
                                    mm = CatBoostClassifier(**cat_params(
                                        P0, float(yp[tr].mean()), args.keep_bagging))
                                    mm.fit(q_tr, eval_set=q_va, use_best_model=True,
                                           early_stopping_rounds=80)
                                    n_it = max(60, int(mm.get_best_iteration()) + 1)
                                    print(f"      내부 조기종료: 검증시즌 {iv_season} "
                                          f"-> {n_it}회", flush=True)
                                    del q_tr, q_va, mm
                                    gc.collect()
                            pp = cat_params(P0, float(yp[tr].mean()), args.keep_bagging)
                            if n_it:
                                pp["iterations"] = n_it
                            p_tr = Pool(fr.loc[tr, feats], yp[tr].astype("int8"),
                                        cat_features=cat_cols, weight=w,
                                        baseline=np.full(int(tr.sum()), bl))
                            p_va = Pool(fr.loc[va, feats], cat_features=cat_cols,
                                        baseline=np.full(int(va.sum()), bl))
                            m = CatBoostClassifier(**pp)
                            m.fit(p_tr)
                            pred = pred + m.predict_proba(p_va)[:, 1]
                            del p_tr, p_va, m
                            gc.collect()
                    except Exception as exc:                      # noqa: BLE001
                        print(f"  {sch:<12}  실패: {type(exc).__name__}: "
                              f"{str(exc)[:50]}", flush=True)
                        continue
                    pred = np.clip(pred, EPS, 1 - EPS)
                    save_prediction(npy, pred, y_va, where=f"{tg}/{sch}")
                    src = f"fit {time.time() - t1:.0f}s"
                m = bss(y_va, pred)
                print(f"  {sch:<12}{m['bss_raw']:>10.1f}{m['bss_centered']:>10.1f}"
                      f"{m['offset']:>+10.4f}{share:>10.3f}  {src}", flush=True)
                rows.append({"target": tg, "scheme": sch, "arm": args.arm,
                             "combo": combo, "fold": fold, "w2022": share,
                             "keep_bagging": bool(args.keep_bagging),
                             "inner_es": bool(args.inner_es), **m})
                pd.DataFrame(rows).to_csv(OUT / "way_weights.csv", index=False)
            del fr, base_fr, static, forward
            gc.collect()

    if rows:
        t = pd.DataFrame(rows)
        p = OUT / "way_weights.csv"
        if p.exists():
            t = pd.concat([pd.read_csv(p), t], ignore_index=True)
        for c in ("keep_bagging", "inner_es"):
            if c not in t.columns:
                t[c] = False
        t.drop_duplicates(["target", "scheme", "arm", "fold", "combo",
                           "keep_bagging", "inner_es"], keep="last").to_csv(
            p, index=False)
        print(f"{chr(10)}saved -> {p}")


if __name__ == "__main__":
    main()
