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
    if scheme == "fw010":
        if is_f is None:
            raise ValueError("fw010 에는 포스트시즌 표식이 필요하다")
        return np.where(is_f, base * 0.10, base)
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
                    choices=("single", "split_ball", "split_count", "split_bc",
                             "split_outs", "split_hand", "split_late"))
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--combo", default="")
    ap.add_argument("--drift-drop", type=int, default=0,
                    help="학습 시즌 간 분포 이동이 큰 수치 피처를 N개 제거한다. "
                         "평가 데이터는 보지 않는다 (학습 시즌 두 개만 비교)")
    ap.add_argument("--mask-augment", action="store_true",
                    help="마스킹으로 원본을 덮어쓰지 않고 **마스킹한 사본을 학습에 덧붙인다**. "
                         "학습 행 수가 2배가 되고 원본 정보가 보존된다. 검증은 원본만 쓴다")
    ap.add_argument("--mask", default="",
                    help="이상치 마스킹. clipQ / nanQ / rowQ / nanrareN 형태. "
                         "예: clip999(양끝 0.1%% 절단) nan995 row999 nanrare20. "
                         "임계는 **학습 행에서만** 구하고 검증에도 같은 상수를 쓴다")
    ap.add_argument("--drift-metric", default="mean",
                    choices=("mean", "quantile", "var", "combo"),
                    help="이동 지표. mean=평균차/표준편차, quantile=십분위 이동 중앙값, "
                         "var=분산비 로그, combo=셋의 순위 평균")
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
    outs = _n("outs_before")                       # 0/1/2
    hmt = frame["handedness_matchup"].astype(str).to_numpy()   # 1_1 1_2 2_1 2_2
    late = _n("late_inning")                       # 0/1

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
            sch0 = [x.strip() for x in args.schemes.split(",") if x.strip()][0]
            if args.drift_drop > 0:
                # 학습 시즌 중 가중치>0 인 최근 두 해를 비교해 이동이 큰 피처를 뺀다.
                # no_trackman 이 손으로 한 것(계측 피처 제거)의 자동화판이다.
                _w0 = weights(sch0, season[tr], fold, half_life, recency_weights,
                              (frame["game_type"].astype(str).to_numpy() == "F")[tr])
                _live = np.unique(season[tr][_w0 > 0])
                if len(_live) >= 2:
                    s_a, s_b = int(_live[-2]), int(_live[-1])
                    idx = np.flatnonzero(tr)
                    ma = season[tr] == s_a
                    mb = season[tr] == s_b
                    drift = {}
                    for c in feats:
                        if c in cat_cols:
                            continue
                        v = pd.to_numeric(fr.iloc[idx][c], errors="coerce").to_numpy(float)
                        a, b = v[ma], v[mb]
                        sd_ = np.nanstd(np.concatenate([a, b]))
                        if not np.isfinite(sd_) or sd_ < 1e-12:
                            continue
                        d_mean = abs(np.nanmean(a) - np.nanmean(b)) / sd_
                        qs = np.arange(0.1, 1.0, 0.1)
                        qa, qb = np.nanquantile(a, qs), np.nanquantile(b, qs)
                        d_qnt = float(np.nanmedian(np.abs(qa - qb))) / sd_
                        # 이름 주의: va 는 바깥의 검증 마스크다. 덮어쓰면 안 된다.
                        var_a, var_b = np.nanvar(a), np.nanvar(b)
                        d_var = abs(np.log((var_a + 1e-12) / (var_b + 1e-12)))
                        drift[c] = {"mean": d_mean, "quantile": d_qnt,
                                    "var": d_var}.get(args.drift_metric,
                                                      (d_mean, d_qnt, d_var))
                    if args.drift_metric == "combo":
                        # 세 지표의 순위를 평균낸다 (스케일이 서로 다르므로)
                        keys = list(drift)
                        rk = np.zeros(len(keys))
                        for j in range(3):
                            order = np.argsort([-drift[k][j] for k in keys])
                            r = np.empty(len(keys)); r[order] = np.arange(len(keys))
                            rk += r
                        drift = {k: -rk[i] for i, k in enumerate(keys)}
                    bad = [c for c, _ in sorted(drift.items(), key=lambda kv: -kv[1])
                           ][:args.drift_drop]
                    feats = [c for c in feats if c not in bad]
                    print(f"  분포이동 제거 {len(bad)}개 ({s_a} vs {s_b}) -> 피처 "
                          f"{len(feats)}개   상위: {bad[:4]}", flush=True)
            if args.interact:
                from train_arms import interactions
                import v85_preprocess_screen as _M
                ix = interactions(frame, feats)
                fr = _M.add_columns(fr, ix)
                feats = list(dict.fromkeys(list(feats) + list(ix)))
                print(f"  상호작용 {len(ix)}열 추가 -> 피처 {len(feats)}개", flush=True)
            # 마스킹은 피처가 확정된 뒤에 적용한다 — 예전에는 상호작용
            # 추가 전에 사본을 떠서 증강 학습이 ix_* 열 부재로 죽었다.
            row_ok = None
            fr_mask = None
            if args.mask:
                # 임계는 학습 행 분위수로만 정한다. 검증/테스트에는 같은 상수를
                # 적용하므로 행 독립이고 평가 분포를 보지 않는다.
                # '+' 로 여러 방식을 겹칠 수 있다 (예: clip995+nanrare20).
                base_fr_m = fr.copy() if args.mask_augment else fr
                for one in [x for x in args.mask.split("+") if x]:
                    m_ = re.fullmatch(r"(clip|nan|row)(\d{2,3})", one)
                    mr_ = re.fullmatch(r"nanrare(\d+)", one)
                    if mr_:
                        k = int(mr_.group(1))
                        n_masked = 0
                        for c in cat_cols:
                            vc = base_fr_m.loc[tr, c].astype(str).value_counts()
                            rare = set(vc[vc < k].index)
                            if not rare:
                                continue
                            col = base_fr_m[c].astype(str)
                            base_fr_m[c] = col.where(~col.isin(rare), "__RARE__")
                            n_masked += len(rare)
                        print(f"  희소범주 {n_masked}개 -> __RARE__ (빈도 {k} 미만)",
                              flush=True)
                    elif m_:
                        mode, qq = m_.group(1), int(m_.group(2))
                        q = 1.0 - qq / (10 ** len(str(qq)))
                        lo_q, hi_q = q, 1.0 - q
                        nums = [c for c in feats if c not in cat_cols]
                        bounds = {}
                        for c in nums:
                            v = pd.to_numeric(base_fr_m.loc[tr, c],
                                              errors="coerce").to_numpy(float)
                            if not np.isfinite(v).any():
                                continue
                            bounds[c] = (float(np.nanquantile(v, lo_q)),
                                         float(np.nanquantile(v, hi_q)))
                        if mode == "row":
                            bad = np.zeros(len(base_fr_m), bool)
                            for c, (lo, hi) in bounds.items():
                                v = pd.to_numeric(base_fr_m[c],
                                                  errors="coerce").to_numpy(float)
                                bad |= (v < lo) | (v > hi)
                            row_ok = ~bad
                            print(f"  이상치 행 {int((~row_ok[tr]).sum()):,}/"
                                  f"{int(tr.sum()):,} ({(~row_ok[tr]).mean():.1%}) "
                                  f"학습 제외", flush=True)
                        else:
                            for c, (lo, hi) in bounds.items():
                                v = pd.to_numeric(base_fr_m[c],
                                                  errors="coerce").to_numpy(float)
                                base_fr_m[c] = (np.clip(v, lo, hi) if mode == "clip"
                                                else np.where((v < lo) | (v > hi),
                                                              np.nan, v))
                            print(f"  {mode} {len(bounds)}열  "
                                  f"[{lo_q:.4f},{hi_q:.4f}]", flush=True)
                    else:
                        print(f"  ! 모르는 마스킹 {one} — 무시", flush=True)
                if args.mask_augment:
                    fr_mask = base_fr_m          # 학습에 덧붙일 사본
                    print(f"  증강: 원본 + 마스킹 사본으로 학습 행 "
                          f"{int(tr.sum()):,} -> {int(tr.sum()) * 2:,}", flush=True)
            assert_features_clean(feats, tg)
            s_tr = season[tr]
            is_f_tr = (frame["game_type"].astype(str).to_numpy() == "F")[tr]

            if args.arm == "split_ball":
                parts = [("b1", ((yv == 1) & (yball == 1)).astype(float)),
                         ("b0", ((yv == 1) & (yball == 0)).astype(float))]
            elif args.arm == "split_count":
                parts = [(f"c{k}", ((yv == 1) & (cnt == k)).astype(float))
                         for k in np.unique(cnt)]
            elif args.arm == "split_outs":        # 아웃카운트 3분할
                parts = [(f"o{k}", ((yv == 1) & (outs == k)).astype(float))
                         for k in (0.0, 1.0, 2.0)]
            elif args.arm == "split_hand":        # 투타 손 조합 4분할
                parts = [(f"h{k}", ((yv == 1) & (hmt == k)).astype(float))
                         for k in sorted(set(hmt.tolist()))]
            elif args.arm == "split_late":        # 후반 이닝 2분할
                parts = [(f"l{k}", ((yv == 1) & (late == k)).astype(float))
                         for k in (0.0, 1.0)]
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
                _cs += f"_mk{args.mask.replace('+', '-')}" if args.mask else ""
                _cs += "_aug" if (args.mask and args.mask_augment) else ""
                _cs += f"_dd{args.drift_drop}" if args.drift_drop else ""
                _cs += f"_{args.drift_metric[:3]}" if (
                    args.drift_drop and args.drift_metric != "mean") else ""
                npy = OUT / f"ww_{tg}__{args.arm}_{sch}{_cs}__{fold}.npy"
                try:
                    w = weights(sch, s_tr, fold, half_life, recency_weights,
                                is_f_tr)
                    if row_ok is not None:      # 이상치 행은 학습에서만 뺀다
                        w = np.where(row_ok[tr], w, 0.0)
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
                            if fr_mask is not None:
                                X_tr = pd.concat([fr.loc[tr, feats],
                                                  fr_mask.loc[tr, feats]],
                                                 ignore_index=True)
                                y_tr = np.concatenate([yp[tr], yp[tr]]).astype("int8")
                                w_tr = np.concatenate([w, w]) * 0.5
                            else:
                                X_tr, y_tr, w_tr = fr.loc[tr, feats],                                     yp[tr].astype("int8"), w
                            p_tr = Pool(X_tr, y_tr, cat_features=cat_cols,
                                        weight=w_tr,
                                        baseline=np.full(len(y_tr), bl))
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
