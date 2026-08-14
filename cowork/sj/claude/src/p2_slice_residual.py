"""P2 (Phase B): 아웃카운트 / 경기중요도(LI) 등 상황 축의 잔차 진단.

원본 47컬럼은 이미 전부 모델에 들어가 있다. 따라서 "아웃카운트가 성공률과
관계있다"는 마진 통계는 무가치하고, "모델이 그 구간에서 체계적으로 틀리는가"만
가치가 있다. 이 스크립트는 후자만 잰다.

잔차검정비 (찬우 라인과 같은 정의):
    z_c = mean_c(y - p) / sqrt(mean_c(p(1-p)) / n_c)
    ratio = mean(z_c^2)        # 잘 보정돼 있으면 ~1.0
1.03 = 무신호, 0.60~0.64 = 무신호(과분산 아님), >= 1.5 = 신호 후보.

이득 추정 두 가지를 함께 낸다.
    oracle : 각 셀 오프셋을 그 셀 자신의 라벨로 맞췄을 때 (in-sample 상한)
    honest : 홀수월에서 오프셋 적합 -> 짝수월에 적용, 반대도. 평균 (out-of-sample)
honest가 0 이하이면 그 축은 셀 추정 분산이 신호보다 크다는 뜻이다.

라운드 수는 전 fold 고정 300이다. early stopping을 쓰지 않는다.
  - 검증 fold에서 early stop -> 검증 라벨이 라운드 수에 샌다.
  - 마지막 학습 시즌(inner fold)에서 early stop -> 라운드 수가 fold 간에 전이되지
    않는다. 실측: Val2024의 inner인 2023이 레짐 단절로 10라운드에서 멈춰 2024
    base가 BSS 683 -> 195로 붕괴했다.
고정 300은 drift 실험의 2022/2024 최적(250~350) 범위이고 누수가 원천적으로 없다.

출력: outputs/p2_slice_residual.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, forecast_base_rate, metrics)

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
VALID_SEASONS = [2022, 2023, 2024]
N_ROUNDS = 300          # 전 fold 고정. 사전 등록 상수.
EPS = 1e-9


# ---------------------------------------------------------------- base model
def build_base_predictions(df):
    """fold별 8시드 배깅 예측. 고정 라운드, drift 보정 base_score."""
    y_all = df[TARGET].to_numpy(np.float64)
    season = df["season"].to_numpy()
    preds = {}
    for vs in VALID_SEASONS:
        tr_mask = season < vs
        va_mask = season == vs
        priors = make_priors(df.loc[tr_mask])
        feat = encode(add_stateless(df, priors))
        cols = [c for c in feat.columns if c not in DROP]
        X = feat[cols].to_numpy(np.float32)
        bs = forecast_base_rate(df, tr_mask, vs)
        d_tr = xgb.DMatrix(X[tr_mask], label=y_all[tr_mask])
        d_va = xgb.DMatrix(X[va_mask], label=y_all[va_mask])

        acc = np.zeros(int(va_mask.sum()))
        for seed in SEEDS:
            bst = xgb.train({**BASE_PARAMS, "base_score": bs, "seed": seed},
                            d_tr, num_boost_round=N_ROUNDS, verbose_eval=False)
            acc += bst.predict(d_va)
        p = acc / len(SEEDS)
        preds[vs] = p
        m = metrics(y_all[va_mask], p,
                    game_type=df.loc[va_mask, "game_type"].astype(str).to_numpy())
        print(f"[base] valid {vs}  rounds {N_ROUNDS}  base_score {bs:.5f}  "
              f"BSS {m['bss_raw']:9.3f}  R {m.get('r_bss', np.nan):9.3f}  "
              f"F {m.get('f_bss', np.nan):10.3f}  pred_mean {m['pred_mean']:.5f}",
              flush=True)
    return preds


# ------------------------------------------------------------------- slices
def make_slices(d):
    li = d["li"].to_numpy()
    li_q = pd.qcut(li, 10, labels=False, duplicates="drop")
    li_b = np.digitize(li, [0.5, 1.0, 2.0, 3.0])
    sd = d["score_diff_pitcher_team"].to_numpy()
    sd_b = np.digitize(sd, [-4, -1, 0, 1, 4])
    inn_b = np.digitize(d["inning"].to_numpy(), [4, 7, 10])
    outs = d["outs_before"].to_numpy()
    cnt = (d["balls_before"].to_numpy() * 3 + d["strikes_before"].to_numpy())
    run = d["num_runners_on"].to_numpy()
    mo = d["game_month"].to_numpy()
    hand = (d["pitcher_hand"].to_numpy() * 2 + d["batter_hand"].to_numpy())
    return {
        "outs": outs,
        "li_decile": li_q,
        "li_bucket": li_b,
        "high_leverage": (li >= 2).astype(int),
        "outs_x_base_state": outs * 8 + d["base_state"].cat.codes.to_numpy(),
        "outs_x_runners": outs * 4 + run,
        "outs_x_count": outs * 12 + cnt,
        "li_bucket_x_count": li_b * 12 + cnt,
        "inning_x_outs": inn_b * 3 + outs,
        "scorediff_x_li": sd_b * 5 + li_b,
        "month_x_outs": mo * 3 + outs,
        "outs_x_hand": outs * 4 + hand,
        # 참조축: 이미 residual expert가 붙어 있는 축 (기준점)
        "count_x_hand": cnt * 4 + hand,
    }


def _honest_gain(d, cell_col, min_cell):
    """홀수월 적합 -> 짝수월 적용, 반대도. 셀별 오프셋의 out-of-sample ΔBrier."""
    gains = []
    odd = d["m"] % 2 == 1
    for fit_mask in (odd, ~odd):
        apply_mask = ~fit_mask
        fit = d[fit_mask].groupby(cell_col).agg(r=("y", "mean"), pm=("p", "mean"),
                                                n=("y", "size"))
        fit = fit[fit["n"] >= max(min_cell // 2, 1)]
        off = (fit["r"] - fit["pm"]).to_dict()
        ap = d[apply_mask]
        adj = ap[cell_col].map(off).fillna(0.0).to_numpy()
        pv, yv = ap["p"].to_numpy(), ap["y"].to_numpy()
        p_new = np.clip(pv + adj, 1e-6, 1 - 1e-6)
        gains.append((np.mean((pv - yv) ** 2) - np.mean((p_new - yv) ** 2))
                     * apply_mask.sum())
    return float(sum(gains) / len(d))


def residual_report(y, p, cells, month, min_cell=200):
    """잔차검정비 + oracle/honest 이득.

    반드시 전역 오프셋을 통제한다. 셀별 오프셋을 적합하면 어떤 분할이든
    전역 편향을 흡수해 버려서, 예측 평균이 1%p만 어긋나 있어도 모든 축이
    똑같이 ~49 BSS의 가짜 '신호'를 낸다. 실측으로 확인된 함정이다.
    따라서 z와 이득 모두 전역 잔차를 뺀 뒤 계산하고, 단일 셀(전역) 통제군의
    이득을 따로 보고해 net을 낸다.
    """
    d = pd.DataFrame({"y": y, "p": p, "c": cells, "m": month, "g": 0})
    d["v"] = p * (1 - p)
    g_resid = float(d["y"].mean() - d["p"].mean())      # 전역 잔차

    g = d.groupby("c").agg(n=("y", "size"), r=("y", "mean"),
                           pm=("p", "mean"), vm=("v", "mean"))
    g = g[g["n"] >= min_cell]
    if len(g) < 2:
        return None
    g["resid"] = g["r"] - g["pm"] - g_resid             # 전역 성분 제거
    g["se"] = np.sqrt(g["vm"] / g["n"])
    g["z"] = g["resid"] / g["se"]
    ratio = float(np.mean(g["z"] ** 2))

    n_tot = len(y)
    oracle = float((g["n"] * g["resid"] ** 2).sum() / n_tot)

    honest_slice = _honest_gain(d, "c", min_cell)
    honest_global = _honest_gain(d, "g", min_cell)      # 통제군: 전역 오프셋만
    honest_net = honest_slice - honest_global

    null = y.mean() * (1 - y.mean())
    top = g.reindex(g["z"].abs().sort_values(ascending=False).index).head(3)
    return {
        "n_cells": int(len(g)), "ratio": ratio,
        "max_abs_z": float(g["z"].abs().max()),
        "global_resid": g_resid,
        "oracle_dbss": 100000 * oracle / null,
        "honest_slice_dbss": 100000 * honest_slice / null,
        "honest_global_dbss": 100000 * honest_global / null,
        "honest_dbss": 100000 * honest_net / null,
        "top_cells": "; ".join(f"c={int(i)} n={int(r.n)} resid={r.resid:+.4f} z={r.z:+.1f}"
                               for i, r in top.iterrows()),
    }


# ---------------------------------------------------------------------- run
df = load()
preds = build_base_predictions(df)

rows = []
for vs in VALID_SEASONS:
    d = df[df["season"] == vs].reset_index(drop=True)
    y = d[TARGET].to_numpy(np.float64)
    p = np.clip(preds[vs], 1e-6, 1 - 1e-6)
    month = d["game_month"].to_numpy()
    slices = make_slices(d)
    bm = metrics(y, p)
    print(f"\n{'='*116}\nvalid {vs}   n={len(d):,}   base BSS {bm['bss_raw']:.3f}   "
          f"pred_mean {bm['pred_mean']:.5f} vs target {bm['target_mean']:.5f}  "
          f"(전역 편향 {100*(bm['pred_mean']-bm['target_mean']):+.3f}%p)\n{'='*116}")
    print(f"{'slice':<20}{'cells':>6}{'ratio':>8}{'max|z|':>8}"
          f"{'oracle':>9}{'honest_all':>11}{'전역통제':>9}{'net ΔBSS':>10}   판정")
    for name, cells in slices.items():
        rep = residual_report(y, p, np.asarray(cells), month)
        if rep is None:
            continue
        verdict = ("신호 후보" if rep["ratio"] >= 1.5 and rep["honest_dbss"] > 0
                   else "무신호")
        rows.append({"valid_season": vs, "slice": name, **rep, "verdict": verdict})
        print(f"{name:<20}{rep['n_cells']:>6}{rep['ratio']:>8.2f}"
              f"{rep['max_abs_z']:>8.1f}{rep['oracle_dbss']:>9.1f}"
              f"{rep['honest_slice_dbss']:>11.1f}{rep['honest_global_dbss']:>9.1f}"
              f"{rep['honest_dbss']:>10.1f}   {verdict}")

res = pd.DataFrame(rows)
res.to_csv(OUT / "p2_slice_residual.csv", index=False)

print("\n" + "=" * 104)
print("두 연도 이상에서 ratio >= 1.5 이고 honest ΔBSS > 0 인 축")
print("=" * 104)
ok = res[(res["ratio"] >= 1.5) & (res["honest_dbss"] > 0)]
hit = ok.groupby("slice").agg(n_season=("valid_season", "nunique"),
                              seasons=("valid_season", lambda s: list(s)),
                              honest_mean=("honest_dbss", "mean"),
                              ratio_mean=("ratio", "mean"))
survivors = hit[hit["n_season"] >= 2]
print(survivors.round(2).to_string() if len(survivors)
      else "  없음 — 게이트 B 미통과. 상황 축 종료하고 Phase D로 이동한다.")

print(f"\nsaved -> {OUT/'p2_slice_residual.csv'}")
