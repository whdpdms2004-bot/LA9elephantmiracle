"""P6 (E1 + E2): 조건부 독립 검증 + 성분 3모델 사후 결합.

E1  결합식 P(success) = (1-p_m)(1-p_r) - p_o 가 행 단위로 성립하려면
    middle 과 reverse 가 x 조건부로도 독립이어야 한다. 두 방식으로 검증한다.
      E1a 슬라이스별 phi 계수와 lift = P(m&r)/(P(m)P(r))
      E1b 모델 잔차 상관 — p_m(x), p_r(x) 학습 후 corr(y_m - p_m, y_r - p_r).
          x가 가진 정보를 다 쓴 뒤에도 남는 의존성이므로 이쪽이 결정적이다.

E2  성분 3모델을 학습해 결합식으로 control_success 를 예측하고 단일 모델과 비교.
      A direct        단일 모델 (기준선)
      B formula       (1-p_m)(1-p_r) - p_o
      C incl_excl     1 - [p_m + p_r - p_mr + p_o],  p_mr 은 네번째 모델
      D blend         w*B + (1-w)*A
      E blend_ie      w*C + (1-w)*A

라벨은 cache/failure_labels.parquet (P5에서 검증 완료). 학습에만 사용한다.
추론 경로에는 성분 라벨이 들어가지 않는다 — p_m/p_r/p_o 는 투구 이전 피처만으로 예측된다.

출력: outputs/p6_component_compose.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics, log_result)

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 300
VALID_SEASONS = [2022, 2023, 2024]
EPS = 1e-6
BLEND_W = [0.15, 0.30, 0.50, 0.70, 1.00]

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
assert lab["row_id"].is_unique and df["row_id"].is_unique
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
assert len(df) == 1_475_092
print(f"label_ok {int(df['label_ok'].sum()):,} / {len(df):,}", flush=True)

season = df["season"].to_numpy()
y_succ = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1


def as_label(col):
    """parquet은 복원 불가 행을 -1로 저장한다. 학습에서 제외하도록 NaN으로 바꾼다."""
    v = df[col].to_numpy(np.float64)
    return np.where(ok, v, np.nan)


y_m, y_r, y_o = as_label("y_middle"), as_label("y_reverse"), as_label("y_outside")
y_mr = np.where(ok, (y_m == 1) & (y_r == 1), np.nan)
for nm, v in [("y_m", y_m), ("y_r", y_r), ("y_o", y_o), ("y_mr", y_mr)]:
    u = np.unique(v[~np.isnan(v)])
    assert set(u.tolist()) <= {0.0, 1.0}, (nm, u)


def forecast_rate(labels, seasons, tr_mask, valid_season):
    """학습 시즌 라벨만으로 검증 시즌 비율 예측 (random walk + drift)."""
    m = tr_mask & ~np.isnan(labels)
    s = pd.Series(labels[m]).groupby(pd.Series(seasons[m])).mean().sort_index()
    if len(s) < 2:
        return float(s.iloc[-1])
    last_s, last_r = float(s.index[-1]), float(s.iloc[-1])
    slope = (last_r - float(s.iloc[0])) / (last_s - float(s.index[0]))
    return float(np.clip(last_r + slope * (valid_season - last_s), 0.005, 0.995))


# ============================================================ E1a 슬라이스 phi
print("\n" + "=" * 92)
print("[E1a] 슬라이스별 middle-reverse 독립성   phi = 이항 상관,  lift = P(m&r)/(P(m)P(r))")
print("=" * 92)
sub = df[ok].copy()
sm, sr = (sub["y_middle"] == 1).to_numpy(), (sub["y_reverse"] == 1).to_numpy()
sub["_m"], sub["_r"] = sm, sr
SLICES = {
    "count_state": sub["balls_before"] * 3 + sub["strikes_before"],
    "hand_matchup": sub["pitcher_hand"] * 2 + sub["batter_hand"],
    "outs": sub["outs_before"],
    "inning_bucket": np.digitize(sub["inning"], [4, 7, 10]),
    "season": sub["season"],
    "game_type": sub["game_type"].astype(str),
    "pitcher_n_bucket": np.digitize(sub["asof_pitcher_n"], [100, 1000, 4000]),
    "li_bucket": np.digitize(sub["li"], [0.5, 1.0, 2.0, 3.0]),
    "base_state": sub["base_state"].astype(str),
    "pitcher_id": sub["pitcher_id"],
}
e1a = []
print(f"{'slice':<18}{'cells':>6}{'phi_min':>9}{'phi_max':>9}{'phi_absmax':>11}"
      f"{'lift_min':>9}{'lift_max':>9}{'|phi|>0.03':>11}")
for name, key in SLICES.items():
    t = pd.DataFrame({"k": np.asarray(key), "m": sm, "r": sr})
    g = t.groupby("k").agg(n=("m", "size"), pm=("m", "mean"), pr=("r", "mean"),
                           pmr=("m", lambda s: 0.0))
    g["pmr"] = t.assign(mr=t.m & t.r).groupby("k")["mr"].mean()
    g = g[(g["n"] >= 2000) & (g["pm"] > 0) & (g["pr"] > 0)]
    if len(g) < 2:
        continue
    denom = np.sqrt(g["pm"] * (1 - g["pm"]) * g["pr"] * (1 - g["pr"]))
    g["phi"] = (g["pmr"] - g["pm"] * g["pr"]) / denom
    g["lift"] = g["pmr"] / (g["pm"] * g["pr"])
    big = int((g["phi"].abs() > 0.03).sum())
    e1a.append({"slice": name, "n_cells": len(g), "phi_min": g["phi"].min(),
                "phi_max": g["phi"].max(), "phi_absmax": g["phi"].abs().max(),
                "lift_min": g["lift"].min(), "lift_max": g["lift"].max(),
                "cells_phi_gt_003": big})
    print(f"{name:<18}{len(g):>6}{g['phi'].min():>9.4f}{g['phi'].max():>9.4f}"
          f"{g['phi'].abs().max():>11.4f}{g['lift'].min():>9.3f}"
          f"{g['lift'].max():>9.3f}{big:>11}")
pd.DataFrame(e1a).to_csv(OUT / "p6_e1a_slice_phi.csv", index=False)


# ================================================================ 모델 학습기
def bag_predict(X, tr_mask, pr_mask, labels, base_score):
    m = tr_mask & ~np.isnan(labels)
    d_tr = xgb.DMatrix(X[m], label=labels[m])
    d_pr = xgb.DMatrix(X[pr_mask])
    acc = np.zeros(int(pr_mask.sum()))
    for seed in SEEDS:
        bst = xgb.train({**BASE_PARAMS, "base_score": base_score, "seed": seed},
                        d_tr, num_boost_round=N_ROUNDS, verbose_eval=False)
        acc += bst.predict(d_pr)
    return acc / len(SEEDS)


rows = []
for vs in VALID_SEASONS:
    tr_mask, va_mask = season < vs, season == vs
    priors = make_priors(df.loc[tr_mask])
    feat = encode(add_stateless(df, priors))
    cols = [c for c in feat.columns if c not in DROP and not c.startswith("y_")
            and c != "label_ok"]
    X = feat[cols].to_numpy(np.float32)
    yv = y_succ[va_mask]
    gt = df.loc[va_mask, "game_type"].astype(str).to_numpy()

    print(f"\n{'='*92}\nvalid {vs}   train {tr_mask.sum():,}  val {va_mask.sum():,}  "
          f"feats {len(cols)}\n{'='*92}", flush=True)

    comp = {}
    for tag, lab_arr in [("succ", y_succ), ("m", y_m), ("r", y_r),
                         ("o", y_o), ("mr", y_mr)]:
        bs = forecast_rate(lab_arr, season, tr_mask, vs)
        comp[tag] = np.clip(bag_predict(X, tr_mask, va_mask, lab_arr, bs), EPS, 1 - EPS)
        act = np.nanmean(lab_arr[va_mask])
        print(f"  [{tag:>4}] base_score {bs:.5f}  pred_mean {comp[tag].mean():.5f}  "
              f"actual {act:.5f}", flush=True)

    p_a = comp["succ"]
    p_m, p_r, p_o, p_mr = comp["m"], comp["r"], comp["o"], comp["mr"]

    # ------------------------------------------------ E1b 모델 잔차 상관
    okv = ok[va_mask]
    res_m = (y_m[va_mask] - p_m)[okv]
    res_r = (y_r[va_mask] - p_r)[okv]
    rho = float(np.corrcoef(res_m, res_r)[0, 1])
    raw_rho = float(np.corrcoef(y_m[va_mask][okv], y_r[va_mask][okv])[0, 1])
    print(f"\n  [E1b] corr(y_m, y_r) raw {raw_rho:+.5f}  ->  "
          f"모델 잔차 상관 {rho:+.5f}   "
          f"{'조건부 독립 성립' if abs(rho) < 0.02 else '의존성 잔존'}", flush=True)

    # ------------------------------------------------------- E2 결합식 비교
    p_b = np.clip((1 - p_m) * (1 - p_r) - p_o, EPS, 1 - EPS)
    p_c = np.clip(1 - (p_m + p_r - p_mr + p_o), EPS, 1 - EPS)
    base_m = metrics(yv, p_a, game_type=gt)

    variants = {"A_direct": p_a, "B_formula": p_b, "C_incl_excl": p_c}
    for w in BLEND_W:
        variants[f"D_blend_{w:.2f}"] = np.clip(w * p_b + (1 - w) * p_a, EPS, 1 - EPS)
        variants[f"E_blendIE_{w:.2f}"] = np.clip(w * p_c + (1 - w) * p_a, EPS, 1 - EPS)

    print(f"\n  {'variant':<18}{'BSS':>11}{'ΔBSS':>10}{'R BSS':>11}{'F BSS':>11}"
          f"{'pred_mean':>11}", flush=True)
    for name, p in variants.items():
        mm = metrics(yv, p, game_type=gt)
        row = {"experiment": "p6_component_compose", "valid_season": vs,
               "variant": name, "bss": mm["bss_raw"],
               "dbss": mm["bss_raw"] - base_m["bss_raw"],
               "r_bss": mm.get("r_bss"), "f_bss": mm.get("f_bss"),
               "brier": mm["brier"], "pred_mean": mm["pred_mean"],
               "resid_corr_mr": rho, "raw_corr_mr": raw_rho}
        rows.append(row)
        log_result(row, "p6_component_compose")
        flag = "  <<<" if name != "A_direct" and row["dbss"] > 0 else ""
        print(f"  {name:<18}{mm['bss_raw']:>11.3f}{row['dbss']:>10.3f}"
              f"{mm.get('r_bss', np.nan):>11.3f}{mm.get('f_bss', np.nan):>11.3f}"
              f"{mm['pred_mean']:>11.5f}{flag}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "p6_component_compose_all.csv", index=False)

print("\n" + "=" * 92)
print("요약 — 채택선 1.5sigma: 2022 +7.1 / 2023 +34.4 / 2024 +9.4")
print("=" * 92)
piv = res.pivot(index="variant", columns="valid_season", values="dbss")
print(piv.round(2).to_string())
print(f"\nsaved -> {OUT/'p6_component_compose_all.csv'}")
