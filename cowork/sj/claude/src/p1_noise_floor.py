"""P1-2 (Phase A3): 노이즈 바닥 측정.

동일 구성에서 seed만 바꿔 8회 반복하고 fold별 BSS 산포를 잰다.
이 sigma가 이후 모든 채택 판단의 단위가 된다 (채택 = sigma x 1.5 이상).

기준선: 원본 47피처 + README §10.7 stateless 파생 (TrackMan 없음, residual 없음).
출력: outputs/p1_noise_floor.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, load, make_priors, add_stateless, encode,
                     folds, fit_predict, metrics, log_result)

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]

df = load()
gt_raw = df["game_type"].astype(str).to_numpy()
month = df["game_month"].to_numpy()

rows = []
for valid_season, (tr_mask, va_mask) in folds(df).items():
    priors = make_priors(df.loc[tr_mask])
    feat = encode(add_stateless(df, priors))
    cols = [c for c in feat.columns if c not in DROP]
    X = feat[cols].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float64)
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_va, y_va = X[va_mask], y[va_mask]
    print(f"\n=== valid {valid_season}  train {tr_mask.sum():,}  "
          f"val {va_mask.sum():,}  feats {len(cols)} ===", flush=True)

    for seed in SEEDS:
        p, info, _ = fit_predict(X_tr, y_tr, X_va, y_va, seed=seed)
        m = metrics(y_va, p, game_type=gt_raw[va_mask], month=month[va_mask])
        row = {"experiment": "p1_noise_floor", "variant": "stateless_base",
               "n_features": len(cols), "valid_season": valid_season,
               "seed": seed, **info,
               **{k: v for k, v in m.items() if k != "month_brier"},
               "month_brier": m["month_brier"]}
        rows.append(row)
        log_result(row, "p1_noise_floor")
        print(f"  seed {seed:>2}  iter {info['best_iter']:>4}  "
              f"BSS {m['bss_raw']:8.3f}  R {m.get('r_bss', float('nan')):8.3f}  "
              f"F {m.get('f_bss', float('nan')):8.3f}  "
              f"brier {m['brier']:.8f}  {info['elapsed_sec']}s", flush=True)

res = pd.DataFrame(rows)
print("\n" + "=" * 74)
print("노이즈 바닥 — 같은 구성, seed만 다름")
print("=" * 74)
summary = res.groupby("valid_season").agg(
    n_seed=("seed", "size"),
    bss_mean=("bss_raw", "mean"), bss_sd=("bss_raw", "std"),
    bss_min=("bss_raw", "min"), bss_max=("bss_raw", "max"),
    r_bss_sd=("r_bss", "std"), f_bss_sd=("f_bss", "std"),
    brier_sd=("brier", "std"), iter_mean=("best_iter", "mean"),
    sec_mean=("elapsed_sec", "mean"))
summary["bss_range"] = summary["bss_max"] - summary["bss_min"]
summary["adopt_threshold_1.5sd"] = (summary["bss_sd"] * 1.5).round(2)
print(summary.round(3).to_string())

# 8-seed 배깅이 노이즈를 얼마나 줄이는지 (이론값 sd/sqrt(8)과 비교용 참고치)
print("\n참고: seed 배깅 시 기대 sd = sd/sqrt(8) =")
print((summary["bss_sd"] / np.sqrt(len(SEEDS))).round(3).to_string())

summary.to_csv(OUT / "p1_noise_floor_summary.csv")
print(f"\nsaved -> {OUT/'p1_noise_floor.csv'} , {OUT/'p1_noise_floor_summary.csv'}")
