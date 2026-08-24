"""P1-3 (Phase A3 재측정): 시즌 drift 보정 후 노이즈 바닥.

p1_noise_floor.py에서 Val2023이 BSS -397 +- 56으로 붕괴했다. 원인 분리:
  (a) 시즌 drift  - 학습 구간 pooled 평균이 검증 시즌보다 높아 base_score가 편향
  (b) F regime 단절 - F 성공률이 2022 0.7087 -> 2023 0.4729로 붕괴 (누수 없이 예측 불가)

(a)는 학습 라벨만으로 보정 가능하다. (b)는 불가능하므로 R-only 지표를 함께 본다.
이 스크립트는 세 구성을 같은 seed 8개로 비교한다.

  raw    : 보정 없음 (p1_noise_floor 재현)
  drift  : base_score = forecast_base_rate(학습 시즌만)
  driftR : drift + R 행만으로 지표 산출

출력: outputs/p1_noise_floor_drift.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, load, make_priors, add_stateless, encode,
                     folds, forecast_base_rate, fit_predict, metrics, log_result)

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]

df = load()
gt_raw = df["game_type"].astype(str).to_numpy()
month = df["game_month"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)

rows = []
for valid_season, (tr_mask, va_mask) in folds(df).items():
    priors = make_priors(df.loc[tr_mask])
    feat = encode(add_stateless(df, priors))
    cols = [c for c in feat.columns if c not in DROP]
    X = feat[cols].to_numpy(np.float32)
    X_tr, y_tr = X[tr_mask], y_all[tr_mask]
    X_va, y_va = X[va_mask], y_all[va_mask]
    gt_va, mo_va = gt_raw[va_mask], month[va_mask]
    is_r = gt_va == "R"

    pooled = float(y_tr.mean())
    fc = forecast_base_rate(df, tr_mask, valid_season)
    actual = float(y_va.mean())
    print(f"\n=== valid {valid_season}  train {tr_mask.sum():,}  val {va_mask.sum():,} "
          f"feats {len(cols)} ===", flush=True)
    print(f"    학습 pooled {pooled:.6f} | 예측 base {fc:.6f} | 실제 {actual:.6f} "
          f"| pooled 오차 {(pooled-actual)*100:+.3f}%p -> 보정 후 {(fc-actual)*100:+.3f}%p",
          flush=True)

    for variant, base_score in [("raw", None), ("drift", fc)]:
        params = {} if base_score is None else {"base_score": base_score}
        for seed in SEEDS:
            p, info, _ = fit_predict(X_tr, y_tr, X_va, y_va, params=params, seed=seed)
            m = metrics(y_va, p, game_type=gt_va, month=mo_va)
            row = {"experiment": "p1_noise_floor_drift", "variant": variant,
                   "base_score": base_score, "n_features": len(cols),
                   "valid_season": valid_season, "seed": seed, **info,
                   **{k: v for k, v in m.items() if k != "month_brier"}}
            rows.append(row)
            log_result(row, "p1_noise_floor_drift")
            print(f"  {variant:>5} seed {seed:>2}  iter {info['best_iter']:>4}  "
                  f"BSS {m['bss_raw']:9.3f}  R {m.get('r_bss', np.nan):9.3f}  "
                  f"F {m.get('f_bss', np.nan):10.3f}  pred_mean {m['pred_mean']:.5f}",
                  flush=True)

res = pd.DataFrame(rows)
print("\n" + "=" * 96)
print("노이즈 바닥 비교 — 같은 구성, seed만 다름")
print("=" * 96)
summ = res.groupby(["valid_season", "variant"]).agg(
    bss_mean=("bss_raw", "mean"), bss_sd=("bss_raw", "std"),
    bss_range=("bss_raw", lambda s: s.max() - s.min()),
    r_bss_mean=("r_bss", "mean"), r_bss_sd=("r_bss", "std"),
    f_bss_mean=("f_bss", "mean"), f_bss_sd=("f_bss", "std"),
    iter_mean=("best_iter", "mean"), sec_mean=("elapsed_sec", "mean"))
summ["adopt_1.5sd_all"] = (summ["bss_sd"] * 1.5).round(2)
summ["adopt_1.5sd_R"] = (summ["r_bss_sd"] * 1.5).round(2)
print(summ.round(3).to_string())

summ.to_csv(OUT / "p1_noise_floor_drift_summary.csv")
print(f"\nsaved -> {OUT/'p1_noise_floor_drift.csv'} , "
      f"{OUT/'p1_noise_floor_drift_summary.csv'}")
