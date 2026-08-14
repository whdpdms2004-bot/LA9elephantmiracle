"""v8 Phase 1 보강 — CatBoost 시드를 5 -> 20 으로 늘려 교란 요인을 분리한다. (sj 작성)

문제
    v8 Phase 1 결과가 anchor(650.54) 대비 -14.66 이다. 앙상블 수준 시드 노이즈가
    약 2~3 이므로 노이즈로 설명되지 않는다. 그런데 v8 은 두 가지를 동시에 바꿨다.

        (a) 하이퍼파라미터 신규 튜닝
        (b) CatBoost 시드 20 -> 5, LightGBM 20 -> 10 축소 (hw 노트북 시간 제약)

    (b) 만으로도 배깅 이득이 줄어 BSS 가 내려간다. 실측: CatBoost 개별 평균 608.11 ->
    5시드 앙상블 627.68 로 배깅이 +19.57 을 만든다. 시드를 더 넣으면 더 올라간다.
    따라서 -14.66 을 "하이퍼파라미터가 나빠졌다"로 읽으면 안 된다.

방법
    train_best_model_v8.py 의 함수와 파라미터를 그대로 import 해서 CatBoost 시드만
    2031~2045 를 추가 학습하고, 시드 수를 5/10/15/20 으로 늘려가며 Phase 1 앙상블
    BSS 를 다시 잰다. LightGBM 은 기존 10개를 재사용한다 (progress 체크포인트).

    하이퍼파라미터는 하나도 건드리지 않는다. 오직 시드 개수만 바꾼다.

출력: submission_v8/extra_seeds_by_sj.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import train_best_model_v8 as V8

EXTRA_CAT_SEEDS = list(range(2031, 2046))          # 15개 추가 -> 총 20개
CURVE_POINTS = [5, 8, 10, 12, 15, 20]
OUT_JSON = V8.OUT_DIR / "extra_seeds_by_sj.json"

t0 = time.time()
raw = pd.read_csv(V8.DATA_DIR / "train.csv")
test_cols = pd.read_csv(V8.DATA_DIR / "test.csv", nrows=0).columns.tolist()
full = V8.add_trend(raw)
V8.log(f"loaded train={raw.shape}", t0)

fit = full[full.season < V8.SANITY_VAL_SEASON].copy()
val = full[full.season == V8.SANITY_VAL_SEASON].copy()
y_fit = fit[V8.TARGET].to_numpy()
y_val = val[V8.TARGET].to_numpy()

num_cols = [c for c in test_cols if c not in ([V8.ID] + V8.BASELINE_CATS)]
extra = [c for c in full.columns if c not in test_cols and c != V8.TARGET]
num_cols = num_cols + [c for c in extra if c not in num_cols]
med = fit[num_cols].median()
cat_maps = V8.build_cat_maps(fit, V8.BASELINE_CATS)
cols = V8.BASELINE_CATS + num_cols

x_fit_cb = V8.matrix_catboost(fit, cols, num_cols, med)
x_val_cb = V8.matrix_catboost(val, cols, num_cols, med)
x_fit_lg = V8.matrix_lightgbm(fit, cols, num_cols, med, cat_maps)
x_val_lg = V8.matrix_lightgbm(val, cols, num_cols, med, cat_maps)
V8.log("행렬 준비 완료", t0)

done = V8.load_progress()

# ---- 기존 5시드 재사용 + 15시드 추가 학습
cat_preds, cat_bss = {}, {}
for seed in V8.CAT_SEEDS + EXTRA_CAT_SEEDS:
    model, best_iter = V8.train_cb_member(seed, x_fit_cb, y_fit, x_val_cb, y_val)
    p = model.predict_proba(x_val_cb)[:, 1]
    cat_preds[seed] = p
    cat_bss[seed] = V8.score(y_val, p)[1]
    V8.log(f"CatBoost seed={seed} best_iter={best_iter} BSS={cat_bss[seed]:8.2f}", t0)

lgb_preds, lgb_bss = {}, {}
for seed in V8.LGB_SEEDS:
    model, best_iter = V8.train_lgb_member(seed, x_fit_lg, y_fit, x_val_lg, y_val)
    p = model.predict(x_val_lg, num_iteration=model.best_iteration)
    lgb_preds[seed] = p
    lgb_bss[seed] = V8.score(y_val, p)[1]
V8.log(f"LightGBM {len(lgb_preds)}개 준비", t0)

lgb_mean = np.mean([lgb_preds[s] for s in V8.LGB_SEEDS], axis=0)
lgb_ens_bss = V8.score(y_val, lgb_mean)[1]

all_cat = V8.CAT_SEEDS + EXTRA_CAT_SEEDS
curve = []
for k in CURVE_POINTS:
    if k > len(all_cat):
        continue
    use = all_cat[:k]
    cb_mean = np.mean([cat_preds[s] for s in use], axis=0)
    cb_bss = V8.score(y_val, cb_mean)[1]
    ens = 0.5 * cb_mean + 0.5 * lgb_mean
    ens_bss = V8.score(y_val, ens)[1]
    curve.append({"cat_seeds": k, "cat_ens_bss": cb_bss, "lgb_ens_bss": lgb_ens_bss,
                  "final_bss": ens_bss,
                  "delta_vs_anchor": ens_bss - V8.KNOWN_ANCHOR_BSS_2024,
                  "pred_mean": float(ens.mean())})

arr = np.array([cat_bss[s] for s in all_cat])
summary = {
    "note": "하이퍼파라미터 불변, CatBoost 시드 개수만 5->20. LightGBM 10개 고정.",
    "anchor_bss_2024": V8.KNOWN_ANCHOR_BSS_2024,
    "cat_individual": {"n": len(arr), "mean": float(arr.mean()),
                       "sd": float(arr.std(ddof=1)), "min": float(arr.min()),
                       "max": float(arr.max())},
    "lgb_individual": {"n": len(lgb_bss), "mean": float(np.mean(list(lgb_bss.values()))),
                       "sd": float(np.std(list(lgb_bss.values()), ddof=1))},
    "lgb_ens_bss": lgb_ens_bss,
    "curve": curve,
    "val_target_mean": float(y_val.mean()),
    "elapsed_sec": round(time.time() - t0, 1),
}
OUT_JSON.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 84)
print("CatBoost 시드 개수 -> Phase 1 앙상블 BSS (하이퍼파라미터 불변)")
print("=" * 84)
print(f"{'cat시드':>8}{'Cat앙상블':>12}{'LGB앙상블':>12}{'최종50:50':>12}"
      f"{'anchor대비':>12}{'예측평균':>11}")
for c in curve:
    print(f"{c['cat_seeds']:>8}{c['cat_ens_bss']:>12.2f}{c['lgb_ens_bss']:>12.2f}"
          f"{c['final_bss']:>12.2f}{c['delta_vs_anchor']:>+12.2f}{c['pred_mean']:>11.5f}")
print(f"\nCatBoost 개별 20개: 평균 {arr.mean():.2f}  sd {arr.std(ddof=1):.2f}  "
      f"범위 {arr.min():.2f}~{arr.max():.2f}")
print(f"검증 실제 평균 {y_val.mean():.6f}")
print(f"\nsaved -> {OUT_JSON}")
