"""V17: 결합 가중치 w 를 세 fold 에서 재측정한다.

문제
    V16 에서 Val2024 최적 w 가 0.30~0.35 (+28.2) 로 나왔다. 현재 사전 등록값
    0.20 은 +23.92 라 4.3점을 남긴다.

    w=0.20 은 submit_024 때 정했다. 그때 성분 라인은 4성분/타자플래툰 없음으로
    단독 695 였고 프로덕션(836)과 격차가 141 이었다. 지금은 단독 748, 격차 88 이다.
    파트너가 강해지면 최적 가중치가 오르는 것은 메커니즘이지 과적합이 아니다.

    그러나 2024 에서 w 를 고르면 안 된다. 그건 이 프로젝트에서 세 번 지적한
    함정이다 (2022 에서 0.5~0.6 을 골라 2024 에 적용했다가 -25.9 를 맞았다).

방법
    같은 구성(V12 G4 = submit_027)으로 2022 / 2023 / 2024 세 fold 의 w 곡선을
    전부 그린다. 세 fold 에서 최적이 함께 이동했으면 메커니즘이고, 2024 만
    이동했으면 과적합이다.

    강한 base 는 fold 마다 다르다.
        2022 / 2023  enhanced_seed_oof_parts 25종 평균
        2024         프로덕션 submit_021 (실제 제출 base)
    2022/2023 은 프로덕션 예측이 없으므로 25종 앙상블로 대신한다. 절대값은
    비교하지 않고 '최적 w 의 위치'만 본다.

출력: outputs/v17_weight_folds.csv
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, OUT, CACHE, BASE_PARAMS, load, metrics

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS = 400
WS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
FOLDS = [2022, 2023, 2024]
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}

# 강한 base
models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
strong = {}
for fold in FOLDS:
    ids = df.loc[season == fold, "row_id"].to_numpy()
    acc, cnt = None, 0
    for mn in models:
        f = OOF_DIR / f"{mn}_fold{fold}.parquet"
        if f.exists():
            v = pd.read_parquet(f).set_index("row_id").reindex(ids)["prediction"].to_numpy()
            acc = v if acc is None else acc + v
            cnt += 1
    strong[fold] = np.clip(acc / cnt, EPS, 1 - EPS)
prod = pd.read_parquet(PROD).set_index("row_id").reindex(
    df.loc[season == 2024, "row_id"].to_numpy())
strong[2024] = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                       EPS, 1 - EPS)
for f in FOLDS:
    print(f"  base {f}  BSS {metrics(y_all[season==f], strong[f])['bss_raw']:9.3f}"
          f"{'  (프로덕션 submit_021)' if f == 2024 else '  (enhanced 25종)'}",
          flush=True)


def extrap(a, tr):
    m = tr & ~np.isnan(a)
    s = pd.Series(a[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


t0, rows = time.time(), []
for vs in FOLDS:
    tr, va = season < vs, season == vs
    train_df = df.loc[tr]
    spec = CF.make_spec(train_df)
    platoon = CF.make_platoon_table(train_df)
    bat = CF.make_batter_platoon_table(train_df, {k: v[tr] for k, v in LAB.items()})
    X = CF.build(df[INPUT_COLS], spec, platoon, bat).to_numpy(np.float32)
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr),
               **params_for(float(np.nanmean(arr[tr])))}
        d_tr = xgb.DMatrix(X[m], label=arr[m]); d_va = xgb.DMatrix(X[va])
        p_tr, p_va = Pool(X[m], arr[m]), Pool(X[va])
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(p_va)[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    p_ie = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
    y_va = y_all[va]
    b = strong[vs]
    bm = metrics(y_va, b)
    null = y_va.mean() * (1 - y_va.mean())
    solo = metrics(y_va, p_ie)["bss_raw"]
    gap = bm["bss_raw"] - solo
    print(f"\nfold {vs}  base {bm['bss_raw']:8.2f}  성분단독 {solo:8.2f}  "
          f"격차 {gap:7.2f}   [{time.time()-t0:.0f}s]", flush=True)
    line = "  ΔBSS  "
    for w in WS:
        q = np.clip(w * p_ie + (1 - w) * b, EPS, 1 - EPS)
        mm = metrics(y_va, q)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (b - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": vs, "w": w, "base_bss": bm["bss_raw"], "solo_bss": solo,
                     "gap": gap, "bss": mm["bss_raw"], "dbss": d, "se_row": se,
                     "t_row": d / se, "pred_mean": mm["pred_mean"],
                     "target_mean": y_va.mean()})
        line += f"{d:>+9.2f}"
    print("  w     " + "".join(f"{w:>9.2f}" for w in WS))
    print(line, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v17_weight_folds.csv", index=False)
print("\n" + "=" * 78)
print("fold 별 최적 w")
print("=" * 78)
for vs in FOLDS:
    s = res[res.fold == vs]
    b = s.sort_values("dbss", ascending=False).iloc[0]
    at20 = s[s.w == 0.20].iloc[0]
    print(f"  {vs}  격차 {b.gap:7.2f}   최적 w={b.w:.2f} ({b.dbss:+.2f})   "
          f"w=0.20 은 {at20.dbss:+.2f}   차이 {b.dbss-at20.dbss:+.2f}")
print("\n세 fold 최적이 함께 이동했으면 메커니즘, 2024만 이동했으면 과적합이다.")
print(f"\nsaved -> {OUT/'v17_weight_folds.csv'}")
