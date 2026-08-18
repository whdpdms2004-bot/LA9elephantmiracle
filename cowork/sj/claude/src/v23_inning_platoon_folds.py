"""V23: 이닝별 플래툰(V22 J_P_hi)을 세 fold 에서 검증.

V22 (Val2024, w=0.25, 프로덕션 836.503 대비)
    J0_current  +36.07  단독 760.08   <- submit_028
    J_P_hi      +39.49  단독 761.96   <- +3.42, 5개 후보 중 최고
    J_all       +39.20  단독 721.31   <- 전부 넣으면 오히려 하락

    P_hi = EB(투수, 타자손, 이닝군) - EB(투수, 타자손)   2단계 차감

주의
    +3.42 는 5개 후보 중 2024 에서 최고를 고른 값이라 선택 편향이 섞여 있다.
    V19 의 +8.44 보다 훨씬 작다. 세 fold 에서 일관되지 않으면 싣지 않는다.

판정: 세 fold 모두 J0 대비 양수인가. 그리고 w=0.25 가 여전히 세 fold 양수인가.
출력: outputs/v23_inning_platoon_folds.csv
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

balls = df["balls_before"].to_numpy()
strikes = df["strikes_before"].to_numpy()
cnt_bucket = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()


def inning_platoon(tr_mask, K=300):
    """EB(투수, 타자손, 이닝군) - EB(투수, 타자손). 2단계 차감."""
    inn = np.digitize(df["inning"].to_numpy(), [4, 7, 10])
    d = pd.DataFrame({"p": pid[tr_mask], "h": bhand[tr_mask],
                      "i": inn[tr_mask], "y": y_all[tr_mask]})
    lg = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "i"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * lg) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * lg) / (g3["size"] + K)
    k3 = pd.MultiIndex.from_arrays([pid, bhand, inn])
    k2 = pd.MultiIndex.from_arrays([pid, bhand])
    v3 = np.where(np.isnan(eb3.reindex(k3).to_numpy()), lg, eb3.reindex(k3).to_numpy())
    v2 = np.where(np.isnan(eb2.reindex(k2).to_numpy()), lg, eb2.reindex(k2).to_numpy())
    sz = g3["size"].reindex(k3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


def count_platoon(tr_mask, K=300):
    """2단계 차감: EB(투수 x 타자손 x 카운트군) - EB(투수 x 타자손)."""
    d = pd.DataFrame({"p": pid[tr_mask], "h": bhand[tr_mask],
                      "c": cnt_bucket[tr_mask], "y": y_all[tr_mask]})
    lg = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "c"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * lg) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * lg) / (g3["size"] + K)
    k3 = pd.MultiIndex.from_arrays([pid, bhand, cnt_bucket])
    k2 = pd.MultiIndex.from_arrays([pid, bhand])
    v3 = np.where(np.isnan(eb3.reindex(k3).to_numpy()), lg, eb3.reindex(k3).to_numpy())
    v2 = np.where(np.isnan(eb2.reindex(k2).to_numpy()), lg, eb2.reindex(k2).to_numpy())
    sz = g3["size"].reindex(k3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


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
    _f = CF.build(df[INPUT_COLS], spec, platoon, bat)
    _cp, _cr = count_platoon(tr)
    _f["count_platoon_split"] = _cp
    _f["count_platoon_rel"] = _cr
    _f["count_platoon_w"] = _cp * _cr
    _ip, _ir = inning_platoon(tr)
    _f["inning_platoon_split"] = _ip
    _f["inning_platoon_rel"] = _ir
    _f["inning_platoon_w"] = _ip * _ir
    X = _f.to_numpy(np.float32)
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
res.to_csv(OUT / "v23_inning_platoon_folds.csv", index=False)
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
