"""V54: Tier E 를 fold 2023·2024 에서 확인한다.

V53 (2024 선별)
    E0 현행            단독 745.17   ΔBSS +38.92
    E2 +TierE F행제외   단독 755.17   ΔBSS +41.56   (+10.00 / +2.63)
    단독과 ΔBSS 가 함께 올랐다. V33 이후 모든 피처 추가가 실패했던 기준을 통과한
    첫 사례다. E2 > E1 이라 규칙 N3(F행 제외)도 실측 확인됐다.

fold 2022 는 구조적으로 불가능하다
    규칙 N1 이 TrackMan 증거 시즌을 2022 이후로 제한한다. fold 2022 는
    증거가 season < 2022 여야 하므로 교집합이 공집합이다.
    09_TIER_E_RESULTS.md 도 같은 이유로 "fold 2022 확증이 구조적으로 불가능"이라
    적고 월 블록 검증으로 대체했다. 여기서는 2023·2024 두 fold 로 판정한다.

    fold 2023 은 증거가 2022 한 시즌뿐이라 커버리지가 낮다. 그 조건에서도
    양수면 신뢰할 만하다.

판정: 두 fold 모두 단독과 ΔBSS 가 E0 이상이어야 채택한다.
출력: outputs/v54_tiere_folds.csv
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

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
TE_DIR = SJ / "claude" / "outputs" / "tier_e"
CW_DIR = SJ / "experiment" / "pitcher_embedding" / "outputs" / "trackman500"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, F_WEIGHT, K = 400, 0.20, 300
WS = [0.15, 0.20, 0.25, 0.30]
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])       # submit_032
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
FOLDS = [2023, 2024]
MIN_EV_SEASON, HALF_LIFE = 2022, 2.0
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)

pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
IS_F = df["game_type"].astype(str).to_numpy() == "F"
row_w = np.where(IS_F, F_WEIGHT, 1.0)
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)


def AND(*a):
    m = np.ones(len(df), bool)
    for x in a:
        m &= (x == 1)
    return np.where(ok, m.astype(float), np.nan)


LAB = {"m": ym, "r": yr, "mr": AND(ym, yr), "ob": AND(yo, yb), "oz": AND(yo, 1 - yb)}

models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
BASE_P = {}
for f in FOLDS:
    fid = df.loc[season == f, "row_id"].to_numpy()
    if f == 2024:
        pr = pd.read_parquet(PROD).set_index("row_id").reindex(fid)
        BASE_P[f] = np.clip(pr["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                            EPS, 1 - EPS)
    else:
        acc, c = None, 0
        for mn in models:
            p = OOF_DIR / f"{mn}_fold{f}.parquet"
            if p.exists():
                v = pd.read_parquet(p).set_index("row_id").reindex(fid)["prediction"].to_numpy()
                acc = v if acc is None else acc + v
                c += 1
        BASE_P[f] = np.clip(acc / c, EPS, 1 - EPS)


def tier_e_profile(fold):
    cw = pd.read_parquet(CW_DIR / f"cutoff_{fold}" / "crosswalk.parquet")[
        ["pitcher_id", "pitcher_trackman_id"]]
    emb = pd.read_parquet(TE_DIR / f"tier_e_cutoff{fold}.parquet")
    dims = [c for c in emb.columns if c.startswith("te_svd_")]
    emb = emb[emb["season"] >= MIN_EV_SEASON]
    m = cw.merge(emb, on="pitcher_trackman_id", how="inner").rename(
        columns={"season": "ev"})
    out = []
    for S in sorted(df["season"].unique()):
        past = m[m["ev"] < S]
        if past.empty:
            continue
        w = np.power(0.5, (S - past["ev"].to_numpy()) / HALF_LIFE)
        g = past.assign(_w=w, **{c: past[c].to_numpy() * w for c in dims}) \
                .groupby("pitcher_id", as_index=False).agg(
                    _w=("_w", "sum"), **{c: (c, "sum") for c in dims})
        for c in dims:
            g[c] = g[c] / g["_w"].clip(lower=1e-9)
        g["season"] = S
        out.append(g.drop(columns="_w"))
    prof = pd.concat(out, ignore_index=True)
    TE = df[["pitcher_id", "season"]].merge(prof, on=["pitcher_id", "season"],
                                            how="left")
    return TE[dims].to_numpy(np.float64), dims


def extrap(a, mask):
    m_ = mask & ~np.isnan(a)
    s = pd.Series(a[m_]).groupby(pd.Series(season[m_])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def features(fold, with_te):
    tr = season < fold
    td = df.loc[tr]
    F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                 CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
    pidx = pd.MultiIndex.from_arrays([pid, bhand])
    for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
        d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": ax[tr], "y": y_all[tr]})
        l0 = float(d["y"].mean())
        g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
        g3 = d.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
        e2 = (g2["sum"] + K * l0) / (g2["size"] + K)
        e3 = (g3["sum"] + K * l0) / (g3["size"] + K)
        i3 = pd.MultiIndex.from_arrays([pid, bhand, ax])
        v2 = np.where(np.isnan(e2.reindex(pidx).to_numpy()), l0,
                      e2.reindex(pidx).to_numpy())
        v3 = np.where(np.isnan(e3.reindex(i3).to_numpy()), l0,
                      e3.reindex(i3).to_numpy())
        sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
        F[f"{tag}_split"], F[f"{tag}_rel"] = v3 - v2, sz / (sz + K)
        F[f"{tag}_w"] = (v3 - v2) * sz / (sz + K)
    cov = 0.0
    if with_te:
        V, dims = tier_e_profile(fold)
        V[IS_F] = np.nan                       # 규칙 N3
        for i, c in enumerate(dims):
            F[c] = V[:, i]
        va = season == fold
        cov = float(np.isfinite(V[va & ~IS_F, 0]).mean())
    return F, cov


def line(X, fold):
    tr, va = season < fold, season == fold
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr),
               **params_for(float(np.nanmean(arr[mm])))}
        Xv = X[va]
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=row_w[mm], missing=np.nan)
        d_va = xgb.DMatrix(Xv, missing=np.nan)
        p_tr = Pool(np.nan_to_num(X[mm], nan=-999.0), arr[mm], weight=row_w[mm])
        Xc = np.nan_to_num(Xv, nan=-999.0)
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(Xc)[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


t0, rows = time.time(), []
for fold in FOLDS:
    va = season == fold
    y, b, bk = y_all[va], BASE_P[fold], bucket_all[va]
    null = y.mean() * (1 - y.mean())
    base_bss = metrics(y, b)["bss_raw"]
    print(f"{chr(10)}fold {fold}   base {base_bss:9.2f}")
    print(f"  {'arm':<16}{'피처':>6}{'커버':>7}{'단독':>10}"
          + "".join(f"{f'w{w:.2f}':>10}" for w in WS) + f"{'032벡터':>10}")
    for arm, with_te in [("E0_current", False), ("E2_tierE", True)]:
        F, cov = features(fold, with_te)
        p_ie = line(F.to_numpy(np.float32), fold)
        np.save(CACHE / f"v54_{arm}_{fold}.npy", p_ie)
        solo = metrics(y, p_ie)["bss_raw"]
        rec = {"fold": fold, "arm": arm, "n_features": F.shape[1],
               "coverage": cov, "solo_bss": solo, "base_bss": base_bss}
        out = f"  {arm:<16}{F.shape[1]:>6}{cov*100:>6.1f}%{solo:>10.2f}"
        for w in WS:
            q = np.clip(w * p_ie + (1 - w) * b, EPS, 1 - EPS)
            d = metrics(y, q)["bss_raw"] - base_bss
            rec[f"d_w{w:.2f}"] = d
            out += f"{d:>+10.2f}"
        wv = BW[bk]
        q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
        d = metrics(y, q)["bss_raw"] - base_bss
        dr = (b - y) ** 2 - (q - y) ** 2
        rec["d_032"] = d
        rec["t_032"] = d / (100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null)
        rows.append(rec)
        print(out + f"{d:>+10.2f}   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v54_tiere_folds.csv", index=False)
print(f"{chr(10)}{'='*70}{chr(10)}E0 대비 (submit_032 구간 벡터 기준){chr(10)}{'='*70}")
piv = res.pivot_table(index="arm", columns="fold", values="d_032")
sol = res.pivot_table(index="arm", columns="fold", values="solo_bss")
print("ΔBSS")
print(piv.subtract(piv.loc["E0_current"], axis=1).round(2).to_string())
print(f"{chr(10)}성분단독")
print(sol.subtract(sol.loc["E0_current"], axis=1).round(2).to_string())
print(f"{chr(10)}두 fold 모두 둘 다 양수면 채택. fold 2022 는 규칙 N1 때문에 "
      f"구조적으로 불가능하다.")
print(f"{chr(10)}saved -> {OUT/'v54_tiere_folds.csv'}")
