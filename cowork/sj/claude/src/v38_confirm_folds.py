"""V38: 살아남은 구성을 세 fold 에서 확인한다.

선별 단계에서 이긴 것들
    V35 P2_msplit   m -> mb, mz (6성분)   단독 +9.83, ΔBSS +1.50
    V37 (실행 중)   성분별 플래툰 계층 차감

    둘 다 2024 선별에서 나온 값이라 선택 편향이 있다. V23 에서 같은 함정을
    겪었으므로 세 fold 확인이 필수다.

arm
    R0  현행 submit_030 구성 (5성분 + 플래툰 4축)
    R1  R0 에 m 분할 (6성분)
    R2  R1 에 V37 승자 (있으면)

가중치
    전역 w 격자와 V30 W1(구간별, 4000+ 만 0.45)을 함께 낸다.
    Public 이 가중치 전이율을 4% 로 못박았으므로 가중치로 이득을 주장하지
    않는다. 구성 비교가 목적이고 가중치는 고정 조건이다.

사용법
    python v38_confirm_folds.py [--with-cp]
        --with-cp 를 주면 V37 의 Q2 형태(성분별 플래툰 2겹 차감)를 R2 로 넣는다.

출력: outputs/v38_confirm_folds.csv
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
N_ROUNDS, F_WEIGHT, K = 400, 0.20, 300
WS = [0.15, 0.20, 0.25, 0.30, 0.35]
BW = np.array([0.25, 0.25, 0.25, 0.25, 0.45])
CUTS = [100, 500, 2000, 4000]
FOLDS = [2022, 2023, 2024]
WITH_CP = "--with-cp" in sys.argv
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
row_w = np.where(df["game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])
idx = pd.MultiIndex.from_arrays([pid, bhand])

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)


def AND(*a):
    m = np.ones(len(df), bool)
    for x in a:
        m &= (x == 1)
    return np.where(ok, m.astype(float), np.nan)


LAB = {"m": ym, "r": yr, "mr": AND(ym, yr), "ob": AND(yo, yb),
       "oz": AND(yo, 1 - yb), "mb": AND(ym, yb), "mz": AND(ym, 1 - yb)}
SET5 = ["m", "r", "mr", "ob", "oz"]
SET6 = ["mb", "mz", "r", "mr", "ob", "oz"]

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


def eb_ph(vals, mask):
    m = mask & ~np.isnan(vals)
    d = pd.DataFrame({"p": pid[m], "h": bhand[m], "y": vals[m]})
    lg = float(d["y"].mean())
    g = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    eb = (g["sum"] + K * lg) / (g["size"] + K)
    v = eb.reindex(idx).to_numpy()
    sz = g["size"].reindex(idx).fillna(0.0).to_numpy()
    return np.where(np.isnan(v), lg, v), lg, sz / (sz + K)


def layered_axis(axis, tr):
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": axis[tr], "y": y_all[tr]})
    lg = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * lg) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * lg) / (g3["size"] + K)
    i3 = pd.MultiIndex.from_arrays([pid, bhand, axis])
    v2 = eb2.reindex(idx).to_numpy(); v3 = eb3.reindex(i3).to_numpy()
    v2 = np.where(np.isnan(v2), lg, v2); v3 = np.where(np.isnan(v3), lg, v3)
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


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


def line(vs, tags, with_cp):
    tr, va = season < vs, season == vs
    td = df.loc[tr]
    F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                 CF.make_batter_platoon_table(td, {k: LAB[k][tr] for k in SET5}))
    for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
        sp, rel = layered_axis(ax, tr)
        F[f"{tag}_split"], F[f"{tag}_rel"] = sp, rel
        F[f"{tag}_w"] = sp * rel
    if with_cp:
        se, sl, _ = eb_ph(y_all, tr)
        for k in SET5:
            eb, lg, rel = eb_ph(LAB[k], tr)
            F[f"cp_{k}"] = (eb - lg) + (se - sl) * (lg / max(sl, 1e-6))
            F[f"cp_{k}_rel"] = rel
    X = F.to_numpy(np.float32)
    p = {}
    for tag in tags:
        arr = LAB[tag]
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr),
               **params_for(float(np.nanmean(arr[tr])))}
        d_tr = xgb.DMatrix(X[m], label=arr[m], weight=row_w[m])
        d_va = xgb.DMatrix(X[va])
        p_tr, p_va = Pool(X[m], arr[m], weight=row_w[m]), Pool(X[va])
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
    fail = sum(-p[t] if t == "mr" else p[t] for t in tags)
    return np.clip(1 - fail, EPS, 1 - EPS), F.shape[1]


ARMS = [("R0_current", SET5, False), ("R1_msplit", SET6, False)]
if WITH_CP:
    ARMS.append(("R2_msplit_cp", SET6, True))

t0, rows = time.time(), []
for vs in FOLDS:
    va = season == vs
    y, b, bk = y_all[va], strong[vs], bucket_all[va]
    wv = BW[bk]
    base_bss = metrics(y, b)["bss_raw"]
    null = y.mean() * (1 - y.mean())
    print(f"\nfold {vs}   base {base_bss:9.2f}")
    print(f"  {'arm':<14}{'피처':>6}{'단독':>10}" +
          "".join(f"{f'w{w:.2f}':>10}" for w in WS) + f"{'W1구간':>10}")
    for name, tags, cp in ARMS:
        p_ie, nf = line(vs, tags, cp)
        np.save(CACHE / f"v38_{name}_{vs}.npy", p_ie)
        solo = metrics(y, p_ie)["bss_raw"]
        out = f"  {name:<14}{nf:>6}{solo:>10.2f}"
        rec = {"fold": vs, "arm": name, "n_features": nf, "solo_bss": solo,
               "base_bss": base_bss}
        for w in WS:
            q = np.clip(w * p_ie + (1 - w) * b, EPS, 1 - EPS)
            d = metrics(y, q)["bss_raw"] - base_bss
            rec[f"d_w{w:.2f}"] = d
            out += f"{d:>+10.2f}"
        q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
        d = metrics(y, q)["bss_raw"] - base_bss
        dr = (b - y) ** 2 - (q - y) ** 2
        rec["d_W1"] = d
        rec["t_W1"] = d / (100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null)
        rows.append(rec)
        print(out + f"{d:>+10.2f}   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v38_confirm_folds.csv", index=False)
print("\n" + "=" * 78)
print("R0 대비 (W1 구간별 가중치 기준)")
print("=" * 78)
piv = res.pivot_table(index="arm", columns="fold", values="d_W1")
delta = piv.subtract(piv.loc["R0_current"], axis=1)
print(delta.round(2).to_string())
sol = res.pivot_table(index="arm", columns="fold", values="solo_bss")
print("\n성분단독 R0 대비")
print(sol.subtract(sol.loc["R0_current"], axis=1).round(2).to_string())
print("\n판정: ΔBSS 와 성분단독이 세 fold 모두 R0 이상이어야 채택한다.")
print(f"\nsaved -> {OUT/'v38_confirm_folds.csv'}")
