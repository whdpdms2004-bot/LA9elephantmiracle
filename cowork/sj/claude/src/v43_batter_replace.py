"""V43: V40 의 arm 설계 결함을 고친다 — '추가'가 아니라 '교체'.

V40 의 결함
    기준 피처(BASE_F)에는 이미 V12 의 타자 성분 테이블 bat_pl_{tag} 가 들어 있다.
    그건 1겹 차감이다. V40 은 거기에 2겹 차감판 bl2_{k} 를 '추가'했다.
    같은 정보의 두 표현을 나란히 넣은 것이라 재표현이고, 지금까지 재표현은
    한 번도 이득을 낸 적이 없다. 실측도 그랬다 (T1 −5.63, 단독 −36.32).

    V19 가 원리를 세운 방식은 '추가'가 아니라 '교체'였다.
        1겹 split -> 2겹 split 으로 바꾸니 +0.16 -> +8.44
    타자 축에서도 교체로 물어야 한다.

arm  (한 번에 잰다 — run 간 시드 노이즈가 ±1.5 라 대조군이 같은 실행 안에 있어야 한다)
    U0  현행           bat_pl_{k} 1겹
    U1  교체           bat_pl_{k} 를 2겹으로 대체
    U2  교체 + rel     2겹 + 타자 표본 신뢰도

    2겹 정의
        d_k(b,ph) = [EB_k(b,ph) − lg_k] + [EB_s(b,ph) − lg_s] × (lg_k / lg_s)
        타자의 전반적 약함을 성분 규모로 환산해 빼면 "이 타자는 특히 k 방식으로
        투수를 못 흔든다"만 남는다.

출력: outputs/v43_batter_replace.csv
"""
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
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W, F_WEIGHT, K = 400, 0.25, 0.20, 300
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
VS = 2024
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
tr, va = season < VS, season == VS

pid = df["pitcher_id"].to_numpy()
bid = df["batter_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
phand = df["pitcher_hand"].to_numpy()
row_w = np.where(df["game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
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
bidx = pd.MultiIndex.from_arrays([bid, phand])
pidx = pd.MultiIndex.from_arrays([pid, bhand])


def eb(vals, keys, index, mask):
    m = mask & ~np.isnan(vals)
    d = pd.DataFrame({f"k{i}": k[m] for i, k in enumerate(keys)})
    d["y"] = vals[m]
    kc = list(d.columns[:-1])
    lg = float(d["y"].mean())
    g = d.groupby(kc)["y"].agg(["sum", "size"])
    t = (g["sum"] + K * lg) / (g["size"] + K)
    v = t.reindex(index).to_numpy()
    sz = g["size"].reindex(index).fillna(0.0).to_numpy()
    return np.where(np.isnan(v), lg, v), lg, sz / (sz + K)


BS_EB, BS_LG, BS_REL = eb(y_all, [bid, phand], bidx, tr)
BC = {}
for tag in COMPONENTS:
    e, lg, rel = eb(LAB[tag], [bid, phand], bidx, tr)
    BC[tag] = {"d2": (e - lg) + (BS_EB - BS_LG) * (lg / max(BS_LG, 1e-6))}


def layered_axis(axis):
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": axis[tr], "y": y_all[tr]})
    lg = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
    e2 = (g2["sum"] + K * lg) / (g2["size"] + K)
    e3 = (g3["sum"] + K * lg) / (g3["size"] + K)
    i3 = pd.MultiIndex.from_arrays([pid, bhand, axis])
    v2 = e2.reindex(pidx).to_numpy(); v3 = e3.reindex(i3).to_numpy()
    v2 = np.where(np.isnan(v2), lg, v2); v3 = np.where(np.isnan(v3), lg, v3)
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


td = df.loc[tr]
BASE_F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
    sp, rel = layered_axis(ax)
    BASE_F[f"{tag}_split"], BASE_F[f"{tag}_rel"] = sp, rel
    BASE_F[f"{tag}_w"] = sp * rel
BPL = [c for c in BASE_F.columns if c.startswith("bat_pl_")]
assert len(BPL) == len(COMPONENTS), f"타자 성분 열 {BPL}"
print(f"기준 피처 {BASE_F.shape[1]}개, 교체 대상 {BPL}", flush=True)


def extrap(a):
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


def run(X):
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr),
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
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b, game_type=gt)["bss_raw"]

t0, rows = time.time(), []
print(f"\n{'arm':<6}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}{'경과':>8}")
for arm in ["U0", "U1", "U2"]:
    F = BASE_F.copy()
    if arm != "U0":
        F = F.drop(columns=BPL)
        for k in COMPONENTS:
            F[f"bl2_{k}"] = BC[k]["d2"]
        if arm == "U2":
            F["bat_rel"] = BS_REL
            for k in COMPONENTS:
                F[f"bl2_{k}_w"] = BC[k]["d2"] * BS_REL
    p_ie = run(F.to_numpy(np.float32))
    np.save(CACHE / f"v43_{arm}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(b / (1 - b)), np.log(p_ie / (1 - p_ie)))[0, 1])
    q = np.clip(W * p_ie + (1 - W) * b, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": arm, "n_features": F.shape[1], "solo_bss": solo, "corr": corr,
                 "bss": mm["bss_raw"], "dbss": d, "se_row": se, "t_row": d / se})
    print(f"{arm:<6}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}"
          f"{d/se:>8.2f}{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v43_batter_replace.csv", index=False)
r0 = res[res.arm == "U0"].iloc[0]
print("\n" + "=" * 56)
print(f"{'arm':<6}{'ΔBSS 대비':>12}{'단독 대비':>12}")
for _, r in res.iterrows():
    print(f"{r.arm:<6}{r.dbss-r0.dbss:>+12.2f}{r.solo_bss-r0.solo_bss:>+12.2f}")
print("\nV19 의 교체 실험은 +0.16 -> +8.44 였다. 타자 축에서도 재현되면 채택 후보.")
print(f"\nsaved -> {OUT/'v43_batter_replace.csv'}")
