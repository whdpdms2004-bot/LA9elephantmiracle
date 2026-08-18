"""V40: 타자 쪽을 판다. V37 이 투수 쪽에서 실패한 이유가 타자 쪽에는 없다.

V37 이 밝힌 것
    투수 키 성분 테이블은 차감을 2겹으로 해도 안 된다(-10.62). 원본 피처에
    asof_pitcher_{success,reverse,middle,ball,strike}_rate 다섯 개가 이미 있어
    정적 테이블이 같은 정보를 '시간 제약 없이' 주기 때문이다.

    타자 쪽은 다르다. 원본에 asof_batter_n / success_rate / middle_rate 셋뿐이다.
    reverse·ball·strike 에 대응하는 타자 피처가 없다. 그래서 V12 의 타자 성분별
    플래툰이 이겼다(+2.44). 그런데 그건 1겹 차감이다.

    > V19 의 원리를 아직 적용 안 한 곳이 타자 쪽 성분 테이블 하나 남았다.

arm
    T0  현행 (V12 G4, 1겹)
    T1  타자 성분 테이블 2겹 차감
            d_k(b,ph) = [EB_k(b,ph) − lg_k] + [EB_s(b,ph) − lg_s] × (lg_k / lg_s)
        타자의 전반적 약함을 성분 규모로 환산해 빼면 "이 타자는 특히 k 방식으로
        투수를 못 흔든다"만 남는다.
    T2  T0 + 타자 x 카운트 계층 차감
            V22 J_B_cnt 는 +0.86 이었고 이닝 축에 밀려 안 실렸다. 이닝이 들어간
            지금 다시 잰다.
    T3  T1 + T2
    T4  T1 + 타자 표본 신뢰도(asof_batter_n) 상호작용

판정: Val2024 선별. 단독 BSS 와 ΔBSS 가 함께 올라야 후보다(V33/V35 기준).
출력: outputs/v40_batter_deepen.csv
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
bn = df["asof_batter_n"].to_numpy(np.float64)
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


def eb_key(vals, keys, index, mask):
    m = mask & ~np.isnan(vals)
    d = pd.DataFrame({f"k{i}": k[m] for i, k in enumerate(keys)})
    d["y"] = vals[m]
    kc = list(d.columns[:-1])
    lg = float(d["y"].mean())
    g = d.groupby(kc)["y"].agg(["sum", "size"])
    eb = (g["sum"] + K * lg) / (g["size"] + K)
    v = eb.reindex(index).to_numpy()
    sz = g["size"].reindex(index).fillna(0.0).to_numpy()
    return np.where(np.isnan(v), lg, v), lg, sz / (sz + K)


BS_EB, BS_LG, BS_REL = eb_key(y_all, [bid, phand], bidx, tr)
BCOMP = {}
for tag in COMPONENTS:
    eb, lg, rel = eb_key(LAB[tag], [bid, phand], bidx, tr)
    BCOMP[tag] = {"eb": eb, "lg": lg, "rel": rel}
    print(f"  타자 {tag:<3} 리그평균 {lg:.4f}", flush=True)


def bat_layer2(tag):
    c = BCOMP[tag]
    return (c["eb"] - c["lg"]) + (BS_EB - BS_LG) * (c["lg"] / max(BS_LG, 1e-6))


def layered_axis(keys, index, axis, base_index):
    """EB(keys, axis) − EB(keys). 2겹 차감."""
    d = pd.DataFrame({f"k{i}": k[tr] for i, k in enumerate(keys)})
    d["a"], d["y"] = axis[tr], y_all[tr]
    kc = list(d.columns[:-2])
    lg = float(d["y"].mean())
    g2 = d.groupby(kc)["y"].agg(["sum", "size"])
    g3 = d.groupby(kc + ["a"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * lg) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * lg) / (g3["size"] + K)
    v2 = eb2.reindex(base_index).to_numpy(); v3 = eb3.reindex(index).to_numpy()
    v2 = np.where(np.isnan(v2), lg, v2); v3 = np.where(np.isnan(v3), lg, v3)
    sz = g3["size"].reindex(index).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


td = df.loc[tr]
BASE_F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
pidx = pd.MultiIndex.from_arrays([pid, bhand])
for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
    sp, rel = layered_axis([pid, bhand], pd.MultiIndex.from_arrays([pid, bhand, ax]),
                           ax, pidx)
    BASE_F[f"{tag}_split"], BASE_F[f"{tag}_rel"] = sp, rel
    BASE_F[f"{tag}_w"] = sp * rel
print(f"기준 피처 {BASE_F.shape[1]}개", flush=True)

BAT_CNT = layered_axis([bid, phand],
                       pd.MultiIndex.from_arrays([bid, phand, cnt_b]), cnt_b, bidx)


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


def build(arm):
    F = BASE_F.copy()
    if arm in ("T1", "T3", "T4"):
        for k in COMPONENTS:
            F[f"bl2_{k}"] = bat_layer2(k)
    if arm in ("T2", "T3"):
        sp, rel = BAT_CNT
        F["bat_cnt_split"], F["bat_cnt_rel"] = sp, rel
        F["bat_cnt_w"] = sp * rel
    if arm == "T4":
        rel = BS_REL
        for k in COMPONENTS:
            F[f"bl2_{k}_w"] = bat_layer2(k) * rel
        F["bat_rel"] = rel
    return F


t0, rows = time.time(), []
print(f"\n{'arm':<6}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}{'경과':>8}")
for arm in ["T0", "T1", "T2", "T3", "T4"]:
    F = build(arm)
    p_ie = run(F.to_numpy(np.float32))
    np.save(CACHE / f"v40_{arm}.npy", p_ie)
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
res.to_csv(OUT / "v40_batter_deepen.csv", index=False)
r0 = res[res.arm == "T0"].iloc[0]
print("\n" + "=" * 56)
print(f"{'arm':<6}{'ΔBSS 대비':>12}{'단독 대비':>12}")
for _, r in res.iterrows():
    print(f"{r.arm:<6}{r.dbss-r0.dbss:>+12.2f}{r.solo_bss-r0.solo_bss:>+12.2f}")
print("\n둘 다 양수인 arm 만 세 fold 로 넘긴다.")
print(f"\nsaved -> {OUT/'v40_batter_deepen.csv'}")
