"""V22: 계층 차감 스플릿 확장 — V19 의 원리를 다른 축으로.

원리 (V19 에서 확립)
    V8  EB(투수, 타자손, 성분) - 리그평균          +0.16  실패 (전역 플래툰과 중복)
    V19 EB(투수, 타자손, 카운트) - EB(투수, 타자손) +8.44  성공 (2단계 차감)

    주효과가 하나가 아니라 계층으로 쌓여 있다. 바로 위 층까지 빼야 새 정보만 남는다.

후보 (전부 2단계 차감)
    P_cnt   EB(투수, 카운트)              - EB(투수)              좌우 무관 카운트 성향
    B_cnt   EB(타자, 카운트)              - EB(타자)
    B_hc    EB(타자, 투수손, 카운트)      - EB(타자, 투수손)      V19 의 완전 대칭
    P_hi    EB(투수, 타자손, 이닝군)      - EB(투수, 타자손)
    P_hb    EB(투수, 타자손, 주자유무)    - EB(투수, 타자손)

절차
    1단계  산포를 먼저 잰다. 2단계 차감 후 남는 sd 가 작으면 정보가 없다.
           기준: 투수 전역 플래툰 sd 0.01775, 카운트별 0.01526 (채택된 것)
    2단계  산포가 충분한 후보만 학습해 ΔBSS 를 잰다.

    구분선 기준: 산포가 크면 '아직 못 보는 정보'일 가능성이 있다. 다만 V8 처럼
    산포가 있어도 상위 층과 중복이면 이득이 없으므로 학습으로 확인해야 한다.

판정: Val2024 전체 BSS, 프로덕션 836.503 대비, w=0.25 (submit_028 구성).
출력: outputs/v22_layered_splits.csv
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

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W = 400, 0.25
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
K = 300
SD_FLOOR = 0.008          # 채택된 스플릿의 절반. 이보다 작으면 학습하지 않는다
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
tr, va = season < 2024, season == 2024
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]

pid = df["pitcher_id"].to_numpy()
bid = df["batter_id"].to_numpy()
phand = df["pitcher_hand"].to_numpy()
bhand = df["batter_hand"].to_numpy()
cnt = CF.count_bucket(df)
inn = np.digitize(df["inning"].to_numpy(), [4, 7, 10])
runners = (df["num_runners_on"].to_numpy() > 0).astype(int)

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}


def layered_split(upper_keys, lower_keys):
    """EB(lower_keys) - EB(upper_keys). lower 가 upper 를 포함해야 한다.

    2단계 차감. 상위 층을 명시적으로 빼야 새 정보만 남는다 (V19 원리).
    """
    d = {f"k{i}": k[tr] for i, k in enumerate(lower_keys)}
    d["y"] = y_all[tr]
    d = pd.DataFrame(d)
    lg = float(d["y"].mean())
    up = [f"k{i}" for i in range(len(upper_keys))]
    lo = [f"k{i}" for i in range(len(lower_keys))]
    gu = d.groupby(up)["y"].agg(["sum", "size"])
    gl = d.groupby(lo)["y"].agg(["sum", "size"])
    ebu = (gu["sum"] + K * lg) / (gu["size"] + K)
    ebl = (gl["sum"] + K * lg) / (gl["size"] + K)
    ku = pd.MultiIndex.from_arrays([k for k in upper_keys]) if len(upper_keys) > 1 \
        else pd.Index(upper_keys[0])
    kl = pd.MultiIndex.from_arrays([k for k in lower_keys])
    vu = ebu.reindex(ku).to_numpy()
    vl = ebl.reindex(kl).to_numpy()
    vu = np.where(np.isnan(vu), lg, vu)
    vl = np.where(np.isnan(vl), lg, vl)
    sz = gl["size"].reindex(kl).fillna(0.0).to_numpy()
    return vl - vu, sz / (sz + K)


CANDS = {
    "P_cnt":  ([pid], [pid, cnt]),
    "B_cnt":  ([bid], [bid, cnt]),
    "B_hc":   ([bid, phand], [bid, phand, cnt]),
    "P_hi":   ([pid, bhand], [pid, bhand, inn]),
    "P_hb":   ([pid, bhand], [pid, bhand, runners]),
}

print("2단계 차감 후 산포 (채택 기준: 투수 전역 0.01775 / 카운트별 0.01526)")
keep, splits = [], {}
for name, (up, lo) in CANDS.items():
    s, r = layered_split(up, lo)
    splits[name] = (s, r)
    hi = r > 0.5
    sd = float(s[hi].std()) if hi.sum() else 0.0
    mark = "학습 진행" if sd >= SD_FLOOR else "산포 부족 - 기각"
    print(f"  {name:<7} sd {sd:.5f}  신뢰행 {100*hi.mean():5.1f}%  "
          f"p1~p99 {np.percentile(s[hi],1):+.4f}~{np.percentile(s[hi],99):+.4f}   {mark}",
          flush=True)
    if sd >= SD_FLOOR:
        keep.append(name)

if not keep:
    print("\n전 후보 산포 부족. 학습 없이 종료한다.")
    sys.exit(0)

train_df = df.loc[tr]
spec = CF.make_spec(train_df)
platoon = CF.make_platoon_table(train_df)
bat = CF.make_batter_platoon_table(train_df, {k: v[tr] for k, v in LAB.items()})
cplat = CF.make_count_platoon_table(train_df)
base_feat = CF.build(df[INPUT_COLS], spec, platoon, bat, cplat)
print(f"\n기준 피처 {base_feat.shape[1]}개 (submit_028 구성)", flush=True)


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


def run(add):
    out = base_feat.copy()
    for nm in add:
        s, r = splits[nm]
        out[f"{nm}_split"] = s
        out[f"{nm}_rel"] = r
        out[f"{nm}_w"] = s * r
    X = out.to_numpy(np.float32)
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr),
               **params_for(float(np.nanmean(arr[tr])))}
        d_tr = xgb.DMatrix(X[m], label=arr[m]); d_va = xgb.DMatrix(X[va])
        p_tr, p_va = Pool(X[m], arr[m]), Pool(X[va])
        acc = np.zeros(int(va.sum()))
        for s_ in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s_}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s_, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(p_va)[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return (np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS),
            out.shape[1])


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())

t0, rows = time.time(), []
arms = [("J0_current", [])] + [(f"J_{n}", [n]) for n in keep]
if len(keep) > 1:
    arms.append(("J_all", keep))
print(f"\n{'arm':<16}{'피처':>5}{'단독BSS':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}",
      flush=True)
for name, add in arms:
    ie, nf = run(add)
    solo = metrics(y_va, ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(p_prod / (1 - p_prod)),
                             np.log(ie / (1 - ie)))[0, 1])
    q = np.clip(W * ie + (1 - W) * p_prod, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - bm["bss_raw"]
    dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "n_features": nf, "solo_bss": solo, "corr": corr,
                 "bss": mm["bss_raw"], "dbss": d, "se_row": se, "t_row": d / se})
    print(f"{name:<16}{nf:>5}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}{d/se:>8.2f}"
          f"   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v22_layered_splits.csv", index=False)
ref = res[res.arm == "J0_current"]["dbss"].iloc[0]
best = res.sort_values("dbss", ascending=False).iloc[0]
print(f"\n기준선 J0 {ref:+.3f}   최고 {best.arm} {best.dbss:+.3f}  "
      f"차이 {best.dbss-ref:+.3f}  t_row {best.t_row:+.2f}")
print(f"\nsaved -> {OUT/'v22_layered_splits.csv'}")
