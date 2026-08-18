"""V19: 팀 ID 범주형 + 카운트별 플래툰 — 남은 미착수 후보 정리.

후보 1: pitcher_team_id / batter_team_id 범주형  (팀통합 6-3, 정희원 보류)
    "3/3 개선했으나 미실행". 팀 카탈로그의 '아직 아무도 안 한 것' 마지막 항목이다.
    현재 sj 라인은 두 ID 를 연속형 숫자로 넣는다 (크기에 순서 의미가 없는데도).
    구분선 기준: 팀 소속은 포수/투수코치/구장 같은 관측 안 되는 요인의 대리라
    '새 정보'일 수 있다. 다만 선수 ID 범주형은 팀 5명이 전원 기각했으므로
    팀 ID(13개)가 선수 ID(792개)와 다르게 작동하는지가 관건이다.

후보 2: 카운트별 플래툰 스플릿
    E1a 실측에서 middle-reverse 의존이 count_state 축에서 6/12 셀이 유의했다
    (|phi|>0.03). 좌우 효과가 카운트에 따라 다를 수 있다.
        split(p, h, count_bucket) = EB(투수 x 타자손 x 카운트군) - EB(투수 x 타자손)
    2단계 차감이다. 전역 플래툰을 빼고 남는 카운트별 편차만 본다.

    V8 에서 성분별 플래툰이 실패한 이유가 "전역이 이미 지배한다"였으므로
    여기서도 2단계 차감이 필수다.

arm
    H0  현행 (submit_027 구성)
    H1  + 팀 ID 를 범주형 코드로 (CatBoost 는 범주형 지정, XGB 는 원핫 대신 코드)
    H2  + 카운트별 플래툰 (볼카운트 3군: 투수우세 / 중립 / 타자우세)
    H3  + 둘 다

판정: Val2024 전체 BSS, 프로덕션 836.503 대비, 균일 w=0.20. 세 fold 확인은
      통과한 arm 에 대해서만 별도로 한다.
출력: outputs/v19_team_count.csv
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
N_ROUNDS, W = 400, 0.20
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
K_EB = 300
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
bhand = df["batter_hand"].to_numpy()
balls = df["balls_before"].to_numpy()
strikes = df["strikes_before"].to_numpy()

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}

# 볼카운트 3군: 투수우세(0) / 중립(1) / 타자우세(2)
cnt_bucket = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))


def count_platoon(K=K_EB):
    """2단계 차감. 전역 플래툰을 뺀 뒤 남는 카운트별 편차만 본다.

    V8 에서 성분별 플래툰이 실패한 이유가 '전역이 이미 지배한다'였으므로
    여기서도 전역을 명시적으로 빼야 새 정보만 남는다.
    """
    m = tr
    d = pd.DataFrame({"p": pid[m], "h": bhand[m], "c": cnt_bucket[m], "y": y_all[m]})
    lg = float(d["y"].mean())
    g_ph = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    eb_ph = (g_ph["sum"] + K * lg) / (g_ph["size"] + K)
    g_phc = d.groupby(["p", "h", "c"])["y"].agg(["sum", "size"])
    eb_phc = (g_phc["sum"] + K * lg) / (g_phc["size"] + K)
    k3 = pd.MultiIndex.from_arrays([pid, bhand, cnt_bucket])
    k2 = pd.MultiIndex.from_arrays([pid, bhand])
    v3 = eb_phc.reindex(k3).to_numpy()
    v2 = eb_ph.reindex(k2).to_numpy()
    v3 = np.where(np.isnan(v3), lg, v3)
    v2 = np.where(np.isnan(v2), lg, v2)
    sz = g_phc["size"].reindex(k3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


train_df = df.loc[tr]
spec = CF.make_spec(train_df)
platoon = CF.make_platoon_table(train_df)
bat = CF.make_batter_platoon_table(train_df, {k: v[tr] for k, v in LAB.items()})
base_feat = CF.build(df[INPUT_COLS], spec, platoon, bat)
print(f"기준 피처 {base_feat.shape[1]}개", flush=True)

cp_split, cp_rel = count_platoon()
hi = cp_rel > 0.5
print(f"카운트별 플래툰 산포  sd {cp_split[hi].std():.5f}  "
      f"p1~p99 {np.percentile(cp_split[hi],1):+.4f}~{np.percentile(cp_split[hi],99):+.4f}"
      f"  신뢰행 {100*hi.mean():.1f}%", flush=True)
print(f"  (참고: 투수 전역 플래툰 sd 0.01775, 타자 0.01300)", flush=True)


def build(team_cat, count_pl):
    out = base_feat.copy()
    if team_cat:
        # 13개 팀. 연속형으로 들어가 있던 걸 명시적 범주 코드로 재표현
        for c in ("pitcher_team_id", "batter_team_id"):
            codes = pd.Series(df[c].astype(str)).astype("category").cat.codes.to_numpy()
            out[f"cat_{c}"] = codes.astype(np.float64)
        out["cat_team_pair"] = (out["cat_pitcher_team_id"] * 13
                                + out["cat_batter_team_id"])
        out["cat_same_team"] = (df["pitcher_team_id"].to_numpy()
                                == df["batter_team_id"].to_numpy()).astype(np.float64)
    if count_pl:
        out["count_platoon_split"] = cp_split
        out["count_platoon_rel"] = cp_rel
        out["count_platoon_w"] = cp_split * cp_rel
    return out.to_numpy(np.float32), list(out.columns)


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


def fit_pair(X, arr, cat_cols, cols):
    """cat_cols 가 있으면 CatBoost 에 DataFrame 을 넘긴다.

    CatBoost 는 float numpy 행렬에 cat_features 를 못 받는다. 해당 컬럼만
    정수 dtype 으로 만든 DataFrame 이 필요하다.
    """
    m = tr & ~np.isnan(arr)
    prm = {**BASE_PARAMS, "base_score": extrap(arr),
           **params_for(float(np.nanmean(arr[tr])))}
    d_tr = xgb.DMatrix(X[m], label=arr[m]); d_va = xgb.DMatrix(X[va])
    if cat_cols:
        F = pd.DataFrame(X, columns=cols)
        for c in cat_cols:
            F[c] = F[c].astype(np.int32)
        p_tr = Pool(F[m], arr[m], cat_features=cat_cols)
        p_va = Pool(F[va], cat_features=cat_cols)
    else:
        p_tr, p_va = Pool(X[m], arr[m]), Pool(X[va])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                               verbose_eval=False).predict(d_va)
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="CPU" if cat_cols else "GPU",
                               verbose=0)
        c.fit(p_tr)
        acc += 0.5 * c.predict_proba(p_va)[:, 1]
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())

t0, rows = time.time(), []
print(f"\n{'arm':<18}{'피처':>5}{'단독BSS':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}",
      flush=True)
for name, tc, cp in [("H0_current", False, False), ("H1_team_cat", True, False),
                     ("H2_count_pl", False, True), ("H3_both", True, True)]:
    X, cols = build(tc, cp)
    cat_cols = [c for c in cols if c.startswith("cat_")] if tc else None
    p = {t: fit_pair(X, a, cat_cols, cols) for t, a in LAB.items()}
    ie = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
    solo = metrics(y_va, ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(p_prod / (1 - p_prod)),
                             np.log(ie / (1 - ie)))[0, 1])
    q = np.clip(W * ie + (1 - W) * p_prod, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - bm["bss_raw"]
    dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "n_features": len(cols), "solo_bss": solo, "corr": corr,
                 "bss": mm["bss_raw"], "dbss": d, "se_row": se, "t_row": d / se})
    print(f"{name:<18}{len(cols):>5}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}{d/se:>8.2f}"
          f"   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v19_team_count.csv", index=False)
ref = res[res.arm == "H0_current"]["dbss"].iloc[0]
best = res.sort_values("dbss", ascending=False).iloc[0]
print(f"\n기준선 H0 {ref:+.3f}   최고 {best.arm} {best.dbss:+.3f}  "
      f"차이 {best.dbss-ref:+.3f}")
print(f"\nsaved -> {OUT/'v19_team_count.csv'}")
