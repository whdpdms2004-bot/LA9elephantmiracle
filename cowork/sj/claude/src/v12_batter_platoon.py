"""V12: 타자 플래툰 스플릿 + 타자 성분 프로파일.

논리
    투수 플래툰이 +13.75(Public 945)를 만든 근거는 팀통합 1-1 이다.
        asof_pitcher_success_rate 에 좌우 구분이 없어 15%p 가 통째로 사라진다.
    같은 논리가 타자 쪽에도 성립한다.
        asof_batter_success_rate 에도 좌우 구분이 없다.

    팀 문서 어디에도 타자 플래툰 스플릿이 없다. 프로덕션 batter_lookup_2025.csv 는
    batter_overall_resid(전체 잔차)만 갖고 좌우 스플릿이 없다. sj 97피처에도 없다.

    구분선 기준 통과: 현재 모델은 handedness_matchup 코드(4값)만 볼 뿐
    '타자 개인의 좌우 편차'를 모른다. 새 정보다.

arm
    G0  현행 (투수 플래툰만)                                 <- 기준선
    G1  + 타자 플래툰 스플릿 (control_success, K=300)
    G2  + 타자 성분 프로파일 (성분 5개의 타자별 EB, 주효과 차감)
    G3  + 둘 다
    G4  + 타자 플래툰을 성분별로 (5개)

주효과 차감은 필수다. V1 에서 정적 레벨이 direct_bss 705.7 -> 187.5 로 붕괴한
것이 증거다. 타자 쪽도 같은 함정이 있다 - asof_batter_success_rate 와 중복된다.

판정: Val2024 전체 BSS, 프로덕션 836.503 대비, 균일 w=0.20 (submit_026 구성).
출력: outputs/v12_batter_platoon.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics)

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS, K_EB = 400, 300
W = 0.20
EPS = 1e-7

RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
         "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
         "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
         "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
PREV_S = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
PREV_M = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]


def logit(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
asof_n = df["asof_pitcher_n"].to_numpy(np.float64)
pid = df["pitcher_id"].to_numpy()
bid = df["batter_id"].to_numpy()
phand = df["pitcher_hand"].to_numpy()
bhand = df["batter_hand"].to_numpy()
ok = df["label_ok"].to_numpy() == 1
tr, va = season < 2024, season == 2024

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}


def eb_split(actor, opp_hand, target, K=K_EB):
    """split(actor, opp_hand) = EB(actor x 상대손) - EB(actor 전체). 주효과 차감."""
    m = tr & ~np.isnan(target)
    d = pd.DataFrame({"a": actor[m], "h": opp_hand[m], "y": target[m]})
    lg = float(d["y"].mean())
    ga = d.groupby("a")["y"].agg(["sum", "size"])
    gh = d.groupby(["a", "h"])["y"].agg(["sum", "size"])
    eb_a = (ga["sum"] + K * lg) / (ga["size"] + K)
    eb_ah = (gh["sum"] + K * lg) / (gh["size"] + K)
    key = pd.MultiIndex.from_arrays([actor, opp_hand])
    lv = np.where(np.isnan(eb_ah.reindex(key).to_numpy()), lg,
                  eb_ah.reindex(key).to_numpy())
    pv = np.where(np.isnan(pd.Series(actor).map(eb_a).to_numpy()), lg,
                  pd.Series(actor).map(eb_a).to_numpy())
    sz = gh["size"].reindex(key).fillna(0.0).to_numpy()
    return lv - pv, sz / (sz + K)


def eb_profile(actor, target, K=K_EB):
    """EB(actor 의 성분 발생률) - 리그평균. 레벨이 아니라 리그 대비 편차."""
    m = tr & ~np.isnan(target)
    d = pd.DataFrame({"a": actor[m], "y": target[m]})
    lg = float(d["y"].mean())
    ga = d.groupby("a")["y"].agg(["sum", "size"])
    eb_a = (ga["sum"] + K * lg) / (ga["size"] + K)
    v = np.where(np.isnan(pd.Series(actor).map(eb_a).to_numpy()), lg,
                 pd.Series(actor).map(eb_a).to_numpy())
    sz = ga["size"].reindex(pd.Index(actor)).fillna(0.0).to_numpy()
    return v - lg, sz / (sz + K)


def build(bat_platoon, bat_profile, bat_comp_platoon):
    priors = make_priors(df.loc[tr])
    base = encode(add_stateless(df, priors))
    cols = [c for c in base.columns if c not in DROP and not c.startswith("y_")
            and c != "label_ok"]
    out = base[cols].copy()
    n = asof_n
    for c in RATES:
        pr = float(df.loc[tr, c].median())
        r = np.where(np.isnan(df[c].to_numpy(np.float64)), pr, df[c].to_numpy(np.float64))
        out[f"prof200_{c}"] = (n * r + 200 * pr) / (n + 200)
    ps = {c: np.where(np.isnan(df[c].to_numpy(np.float64)),
                      float(df.loc[tr, c].median()),
                      df[c].to_numpy(np.float64)) for c in PREV_S + PREV_M}
    out["prev_trend_s"] = ps[PREV_S[0]] - ps[PREV_S[2]]
    out["prev_trend_m"] = ps[PREV_M[0]] - ps[PREV_M[2]]
    out["prev_std_s"] = np.std(np.vstack([ps[c] for c in PREV_S]), axis=0)
    out["prev_std_m"] = np.std(np.vstack([ps[c] for c in PREV_M]), axis=0)
    out["prev_miss_cnt"] = sum(np.isnan(df[c].to_numpy(np.float64)).astype(np.float64)
                               for c in PREV_S + PREV_M)
    for k, (cs, cm) in enumerate(zip(PREV_S, PREV_M)):
        out[f"faildir_{k}"] = ps[cm] - (1 - ps[cs])
    out["rel200"] = n / (n + 200.0)
    sp, rel = eb_split(pid, bhand, y_all)                 # 투수 플래툰 (현행)
    out["platoon_split"], out["platoon_split_rel"] = sp, rel
    out["platoon_split_w"] = sp * rel
    if bat_platoon:
        bs, br = eb_split(bid, phand, y_all)              # 타자 플래툰 (신규)
        out["bat_platoon_split"], out["bat_platoon_rel"] = bs, br
        out["bat_platoon_split_w"] = bs * br
    if bat_profile:
        for tag, arr in LAB.items():                      # 타자 성분 프로파일
            v, r_ = eb_profile(bid, arr)
            out[f"bat_prof_{tag}"] = v
        out["bat_prof_rel"] = r_
    if bat_comp_platoon:
        for tag, arr in LAB.items():                      # 타자 성분별 플래툰
            v, r_ = eb_split(bid, phand, arr)
            out[f"bat_pl_{tag}"] = v
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


def fit_pair(X, arr):
    m = tr & ~np.isnan(arr)
    prm = {**BASE_PARAMS, "base_score": extrap(arr),
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
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


# 타자 플래툰의 크기를 먼저 본다 — 투수 쪽과 비교
sp_p, rel_p = eb_split(pid, bhand, y_all)
sp_b, rel_b = eb_split(bid, phand, y_all)
print("플래툰 스플릿 산포 비교 (K=300, 신뢰도 0.5 이상 행)")
for nm, s_, r_ in [("투수(현행)", sp_p, rel_p), ("타자(신규)", sp_b, rel_b)]:
    h = r_ > 0.5
    print(f"  {nm}  sd {s_[h].std():.5f}  p1~p99 "
          f"{np.percentile(s_[h],1):+.4f}~{np.percentile(s_[h],99):+.4f}  "
          f"신뢰행 {100*h.mean():.1f}%", flush=True)

prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())

ARMS = [("G0_pitcher_only", False, False, False),
        ("G1_bat_platoon", True, False, False),
        ("G2_bat_profile", False, True, False),
        ("G3_bat_both", True, True, False),
        ("G4_bat_comp_platoon", True, False, True)]

t0, rows = time.time(), []
print(f"\n{'arm':<22}{'피처':>5}{'단독BSS':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}"
      f"{'R':>10}{'F':>10}", flush=True)
for name, bp, bpr, bcp in ARMS:
    X, cols = build(bp, bpr, bcp)
    p = {t: fit_pair(X, a) for t, a in LAB.items()}
    ie = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
    solo = metrics(y_va, ie)["bss_raw"]
    corr = float(np.corrcoef(logit(p_prod), logit(ie))[0, 1])
    q = np.clip(W * ie + (1 - W) * p_prod, EPS, 1 - EPS)      # 균일 적용 (026 구성)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - bm["bss_raw"]
    dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "n_features": len(cols), "solo_bss": solo, "corr": corr,
                 "bss": mm["bss_raw"], "dbss": d, "se_row": se, "t_row": d / se,
                 "r_bss": mm["r_bss"], "f_bss": mm["f_bss"]})
    print(f"{name:<22}{len(cols):>5}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}{d/se:>8.2f}"
          f"{mm['r_bss']:>10.2f}{mm['f_bss']:>10.2f}   [{time.time()-t0:.0f}s]",
          flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v12_batter_platoon.csv", index=False)
ref = res[res.arm == "G0_pitcher_only"]["dbss"].iloc[0]
best = res.sort_values("dbss", ascending=False).iloc[0]
print(f"\n기준선 G0 {ref:+.3f}   최고 {best.arm} {best.dbss:+.3f}  "
      f"차이 {best.dbss-ref:+.3f}  t_row {best.t_row:+.2f}")
print(f"  [submit_026 은 20시드 +21.72, 여기는 8시드라 G0 가 그 근처여야 정상]")
print(f"\nsaved -> {OUT/'v12_batter_platoon.csv'}")
