"""P9: 실제 제출 시스템(836) base 위에서 P7/D2 이득이 남는지 — 결정적 테스트.

지금까지 실험은 전부 77피처 간이 base(Val2024 BSS 687) 위에서 측정했다.
세 fold를 관통하는 패턴은 "ΔBSS가 base 품질과 역상관"이다.

    fold      base BSS    D2 ΔBSS
    2022        2303       -20.9
    2024         687       +16.1
    2023       -1147       +57.9

실제 제출 base는 836이므로 이득이 줄거나 사라질 수 있다. 추측으로 두지 않고 잰다.

프로덕션 예측 출처
    experiment/model_optimization/pitcher_cluster_matchup/reports/
        reverse20_submission_oof.parquet   (Val2024 253,507행)
        컬럼 submit017/019/020/021_* — 021이 Val2024 836.503 시스템

두 가지를 테스트한다.
  (A) P7 성분결합  : w*p_ie + (1-w)*p_prod, R 행에만 적용
                     p_ie 는 독립적인 예측이므로 혼합이 그대로 성립한다.
  (B) D2 유형잔차  : logit(p_prod) + scale*delta_type
                     주의 — delta_type 은 내 base 의 잔차로 학습됐다. 프로덕션 base
                     가 이미 고친 부분과 겹칠 수 있으므로 이 테스트는 보수적 근사다.
                     그래도 이득이 남으면 프로덕션이 놓친 성분을 잡는다는 뜻이다.

출력: outputs/p9_production_port.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, forecast_base_rate, metrics)

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 300
OOF_SEASONS = [2020, 2021, 2022, 2023]
WS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
SCALES = [0.10, 0.25, 0.50, 0.75, 1.00]
LAM_PROF, LEAVES, FEATSET = 200, 6, "profile"      # D2가 2022에서 고른 설정
EPS = 1e-7


def logit(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
asof_n = df["asof_pitcher_n"].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1


def as_label(c):
    return np.where(ok, df[c].to_numpy(np.float64), np.nan)


y_m, y_r, y_o = as_label("y_middle"), as_label("y_reverse"), as_label("y_outside")
y_mr = np.where(ok, (y_m == 1) & (y_r == 1), np.nan)

# ------------------------------------------------------- 프로덕션 예측 정렬
prod = pd.read_parquet(PROD)
va = season == 2024
va_ids = df.loc[va, "row_id"].to_numpy()
prod = prod.set_index("row_id").reindex(va_ids)
assert prod["control_success"].to_numpy().astype(int).tolist() == \
       y_all[va].astype(int).tolist(), "row 정렬 불일치"
P_COLS = ["submit017_reconstructed", "submit019_reconstructed",
          "submit020_reverse20_s055_tabm", "submit021_reverse20_s040_tabm"]
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
null = y_va.mean() * (1 - y_va.mean())
print(f"Val2024 {va.sum():,}행  R {is_r.sum():,} / F {(~is_r).sum():,}", flush=True)
for c in P_COLS:
    m = metrics(y_va, prod[c].to_numpy(), game_type=gt)
    print(f"  {c:<32} BSS {m['bss_raw']:9.3f}  R {m['r_bss']:9.3f}  F {m['f_bss']:9.3f}",
          flush=True)
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
base_m = metrics(y_va, p_prod, game_type=gt)


def build_X(tr_mask):
    priors = make_priors(df.loc[tr_mask])
    feat = encode(add_stateless(df, priors))
    cols = [c for c in feat.columns if c not in DROP and not c.startswith("y_")
            and c != "label_ok"]
    return feat[cols].to_numpy(np.float32)


def bag(X, tr_mask, pr_mask, labels, base_score):
    m = tr_mask & ~np.isnan(labels)
    d_tr = xgb.DMatrix(X[m], label=labels[m])
    d_pr = xgb.DMatrix(X[pr_mask])
    acc = np.zeros(int(pr_mask.sum()))
    for s in SEEDS:
        acc += xgb.train({**BASE_PARAMS, "base_score": base_score, "seed": s},
                         d_tr, num_boost_round=N_ROUNDS,
                         verbose_eval=False).predict(d_pr)
    return acc / len(SEEDS)


def forecast_rate(labels, tr_mask, vs):
    m = tr_mask & ~np.isnan(labels)
    s = pd.Series(labels[m]).groupby(pd.Series(season[m])).mean().sort_index()
    ls, lr = float(s.index[-1]), float(s.iloc[-1])
    slope = (lr - float(s.iloc[0])) / (ls - float(s.index[0]))
    return float(np.clip(lr + slope * (vs - ls), 0.005, 0.995))


# ------------------------------------------------------------ 성분 예측 (A)
tr = season < 2024
X = build_X(tr)
comp = {}
for tag, arr in [("m", y_m), ("r", y_r), ("o", y_o), ("mr", y_mr)]:
    comp[tag] = np.clip(bag(X, tr, va, arr, forecast_rate(arr, tr, 2024)), EPS, 1 - EPS)
    print(f"  [{tag}] pred_mean {comp[tag].mean():.5f}", flush=True)
p_ie = np.clip(1 - (comp["m"] + comp["r"] - comp["mr"] + comp["o"]), EPS, 1 - EPS)
print(f"  p_ie BSS {metrics(y_va, p_ie)['bss_raw']:.3f} (단독)", flush=True)

# ------------------------------------------------------- 유형 잔차 delta (B)
print("순방향 체인 OOF -> 유형 잔차", flush=True)
oof = np.full(len(df), np.nan)
for s in OOF_SEASONS:
    t, v = season < s, season == s
    oof[v] = bag(build_X(t), t, v, y_all, forecast_base_rate(df, t, s))
p_my = np.clip(bag(X, tr, va, y_all, forecast_base_rate(df, tr, 2024)), EPS, 1 - EPS)
print(f"  내 base Val2024 BSS {metrics(y_va, p_my)['bss_raw']:.3f}", flush=True)

RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
         "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
         "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
         "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
PREV_S = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
PREV_M = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]
prof = {}
for c in RATES:
    pr = float(df.loc[tr, c].median())
    r = np.where(np.isnan(df[c].to_numpy(np.float64)), pr, df[c].to_numpy(np.float64))
    prof[c] = (asof_n * r + LAM_PROF * pr) / (asof_n + LAM_PROF)
for c in PREV_S + PREV_M:
    pr = float(df.loc[tr, c].median())
    v = df[c].to_numpy(np.float64)
    prof[c] = np.where(np.isnan(v), pr, v)
    prof["miss_" + c] = np.isnan(v).astype(np.float64)
prof["log1p_n"] = np.log1p(asof_n)
prof["bs_gap"] = prof[RATES[3]] - prof[RATES[4]]
prof["trend"] = prof[PREV_S[0]] - prof[PREV_S[2]]
for k, (cs, cm) in enumerate(zip(PREV_S, PREV_M)):
    prof[f"faildir_{k}"] = prof[cm] - (1 - prof[cs])
P = pd.DataFrame(prof).to_numpy(np.float32)

src = (season < 2024) & ~np.isnan(oof)
d_tr = xgb.DMatrix(P[src], label=y_all[src])
d_tr.set_base_margin(logit(oof[src]))
acc = np.zeros(int(va.sum()))
for s in SEEDS:
    dv = xgb.DMatrix(P[va])
    dv.set_base_margin(logit(p_my))
    acc += xgb.train({**BASE_PARAMS, "max_leaves": LEAVES, "seed": s},
                     d_tr, num_boost_round=N_ROUNDS,
                     verbose_eval=False).predict(dv, output_margin=True) - logit(p_my)
delta_type = acc / len(SEEDS)
print(f"  delta_type |mean| {np.abs(delta_type).mean():.5f}", flush=True)

# ----------------------------------------------------------------- 평가
rows = []


def ev(name, p, note=""):
    p = np.clip(p, EPS, 1 - EPS)
    mm = metrics(y_va, p, game_type=gt)
    d = mm["bss_raw"] - base_m["bss_raw"]
    dr_ = (p_prod - y_va) ** 2 - (p - y_va) ** 2
    se = 100000 * float(dr_.std(ddof=1) / np.sqrt(len(dr_))) / null
    rows.append({"variant": name, "bss": mm["bss_raw"], "dbss": d,
                 "se_row": se, "t_row": d / se if se > 0 else np.nan,
                 "r_bss": mm["r_bss"], "dr": mm["r_bss"] - base_m["r_bss"],
                 "f_bss": mm["f_bss"], "df": mm["f_bss"] - base_m["f_bss"],
                 "brier": mm["brier"], "pred_mean": mm["pred_mean"], "note": note})


ev("prod_submit021_base", p_prod, "기준")
for w in WS:
    q = p_prod.copy()
    q[is_r] = w * p_ie[is_r] + (1 - w) * p_prod[is_r]
    ev(f"A_ieR_w{w:.2f}", q, "P7 성분결합 R한정")
for sc in SCALES:
    ev(f"B_type_s{sc:.2f}", sigmoid(logit(p_prod) + sc * delta_type), "D2 유형잔차(근사)")
best_w = None
for w in WS:
    for sc in SCALES:
        q = p_prod.copy()
        q[is_r] = w * p_ie[is_r] + (1 - w) * p_prod[is_r]
        ev(f"C_both_w{w:.2f}_s{sc:.2f}", sigmoid(logit(q) + sc * delta_type), "결합")

res = pd.DataFrame(rows)
res.to_csv(OUT / "p9_production_port.csv", index=False)
print("\n" + "=" * 100)
print(f"★ 프로덕션 base(submit_021) Val2024 BSS {base_m['bss_raw']:.3f} 대비")
print("=" * 100)
print(f"{'variant':<24}{'BSS':>11}{'ΔBSS':>10}{'SE_row':>8}{'t_row':>8}"
      f"{'ΔR':>9}{'ΔF':>10}")
for _, r in res.sort_values("dbss", ascending=False).head(22).iterrows():
    print(f"{r.variant:<24}{r.bss:>11.3f}{r.dbss:>10.3f}{r.se_row:>8.3f}"
          f"{r.t_row:>8.2f}{r.dr:>9.3f}{r.df:>10.3f}")
print(f"\nsaved -> {OUT/'p9_production_port.csv'}")
