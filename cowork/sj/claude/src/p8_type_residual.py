"""P8-D2 (Phase D): 투수 유형 residual — 프로파일 기반.

사용자 아이디어: 개인화가 어려운 저투구 투수를 이전 경기 투구 정보로 유형화한다.
구현은 하드 클러스터가 아니라 C3 안 — 프로파일 벡터를 그대로 residual 모델에 넣는다.
"유형"은 결국 프로파일 -> 잔차 함수이고, 클러스터는 그 함수를 계단으로 근사한다.

프로파일은 test.csv에 그대로 있는 asof 컬럼만 쓴다. 복원도 클러스터 룩업도 없다.
따라서 유형 배정이 행 단위로 끝나고 test 행 간 참조가 원천적으로 없다.

residual 학습은 base_margin 을 써서 로짓 보정을 직접 학습한다.
    d_train.set_base_margin(logit(p_oof))     target = y
    delta = predict(output_margin=True) - logit(p_base)
    p_final = sigmoid(logit(p_base) + scale * delta)

판정: 전체 BSS 단일. 선택 2022, 부호 확인 2023, Val2024 최종 게이트 1회.
      분모는 sigma_pair / SE_row. asof_pitcher_n 구간별 분해를 항상 남긴다.

출력: outputs/p8_type_residual_all.csv / _val2024.csv / _nbucket.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, BASE_PARAMS, load, make_priors, add_stateless,
                     encode, forecast_base_rate, metrics)

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 300
VALID_SEASONS = [2022, 2023, 2024]
OOF_SEASONS = [2020, 2021, 2022, 2023]
LAM_PROF = [50, 200, 500]
SCALES = [0.25, 0.50, 0.75, 1.00]
LEAVES = [6, 18]
FEATSETS = ["profile", "profile_ctx"]
N_BUCKETS = [0, 1, 100, 500, 1000, 4000, 10 ** 9]
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
pid = df["pitcher_id"].to_numpy()
asof_n = df["asof_pitcher_n"].to_numpy(np.float64)


def build_profile(tr_mask, lam_prof, with_ctx):
    """제공 asof 컬럼만으로 투수 프로파일. prior는 학습 fold에서만 계산."""
    out = {}
    n = asof_n
    for c in RATES:
        prior = float(df.loc[tr_mask, c].median())
        r = df[c].to_numpy(np.float64)
        miss = np.isnan(r)
        r = np.where(miss, prior, r)
        out[f"prof_{c}"] = (n * r + lam_prof * prior) / (n + lam_prof)
    for c in PREV_S + PREV_M:
        prior = float(df.loc[tr_mask, c].median())
        v = df[c].to_numpy(np.float64)
        out[f"prof_{c}"] = np.where(np.isnan(v), prior, v)
        out[f"miss_{c}"] = np.isnan(v).astype(np.float64)
    out["prof_log1p_n"] = np.log1p(n)
    out["prof_ball_strike_gap"] = (out["prof_asof_pitcher_ball_rate"]
                                   - out["prof_asof_pitcher_strike_rate"])
    out["prof_trend"] = (out[f"prof_{PREV_S[0]}"] - out[f"prof_{PREV_S[2]}"])
    # 최근 실패가 middle 때문인가 — 최근 폼은 success/middle 둘만 제공된다
    for k, (cs, cm) in enumerate(zip(PREV_S, PREV_M)):
        out[f"prof_faildir_{k}"] = out[f"prof_{cm}"] - (1 - out[f"prof_{cs}"])
    if with_ctx:
        out["ctx_count"] = (df["balls_before"].to_numpy() * 3
                            + df["strikes_before"].to_numpy())
        out["ctx_hand"] = (df["pitcher_hand"].to_numpy() * 2
                           + df["batter_hand"].to_numpy())
        out["ctx_outs"] = df["outs_before"].to_numpy()
        out["ctx_season"] = season
    return pd.DataFrame(out).to_numpy(np.float32), list(out)


def build_X(tr_mask):
    priors = make_priors(df.loc[tr_mask])
    feat = encode(add_stateless(df, priors))
    cols = [c for c in feat.columns if c not in DROP]
    return feat[cols].to_numpy(np.float32)


def per_seed_predict(X, tr_mask, pr_mask, base_score):
    d_tr = xgb.DMatrix(X[tr_mask], label=y_all[tr_mask])
    d_pr = xgb.DMatrix(X[pr_mask])
    out = np.empty((len(SEEDS), int(pr_mask.sum())))
    for i, s in enumerate(SEEDS):
        bst = xgb.train({**BASE_PARAMS, "base_score": base_score, "seed": s},
                        d_tr, num_boost_round=N_ROUNDS, verbose_eval=False)
        out[i] = bst.predict(d_pr)
    return out


print("순방향 체인 OOF 생성", flush=True)
oof = np.full(len(df), np.nan)
for s in OOF_SEASONS:
    tr, va = season < s, season == s
    oof[va] = per_seed_predict(build_X(tr), tr, va,
                               forecast_base_rate(df, tr, s)).mean(axis=0)
    print(f"  OOF {s}  BSS {metrics(y_all[va], oof[va])['bss_raw']:9.3f}", flush=True)

rows = []
store24 = {}
for vs in VALID_SEASONS:
    tr_mask, va_mask = season < vs, season == vs
    per = per_seed_predict(build_X(tr_mask), tr_mask, va_mask,
                           forecast_base_rate(df, tr_mask, vs))
    p_base = np.clip(per.mean(axis=0), EPS, 1 - EPS)
    y_va = y_all[va_mask]
    gt_va = df.loc[va_mask, "game_type"].astype(str).to_numpy()
    base_m = metrics(y_va, p_base, game_type=gt_va)
    null = y_va.mean() * (1 - y_va.mean())
    lg_base = logit(p_base)
    lg_seed = logit(np.clip(per, EPS, 1 - EPS))
    src = (season < vs) & ~np.isnan(oof)
    print(f"\n{'='*104}\nvalid {vs}   base BSS {base_m['bss_raw']:.3f}   "
          f"OOF 학습행 {src.sum():,}\n{'='*104}", flush=True)

    for fs in FEATSETS:
        for lam in LAM_PROF:
            P, names = build_profile(tr_mask, lam, fs == "profile_ctx")
            d_tr = xgb.DMatrix(P[src], label=y_all[src])
            d_tr.set_base_margin(logit(oof[src]))
            d_va = xgb.DMatrix(P[va_mask])
            d_va.set_base_margin(lg_base)
            for lv in LEAVES:
                acc = np.zeros(int(va_mask.sum()))
                for s in SEEDS:
                    bst = xgb.train({**BASE_PARAMS, "max_leaves": lv, "seed": s},
                                    d_tr, num_boost_round=N_ROUNDS, verbose_eval=False)
                    acc += bst.predict(d_va, output_margin=True) - lg_base
                delta = acc / len(SEEDS)
                for sc in SCALES:
                    adj = sc * delta
                    p_new = sigmoid(lg_base + adj)
                    mm = metrics(y_va, p_new, game_type=gt_va)
                    dbss = mm["bss_raw"] - base_m["bss_raw"]
                    d_row = (p_base - y_va) ** 2 - (p_new - y_va) ** 2
                    se_row = 100000 * float(d_row.std(ddof=1) / np.sqrt(len(d_row))) / null
                    ds = np.array([
                        100000 * (np.mean((np.clip(per[i], EPS, 1 - EPS) - y_va) ** 2)
                                  - np.mean((sigmoid(lg_seed[i] + adj) - y_va) ** 2)) / null
                        for i in range(len(SEEDS))])
                    sp = float(ds.std(ddof=1))
                    key = (fs, lam, lv, sc)
                    rows.append({"valid_season": vs, "featset": fs, "lam_prof": lam,
                                 "leaves": lv, "scale": sc, "n_features": len(names),
                                 "base_bss": base_m["bss_raw"], "bss": mm["bss_raw"],
                                 "dbss": dbss, "sigma_pair": sp,
                                 "t_pair": dbss / sp if sp > 0 else np.nan,
                                 "se_row": se_row,
                                 "t_row": dbss / se_row if se_row > 0 else np.nan,
                                 "r_bss": mm.get("r_bss"), "f_bss": mm.get("f_bss"),
                                 "brier": mm["brier"], "pred_mean": mm["pred_mean"],
                                 "abs_delta_mean": float(np.abs(adj).mean())})
                    if vs == 2024:
                        store24[key] = p_new
        print(f"  [{fs}] done", flush=True)

    cur = pd.DataFrame([r for r in rows if r["valid_season"] == vs])
    top = cur.sort_values("dbss", ascending=False).head(8)
    print(f"\n  {'featset':<14}{'lam':>5}{'lv':>4}{'sc':>6}{'BSS':>11}{'ΔBSS':>9}"
          f"{'σ_pair':>8}{'t_pair':>8}{'SE_row':>8}{'t_row':>8}", flush=True)
    for _, r in top.iterrows():
        print(f"  {r.featset:<14}{int(r.lam_prof):>5}{int(r.leaves):>4}{r.scale:>6.2f}"
              f"{r.bss:>11.3f}{r.dbss:>9.3f}{r.sigma_pair:>8.3f}{r.t_pair:>8.2f}"
              f"{r.se_row:>8.3f}{r.t_row:>8.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "p8_type_residual_all.csv", index=False)
res[res.valid_season == 2024].sort_values("dbss", ascending=False).to_csv(
    OUT / "p8_type_residual_val2024.csv", index=False)

KEY = ["featset", "lam_prof", "leaves", "scale"]
sel = res[res.valid_season == 2022].sort_values("dbss", ascending=False).iloc[0]
chosen = res[np.logical_and.reduce([res[k] == sel[k] for k in KEY])].set_index("valid_season")
print("\n" + "=" * 104)
print(f"정직한 게이트 — 2022 선택: {sel.featset} lam={int(sel.lam_prof)} "
      f"leaves={int(sel.leaves)} scale={sel.scale}")
print("=" * 104)
print(chosen[["base_bss", "bss", "dbss", "sigma_pair", "t_pair", "se_row",
              "t_row", "r_bss", "f_bss"]].round(3).to_string())

g = chosen.loc[2024]
print("\n" + "=" * 104)
print("★ Val2024 최종 게이트")
print("=" * 104)
print(f"  base BSS   {g.base_bss:.3f}\n  type BSS   {g.bss:.3f}\n"
      f"  ΔBSS       {g.dbss:+.3f}\n"
      f"  sigma_pair {g.sigma_pair:.3f}  t_pair {g.t_pair:+.2f}\n"
      f"  SE_row     {g.se_row:.3f}  t_row  {g.t_row:+.2f}")

print("\nVal2024 상위 12 (참고 — 선택은 2022)")
print(res[res.valid_season == 2024].sort_values("dbss", ascending=False).head(12)[
    ["featset", "lam_prof", "leaves", "scale", "bss", "dbss", "t_pair", "t_row"]
].round(3).to_string(index=False))

# ---------------------------------------- asof_pitcher_n 구간별 분해 (2024)
key24 = (sel.featset, sel.lam_prof, sel.leaves, sel.scale)
if key24 in store24:
    tr24, va24 = season < 2024, season == 2024
    p_b24 = np.clip(per_seed_predict(build_X(tr24), tr24, va24,
                                     forecast_base_rate(df, tr24, 2024)).mean(axis=0),
                    EPS, 1 - EPS)
    p_n24, y24, n24 = store24[key24], y_all[va24], asof_n[va24]
    buck = pd.cut(n24, N_BUCKETS, right=False,
                  labels=["0", "1-99", "100-499", "500-999", "1000-3999", "4000+"])
    nb = []
    for b in buck.categories:
        m = np.asarray(buck == b)
        if m.sum() < 100:
            continue
        nl = y24[m].mean() * (1 - y24[m].mean())
        if nl <= 0:
            continue
        bo = np.mean((p_b24[m] - y24[m]) ** 2)
        bn = np.mean((p_n24[m] - y24[m]) ** 2)
        nb.append({"bucket": b, "n": int(m.sum()), "share_pct": 100 * m.mean(),
                   "local_dbss": 100000 * (bo - bn) / nl,
                   "contrib_to_total_dbss": 100000 * (bo - bn) * m.sum() / len(y24)
                   / (y24.mean() * (1 - y24.mean()))})
    nbdf = pd.DataFrame(nb)
    nbdf.to_csv(OUT / "p8_type_residual_nbucket.csv", index=False)
    print("\n" + "=" * 104)
    print("Val2024 asof_pitcher_n 구간별 ΔBSS — 저투구 구간에서 실제로 개선되는가")
    print("=" * 104)
    print(nbdf.round(3).to_string(index=False))
    print(f"\n  구간 기여 합계 {nbdf['contrib_to_total_dbss'].sum():+.3f} "
          f"(전체 {g.dbss:+.3f})")

print(f"\nsaved -> {OUT/'p8_type_residual_all.csv'}")
