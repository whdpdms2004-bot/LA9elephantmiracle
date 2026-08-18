"""P3-D1 (Phase D): 투수 절편 residual — 부분 풀링.

완전분리 투수별 모델은 자체 실험에서 이미 기각됐다 (0723 A라운드, LogLoss
0.5865 vs 전체 0.5760). 여기서는 전체 모델 예측에 투수별 로짓 오프셋만 얹는다.

    delta_p = (r_p / v_p) * n_p/(n_p + lambda) * scale        # Newton 1스텝 + 수축
    p_final = sigmoid(logit(p_base) + delta_p)

  r_p : 투수 p의 OOF 잔차 가중평균 (y - p_base)
  v_p : 투수 p의 p(1-p) 가중평균
  n_p : recency 가중 표본수
  대상 : 학습 시즌 누적 투구수 >= T 인 투수만. 미달/신규는 delta=0 (base 보존)

OOF는 순방향 체인으로 만든다 (2020은 2019로, 2021은 2019~2020으로, ...).
검증 시즌 라벨은 오프셋 산출에 일절 쓰지 않는다.

판정 (P7에서 교정한 규약)
  지표      전체 BSS 단일. Val2024가 최종 게이트다.
  선택      2022 전체 BSS에서만. 2023은 부호 확인 (F 붕괴로 게이트 부적합).
  분모      sigma_pair = 시드별 dBSS 산포     (다시 돌리면 뒤집히나)
            SE_row     = 행별 쌍대 Brier 차   (다른 25만 행으로 전이되나)
            sigma_abs 는 참고로만. 같은 base를 공유하는 변형 판정에는 과도한 자다.
  분해      asof_pitcher_n 구간별 dBSS 를 항상 기록한다. 이 계층이 노리는 구간에서
            실제로 개선되는지가 설계 자체의 검증이다.

출력: outputs/p3_pitcher_offset_all.csv  (전 그리드)
      outputs/p3_pitcher_offset_val2024.csv  (Val2024 헤드라인)
      outputs/p3_pitcher_offset_nbucket.csv  (투구수 구간 분해)
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

THRESHOLDS = [100, 200, 500]
LAMBDAS = [50, 200, 500, 1000, 2000]
SCALES = [0.25, 0.50, 0.75, 1.00]
HALF_LIVES = [1.0, 2.0, 1e9]
N_BUCKETS = [0, 1, 100, 500, 1000, 4000, 10 ** 9]
EPS = 1e-7


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
pid = df["pitcher_id"].to_numpy()
asof_n = df["asof_pitcher_n"].to_numpy()


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


# ------------------------------------------------ 순방향 체인 OOF (배깅 평균)
print("순방향 체인 OOF 생성", flush=True)
oof = np.full(len(df), np.nan)
for s in OOF_SEASONS:
    tr, va = season < s, season == s
    X = build_X(tr)
    oof[va] = per_seed_predict(X, tr, va, forecast_base_rate(df, tr, s)).mean(axis=0)
    print(f"  OOF {s}  n={va.sum():,}  BSS {metrics(y_all[va], oof[va])['bss_raw']:9.3f}",
          flush=True)

rows, nbuck = [], []
for vs in VALID_SEASONS:
    tr_mask, va_mask = season < vs, season == vs
    X = build_X(tr_mask)
    per = per_seed_predict(X, tr_mask, va_mask, forecast_base_rate(df, tr_mask, vs))
    p_base = np.clip(per.mean(axis=0), EPS, 1 - EPS)
    y_va = y_all[va_mask]
    gt_va = df.loc[va_mask, "game_type"].astype(str).to_numpy()
    pid_va, n_va = pid[va_mask], asof_n[va_mask]
    base_m = metrics(y_va, p_base, game_type=gt_va)
    null = y_va.mean() * (1 - y_va.mean())

    abs_bss = np.array([metrics(y_va, np.clip(per[i], EPS, 1 - EPS))["bss_raw"]
                        for i in range(len(SEEDS))])
    sigma_abs = float(abs_bss.std(ddof=1))
    print(f"\n{'='*104}\nvalid {vs}   base BSS {base_m['bss_raw']:.3f}   "
          f"n={va_mask.sum():,}   sigma_abs(단일시드) {sigma_abs:.2f}\n{'='*104}",
          flush=True)

    src = (season < vs) & ~np.isnan(oof)
    s_season, s_pid = season[src], pid[src]
    s_p = np.clip(oof[src], EPS, 1 - EPS)
    s_res, s_var = y_all[src] - s_p, s_p * (1 - s_p)
    raw_cnt = pd.Series(pid[tr_mask]).value_counts()
    lg_base = logit(p_base)
    lg_seed = logit(np.clip(per, EPS, 1 - EPS))

    for hl in HALF_LIVES:
        w = 0.5 ** ((vs - 1 - s_season) / hl) if hl < 1e8 else np.ones(src.sum())
        agg = pd.DataFrame({"pid": s_pid, "w": w, "wr": w * s_res,
                            "wv": w * s_var}).groupby("pid").sum()
        delta_raw = (agg["wr"] / agg["w"]) / (agg["wv"] / agg["w"])
        n_p = agg["w"]
        for T in THRESHOLDS:
            eligible = set(raw_cnt[raw_cnt >= T].index)
            cover = np.isin(pid_va, list(eligible))
            for lam in LAMBDAS:
                dmap = (delta_raw * (n_p / (n_p + lam))).to_dict()
                d_va = np.where(cover, np.array([dmap.get(q, 0.0) for q in pid_va]), 0.0)
                for sc in SCALES:
                    adj = sc * d_va
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

                    rows.append({"valid_season": vs, "threshold": T, "lam": lam,
                                 "scale": sc, "half_life": hl,
                                 "n_pitchers": len(eligible),
                                 "coverage_pct": 100 * cover.mean(),
                                 "base_bss": base_m["bss_raw"], "bss": mm["bss_raw"],
                                 "dbss": dbss, "sigma_abs": sigma_abs,
                                 "sigma_pair": sp,
                                 "t_pair": dbss / sp if sp > 0 else np.nan,
                                 "se_row": se_row,
                                 "t_row": dbss / se_row if se_row > 0 else np.nan,
                                 "r_bss": mm.get("r_bss"), "f_bss": mm.get("f_bss"),
                                 "brier": mm["brier"], "pred_mean": mm["pred_mean"],
                                 "abs_delta_mean": float(np.abs(adj).mean())})

    cur = pd.DataFrame([r for r in rows if r["valid_season"] == vs])
    top = cur.sort_values("dbss", ascending=False).head(10)
    print(f"  {'T':>5}{'lam':>6}{'scale':>6}{'hl':>5}{'cov%':>7}{'BSS':>11}"
          f"{'ΔBSS':>9}{'σ_pair':>8}{'t_pair':>8}{'SE_row':>8}{'t_row':>8}", flush=True)
    for _, r in top.iterrows():
        print(f"  {int(r.threshold):>5}{int(r.lam):>6}{r.scale:>6.2f}"
              f"{('inf' if r.half_life>1e8 else f'{r.half_life:.0f}'):>5}"
              f"{r.coverage_pct:>7.2f}{r.bss:>11.3f}{r.dbss:>9.3f}"
              f"{r.sigma_pair:>8.3f}{r.t_pair:>8.2f}{r.se_row:>8.3f}{r.t_row:>8.2f}",
              flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "p3_pitcher_offset_all.csv", index=False)

# ------------------------------------------------------- 정직한 게이트
KEY = ["threshold", "lam", "scale", "half_life"]
sel = res[res.valid_season == 2022].sort_values("dbss", ascending=False).iloc[0]
chosen = res[np.logical_and.reduce([res[k] == sel[k] for k in KEY])].set_index("valid_season")

print("\n" + "=" * 104)
print(f"정직한 게이트 — 2022 전체 BSS로 선택: T={int(sel.threshold)} "
      f"lam={int(sel.lam)} scale={sel.scale} half_life={sel.half_life}")
print("=" * 104)
print(chosen[["coverage_pct", "base_bss", "bss", "dbss", "sigma_pair", "t_pair",
              "se_row", "t_row", "r_bss", "f_bss"]].round(3).to_string())

g24 = chosen.loc[2024]
print("\n" + "=" * 104)
print("★ Val2024 최종 게이트")
print("=" * 104)
print(f"  base BSS        {g24.base_bss:.3f}")
print(f"  offset BSS      {g24.bss:.3f}")
print(f"  ΔBSS            {g24.dbss:+.3f}")
print(f"  sigma_pair      {g24.sigma_pair:.3f}   t_pair {g24.t_pair:+.2f}")
print(f"  SE_row          {g24.se_row:.3f}   t_row  {g24.t_row:+.2f}")
print(f"  커버리지         {g24.coverage_pct:.2f}%")
print(f"  참고 sigma_abs  {g24.sigma_abs:.3f} (단일시드) / "
      f"{g24.sigma_abs/np.sqrt(len(SEEDS)):.3f} (8시드 배깅)")

# Val2024 헤드라인 — 상위 그리드 전체
v24 = res[res.valid_season == 2024].sort_values("dbss", ascending=False)
v24.to_csv(OUT / "p3_pitcher_offset_val2024.csv", index=False)
print("\nVal2024 상위 12 (참고 — 선택은 2022에서 했다)")
print(v24.head(12)[["threshold", "lam", "scale", "half_life", "coverage_pct",
                    "bss", "dbss", "t_pair", "t_row"]].round(3).to_string(index=False))

# --------------------------------------------- asof_pitcher_n 구간별 분해
print("\n" + "=" * 104)
print("Val2024 asof_pitcher_n 구간별 ΔBSS — 계층 설계가 노리는 구간에서 개선되는가")
print("=" * 104)
tr24, va24 = season < 2024, season == 2024
X24 = build_X(tr24)
per24 = per_seed_predict(X24, tr24, va24, forecast_base_rate(df, tr24, 2024))
p_b24 = np.clip(per24.mean(axis=0), EPS, 1 - EPS)
y24, n24, pid24 = y_all[va24], asof_n[va24], pid[va24]
raw_cnt24 = pd.Series(pid[tr24]).value_counts()
src24 = (season < 2024) & ~np.isnan(oof)
w24 = (0.5 ** ((2023 - season[src24]) / sel.half_life)
       if sel.half_life < 1e8 else np.ones(src24.sum()))
sp24 = np.clip(oof[src24], EPS, 1 - EPS)
a24 = pd.DataFrame({"pid": pid[src24], "w": w24, "wr": w24 * (y_all[src24] - sp24),
                    "wv": w24 * sp24 * (1 - sp24)}).groupby("pid").sum()
dm = ((a24["wr"] / a24["w"]) / (a24["wv"] / a24["w"])
      * (a24["w"] / (a24["w"] + sel.lam))).to_dict()
elig = set(raw_cnt24[raw_cnt24 >= sel.threshold].index)
d24v = np.where(np.isin(pid24, list(elig)),
                np.array([dm.get(q, 0.0) for q in pid24]), 0.0) * sel.scale
p_n24 = sigmoid(logit(p_b24) + d24v)
buck = pd.cut(n24, N_BUCKETS, right=False,
              labels=["0", "1-99", "100-499", "500-999", "1000-3999", "4000+"])
for b in buck.categories:
    m = np.asarray(buck == b)
    if m.sum() < 100:
        continue
    nl = y24[m].mean() * (1 - y24[m].mean())
    if nl <= 0:
        continue
    bo = np.mean((p_b24[m] - y24[m]) ** 2)
    bn = np.mean((p_n24[m] - y24[m]) ** 2)
    contrib = 100000 * (bo - bn) * m.sum() / len(y24) / (y24.mean() * (1 - y24.mean()))
    nbuck.append({"bucket": b, "n": int(m.sum()), "share_pct": 100 * m.mean(),
                  "delta_applied_pct": 100 * float((d24v[m] != 0).mean()),
                  "local_dbss": 100000 * (bo - bn) / nl,
                  "contrib_to_total_dbss": contrib})
nb = pd.DataFrame(nbuck)
nb.to_csv(OUT / "p3_pitcher_offset_nbucket.csv", index=False)
print(nb.round(3).to_string(index=False))
print(f"\n  구간 기여 합계 {nb['contrib_to_total_dbss'].sum():+.3f} "
      f"(전체 ΔBSS {g24.dbss:+.3f}와 일치해야 한다)")
print(f"\nsaved -> {OUT/'p3_pitcher_offset_all.csv'}, "
      f"{OUT/'p3_pitcher_offset_val2024.csv'}, {OUT/'p3_pitcher_offset_nbucket.csv'}")
