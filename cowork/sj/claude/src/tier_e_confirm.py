"""Tier E variant F 확인 검증 + R 전용 모델(P0-6 G) 결합.

variant F = SVD dim4 / 증거 시즌 >= 2022 / 신뢰도 열 제거 / half_life 2
  ablation 결과: R-only  F23 +1.16 / F24 +19.58,  F24 dAUC +0.00097  -> G1 통과

한계: 증거>=2022 필터는 fold 2022 에서 정의되지 않는다(2022 가 단절 연도 자체).
      따라서 세 번째 fold 로 확인할 수 없다. 대신 검증 시즌을 월 블록으로 쪼개
      블록별 부호 일치를 본다 (06_COMPUTE_STRATEGY §1 의 축소판).

결합: P0-6 의 G(부분 분리) 와 함께 적용하면 이득이 더해지는지 확인.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import tier_e_eval as T

DATA, OUT = T.DATA, T.OUT
ROUNDS = 250


def bss(p, y):
    r = y.mean()
    return (1 - np.mean((p - y) ** 2) / (r * (1 - r))) * 1e5


df = pd.read_csv(f"{DATA}/train.csv")
tm = pd.read_csv(f"{DATA}/trackman_history.csv", usecols=sorted(set(T.TM_FP)))
asof = [c for c in df.columns if c.startswith("asof_")]
ids = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
sit = [c for c in df.columns if c not in asof + ids + ["row_id", T.TARGET]]
gt = df.game_type.astype(str).values
fp_main = df[T.MAIN_FP].copy()
month = df.game_month.values.copy()
for c in ["top_bottom", "base_state", "game_type"]:
    df[c] = df[c].astype("category").cat.codes
BASE = sit + asof
y = df[T.TARGET].values.astype(np.float64)
IS_F = gt == "F"
seasons = sorted(df.season.unique())


def make_variant_f(FOLD):
    emb = pd.read_parquet(f"{T.TE}/tier_e_cutoff{FOLD}.parquet")
    cw = T.soft_crosswalk(fp_main, tm, FOLD, tau=0.05, s_min=0.60)
    cw = cw[cw.evidence_season >= 2022]
    prof = T.build_asof(cw, emb, seasons, half_life=2.0)
    dims = [c for c in prof.columns if c.startswith("te_svd_")][:4]
    keep = dims + ["te_seasons", "te_season_gap", "te_available"]
    d = df.merge(prof[["pitcher_id", "season"] + keep], on=["pitcher_id", "season"],
                 how="left")
    d["te_available"] = d.te_available.fillna(0)
    return d, keep


def fit(d, cols, mask_tr, mask_pr):
    dtr = xgb.DMatrix(d.loc[mask_tr, cols], label=y[mask_tr], missing=np.nan)
    dpr = xgb.DMatrix(d.loc[mask_pr, cols], missing=np.nan)
    b = xgb.train(T.PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    return b.predict(dpr)


rows = []
for FOLD in [2023, 2024]:
    d, TEC = make_variant_f(FOLD)
    tr, va = (d.season < FOLD).values, (d.season == FOLD).values
    yv, fv, mv = y[va], IS_F[va], month[va]
    print(f"\n===== fold {FOLD}  커버리지 {d.loc[va,'te_available'].mean():.1%}", flush=True)

    P = {}
    P["1 BASE"] = fit(d, BASE, tr, va)
    P["2 BASE+TierE_F"] = fit(d, BASE + TEC, tr, va)
    # G 부분 분리: R 행은 R 전용 모델
    for nm, cols in [("3 G(R전용)", BASE), ("4 G + TierE_F", BASE + TEC)]:
        p = fit(d, cols, tr, va).copy()
        pR = fit(d, cols, tr & ~IS_F, va)
        p[~fv] = pR[~fv]
        P[nm] = p

    for name, p in P.items():
        rec = dict(fold=FOLD, config=name, bss=float(bss(p, yv)),
                   auc=float(roc_auc_score(yv, p)),
                   bss_R=float(bss(p[~fv], yv[~fv])), bss_F=float(bss(p[fv], yv[fv])))
        # 월 블록별 R-only ΔBrier 부호 (블록 안정성)
        rec["p"] = p
        rows.append(rec)
        print(f"  {name:16s} BSS={rec['bss']:9.2f}  AUC={rec['auc']:.5f}  "
              f"R={rec['bss_R']:9.2f}  F={rec['bss_F']:10.2f}", flush=True)

    # 월 블록 안정성: TierE 추가가 R-only Brier 를 블록마다 개선하는가
    base_p = P["1 BASE"]
    te_p = P["2 BASE+TierE_F"]
    print("  월 블록별 R-only ΔBrier (음수=개선):", flush=True)
    sign_ok = 0
    blocks = sorted(set(mv[~fv]))
    for m in blocks:
        sel = (~fv) & (mv == m)
        if sel.sum() < 5000:
            continue
        dB = float(np.mean((te_p[sel] - yv[sel]) ** 2) - np.mean((base_p[sel] - yv[sel]) ** 2))
        sign_ok += dB < 0
        print(f"    {int(m)}월 n={int(sel.sum()):6d}  ΔBrier={dB:+.6e}  "
              f"{'개선' if dB<0 else '악화'}", flush=True)
    print(f"  -> 개선 블록 {sign_ok}/{len([m for m in blocks if ((~fv)&(mv==m)).sum()>=5000])}",
          flush=True)

res = pd.DataFrame([{k: v for k, v in r.items() if k != "p"} for r in rows])
res.to_csv(f"{OUT}/tier_e_confirm.csv", index=False)
print("\n" + "=" * 92)
piv = res.pivot_table(index="config", columns="fold", values=["bss", "bss_R", "auc"])
print(piv.round(5).to_string())
print("\n기준(1 BASE) 대비 R-only ΔBSS:")
for cfg in res.config.unique():
    if cfg == "1 BASE":
        continue
    o = []
    for f in [2023, 2024]:
        b = res[(res.fold == f) & (res.config == "1 BASE")].bss_R.iloc[0]
        v = res[(res.fold == f) & (res.config == cfg)].bss_R.iloc[0]
        o.append(v - b)
    tot = [res[(res.fold == f) & (res.config == cfg)].bss.iloc[0]
           - res[(res.fold == f) & (res.config == "1 BASE")].bss.iloc[0] for f in [2023, 2024]]
    print(f"  {cfg:16s} R: F23 {o[0]:+8.2f} / F24 {o[1]:+8.2f}   "
          f"전체: {tot[0]:+8.2f} / {tot[1]:+8.2f}   "
          f"-> {'통과' if o[0] > 0 and o[1] > 0 else '실패'}")
