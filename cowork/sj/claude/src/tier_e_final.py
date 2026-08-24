"""Tier E 최종 변형: F(game_type) 행을 TierE 에서 제외.

앞선 결과의 문제
  구성 2 (BASE+TierE_F 전체 적용): R 은 F23 +1.16 / F24 +19.58 로 통과했으나
                                   F 집단이 F23 에서 -16006 -> -16654 (-648) 악화,
                                   그 결과 전체가 F23 -66.54.
  구성 4 (G + TierE 전체 적용):     F24 최고(+46.62)지만 F23 R -7.24 로 실패.

가설: TierE 는 R 에서만 유효하다. F 는 체제 단절 집단이고 TrackMan 프로파일이
      F 의 운영 기준 변화를 설명하지 못한다. 따라서 dispatch 로 분리한다.

구성
  5  R행 -> BASE+TierE (전체 학습) / F행 -> BASE
  6  R행 -> BASE+TierE (R만 학습)  / F행 -> BASE          <- G 결합 + F 보호
  7  R행 -> BASE+TierE (R만 학습)  / F행 -> BASE (R제외 학습 없음, 전체 학습)
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
for c in ["top_bottom", "base_state", "game_type"]:
    df[c] = df[c].astype("category").cat.codes
BASE = sit + asof
y = df[T.TARGET].values.astype(np.float64)
IS_F = gt == "F"
seasons = sorted(df.season.unique())


def variant_f(FOLD):
    emb = pd.read_parquet(f"{T.TE}/tier_e_cutoff{FOLD}.parquet")
    cw = T.soft_crosswalk(fp_main, tm, FOLD, tau=0.05, s_min=0.60)
    cw = cw[cw.evidence_season >= 2022]
    prof = T.build_asof(cw, emb, seasons, half_life=2.0)
    dims = [c for c in prof.columns if c.startswith("te_svd_")][:4]
    keep = dims + ["te_seasons", "te_season_gap", "te_available"]
    d = df.merge(prof[["pitcher_id", "season"] + keep], on=["pitcher_id", "season"], how="left")
    d["te_available"] = d.te_available.fillna(0)
    return d, keep


def fit(d, cols, mtr, mpr):
    dtr = xgb.DMatrix(d.loc[mtr, cols], label=y[mtr], missing=np.nan)
    dpr = xgb.DMatrix(d.loc[mpr, cols], missing=np.nan)
    b = xgb.train(T.PARAMS, dtr, num_boost_round=ROUNDS, verbose_eval=False)
    return b.predict(dpr)


rows = []
for FOLD in [2023, 2024]:
    d, TEC = variant_f(FOLD)
    tr, va = (d.season < FOLD).values, (d.season == FOLD).values
    yv, fv = y[va], IS_F[va]
    print(f"\n===== fold {FOLD}", flush=True)

    p_base = fit(d, BASE, tr, va)
    p_te_all = fit(d, BASE + TEC, tr, va)
    p_te_R = fit(d, BASE + TEC, tr & ~IS_F, va)
    p_base_R = fit(d, BASE, tr & ~IS_F, va)

    P = {"1 BASE": p_base, "2 BASE+TierE (전체)": p_te_all}
    q = p_base.copy(); q[~fv] = p_te_all[~fv]; P["5 R:TierE(전체학습) / F:BASE"] = q
    q = p_base.copy(); q[~fv] = p_te_R[~fv];   P["6 R:TierE(R학습) / F:BASE"] = q
    q = p_base.copy(); q[~fv] = p_base_R[~fv]; P["3 G(R전용) / F:BASE"] = q

    for name, p in P.items():
        rec = dict(fold=FOLD, config=name, bss=float(bss(p, yv)),
                   auc=float(roc_auc_score(yv, p)),
                   bss_R=float(bss(p[~fv], yv[~fv])), bss_F=float(bss(p[fv], yv[fv])))
        rows.append(rec)
        print(f"  {name:30s} BSS={rec['bss']:9.2f}  AUC={rec['auc']:.5f}  "
              f"R={rec['bss_R']:9.2f}  F={rec['bss_F']:10.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/tier_e_final.csv", index=False)
print("\n" + "=" * 100)
print("게이트: R-only 양 fold 개선 + 전체 양 fold 개선 + F24 AUC 상승")
J = {2022: 0.10, 2023: 0.25, 2024: 0.65}
for cfg in res.config.unique():
    if cfg == "1 BASE":
        continue
    o, tot = [], []
    for f in [2023, 2024]:
        b = res[(res.fold == f) & (res.config == "1 BASE")].iloc[0]
        v = res[(res.fold == f) & (res.config == cfg)].iloc[0]
        o.append(v.bss_R - b.bss_R); tot.append(v.bss - b.bss)
    dA = (res[(res.fold == 2024) & (res.config == cfg)].auc.iloc[0]
          - res[(res.fold == 2024) & (res.config == "1 BASE")].auc.iloc[0])
    wj = 0.25 * tot[0] + 0.65 * tot[1]
    ok = (o[0] > 0 and o[1] > 0 and tot[0] > 0 and tot[1] > 0)
    print(f"  {cfg:30s} R: {o[0]:+8.2f} / {o[1]:+8.2f}   전체: {tot[0]:+8.2f} / {tot[1]:+8.2f}"
          f"   가중J {wj:+8.2f}   dAUC {dA:+.5f}  -> {'통과' if ok else '실패'}")
