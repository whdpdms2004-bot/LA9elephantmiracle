"""Tier E ablation — F23 악화 원인 진단.

1차 결과
  fold 2024: dBSS +24.21  dAUC +0.00061  dBSS_R +19.15   <- 개선, AUC 최초 상승
  fold 2023: dBSS -50.98  dAUC -0.00094  dBSS_R -25.44   <- 악화, 게이트 실패

가설
  H1 체제 불일치: fold 2023 의 프로파일 증거는 대부분 2022 이전(측정계 단절 전)이다.
                  증거를 2022+ 로 제한하면 개선될 것.
  H2 차원 과다  : dim 12 는 BASE 43 대비 크다. dim 4 로 줄이면 잡음이 줄 것.
  H3 신뢰도 열  : cw_* 4개와 te_available 이 커버리지 대리변수로 과적합할 것.
  H4 최근성     : half_life 2 -> 1 로 최근 시즌을 더 강하게 가중.
"""
import argparse, json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import tier_e_eval as T

DATA = T.DATA
OUT = T.OUT


def bss(p, y):
    r = y.mean()
    return (1 - np.mean((p - y) ** 2) / (r * (1 - r))) * 1e5


def build_asof_filtered(cw, emb, seasons, half_life, min_evidence_season):
    if min_evidence_season is not None:
        cw = cw[cw.evidence_season >= min_evidence_season]
        if cw.empty:
            return None
    return T.build_asof(cw, emb, seasons, half_life=half_life)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=250)
    args = ap.parse_args()

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

    VARIANTS = [
        # name,                      dim,  drop_conf, half_life, min_evidence_season
        ("A dim12 hl2 (1차)",         12,  False, 2.0, None),
        ("B dim4",                     4,  False, 2.0, None),
        ("C dim6 + conf 제거",         6,  True,  2.0, None),
        ("D dim12 hl1",               12,  False, 1.0, None),
        ("E dim12 증거>=2022",        12,  False, 2.0, 2022),
        ("F dim4 증거>=2022 conf제거", 4,  True,  2.0, 2022),
    ]

    rows = []
    for FOLD in [2023, 2024]:
        emb = pd.read_parquet(f"{T.TE}/tier_e_cutoff{FOLD}.parquet")
        cw = T.soft_crosswalk(fp_main, tm, FOLD, tau=0.05, s_min=0.60)
        tr, va = (df.season < FOLD).values, (df.season == FOLD).values
        yv, fv = y[va], IS_F[va]

        dtr = xgb.DMatrix(df.loc[tr, BASE], label=y[tr], missing=np.nan)
        dva = xgb.DMatrix(df.loc[va, BASE], missing=np.nan)
        b = xgb.train(T.PARAMS, dtr, num_boost_round=args.rounds, verbose_eval=False)
        p = b.predict(dva)
        base_rec = dict(fold=FOLD, variant="BASE", bss=float(bss(p, yv)),
                        auc=float(roc_auc_score(yv, p)), bss_R=float(bss(p[~fv], yv[~fv])),
                        bss_F=float(bss(p[fv], yv[fv])), coverage=np.nan, n_te=0)
        rows.append(base_rec)
        print(f"\n=== fold {FOLD}")
        print(f"  {'BASE':30s} BSS={base_rec['bss']:9.2f}  AUC={base_rec['auc']:.5f}  "
              f"R={base_rec['bss_R']:9.2f}", flush=True)

        for name, dim, drop_conf, hl, mes in VARIANTS:
            prof = build_asof_filtered(cw, emb, seasons, hl, mes)
            if prof is None:
                continue
            dims = [c for c in prof.columns if c.startswith("te_svd_")][:dim]
            conf = ["cw_top1_sim", "cw_margin", "cw_entropy", "te_n_cells"]
            keep = dims + ["te_seasons", "te_season_gap", "te_available"]
            if not drop_conf:
                keep += conf
            d = df.merge(prof[["pitcher_id", "season"] + keep], on=["pitcher_id", "season"],
                         how="left")
            d["te_available"] = d.te_available.fillna(0)
            cov = float(d.loc[va, "te_available"].mean())
            cols = BASE + keep
            dtr = xgb.DMatrix(d.loc[tr, cols], label=y[tr], missing=np.nan)
            dva = xgb.DMatrix(d.loc[va, cols], missing=np.nan)
            b = xgb.train(T.PARAMS, dtr, num_boost_round=args.rounds, verbose_eval=False)
            p = b.predict(dva)
            rec = dict(fold=FOLD, variant=name, bss=float(bss(p, yv)),
                       auc=float(roc_auc_score(yv, p)), bss_R=float(bss(p[~fv], yv[~fv])),
                       bss_F=float(bss(p[fv], yv[fv])), coverage=cov, n_te=len(keep))
            rows.append(rec)
            print(f"  {name:30s} BSS={rec['bss']:9.2f} ({rec['bss']-base_rec['bss']:+8.2f})  "
                  f"AUC={rec['auc']:.5f} ({rec['auc']-base_rec['auc']:+.5f})  "
                  f"R={rec['bss_R']:9.2f} ({rec['bss_R']-base_rec['bss_R']:+7.2f})  "
                  f"cov={cov:.1%}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/tier_e_ablation.csv", index=False)
    print("\n" + "=" * 96)
    print("게이트 G1: R-only 가 F23·F24 모두 개선되어야 채택")
    b23 = res[(res.fold == 2023) & (res.variant == "BASE")].iloc[0]
    b24 = res[(res.fold == 2024) & (res.variant == "BASE")].iloc[0]
    for v in res.variant.unique():
        if v == "BASE":
            continue
        r23 = res[(res.fold == 2023) & (res.variant == v)]
        r24 = res[(res.fold == 2024) & (res.variant == v)]
        if r23.empty or r24.empty:
            continue
        d23R = r23.bss_R.iloc[0] - b23.bss_R
        d24R = r24.bss_R.iloc[0] - b24.bss_R
        d23 = r23.bss.iloc[0] - b23.bss
        d24 = r24.bss.iloc[0] - b24.bss
        dA = (r24.auc.iloc[0] - b24.auc)
        ok = "통과" if (d23R > 0 and d24R > 0) else "실패"
        print(f"  {v:30s} R: F23 {d23R:+8.2f} / F24 {d24R:+8.2f}  "
              f"전체: {d23:+8.2f} / {d24:+8.2f}  F24 dAUC={dA:+.5f}  -> {ok}")


if __name__ == "__main__":
    main()
