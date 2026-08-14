"""Tier E 평가: soft crosswalk 로 main 에 결합 후 F23/F24 에서 BASE 대비 측정.

crosswalk: 팀의 상황 지문(fingerprint) 코사인 방식을 재사용하되 hard 1:1 대신
           상위 k 후보의 softmax 가중 평균(soft)으로 프로파일을 만든다.
           동반 신뢰도: cw_top1_sim / cw_margin / cw_entropy / cw_eff_cand

as-of: 시즌 S 의 main 행은 trackman 시즌 < S 만 사용. recency half-life 2 시즌 가중.
투입: SVD 잔차 좌표 (절대 확률 아님) -> 04_PREPROCESSING_SPEC §3-4 하드 규칙 준수.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import roc_auc_score

DATA = "/mnt/user-data/uploads/LGAIMERS/data"
TE = "/home/claude/work/outputs/tier_e"
OUT = "/home/claude/work/outputs"
TARGET = "control_success"

MAIN_FP = ["season", "pitcher_id", "pitcher_hand", "game_month", "game_dayofweek",
           "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
           "batter_hand"]
TM_FP = ["season", "pitcher_trackman_id", "pitcher_hand", "game_month", "game_dayofweek",
         "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
         "batter_hand"]


def state_code(f, is_main):
    month = f.game_month.to_numpy(np.int64) - 1
    day = f.game_dayofweek.to_numpy(np.int64)
    inning = np.minimum(f.inning.to_numpy(np.int64), 20)
    if is_main:
        bottom = f.top_bottom.astype(str).eq("B").to_numpy(np.int64)
        bright = (f.batter_hand.to_numpy(np.int64) == 2).astype(np.int64)
    else:
        bottom = f.top_bottom.astype(str).eq("Bottom").to_numpy(np.int64)
        bright = f.batter_hand.astype(str).eq("Right").to_numpy(np.int64)
    code = month.copy()
    for v, b in [(day, 7), (inning, 21), (bottom, 2),
                 (f.balls_before.to_numpy(np.int64), 4),
                 (f.strikes_before.to_numpy(np.int64), 3),
                 (f.outs_before.to_numpy(np.int64), 3), (bright, 2)]:
        code = code * b + v
    return code


def nrm(M):
    n = np.sqrt(M.multiply(M).sum(axis=1)).A1.clip(1e-9)
    return sparse.diags(1.0 / n) @ M


def soft_crosswalk(main, tm, cutoff, topk=5, tau=0.05, s_min=0.60):
    """반환: (pitcher_id, evidence_season, pitcher_trackman_id, w) long table."""
    a = main[main.season < cutoff][MAIN_FP].copy()
    b = tm[tm.season < cutoff][TM_FP].copy()
    a["st"] = state_code(a, True)
    b["st"] = state_code(b, False)
    recs = []
    for season in np.intersect1d(a.season.unique(), b.season.unique()):
        A, B = a[a.season.eq(season)], b[b.season.eq(season)]
        mid = np.sort(A.pitcher_id.unique())
        tid = np.sort(B.pitcher_trackman_id.unique())
        if len(mid) == 0 or len(tid) < 2:
            continue
        mi = {v: i for i, v in enumerate(mid)}
        ti = {v: i for i, v in enumerate(tid)}
        states = np.union1d(A.st.unique(), B.st.unique())
        si = {v: i for i, v in enumerate(states)}
        ag = A.groupby(["pitcher_id", "st"], sort=False).size().reset_index(name="n")
        bg = B.groupby(["pitcher_trackman_id", "st"], sort=False).size().reset_index(name="n")
        MA = sparse.csr_matrix((ag.n, (ag.pitcher_id.map(mi), ag.st.map(si))),
                              shape=(len(mid), len(states)), dtype=np.float32)
        MB = sparse.csr_matrix((bg.n, (bg.pitcher_trackman_id.map(ti), bg.st.map(si))),
                              shape=(len(tid), len(states)), dtype=np.float32)
        S = (nrm(MA) @ nrm(MB).T).toarray()
        mh = A.groupby("pitcher_id").pitcher_hand.first().reindex(mid).to_numpy()
        th = np.where(B.groupby("pitcher_trackman_id").pitcher_hand.first()
                      .reindex(tid).to_numpy() == "Left", 1, 2)
        S[mh[:, None] != th[None, :]] = -1.0
        k = min(topk, S.shape[1])
        top = np.argsort(S, axis=1)[:, -k:][:, ::-1]
        sim = np.take_along_axis(S, top, axis=1)
        w = np.exp((sim - sim[:, :1]) / tau)
        w[sim < s_min] = 0.0
        rs = w.sum(1, keepdims=True)
        keep = rs[:, 0] > 0
        w = np.divide(w, rs, out=np.zeros_like(w), where=rs > 0)
        ent = -(np.where(w > 0, w * np.log(w.clip(1e-12)), 0)).sum(1)
        for r in np.flatnonzero(keep):
            for j in range(k):
                if w[r, j] > 1e-4:
                    recs.append((int(season), int(mid[r]), int(tid[top[r, j]]),
                                 float(w[r, j]), float(sim[r, 0]),
                                 float(sim[r, 0] - sim[r, 1]), float(ent[r])))
    if recs:
        _s = np.array([r[4] for r in recs])
        print(f"    [cw] top1_sim: p10={np.percentile(_s,10):.3f} "
              f"median={np.median(_s):.3f} p90={np.percentile(_s,90):.3f}", flush=True)
    return pd.DataFrame(recs, columns=["evidence_season", "pitcher_id",
                                       "pitcher_trackman_id", "w", "cw_top1_sim",
                                       "cw_margin", "cw_entropy"])


def build_asof(cw, emb, seasons, half_life=2.0):
    """시즌 S 의 main 행용 프로파일: trackman 시즌 < S, recency 가중."""
    dims = [c for c in emb.columns if c.startswith("te_svd_")]
    # (pitcher_id, evidence_season) 단위로 soft 가중 평균 -> main 좌표계
    m = cw.merge(emb, left_on=["pitcher_trackman_id", "evidence_season"],
                 right_on=["pitcher_trackman_id", "season"], how="inner")
    if m.empty:
        return None
    for c in dims:
        m[c] = m[c] * m.w
    agg = m.groupby(["pitcher_id", "evidence_season"], as_index=False).agg(
        **{c: (c, "sum") for c in dims},
        wsum=("w", "sum"), cw_top1_sim=("cw_top1_sim", "first"),
        cw_margin=("cw_margin", "first"), cw_entropy=("cw_entropy", "first"),
        te_n_cells=("te_n_cells", "mean"))
    for c in dims:
        agg[c] = agg[c] / agg.wsum.clip(lower=1e-9)

    rows = []
    for S in seasons:
        past = agg[agg.evidence_season < S].copy()
        if past.empty:
            continue
        rw = np.power(0.5, (S - past.evidence_season) / half_life)
        past["_rw"] = rw
        for c in dims:
            past[c] = past[c] * past._rw
        g = past.groupby("pitcher_id", as_index=False).agg(
            **{c: (c, "sum") for c in dims}, rw=("_rw", "sum"),
            te_seasons=("evidence_season", "nunique"),
            te_last_season=("evidence_season", "max"),
            cw_top1_sim=("cw_top1_sim", "mean"), cw_margin=("cw_margin", "mean"),
            cw_entropy=("cw_entropy", "mean"), te_n_cells=("te_n_cells", "mean"))
        for c in dims:
            g[c] = g[c] / g.rw.clip(lower=1e-9)
        g["season"] = S
        g["te_season_gap"] = S - g.te_last_season
        g["te_available"] = 1
        rows.append(g.drop(columns=["rw", "te_last_season"]))
    return pd.concat(rows, ignore_index=True) if rows else None


def bss(p, y):
    r = y.mean()
    return (1 - np.mean((p - y) ** 2) / (r * (1 - r))) * 1e5


PARAMS = dict(max_depth=0, grow_policy="lossguide", max_leaves=18, eta=0.03,
              subsample=0.8, colsample_bytree=0.6, min_child_weight=64, reg_lambda=2.0,
              objective="binary:logistic", eval_metric="logloss", tree_method="hist",
              nthread=2, seed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="2023,2024")
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--smin", type=float, default=0.60)
    args = ap.parse_args()

    df = pd.read_csv(f"{DATA}/train.csv")
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", usecols=sorted(set(TM_FP)))
    asof = [c for c in df.columns if c.startswith("asof_")]
    ids = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
    sit = [c for c in df.columns if c not in asof + ids + ["row_id", TARGET]]
    gt = df.game_type.astype(str).values
    # crosswalk 지문은 원본 문자열(top_bottom = 'T'/'B')이 필요하다.
    # 아래 인코딩을 하면 'T'/'B' 가 0/1 코드로 바뀌어 지문의 초/말 비트가 죽는다.
    fp_main = df[MAIN_FP].copy()
    for c in ["top_bottom", "base_state", "game_type"]:
        df[c] = df[c].astype("category").cat.codes
    BASE = sit + asof
    y = df[TARGET].values.astype(np.float64)
    IS_F = gt == "F"

    rows = []
    for FOLD in [int(x) for x in args.folds.split(",")]:
        t0 = time.time()
        emb = pd.read_parquet(f"{TE}/tier_e_cutoff{FOLD}.parquet")
        cw = soft_crosswalk(fp_main, tm, FOLD, tau=args.tau, s_min=args.smin)
        npit = cw.pitcher_id.nunique()
        prof = build_asof(cw, emb, sorted(df.season.unique()))
        d = df.merge(prof, on=["pitcher_id", "season"], how="left")
        d["te_available"] = d.te_available.fillna(0)
        TEC = [c for c in prof.columns if c not in ("pitcher_id", "season")]
        tr, va = (d.season < FOLD).values, (d.season == FOLD).values
        cov = float(d.loc[va, "te_available"].mean())
        print(f"\n=== fold {FOLD}: crosswalk 투수 {npit} / {df.pitcher_id.nunique()} "
              f"({npit/df.pitcher_id.nunique():.1%})  검증행 커버리지 {cov:.1%}  "
              f"TierE 열 {len(TEC)}  ({time.time()-t0:.0f}s)", flush=True)

        for name, cols in [("BASE", BASE), ("BASE+TierE", BASE + TEC)]:
            dtr = xgb.DMatrix(d.loc[tr, cols], label=y[tr], missing=np.nan)
            dva = xgb.DMatrix(d.loc[va, cols], missing=np.nan)
            b = xgb.train(PARAMS, dtr, num_boost_round=args.rounds, verbose_eval=False)
            p = b.predict(dva)
            yv, fv = y[va], IS_F[va]
            rec = dict(fold=FOLD, config=name, n_features=len(cols),
                       bss=float(bss(p, yv)), auc=float(roc_auc_score(yv, p)),
                       bss_R=float(bss(p[~fv], yv[~fv])), bss_F=float(bss(p[fv], yv[fv])),
                       brier=float(np.mean((p - yv) ** 2)), pred_mean=float(p.mean()),
                       coverage=cov, cw_pitchers=npit)
            rows.append(rec)
            print(f"  {name:12s} BSS={rec['bss']:9.2f}  AUC={rec['auc']:.5f}  "
                  f"R={rec['bss_R']:9.2f}  F={rec['bss_F']:10.2f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/tier_e_eval.csv", index=False)
    print()
    piv = res.pivot_table(index="config", columns="fold", values=["bss", "auc", "bss_R"])
    print(piv.round(5).to_string())
    for f in res.fold.unique():
        s = res[res.fold.eq(f)].set_index("config")
        print(f"  fold {f}: ΔBSS={s.loc['BASE+TierE','bss']-s.loc['BASE','bss']:+.2f}  "
              f"ΔAUC={s.loc['BASE+TierE','auc']-s.loc['BASE','auc']:+.5f}  "
              f"ΔBSS_R={s.loc['BASE+TierE','bss_R']-s.loc['BASE','bss_R']:+.2f}")


if __name__ == "__main__":
    main()
