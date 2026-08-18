"""V66: TrackMan 물리 요약을 성분 모델에 넣는다 — 성공률이 아니라 '실패 유형'을 겨냥해서.

왜 아직도 안 해본 자리인가
    내 성분 모델은 111피처로만 돈다. 프로덕션의 tm500_* 요약 72개가 하나도 없다.
    V53 에서 넣은 것은 Tier E SVD 12차원, 즉 조건부 반응 '잔차'이지 원시 물리 요약이
    아니다. 원시 tm500_* 는 성분 라인에 한 번도 들어간 적이 없다.

왜 될 만한가 — Phase-0 의 결론을 뒤집는 게 아니라 다른 타깃에 적용하는 것이다
    Phase-0 P0-4: (투수, 시즌) oracle 상한 720.9. 팀 단일 모델이 이미 넘었다.
    "TrackMan 요약이 담는 것은 본질적으로 투수-시즌 상수라 여지가 얇다"
    -> 이 결론은 '성공률' 타깃에 대한 것이다.

    그런데 실패의 '종류'는 물리와 강하게 붙는다. V52 유형표가 증거다.
        유형 0  horz_break 23.5  구성_r 0.544  구성_ob 0.208
        유형 3  horz_break 13.7  구성_r 0.392  구성_ob 0.282
    큰 횡변화 투수는 반대 코스로 실패하고 제구형은 볼로 빠진다.
    성공률에 얇은 정보가 유형 분해에는 두꺼울 수 있다. 아무도 안 쟀다.

    그리고 V65 가 증명했다: 성분 분해의 가치는 base 와의 '비상관'에서 온다
    (직접 모델 단독 751.20 > 성분 745.30 인데 결합은 +23.43 < +41.04).
    성분 모델이 물리를 쓰면 base 와 더 달라질 수도, 더 같아질 수도 있다.
    상관을 함께 찍어서 어느 쪽인지 본다.

규칙
    N2  crosswalk 신뢰도 열(cw_*)은 넣지 않는다. 커버리지 대리변수로 작동한다.
    N3  TrackMan 유래 피처를 game_type=F 행에 적용하지 않는다.
    tm500_* 은 이미 strict as-of (시즌 S 행은 season < S 증거만)로 만들어져 있다.
    결측 50.1% 는 XGBoost 가 그대로 받고 CatBoost 는 -999 로 채운다.

arm
    P0  현행 111피처
    P1  + tm500 물리 mean 만 (8지표 x latest/recent = 16)
    P2  + tm500 non-between 전체 (47)
    P3  P2 에서 F행 NaN (규칙 N3)
    P4  P3 + Tier E SVD 12 (submit_033 구성과 합침)

판정: 두 fold 모두 단독과 ΔBSS 가 함께 올라야 한다.
      V61 기준 내부 +3 미만은 제출본 교체 근거로 쓰지 않는다.
출력: outputs/v66_trackman_components.csv
"""
import json
import re
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

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
TM = MO / "trackman500_asof_train.parquet"
TMAN = MO / "trackman500_asof_manifest.json"
TE_DIR = SJ / "claude" / "outputs" / "tier_e"
CW_DIR = SJ / "experiment" / "pitcher_embedding" / "outputs" / "trackman500"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, F_WEIGHT, K = 400, 0.20, 300
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
FOLDS = [2023, 2024]
METRICS8 = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
            "extension", "rel_height", "rel_side", "zone_speed"]
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
IS_F = df["game_type"].astype(str).to_numpy() == "F"
row_w = np.where(IS_F, F_WEIGHT, 1.0)
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])

fc = json.load(open(TMAN, encoding="utf-8"))["feature_columns"]
TM_ALL = [c for c in fc if c.startswith("tm500_") and "between" not in c]   # 규칙 N2
TM_MEAN = [c for c in TM_ALL if c.endswith("_mean")
           and any(m in c for m in METRICS8)]
tmdf = df[["row_id"]].merge(pd.read_parquet(TM, columns=["row_id"] + TM_ALL),
                            on="row_id", how="left")
TMV = tmdf[TM_ALL].to_numpy(np.float64)
COV = np.isfinite(TMV[:, TM_ALL.index("tm500_total_pitches")])
print(f"tm500 피처 {len(TM_ALL)}개 (mean만 {len(TM_MEAN)})   "
      f"행 커버 {COV.mean():.1%}   cw_* 는 규칙 N2 로 제외", flush=True)

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)


def AND(*a):
    m = np.ones(len(df), bool)
    for x in a:
        m &= (x == 1)
    return np.where(ok, m.astype(float), np.nan)


LAB = {"m": ym, "r": yr, "mr": AND(ym, yr), "ob": AND(yo, yb), "oz": AND(yo, 1 - yb)}

models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
BASE_P = {}
for f in FOLDS:
    fid = df.loc[season == f, "row_id"].to_numpy()
    if f == 2024:
        pr = pd.read_parquet(PROD).set_index("row_id").reindex(fid)
        BASE_P[f] = np.clip(pr["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                            EPS, 1 - EPS)
    else:
        acc, c = None, 0
        for mn in models:
            p = OOF_DIR / f"{mn}_fold{f}.parquet"
            if p.exists():
                v = pd.read_parquet(p).set_index("row_id").reindex(fid)["prediction"].to_numpy()
                acc = v if acc is None else acc + v
                c += 1
        BASE_P[f] = np.clip(acc / c, EPS, 1 - EPS)


def tier_e(fold):
    cw = pd.read_parquet(CW_DIR / f"cutoff_{fold}" / "crosswalk.parquet")[
        ["pitcher_id", "pitcher_trackman_id"]]
    emb = pd.read_parquet(TE_DIR / f"tier_e_cutoff{fold}.parquet")
    dims = [c for c in emb.columns if c.startswith("te_svd_")]
    emb = emb[emb["season"] >= 2022]
    link = cw.merge(emb, on="pitcher_trackman_id", how="inner").rename(
        columns={"season": "ev"})
    out = []
    for S in sorted(df["season"].unique()):
        past = link[link["ev"] < S]
        if past.empty:
            continue
        w = np.power(0.5, (S - past["ev"].to_numpy()) / 2.0)
        g = past.assign(_w=w, **{c: past[c].to_numpy() * w for c in dims}) \
                .groupby("pitcher_id", as_index=False).agg(
                    _w=("_w", "sum"), **{c: (c, "sum") for c in dims})
        for c in dims:
            g[c] = g[c] / g["_w"].clip(lower=1e-9)
        out.append(g.drop(columns="_w").assign(season=S))
    prof = pd.concat(out, ignore_index=True)
    TE = df[["pitcher_id", "season"]].merge(prof, on=["pitcher_id", "season"],
                                            how="left")
    return TE[dims].to_numpy(np.float64), dims


def base_features(fold):
    tr = season < fold
    td = df.loc[tr]
    F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                 CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
    pidx = pd.MultiIndex.from_arrays([pid, bhand])
    for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
        d2 = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": ax[tr], "y": y_all[tr]})
        l0 = float(d2["y"].mean())
        g2 = d2.groupby(["p", "h"])["y"].agg(["sum", "size"])
        g3 = d2.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
        e2 = (g2["sum"] + K * l0) / (g2["size"] + K)
        e3 = (g3["sum"] + K * l0) / (g3["size"] + K)
        i3 = pd.MultiIndex.from_arrays([pid, bhand, ax])
        v2 = np.where(np.isnan(e2.reindex(pidx).to_numpy()), l0,
                      e2.reindex(pidx).to_numpy())
        v3 = np.where(np.isnan(e3.reindex(i3).to_numpy()), l0,
                      e3.reindex(i3).to_numpy())
        sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
        F[f"{tag}_split"], F[f"{tag}_rel"] = v3 - v2, sz / (sz + K)
        F[f"{tag}_w"] = (v3 - v2) * sz / (sz + K)
    return F


def extrap(a, tr):
    m_ = tr & ~np.isnan(a)
    s = pd.Series(a[m_]).groupby(pd.Series(season[m_])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def line(X, fold):
    tr, va = season < fold, season == fold
    p = {}
    Xv = X[va]
    Xc = np.nan_to_num(Xv, nan=-999.0)
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr),
               **params_for(float(np.nanmean(arr[mm])))}
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=row_w[mm], missing=np.nan)
        d_va = xgb.DMatrix(Xv, missing=np.nan)
        p_tr = Pool(np.nan_to_num(X[mm], nan=-999.0), arr[mm], weight=row_w[mm])
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(Xc)[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


t0, rows = time.time(), []
lg = lambda z: np.log(z / (1 - z))
for fold in FOLDS:
    va = season == fold
    BF = base_features(fold)
    TEV, tedims = tier_e(fold)
    y, b = y_all[va], BASE_P[fold]
    null = y.mean() * (1 - y.mean())
    ref = metrics(y, b)["bss_raw"]
    wv = BW[bucket_all[va]]
    print(f"{chr(10)}fold {fold}   base {ref:9.2f}")
    print(f"  {'arm':<16}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}")
    for arm in ["P0", "P1", "P2", "P3", "P4"]:
        F = BF.copy()
        cols = {"P1": TM_MEAN, "P2": TM_ALL, "P3": TM_ALL, "P4": TM_ALL}.get(arm)
        if cols:
            V = TMV[:, [TM_ALL.index(c) for c in cols]].copy()
            if arm in ("P3", "P4"):
                V[IS_F] = np.nan                      # 규칙 N3
            for i, c in enumerate(cols):
                F[c] = V[:, i]
        if arm == "P4":
            W = TEV.copy()
            W[IS_F] = np.nan
            for i, c in enumerate(tedims):
                F[c] = W[:, i]
        p_ie = line(F.to_numpy(np.float32), fold)
        np.save(CACHE / f"v66_{arm}_{fold}.npy", p_ie)
        solo = metrics(y, p_ie)["bss_raw"]
        corr = float(np.corrcoef(lg(b), lg(p_ie))[0, 1])
        q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
        d = metrics(y, q)["bss_raw"] - ref
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": fold, "arm": arm, "n_features": F.shape[1],
                     "solo_bss": solo, "corr": corr, "dbss": d, "t_row": d / se})
        print(f"  {arm:<16}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}"
              f"{d/se:>8.2f}   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v66_trackman_components.csv", index=False)
piv = res.pivot_table(index="arm", columns="fold", values="dbss")
sol = res.pivot_table(index="arm", columns="fold", values="solo_bss")
cor = res.pivot_table(index="arm", columns="fold", values="corr")
print(f"{chr(10)}{'='*62}{chr(10)}P0 대비{chr(10)}{'='*62}")
print("ΔBSS")
print(piv.subtract(piv.loc["P0"], axis=1).round(2).to_string())
print(f"{chr(10)}성분단독")
print(sol.subtract(sol.loc["P0"], axis=1).round(2).to_string())
print(f"{chr(10)}base 와의 logit 상관 (낮을수록 결합에 유리 — V65)")
print(cor.round(4).to_string())
print(f"{chr(10)}saved -> {OUT/'v66_trackman_components.csv'}")
