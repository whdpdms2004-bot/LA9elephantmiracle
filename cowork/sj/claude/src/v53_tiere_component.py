"""V53: Tier E 조건부 반응 프로파일을 성분 라인에 넣는다.

왜 지금인가
    09_TIER_E_RESULTS.md 는 Tier E 가 게이트를 통과했다고 기록한다.
        채택 구성 R행->TierE / F행->BASE
        R-only F23 +1.16 / F24 +19.58,  전체 F23 +1.04 / F24 +17.27
        F24 dAUC +0.00073   <- 이 프로젝트에서 AUC 가 움직인 첫 사례
        (기존 946개 fold-2024 실험의 AUC 는 0.5329~0.5501 에 갇혀 있었고
         BSS 774->815 동안 순증이 0 이었다)

    그런데 submit_032 의 feature_sets 211개를 스캔하니 te/u_/p_ 계열이 0개다.
    검증만 되고 어떤 제출본에도 통합된 적이 없다. 그 문서의 '다음 단계 1순위'인
    '211피처 실제 파이프라인에서 재검증'이 미실행 상태다.

    Phase-0 P0-4 는 남은 헤드룸을 (투수 x 시즌 x 타자손) 조건부 하나로 지목했고
    (oracle 상한 Val2024 917.7) Tier E 가 그 축을 TrackMan 에서 만드는 유일한 산출물이다.

여기서 유리한 점
    그 문서가 걱정한 것은 '기존 tm500_* 72개와의 중복'이었다.
    그런데 성분 라인에는 TrackMan 피처가 0개다. 중복이 아예 없다.
    프로덕션보다 성분 라인에서 더 살아날 여지가 있다.

붙이는 방법
    crosswalk (cutoff 2024, 하드 1:1, 투수 295명)로 pitcher_id <-> pitcher_trackman_id
    tier_e_cutoff2024.parquet 의 te_svd_00..11 을 (trackman_id, season) 에서 가져와
    시즌 S 행에는 evidence season < S 만, recency half-life 2 시즌 가중으로 평균.

    규칙 N1  TrackMan 증거 시즌을 2022 이후로 제한 (단절 이전은 섞지 않는다)
    규칙 N3  TrackMan 유래 피처를 game_type=F 행에 적용하지 않는다
    규칙 N2  crosswalk 신뢰도 열은 GBDT 에 넣지 않는다 (커버리지 대리변수)

arm
    E0  현행
    E1  + te_svd 12차원 (전 행)
    E2  + te_svd, F행은 NaN            <- 규칙 N3 준수, 문서의 채택 구성
    E3  E2 + te_available 플래그

출력: outputs/v53_tiere_component.csv
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

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
TE_DIR = SJ / "claude" / "outputs" / "tier_e"
CW_DIR = SJ / "experiment" / "pitcher_embedding" / "outputs" / "trackman500"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W, F_WEIGHT, K = 400, 0.25, 0.20, 300
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
VS, MIN_EV_SEASON, HALF_LIFE = 2024, 2022, 2.0
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
tr, va = season < VS, season == VS

pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
gtype = df["game_type"].astype(str).to_numpy()
IS_F = gtype == "F"
row_w = np.where(IS_F, F_WEIGHT, 1.0)
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])

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

# ---------------------------------------------- Tier E as-of 프로파일
cw = pd.read_parquet(CW_DIR / f"cutoff_{VS}" / "crosswalk.parquet")[
    ["pitcher_id", "pitcher_trackman_id"]]
emb = pd.read_parquet(TE_DIR / f"tier_e_cutoff{VS}.parquet")
DIMS = [c for c in emb.columns if c.startswith("te_svd_")]
emb = emb[emb["season"] >= MIN_EV_SEASON]           # 규칙 N1
m = cw.merge(emb, on="pitcher_trackman_id", how="inner").rename(
    columns={"season": "ev"})
print(f"crosswalk 투수 {cw.pitcher_id.nunique()} / {df.pitcher_id.nunique()}   "
      f"Tier E 차원 {len(DIMS)}   증거 시즌 {sorted(m.ev.unique())}")

rows = []
for S in sorted(df["season"].unique()):
    past = m[m.ev < S]
    if past.empty:
        continue
    w = np.power(0.5, (S - past["ev"].to_numpy()) / HALF_LIFE)
    g = past.assign(_w=w, **{c: past[c].to_numpy() * w for c in DIMS}) \
            .groupby("pitcher_id", as_index=False).agg(
                _w=("_w", "sum"), **{c: (c, "sum") for c in DIMS})
    for c in DIMS:
        g[c] = g[c] / g["_w"].clip(lower=1e-9)
    g["season"] = S
    rows.append(g.drop(columns="_w"))
prof = pd.concat(rows, ignore_index=True)
prof["te_available"] = 1.0
TE = df[["pitcher_id", "season"]].merge(prof, on=["pitcher_id", "season"], how="left")
TEV = TE[DIMS].to_numpy(np.float64)
AVAIL = TE["te_available"].fillna(0.0).to_numpy()
print(f"행 커버리지  전체 {AVAIL.mean():.1%}   2024 {AVAIL[va].mean():.1%}   "
      f"R행 2024 {AVAIL[va & ~IS_F].mean():.1%}", flush=True)

td = df.loc[tr]
BASE_F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
pidx = pd.MultiIndex.from_arrays([pid, bhand])
for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": ax[tr], "y": y_all[tr]})
    l0 = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
    e2 = (g2["sum"] + K * l0) / (g2["size"] + K)
    e3 = (g3["sum"] + K * l0) / (g3["size"] + K)
    i3 = pd.MultiIndex.from_arrays([pid, bhand, ax])
    v2 = np.where(np.isnan(e2.reindex(pidx).to_numpy()), l0, e2.reindex(pidx).to_numpy())
    v3 = np.where(np.isnan(e3.reindex(i3).to_numpy()), l0, e3.reindex(i3).to_numpy())
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    BASE_F[f"{tag}_split"], BASE_F[f"{tag}_rel"] = v3 - v2, sz / (sz + K)
    BASE_F[f"{tag}_w"] = (v3 - v2) * sz / (sz + K)
print(f"기준 피처 {BASE_F.shape[1]}개 (TrackMan 피처 0개 — 중복 없음)", flush=True)


def extrap(a, mask):
    m_ = mask & ~np.isnan(a)
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


def line(X):
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr),
               **params_for(float(np.nanmean(arr[mm])))}
        Xv = X[va]
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=row_w[mm], missing=np.nan)
        d_va = xgb.DMatrix(Xv, missing=np.nan)
        p_tr = Pool(np.nan_to_num(X[mm], nan=-999.0), arr[mm], weight=row_w[mm])
        Xc = np.nan_to_num(Xv, nan=-999.0)
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


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b, game_type=gt)["bss_raw"]


def build(arm):
    F = BASE_F.copy()
    if arm == "E0":
        return F
    V = TEV.copy()
    if arm in ("E2", "E3"):
        V[IS_F] = np.nan                      # 규칙 N3
    for i, c in enumerate(DIMS):
        F[c] = V[:, i]
    if arm == "E3":
        F["te_available"] = np.where(IS_F, 0.0, AVAIL)
    return F


t0, res_rows = time.time(), []
NAME = {"E0": "현행", "E1": "+TierE 전행", "E2": "+TierE F행제외",
        "E3": "E2+가용플래그"}
print(f"{chr(10)}{'arm':<20}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}"
      f"{'t_row':>8}{'경과':>8}")
for arm in ["E0", "E1", "E2", "E3"]:
    F = build(arm)
    p_ie = line(F.to_numpy(np.float32))
    np.save(CACHE / f"v53_{arm}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(b / (1 - b)), np.log(p_ie / (1 - p_ie)))[0, 1])
    q = np.clip(W * p_ie + (1 - W) * b, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    res_rows.append({"arm": arm, "n_features": F.shape[1], "solo_bss": solo,
                     "corr": corr, "dbss": d, "t_row": d / se,
                     "r_bss": mm.get("r_bss", np.nan), "f_bss": mm.get("f_bss", np.nan)})
    print(f"{arm} {NAME[arm]:<17}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}"
          f"{d:>+9.2f}{d/se:>8.2f}{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(res_rows)
res.to_csv(OUT / "v53_tiere_component.csv", index=False)
r0 = res[res.arm == "E0"].iloc[0]
print(f"{chr(10)}{'='*58}{chr(10)}{'arm':<20}{'ΔBSS 대비':>12}{'단독 대비':>12}")
for _, r in res.iterrows():
    print(f"{r.arm} {NAME[r.arm]:<17}{r.dbss-r0.dbss:>+12.2f}"
          f"{r.solo_bss-r0.solo_bss:>+12.2f}")
print(f"{chr(10)}09_TIER_E_RESULTS.md 는 43피처 BASE 위에서 전체 F24 +17.27 을 냈다.")
print(f"성분 라인은 TrackMan 중복이 없으므로 그 이득이 살아있어야 한다.")
print(f"{chr(10)}saved -> {OUT/'v53_tiere_component.csv'}")
