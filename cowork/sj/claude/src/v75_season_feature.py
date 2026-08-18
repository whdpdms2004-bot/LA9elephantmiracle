"""V75: season 을 피처로 쓰는 것이 옳은가 — 감사(V74 §1-C)가 남긴 미검증 항목.

문제
    학습 season 값 [2019, 2020, 2021, 2022, 2023]
    검증 season 값 [2024]
    검증 시즌 값은 학습에 한 번도 없던 값이다. 트리는 학습 최대값 이상을 한 잎으로
    묶으므로, 모델이 시즌 드리프트를 season 분할로 배웠다면 검증에서는 마지막 시즌
    규칙이 그대로 적용된다.

    base_score 외삽(성분별 2025 기저율)이 이를 보정하는 구조이긴 하다.
    그러나 season 을 빼는 편이 나은지 한 번도 안 재봤다.

    실제 시즌 드리프트가 크다 (계획서 §2-1)
        m  +41.0%   r  +36.7%   ob  -23.6%   oz  -12.2%   (2019 -> 2024)
    이만큼 움직이는데 season 을 그대로 주면 트리가 그 축으로 규칙을 만들 유인이 크다.

arm
    S0  현행 (season 포함)
    S1  season 제거
    S2  season 제거 + 시즌 순번(0..n)으로 대체   <- 순서 정보는 남기되 절대값은 뺀다
    S3  season 제거 + 최근 시즌 recency 가중 0.85

판정: 두 fold 모두 단독과 ΔBSS 가 함께 올라야 채택.
출력: outputs/v75_season_feature.csv
"""
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
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, F_WEIGHT, K = 400, 0.20, 300
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
FOLDS = [2023, 2024]
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
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt3 = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
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


def base_features(fold):
    tr = season < fold
    td = df.loc[tr]
    F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                 CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}),
                 CF.make_count_platoon_table(td), CF.make_inning_platoon_table(td))
    return F


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def line(X, fold, recency):
    tr, va = season < fold, season == fold
    W = np.where(IS_F, F_WEIGHT, 1.0)
    if recency != 1.0:
        W = W * (recency ** (int(season[tr].max()) - season))
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & ~np.isnan(arr)
        s_ = pd.Series(arr[mm]).groupby(pd.Series(season[mm])).mean().sort_index()
        bs = float(np.clip(float(s_.iloc[-1]) + (float(s_.iloc[-1]) - float(s_.iloc[0]))
                           / (float(s_.index[-1]) - float(s_.index[0])), 0.005, 0.995))
        prm = {**BASE_PARAMS, "base_score": bs,
               **params_for(float(np.nanmean(arr[mm])))}
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=W[mm])
        d_va = xgb.DMatrix(X[va])
        p_tr = Pool(X[mm], arr[mm], weight=W[mm])
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(X[va])[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


t0, rows = time.time(), []
lgf = lambda z: np.log(z / (1 - z))
NAME = {"S0": "현행 season 포함", "S1": "season 제거",
        "S2": "season -> 순번", "S3": "season 제거 + recency 0.85"}
for fold in FOLDS:
    va = season == fold
    BF = base_features(fold)
    has = "season" in BF.columns
    y, b = y_all[va], BASE_P[fold]
    null = y.mean() * (1 - y.mean())
    ref = metrics(y, b)["bss_raw"]
    wv = BW[bucket_all[va]]
    print(f"{chr(10)}fold {fold}   base {ref:9.2f}   season 열 존재 {has}")
    print(f"  {'arm':<26}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}")
    for arm in ["S0", "S1", "S2", "S3"]:
        F = BF.copy()
        rec_w = 1.0
        if arm in ("S1", "S3"):
            F = F.drop(columns=["season"])
        if arm == "S2":
            F["season"] = (season - int(season.min())).astype(np.float32)
        if arm == "S3":
            rec_w = 0.85
        p_ie = line(F.to_numpy(np.float32), fold, rec_w)
        np.save(CACHE / f"v75_{arm}_{fold}.npy", p_ie)
        solo = metrics(y, p_ie)["bss_raw"]
        corr = float(np.corrcoef(lgf(b), lgf(p_ie))[0, 1])
        q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
        d = metrics(y, q)["bss_raw"] - ref
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": fold, "arm": arm, "n_features": F.shape[1],
                     "solo_bss": solo, "corr": corr, "dbss": d, "t_row": d / se})
        print(f"  {arm} {NAME[arm]:<23}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}"
              f"{d:>+9.2f}{d/se:>8.2f}   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v75_season_feature.csv", index=False)
piv = res.pivot_table(index="arm", columns="fold", values="dbss")
sol = res.pivot_table(index="arm", columns="fold", values="solo_bss")
print(f"{chr(10)}{'='*58}{chr(10)}S0 대비{chr(10)}{'='*58}")
print("ΔBSS")
print(piv.subtract(piv.loc["S0"], axis=1).round(2).to_string())
print(f"{chr(10)}성분단독")
print(sol.subtract(sol.loc["S0"], axis=1).round(2).to_string())
print(f"{chr(10)}season 은 검증 시즌 값이 학습 범위 밖이라 트리가 외삽할 수 없는 축이다.")
print(f"빼는 쪽이 나으면 전처리 재설계의 기본값을 바꾼다.")
print(f"{chr(10)}saved -> {OUT/'v75_season_feature.csv'}")
