"""V65: 내 피처로 '성공률을 직접' 예측해 앙상블 멤버로 넣는다.

여기까지의 지형
    Public 979 (submit_032), 목표 1200. 내부 레버는 전부 +1~3 이다.
    V61 이 확정: 내부 +-2.5 는 Public 에 대해 정보가 없다.
    레벨 레버는 기대값 10점대다 (추종률 31~85%, 드리프트 감속 중).
    Public 979 의 대부분은 base(프로덕션 스택, Val2024 836.5)에서 나온다.

아직 안 해본 것
    내 111피처(계층 차감 플래툰 4축 포함)는 프로덕션 211피처에 없다.
    그런데 그 피처를 '성분 분해' 경로로만 썼다.
    같은 피처로 control_success 를 직접 예측한 모델을 앙상블 멤버로 넣어본 적이 없다.

    성분 라인이 단독 755 를 내는 피처다. 직접 모델은 보통 그보다 높게 나오고
    (분해 없이 목표를 바로 맞추므로), base 와 잔차가 다르면 결합에서 값을 한다.

    프로덕션이 기록한 실패는 '군집/matchup 을 GBDT 원시 피처로' 넣은 것이었다.
    여기서는 그런 표현이 아니라 '별도 모델의 예측'을 멤버로 넣는다 —
    02_EMBEDDING_METHODS.md 의 M3(residual expert)에 해당하고, 팀이 공동 SVD 로
    상관 0.09 를 얻어 이득을 본 전례가 있는 방식이다.

arm
    Z0  base 단독
    Z1  base + 성분 라인            (= 현행 submit_032 구성)
    Z2  base + 직접 모델
    Z3  base + 성분 라인 + 직접 모델  (가중치는 세 fold 규칙으로 고정 격자)

    직접 모델은 성분 모델과 같은 설정(8시드 x XGB+CatBoost, 400라운드)으로
    control_success 를 학습한다. base_score 는 forecast_base_rate 외삽.

판정: 두 fold 모두에서 Z2/Z3 가 Z1 이상이어야 의미가 있다.
      V61 기준 내부 +3 미만은 제출본 교체 근거로 쓰지 않는다.
출력: outputs/v65_direct_member.csv
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
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, metrics,
                     forecast_base_rate)

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
WD = [0.10, 0.15, 0.20, 0.25, 0.30]
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


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def features(fold):
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
    return F.to_numpy(np.float32)


def fit_target(X, fold, arr, base_score, prm_extra):
    tr, va = season < fold, season == fold
    mm = tr & ~np.isnan(arr)
    prm = {**BASE_PARAMS, "base_score": base_score, **prm_extra}
    d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=row_w[mm])
    d_va = xgb.DMatrix(X[va])
    p_tr = Pool(X[mm], arr[mm], weight=row_w[mm])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                               verbose_eval=False).predict(d_va)
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        acc += 0.5 * c.predict_proba(X[va])[:, 1]
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


def extrap(a, tr):
    m_ = tr & ~np.isnan(a)
    s = pd.Series(a[m_]).groupby(pd.Series(season[m_])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


t0, rows = time.time(), []
for fold in FOLDS:
    tr, va = season < fold, season == fold
    X = features(fold)
    y, b = y_all[va], BASE_P[fold]
    null = y.mean() * (1 - y.mean())
    ref = metrics(y, b)["bss_raw"]
    wv = BW[bucket_all[va]]

    p = {t: fit_target(X, fold, LAB[t], extrap(LAB[t], tr), params_for(
        float(np.nanmean(LAB[t][tr])))) for t in COMPONENTS}
    comp = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
    direct = fit_target(X, fold, y_all, forecast_base_rate(df, tr, fold),
                        {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0})
    np.save(CACHE / f"v65_direct_{fold}.npy", direct)
    np.save(CACHE / f"v65_comp_{fold}.npy", comp)

    lg = lambda z: np.log(z / (1 - z))
    print(f"{chr(10)}fold {fold}   base {ref:9.2f}   성분단독 "
          f"{metrics(y, comp)['bss_raw']:9.2f}   직접단독 "
          f"{metrics(y, direct)['bss_raw']:9.2f}")
    print(f"  상관(logit)  base x 성분 {np.corrcoef(lg(b), lg(comp))[0,1]:.4f}   "
          f"base x 직접 {np.corrcoef(lg(b), lg(direct))[0,1]:.4f}   "
          f"성분 x 직접 {np.corrcoef(lg(comp), lg(direct))[0,1]:.4f}", flush=True)

    def rec(name, q):
        d = metrics(y, q)["bss_raw"] - ref
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": fold, "arm": name, "bss": metrics(y, q)["bss_raw"],
                     "dbss": d, "t_row": d / se})
        return d, d / se

    d, t = rec("Z0_base", b)
    print(f"  {'Z0 base':<26}{d:>+9.2f}{t:>8.2f}")
    q = np.clip(wv * comp + (1 - wv) * b, EPS, 1 - EPS)
    d, t = rec("Z1_comp", q)
    print(f"  {'Z1 base+성분':<26}{d:>+9.2f}{t:>8.2f}")
    for wdd in WD:
        q = np.clip(wdd * direct + (1 - wdd) * b, EPS, 1 - EPS)
        d, t = rec(f"Z2_direct_w{wdd:.2f}", q)
        print(f"  {'Z2 base+직접 w'+f'{wdd:.2f}':<26}{d:>+9.2f}{t:>8.2f}")
    for wdd in WD:
        q = np.clip(wv * comp + wdd * direct + (1 - wv - wdd) * b, EPS, 1 - EPS)
        d, t = rec(f"Z3_both_w{wdd:.2f}", q)
        print(f"  {'Z3 base+성분+직접 w'+f'{wdd:.2f}':<26}{d:>+9.2f}{t:>8.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v65_direct_member.csv", index=False)
piv = res.pivot_table(index="arm", columns="fold", values="dbss")
piv["최악"] = piv.min(axis=1)
print(f"{chr(10)}{'='*62}{chr(10)}fold 별 ΔBSS (base 대비){chr(10)}{'='*62}")
print(piv.round(2).sort_values("최악", ascending=False).to_string())
z1 = piv.loc["Z1_comp"]
print(f"{chr(10)}현행 Z1 대비 개선된 arm 만 의미가 있다 "
      f"(2023 {z1[2023]:+.2f} / 2024 {z1[2024]:+.2f}).")
print(f"{chr(10)}saved -> {OUT/'v65_direct_member.csv'}")
