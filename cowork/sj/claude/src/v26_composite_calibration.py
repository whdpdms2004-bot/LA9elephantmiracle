"""V26: 합성식 계수를 OOF 에서 적합한다.

현재 가정
    P(success) = 1 - (1*p_m + 1*p_r - 1*p_mr + 1*p_ob + 1*p_oz)

    계수 (1,1,-1,1,1) 은 다섯 성분이 완벽히 보정돼 있다고 가정한다.
    V5 에서 M or R or O == 실패 가 정확히 성립함을 확인했으므로 '참 확률'에
    대해서는 맞는 식이다. 그러나 '추정 확률'에는 각자의 편향이 있고, 다섯
    모델의 오차가 선형 결합될 때 자동으로 상쇄되지 않는다.
    기저율이 10배 차이(r 0.229 vs oz 0.023)라 희소 성분일수록 편향이 크다.

제안
    P = 1 - (c_m*p_m + c_r*p_r + c_mr*p_mr + c_ob*p_ob + c_oz*p_oz) - c0
    계수를 순방향 체인 OOF 에서 최소제곱으로 적합한다. 검증 시즌은 쓰지 않는다.

    구분선 기준: '재표현'이 아니라 지금까지 검증한 적 없는 가정을 푸는 것이다.
    다만 계수 6개짜리 사후 보정이므로 과적합 위험이 있어 OOF 적합이 필수다.

arm
    K0  고정 계수 (현행)
    K1  OOF 적합 계수 (제약 없음)
    K2  OOF 적합 + 부호 제약 (c_mr <= 0, 나머지 >= 0)
    K3  전역 절편만 (c0 만 적합, 계수는 고정)   <- 레벨 보정만의 효과 분리

OOF 구성
    2022, 2023 을 순방향으로 만든다 (각각 그 이전 시즌으로 학습).
    2024 는 최종 게이트라 계수 적합에 쓰지 않는다.

판정: Val2024 전체 BSS, 프로덕션 836.503 대비, w=0.25.
출력: outputs/v26_composite_calibration.csv
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

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W = 400, 0.25
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
SIGN = np.array([1.0, 1.0, -1.0, 1.0, 1.0])      # 현행 고정 계수
OOF_FOLDS = [2022, 2023]
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}


def extrap(a, tr):
    m = tr & ~np.isnan(a)
    s = pd.Series(a[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def component_preds(vs):
    """vs 시즌에 대한 5성분 예측. vs 이전 시즌만으로 학습한다."""
    tr, va = season < vs, season == vs
    td = df.loc[tr]
    spec = CF.make_spec(td)
    pl = CF.make_platoon_table(td)
    bt = CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()})
    cp = CF.make_count_platoon_table(td)
    ip = CF.make_inning_platoon_table(td)
    X = CF.build(df[INPUT_COLS], spec, pl, bt, cp, ip).to_numpy(np.float32)
    out = np.empty((int(va.sum()), len(COMPONENTS)))
    for j, tag in enumerate(COMPONENTS):
        arr = LAB[tag]
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr),
               **params_for(float(np.nanmean(arr[tr])))}
        d_tr = xgb.DMatrix(X[m], label=arr[m]); d_va = xgb.DMatrix(X[va])
        p_tr, p_va = Pool(X[m], arr[m]), Pool(X[va])
        acc = np.zeros(int(va.sum()))
        for s_ in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s_}, d_tr,
                                   num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s_, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(p_va)[:, 1]
        out[:, j] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return out


t0 = time.time()
P, Y = {}, {}
for vs in OOF_FOLDS + [2024]:
    P[vs] = component_preds(vs)
    Y[vs] = y_all[season == vs]
    comp = np.clip(1 - P[vs] @ SIGN, EPS, 1 - EPS)
    print(f"  fold {vs} 성분 예측 완료  고정계수 합성 BSS "
          f"{metrics(Y[vs], comp)['bss_raw']:8.2f}   [{time.time()-t0:.0f}s]", flush=True)

# ------------------------------------------------- OOF 계수 적합 (2022+2023)
Xo = np.vstack([P[v] for v in OOF_FOLDS])
yo_ = np.concatenate([Y[v] for v in OOF_FOLDS])
A = np.hstack([Xo, np.ones((len(Xo), 1))])
target = 1.0 - yo_                       # P(fail) 을 맞춘다
c_free, *_ = np.linalg.lstsq(A, target, rcond=None)
c_sign = c_free.copy()
c_sign[:5] = np.where(SIGN > 0, np.maximum(c_sign[:5], 0.0),
                      np.minimum(c_sign[:5], 0.0))
c0_only = np.append(SIGN, float(np.mean(target - Xo @ SIGN)))

print("\nOOF(2022+2023) 적합 계수")
names = COMPONENTS + ["절편"]
print(f"  {'':<8}" + "".join(f"{n:>9}" for n in names))
print(f"  {'고정':<8}" + "".join(f"{v:>9.4f}" for v in np.append(SIGN, 0.0)))
print(f"  {'자유':<8}" + "".join(f"{v:>9.4f}" for v in c_free))
print(f"  {'부호제약':<8}" + "".join(f"{v:>9.4f}" for v in c_sign))
print(f"  {'절편만':<8}" + "".join(f"{v:>9.4f}" for v in c0_only), flush=True)

prod = pd.read_parquet(PROD).set_index("row_id").reindex(
    df.loc[season == 2024, "row_id"].to_numpy())
y_va = Y[2024]
gt = prod["game_type"].astype(str).to_numpy()
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())

rows = []
print(f"\n{'arm':<14}{'합성단독':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}{'pred_mean':>11}")
for name, c in [("K0_fixed", np.append(SIGN, 0.0)), ("K1_free", c_free),
                ("K2_signed", c_sign), ("K3_intercept", c0_only)]:
    ie = np.clip(1 - (P[2024] @ c[:5] + c[5]), EPS, 1 - EPS)
    solo = metrics(y_va, ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(p_prod / (1 - p_prod)),
                             np.log(ie / (1 - ie)))[0, 1])
    q = np.clip(W * ie + (1 - W) * p_prod, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - bm["bss_raw"]
    dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "solo_bss": solo, "corr": corr, "bss": mm["bss_raw"],
                 "dbss": d, "se_row": se, "t_row": d / se,
                 "pred_mean": mm["pred_mean"],
                 **{f"c_{k}": v for k, v in zip(names, c)}})
    print(f"{name:<14}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}{d/se:>8.2f}"
          f"{mm['pred_mean']:>11.5f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v26_composite_calibration.csv", index=False)
ref = res[res.arm == "K0_fixed"]["dbss"].iloc[0]
best = res.sort_values("dbss", ascending=False).iloc[0]
print(f"\n기준선 K0 {ref:+.3f}   최고 {best.arm} {best.dbss:+.3f}  "
      f"차이 {best.dbss-ref:+.3f}")
print(f"실제 2024 평균 {y_va.mean():.6f}")
print(f"\nsaved -> {OUT/'v26_composite_calibration.csv'}")
