"""V64: V63 상위 후보를 fold 2023/2024 에서 확인한다.

V63 (45 arm 전체 구성, fold 2024)
    arm                  ΔBSS 대비   단독 대비   제외%
    R_firstseason_drop     +3.33      -1.93     5.28%
    S0.5_w0.5              +1.97      +7.26     0.00%
    V50_drop               +1.66      -0.98     2.76%
    A15_w05                +1.61      +8.90     0.00%
    V50_w05                +1.51      +2.60     0.00%
    ...
    V500_drop              -7.33     -89.53    20.78%
    A20_drop               -2.96      -6.35    19.18%

    체계적 패턴: 제거는 볼륨 비용을 물고 가중치 축소는 안 문다.
    V9 에서 잰 '학습 행 절반 = -4.39' 가 대량 제거 arm 에서 그대로 재현된다.
    V10 에서 F행을 버리는 대신 0.20 을 준 것이 +4.07 이었던 것과 같은 구조다.

    단독 BSS 로는 둘이 앞선다 - A15_w05 +8.90, S0.5_w0.5 +7.26.
    서로 다른 축(절대 짧음 vs 상대적 짧음)이라 합칠 여지가 있는데
    V63 의 조합 arm 은 A10 을 썼고 승자인 A15 와의 조합은 미측정이다.

arm
    Y0  기준선
    Y1  S0.5 w0.5      평소의 절반 미만 등판에 가중치 0.5
    Y2  A15 w0.5       15구 미만 등판에 가중치 0.5
    Y3  Y1 + Y2 (곱)
    Y4  Y1 + Y2 를 각 0.7 로 완화
    Y5  시즌 첫 등판 제외 (V63 최고 ΔBSS 였으나 단독이 음수)

판정: 두 fold 모두에서 단독과 ΔBSS 가 함께 올라야 채택.
      V61 기준으로 내부 +3 미만은 제출본 교체 근거로 쓰지 않는다.
출력: outputs/v64_filter_folds.csv
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
NVOL = df["asof_pitcher_n"].to_numpy()
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


def outing_stats(tr_mask):
    """등판 분할과 '평소 대비' 비율. 중앙값은 학습 시즌만으로 계산한다."""
    o = np.argsort(pid.astype(np.int64) * 10_000_000 + NVOL, kind="stable")
    pv = df["asof_pitcher_prev1_game_success_rate"].to_numpy()[o]
    gp = pid[o]
    chg = np.r_[True, (gp[1:] != gp[:-1])
                | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
    outing = np.empty(len(df), dtype=np.int64)
    outing[o] = np.cumsum(chg) - 1
    od = pd.DataFrame({"outing": outing, "pid": pid, "inn": df["inning"].to_numpy(),
                       "season": season, "tr": tr_mask, "nv": NVOL})
    agg = od.groupby("outing").agg(n=("outing", "size"), pid=("pid", "first"),
                                   first_inn=("inn", "min"),
                                   season=("season", "first"),
                                   ntr=("tr", "sum"), nv=("nv", "min"))
    agg["start"] = (agg["first_inn"] == 1).astype(int)
    med = agg[agg["ntr"] > 0].groupby(["pid", "start"])["n"].median().rename("med")
    agg = agg.join(med, on=["pid", "start"])
    agg["ratio"] = agg["n"] / agg["med"].clip(lower=1)
    agg["first_of_season"] = (agg.sort_values("nv").groupby(["pid", "season"])
                              .cumcount() == 0).astype(int)
    return (np.nan_to_num(agg["ratio"].reindex(outing).to_numpy(), nan=1.0),
            agg["n"].reindex(outing).to_numpy(),
            agg["first_of_season"].reindex(outing).to_numpy() == 1)


def extrap(a, tr_mask, keep):
    m_ = tr_mask & keep & ~np.isnan(a)
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


def line(X, fold, keep, wmul):
    tr, va = season < fold, season == fold
    w = np.where(IS_F, F_WEIGHT, 1.0) * wmul
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & keep & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr, keep),
               **params_for(float(np.nanmean(arr[mm])))}
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=w[mm])
        d_va = xgb.DMatrix(X[va])
        p_tr = Pool(X[mm], arr[mm], weight=w[mm])
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


ALL = np.ones(len(df), bool)
ONE = np.ones(len(df))
t0, rows = time.time(), []
for fold in FOLDS:
    tr, va = season < fold, season == fold
    RATIO, OUT_N, FIRST_S = outing_stats(tr)
    S5, A15 = RATIO < 0.5, OUT_N < 15
    ARMS = [
        ("Y0_baseline", ALL, ONE),
        ("Y1_S05_w05", ALL, np.where(S5, 0.5, 1.0)),
        ("Y2_A15_w05", ALL, np.where(A15, 0.5, 1.0)),
        ("Y3_both_w05", ALL, np.where(S5, 0.5, 1.0) * np.where(A15, 0.5, 1.0)),
        ("Y4_both_w07", ALL, np.where(S5, 0.7, 1.0) * np.where(A15, 0.7, 1.0)),
        ("Y5_firstseason_drop", ~FIRST_S, ONE),
    ]
    X = features(fold)
    y, b = y_all[va], BASE_P[fold]
    null = y.mean() * (1 - y.mean())
    ref = metrics(y, b)["bss_raw"]
    wv = BW[bucket_all[va]]
    print(f"{chr(10)}fold {fold}   base {ref:9.2f}   "
          f"짧은등판 {S5[tr].mean()*100:.2f}%  15구미만 {A15[tr].mean()*100:.2f}%")
    print(f"  {'arm':<22}{'학습행':>10}{'단독':>10}{'ΔBSS':>9}{'t_row':>8}")
    for name, keep, wmul in ARMS:
        p_ie = line(X, fold, keep, wmul)
        np.save(CACHE / f"v64_{name}_{fold}.npy", p_ie)
        solo = metrics(y, p_ie)["bss_raw"]
        q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
        d = metrics(y, q)["bss_raw"] - ref
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": fold, "arm": name, "n_train": int((tr & keep).sum()),
                     "solo_bss": solo, "dbss": d, "t_row": d / se})
        print(f"  {name:<22}{int((tr & keep).sum()):>10,}{solo:>10.2f}{d:>+9.2f}"
              f"{d/se:>8.2f}   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v64_filter_folds.csv", index=False)
piv = res.pivot_table(index="arm", columns="fold", values="dbss")
sol = res.pivot_table(index="arm", columns="fold", values="solo_bss")
print(f"{chr(10)}{'='*66}{chr(10)}기준선 대비{chr(10)}{'='*66}")
print("ΔBSS")
print(piv.subtract(piv.loc["Y0_baseline"], axis=1).round(2).to_string())
print(f"{chr(10)}성분단독")
print(sol.subtract(sol.loc["Y0_baseline"], axis=1).round(2).to_string())
print(f"{chr(10)}두 fold 모두 둘 다 양수여야 채택. V61 기준 내부 +3 미만은 "
      f"제출본 교체 근거로 쓰지 않는다.")
print(f"{chr(10)}saved -> {OUT/'v64_filter_folds.csv'}")
