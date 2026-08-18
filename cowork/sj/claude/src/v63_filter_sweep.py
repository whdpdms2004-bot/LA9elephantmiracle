"""V63: 학습 표본 선택 조합 스윕 (전체 구성 일괄).

지시: "방금 말한 전처리 매우 다양한 조합으로 실험 돌려봐"

한 번에 크게 돌린다 (지시: "큰 실험 한번에 돌려 빠르게 안 돌려도 돼")
    스크리닝을 따로 두지 않고 처음부터 전체 구성(5성분 x 8시드 x XGB+CatBoost)으로
    45개 arm 을 전부 잰다. arm 당 ~300초, 총 3~4시간.
    재확인 단계가 필요 없고 시드 잡음이 작아 +-1 수준 차이도 읽을 수 있다.

등판 분할 (V62 에서 검증)
    asof_pitcher_prev1_game_success_rate 가 경기 단위로만 갱신되는 성질을 쓴다.
    투수별 asof_pitcher_n 순 정렬 -> 그 값이 일정한 구간이 한 등판.
    등판 45,121개, 선발 중앙 90구 / 구원 중앙 16구. 야구 상식과 맞는다.

거르는 축
    S  짧은 등판   자기 평소(선발/구원별 중앙) 대비 비율 < th
    A  절대 짧음   등판 총 투구 수 < th
    V  저물량      asof_pitcher_n < th
    L  긴 등판     비율 > th        (피로 구간, 대칭 확인용)
    R  시즌 첫 등판 (감각 회복 전)
    처리는 제외(drop) 또는 가중치 축소(w)

    V9 에서 학습 행 절반 감소 비용을 -4.39 dBSS 로 쟀다. 2~5% 제거의 볼륨 비용은
    -0.2 ~ -0.4 수준이므로, 거른 행이 진짜 잡음이면 순이득이 난다.

판정: V61 이 확정한 대로 내부 +3 미만은 제출 근거로 쓰지 않는다.
      스크리닝은 방향만 본다.
출력: outputs/v63_filter_sweep.csv
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
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, F_WEIGHT, K = 400, 0.20, 300
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
VS = 2024
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
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
IS_F = df["game_type"].astype(str).to_numpy() == "F"
NVOL = df["asof_pitcher_n"].to_numpy()
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])

o = np.argsort(pid.astype(np.int64) * 10_000_000 + NVOL, kind="stable")
pv = df["asof_pitcher_prev1_game_success_rate"].to_numpy()[o]
gp = pid[o]
chg = np.r_[True, (gp[1:] != gp[:-1]) | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
outing = np.empty(len(df), dtype=np.int64)
outing[o] = np.cumsum(chg) - 1
od = pd.DataFrame({"outing": outing, "pid": pid, "inn": df["inning"].to_numpy(),
                   "season": season, "tr": tr, "nv": NVOL})
agg = od.groupby("outing").agg(n=("outing", "size"), pid=("pid", "first"),
                               first_inn=("inn", "min"), season=("season", "first"),
                               ntr=("tr", "sum"), nv=("nv", "min"))
agg["start"] = (agg["first_inn"] == 1).astype(int)
med = agg[agg["ntr"] > 0].groupby(["pid", "start"])["n"].median().rename("med")
agg = agg.join(med, on=["pid", "start"])
agg["ratio"] = agg["n"] / agg["med"].clip(lower=1)
agg["first_of_season"] = (agg.sort_values("nv").groupby(["pid", "season"])
                          .cumcount() == 0).astype(int)
RATIO = np.nan_to_num(agg["ratio"].reindex(outing).to_numpy(), nan=1.0)
OUT_N = agg["n"].reindex(outing).to_numpy()
FIRST_S = agg["first_of_season"].reindex(outing).to_numpy() == 1
print(f"등판 {len(agg):,}개  선발 중앙 {int(agg.loc[agg.start==1,'n'].median())}구  "
      f"구원 중앙 {int(agg.loc[agg.start==0,'n'].median())}구", flush=True)

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
    v2 = np.where(np.isnan(e2.reindex(pidx).to_numpy()), l0, e2.reindex(pidx).to_numpy())
    v3 = np.where(np.isnan(e3.reindex(i3).to_numpy()), l0, e3.reindex(i3).to_numpy())
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    F[f"{tag}_split"], F[f"{tag}_rel"] = v3 - v2, sz / (sz + K)
    F[f"{tag}_w"] = (v3 - v2) * sz / (sz + K)
X = F.to_numpy(np.float32)


def extrap(a, keep):
    m_ = tr & keep & ~np.isnan(a)
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


def line(keep, wmul):
    p = {}
    w = np.where(IS_F, F_WEIGHT, 1.0) * wmul
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & keep & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, keep),
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


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
y_va = y_all[va]
ref = metrics(y_va, b)["bss_raw"]
wv = BW[bucket_all[va]]
ALL = np.ones(len(df), bool)
ONE = np.ones(len(df))

ARMS = [("C0_baseline", ALL, ONE)]
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    S = RATIO < th
    ARMS.append((f"S{th}_drop", ~S, ONE))
    for ww in [0.25, 0.5, 0.75]:
        ARMS.append((f"S{th}_w{ww}", ALL, np.where(S, ww, 1.0)))
for th in [5, 10, 15, 20]:
    A = OUT_N < th
    ARMS.append((f"A{th}_drop", ~A, ONE))
    ARMS.append((f"A{th}_w05", ALL, np.where(A, 0.5, 1.0)))
for th in [50, 100, 200, 500]:
    V = NVOL < th
    ARMS.append((f"V{th}_drop", ~V, ONE))
    ARMS.append((f"V{th}_w05", ALL, np.where(V, 0.5, 1.0)))
for th in [1.5, 2.0]:
    L = RATIO > th
    ARMS.append((f"L{th}_w05", ALL, np.where(L, 0.5, 1.0)))
ARMS.append(("R_firstseason_drop", ~FIRST_S, ONE))
ARMS.append(("R_firstseason_w05", ALL, np.where(FIRST_S, 0.5, 1.0)))
# 조합
S5, V1c, A10 = RATIO < 0.5, NVOL < 100, OUT_N < 10
ARMS += [
    ("X_S5w05_V100drop", ~V1c, np.where(S5, 0.5, 1.0)),
    ("X_S5drop_V100w05", ~S5, np.where(V1c, 0.5, 1.0)),
    ("X_S5w05_A10w05", ALL, np.where(S5 | A10, 0.5, 1.0)),
    ("X_S5w05_V100w05", ALL, np.where(S5, 0.5, 1.0) * np.where(V1c, 0.5, 1.0)),
]

t0, rows = time.time(), []
print(f"{chr(10)}전체 구성 {len(ARMS)}개 arm (8시드 x XGB+CatBoost)  예상 {len(ARMS)*5:.0f}분")
print(f"{'arm':<22}{'학습행':>10}{'제외%':>8}{'단독':>10}{'ΔBSS':>9}{'경과':>8}")
for name, keep, wmul in ARMS:
    p_ie = line(keep, wmul)
    solo = metrics(y_va, p_ie)["bss_raw"]
    q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
    d = metrics(y_va, q)["bss_raw"] - ref
    ntr = int((tr & keep).sum())
    drop = 100 * (1 - ntr / int(tr.sum()))
    rows.append({"arm": name, "n_train": ntr, "drop_pct": drop,
                 "solo_bss": solo, "dbss": d})
    print(f"{name:<22}{ntr:>10,}{drop:>7.2f}%{solo:>10.2f}{d:>+9.2f}"
          f"{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v63_filter_sweep.csv", index=False)
r0 = res[res.arm == "C0_baseline"].iloc[0]
res["d_vs_base"] = res["dbss"] - r0.dbss
res["solo_vs_base"] = res["solo_bss"] - r0.solo_bss
print(f"{chr(10)}{'='*72}{chr(10)}기준선 대비 상위 12{chr(10)}{'='*72}")
top = res.sort_values("d_vs_base", ascending=False).head(12)
print(f"{'arm':<22}{'ΔBSS 대비':>12}{'단독 대비':>12}{'제외%':>9}")
for _, r in top.iterrows():
    print(f"{r.arm:<22}{r.d_vs_base:>+12.2f}{r.solo_vs_base:>+12.2f}{r.drop_pct:>8.2f}%")
print(f"{chr(10)}하위 5")
for _, r in res.sort_values("d_vs_base").head(5).iterrows():
    print(f"{r.arm:<22}{r.d_vs_base:>+12.2f}{r.solo_vs_base:>+12.2f}{r.drop_pct:>8.2f}%")
print(f"{chr(10)}전체 구성으로 잰 값이라 재확인 단계가 없다. 상위 arm 은 fold 2023 으로 넘긴다.")
print(f"{chr(10)}saved -> {OUT/'v63_filter_sweep.csv'}")
