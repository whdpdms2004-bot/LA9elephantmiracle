"""V71: 피처를 2차로 확장하면 설명력이 오르는가 — 세 종류를 갈라서 잰다.

지시: "지금 피쳐를 1차로만 보고 있잖아 2차로 바꾸면 더 설명력이 높아지는 피쳐가
       있지는 않을지 실험"

먼저 갈라야 할 것 — GBDT 에서 '2차'는 한 가지가 아니다
    트리는 단일 피처의 단조 변환에 불변이다. x^2 를 넣어도 분할 지점만 바뀌고
    정보가 늘지 않는다. 반면 곱/차/비 같은 교차항은 축 정렬 분할로 근사하기 어렵다.
    따라서 세 질문을 따로 물어야 한다.

        C1  제곱항        -> 이론상 0 이어야 한다. 맞으면 측정이 건전하다는 확인.
        C2  곱 상호작용    -> 트리가 못 만드는 것. 여기가 진짜 후보.
        C3  차 / 비       -> 상대 강도, 폼 편차. 스케일이 달라 곱과 성격이 다르다.
        C4  C2 + C3

    이미 손으로 넣은 2차항이 하나 있다: platoon_w = split * rel.
    그게 되는 걸 봤으니 같은 종류를 더 찾는 실험이다.

교차항 선택 (사람이 고른 12쌍)
    축 교차   platoon_split x cnt_split,  platoon_split x inn_split,
              cnt_split x inn_split                <- 계층차감 축들의 상호작용.
                                                      지금 각 축은 서로 독립으로만 들어간다.
    상성      pitcher_success x batter_success,  pitcher_middle x batter_middle
    신뢰도    pitcher_success x log1p(n),  platoon_split x log1p(n)
    상황      pitcher_success x li,  pitcher_reverse x fastball_rate
    폼        prev1 - career,  prev3 - career,  prev5 - career
    상대      pitcher_success - batter_success

DL 축과의 관계
    attention 은 2차 상호작용을 구조로 학습한다(V69/V70).
    이 실험은 '명시적 2차항을 GBDT 에 주기' vs 'attention 이 배우기' 의 비교 기준이 된다.
    C2 가 크면 상호작용이 실재하는데 트리가 못 잡고 있다는 뜻이고,
    C2 가 0 이면 트리가 이미 잡고 있어 attention 의 여지도 그만큼 좁다.

판정: 두 fold 모두 단독과 ΔBSS 가 함께 올라야 한다.
출력: outputs/v71_second_order.csv
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
N_ROUNDS, K = 400, 300
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

o_ = np.argsort(pid.astype(np.int64) * 10_000_000 + NVOL, kind="stable")
pv = df["asof_pitcher_prev1_game_success_rate"].to_numpy()[o_]
gp = pid[o_]
chg = np.r_[True, (gp[1:] != gp[:-1]) | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
outing = np.empty(len(df), dtype=np.int64)
outing[o_] = np.cumsum(chg) - 1
ag = pd.DataFrame({"outing": outing, "pid": pid,
                   "inn": df["inning"].to_numpy()}).groupby("outing").agg(
    n=("outing", "size"), pid=("pid", "first"), first_inn=("inn", "min"))
ag["start"] = (ag["first_inn"] == 1).astype(int)
ag = ag.join(ag.groupby(["pid", "start"])["n"].median().rename("med"),
             on=["pid", "start"])
SHORT = np.nan_to_num((ag["n"] / ag["med"].clip(lower=1)).reindex(outing).to_numpy(),
                      nan=1.0) < 0.5
ROW_W = np.where(IS_F, 0.20, 1.0) * np.where(SHORT, 0.5, 1.0)

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


SQ_BASE = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
           "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
           "asof_batter_success_rate", "li", "platoon_split", "cnt_split",
           "inn_split"]
PRODUCTS = [
    ("platoon_split", "cnt_split"), ("platoon_split", "inn_split"),
    ("cnt_split", "inn_split"),
    ("asof_pitcher_success_rate", "asof_batter_success_rate"),
    ("asof_pitcher_middle_rate", "asof_batter_middle_rate"),
    ("asof_pitcher_success_rate", "li"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_fastball_rate"),
]
DIFFS = [
    ("asof_pitcher_prev1_game_success_rate", "asof_pitcher_success_rate"),
    ("asof_pitcher_prev3_game_success_rate", "asof_pitcher_success_rate"),
    ("asof_pitcher_prev5_game_success_rate", "asof_pitcher_success_rate"),
    ("asof_pitcher_prev1_game_middle_rate", "asof_pitcher_middle_rate"),
    ("asof_pitcher_success_rate", "asof_batter_success_rate"),
]


def add_second_order(F, kind):
    G = F.copy()
    lg_n = np.log1p(NVOL)
    if kind in ("C1",):
        for c in SQ_BASE:
            if c in G.columns:
                G[f"sq_{c}"] = G[c].to_numpy() ** 2
    if kind in ("C2", "C4"):
        for a, b in PRODUCTS:
            if a in G.columns and b in G.columns:
                G[f"x_{a[:14]}_{b[:14]}"] = G[a].to_numpy() * G[b].to_numpy()
        G["x_psucc_logn"] = G["asof_pitcher_success_rate"].to_numpy() * lg_n
        G["x_split_logn"] = G["platoon_split"].to_numpy() * lg_n
    if kind in ("C3", "C4"):
        for a, b in DIFFS:
            if a in G.columns and b in G.columns:
                G[f"d_{a[:16]}_{b[:12]}"] = G[a].to_numpy() - G[b].to_numpy()
        pb = G["asof_pitcher_success_rate"].to_numpy()
        bb = G["asof_batter_success_rate"].to_numpy()
        G["r_pitch_bat"] = pb / np.clip(bb, 1e-3, None)
    return G


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def line(X, fold):
    tr, va = season < fold, season == fold
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & ~np.isnan(arr)
        s_ = pd.Series(arr[mm]).groupby(pd.Series(season[mm])).mean().sort_index()
        bs = float(np.clip(float(s_.iloc[-1]) + (float(s_.iloc[-1]) - float(s_.iloc[0]))
                           / (float(s_.index[-1]) - float(s_.index[0])), 0.005, 0.995))
        prm = {**BASE_PARAMS, "base_score": bs,
               **params_for(float(np.nanmean(arr[mm])))}
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=ROW_W[mm])
        d_va = xgb.DMatrix(X[va])
        p_tr = Pool(X[mm], arr[mm], weight=ROW_W[mm])
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
NAME = {"C0": "현행 1차", "C1": "+제곱항", "C2": "+곱 상호작용",
        "C3": "+차/비", "C4": "+곱+차/비"}
for fold in FOLDS:
    va = season == fold
    BF = base_features(fold)
    y, b = y_all[va], BASE_P[fold]
    null = y.mean() * (1 - y.mean())
    ref = metrics(y, b)["bss_raw"]
    wv = BW[bucket_all[va]]
    print(f"{chr(10)}fold {fold}   base {ref:9.2f}")
    print(f"  {'arm':<16}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}")
    for arm in ["C0", "C1", "C2", "C3", "C4"]:
        F = BF if arm == "C0" else add_second_order(BF, arm)
        p_ie = line(F.to_numpy(np.float32), fold)
        np.save(CACHE / f"v71_{arm}_{fold}.npy", p_ie)
        solo = metrics(y, p_ie)["bss_raw"]
        corr = float(np.corrcoef(lgf(b), lgf(p_ie))[0, 1])
        q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
        d = metrics(y, q)["bss_raw"] - ref
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": fold, "arm": arm, "n_features": F.shape[1],
                     "solo_bss": solo, "corr": corr, "dbss": d, "t_row": d / se})
        print(f"  {arm} {NAME[arm]:<13}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}"
              f"{d:>+9.2f}{d/se:>8.2f}   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v71_second_order.csv", index=False)
piv = res.pivot_table(index="arm", columns="fold", values="dbss")
sol = res.pivot_table(index="arm", columns="fold", values="solo_bss")
print(f"{chr(10)}{'='*58}{chr(10)}C0 대비{chr(10)}{'='*58}")
print("ΔBSS")
print(piv.subtract(piv.loc["C0"], axis=1).round(2).to_string())
print(f"{chr(10)}성분단독")
print(sol.subtract(sol.loc["C0"], axis=1).round(2).to_string())
print(f"{chr(10)}C1(제곱)이 0 근처면 측정이 건전하다는 확인이다 — 트리는 단조 변환에 불변.")
print(f"C2(곱)가 크면 상호작용이 실재하는데 트리가 못 잡고 있다는 뜻이고,")
print(f"0 이면 트리가 이미 잡고 있어 attention(V69/V70)의 여지도 그만큼 좁다.")
print(f"{chr(10)}saved -> {OUT/'v71_second_order.csv'}")
