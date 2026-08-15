"""V33: 계층 차감 축을 더 판다 — 아웃카운트·경기중요도·주자상황·점수차·상대팀·시기.

왜 이 방향인가 (Public 결과가 정해줬다)
    submit_024  성분(투수 플래툰 1축)      Val2024 +13.75  Public 945
    submit_029  성분(4축) w_eff 0.684      Val2024 +16.12  Public 963
    submit_030  성분(4축) w    0.250       Val2024 +40.48  Public 964

    029 와 030 은 피처가 같고 가중치만 다르다. 내부 24점 차이가 Public 1점이다.
    024 와 030 은 축이 1개 -> 4개로 늘었다. 내부 축 기여 합 +13.86 이 Public +19 다.

    > Val2024 는 '정보 추가'에는 예측력이 있고(+13.9 -> +19),
    >          '가중치 조정'에는 없다(+24 -> +1).

    그러니 가중치는 그만 만지고 축을 늘린다.
    V32 도 같은 방향을 가리킨다 — 2023 붕괴는 레벨이 아니라 형태이고(오라클
    레벨 보정 후 격차가 987 -> 1095 로 오히려 벌어진다) 가중치로 못 푼다.

축 (전부 계층 차감 — V19 의 원리)
    split = EB(투수, 타자손, 축) − EB(투수, 타자손)
    주효과를 안 빼면 원본 피처와 중복이고 값이 죽는다(V8 +0.16 vs V19 +8.44).

    M_outs    아웃카운트          사용자가 처음 지목한 축
    M_li      경기중요도 li 4분위  사용자가 처음 지목한 축
    M_base    주자상황 8종
    M_score   점수차 5구간
    M_opp     상대팀 (EB(투수, 상대팀) − EB(투수), 타자손 대신 팀)
    M_month   시즌 내 시기

    현행 4축(투수·타자·카운트·이닝) 위에 하나씩 얹는다. V22 에서 전부 넣으면
    성분단독이 760 -> 721 로 무너지는 걸 봤다. 한 번에 하나다.

1단계: Val2024 로 선별 (w=0.25, 프로덕션 836.503 대비)
2단계: 이긴 축만 세 fold 확인 (별도 스크립트)

출력: outputs/v33_new_axes.csv
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

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W, F_WEIGHT = 400, 0.25, 0.20
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

pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
gtype = df["game_type"].astype(str).to_numpy()
row_w = np.where(gtype == "F", F_WEIGHT, 1.0)

balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
AX = {
    "cnt": np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1)),
    "inn": np.digitize(df["inning"].to_numpy(), [4, 7, 10]),
    "outs": df["outs_before"].to_numpy(),
    "li": np.digitize(df["li"].to_numpy(),
                      np.nanquantile(df.loc[tr, "li"].to_numpy(), [.25, .5, .75])),
    "base": df["base_state"].cat.codes.to_numpy(),
    "score": np.digitize(df["score_diff_pitcher_team"].to_numpy(), [-3, 0, 1, 4]),
    "month": df["game_month"].to_numpy(),
}
OPP = df["batter_team_id"].to_numpy()


def layered(axis, K=300, second=None):
    """EB(투수, second, 축) − EB(투수, second). second 가 None 이면 (투수, 축)−(투수)."""
    keys = [pid] if second is None else [pid, second]
    d = pd.DataFrame({f"k{i}": k[tr] for i, k in enumerate(keys)})
    d["a"], d["y"] = axis[tr], y_all[tr]
    kc = [f"k{i}" for i in range(len(keys))]
    lg = float(d["y"].mean())
    g2 = d.groupby(kc)["y"].agg(["sum", "size"])
    g3 = d.groupby(kc + ["a"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * lg) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * lg) / (g3["size"] + K)
    i2 = pd.MultiIndex.from_arrays(keys) if len(keys) > 1 else pd.Index(keys[0])
    i3 = pd.MultiIndex.from_arrays(keys + [axis])
    v2 = eb2.reindex(i2).to_numpy(); v3 = eb3.reindex(i3).to_numpy()
    v2 = np.where(np.isnan(v2), lg, v2); v3 = np.where(np.isnan(v3), lg, v3)
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}

td = df.loc[tr]
BASE_F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
for tag in ["cnt", "inn"]:
    sp, rel = layered(AX[tag], second=bhand)
    BASE_F[f"{tag}_split"], BASE_F[f"{tag}_rel"] = sp, rel
    BASE_F[f"{tag}_w"] = sp * rel
print(f"기준 피처 {BASE_F.shape[1]}개 (submit_030 구성)", flush=True)


def extrap(a):
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


def run(X):
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr),
               **params_for(float(np.nanmean(arr[tr])))}
        d_tr = xgb.DMatrix(X[m], label=arr[m], weight=row_w[m])
        d_va = xgb.DMatrix(X[va])
        p_tr, p_va = Pool(X[m], arr[m], weight=row_w[m]), Pool(X[va])
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(p_va)[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b, game_type=gt)["bss_raw"]

ARMS = [("M0_current", None, None),
        ("M_outs", AX["outs"], bhand),
        ("M_li", AX["li"], bhand),
        ("M_base", AX["base"], bhand),
        ("M_score", AX["score"], bhand),
        ("M_opp", OPP, None),
        ("M_month", AX["month"], bhand)]

t0, rows = time.time(), []
print(f"\n{'arm':<13}{'피처':>6}{'성분단독':>11}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}{'경과':>8}")
for name, axis, second in ARMS:
    F = BASE_F.copy()
    if axis is not None:
        sp, rel = layered(axis, second=second)
        F["new_split"], F["new_rel"], F["new_w"] = sp, rel, sp * rel
    p_ie = run(F.to_numpy(np.float32))
    np.save(CACHE / f"v33_{name}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(b / (1 - b)),
                             np.log(p_ie / (1 - p_ie)))[0, 1])
    q = np.clip(W * p_ie + (1 - W) * b, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "n_features": F.shape[1], "solo_bss": solo,
                 "corr": corr, "bss": mm["bss_raw"], "dbss": d, "se_row": se,
                 "t_row": d / se})
    print(f"{name:<13}{F.shape[1]:>6}{solo:>11.2f}{corr:>8.4f}{d:>+9.2f}"
          f"{d/se:>8.2f}{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v33_new_axes.csv", index=False)
r0 = res[res.arm == "M0_current"]["dbss"].iloc[0]
res["vs_current"] = res["dbss"] - r0
print("\n" + "=" * 72)
print(f"{'arm':<13}{'현행 대비':>12}")
for _, r in res.sort_values("vs_current", ascending=False).iterrows():
    print(f"{r.arm:<13}{r.vs_current:>+12.2f}")
print("\n최고 축만 세 fold 로 확인한다. 후보 6개 중 2024 최고를 고른 것이므로")
print("선택 편향이 섞여 있다 (V23 에서 같은 함정을 겪었다).")
res.to_csv(OUT / "v33_new_axes.csv", index=False)
print(f"\nsaved -> {OUT/'v33_new_axes.csv'}")
