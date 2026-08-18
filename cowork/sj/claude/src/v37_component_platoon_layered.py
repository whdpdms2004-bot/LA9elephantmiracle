"""V37: V8(성분별 플래툰)을 계층 차감으로 다시 한다.

왜 다시 하는가
    V8 은 성분별 플래툰을 넣고 +0.16 으로 기각했다. 그때 구성은
        split_k(p,h) = EB_k(p,h) − 리그평균_k                      1겹 차감
    V19 가 밝힌 것: 1겹 차감은 값이 죽고 2겹 차감은 +8.44 를 낸다.
    구조가 같은 피처가 차감 겹수만으로 50배 차이가 났다.

    > "버렸던 것도 다시 본다" — V8 은 아이디어가 틀린 게 아니라 차감이 모자랐을
    > 가능성이 크다.

이번 구성
    투수의 '전반적 실력'을 빼고 '실패 방식의 편향'만 남긴다.

        d_k(p,h) = [EB_k(p,h) − EB_k(전체)] − [EB_s(p,h) − EB_s(전체)] × (base_k/base_s)

    앞항은 이 투수가 성분 k 를 얼마나 더 내는가, 뒷항은 그 투수가 원래 얼마나
    실패하는가를 성분 k 규모로 환산한 것이다. 뒤를 빼면 "이 투수는 못하는데,
    특히 k 방식으로 못한다"만 남는다. 이건 기존 플래툰 피처와 중복되지 않는다.

    arm 으로 차감 겹수를 직접 비교한다. 이게 이 실험의 요점이다.
        Q0  현행 (성분별 테이블 없음)
        Q1  1겹  EB_k(p,h) − 리그평균_k            <- V8 재현
        Q2  2겹  Q1 − 투수 전반 편차 (스케일 보정)  <- 본 제안
        Q3  2겹 + 상대 표본 가중치 rel 동반
        Q4  Q3 를 성분 모델별로 자기 성분만 투입     <- 자기 라벨 누수 위험 점검

주의 — 자기 라벨 누수
    성분 k 의 테이블을 성분 k 모델에 넣으면 정적 테이블이라 자기 라벨이 샌다.
    V1 에서 정적 '레벨'을 넣었더니 direct_bss 705.7 -> 187.5 로 붕괴했다.
    Q1~Q3 은 다섯 성분의 테이블을 다섯 모델 전부에 넣는다(공유). Q4 만 자기
    성분을 자기 모델에 넣어 그 위험을 직접 잰다. 단독 BSS 가 무너지면 확정이다.

출력: outputs/v37_component_platoon.csv
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
N_ROUNDS, W, F_WEIGHT, K = 400, 0.25, 0.20, 300
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
row_w = np.where(df["game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, ((ym == 1) & (yr == 1)).astype(float), np.nan),
       "ob": np.where(ok, ((yo == 1) & (yb == 1)).astype(float), np.nan),
       "oz": np.where(ok, ((yo == 1) & (yb == 0)).astype(float), np.nan)}

idx = pd.MultiIndex.from_arrays([pid, bhand])


def eb_ph(vals, mask):
    """EB(투수, 타자손) 과 상대 표본 가중치. vals 의 NaN 은 제외한다."""
    m = mask & ~np.isnan(vals)
    d = pd.DataFrame({"p": pid[m], "h": bhand[m], "y": vals[m]})
    lg = float(d["y"].mean())
    g = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    eb = (g["sum"] + K * lg) / (g["size"] + K)
    v = eb.reindex(idx).to_numpy()
    sz = g["size"].reindex(idx).fillna(0.0).to_numpy()
    return np.where(np.isnan(v), lg, v), lg, sz / (sz + K)


SUCC_EB, SUCC_LG, SUCC_REL = eb_ph(y_all, tr)
COMP = {}
for tag in COMPONENTS:
    eb, lg_, rel = eb_ph(LAB[tag], tr)
    COMP[tag] = {"eb": eb, "lg": lg_, "rel": rel,
                 "scale": lg_ / max(SUCC_LG, 1e-6)}
    print(f"  {tag:<3} 리그평균 {lg_:.4f}   스케일 {COMP[tag]['scale']:.4f}",
          flush=True)


def split_of(tag, depth):
    c = COMP[tag]
    one = c["eb"] - c["lg"]
    if depth == 1:
        return one
    # 투수 전반 실패 편향을 성분 규모로 환산해 뺀다.
    # 성공률 편차의 부호를 뒤집어야 실패 방향이 된다.
    return one + (SUCC_EB - SUCC_LG) * c["scale"]


td = df.loc[tr]
BASE_F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))


def layered_axis(axis):
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": axis[tr], "y": y_all[tr]})
    lg_ = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * lg_) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * lg_) / (g3["size"] + K)
    i3 = pd.MultiIndex.from_arrays([pid, bhand, axis])
    v2 = eb2.reindex(idx).to_numpy(); v3 = eb3.reindex(i3).to_numpy()
    v2 = np.where(np.isnan(v2), lg_, v2); v3 = np.where(np.isnan(v3), lg_, v3)
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
    sp, rel = layered_axis(ax)
    BASE_F[f"{tag}_split"], BASE_F[f"{tag}_rel"] = sp, rel
    BASE_F[f"{tag}_w"] = sp * rel
print(f"기준 피처 {BASE_F.shape[1]}개", flush=True)


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


def fit_one(X, tag):
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
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b, game_type=gt)["bss_raw"]

ARMS = [("Q0_none", 0, False, False),
        ("Q1_depth1", 1, False, False),
        ("Q2_depth2", 2, False, False),
        ("Q3_depth2_rel", 2, True, False),
        ("Q4_own_only", 2, True, True)]

t0, rows = time.time(), []
print(f"\n{'arm':<15}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}{'경과':>8}")
for name, depth, with_rel, own in ARMS:
    p = {}
    for tag in COMPONENTS:
        F = BASE_F.copy()
        if depth:
            src = [tag] if own else COMPONENTS
            for k in src:
                F[f"cp_{k}"] = split_of(k, depth)
                if with_rel:
                    F[f"cp_{k}_rel"] = COMP[k]["rel"]
        p[tag] = fit_one(F.to_numpy(np.float32), tag)
    nfeat = F.shape[1]
    p_ie = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
    np.save(CACHE / f"v37_{name}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(b / (1 - b)), np.log(p_ie / (1 - p_ie)))[0, 1])
    q = np.clip(W * p_ie + (1 - W) * b, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "n_features": nfeat, "solo_bss": solo, "corr": corr,
                 "bss": mm["bss_raw"], "dbss": d, "se_row": se, "t_row": d / se})
    print(f"{name:<15}{nfeat:>6}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}"
          f"{d/se:>8.2f}{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v37_component_platoon.csv", index=False)
r0 = res[res.arm == "Q0_none"]["dbss"].iloc[0]
s0 = res[res.arm == "Q0_none"]["solo_bss"].iloc[0]
print("\n" + "=" * 64)
print(f"{'arm':<15}{'ΔBSS 대비':>12}{'단독 대비':>12}")
for _, r in res.iterrows():
    print(f"{r.arm:<15}{r.dbss-r0:>+12.2f}{r.solo_bss-s0:>+12.2f}")
q1 = res[res.arm == "Q1_depth1"].iloc[0]
q2 = res[res.arm == "Q2_depth2"].iloc[0]
print(f"\n차감 겹수 효과 (Q2 − Q1) {q2.dbss-q1.dbss:+.2f}")
print("V19 에서는 같은 비교가 +0.16 -> +8.44 였다. 재현되면 원리가 확정된다.")
print(f"\nsaved -> {OUT/'v37_component_platoon.csv'}")
