"""V35: 실패 분류를 더 잘게 쪼갠다 + 투수x타자 직접 대결.

V33 이 말해준 것
    arm         성분단독    현행 대비
    M0_current   746.85      +0.00
    M_base       734.57      +1.64   <- 6개 중 유일한 양수, 선택편향 감안하면 노이즈
    M_outs       733.71      -0.70
    M_li         721.30      -2.72
    M_score      707.84      -2.41
    M_opp        646.92      -4.94
    M_month      643.75      -7.33

    모든 축이 성분단독을 떨어뜨렸다(746.85 -> 707~735). 111 피처에서 포화다.
    투수x타자손x축 계열의 계층 차감은 네 축(손,카운트,이닝,타자)으로 끝났다.

그래서 축이 아니라 구조를 바꾼다

  (1) 실패 분류를 더 잘게
      OUTSIDE 를 ball 로 쪼갠 것이 +2.64 였다(V6). 같은 수를 큰 성분에 쓴다.
          fail = M ∪ R ∪ O,  O = fail ∧ ¬M ∧ ¬R
          P(fail) = P(M) + P(R) − P(M∩R) + P(ob) + P(oz)
      M 을 ball 로 분할하면 P(M) = P(Mb) + P(Mz), R 도 같다. 포함-배제는 그대로.
          P(fail) = p_mb + p_mz + p_rb + p_rz − p_mr + p_ob + p_oz
      기저율 m 0.106, r 0.229 라 쪼개도 학습 가능한 크기다.

  (2) 투수 x 타자 직접 대결
      V33 은 상대'팀'을 쟀고(-4.94) 개별 타자는 안 쟀다. 계층 차감으로
          EB(투수, 타자) − EB(투수, 타자손)
      희소하지만 EB 수축(K=300)이 처리한다. 프로덕션 이름이
      pitcher_cluster_matchup 이라 base 에 이미 있을 수 있으나 성분 라인에는 없다.

  (3) 용량
      111 피처인데 max_leaves 는 성분별 8/12/18 이다. 피처가 늘어난 뒤로
      한 번도 안 늘렸다.

arm
    P0_current   5성분 (현행)
    P1_rsplit    6성분  r -> rb, rz
    P2_msplit    6성분  m -> mb, mz
    P3_both      7성분  둘 다
    P4_matchup   5성분 + 투수x타자 계층 차감
    P5_capacity  5성분, max_leaves 2배

판정: Val2024 선별 후 이긴 것만 세 fold 확인.
출력: outputs/v35_finer_components.csv
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
bid = df["batter_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
row_w = np.where(df["game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
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


BASE5 = {"m": ym, "r": yr, "mr": AND(ym, yr),
         "ob": AND(yo, yb), "oz": AND(yo, 1 - yb)}
LAB_ALL = dict(BASE5)
LAB_ALL.update({"mb": AND(ym, yb), "mz": AND(ym, 1 - yb),
                "rb": AND(yr, yb), "rz": AND(yr, 1 - yb)})

# 분해 검증 — 라벨 수준에서 포함-배제가 정확히 성립하는가
v = ok & ~np.isnan(ym)
fail = 1 - y_all
for name, expr in [
        ("5성분", BASE5["m"] + BASE5["r"] - BASE5["mr"] + BASE5["ob"] + BASE5["oz"]),
        ("7성분", LAB_ALL["mb"] + LAB_ALL["mz"] + LAB_ALL["rb"] + LAB_ALL["rz"]
         - LAB_ALL["mr"] + LAB_ALL["ob"] + LAB_ALL["oz"])]:
    bad = int((np.abs(expr[v] - fail[v]) > 1e-9).sum())
    print(f"  {name} 포함-배제 불일치 {bad}행 / {int(v.sum()):,}행")
    assert bad == 0, f"{name} 분해가 성립하지 않는다"
print("  기저율 " + "  ".join(f"{k} {np.nanmean(x[tr]):.3f}"
                            for k, x in LAB_ALL.items()), flush=True)


def layered(second, axis, K=300):
    d = pd.DataFrame({"p": pid[tr], "s": second[tr], "a": axis[tr], "y": y_all[tr]})
    lg = float(d["y"].mean())
    g2 = d.groupby(["p", "s"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "s", "a"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * lg) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * lg) / (g3["size"] + K)
    i2 = pd.MultiIndex.from_arrays([pid, second])
    i3 = pd.MultiIndex.from_arrays([pid, second, axis])
    v2 = eb2.reindex(i2).to_numpy(); v3 = eb3.reindex(i3).to_numpy()
    v2 = np.where(np.isnan(v2), lg, v2); v3 = np.where(np.isnan(v3), lg, v3)
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


def matchup(K=300):
    """EB(투수, 타자) − EB(투수, 타자손). 개별 대결 편차."""
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "b": bid[tr], "y": y_all[tr]})
    lg = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "b"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * lg) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * lg) / (g3["size"] + K)
    v2 = eb2.reindex(pd.MultiIndex.from_arrays([pid, bhand])).to_numpy()
    v3 = eb3.reindex(pd.MultiIndex.from_arrays([pid, bid])).to_numpy()
    v2 = np.where(np.isnan(v2), lg, v2); v3 = np.where(np.isnan(v3), lg, v3)
    sz = g3["size"].reindex(pd.MultiIndex.from_arrays([pid, bid])).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


td = df.loc[tr]
BASE_F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v_[tr] for k, v_ in BASE5.items()}))
for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
    sp, rel = layered(bhand, ax)
    BASE_F[f"{tag}_split"], BASE_F[f"{tag}_rel"] = sp, rel
    BASE_F[f"{tag}_w"] = sp * rel
print(f"기준 피처 {BASE_F.shape[1]}개", flush=True)


def extrap(a):
    m = tr & ~np.isnan(a)
    s = pd.Series(a[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


def params_for(rate, mult=1):
    if rate < 0.06:
        p = {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    elif rate < 0.15:
        p = {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    else:
        p = {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}
    p["max_leaves"] *= mult
    return p


def fit(X, tags, mult=1):
    p = {}
    for tag in tags:
        arr = LAB_ALL[tag]
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr),
               **params_for(float(np.nanmean(arr[tr])), mult)}
        d_tr = xgb.DMatrix(X[m], label=arr[m], weight=row_w[m])
        d_va = xgb.DMatrix(X[va])
        p_tr, p_va = Pool(X[m], arr[m], weight=row_w[m]), Pool(X[va])
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6 * mult,
                                   learning_rate=0.05, l2_leaf_reg=6.0,
                                   loss_function="Logloss", random_seed=s,
                                   task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(p_va)[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return p


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b, game_type=gt)["bss_raw"]

POS = {"m", "r", "mb", "mz", "rb", "rz", "ob", "oz"}
ARMS = [("P0_current", ["m", "r", "mr", "ob", "oz"], False, 1),
        ("P1_rsplit", ["m", "rb", "rz", "mr", "ob", "oz"], False, 1),
        ("P2_msplit", ["mb", "mz", "r", "mr", "ob", "oz"], False, 1),
        ("P3_both", ["mb", "mz", "rb", "rz", "mr", "ob", "oz"], False, 1),
        ("P4_matchup", ["m", "r", "mr", "ob", "oz"], True, 1),
        ("P5_capacity", ["m", "r", "mr", "ob", "oz"], False, 2)]

t0, rows = time.time(), []
print(f"\n{'arm':<14}{'성분':>5}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}"
      f"{'t_row':>8}{'경과':>8}")
for name, tags, use_mu, mult in ARMS:
    F = BASE_F.copy()
    if use_mu:
        sp, rel = matchup()
        F["mu_split"], F["mu_rel"], F["mu_w"] = sp, rel, sp * rel
    p = fit(F.to_numpy(np.float32), tags, mult)
    fail_hat = sum(p[t] if t in POS else -p[t] for t in tags)
    p_ie = np.clip(1 - fail_hat, EPS, 1 - EPS)
    np.save(CACHE / f"v35_{name}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(b / (1 - b)), np.log(p_ie / (1 - p_ie)))[0, 1])
    q = np.clip(W * p_ie + (1 - W) * b, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "n_comp": len(tags), "n_features": F.shape[1],
                 "solo_bss": solo, "corr": corr, "bss": mm["bss_raw"], "dbss": d,
                 "se_row": se, "t_row": d / se, "pred_mean": mm["pred_mean"]})
    print(f"{name:<14}{len(tags):>5}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}"
          f"{d:>+9.2f}{d/se:>8.2f}{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v35_finer_components.csv", index=False)
r0 = res[res.arm == "P0_current"]["dbss"].iloc[0]
res["vs_current"] = res["dbss"] - r0
print("\n" + "=" * 60)
for _, r in res.sort_values("vs_current", ascending=False).iterrows():
    print(f"{r.arm:<14}{r.vs_current:>+10.2f}   단독 {r.solo_bss:8.2f}")
print("\n단독 BSS 가 함께 올라야 진짜다. ΔBSS 만 오르고 단독이 떨어지면")
print("base 와의 상보성이 늘어난 것뿐이고 V33 에서 그런 것들은 전이가 없었다.")
print(f"\nsaved -> {OUT/'v35_finer_components.csv'}")
