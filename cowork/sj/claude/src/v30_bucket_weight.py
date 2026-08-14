"""V30: 결합 가중치를 투구량 구간별로 나눈다.

논리
    V17 에서 확인한 것: w* 는 (base − 성분단독) 격차에 단조 감소한다.
        fold 2022  격차   −3   ->  w* 0.50
        fold 2024  격차  +89   ->  w* 0.35
        fold 2023  격차 +1134  ->  w* 0.10

    그런데 격차는 fold 안에서도 균일하지 않다. D2 진단에서
        asof_pitcher_n 1–99      local ΔBSS +87.75
        asof_pitcher_n 4000+     local ΔBSS  +3.10
    28배 차이가 났다. 저물량 구간은 base 가 약하고(표본이 얇아
    asof_pitcher_success_rate 가 못 맞춘다) 성분 라인은 상대적으로 강하다.

    전역 w 하나는 이 이질성을 평균으로 뭉갠다. 구간별 w 가 맞다.

    이건 '재표현'이 아니다. 지금까지 쓰지 않은 정보(구간별 격차)를 쓴다.

위험
    파라미터가 1개 -> 5개가 된다. 게이트 fold 에서 고르면 과적합이다.
    그래서 규칙을 그대로 쓴다: 구간별로 '세 fold 모두 양수인 최대 w'.
    추가로 단조 제약(격차가 클수록 w 가 작다) 버전도 함께 잰다.

arm
    W0  전역 w=0.25                    현행 submit_030
    W1  구간별 w = 세 fold 양수 최대    규칙 그대로
    W2  W1 에 단조 제약                 격차 순서로 비증가
    W3  구간별 2024 최적 (참고용)       상한. 채택하지 않는다

성분 라인은 submit_030 구성(플래툰 4축)이다. 절편은 넣지 않는다 —
2022/2023 OOF 에서 적합한 값이라 그 두 fold 에 되먹임된다.

fold 별 성분 예측을 cache/v30_pie_{fold}.npy 에 저장해 이후 실험에서 재사용한다.
출력: outputs/v30_bucket_weight.csv
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

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS = 400
WS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
FOLDS = [2022, 2023, 2024]
CUTS = [100, 500, 2000, 4000]
BNAME = ["0-99", "100-499", "500-1999", "2000-3999", "4000+"]
F_WEIGHT = 0.20
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
bucket = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
gtype = df["game_type"].astype(str).to_numpy()
row_w = np.where(gtype == "F", F_WEIGHT, 1.0)

balls = df["balls_before"].to_numpy()
strikes = df["strikes_before"].to_numpy()
cnt_bucket = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_bucket = np.digitize(df["inning"].to_numpy(), [4, 7, 10])
pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()


def layered_split(tr_mask, axis, K=300):
    """EB(투수, 타자손, axis) - EB(투수, 타자손). 2단계 차감."""
    d = pd.DataFrame({"p": pid[tr_mask], "h": bhand[tr_mask],
                      "a": axis[tr_mask], "y": y_all[tr_mask]})
    lg = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
    eb2 = (g2["sum"] + K * lg) / (g2["size"] + K)
    eb3 = (g3["sum"] + K * lg) / (g3["size"] + K)
    k3 = pd.MultiIndex.from_arrays([pid, bhand, axis])
    k2 = pd.MultiIndex.from_arrays([pid, bhand])
    v3 = np.where(np.isnan(eb3.reindex(k3).to_numpy()), lg, eb3.reindex(k3).to_numpy())
    v2 = np.where(np.isnan(eb2.reindex(k2).to_numpy()), lg, eb2.reindex(k2).to_numpy())
    sz = g3["size"].reindex(k3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}

models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
strong = {}
for fold in FOLDS:
    ids = df.loc[season == fold, "row_id"].to_numpy()
    acc, cnt = None, 0
    for mn in models:
        f = OOF_DIR / f"{mn}_fold{fold}.parquet"
        if f.exists():
            v = pd.read_parquet(f).set_index("row_id").reindex(ids)["prediction"].to_numpy()
            acc = v if acc is None else acc + v
            cnt += 1
    strong[fold] = np.clip(acc / cnt, EPS, 1 - EPS)
prod = pd.read_parquet(PROD).set_index("row_id").reindex(
    df.loc[season == 2024, "row_id"].to_numpy())
strong[2024] = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                       EPS, 1 - EPS)


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


def component_line(vs):
    """submit_030 구성으로 vs 시즌 성분 라인을 낸다. 캐시가 있으면 읽는다."""
    cf = CACHE / f"v30_pie_{vs}.npy"
    if cf.exists():
        return np.load(cf)
    tr, va = season < vs, season == vs
    td = df.loc[tr]
    feat = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                    CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
    for tag, axis in [("count", cnt_bucket), ("inning", inn_bucket)]:
        sp, rel = layered_split(tr, axis)
        feat[f"{tag}_platoon_split"] = sp
        feat[f"{tag}_platoon_rel"] = rel
        feat[f"{tag}_platoon_w"] = sp * rel
    X = feat.to_numpy(np.float32)
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr),
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
    out = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
    np.save(cf, out)
    return out


t0, rows = time.time(), []
PIE, B, Y, BK = {}, {}, {}, {}
for vs in FOLDS:
    va = season == vs
    PIE[vs], B[vs], Y[vs], BK[vs] = component_line(vs), strong[vs], y_all[va], bucket[va]
    g = metrics(Y[vs], B[vs])["bss_raw"] - metrics(Y[vs], PIE[vs])["bss_raw"]
    print(f"fold {vs}  base {metrics(Y[vs], B[vs])['bss_raw']:9.2f}  성분단독 "
          f"{metrics(Y[vs], PIE[vs])['bss_raw']:9.2f}  격차 {g:8.2f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)


def full_bss(vs, wvec):
    """구간별 가중치 벡터로 전체 fold BSS 를 낸다."""
    w = np.asarray(wvec, float)[BK[vs]]
    q = np.clip(w * PIE[vs] + (1 - w) * B[vs], EPS, 1 - EPS)
    return metrics(Y[vs], q)["bss_raw"], q


# ---------------------------------------------- 구간 x w 격자 (구간 하나만 변경)
print("\n" + "=" * 92)
print("구간별 ΔBSS — 해당 구간만 w 를 바꾸고 나머지는 0.25 고정")
print("=" * 92)
grid = {}
for vs in FOLDS:
    base_vec = [0.25] * 5
    ref = full_bss(vs, base_vec)[0]
    print(f"\nfold {vs}   (전역 0.25 = {ref:.2f})")
    print(f"  {'구간':<11}{'비중':>7}{'격차':>9}" + "".join(f"{w:>8.2f}" for w in WS))
    for k in range(5):
        m = BK[vs] == k
        share = float(m.mean())
        gap_k = (metrics(Y[vs][m], B[vs][m])["bss_raw"]
                 - metrics(Y[vs][m], PIE[vs][m])["bss_raw"])
        line = f"  {BNAME[k]:<11}{share*100:>6.2f}%{gap_k:>9.1f}"
        for w in WS:
            v = base_vec.copy(); v[k] = w
            d = full_bss(vs, v)[0] - ref
            grid[(vs, k, w)] = d
            rows.append({"fold": vs, "bucket": BNAME[k], "share": share,
                         "gap": gap_k, "w": w, "dbss_vs_global25": d})
            line += f"{d:>+8.2f}"
        print(line, flush=True)

# ---------------------------------------------- arm 구성
gaps = {k: float(np.mean([grid_gap for grid_gap in
                          [(metrics(Y[v][BK[v] == k], B[v][BK[v] == k])["bss_raw"]
                            - metrics(Y[v][BK[v] == k], PIE[v][BK[v] == k])["bss_raw"])
                           for v in FOLDS]])) for k in range(5)}

w1 = []
for k in range(5):
    good = [w for w in WS if all(grid[(v, k, w)] >= 0 for v in FOLDS)]
    w1.append(max(good) if good else 0.25)
order = sorted(range(5), key=lambda k: gaps[k])      # 격차 오름차순
w2, cap = [0.0] * 5, 1.0
for k in order:
    cap = min(cap, w1[k])
    w2[k] = cap
w3 = [max(WS, key=lambda w: grid[(2024, k, w)]) for k in range(5)]

print("\n" + "=" * 92)
print("arm 별 구간 가중치")
print("=" * 92)
print(f"  {'구간':<11}{'평균격차':>10}{'W0':>7}{'W1':>7}{'W2':>7}{'W3':>7}")
for k in range(5):
    print(f"  {BNAME[k]:<11}{gaps[k]:>10.1f}{0.25:>7.2f}{w1[k]:>7.2f}"
          f"{w2[k]:>7.2f}{w3[k]:>7.2f}")

print("\n" + "=" * 92)
print("fold 전체 ΔBSS (base 대비)")
print("=" * 92)
print(f"  {'arm':<20}" + "".join(f"{v:>12}" for v in FOLDS))
final = []
for name, vec in [("W0_global_0.25", [0.25] * 5), ("W1_bucket_rule", w1),
                  ("W2_monotone", w2), ("W3_2024_optimal", w3)]:
    line, rec = f"  {name:<20}", {"arm": name}
    for vs in FOLDS:
        bb = metrics(Y[vs], B[vs])["bss_raw"]
        bss, q = full_bss(vs, vec)
        d = bss - bb
        null = Y[vs].mean() * (1 - Y[vs].mean())
        dr = (B[vs] - Y[vs]) ** 2 - (q - Y[vs]) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rec[f"dbss_{vs}"], rec[f"t_{vs}"] = d, d / se
        line += f"{d:>+12.2f}"
    rec["weights"] = "|".join(f"{x:.2f}" for x in vec)
    final.append(rec)
    print(line, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v30_bucket_weight.csv", index=False)
fin = pd.DataFrame(final)
fin.to_csv(OUT / "v30_bucket_arms.csv", index=False)
w0r = fin[fin.arm == "W0_global_0.25"].iloc[0]
w1r = fin[fin.arm == "W1_bucket_rule"].iloc[0]
w2r = fin[fin.arm == "W2_monotone"].iloc[0]
print("\n" + "=" * 92)
print(f"W1 − W0   2022 {w1r.dbss_2022-w0r.dbss_2022:+7.2f}   "
      f"2023 {w1r.dbss_2023-w0r.dbss_2023:+7.2f}   "
      f"2024 {w1r.dbss_2024-w0r.dbss_2024:+7.2f}")
print(f"W2 − W0   2022 {w2r.dbss_2022-w0r.dbss_2022:+7.2f}   "
      f"2023 {w2r.dbss_2023-w0r.dbss_2023:+7.2f}   "
      f"2024 {w2r.dbss_2024-w0r.dbss_2024:+7.2f}")
print("\n판정: 세 fold 모두 W0 이상이어야 채택한다. 2024 만 오르면 기각.")
print(f"\nsaved -> {OUT/'v30_bucket_weight.csv'}, {OUT/'v30_bucket_arms.csv'}")
