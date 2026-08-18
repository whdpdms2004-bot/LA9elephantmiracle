"""V18: 극단 예측 cap — 2023 붕괴를 막고 w 를 풀 수 있는지.

동기
    V17 결론: w=0.20 을 못 넘는 이유는 오직 2023 이다.
        w      2022     2023     2024
        0.20  +35.15   +4.58   +23.80   <- 세 fold 모두 양수인 최대값
        0.25  +41.24  -12.36   +26.51
        0.35  +50.18  -67.92   +28.01
    2024 에서 +4.2 를 더 먹으려면 2023 에서 -72 를 감수해야 해서 기대값이 음수다.

    찬우의 극단 cap 이 정확히 그 붕괴를 막는 장치다 (팀 T1).
        정상 연도 -10점, 레짐 변화 시 +800점 방어
        2023 붕괴 원인은 예측 상위 10분위 = 저표본 투수 과신

    cap 이 2023 하방을 줄이면 w 를 올릴 수 있고, 그러면 V17 의 기대값 계산이
    뒤집힌다. cap 자체보다 'cap 이 다른 이득을 여는가'가 이번 질문이다.

설계
    cap 은 최종 예측에 적용한다. 두 형태를 본다.
        abs   고정 상하한  [lo, 1-lo]
        quant 학습 시즌 예측 분포의 분위수로 상하한
    저표본 투수에 한정하는 변형도 본다 (찬우 진단이 저표본을 지목했다).
        cond  asof_pitcher_n < N 인 행에만 cap

    각 cap 아래에서 w 그리드를 다시 그려 세 fold 가 모두 양수인 최대 w 를 찾는다.

판정: '세 fold 모두 양수인 최대 w' 가 0.20 보다 커지는가. 커지면 그만큼이 순이득이다.
출력: outputs/v18_extreme_cap.csv
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
WS = [0.20, 0.25, 0.30, 0.35, 0.40]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
FOLDS = [2022, 2023, 2024]
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
asof_n = df["asof_pitcher_n"].to_numpy(np.float64)
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


t0 = time.time()
IE, TRAIN_PRED = {}, {}
for vs in FOLDS:
    tr, va = season < vs, season == vs
    train_df = df.loc[tr]
    spec = CF.make_spec(train_df)
    platoon = CF.make_platoon_table(train_df)
    bat = CF.make_batter_platoon_table(train_df, {k: v[tr] for k, v in LAB.items()})
    X = CF.build(df[INPUT_COLS], spec, platoon, bat).to_numpy(np.float32)
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr),
               **params_for(float(np.nanmean(arr[tr])))}
        d_tr = xgb.DMatrix(X[m], label=arr[m]); d_va = xgb.DMatrix(X[va])
        p_tr, p_va = Pool(X[m], arr[m]), Pool(X[va])
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
    IE[vs] = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
    print(f"  fold {vs} 성분 라인 완료  단독 "
          f"{metrics(y_all[va], IE[vs])['bss_raw']:9.2f}  [{time.time()-t0:.0f}s]",
          flush=True)


def apply_cap(p, mode, param, n_va):
    if mode == "none":
        return p
    if mode == "abs":
        return np.clip(p, param, 1 - param)
    if mode == "quant":
        lo, hi = np.quantile(p, param), np.quantile(p, 1 - param)
        return np.clip(p, lo, hi)
    if mode == "cond":                      # 저표본 투수에만 cap
        lo, hi = param
        out = p.copy()
        m = n_va < 1000
        out[m] = np.clip(out[m], lo, 1 - lo)
        return out
    raise ValueError(mode)


CAPS = [("none", None), ("abs", 0.35), ("abs", 0.40), ("abs", 0.42),
        ("quant", 0.02), ("quant", 0.05), ("cond", (0.38, None)),
        ("cond", (0.42, None))]

rows = []
for mode, param in CAPS:
    tag = f"{mode}" + ("" if param is None else
                       f"_{param if not isinstance(param, tuple) else param[0]}")
    for vs in FOLDS:
        va = season == vs
        y_va, b, n_va = y_all[va], strong[vs], asof_n[va]
        bm = metrics(y_va, b)
        null = y_va.mean() * (1 - y_va.mean())
        for w in WS:
            q = np.clip(w * IE[vs] + (1 - w) * b, EPS, 1 - EPS)
            q = apply_cap(q, mode, param, n_va)
            mm = metrics(y_va, q)
            rows.append({"cap": tag, "fold": vs, "w": w,
                         "bss": mm["bss_raw"],
                         "dbss": mm["bss_raw"] - bm["bss_raw"]})

res = pd.DataFrame(rows)
res.to_csv(OUT / "v18_extreme_cap.csv", index=False)

print("\n" + "=" * 86)
print("cap 별 fold x w 의 ΔBSS  (세 fold 모두 양수인 최대 w 를 찾는다)")
print("=" * 86)
print(f"{'cap':<14}{'fold':>6}" + "".join(f"{w:>10.2f}" for w in WS))
best_w = {}
for tag in res["cap"].unique():
    for vs in FOLDS:
        s = res[(res.cap == tag) & (res.fold == vs)].sort_values("w")
        print(f"{tag if vs == FOLDS[0] else '':<14}{vs:>6}"
              + "".join(f"{v:>+10.2f}" for v in s["dbss"]))
    piv = res[res.cap == tag].pivot(index="w", columns="fold", values="dbss")
    okw = piv[(piv > 0).all(axis=1)]
    best_w[tag] = (okw.index.max() if len(okw) else None,
                   float(okw.loc[okw.index.max(), 2024]) if len(okw) else None)
    print()

print("=" * 86)
print("세 fold 모두 양수인 최대 w 와 그때의 Val2024 ΔBSS")
print("=" * 86)
base = best_w.get("none", (None, None))
for tag, (w, d24) in best_w.items():
    mark = ""
    if w is not None and base[0] is not None and d24 is not None:
        mark = f"   현행 대비 {d24 - base[1]:+.2f}" if tag != "none" else "   <- 현행"
    print(f"  {tag:<14} 최대 w = {w if w else '없음'}"
          f"   Val2024 {f'{d24:+.2f}' if d24 is not None else '-'}{mark}")
print(f"\nsaved -> {OUT/'v18_extreme_cap.csv'}")
