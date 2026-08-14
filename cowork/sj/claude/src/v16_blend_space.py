"""V16: 결합 공간 — 확률 공간 vs 로짓 공간.

동기
    프로덕션 metadata 는 blend_space: "logit" 이다. 자기 내부 블렌드를 로짓에서 한다.
    그런데 sj 성분 결합은 확률 공간에서 하고 있다.
        확률: w*p_ie + (1-w)*p_prod
        로짓: sigmoid(w*logit(p_ie) + (1-w)*logit(p_prod))
    둘은 다른 연산이다. 로짓 결합은 극단값을 덜 끌어당기고 기하평균에 가깝다.

    구분선 기준으로는 '새 정보'가 아니라 '같은 정보의 다른 결합'이라 기대는 낮다.
    다만 프로덕션이 그 공간을 쓴다는 근거가 있고 비용이 1회전이라 잰다.

같이 보는 것
    - 로짓 결합에서의 최적 w (확률 공간과 다를 수 있다)
    - 예측 평균 편향. 팀 벌점 401,000 x 오차^2 으로 환산해 남은 여지를 확인한다
      (hw v8 에서 +1.04%p = 43 BSS 였다. sj 라인은 얼마인가)
    - p_ie 를 캐시에 저장해 이후 라운드가 재학습 없이 쓰도록 한다

판정: Val2024 전체 BSS, 프로덕션 836.503 대비.
출력: outputs/v16_blend_space.csv, cache/v16_p_ie.npy
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
N_ROUNDS = 400
WS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
tr, va = season < 2024, season == 2024
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


def logit(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


train_df = df.loc[tr]
spec = CF.make_spec(train_df)
platoon = CF.make_platoon_table(train_df)
bat_platoon = CF.make_batter_platoon_table(train_df, {k: v[tr] for k, v in LAB.items()})
X = CF.build(df[INPUT_COLS], spec, platoon, bat_platoon).to_numpy(np.float32)
print(f"피처 {X.shape[1]}개 (V12 G4 = submit_027 구성)", flush=True)


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


t0 = time.time()
p = {}
for tag in COMPONENTS:
    arr = LAB[tag]
    m = tr & ~np.isnan(arr)
    prm = {**BASE_PARAMS, "base_score": extrap(arr),
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
p_ie = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
np.save(CACHE / "v16_p_ie.npy", p_ie)
print(f"성분 라인 완료 {time.time()-t0:.0f}s  단독 BSS "
      f"{metrics(y_all[va], p_ie)['bss_raw']:.2f}", flush=True)

prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())

rows = []
print(f"\n{'space':<10}" + "".join(f"w{w:<7.2f}" for w in WS), flush=True)
for space in ("prob", "logit"):
    line = f"{space:<10}"
    for w in WS:
        q = (w * p_ie + (1 - w) * p_prod if space == "prob"
             else sigmoid(w * logit(p_ie) + (1 - w) * logit(p_prod)))
        q = np.clip(q, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"space": space, "w": w, "bss": mm["bss_raw"], "dbss": d,
                     "se_row": se, "t_row": d / se, "pred_mean": mm["pred_mean"],
                     "r_bss": mm["r_bss"], "f_bss": mm["f_bss"]})
        line += f"{d:>+8.2f}"
    print(line, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v16_blend_space.csv", index=False)

print("\n예측 평균 편향과 남은 여지 (팀 벌점 401,000 x 오차^2)")
tgt = y_va.mean()
for space in ("prob", "logit"):
    r = res[(res.space == space) & (res.w == 0.20)].iloc[0]
    err = r.pred_mean - tgt
    print(f"  {space:<6} w=0.20  pred_mean {r.pred_mean:.6f}  실제 {tgt:.6f}  "
          f"편향 {err*100:+.4f}%p  벌점 {100000*err**2/null:.2f} BSS")
print(f"  참고: 프로덕션 단독 pred_mean {bm['pred_mean']:.6f}  "
      f"벌점 {100000*(bm['pred_mean']-tgt)**2/null:.2f} BSS")

pb = res[(res.space == "prob") & (res.w == 0.20)].iloc[0]
lb = res[res.space == "logit"].sort_values("dbss", ascending=False).iloc[0]
print(f"\n확률공간 w=0.20 (submit_027)  ΔBSS {pb.dbss:+.3f}  t_row {pb.t_row:+.2f}")
print(f"로짓공간 최고 w={lb.w:.2f}       ΔBSS {lb.dbss:+.3f}  t_row {lb.t_row:+.2f}  "
      f"차이 {lb.dbss-pb.dbss:+.3f}")
print(f"\nsaved -> {OUT/'v16_blend_space.csv'}, {CACHE/'v16_p_ie.npy'}")
