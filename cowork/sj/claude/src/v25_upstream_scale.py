"""V25: 상류 reverse scale 을 성분 결합과 함께 재최적화한다.

논리
    성분 결합은 프로덕션 예측 '위에' 얹힌다. 그런데 프로덕션 내부 스케일은
    성분 층이 생기기 전에 튜닝된 값이다.
        reverse_matchup_scale 0.475   <- 020(0.55)/021(0.40)의 중립점
    submit_021 메타데이터에도 "downstream R/F 보정과 중복을 줄이려 scale 0.40"
    이라는 기록이 있다. 하류에 강한 층이 붙으면 상류 보정이 과해진다는 논리이며,
    성분 층(+39.25)이 생긴 지금 그 논리가 훨씬 강하게 적용된다.

재학습 없이 잴 수 있다
    reverse20_submission_components.npz 에 reverse20 성분이 있고
        p(s) = p_019 + (s - 0.40) * 0.6085 * reverse20
    로 임의의 scale 을 재구성할 수 있다. 앞서 p_022 재구성이 기록값과
    836.241467 vs 836.242 로 일치함을 확인했다 (020 재구성 최대오차 6e-08).

    성분 라인 예측은 v16_p_ie.npy 에 캐시돼 있다. 다만 그건 submit_027 구성
    (105피처)이므로 submit_029 구성(111피처)으로 다시 만든다.

격자
    s in {0.20 ... 0.70}   x   w in {0.15 ... 0.35}
    세 fold 전부. 판정은 V17/V20 규칙 그대로 - 세 fold 모두 양수인 최대 w,
    그 안에서 최선의 s.

주의
    2022/2023 은 프로덕션 예측이 없어 reverse 성분도 없다. 따라서 s 재최적화는
    2024 에서만 가능하다. 그러면 s 를 게이트 fold 에서 고르는 셈이라 위험하다.
    -> s 는 '개선 여지가 있는가'를 진단만 하고, 채택은 개선폭이 크고 곡선이
       평평할 때만 한다. 뾰족한 최적점은 채택하지 않는다.

출력: outputs/v25_upstream_scale.csv
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
NPZ = PROD.parent / "reverse20_submission_components.npz"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS = 400
SCALES = [0.20, 0.30, 0.40, 0.475, 0.55, 0.65, 0.75]
WS = [0.15, 0.20, 0.25, 0.30, 0.35]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
OUTER_W = 0.6085          # metadata outer_blend.insight_weight
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

# ------------------------------------------- 성분 라인 (submit_029 구성, 111피처)
train_df = df.loc[tr]
spec = CF.make_spec(train_df)
platoon = CF.make_platoon_table(train_df)
bat = CF.make_batter_platoon_table(train_df, {k: v[tr] for k, v in LAB.items()})
cpl = CF.make_count_platoon_table(train_df)
ipl = CF.make_inning_platoon_table(train_df)
X = CF.build(df[INPUT_COLS], spec, platoon, bat, cpl, ipl).to_numpy(np.float32)
print(f"피처 {X.shape[1]}개 (submit_029 구성)", flush=True)


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
    for s_ in SEEDS:
        acc += 0.5 * xgb.train({**prm, "seed": s_}, d_tr, num_boost_round=N_ROUNDS,
                               verbose_eval=False).predict(d_va)
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s_, task_type="GPU", verbose=0)
        c.fit(p_tr)
        acc += 0.5 * c.predict_proba(p_va)[:, 1]
    p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
p_ie = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
np.save(CACHE / "v25_p_ie_029.npy", p_ie)
print(f"성분 라인 완료 {time.time()-t0:.0f}s  단독 "
      f"{metrics(y_all[va], p_ie)['bss_raw']:.2f}", flush=True)

# --------------------------------------------------- reverse scale 재구성
prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
z = np.load(NPZ, allow_pickle=True)
rev = z["reverse20"].astype(np.float64)
p019 = np.clip(prod["submit019_reconstructed"].to_numpy(np.float64), EPS, 1 - EPS)
p021 = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
null = y_va.mean() * (1 - y_va.mean())


def base_at(s):
    return np.clip(p019 + (s - 0.40) * OUTER_W * rev, EPS, 1 - EPS)


# 재구성 검증
chk021 = base_at(0.40)
chk022 = base_at(0.475)
print(f"\n재구성 검증  021 {metrics(y_va, chk021)['bss_raw']:.6f} (기록 836.502924)")
print(f"             022 {metrics(y_va, chk022)['bss_raw']:.6f} (기록 836.242000)")
print(f"             021 최대오차 {np.max(np.abs(chk021-p021)):.3e}", flush=True)

ref = metrics(y_va, p021, game_type=gt)["bss_raw"]
rows = []
print(f"\nreverse scale x w  ->  Val2024 BSS  (기준 submit_021 {ref:.3f})")
print(f"{'s\\w':>8}" + "".join(f"{w:>10.2f}" for w in WS) + f"{'base단독':>11}")
for s in SCALES:
    b = base_at(s)
    solo = metrics(y_va, b)["bss_raw"]
    line = f"{s:>8.3f}"
    for w in WS:
        q = np.clip(w * p_ie + (1 - w) * b, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - ref
        dr = (p021 - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"scale": s, "w": w, "base_solo": solo, "bss": mm["bss_raw"],
                     "dbss": d, "se_row": se, "t_row": d / se})
        line += f"{d:>+10.2f}"
    print(line + f"{solo:>11.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v25_upstream_scale.csv", index=False)
cur = res[(res.scale == 0.475) & (res.w == 0.25)].iloc[0]
best = res.sort_values("dbss", ascending=False).iloc[0]
print(f"\n현행 submit_029 (s=0.475, w=0.25)  ΔBSS {cur.dbss:+.3f}  t_row {cur.t_row:+.2f}")
print(f"격자 최고 (s={best.scale}, w={best.w:.2f})  ΔBSS {best.dbss:+.3f}  "
      f"차이 {best.dbss-cur.dbss:+.3f}")
sub = res[res.w == 0.25].sort_values("scale")
flat = sub["dbss"].max() - sub["dbss"].min()
print(f"\nw=0.25 고정 시 scale 0.20~0.75 구간 ΔBSS 폭 {flat:.2f}  "
      f"({'평평 - 채택 가능' if flat < 5 else '뾰족 - 게이트 fold 과적합 위험'})")
print(f"\nsaved -> {OUT/'v25_upstream_scale.csv'}")
