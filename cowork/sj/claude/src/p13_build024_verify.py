"""P13-A: submit_024 산출물 생성 전 검증.

component_features.py 의 공유 빌더로 P12 결과(프로덕션 836.503 -> +13.81)를
재현하는지 먼저 확인한다. 재현되지 않으면 패키징하지 않는다.

절차
  1. 2019~2023 으로 spec/platoon 생성 -> 성분 4모델 학습 -> 2024 예측
  2. 프로덕션 submit_022 예측에 R 한정 w=0.20 혼합
  3. P12 의 +13.81 (submit_021 기준) 과 정합한지 확인
     주의: 024 는 022 위에 얹으므로 기준선이 836.242 다 (021 은 836.503)
  4. 행 독립성 기계 검증 — 단독 행 예측 == 전체 예측[i]

출력: outputs/p13_build024_verify.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, OUT, CACHE, BASE_PARAMS, load, metrics

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS = 400
W_BLEND = 0.20
EPS = 1e-7
COMPONENTS = ["m", "r", "o", "mr"]

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1

INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
tr, va = season < 2024, season == 2024

train_df = df.loc[tr]
spec = CF.make_spec(train_df)
platoon = CF.make_platoon_table(train_df)
print(f"spec 생성 완료  platoon rows {len(platoon)}  "
      f"league_mean {platoon.attrs['league_mean']:.6f}", flush=True)

feat_all = CF.build(df[INPUT_COLS], spec, platoon)
spec["columns"] = list(feat_all.columns)
X = feat_all.to_numpy(np.float32)
print(f"피처 {X.shape[1]}개  결측 {int(np.isnan(X).sum()):,}", flush=True)

labels = {
    "m": np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan),
    "r": np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan),
    "o": np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan),
}
labels["mr"] = np.where(ok, (labels["m"] == 1) & (labels["r"] == 1), np.nan)


def extrap(arr, tr_mask, vs):
    m = tr_mask & ~np.isnan(arr)
    s = pd.Series(arr[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    slope = (last - float(s.iloc[0])) / (float(s.index[-1]) - float(s.index[0]))
    return float(np.clip(last + slope, 0.005, 0.995))


boosters, base_scores, preds = {}, {}, {}
for tag in COMPONENTS:
    arr = labels[tag]
    bs = extrap(arr, tr, 2024)
    base_scores[tag] = bs
    m = tr & ~np.isnan(arr)
    d_tr = xgb.DMatrix(X[m], label=arr[m])
    d_va = xgb.DMatrix(X[va])
    acc = np.zeros(int(va.sum()))
    bl = []
    for s in SEEDS:
        b = xgb.train({**BASE_PARAMS, "base_score": bs, "seed": s}, d_tr,
                      num_boost_round=N_ROUNDS, verbose_eval=False)
        bl.append(b)
        acc += b.predict(d_va)
    boosters[tag] = bl
    preds[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    print(f"  [{tag:>2}] base_score {bs:.5f}  pred_mean {preds[tag].mean():.5f}  "
          f"actual {np.nanmean(arr[va]):.5f}", flush=True)

p_ie = np.clip(1 - (preds["m"] + preds["r"] - preds["mr"] + preds["o"]),
               EPS, 1 - EPS)
y_va = y_all[va]
print(f"\np_ie Val2024 BSS {metrics(y_va, p_ie)['bss_raw']:.3f}", flush=True)

prod = pd.read_parquet(PROD).set_index("row_id").reindex(
    df.loc[va, "row_id"].to_numpy())
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
rows = []
for col, name in [("submit021_reverse20_s040_tabm", "submit_021"),
                  ("submit019_reconstructed", "submit_019")]:
    p_b = np.clip(prod[col].to_numpy(np.float64), EPS, 1 - EPS)
    bm = metrics(y_va, p_b, game_type=gt)
    null = y_va.mean() * (1 - y_va.mean())
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        q = p_b.copy()
        q[is_r] = w * p_ie[is_r] + (1 - w) * p_b[is_r]
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (p_b - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"base": name, "base_bss": bm["bss_raw"], "w": w,
                     "bss": mm["bss_raw"], "dbss": d, "se_row": se,
                     "t_row": d / se, "r_bss": mm["r_bss"], "f_bss": mm["f_bss"]})
res = pd.DataFrame(rows)
print("\n" + "=" * 88)
print("공유 빌더 재현 확인 (P12: submit_021 base 에서 w=0.20 -> +13.81)")
print("=" * 88)
print(res.round(3).to_string(index=False))

sel = res[(res.base == "submit_021") & (res.w == 0.20)].iloc[0]
print(f"\n재현값 {sel.dbss:+.3f}  vs  P12 +13.810   "
      f"차이 {sel.dbss - 13.810:+.3f}", flush=True)

# --------------------------------------------------- 행 독립성 기계 검증
print("\n" + "=" * 88)
print("행 독립성 — predict(단독 행) == predict(전체)[i]")
print("=" * 88)
rng = np.random.default_rng(20260814)
idx_va = np.where(va)[0]
sample = rng.choice(idx_va, size=200, replace=False)
sub = df.iloc[sample][INPUT_COLS].reset_index(drop=True)
X_sub_single = np.vstack([
    CF.matrix(sub.iloc[[i]], spec, platoon) for i in range(len(sub))])
X_sub_batch = CF.matrix(sub, spec, platoon)
feat_same = bool(np.allclose(np.nan_to_num(X_sub_single, nan=-999),
                             np.nan_to_num(X_sub_batch, nan=-999)))
print(f"  피처 행렬 동일: {feat_same}")

pos = {int(v): i for i, v in enumerate(idx_va)}
maxdiff = 0.0
for tag in COMPONENTS:
    d_single = xgb.DMatrix(X_sub_batch)
    p_single = np.mean([b.predict(d_single) for b in boosters[tag]], axis=0)
    p_full = preds[tag][[pos[int(s)] for s in sample]]
    maxdiff = max(maxdiff, float(np.max(np.abs(p_single - p_full))))
print(f"  성분 예측 최대 절대차: {maxdiff:.3e}")
row_ok = feat_same and maxdiff < 1e-6
print(f"  판정: {'통과' if row_ok else '실패'}")

res.to_csv(OUT / "p13_build024_verify.csv", index=False)
if row_ok and abs(sel.dbss - 13.810) < 1.0:
    ART = CACHE / "submit024_artifacts"
    ART.mkdir(exist_ok=True)
    json.dump(spec, open(ART / "spec_val2024.json", "w"), indent=1)
    platoon.to_csv(ART / "platoon_val2024.csv", index=False)
    json.dump(base_scores, open(ART / "base_scores_val2024.json", "w"), indent=1)
    print(f"\n검증 통과 -> {ART} 에 Val2024 산출물 저장")
    print("다음: 2019~2024 전체 재학습으로 제출용 산출물 생성")
else:
    print("\n검증 실패 — 패키징하지 않는다")
