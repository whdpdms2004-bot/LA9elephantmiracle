"""V79: 최종 블렌드 예측의 평균이 한 해 앞에서 어디에 떨어지는가.

질문
    제출물의 예측 평균이 스모크에서 0.4803 이다. 성분 라인의 base_score 외삽은
    2025 를 0.4704 로 보는데, 블렌드가 base 쪽으로 끌어올린 결과다.
    문제는 스모크의 'test' 가 2024 행에 라벨만 바꾼 것이라 asof_* 피처가
    2024 실적을 담고 있다는 점이다. 진짜 피처를 주면 base 도 따라 내려간다.

    그래서 진짜 피처로 한 해 앞을 예측하는 상황 — fold 2024 — 에서
    최종 블렌드의 평균을 직접 잰다. 실제 제출과 조건이 같다.

읽는 법
    블렌드 평균이 0.4861 근처면 파이프라인이 평균을 맞추고 있는 것이고,
    0.487~0.490 이면 그 편차가 2025 에도 그대로 남는다.
    페널티 = 401,000 x 오프셋^2 이므로 0.008 이면 26점, 0.014 면 78점이다.

    같이 찍는 bss_centered 는 오프셋을 제거한 BSS 다. bss_raw 와의 차이가
    '평균을 맞춰서 번 점수'이고, bss_centered 자체가 '신호로 번 점수'다.

실행: python v79_blend_mean.py
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
N_ROUNDS, F_WEIGHT = 400, 0.20
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
EPS = 1e-7
FOLD = 2024

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
bucket = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
ROW_W = np.where(df["game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)

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

tr, va = season < FOLD, season == FOLD
y = y_all[va]
td = df.loc[tr]
F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
             CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}),
             CF.make_count_platoon_table(td), CF.make_inning_platoon_table(td))
X = F.to_numpy(np.float32)

base = np.clip(pd.read_parquet(PROD).set_index("row_id")
               .reindex(df.loc[va, "row_id"].to_numpy())
               ["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


t0 = time.time()
print(f"fold {FOLD}   실제 평균 {y.mean():.6f}   n={len(y):,}")
print(f"{chr(10)}성분별 base_score 외삽 (학습 시즌 추세를 한 해 전진)")
print(f"  {'성분':>4}{'학습평균':>10}{'외삽 base':>12}{'예측평균':>11}{'실제평균':>11}"
      f"{'오차':>10}")

p, comp_off = {}, {}
Xv, Xc = X[va], np.nan_to_num(X[va], nan=-999.0)
for tag in COMPONENTS:
    arr = LAB[tag]
    mm = tr & ~np.isnan(arr)
    s_ = pd.Series(arr[mm]).groupby(pd.Series(season[mm])).mean().sort_index()
    bs = float(np.clip(float(s_.iloc[-1]) + (float(s_.iloc[-1]) - float(s_.iloc[0]))
                       / (float(s_.index[-1]) - float(s_.index[0])), 0.005, 0.995))
    prm = {**BASE_PARAMS, "base_score": bs, **params_for(float(np.nanmean(arr[mm])))}
    d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=ROW_W[mm], missing=np.nan)
    acc, n = np.zeros(int(va.sum())), 0
    for s in SEEDS:
        acc += xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                         verbose_eval=False).predict(xgb.DMatrix(Xv, missing=np.nan))
        n += 1
    p_tr = Pool(np.nan_to_num(X[mm], nan=-999.0), arr[mm], weight=ROW_W[mm])
    for s in SEEDS:
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        acc += c.predict_proba(Xc)[:, 1]
        n += 1
    p[tag] = np.clip(acc / n, EPS, 1 - EPS)
    act = float(np.nanmean(LAB[tag][va]))
    comp_off[tag] = p[tag].mean() - act
    print(f"  {tag:>4}{float(np.nanmean(arr[mm])):>10.4f}{bs:>12.4f}"
          f"{p[tag].mean():>11.4f}{act:>11.4f}{p[tag].mean()-act:>+10.4f}"
          f"   [{time.time()-t0:.0f}s]", flush=True)

line = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
w = BW[bucket[va]]
blend = np.clip(w * line + (1 - w) * base, EPS, 1 - EPS)

print(f"{chr(10)}{'='*92}")
print("최종 예측의 평균이 어디에 떨어지는가")
print("=" * 92)
print(f"  {'예측':<14}{'평균':>11}{'오프셋':>11}{'페널티':>10}"
      f"{'BSS':>11}{'오프셋제거':>12}{'평균으로번점수':>16}")
rows = []
for nm, q in [("프로덕션 base", base), ("성분 라인", line), ("블렌드 (제출)", blend)]:
    m = metrics(y, q)
    gain = m["bss_raw"] - m["bss_centered"]
    print(f"  {nm:<14}{m['pred_mean']:>11.6f}{m['offset']:>+11.6f}"
          f"{401000*m['offset']**2:>10.1f}{m['bss_raw']:>11.2f}"
          f"{m['bss_centered']:>12.2f}{gain:>+16.2f}")
    rows.append({"name": nm, **{k: m[k] for k in
                 ("pred_mean", "offset", "bss_raw", "bss_centered")}})

pd.DataFrame(rows).to_csv(OUT / "v79_blend_mean.csv", index=False)

off = blend.mean() - y.mean()
print(f"{chr(10)}{'='*92}")
print("2025 로 넘길 때의 함의")
print("=" * 92)
print(f"  fold 2024 에서 블렌드 오프셋은 {off:+.6f} 다.")
print(f"  같은 편차가 2025 에도 남으면 페널티 약 {401000*off**2:.1f} 점.")
print(f"  참고 — 스모크(2024 행에 라벨만 2025) 출력 평균은 0.4803 이었다.")
print(f"         진짜 피처를 주면 여기 값처럼 떨어진다는 것이 이 실험의 요지다.")
print(f"{chr(10)}  '평균으로 번 점수' 가 크면 그 구성의 BSS 는 신호가 아니라")
print(f"  평균 정렬에서 나온 것이다. 다른 fold 로 옮겨갈 때 같이 가지 않는다.")
print(f"{chr(10)}총 {time.time()-t0:.0f}초   saved -> {OUT / 'v79_blend_mean.csv'}")
