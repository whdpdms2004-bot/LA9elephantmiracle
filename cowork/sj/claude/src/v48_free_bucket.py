"""V48: 구간별 가중치 벡터를 자유롭게 푼다 — 세 fold 를 다 보여주고. (CPU 전용)

여기까지 온 경로
    V44  2024 캐시 예측으로 탐욕 앙상블      890.27  (+53.8)
    V45  해석적 NNLS, K 제약                 K=1 이 이미 상한에 근접
    V46  분해 — 이득의 대부분은 '가중치'      라인 교체는 +1.98 뿐
    V47  세 fold 풀링으로 다시 풀기          w 가 0.230 으로 회귀
    (즉석) 라인 단순 평균                    세 fold 전부에서 이득 0

    "여러 조합 시험해보고 결과로만 평가해" 를 그대로 했더니
    2024 한 fold 에서 커 보였던 것들이 세 fold 를 보여주면 전부 사라졌다.
    남은 자유도는 '구간별 가중치 벡터' 하나다. V30 에서 규칙으로 골랐던 것을
    이번엔 규칙 없이 최적화로 푼다.

푸는 것
    w = (w0, w1, w2, w3, w4)   asof_pitcher_n 구간 [0,100,500,2000,4000)
    목적함수 두 가지를 따로 최적화한다.
        SUM    세 fold ΔBSS 합 최대
        WORST  세 fold 중 최악 ΔBSS 최대 (minimax)
    규칙으로 자르지 않는다. 최적화가 알아서 답을 낸다.

    비교군: 전역 0.20 / 0.23 / 0.25, V30 규칙해 (0.25 x4, 0.45),
            현행 submit_030 (전역 0.25), submit_031 (V30 규칙해)

출력: outputs/v48_free_bucket.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
FOLDS = [2022, 2023, 2024]
CUTS = [100, 500, 2000, 4000]
BNAME = ["0-99", "100-499", "500-1999", "2000-3999", "4000+"]
EPS = 1e-7

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)

models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
B, LN, Y, BK, NL = {}, {}, {}, {}, {}
for f in FOLDS:
    fid = df.loc[season == f, "row_id"].to_numpy()
    if f == 2024:
        pr = pd.read_parquet(PROD).set_index("row_id").reindex(fid)
        B[f] = np.clip(pr["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                       EPS, 1 - EPS)
    else:
        acc, c = None, 0
        for mn in models:
            p = OOF_DIR / f"{mn}_fold{f}.parquet"
            if p.exists():
                v = pd.read_parquet(p).set_index("row_id").reindex(fid)["prediction"].to_numpy()
                acc = v if acc is None else acc + v
                c += 1
        B[f] = np.clip(acc / c, EPS, 1 - EPS)
    LN[f] = np.clip(np.load(CACHE / f"v38_R0_current_{f}.npy"), EPS, 1 - EPS)
    Y[f] = y_all[season == f]
    BK[f] = bucket_all[season == f]
    NL[f] = Y[f].mean() * (1 - Y[f].mean())

REF = {f: 100000 * (NL[f] - ((B[f] - Y[f]) ** 2).mean()) / NL[f] for f in FOLDS}


def delta(f, wvec):
    w = np.asarray(wvec, float)[BK[f]]
    q = np.clip(w * LN[f] + (1 - w) * B[f], EPS, 1 - EPS)
    return 100000 * (NL[f] - ((q - Y[f]) ** 2).mean()) / NL[f] - REF[f]


def squash(u):
    return 0.85 / (1 + np.exp(-np.asarray(u)))


def opt(kind, starts=6):
    best = None
    for i in range(starts):
        u0 = np.log(np.full(5, 0.25) / (0.85 - 0.25)) + \
            np.random.default_rng(7000 + i).normal(0, 0.6, 5)
        if kind == "sum":
            def obj(u):
                return -sum(delta(f, squash(u)) for f in FOLDS)
        else:
            def obj(u):
                return -min(delta(f, squash(u)) for f in FOLDS)
        r = minimize(obj, u0, method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    return squash(best.x)


print("fold 별 base / 성분단독")
for f in FOLDS:
    solo = 100000 * (NL[f] - ((LN[f] - Y[f]) ** 2).mean()) / NL[f]
    print(f"  {f}  base {REF[f]:9.2f}   성분단독 {solo:9.2f}")

ARMS = [("전역 0.20", [0.20] * 5), ("전역 0.23", [0.23] * 5),
        ("전역 0.25 (submit_030)", [0.25] * 5),
        ("V30 규칙해 (submit_031)", [0.25, 0.25, 0.25, 0.25, 0.45]),
        ("자유해 SUM", opt("sum")), ("자유해 WORST", opt("worst"))]

rows = []
print(f"{chr(10)}{'='*104}")
print(f"{'arm':<24}" + "".join(f"{b:>10}" for b in BNAME)
      + "".join(f"{f'Δ{f}':>11}" for f in FOLDS) + f"{'최악':>9}{'합':>9}")
print("=" * 104)
for name, wv in ARMS:
    ds = [delta(f, wv) for f in FOLDS]
    rows.append({"arm": name, **{f"w_{b}": v for b, v in zip(BNAME, wv)},
                 **{f"d{f}": d for f, d in zip(FOLDS, ds)},
                 "worst": min(ds), "sum": sum(ds)})
    print(f"{name:<24}" + "".join(f"{v:>10.3f}" for v in wv)
          + "".join(f"{d:>+11.2f}" for d in ds)
          + f"{min(ds):>+9.2f}{sum(ds):>+9.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v48_free_bucket.csv", index=False)
bs = res.sort_values("sum", ascending=False).iloc[0]
bw = res.sort_values("worst", ascending=False).iloc[0]
cur = res[res.arm.str.contains("submit_031")].iloc[0]
print(f"{chr(10)}합 최고    {bs.arm:<24} 합 {bs['sum']:+8.2f}  최악 {bs.worst:+8.2f}")
print(f"최악 최고  {bw.arm:<24} 합 {bw['sum']:+8.2f}  최악 {bw.worst:+8.2f}")
print(f"현행 031   {cur.arm:<24} 합 {cur['sum']:+8.2f}  최악 {cur.worst:+8.2f}")
print(f"{chr(10)}saved -> {OUT/'v48_free_bucket.csv'}")
