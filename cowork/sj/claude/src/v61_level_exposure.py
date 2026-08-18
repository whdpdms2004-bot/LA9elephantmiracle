"""V61: 레벨 노출을 정확히 재고, 예보 기반 보정이 fold 에서 통하는지 확인한다.

왜 지금 이걸 보는가
    Public ~= Val2024 + 90 이 네 제출에서 일관된다.
        023  836.5 -> 916.70    024  850.3 -> 945.0
        030  877.0 -> 964       032  879.1 -> 979
    1200 이면 Val2024 ~1110 이 필요하고 지금 879 다. 내부 +231.

    그런데 Phase-0 P0-4 가 판별력 한계를 이미 재놨다 — asof_* 19개가 AUC 의 거의
    전부(0.5202 -> 0.5437)이고 ID 를 더 줘도 +0.0009 다. AUC ~0.55 가 구속조건이다.
    따라서 남은 큰 레버는 판별력이 아니라 '레벨'이다.

    팀 벌점 공식  손실 ~= 401,000 x (pred_mean - target_mean)^2
        0.5%p -> 10   1.0%p -> 40   1.6%p -> 103   2.0%p -> 160

시즌 추이 (실측)
    2019 .5647  2020 .5327  2021 .5328  2022 .5289  2023 .5000  2024 .4861
    최근 3년 변화 -0.0039 / -0.0289 / -0.0139, 평균 -0.0156 -> 2025 ~ 0.4705
    찬우 params.json 도 target_rate 0.47353 로 같은 구간을 본다.

    그런데 submit_033 은 2024형 입력에서 pred_mean 0.4806 을 낸다.

반례도 있다
    submit_023 = 022 + logit offset -> Public 916.70
    submit_024 = 022 + 성분결합      -> Public 945.0
    offset 단독은 도움이 안 됐다. 그래서 임의로 밀지 않고 fold 로 검증한다.

측정
    fold V 마다 season < V 만으로 2025 식 예보 f_V 를 만들고(외삽 3종),
    최종 예측을 로짓에서 이동시켜 pred_mean = f_V 로 맞춘 뒤 BSS 변화를 본다.
    오라클(실제 평균에 맞추기)도 같이 재서 '완벽한 레벨'의 상한을 본다.

    예보가 세 fold 모두에서 이득이면 2025 에 적용할 근거가 된다.
    한 fold 라도 크게 손해면 적용하지 않는다.

출력: outputs/v61_level_exposure.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load, metrics

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
FOLDS = [2023, 2024]
EPS = 1e-7

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
rate = df.groupby("season")[TARGET].mean()
print("시즌별 실제 성공률")
print("  " + "  ".join(f"{int(s)} {v:.4f}" for s, v in rate.items()))


def forecasts(upto):
    """season < upto 만으로 만든 다음 시즌 예보 3종."""
    r = rate[rate.index < upto]
    s = r.index.to_numpy(float)
    v = r.to_numpy()
    out = {}
    out["전체선형"] = float(np.polyval(np.polyfit(s, v, 1), upto))
    if len(v) >= 3:
        out["최근3선형"] = float(np.polyval(np.polyfit(s[-3:], v[-3:], 1), upto))
    out["차분외삽"] = float(v[-1] + (v[-1] - v[-2]))
    out["harness"] = float(v[-1] + (v[-1] - v[0]) / (s[-1] - s[0]))
    out["직전유지"] = float(v[-1])
    return {k: float(np.clip(x, 0.3, 0.7)) for k, x in out.items()}


models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
BASE_P = {}
for f in FOLDS:
    fid = df.loc[season == f, "row_id"].to_numpy()
    if f == 2024:
        pr = pd.read_parquet(PROD).set_index("row_id").reindex(fid)
        BASE_P[f] = np.clip(pr["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                            EPS, 1 - EPS)
    else:
        acc, c = None, 0
        for mn in models:
            p = OOF_DIR / f"{mn}_fold{f}.parquet"
            if p.exists():
                v = pd.read_parquet(p).set_index("row_id").reindex(fid)["prediction"].to_numpy()
                acc = v if acc is None else acc + v
                c += 1
        BASE_P[f] = np.clip(acc / c, EPS, 1 - EPS)


def shift_to(p, target):
    z = np.log(p / (1 - p))

    def fn(c):
        return float((1 / (1 + np.exp(-(z + c)))).mean() - target)

    lo, hi = -6.0, 6.0
    if fn(lo) * fn(hi) > 0:
        return p, 0.0
    c = brentq(fn, lo, hi, xtol=1e-12)
    return np.clip(1 / (1 + np.exp(-(z + c))), EPS, 1 - EPS), c


rows = []
for f in FOLDS:
    va = season == f
    y = y_all[va]
    b = BASE_P[f]
    pie = np.clip(np.load(CACHE / f"v57_F1_tierE_v1_{f}.npy"), EPS, 1 - EPS)
    w = BW[bucket_all[va]]
    q = np.clip(w * pie + (1 - w) * b, EPS, 1 - EPS)
    ref = metrics(y, q)["bss_raw"]
    actual = float(y.mean())
    fc = forecasts(f)
    print(f"{chr(10)}fold {f}   실제 {actual:.5f}   현재 예측평균 {q.mean():.5f}   "
          f"편차 {(q.mean()-actual)*100:+.3f}%p   BSS {ref:.2f}")
    print(f"  {'방법':<12}{'예보':>9}{'오차%p':>9}{'이동후 BSS':>12}{'ΔBSS':>10}")
    qo, _ = shift_to(q, actual)
    print(f"  {'오라클':<12}{actual:>9.5f}{0.0:>9.3f}{metrics(y, qo)['bss_raw']:>12.2f}"
          f"{metrics(y, qo)['bss_raw']-ref:>+10.2f}")
    rows.append({"fold": f, "method": "oracle", "target": actual, "err_pp": 0.0,
                 "bss": metrics(y, qo)["bss_raw"],
                 "dbss": metrics(y, qo)["bss_raw"] - ref})
    for name, t in fc.items():
        qs, _ = shift_to(q, t)
        m = metrics(y, qs)["bss_raw"]
        rows.append({"fold": f, "method": name, "target": t,
                     "err_pp": (t - actual) * 100, "bss": m, "dbss": m - ref})
        print(f"  {name:<12}{t:>9.5f}{(t-actual)*100:>+9.3f}{m:>12.2f}{m-ref:>+10.2f}")

res = pd.DataFrame(rows)
res.to_csv(OUT / "v61_level_exposure.csv", index=False)

print(f"{chr(10)}{'='*72}{chr(10)}fold 별 ΔBSS{chr(10)}{'='*72}")
piv = res.pivot_table(index="method", columns="fold", values="dbss")
piv["최악"] = piv.min(axis=1)
print(piv.round(2).sort_values("최악", ascending=False).to_string())

print(f"{chr(10)}2025 예보 (season <= 2024 전부 사용)")
f25 = forecasts(2025)
for k, v in f25.items():
    print(f"  {k:<12}{v:.5f}")
cur = 0.480628      # submit_033 6단계 검증 예측 평균 (2024형 입력)
print(f"{chr(10)}submit_033 예측평균(2024형 입력) {cur:.5f}")
for k, v in f25.items():
    d = (cur - v)
    print(f"  vs {k:<10} 오차 {d*100:+.3f}%p   벌점 추정 {401000*d*d:>7.1f} BSS")
print(f"{chr(10)}주의: 위 벌점은 '2025 입력에서도 같은 평균이 나온다'는 가정이다.")
print(f"실제로는 asof_* 가 2025 수준으로 내려가 예측도 함께 내려간다.")
print(f"{chr(10)}saved -> {OUT/'v61_level_exposure.csv'}")
