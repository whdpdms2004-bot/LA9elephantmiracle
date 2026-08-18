"""V36: 합성 라인의 사후 보정 — 절편만이 아니라 형태까지. (CPU 전용)

V32 가 남긴 질문
    2023 붕괴는 레벨이 아니라 '형태'다. 레벨은 절편 하나로 잡았고(V26 K3, +1.37)
    형태는 아직 아무것도 안 했다. 단조 재보정이 형태를 잡을 수 있는가.

V26 과의 차이
    V26 K1 은 '성분별 계수 6개'를 적합해서 과적합했다(c_mr −1.00 -> −2.54, −11점).
    여기서는 성분을 건드리지 않고 합성 결과 p_ie 하나에 단조 사상만 씌운다.
    파라미터가 1~2개(Platt/beta)거나 단조 제약이 있다(isotonic).

적합 프로토콜
    순방향 OOF 2022+2023 에서 적합 -> 2024 에서 평가.  (게이트 fold 미사용)
    역방향 2023+2024 -> 2022 도 같이 재서 사상이 fold 간에 안정한지 본다.
    안정하지 않으면 2024 에서 좋아도 채택하지 않는다.

arm
    C0 identity      현행
    C1 intercept     확률에 상수 (V26 K3)
    C2 platt         로짓 아핀  a*z + b
    C3 isotonic      단조 비모수
    C4 beta          a*log(p) − b*log(1−p) + c

출력: outputs/v36_composite_recalib.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load, metrics

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
FOLDS = [2022, 2023, 2024]
BW = np.array([0.25, 0.25, 0.25, 0.25, 0.45])       # V30 W1
CUTS = [100, 500, 2000, 4000]
EPS = 1e-7

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)

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

P = {f: np.load(CACHE / f"v30_pie_{f}.npy") for f in FOLDS}
Y = {f: y_all[season == f] for f in FOLDS}
BK = {f: bucket_all[season == f] for f in FOLDS}


def lg(p):
    return np.log(p / (1 - p))


def fit_map(kind, p, y):
    if kind == "identity":
        return lambda q: q
    if kind == "intercept":
        c = float(np.mean(y - p))
        return lambda q, c=c: np.clip(q + c, EPS, 1 - EPS)
    if kind == "platt":
        m = LogisticRegression(C=1e6, max_iter=1000).fit(lg(p).reshape(-1, 1), y)
        a, b = float(m.coef_[0][0]), float(m.intercept_[0])
        return lambda q, a=a, b=b: np.clip(1 / (1 + np.exp(-(a * lg(q) + b))),
                                           EPS, 1 - EPS)
    if kind == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
        return lambda q, ir=ir: np.clip(ir.predict(q), EPS, 1 - EPS)
    if kind == "beta":
        Xb = np.column_stack([np.log(p), -np.log(1 - p)])
        m = LogisticRegression(C=1e6, max_iter=1000).fit(Xb, y)
        a, b = m.coef_[0]
        c = float(m.intercept_[0])
        return lambda q, a=a, b=b, c=c: np.clip(
            1 / (1 + np.exp(-(a * np.log(q) - b * np.log(1 - q) + c))), EPS, 1 - EPS)
    raise ValueError(kind)


KINDS = ["identity", "intercept", "platt", "isotonic", "beta"]
SETUPS = [("정방향 22+23 -> 24", [2022, 2023], 2024),
          ("역방향 23+24 -> 22", [2023, 2024], 2022),
          ("역방향 22+24 -> 23", [2022, 2024], 2023)]

rows = []
for label, fit_folds, ev in SETUPS:
    pf = np.concatenate([P[f] for f in fit_folds])
    yf = np.concatenate([Y[f] for f in fit_folds])
    y, b, bk = Y[ev], strong[ev], BK[ev]
    w = BW[bk]
    ref_solo = metrics(y, P[ev])["bss_raw"]
    ref_mix = metrics(y, np.clip(w * P[ev] + (1 - w) * b, EPS, 1 - EPS))["bss_raw"]
    print(f"\n{label}    성분단독 {ref_solo:9.2f}   결합 {ref_mix:9.2f}")
    print(f"  {'arm':<12}{'단독':>11}{'Δ단독':>10}{'결합':>11}{'Δ결합':>10}")
    for kind in KINDS:
        g = fit_map(kind, pf, yf)
        q = g(P[ev])
        solo = metrics(y, q)["bss_raw"]
        mix = metrics(y, np.clip(w * q + (1 - w) * b, EPS, 1 - EPS))["bss_raw"]
        rows.append({"setup": label, "eval_fold": ev, "arm": kind, "solo": solo,
                     "d_solo": solo - ref_solo, "mix": mix, "d_mix": mix - ref_mix})
        print(f"  {kind:<12}{solo:>11.2f}{solo-ref_solo:>+10.2f}"
              f"{mix:>11.2f}{mix-ref_mix:>+10.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v36_composite_recalib.csv", index=False)

print("\n" + "=" * 76)
print("세 setup 전부에서 결합 이득이 양수인 arm 만 채택 가능하다")
print("=" * 76)
piv = res.pivot_table(index="arm", columns="eval_fold", values="d_mix")
piv["min"] = piv.min(axis=1)
print(piv.round(2).sort_values("min", ascending=False).to_string())
best = piv.sort_values("min", ascending=False).index[0]
if piv.loc[best, "min"] > 0:
    print(f"\n채택 후보: {best}   최악 fold 이득 {piv.loc[best,'min']:+.2f}")
else:
    print("\n전부 어느 한 fold 에서 음수 -> 기각. 형태 보정은 fold 간 전이가 없다.")
print(f"\nsaved -> {OUT/'v36_composite_recalib.csv'}")
