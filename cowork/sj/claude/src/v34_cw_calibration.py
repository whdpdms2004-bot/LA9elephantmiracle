"""V34: 찬우(cw) 의 보정 계열을 sj 라인에 적용한다. (CPU 전용)

출처
    cowork/cw/submit_v8_mlp.zip 의 model/params.json 에서 읽었다.
        target_rate      0.47353
        logit_center_C0  −0.0496 / −0.0492 / −0.0528   (계열별)
        logit_target_C1  −0.10598
        logit_scale      0.85 (A계열) / 1.0 (B,C계열)
        cap              0.15 / 0.20
        floor/ceil       0.15 / 0.85

    구성을 복원하면
        z  = logit(p) − C0        로짓 중심을 0 으로
        z' = z × scale + C1       스케일 축소 후 목표 레벨로 이동
        p' = sigmoid(z')
        p'' = clip(p', target − cap, target + cap)

    핵심은 scale < 1 이다. 전체 로짓 폭을 줄여 base rate 쪽으로 수축시킨다.

V18 과 무엇이 다른가
    V18 은 '극단값 cap' 만 쟀고 기각했다. cap 은 꼬리만 자른다.
    scale 은 모든 행을 비례 축소한다. V32 에서 2023 붕괴가 '형태' 문제로
    판명됐으므로(레벨 보정으로 안 풀림) 형태를 건드리는 scale 은 다른 후보다.

    찬우는 blend 비율을 리더보드 점수로 정했다(params.json 의 blend_source
    "LB-measured"). 그 부분은 가져오지 않는다 (RULES §2).
    가져오는 것은 보정 형태뿐이고, 파라미터는 세 fold 규칙으로 정한다.

측정
    성분 라인과 최종 결합 예측 양쪽에 적용해 본다.
    target 은 오라클이 아니라 harness.forecast_base_rate 외삽값을 쓴다.

출력: outputs/v34_cw_calibration.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load, metrics, forecast_base_rate

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
FOLDS = [2022, 2023, 2024]
SCALES = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
CAPS = [0.10, 0.15, 0.20, 1.00]
BW = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.45}      # V30 W1
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


def calibrate(p, target, scale, cap):
    z = np.log(p / (1 - p))
    z = (z - z.mean()) * scale + np.log(target / (1 - target))
    q = 1 / (1 + np.exp(-z))
    return np.clip(q, max(EPS, target - cap), min(1 - EPS, target + cap))


rows = []
for vs in FOLDS:
    va = season == vs
    y, b, bk = y_all[va], strong[vs], bucket_all[va]
    tgt = forecast_base_rate(df, season < vs, vs)
    pie = np.load(CACHE / f"v30_pie_{vs}.npy")
    w = np.array([BW[k] for k in range(5)])[bk]
    mix = np.clip(w * pie + (1 - w) * b, EPS, 1 - EPS)
    ref = metrics(y, mix)["bss_raw"]
    print(f"\nfold {vs}   외삽 target {tgt:.5f}  실제 {y.mean():.5f}  "
          f"(오차 {(tgt-y.mean())*100:+.3f}%p)   기준 {ref:.2f}")
    for lname, line in [("성분", pie), ("결합", mix)]:
        print(f"  {lname:<5}" + "".join(f"{f'cap{c:g}':>11}" for c in CAPS))
        for sc in SCALES:
            out = f"  s={sc:<4.2f}"
            for cp in CAPS:
                q = calibrate(line, tgt, sc, cp)
                fin = (q if lname == "성분" else None)
                if lname == "성분":
                    fin = np.clip(w * q + (1 - w) * b, EPS, 1 - EPS)
                else:
                    fin = q
                d = metrics(y, fin)["bss_raw"] - ref
                rows.append({"fold": vs, "apply_to": lname, "scale": sc,
                             "cap": cp, "dbss": d})
                out += f"{d:>+11.2f}"
            print(out, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v34_cw_calibration.csv", index=False)

print("\n" + "=" * 84)
print("세 fold 모두 양수인 조합")
print("=" * 84)
piv = res.pivot_table(index=["apply_to", "scale", "cap"], columns="fold",
                      values="dbss")
good = piv[(piv > 0).all(axis=1)]
if len(good):
    good = good.assign(min_fold=good.min(axis=1)).sort_values("min_fold",
                                                             ascending=False)
    print(good.head(12).to_string())
    b = good.index[0]
    print(f"\n최선: {b[0]} 라인, scale {b[1]}, cap {b[2]}   "
          f"세 fold 최소 이득 {good['min_fold'].iloc[0]:+.2f}")
else:
    print("  없음. 세 fold 를 동시에 만족하는 조합이 없다 -> 기각.")
    print(piv.assign(min_fold=piv.min(axis=1)).sort_values(
        "min_fold", ascending=False).head(8).to_string())
print(f"\nsaved -> {OUT/'v34_cw_calibration.csv'}")
