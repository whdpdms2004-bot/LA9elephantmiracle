"""V32: 성분 라인의 fold 별 붕괴가 '레벨'인가 '형태'인가.

동기
    w 를 못 올리는 유일한 이유는 2023 이다.
        fold 2022  base 2296.9   성분단독 2241.4   격차   +55
        fold 2023  base −140.2   성분단독 −1127.1  격차  +987
        fold 2024  base  836.5   성분단독  763.2   격차   +73
    2023 에서만 성분 라인이 base 보다 987 나쁘다. 이 하나 때문에 w=0.25 에
    묶여 있다. 2023 붕괴의 정체를 알면 w 상한이 풀릴 수 있다.

    Public 결과가 이 질문을 급하게 만들었다.
        submit_029  Val2024 +16.12 (w_eff 0.684)  ->  Public 963
        submit_030  Val2024 +40.48 (w    0.250)  ->  Public 964
    내부 24점 차이가 Public 1점이다. 2025 에서는 w 가 거의 무관하다.
    2024 최적(0.35)보다 2022 최적(0.50)에 가까운 환경이라는 뜻이다.

가설
    A 레벨: 예측 평균이 실제 평균에서 벗어났다. 상수 이동으로 회복된다.
    B 형태: 순위/기울기가 틀렸다. 이동으로 회복되지 않는다.

    A 라면 시즌 레벨 예보를 고치는 것이 답이고, 그건 이미 base_score
    외삽으로 하고 있으니 개선 여지가 크다. B 라면 w 상한은 진짜다.

측정
    각 fold 에서
        raw       그대로
        shift     확률에 상수를 더해 예측 평균 = 실제 평균 (오라클)
        logit     로짓에 상수를 더해 예측 평균 = 실제 평균 (오라클)
    오라클이므로 채택 가능한 방법이 아니다. '레벨을 완벽히 맞추면 얼마나
    회복되는가'라는 상한을 재는 것이다.

출력: outputs/v32_level_vs_shape.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load, metrics

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
FOLDS = [2022, 2023, 2024]
EPS = 1e-7

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)

models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
strong = {}
for fold in FOLDS:
    ids = df.loc[season == fold, "row_id"].to_numpy()
    acc, cnt = None, 0
    for mn in models:
        f = OOF_DIR / f"{mn}_fold{fold}.parquet"
        if f.exists():
            acc = (pd.read_parquet(f).set_index("row_id").reindex(ids)["prediction"]
                   .to_numpy() if acc is None else acc + pd.read_parquet(f)
                   .set_index("row_id").reindex(ids)["prediction"].to_numpy())
            cnt += 1
    strong[fold] = np.clip(acc / cnt, EPS, 1 - EPS)
prod = pd.read_parquet(PROD).set_index("row_id").reindex(
    df.loc[season == 2024, "row_id"].to_numpy())
strong[2024] = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                       EPS, 1 - EPS)


def shift_prob(p, target):
    return np.clip(p + (target - p.mean()), EPS, 1 - EPS)


def shift_logit(p, target):
    z = np.log(p / (1 - p))

    def f(c):
        return float((1 / (1 + np.exp(-(z + c)))).mean() - target)

    c = brentq(f, -5.0, 5.0, xtol=1e-10)
    return np.clip(1 / (1 + np.exp(-(z + c))), EPS, 1 - EPS), c


rows = []
print(f"{'fold':<6}{'라인':<10}{'raw':>11}{'prob이동':>11}{'logit이동':>11}"
      f"{'pred_mean':>11}{'실제':>10}{'편향%p':>9}{'레벨기여':>10}")
for vs in FOLDS:
    va = season == vs
    y = y_all[va]
    t = float(y.mean())
    lines = {"base": strong[vs]}
    cf = CACHE / f"v30_pie_{vs}.npy"
    if cf.exists():
        lines["성분"] = np.load(cf)
    for name, p in lines.items():
        raw = metrics(y, p)["bss_raw"]
        sp = metrics(y, shift_prob(p, t))["bss_raw"]
        sl_p, c = shift_logit(p, t)
        sl = metrics(y, sl_p)["bss_raw"]
        bias = (p.mean() - t) * 100
        rows.append({"fold": vs, "line": name, "raw": raw, "shift_prob": sp,
                     "shift_logit": sl, "pred_mean": p.mean(), "target": t,
                     "bias_pp": bias, "level_gain": max(sp, sl) - raw,
                     "logit_c": c})
        print(f"{vs:<6}{name:<10}{raw:>11.2f}{sp:>11.2f}{sl:>11.2f}"
              f"{p.mean():>11.5f}{t:>10.5f}{bias:>+9.3f}{max(sp,sl)-raw:>+10.2f}")

res = pd.DataFrame(rows)
res.to_csv(OUT / "v32_level_vs_shape.csv", index=False)

print("\n" + "=" * 96)
print("2023 격차 분해")
print("=" * 96)
s = res[res.fold == 2023].set_index("line")
if "성분" in s.index:
    gap_raw = s.loc["base", "raw"] - s.loc["성분", "raw"]
    gap_lvl = (s.loc["base", ["shift_prob", "shift_logit"]].max()
               - s.loc["성분", ["shift_prob", "shift_logit"]].max())
    print(f"  raw 격차          {gap_raw:9.2f}")
    print(f"  레벨 보정 후 격차 {gap_lvl:9.2f}")
    print(f"  레벨이 설명하는 몫 {gap_raw-gap_lvl:9.2f}  "
          f"({(gap_raw-gap_lvl)/gap_raw*100:.1f}%)")
    print()
    if gap_raw - gap_lvl > 0.6 * gap_raw:
        print("  -> 레벨 문제다. 시즌 레벨 예보를 고치면 w 상한이 풀린다.")
    elif gap_raw - gap_lvl < 0.3 * gap_raw:
        print("  -> 형태 문제다. w 상한은 진짜이고 성분 라인 자체를 고쳐야 한다.")
    else:
        print("  -> 절반씩이다. 레벨 보정으로 일부 풀리지만 형태도 손봐야 한다.")
print(f"\nsaved -> {OUT/'v32_level_vs_shape.csv'}")
