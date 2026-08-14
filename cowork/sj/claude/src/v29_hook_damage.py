"""V29: 훅 중복이 실제로 얼마짜리 손해였는지 잰다.

submit_027/028/029 는 component_blend 를 각각 2/3/4회 적용한다.
    q_1 = w*p_ie + (1-w)*b
    q_k = w*p_ie + (1-w)*q_{k-1}
    => q_k = (1 - (1-w)^k) * p_ie + (1-w)^k * b
    w = 0.25 -> w_eff = 0.4375 / 0.5781 / 0.6836

성분 라인 예측(p_ie)은 submit_029 구성으로 v25_p_ie_029.npy 에 캐시돼 있다.
따라서 재학습 없이 각 패키지가 '실제로' 낸 Val2024 점수를 잴 수 있다.

주의: 캐시된 p_ie 는 절편 보정 전(submit_029 식)이다. 027/028 은 피처 구성이
조금씩 달라 p_ie 가 정확히 같지는 않지만, 훅 중복의 크기를 재는 데는 충분하다.
w_eff 열이 요점이다.

출력: outputs/v29_hook_damage.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load, metrics

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
C0, W, EPS = 0.0077, 0.25, 1e-7

df = load()
season = df["season"].to_numpy()
va = season == 2024
y = df[TARGET].to_numpy(np.float64)[va]
prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
null = y.mean() * (1 - y.mean())
p_ie = np.load(CACHE / "v25_p_ie_029.npy")
ref = metrics(y, b, game_type=gt)["bss_raw"]

rows = []
print(f"기준 submit_021  {ref:.3f}    성분단독 {metrics(y, p_ie)['bss_raw']:.2f}")
print(f"{'패키지':<14}{'훅':>4}{'w_eff':>8}{'절편':>7}{'BSS':>11}{'ΔBSS':>9}{'t_row':>8}")
plan = [("submit_026", 1, 0.0), ("submit_027", 2, 0.0), ("submit_028", 3, 0.0),
        ("submit_029", 4, 0.0), ("submit_030", 1, C0), ("의도했던_027~029", 1, 0.0)]
for name, k, c0 in plan:
    w_eff = 1.0 - (1.0 - W) ** k
    ie = np.clip(p_ie - c0, EPS, 1 - EPS)
    q = np.clip(w_eff * ie + (1 - w_eff) * b, EPS, 1 - EPS)
    mm = metrics(y, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y) ** 2 - (q - y) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"pkg": name, "hooks": k, "w_eff": w_eff, "intercept": c0,
                 "bss": mm["bss_raw"], "dbss": d, "t_row": d / se,
                 "pred_mean": mm["pred_mean"]})
    print(f"{name:<14}{k:>4}{w_eff:>8.4f}{c0:>7.4f}{mm['bss_raw']:>11.3f}"
          f"{d:>+9.2f}{d/se:>8.2f}")

res = pd.DataFrame(rows)
res.to_csv(OUT / "v29_hook_damage.csv", index=False)
best = res.sort_values("dbss", ascending=False).iloc[0]
p029 = res[res.pkg == "submit_029"].iloc[0]
p030 = res[res.pkg == "submit_030"].iloc[0]
print(f"\n최고 {best.pkg} {best.dbss:+.2f}")
print(f"submit_029(실제 훅4) {p029.dbss:+.2f}  vs  submit_030(훅1+절편) {p030.dbss:+.2f}"
      f"   차이 {p030.dbss - p029.dbss:+.2f}")
print(f"\nsaved -> {OUT/'v29_hook_damage.csv'}")
