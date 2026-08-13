"""P0-1: 선택 과적합 진단.

예측 벡터 없이, 기존 탐색 그리드의 F23 / F24 순위 상관으로 판정한다.
두 fold의 순위가 무관하면 F24 최댓값 선택은 노이즈 추적이다.

입력: experiment/model_optimization/pitcher_cluster_matchup/reports/*.csv
      (f23_delta_brier, f24_delta_brier 컬럼을 가진 모든 그리드)
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

BASE = ("/mnt/user-data/uploads/LGAIMERS/experiment/model_optimization"
        "/pitcher_cluster_matchup/reports")
OUT = "/home/claude/work/outputs"

FILES = [
    # 구조 선택 그리드
    "joint_svd_cluster_grid.csv",
    "deep_pitcher_cluster_grid.csv",
    # 강도 선택 그리드
    "dual_matchup_scale_tuning.csv",
    "reverse_batter_seed_joint_tuning.csv",
    "dual_reverse_batter_scale_tuning.csv",
    "large_xgb_correction_grid.csv",
    # 최종 후보 선발
    "joint_svd_outer_search_with_stability.csv",
    "joint_svd_stable_search.csv",
]

rows = []
for f in FILES:
    d = pd.read_csv(f"{BASE}/{f}").dropna(subset=["f23_delta_brier", "f24_delta_brier"])
    if len(d) < 5:
        continue
    rho, p = spearmanr(d.f23_delta_brier, d.f24_delta_brier)
    both = ((d.f23_delta_brier < 0) & (d.f24_delta_brier < 0)).mean()
    i24, i23 = d.f24_delta_brier.idxmin(), d.f23_delta_brier.idxmin()
    pct23 = (d.f23_delta_brier < d.loc[i24, "f23_delta_brier"]).mean()
    pct24 = (d.f24_delta_brier < d.loc[i23, "f24_delta_brier"]).mean()
    k = max(3, int(len(d) * 0.1))
    t23 = set(d.nsmallest(k, "f23_delta_brier").index)
    t24 = set(d.nsmallest(k, "f24_delta_brier").index)
    rows.append(dict(grid=f.replace(".csv", ""), n=len(d), spearman=rho, p_value=p,
                     both_improve=both, f24best_f23_pct=pct23, f23best_f24_pct=pct24,
                     top10_jaccard=len(t23 & t24) / len(t23 | t24)))
    print(f"{rows[-1]['grid']:44s} n={len(d):5d} spearman={rho:+.3f} "
          f"both={both:.1%} F24best_F23pct={pct23:.1%} jaccard={rows[-1]['top10_jaccard']:.2f}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/p0_selection_overfit.csv", index=False)

# 구조만 바꿀 때 (강도 고정) 의 전이 여부
d = pd.read_csv(f"{BASE}/deep_pitcher_cluster_grid.csv")
g = d.groupby("pitcher_config")[["f23_delta_brier", "f24_delta_brier"]].min()
print(f"\npitcher_config 단위 최선값끼리: n={len(g)} "
      f"spearman={spearmanr(g.f23_delta_brier, g.f24_delta_brier)[0]:+.3f}")
