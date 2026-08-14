"""P0-8: nested 선택 감사 — 공동 SVD 후보의 낙관 편향을 그리드에서 직접 측정.

joint_svd_*_search.csv 는 후보별로 f23_delta_brier / f24_delta_brier / outer_bss_2024 를
모두 갖고 있다. 따라서 재실행 없이 다음을 비교할 수 있다.

  현행  : argmin f24_delta_brier 로 고른 뒤 그 후보의 outer_bss_2024 를 보고한다  <- 낙관
  정직  : argmin f23_delta_brier 로 고른 뒤 그 후보의 outer_bss_2024 를 확인한다
  robust: argmin robust_objective 로 고른 뒤 확인한다

주의: outer_bss_2024 자체가 2024로 적합한 outer_insight_weight 를 쓰므로,
      정직한 값도 여전히 blend 가중치 낙관을 포함한다. 즉 아래 격차는
      '구조·강도 선택'에서 오는 낙관 편향의 하한이다.
"""
import pandas as pd
import numpy as np

BASE = ("/mnt/user-data/uploads/LGAIMERS/experiment/model_optimization"
        "/pitcher_cluster_matchup/reports")
OUT = "/home/claude/work/outputs"

rows = []
for f in ["joint_svd_stable_search.csv", "joint_svd_outer_search_with_stability.csv"]:
    d = pd.read_csv(f"{BASE}/{f}")
    for crit, sub in [("all", d)] + [(f"criterion={c}", g) for c, g in d.groupby("criterion")]:
        sub = sub.dropna(subset=["f23_delta_brier", "f24_delta_brier", "outer_bss_2024"])
        if len(sub) < 3:
            continue
        picks = {
            "현행: F24 최소": sub.f24_delta_brier.idxmin(),
            "정직: F23 최소": sub.f23_delta_brier.idxmin(),
            "robust_objective 최소": sub.robust_objective.idxmin(),
            "min(F23,F24) 최소": sub.assign(
                w=sub[["f23_delta_brier", "f24_delta_brier"]].max(axis=1)).w.idxmin(),
        }
        base = sub.loc[picks["현행: F24 최소"], "outer_bss_2024"]
        for how, i in picks.items():
            r = sub.loc[i]
            rows.append(dict(grid=f.replace(".csv", ""), subset=crit, selection=how,
                             n=len(sub), bss_2024=float(r.outer_bss_2024),
                             gap_vs_f24pick=float(r.outer_bss_2024 - base),
                             f23_delta=float(r.f23_delta_brier),
                             f24_delta=float(r.f24_delta_brier),
                             success_scale=r.get("success_scale"),
                             current_scale=r.get("current_scale"),
                             joint_scale=r.get("joint_scale"),
                             insight_weight=float(r.outer_insight_weight),
                             config=r.config))

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/p0_nested_selection_audit.csv", index=False)
pd.set_option("display.width", 250)
for (g, s), sub in res.groupby(["grid", "subset"]):
    print(f"\n=== {g}  [{s}]  n={sub.n.iloc[0]}")
    print(sub[["selection", "bss_2024", "gap_vs_f24pick", "f23_delta", "f24_delta",
               "success_scale", "current_scale", "joint_scale", "insight_weight"]]
          .to_string(index=False))

# 추가: 스케일 그리드 내부에서 F23 최적 스케일을 F24 에 적용하면?
print("\n" + "=" * 100)
print("스케일 선택만 바꿀 때 (구성 고정, joint_svd_stable_search)")
d = pd.read_csv(f"{BASE}/joint_svd_stable_search.csv")
best_cfg = d.loc[d.f24_delta_brier.idxmin(), "config"]
g = d[d.config.eq(best_cfg)]
print(f"F24-best 구성 {best_cfg}: 해당 구성의 후보 수 {len(g)}")
print(g[["criterion", "success_scale", "current_scale", "joint_scale",
         "f23_delta_brier", "f24_delta_brier", "outer_bss_2024"]].to_string(index=False))
