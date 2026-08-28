"""항목 M 사전 고정 관문 — 2스트라이크 구간 AUC 가 오르는가.

PLAN_next.md §5: 피처 추가의 실측 전이율은 0.06 이다. 전체 BSS 가 올라도 그게
목표한 자리에서 온 게 아니면 채택하지 않는다. 그래서 **1차 관문은 구간 AUC**,
전체 BSS 는 2차다.

CatBoost GPU 비결정성으로 재실행 잡음이 ±2 BSS 다. 그래서 대조군은 **같은 세션에서
같은 설정으로 돌린 것**이어야 한다 (results.csv 의 등록 모델과 비교하지 않는다).

    python gate_M.py --cand M_cntdev --base M_base
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import gbp_common as G
from common import load_labels, load_pred

TARGETS = [("a1_count", "0-2"), ("a1_count", "1-2"), ("a1b_phase", "2스트라이크")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()

    rows = []
    for s in (2024, 2022):
        lab = load_labels(s)
        ax = G.load_axes(s)
        y = lab.y.to_numpy()
        isR = (lab.game_type == "R").to_numpy()
        pc, pb = load_pred(a.cand, s, lab), load_pred(a.base, s, lab)
        for scope, m0 in (("all", np.ones(len(y), bool)), ("R", isR)):
            for axis, b in TARGETS + [("(전체)", "(전체)")]:
                m = m0 if axis == "(전체)" else m0 & (ax[axis] == b).to_numpy()
                yy = y[m]
                r = float(yy.mean())
                nb = r * (1 - r)
                rows.append({
                    "season": s, "scope": scope, "bin": b, "n": int(m.sum()),
                    "auc_base": G.auc(yy, pb[m]), "auc_cand": G.auc(yy, pc[m]),
                    "bss_base": 100000 * (1 - ((pb[m] - yy) ** 2).mean() / nb),
                    "bss_cand": 100000 * (1 - ((pc[m] - yy) ** 2).mean() / nb)})
    d = pd.DataFrame(rows)
    d["d_auc"] = d.auc_cand - d.auc_base
    d["d_bss"] = d.bss_cand - d.bss_base
    d.to_csv(G.OUT / f"gate_M_{a.cand}.csv", index=False)

    print(f"=== 후보 {a.cand}  대조 {a.base} ===")
    print(d[["season", "scope", "bin", "n", "auc_base", "auc_cand", "d_auc",
             "bss_base", "bss_cand", "d_bss"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4f}" if abs(v) < 3 else f"{v:,.1f}"))

    g1 = d[(d["bin"] == "2스트라이크")]
    dec = d[(d["bin"] == "(전체)") & (d.season == 2024) & (d.scope == "all")].iloc[0]
    gu = d[(d["bin"] == "(전체)") & (d.season == 2022) & (d.scope == "R")].iloc[0]
    print()
    print(f"1차 관문 · 2스트라이크 AUC   두 시즌 다 상승? "
          f"{bool((g1.d_auc > 0).all())}   ({g1.d_auc.round(5).tolist()})")
    print(f"2차 관문 · 2024 all BSS      {dec.d_bss:+.1f}  (잡음 바닥 ±2)")
    print(f"2차 관문 · 2022 R  BSS 비하락 {gu.d_bss:+.1f}")


if __name__ == "__main__":
    main()
