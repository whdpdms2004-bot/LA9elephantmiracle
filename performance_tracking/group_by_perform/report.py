"""out/*.csv -> out/tables.md. 표만 만든다. 해석은 RESULTS.md 가 사람 손으로 쓴다.

PLAN.md §6 의 관문을 그대로 코드로 적용해 판정 열을 붙인다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gbp_common as G

REF = "sj_grid_w060"          # Public 실측이 있는 현행 챔피언
CAND = "sj_stdmlp"            # 미채점 후보
SCOPE = "R"                   # 2022 F 는 기저율 0.69~0.74 - 축 비교는 R 에서만


def md(df: pd.DataFrame, floats: str = "{:,.0f}") -> str:
    return df.to_markdown(floatfmt=".4g")


def add_gap(t: pd.DataFrame) -> pd.DataFrame:
    """축 내부 가중평균 대비 격차.

    축이 정보를 담고 있으면 (예: 투수 수준 5분위) 구간을 나누는 순간 구간 간
    resolution 이 전체 BSS 에서 빠져나가 **모든** 구간이 전체보다 낮아진다.
    전체 BSS 와 비교하면 그걸 약점으로 잘못 읽는다. 축 자기 평균과 비교한다.
    """
    t = t.copy()
    w = t.share / t.groupby(["model", "season", "axis"]).share.transform("sum")
    t["axis_mean"] = (t.bss_local * w).groupby(
        [t.model, t.season, t.axis]).transform("sum")
    t["gap_w"] = t.bss_local - t.axis_mean
    t["head_w"] = t.gap_w * t.share          # 시즌 BSS 로 환산한 몫
    return t


def main() -> None:
    d = add_gap(pd.read_csv(G.OUT / "by_bin.csv").query("scope == @SCOPE"))
    d_all = add_gap(pd.read_csv(G.OUT / "by_bin.csv").query("scope == 'all'"))
    ov = pd.read_csv(G.OUT / "overall.csv")
    rec = pd.read_csv(G.OUT / "recal_axis.csv")
    L: list[str] = []
    P = L.append

    P("# group_by_perform — 생성 표")
    P("")
    P("`report.py` 가 만든다. **손으로 고치지 않는다** — 해석은 `../RESULTS.md`.")
    P(f"기준 모델 `{REF}` · 후보 `{CAND}` · 축 비교 범위 `game_type == {SCOPE}`.")
    P("")

    P("## 1. 전체 — scope 가 결론을 바꾼다")
    P("")
    for k, col, r in (("BSS", "bss_local", 1), ("AUC", "auc", 4)):
        P(f"### {k}")
        P("")
        P(md(ov.pivot_table(index="model", columns=["season", "scope"], values=col).round(r)))
        P("")

    P("## 2. 축 순위 — 구간 격차의 크기 (share 가중 |gap_w| 합)")
    P("")
    a = (d[d.model == REF].groupby(["season", "axis_title"])
         .apply(lambda g: (g.share * g.gap_w.abs()).sum(), include_groups=False)
         .unstack(0))
    a["평균"] = a.mean(1)
    P(md(a.sort_values("평균", ascending=False).round(1)))
    P("")
    P("숫자가 클수록 그 축이 성능을 갈라놓는다. 지시받은 4축 중 `A1 볼카운트` 만 상위다.")
    P("")

    P("## 3. 두 시즌 모두 축 평균에 못 미치는 구간")
    P("")
    p = (d[d.model == REF].pivot_table(index=["axis_title", "bin"], columns="season",
                                       values=["gap_w", "head_w", "auc", "n"]))
    p["합"] = p[("head_w", 2022)] + p[("head_w", 2024)]
    lo = p[(p[("gap_w", 2022)] < 0) & (p[("gap_w", 2024)] < 0)].sort_values("합")
    hi = p[(p[("gap_w", 2022)] > 0) & (p[("gap_w", 2024)] > 0)].sort_values("합", ascending=False)
    P("### 열세 — 두 시즌 다 음수인 것만 (안정적인 약점)")
    P("")
    P(md(lo.head(15).round(3)))
    P("")
    P("### 우세")
    P("")
    P(md(hi.head(8).round(3)))
    P("")

    P("## 4. 지시받은 4축 상세")
    P("")
    for ax in G.PRIMARY:
        t = d[(d.axis == ax) & (d.model.isin([REF, CAND]))]
        order = [b for b in G.AXES[ax][1] if b in set(t["bin"])]
        pv = (t.pivot_table(index="bin", columns=["model", "season"],
                            values=["n", "rate", "bias", "auc", "bss_local", "gap_w"])
                .reindex(order))
        P(f"### {G.AXES[ax][0]}")
        P("")
        P(md(pv.round(4)))
        P("")

    P("## 5. 멤버 분담 후보 — 같은 구간에서 모델 간 bss_local 차 (2024, R)")
    P("")
    ms = G.MAIN + G.MAIN_2024_ONLY
    t = d[(d.season == 2024) & (d.model.isin(ms))]
    g = t.pivot_table(index=["axis_title", "bin"], columns="model", values="bss_local")
    cols = [c for c in g.columns]
    g["spread"] = g[cols].max(1) - g[cols].min(1)
    g["best"], g["worst"] = g[cols].idxmax(1), g[cols].idxmin(1)
    g["n"] = t.groupby(["axis_title", "bin"]).n.first()
    g["관문"] = np.where(g.spread >= 30, "후보", "-")
    P(md(g.sort_values("spread", ascending=False)
          .head(12)[["n", "spread", "best", "worst", "관문"]].round(0)))
    P("")

    P("## 6. 구간 보정이 시즌을 건너는가 — 관문 `shift_net >= 3.0`")
    P("")
    r = rec[rec["eval"] == 2024].pivot_table(index="axis_title", columns="model",
                                             values="shift_net")
    r["최대"] = r.max(1)
    r["관문"] = np.where(r["최대"] >= 3.0, "통과", "미달")
    P(md(r.round(1)))
    P("")
    P("`shift_net` = 구간별 상수 보정 이득 − 전역 상수 보정 이득 "
      "(2022 적합 → 동결 → 2024 평가, R 만).")
    P("")

    P("## 7. 참고 — scope=all 로 보면 어떻게 달라지나")
    P("")
    aa = (d_all[d_all.model == REF].groupby(["season", "axis_title"])
          .apply(lambda g: (g.share * g.gap_w.abs()).sum(), include_groups=False).unstack(0))
    aa["평균"] = aa.mean(1)
    P(md(aa.sort_values("평균", ascending=False).round(1)))
    P("")

    (G.OUT / "tables.md").write_text("\n".join(L), encoding="utf-8")
    print(f"out/tables.md  {len(L)}줄")


if __name__ == "__main__":
    main()
