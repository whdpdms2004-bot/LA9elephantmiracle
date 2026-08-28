# -*- coding: utf-8 -*-
"""구간별 약점 지도 — 축 x 구간 BSS/AUC 를 단일 HTML 로.

    python make_map.py                       # 기본: sj_stdmlp
    python make_map.py --model sj_grid_w060
    python make_map.py --out ../DIAG.html

`group_score.py` 가 만든 `out/by_bin.csv` · `out/overall.csv` 를 읽는다.
막대는 **그 구간 AUC - 전체 AUC** 이고 두 시즌을 나란히 그린다 —
한 시즌만 보면 우연을 약점으로 잡기 때문이다 (PLAN.md 자기적합 금지 규약).

서식은 `_map_template.html` 이고 `__DATA__` 자리에 JSON 을 인라인한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TEMPLATE = HERE / "_map_template.html"
SEASONS = ("2024", "2022")          # 앞이 주 판정, 뒤가 관문 (README 규칙 1)


def build(model: str) -> dict:
    b = pd.read_csv(OUT / "by_bin.csv")
    o = pd.read_csv(OUT / "overall.csv")

    ov = {}
    for s in SEASONS:
        m = o[(o.model == model) & (o.season == int(s)) & (o.scope == "all")]
        if m.empty:
            raise SystemExit(
                f"overall.csv 에 {model} / {s} / all 이 없다. group_score.py 를 먼저 돌린다.")
        r = m.iloc[0]
        ov[s] = dict(n=int(r.n), rate=float(r.rate), auc=float(r.auc),
                     bss=float(r.bss_local), bias=float(r.bias),
                     pred_mean=float(r.pred_mean))

    sub = b[(b.model == model) & (b.scope == "all")]
    if sub.empty:
        raise SystemExit(f"by_bin.csv 에 {model} 이 없다.")

    axes = []
    for ax, g in sub.groupby("axis"):
        bins = []
        for bn in g[g.season == int(SEASONS[0])]["bin"].tolist():
            row, ok = {"bin": bn}, True
            for s in SEASONS:
                m = g[(g.season == int(s)) & (g["bin"] == bn)]
                if m.empty:
                    ok = False
                    continue
                m = m.iloc[0]
                row[s] = dict(n=int(m.n), share=float(m.share), rate=float(m.rate),
                              auc=float(m.auc), bss=float(m.bss_local),
                              deficit=float(m.deficit), bias=float(m.bias),
                              pred_mean=float(m.pred_mean))
            if ok:
                bins.append(row)
        if not bins:
            continue

        def spread(s: str) -> float:
            v = [x[s]["auc"] for x in bins]
            return max(v) - min(v)

        # 안정성 = 구간의 (AUC - 전체AUC) 부호가 두 시즌 같은 비율.
        # 부호가 뒤집히는 축은 고칠 대상 자체가 없다 (RESULTS.md §4).
        same = sum(
            (x[SEASONS[0]]["auc"] - ov[SEASONS[0]]["auc"] >= 0)
            == (x[SEASONS[1]]["auc"] - ov[SEASONS[1]]["auc"] >= 0)
            for x in bins)

        axes.append(dict(axis=ax, title=g.axis_title.iloc[0], bins=bins,
                         spread24=spread(SEASONS[0]), spread22=spread(SEASONS[1]),
                         stable=same / len(bins), nbins=len(bins)))

    axes.sort(key=lambda a: -a["spread24"])
    for i, a in enumerate(axes):
        a["rank"] = i + 1
    return dict(model=model, overall=ov, axes=axes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sj_stdmlp", help="by_bin.csv 의 model 이름")
    ap.add_argument("--out", default=str(OUT / "axes-map.html"))
    a = ap.parse_args()

    d = build(a.model)
    payload = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</script", "<\\/script")   # 인라인 JSON 이 script 를 끊지 않게

    tpl = TEMPLATE.read_text(encoding="utf-8")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace("__DATA__", payload), encoding="utf-8", newline="\n")

    print(f"{a.model}  ->  {out}  ({out.stat().st_size:,} bytes)")
    print(f"{'#':>3} {'axis':12} {'AUC폭24':>9} {'AUC폭22':>9} {'부호일치':>8}  title")
    for x in d["axes"]:
        print(f"{x['rank']:>3} {x['axis']:12} {x['spread24']:9.4f} "
              f"{x['spread22']:9.4f} {x['stable']:8.0%}  {x['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
