# -*- coding: utf-8 -*-
"""[중재] 두 계기가 갈린 팀 멤버 판정을 **세 번째 폴드**로 가린다.

## 무엇이 갈렸나

    월전방분할 (시즌 **내부**: 월3~6 <-> 월7~10)   hw 가중 0.00 · 이득 없음
    §31 규약   (시즌 **간**: fit2022 -> eval2024)    hw 가중 0.05~0.10 · +3.5

배포는 시즌 간 전이라 §31 이 구조적으로 가깝다. 그러나 §31 의 fit 폴드 2022 는
`game_type` **구조 단절 이전**이다 (F 성공률 0.7087 vs 2024 의 0.4593).
그래서 그 답도 못 믿는다.

## 중재자 — fit(2023) -> eval(2024)

`run_arm.py` 가 이미 적어뒀다.

> fold 2023 은 판정 수치로 싣지 않는다. 다만 **결합 가중치를 적합하는 폴드로는
> 쓴다** — 단절 이후라 2024 와 같은 레짐이고 인접 시즌이라 배포와 동형이다.

## ★ 이 계기의 한계 — 반드시 같이 읽을 것

fold 2023 예측은 **<=2022(단절 이전) 학습 모델**이 만든다. 그래서 원시 BSS 가
크게 음수다 (mlp -951.7 · hw -1579.6). 배포는 2019~2024 학습이라 단절 이후
시즌을 **둘** 포함한다 — 이 점에서 fold 2023 도 배포와 완전히 같지는 않다.

**전원에게 같은 `calib` 을 걸어 중심 문제를 상쇄시킨 뒤 상대 정보량만 본다.**
절대값은 읽지 않는다.

## 판정

세 계기가 **모두** hw 에 가중을 주면 채택, 모두 0 이면 기각,
갈리면 판단 보류하고 제출 점수를 기다린다.

    python arbitrate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
PT = ROOT / "performance_tracking"
sys.path.insert(0, str(PT / "tools"))
sys.path.insert(0, str(FINAL / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

STEP = 0.05


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def simplex(k, step=STEP):
    if k == 1:
        return [np.array([1.0])]
    g = np.arange(0.0, 1.0 + 1e-9, step)
    out = []
    for c in itertools.product(g, repeat=k - 1):
        s = sum(c)
        if s <= 1.0 + 1e-9:
            out.append(np.array(list(c) + [1.0 - s]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(FINAL / "submit" / "submit_sj_stdmlp.zip"))
    a = ap.parse_args()

    from common import load_labels
    from run_arm import calib
    P = FINAL / "preds"
    par = json.loads(zipfile.ZipFile(a.zip).read("model/cw/model/params.json"))

    def cal(p, q):
        eps = 1e-6
        p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
        lg = np.log(p / (1 - p))
        o = 1.0 / (1.0 + np.exp(-(q["logit_scale"] * (lg - q["logit_center_C0"])
                                  + q["logit_target_C1"])))
        return np.clip(o, max(eps, q["target_rate"] - q["cap"]),
                       min(1 - eps, q["target_rate"] + q["cap"]))

    W = np.array([par["blend_w_cb"], par["blend_w_ft"], par["blend_w_mlp"]])
    MEM = ["sj3way_nv", "hw_v12_honest", "ye_hand"]
    D = {}
    for fold in (2024, 2023, 2022):
        need = [P / ("%s_%d.npy" % (n, fold))
                for n in ("E2_var_cb2_a0.15", "FTX_q64_168", "PREP_std_176")]
        if not all(p.exists() for p in need):
            print("  fold%d — cw 멤버 예측이 아직 없다: %s"
                  % (fold, [p.name for p in need if not p.exists()]))
            continue
        L = load_labels(fold)
        M = np.column_stack([cal(np.load(need[0]), par["model_cb"]),
                             cal(np.load(need[1]), par["model_ft"]),
                             cal(np.load(need[2]), par["model_mlp"])])
        b = L[["row_id", "y"]].copy()
        b["cw"] = np.clip(L["y"].mean() + (M - L["y"].mean()) @ W, 1e-6, 1 - 1e-6)
        for n in MEM:
            f = PT / "val" / ("%s_%d.csv" % (n, fold))
            if f.exists():
                b = b.merge(pd.read_csv(f)[["row_id", "pred"]]
                            .rename(columns={"pred": n}), on="row_id", how="inner")
        D[fold] = b
        print("  fold%d  공통 %s행 · 멤버 %s"
              % (fold, f"{len(b):,}", [c for c in b.columns if c not in ("row_id", "y")]))

    folds = sorted(D)
    names = [c for c in D[folds[0]].columns if c not in ("row_id", "y")]
    for f in folds:
        names = [n for n in names if n in D[f].columns]
    print("\n공통 멤버: %s" % names)

    # 전원 같은 calib — 중심 문제를 상쇄시킨다
    E = {}
    for f in folds:
        y = D[f]["y"].to_numpy(np.float64)
        E[f] = (np.column_stack([calib(D[f][n].to_numpy(np.float64), y)[2]
                                 for n in names]), y)
        print("  fold%d 원시 cw %8.1f -> calib 후 %8.1f"
              % (f, bss(D[f]["cw"].to_numpy(np.float64), y),
                 bss(E[f][0][:, 0], y)))

    def run(fit_f, ev_f, cols):
        Mf, yf = E[fit_f][0][:, cols], E[fit_f][1]
        Me, ye = E[ev_f][0][:, cols], E[ev_f][1]
        cand = simplex(len(cols))
        rf, re_ = yf.mean(), ye.mean()
        sc = [bss(np.clip(rf + (Mf - rf) @ w, 1e-6, 1 - 1e-6), yf) for w in cand]
        w = cand[int(np.argmax(sc))]
        return bss(np.clip(re_ + (Me - re_) @ w, 1e-6, 1 - 1e-6), ye), w

    protos = []
    if 2022 in D and 2024 in D:
        protos.append(("§31  fit2022 -> eval2024", 2022, 2024))
    if 2023 in D and 2024 in D:
        protos.append(("★중재 fit2023 -> eval2024", 2023, 2024))
    if 2024 in D and 2022 in D:
        protos.append(("역   fit2024 -> eval2022", 2024, 2022))

    print("\n" + "=" * 92)
    print("[중재] 세 계기가 hw 에 가중을 주는가")
    print("=" * 92)
    combos = []
    ic = names.index("cw")
    for k in range(1, len(names) + 1):
        for cb_ in itertools.combinations(range(len(names)), k):
            if ic in cb_:
                combos.append(cb_)
    for tag, ff, ef in protos:
        print("\n%s" % tag)
        print("  %-34s %11s   %s" % ("조합", "eval BSS", "fit 가중"))
        rows = []
        for cb_ in combos:
            s, w = run(ff, ef, list(cb_))
            rows.append((s, [names[i] for i in cb_], w))
        for s, sub, w in sorted(rows, reverse=True)[:5]:
            print("  %-34s %11.1f   %s"
                  % ("+".join(sub), s, np.array2string(w, precision=2)))
        hw_i = names.index("hw_v12_honest") if "hw_v12_honest" in names else None
        if hw_i is not None:
            best = max(rows)
            got = ("hw_v12_honest" in best[1]
                   and best[2][best[1].index("hw_v12_honest")] > 0.001)
            print("  → 최고 조합에 hw 가중 %s" % ("있음" if got else "없음(0)"))
    print("\n(세 계기가 모두 같은 방향이면 결론, 갈리면 제출 점수를 기다린다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
