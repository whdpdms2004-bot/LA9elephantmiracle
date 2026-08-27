# -*- coding: utf-8 -*-
"""[항목 E] **F 부분군 라우팅** — hw 의 발견을 새 멤버 구성으로 다시 판정한다.

## 근거

hw 가 `cowork/hw/hw_sj_segmented_blend_analysis.md` 에서 보고했다:

> ★ game_type=F 에서 상관이 0.748 로 뚝 떨어지고, 이득이 +56.1 로 전체 중 최대

재현해보니 **방향은 맞지만 주인공이 다르다.** F 에서 가장 탈상관인 쌍은 hw 가 아니라
**이미 결합에 들어가 있는 `cw × sj3way`(0.643, 팀 전체 최저)** 다.

    부분군      rho(cw,sj3way)   rho(hw,sj3way)   rho(hw,cw)
    R (223,497)     0.9069          0.9555         0.9217
    F ( 30,010)     0.6430          0.8879         0.7520

F 는 행의 11.8% 지만 멤버 간 BSS 폭이 68 로 R(1.9)의 **36배**인 전장이다.

## 앞선 시도와 이번의 차이

이전 측정은 **옛 cw 모듈**로 했고 정방향 +11.3 / 역방향 −6.2 로 갈렸다.
볼카운트 구간가중과 같은 패턴이었고, 그때는 전역 w 로 **λ 수축**하면
양방향 생존(+1.5/+1.2)했다. 여기서는

1. 멤버를 **새 것**으로 바꾼다 (`cb2 + std_mlp` 조합 = `sj_stdmlp`)
2. 수축 λ 를 처음부터 격자로 훑는다
3. 배포 순서(모델별 `apply_calibration` 후 확률결합)를 그대로 재현한다

## 판정

**두 방향 모두 양수여야 채택.** 정방향만 크고 역방향이 음수면 그 이득은
시간에 안 걸린 우연이다 — 볼카운트에서 이미 겪었다.

    python route_F.py
"""

from __future__ import annotations

import argparse
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

FIT_M = (3, 4, 5, 6)
GRID = np.arange(0.20, 0.901, 0.01)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def cal(p, q):
    eps = 1e-6
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    lg = np.log(p / (1 - p))
    o = 1.0 / (1.0 + np.exp(-(q["logit_scale"] * (lg - q["logit_center_C0"])
                              + q["logit_target_C1"])))
    return np.clip(o, max(eps, q["target_rate"] - q["cap"]),
                   min(1 - eps, q["target_rate"] + q["cap"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(FINAL / "submit" / "submit_sj_stdmlp.zip"))
    a = ap.parse_args()

    from common import load_labels
    P = FINAL / "preds"
    z = zipfile.ZipFile(a.zip)
    par = json.loads(z.read("model/cw/model/params.json"))
    W = np.array([par["blend_w_cb"], par["blend_w_ft"], par["blend_w_mlp"]])

    L = load_labels(2024)
    y = L["y"].to_numpy(np.float64)
    m = L["game_month"].to_numpy()
    g = L["game_type"].to_numpy().astype(str)

    M = np.column_stack([cal(np.load(P / "E2_var_cb2_a0.15_2024.npy"), par["model_cb"]),
                         cal(np.load(P / "S1_base__ft_2024.npy"), par["model_ft"]),
                         cal(np.load(P / "PREP_std_176_2024.npy"), par["model_mlp"])])
    r0 = y.mean()
    cw = np.clip(r0 + (M - r0) @ W, 1e-6, 1 - 1e-6)

    def col(f):
        return (pd.read_csv(PT / "val" / f).set_index("row_id")
                .loc[L["row_id"]]["pred"].to_numpy(np.float64))

    sj = col("sj3way_2024.csv")
    hw = col("hw_v12_2024.csv")
    yn = col("yn_fa10c_2024.csv")

    F = g == "F"
    R = ~F
    print("=" * 88)
    print("[항목 E] F 부분군 라우팅 — 새 멤버 (cw = sj_stdmlp 모듈)")
    print("=" * 88)
    print("\n부분군 상관")
    print("  %-4s %9s   %14s %14s %12s"
          % ("", "n", "rho(cw,sj3way)", "rho(hw,sj3way)", "rho(hw,cw)"))
    for nm, mask in (("전체", np.ones(len(y), bool)), ("R", R), ("F", F)):
        print("  %-4s %9s   %14.4f %14.4f %12.4f"
              % (nm, format(int(mask.sum()), ","),
                 np.corrcoef(cw[mask], sj[mask])[0, 1],
                 np.corrcoef(hw[mask], sj[mask])[0, 1],
                 np.corrcoef(hw[mask], cw[mask])[0, 1]))
    print("\n부분군 단독 BSS")
    for nm, mask in (("R", R), ("F", F)):
        print("  %-2s  cw %8.1f · sj3way %8.1f · hw %8.1f · yn %8.1f"
              % (nm, bss(cw[mask], y[mask]), bss(sj[mask], y[mask]),
                 bss(hw[mask], y[mask]), bss(yn[mask], y[mask])))

    f1 = np.isin(m, FIT_M)

    def bestw(mask, A_, B_):
        rf = y[mask].mean()
        sc = [bss(np.clip(rf + w * (A_[mask] - rf) + (1 - w) * (B_[mask] - rf),
                          1e-6, 1 - 1e-6), y[mask]) for w in GRID]
        return float(GRID[int(np.argmax(sc))])

    def run(fitm, evm, lam, third=None):
        """전역 w 를 부분군 w 로 lam 만큼 당긴다. third 가 있으면 F 에만 3멤버."""
        re_ = y[evm].mean()
        w0 = bestw(fitm, cw, sj)
        out = np.empty(evm.sum())
        for mask in (R, F):
            wg = bestw(fitm & mask, cw, sj)
            w = lam * wg + (1 - lam) * w0
            sel = mask[evm]
            out[sel] = re_ + w * (cw[evm][sel] - re_) + (1 - w) * (sj[evm][sel] - re_)
        if third is not None:
            # F 에서만 3멤버. 합=1·비음수 격자 (0.05)
            best, bw = -9e9, None
            for wa in np.arange(0, 1.001, 0.05):
                for wb in np.arange(0, 1.001 - wa + 1e-9, 0.05):
                    wc = 1 - wa - wb
                    rf = y[fitm & F].mean()
                    p = rf + wa * (cw[fitm & F] - rf) + wb * (sj[fitm & F] - rf) \
                        + wc * (third[fitm & F] - rf)
                    s = bss(np.clip(p, 1e-6, 1 - 1e-6), y[fitm & F])
                    if s > best:
                        best, bw = s, (wa, wb, wc)
            wa, wb, wc = bw
            selF = F[evm]
            out[selF] = (re_ + wa * (cw[evm][selF] - re_) + wb * (sj[evm][selF] - re_)
                         + wc * (third[evm][selF] - re_))
        return bss(np.clip(out, 1e-6, 1 - 1e-6), y[evm])

    print("\n" + "-" * 88)
    print("R/F 라우팅 + 수축 (전역 w 로 당기는 정도 lam)")
    print("%-28s %12s %12s   %s" % ("구성", "정방향 Δ", "역방향 Δ", "판정"))
    print("-" * 88)
    b_f = run(f1, ~f1, 0.0)
    b_r = run(~f1, f1, 0.0)
    print("%-28s %12.1f %12.1f   기준 (라우팅 없음)" % ("lam=0.0", b_f, b_r))
    for lam in (0.2, 0.3, 0.5, 0.7, 1.0):
        d1 = run(f1, ~f1, lam) - b_f
        d2 = run(~f1, f1, lam) - b_r
        ok = "양방향 이득" if d1 > 0 and d2 > 0 else ""
        print("%-28s %+12.1f %+12.1f   %s" % ("lam=%.1f" % lam, d1, d2, ok))
    print()
    for nm, th in (("F 에 hw 추가", hw), ("F 에 yn 추가", yn)):
        d1 = run(f1, ~f1, 0.0, th) - b_f
        d2 = run(~f1, f1, 0.0, th) - b_r
        ok = "양방향 이득" if d1 > 0 and d2 > 0 else ""
        print("%-28s %+12.1f %+12.1f   %s" % (nm, d1, d2, ok))
    print("\n(두 방향 모두 양수여야 채택. 정방향만 크면 시간에 안 걸린 우연이다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
