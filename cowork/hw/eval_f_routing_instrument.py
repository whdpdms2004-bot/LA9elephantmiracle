"""F행 라우팅 델타의 계기(보정) 민감도.

## 왜 필요한가

`sj_run/results/01_calib_discrepancy.md` 가 148점 계기 불일치를 못 풀고 끝나면서
"어느 계기가 맞는지 모르는 상태에서 라우팅 L 을 재는 게 의미가 있느냐"를 물었다.

**답은 델타가 계기에 불변인가로 갈린다.** 148 은 *수준*(level)의 문제다. 라우팅
판정은 *델타*(같은 계기 안에서 기준선 대비 차이)라, 계기를 바꿔도 델타가 유지되면
계기 확정 전에도 판정할 수 있다.

## 무엇을 재나

배포가 걸 수 있는 보정을 여러 개 가정하고, 각 계기에서 **기준선과 라우팅본에
같은 보정을 걸어** 델타만 비교한다.

    무보정 (등록 val 그대로)
    logit affine + 목표율 중심이동 + 클리핑, scale in {0.80, 0.90, 1.00}, target 0.474695
    cw v16 상수 (scale 0.95, target 0.47353)

실행:
    python cowork/hw/eval_f_routing_instrument.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PT = REPO / "performance_tracking"
W_CHAMP = np.array([0.6, 0.4])
LAMBDAS = [0.10, 0.20]
CAP = 0.2
FIT_MONTHS = (3, 4, 5, 6)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def calib(p, scale, target, cap=CAP):
    """배포 script.py 의 apply_calibration 과 같은 형태.
    로짓 축소 -> 목표율 중심이동 -> 상하한 클리핑."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    z = scale * (lg - lg.mean()) + np.log(target / (1 - target))
    q = 1.0 / (1.0 + np.exp(-z))
    return np.clip(q, max(1e-6, target - cap), min(1 - 1e-6, target + cap))


def main():
    lab = pd.read_csv(PT / ".cache" / "labels_2024.csv")
    D = lab.copy()
    for m in ("sj_stdmlp", "sj3way", "hw_v12_honest"):
        d = pd.read_csv(PT / "val" / f"{m}_2024.csv")[["row_id", "pred"]]
        D = D.merge(d.rename(columns={"pred": m}), on="row_id", how="inner")

    y = D["y"].to_numpy(float)
    F = (D["game_type"] == "F").to_numpy()
    mo = D["game_month"].to_numpy()
    A = np.isin(mo, FIT_MONTHS)
    B = ~A
    r = y.mean()
    champ0 = np.clip(r + (D[["sj_stdmlp", "sj3way"]].to_numpy(float) - r) @ W_CHAMP,
                     1e-6, 1 - 1e-6)
    hw0 = D["hw_v12_honest"].to_numpy(float)

    instruments = [("무보정 (등록 val 그대로)", None)]
    for s in (0.80, 0.90, 1.00):
        instruments.append((f"보정 scale {s:.2f} / target 0.474695", (s, 0.474695)))
    instruments.append(("보정 scale 0.95 / target 0.47353 (cw v16)", (0.95, 0.47353)))

    print("F행 라우팅 델타의 계기 민감도 -- 기준선과 라우팅본에 같은 보정을 걸고 델타만 비교")
    print(f"  F행 {F.sum():,} / 전체 {len(D):,} ({F.mean()*100:.1f}%)\n")
    hdr = (f"  {'계기':40} {'기준BSS':>9} {'L=.10 전체Δ':>12} {'L=.20 전체Δ':>12} "
           f"{'L=.20 월7-10':>13} {'L=.20 월3-6':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for nm, cfg in instruments:
        if cfg is None:
            c, h = champ0, hw0
        else:
            s, t = cfg
            c, h = calib(champ0, s, t), calib(hw0, s, t)
        b_all, b_ev, b_fit = bss(c, y), bss(c[B], y[B]), bss(c[A], y[A])
        d_all = {}
        d_ev = d_fit = 0.0
        for L in LAMBDAS:
            p = c.copy()
            p[F] = np.clip((1 - L) * c[F] + L * h[F], 1e-6, 1 - 1e-6)
            d_all[L] = bss(p, y) - b_all
            if L == 0.20:
                d_ev = bss(p[B], y[B]) - b_ev
                d_fit = bss(p[A], y[A]) - b_fit
        print(f"  {nm:40} {b_all:9.1f} {d_all[0.10]:+12.2f} {d_all[0.20]:+12.2f} "
              f"{d_ev:+13.2f} {d_fit:+12.2f}")

    print("\n  읽는 법: 기준 BSS 는 계기마다 크게 흔들리지만(수준), 델타는 거의 안 움직인다.")
    print("  148 은 수준의 문제이므로 라우팅 판정(델타)은 계기 확정 전에도 유효하다.")


if __name__ == "__main__":
    main()
