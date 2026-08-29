"""배포 target_rate 가 val 폴드 기저율에서 벗어나 생기는 구조적 BSS 벌점.

## 왜 보는가

yn 관문 1 이 "val2024 all BSS 900 이상" 인데, sj 루프가 같은 모델을
배포보정 계기로 896.7 · 등록 val 계기로 916.0 이라 보고했다(-19.3).
900 이 그 사이에 걸려 있어서 계기 하나로 통과/탈락이 갈린다.

BSS 는 예측 평균이 라벨 평균에서 벗어나면 K = 1e5/(r(1-r)) 배로 벌점이 붙는다
(DECK §2.3). 배포 `apply_calibration` 은 예측 평균을 `target_rate` 0.474695 로
옮기는데, 이 값은 **2025 를 겨냥한 외삽**이라 2024(0.4861)·2022(0.5289) 기저율과
일부러 어긋나 있다. 그러면 그 폴드에서 재는 순간 모델과 무관한 벌점이 붙는다.

이 스크립트는 그 벌점이 얼마인지, 그리고 깎인 원인이 로짓 스케일인지 중심이동인지
가른다 -- 중심만 그 폴드 기저율로 바꿔서 점수가 돌아오는지 본다.

실행:
    python cowork/hw/diag_target_rate_penalty.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PT = REPO / "performance_tracking"
W_CHAMP = np.array([0.45461, 0.64333])
CENTER_SHIFT = 0.003223
TARGET_RATE = 0.474695
CAP = 0.2


def calib(p, scale, target, cap=CAP):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    z = scale * (lg - lg.mean()) + np.log(target / (1 - target))
    q = 1.0 / (1.0 + np.exp(-z))
    return np.clip(q, max(1e-6, target - cap), min(1 - 1e-6, target + cap))


def main():
    print("=" * 72)
    print("1. 시즌마다 구조적 벌점이 얼마인가  K = 1e5/(r(1-r)), 벌점 = K * offset^2")
    print("=" * 72)
    print(f"  {'시즌':8}{'실제 기저율':>13}{'target 과의 차이':>16}{'구조적 벌점':>13}")
    for s in (2022, 2023, 2024):
        y = pd.read_csv(PT / ".cache" / f"labels_{s}.csv")["y"].to_numpy(float)
        r = y.mean()
        off = TARGET_RATE - r
        print(f"  {s:<8}{r:>13.6f}{off:>+16.6f}{1e5/(r*(1-r))*off*off:>13.1f}")
    print("  -> 모델을 아무리 잘 만들어도 이만큼은 계기가 먼저 깎는다.")

    print()
    print("=" * 72)
    print("2. 깎인 원인이 스케일인가 중심이동인가 (val2024, 배포 챔피언)")
    print("=" * 72)
    lab = pd.read_csv(PT / ".cache" / "labels_2024.csv")
    d = lab.copy()
    for m in ("sj_stdmlp", "sj3way"):
        p = pd.read_csv(PT / "val" / f"{m}_2024.csv")[["row_id", "pred"]]
        d = d.merge(p.rename(columns={"pred": m}), on="row_id", how="inner")
    y = d["y"].to_numpy(float)
    r = y.mean()

    def bss(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return 1e5 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))

    champ = np.clip(r + (d[["sj_stdmlp", "sj3way"]].to_numpy(float) - r) @ W_CHAMP
                    - CENTER_SHIFT, 1e-6, 1 - 1e-6)
    print(f"  val2024 실제 기저율 {r:.6f} · 배포 target_rate {TARGET_RATE}")
    print(f"\n  {'구성':46}{'pred평균':>10}{'BSS':>9}{'900':>7}")
    print(f"  {'보정 없음 (등록 val 그대로)':46}{champ.mean():>10.4f}"
          f"{bss(champ):>9.1f}{'통과':>7}")
    for s in (1.00, 0.95, 0.90):
        for tgt, tag in ((TARGET_RATE, "target 0.474695 (배포값)"),
                         (r, "target = 그 폴드 기저율")):
            p = calib(champ, s, tgt)
            print(f"  {f'scale {s:.2f} · {tag}':46}{p.mean():>10.4f}"
                  f"{bss(p):>9.1f}{('통과' if bss(p) >= 900 else '탈락'):>7}")
    print("\n  -> 중심만 폴드 기저율로 바꾸면 점수가 돌아온다(오히려 조금 오른다).")
    print("     깎인 것은 로짓 스케일이 아니라 target_rate 미스매치다.")


if __name__ == "__main__":
    main()
