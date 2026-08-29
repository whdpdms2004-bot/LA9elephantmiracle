"""관문 델타의 폴드 표본오차를 잰다 -- 05_gate_noise_hw.md 재현 코드.

## 왜

score_val.py 의 TOL 은 0.0 이라 -0.1 도 "하락" 으로 읽어 기각한다. 주석은
"시드 산포보다 작은 차이를 상승으로 읽지 않는다" 인데 값이 0 이라 구현이 안 돼
있었다. 고치려면 "얼마까지가 잡음인가" 를 알아야 하는데, 이 스크립트가 그것을
잰다 -- 그리고 **쌍마다 다르다**는 것이 결론이다(고정 TOL 로 못 고치는 이유).

## 무엇을

페어드 행 부트스트랩: 같은 행 인덱스를 두 예측에 동시에 리샘플한다. 모델 간
공통 변동이 상쇄돼 "이 두 모델의 차이" 에만 붙는 오차가 남는다.

재학습 잡음(PROGRESS_workflow §3.5 의 ±2 BSS)은 안 잡힌다 -- 그건 같은 구성을
다시 학습해야 나오는 값이다. 이 CI 는 하한이다.

실행:
    python cowork/hw/measure_gate_noise.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "performance_tracking" / "tools"))

from common import load_labels, load_pred  # noqa: E402

N_BOOT = 400
PAIRS = [("sj_stdmlp", "sj_grid_w060"), ("sj_grid_w060", "sj_e2var"),
         ("sj_stdmlp", "sj_e2var"), ("sj_e2var", "sj_cb_ft_fonly")]
CELLS = [(2024, "all"), (2022, "R")]


def bss(y, p):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def cell(season, subgroup, *names):
    lab = load_labels(season)
    y = lab["y"].to_numpy(np.float64)
    ps = [np.clip(load_pred(n, season, lab), 1e-9, 1 - 1e-9) for n in names]
    if subgroup in ("R", "F"):
        m = lab["game_type"].to_numpy() == subgroup
        y = y[m]
        ps = [p[m] for p in ps]
    return (y, *ps)


def boot_ci(y, cand, base, n_boot=N_BOOT, seed=1):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        d[k] = bss(y[i], cand[i]) - bss(y[i], base[i])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d.mean(), lo, hi, (hi - lo) / 2


def main():
    print("=" * 74)
    print("1. 서로 다른 모델끼리 -- CI 반폭이 크다")
    print("=" * 74)
    print(f"  {'쌍':34} {'시즌·구간':11} {'Δ':>9} {'CI 반폭':>9}")
    for a, b in PAIRS:
        for season, sub in CELLS:
            try:
                y, pa, pb = cell(season, sub, a, b)
            except Exception as e:
                print(f"  {a+' vs '+b:34} {str(season)+' '+sub:11}  (스킵 {type(e).__name__})")
                continue
            m, lo, hi, hw = boot_ci(y, pa, pb)
            print(f"  {a+' vs '+b:34} {str(season)+' '+sub:11} {m:+9.2f} {hw:>9.2f}")

    print()
    print("=" * 74)
    print("2. 거의 같은 모델끼리 -- 소폭 변경 후보를 모사 (eps 만큼만 민다)")
    print("=" * 74)
    print(f"  {'eps':>5} {'rho':>9} {'시즌·구간':11} {'Δ':>9} {'CI 반폭':>9}  판정")
    for season, sub in CELLS:
        y, a, b = cell(season, sub, "sj_stdmlp", "sj_grid_w060")
        for eps in (0.30, 0.10, 0.03):
            cand = np.clip((1 - eps) * a + eps * b, 1e-9, 1 - 1e-9)
            m, lo, hi, hw = boot_ci(y, cand, a)
            v = "유의" if (lo > 0 or hi < 0) else "무증거(0 포함)"
            rho = np.corrcoef(cand, a)[0, 1]
            print(f"  {eps:5.2f} {rho:9.5f} {str(season)+' '+sub:11} "
                  f"{m:+9.2f} {hw:>9.2f}  {v}")
        print()

    print("  결론: 두 예측이 닮을수록 CI 가 좁아진다(페어드라 공통 변동이 상쇄).")
    print("        3피처만 바뀐 후보는 ±0.2~0.6 까지 분해되고, 다른 모델 비교는")
    print("        ±8~25 다. 상수 TOL 하나로 덮을 수 없다 -- 쌍마다 재야 한다.")


if __name__ == "__main__":
    main()
