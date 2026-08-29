"""F행 전용 라우팅 재판정 -- 정직 hw 예측(hw_v12_honest)으로 다시 잰다.

## 왜 다시 재나

`results.csv` 의 `sj_route_F` 는 정방향 +8.7 / 역방향 -5.9 로 기각됐다. 그때 쓴
`val/hw_v12_2024.csv` 가 나중에 오염(eval_set 으로 평가시즌 라벨을 조기종료에
사용)으로 밝혀졌고, 정직본은 F 614.8 -> 488.3 으로 크게 내려갔다.
**입력이 달라졌으므로 결론도 다시 재야 한다.**

## 무엇을 재나

    R행: 챔피언 그대로       p = r + 0.45461*(p_cw모듈 - r) + 0.64333*(p_sj3way - r) - 0.003223
    F행: 챔피언에 hw 를 L    p_F = (1-L)*p_champ + L*p_hw

챔피언 가중은 **배포본 그대로**다 (build_submit_zip.py:10, DECK D4 "한 번도 안 바꿨다").
★ 2026-08-29 정정: 처음엔 {0.6, 0.4}로 잘못 썼다. 그건 sj_grid_w060.md 의 *후보*이고,
채택된 제출 4b 는 팀 가중을 바꾸지 않았다. 기준선이 바뀌면 결론도 바뀐다 --
역방향이 +0.17 에서 -1.32 로 뒤집힌다.
재적합 기준선을 쓰면 기준선의 전이오차가 이득으로 둔갑한다 -- 항목 P 가 정확히
그 함정에서 뒤집혔다(`group_by_perform/RESULTS.md` §9). 그래서 고정값을 쓴다.

## 판정 규약

- 정직 월전방분할 **양방향** (월3-6 적합구간 / 월7-10 평가구간을 서로 바꿔가며)
- 행 부트스트랩 400회로 Δ 의 95% 신뢰구간
- 2022 R 비하락 관문 (F행만 건드리므로 구조적으로 0.00 이 나와야 정상)

재학습 없이 등록된 val 예측 파일만으로 계산한다.

실행:
    python cowork/hw/eval_f_routing_honest.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PT = REPO / "performance_tracking"
W_CHAMP = np.array([0.45461, 0.64333])   # build_submit_zip.py:10 · DECK D4 "한 번도 안 바꿨다"
CENTER_SHIFT = 0.003223
LAMBDAS = [0.10, 0.20]
N_BOOT = 400
FIT_MONTHS = (3, 4, 5, 6)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def load_pred(name, season):
    f = PT / "val" / f"{name}_{season}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)[["row_id", "pred"]].rename(columns={"pred": name})


def build(season, cw_member, sj_member, hw_member="hw_v12_honest"):
    D = pd.read_csv(PT / ".cache" / f"labels_{season}.csv")
    for m in (cw_member, sj_member, hw_member):
        d = load_pred(m, season)
        if d is None:
            raise FileNotFoundError(f"{m}_{season}.csv 없음")
        D = D.merge(d, on="row_id", how="inner")
    y = D["y"].to_numpy(float)
    r = y.mean()
    P = D[[cw_member, sj_member]].to_numpy(float)
    champ = np.clip(r + (P - r) @ W_CHAMP - CENTER_SHIFT, 1e-6, 1 - 1e-6)
    return D, y, champ, D[hw_member].to_numpy(float)


def route(champ, hw, F, L):
    p = champ.copy()
    p[F] = np.clip((1 - L) * champ[F] + L * hw[F], 1e-6, 1 - 1e-6)
    return p


def main():
    print("=" * 78)
    print("2024 -- 챔피언 = 0.45461*sj_stdmlp + 0.64333*sj3way - 0.003223 (배포본)")
    print("=" * 78)
    D, y, champ, hw = build(2024, "sj_stdmlp", "sj3way")
    F = (D["game_type"] == "F").to_numpy()
    mo = D["game_month"].to_numpy()
    A = np.isin(mo, FIT_MONTHS)
    B = ~A
    print(f"  병합 {len(D):,}행 · F {F.sum():,}행({F.mean()*100:.1f}%)")
    print(f"  챔피언 val2024 all={bss(champ, y):.1f} "
          f"F={bss(champ[F], y[F]):.1f} · hw_honest F={bss(hw[F], y[F]):.1f}")

    rng = np.random.default_rng(2026)
    print(f"\n  {'구간':14} {'L':>5} {'평균Δ':>9} {'2.5%':>9} {'97.5%':>9}  판정")
    for L in LAMBDAS:
        p = route(champ, hw, F, L)
        for nm, msk in [("2024 월7-10", B), ("2024 월3-6", A),
                        ("2024 전체", np.ones(len(y), bool))]:
            yy, cc, pp = y[msk], champ[msk], p[msk]
            n = len(yy)
            ds = np.array([bss(pp[i], yy[i]) - bss(cc[i], yy[i])
                           for i in (rng.integers(0, n, n) for _ in range(N_BOOT))])
            lo, hi = np.percentile(ds, [2.5, 97.5])
            v = "0을 넘지 않음(유의)" if lo > 0 else (
                "0 포함(잡음과 구분 안 됨)" if hi > 0 else "유의하게 음수")
            print(f"  {nm:14} {L:5.2f} {ds.mean():+9.2f} {lo:+9.2f} {hi:+9.2f}  {v}")
        print()

    print("=" * 78)
    print("2022 관문 -- sj3way_2022 가 없어 sj3way_nv 로 근사 (챔피언과 구성 다름)")
    print("=" * 78)
    D2, y2, champ2, hw2 = build(2022, "sj_stdmlp", "sj3way_nv")
    F2 = (D2["game_type"] == "F").to_numpy()
    R2 = ~F2
    print(f"  병합 {len(D2):,}행 · F {F2.sum():,}행 · 2022 F기저율 {y2[F2].mean():.4f}")
    print(f"  챔피언(근사) all={bss(champ2, y2):.1f} R={bss(champ2[R2], y2[R2]):.1f} "
          f"F={bss(champ2[F2], y2[F2]):.1f} · hw_honest F={bss(hw2[F2], y2[F2]):.1f}")
    ba, br, bf = bss(champ2, y2), bss(champ2[R2], y2[R2]), bss(champ2[F2], y2[F2])
    print(f"\n  {'L':>5} {'2022 all Δ':>12} {'2022 R Δ(관문)':>16} {'2022 F Δ':>10}")
    for L in LAMBDAS:
        p2 = route(champ2, hw2, F2, L)
        print(f"  {L:5.2f} {bss(p2, y2)-ba:+12.2f} {bss(p2[R2], y2[R2])-br:+16.2f} "
              f"{bss(p2[F2], y2[F2])-bf:+10.2f}")
    print("\n  (R행을 건드리지 않으므로 2022 R Δ 는 구조적으로 0.00 이어야 정상)")


if __name__ == "__main__":
    main()
