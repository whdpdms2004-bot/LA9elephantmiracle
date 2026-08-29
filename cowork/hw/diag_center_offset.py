"""배포본 출력 평균이 목표에서 벗어나 생기는 손실과, 보정 시 회수량.

## 왜 보는가

BSS 는 예측 평균이 라벨 평균에서 벗어나면 K = 1e5/(r(1-r)) ~= 401,000 배로
벌점이 붙는다. 배포본은 `center_shift` 로 이미 한 번 맞췄는데
(`sj_stdmlp.md:195`), 최종 평균이 0.476804 로 자기 target_rate 0.474695 보다
**+0.0021 높다**. DECK 부록 A 가 "최대 손실 ~1.8점" 이라 적어둔 그 자리다.

여기서는 (1) 그 손실이 참 기저율 가정에 따라 얼마나 되는지, (2) 참값이
불확실할 때 어디로 맞추는 게 최적인지, (3) 이 종류 변경의 전이율은 얼마인지를
같이 본다.

`target_rate` 자체는 두 독립 추정이 0.0012 안에서 일치해 문제가 없어 보인다 --
train 6년 선형 외삽 0.474695(배포) 와 cw v16 의 0.47353. 이 스크립트가 겨냥하는
것은 target 이 아니라 **출력이 그 target 을 못 맞추는 잔차**다.

실행:
    python cowork/hw/diag_center_offset.py
"""
import numpy as np

DEPLOY_MEAN = 0.476804      # sj_stdmlp.md:195 (DECK 부록 A)
T_TRAIN = 0.474695          # 배포 target_rate, train 6년 선형 외삽
T_ALT = 0.47353             # cw v16 params.json 의 독립 추정


def K(r):
    return 1e5 / (r * (1 - r))


def main():
    print("=" * 70)
    print("1. 지금 중심이 얼마나 어긋나 있고 몇 점인가")
    print("=" * 70)
    print(f"  배포본 실제 출력 평균  {DEPLOY_MEAN:.6f}\n")
    print(f"  {'참 기저율 가정':28}{'offset':>11}{'손실(BSS)':>11}")
    for t, tag in ((T_TRAIN, "0.474695 (train 외삽)"),
                   (T_ALT, "0.47353 (cw v16 독립추정)"),
                   ((T_TRAIN + T_ALT) / 2, "둘의 중간")):
        off = DEPLOY_MEAN - t
        print(f"  {tag:28}{off:>+11.6f}{K(t)*off*off:>11.2f}")

    print()
    print("=" * 70)
    print("2. 참값이 불확실할 때 어디로 맞추나")
    print("=" * 70)
    print("  E[벌점] = K * ((목표 - 평균참값)^2 + 참값분산)")
    print("  분산항은 목표를 어디로 두든 같으므로 평균 참값에 맞추는 것이 최적이다.")
    mid = (T_TRAIN + T_ALT) / 2
    need = DEPLOY_MEAN - mid
    print(f"\n  두 추정의 중간      {mid:.6f}")
    print(f"  필요한 추가 shift   {need:+.6f}")
    print(f"  회수 기대치         {K(mid)*need*need:.2f} BSS")

    print()
    print("=" * 70)
    print("3. 이 종류 변경의 전이율 (PLAN_next §0 실측표)")
    print("=" * 70)
    print("  결합가중 - 합 이탈(해로움)   val -6.1  -> Public -6.7   전이율 1.1")
    print("  피처 추가 (id_freq)         val +21.0 -> Public +1.33  전이율 0.06")
    print("  표현 교체 (std_mlp)         val +22.3 -> Public +1.68  전이율 0.075")
    print("\n  -> 평균·분산 층은 팀 표에서 유일하게 1.0 이상으로 전이된다.")
    print("     회수분이 val 에만 남지 않고 Public 에 거의 그대로 나온다는 뜻이다.")


if __name__ == "__main__":
    main()
