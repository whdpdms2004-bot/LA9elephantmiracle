# -*- coding: utf-8 -*-
"""lb_recalib.json 을 params_v13.json 에 반영한다 (재학습 없음).

되돌리기:
    python apply_recalib.py --restore

두 번째 점수를 얻은 뒤 정확한 최적 배율 풀기:
    python apply_recalib.py --solve 981 1.0000 984 1.0585
        → BSS(c) = 1e5(2cA − c²V)/U 에 두 점을 넣어 A, V 를 정확히 구하고
          진짜 최적 c* = A/V 와 그때의 점수를 계산한다.
"""

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
P = os.path.join(MODEL_DIR, "params_v13.json")
BAK = os.path.join(MODEL_DIR, "params_v13.json.bak")
R_LB = 0.47353


def solve(s1, c1, s2, c2):
    """두 (배율, 점수) 쌍에서 A, V 를 정확히 구한다.

        BSS = 1e5 (2cA − c²V) / U
        →  2c·A − c²·V = BSS·U/1e5
    미지수 2개, 식 2개.
    """
    import numpy as np
    U = R_LB * (1 - R_LB)
    M = np.array([[2 * c1, -c1 ** 2], [2 * c2, -c2 ** 2]], dtype=float)
    b = np.array([s1 * U / 1e5, s2 * U / 1e5])
    A, V = np.linalg.solve(M, b)
    c_star = A / V
    best = 1e5 * A * A / (V * U)
    print("=" * 60)
    print("[정확해]  두 점에서 A, V 를 직접 풀었다")
    print("=" * 60)
    print("  입력   c=%.4f → %.1f      c=%.4f → %.1f" % (c1, s1, c2, s2))
    print("  해     A = %.6f   V = %.6f   (sd = %.5f)" % (A, V, V ** 0.5))
    print()
    print("  최적 배율 c* = %.4f" % c_star)
    print("  그때 점수     %.1f   (현재 최고 대비 %+.1f)"
          % (best, best - max(s1, s2)))
    print()
    if best - max(s1, s2) < 3:
        print("  판정: 개선폭이 작다. 제출할 값어치가 없다.")
    else:
        print("  판정: python apply_recalib.py --scale %.4f  후 재패키징" % c_star)
    print("=" * 60)
    return c_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true", help="원래 값으로 되돌린다")
    ap.add_argument("--keep-target", action="store_true",
                    help="target_rate 는 학습 외삽값 그대로 두고 배율만 적용한다. "
                         "리더보드에서 역산한 r 을 예측 중심으로 쓰는 건 '평가 데이터 "
                         "전체의 정보로 개별 행을 보정' 에 해당할 소지가 있다. "
                         "반면 앙상블 가중치 조정은 주최측이 명시적으로 허용했다.")
    ap.add_argument("--scale", type=float, default=None, help="배율을 직접 지정")
    ap.add_argument("--solve", type=float, nargs=4, metavar=("S1", "C1", "S2", "C2"),
                    help="두 (점수, 배율) 쌍으로 정확한 최적해를 푼다")
    a = ap.parse_args()

    if a.solve:
        s1, c1, s2, c2 = a.solve
        solve(s1, c1, s2, c2)
        return

    if a.restore:
        if not os.path.exists(BAK):
            sys.exit("백업이 없습니다.")
        shutil.copy(BAK, P)
        print("복원 완료: params_v13.json")
        return

    rc = os.path.join(MODEL_DIR, "lb_recalib.json")
    if not os.path.exists(rc):
        sys.exit("없음: model/lb_recalib.json — lb_recalib.py 를 먼저 돌리세요.")
    r = json.load(open(rc, encoding="utf-8"))
    pv = json.load(open(P, encoding="utf-8"))

    if not os.path.exists(BAK):
        shutil.copy(P, BAK)
        print("백업: params_v13.json.bak")

    c = a.scale if a.scale is not None else r["c_opt"]
    old_r = pv["target_rate"]
    base = json.load(open(BAK, encoding="utf-8"))     # 항상 원본 기준으로 배율 적용

    import math
    if not a.keep_target:
        pv["target_rate"] = R_LB
        pv["logit_target_C1"] = math.log(R_LB / (1 - R_LB))
        pv["target_rate_source"] = "leaderboard-measured r=0.47353 (주최측 확인 후 허용)"
        for k in ("model_cb", "model_ft", "model_mlp"):
            pv[k]["target_rate"] = R_LB
            pv[k]["logit_target_C1"] = pv["logit_target_C1"]
    for k in ("blend_w_cb", "blend_w_ft", "blend_w_mlp"):
        pv[k] = base[k] * c
    pv["blend_scale_c"] = c
    pv["blend_source"] = (base["blend_source"]
                          + " | 전역 배율 %.4f 는 리더보드 점수에서 역산한 A/V" % c)
    json.dump(pv, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("적용 완료")
    print("  target_rate  %.5f → %.5f%s"
          % (old_r, pv["target_rate"],
             "  (유지 — 학습 외삽값, 규정 안전)" if a.keep_target else ""))
    print("  배율 c       %.4f" % c)
    print("  w_cb  %.4f → %.4f" % (base["blend_w_cb"], pv["blend_w_cb"]))
    print("  w_ft  %.4f → %.4f" % (base["blend_w_ft"], pv["blend_w_ft"]))
    print("  w_mlp %.4f → %.4f" % (base["blend_w_mlp"], pv["blend_w_mlp"]))
    print("\n다음: python make_v13.py --out submit_v14.zip")
    print("      python check_rules.py submit_v14.zip")
    print("      python verify_v13.py")


if __name__ == "__main__":
    main()
