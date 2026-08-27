# -*- coding: utf-8 -*-
"""리더보드 점수로 v13 을 재보정한다.

주최측 확인 결과 리더보드 역추출이 허용되므로, 981 이라는 점수 자체를 정보로 쓴다.

원리
    BSS = 1e5 × (2A − V) / U      A = E[d(y−r)],  V = E[d²],  d = p − r
    V 는 예측만으로 로컬에서 잴 수 있다. A 는 라벨이 필요한데 점수가 알려준다.
        A = ( BSS·U/1e5 + V ) / 2
    예측 전체를 c 배 하면 BSS(c) = 1e5(2cA − c²V)/U 이고, 최적 c = A/V.
    최적에서 BSS* = 1e5·A²/(V·U).

두 가지를 되돌린다
    [1] target_rate  0.47469(학습 외삽) → 0.47353(리더보드 실측)
        오차 0.00116 → 0.  벌점 401,027 × 0.00116² ≈ 0.5점 회수.
    [2] 전역 배율     예측 편차를 c 배 해 분산을 최적점에 맞춘다.

측정은 2024 행의 season 을 2025 로 바꾼 분포에서 한다 (평가 데이터 미사용).

실행:
    python lb_recalib.py --lb 981
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))
STAGE = os.path.join(HERE, "_v13", "model")
sys.path.insert(0, HERE)

import script_v13 as S                              # noqa: E402

NROWS = 60000
R_LB = 0.47353          # 리더보드 포물선 적합으로 실측한 2025 평균 제구 성공률


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb", type=float, default=981.0, help="그 구성의 실제 리더보드 점수")
    ap.add_argument("--rows", type=int, default=NROWS)
    ap.add_argument("--a-fixed", type=float, default=None,
                    help="이전 측정에서 얻은 A 를 그대로 쓴다. 시드 수를 바꿔 V 만 달라진 경우.")
    a = ap.parse_args()

    if not os.path.exists(os.path.join(STAGE, "cb.npz")):
        sys.exit("없음: _v13/model — make_v13.py 를 먼저 돌리세요.")
    # 모델 파일은 STAGE 에서 읽되, **가중치는 반드시 원본(base)** 을 쓴다.
    # STAGE/params.json 은 직전 패키징 결과라 이미 배율이 적용돼 있을 수 있다.
    # 그걸로 V 를 재면 "이미 키운 예측"을 다시 재는 셈이 되어 c 가 거꾸로 나온다.
    params = json.load(open(os.path.join(STAGE, "params.json"), encoding="utf-8"))
    P13 = os.path.join(HERE, "model", "params_v13.json")
    BAK = P13 + ".bak"
    base = json.load(open(BAK if os.path.exists(BAK) else P13, encoding="utf-8"))
    r_old = params["target_rate"]
    wcb, wft, wmlp = (base["blend_w_cb"], base["blend_w_ft"], base["blend_w_mlp"])
    print("  가중치는 원본 기준: (%.4f, %.4f, %.4f)" % (wcb, wft, wmlp), flush=True)
    if os.path.exists(BAK):
        print("  (출처: params_v13.json.bak — 배율 적용 전 값)", flush=True)

    X = np.load(os.path.join(WORK, "X168.npy"), mmap_mode="r")
    season = np.load(os.path.join(WORK, "season.npy"))
    names = json.load(open(os.path.join(WORK, "meta.json")))["names80"]
    si = names.index("season")
    all24 = np.where(season == 2024)[0]
    idx = np.sort(np.random.default_rng(11).choice(all24, min(a.rows, len(all24)),
                                                   replace=False))
    Xs = np.asarray(X[idx], dtype=np.float32).copy()
    Xs[:, si] = 2025.0
    print("표본 %d행 (season→2025)" % len(idx), flush=True)

    def load(n):
        with np.load(os.path.join(STAGE, n)) as z:
            return {k: z[k] for k in z.files}
    cbz = load("cb.npz"); prep = load("prep.npz")
    bnds = [prep["b%d" % j] for j in range(int(prep["n_col"][0]))]

    print("CatBoost...", flush=True)
    acc = np.zeros(len(Xs)); n_cb = int(cbz["n_seeds"][0])
    for i in range(n_cb):
        acc += S.cb_predict(Xs, {k[3:]: v for k, v in cbz.items()
                                 if k.startswith("s%d_" % i)})
    p_cb = S.apply_calibration(acc / n_cb, params["model_cb"])
    import torch
    print("FT...", flush=True)
    p_ft = S.apply_calibration(
        S.torch_predict(S.prep_apply(Xs, bnds, False),
                        torch.load(os.path.join(STAGE, "ft.pt"), map_location="cpu",
                                   weights_only=False), "ft"), params["model_ft"])
    print("MLP...", flush=True)
    p_mlp = S.apply_calibration(
        S.torch_predict(S.prep_apply(Xs, bnds, True),
                        torch.load(os.path.join(STAGE, "mlp.pt"), map_location="cpu",
                                   weights_only=False), "mlp"), params["model_mlp"])

    # 현재 제출본과 동일한 블렌드
    p = r_old + wcb * (p_cb - r_old) + wft * (p_ft - r_old) + wmlp * (p_mlp - r_old)

    U = R_LB * (1 - R_LB); K = 1e5 / U
    d = p - R_LB
    V = float((d * d).mean())
    if a.a_fixed is not None:
        # 시드 수만 바뀐 경우: A 는 신호 성분이라 시드 평균으로 변하지 않는다.
        #   예측 = 신호 + 시드노이즈,  A = E[d(y−r)] = A_신호 (노이즈는 y 와 무상관)
        #   V = V_신호 + V_노이즈/n  → 시드가 늘면 V 만 줄고 A 는 그대로다.
        A = a.a_fixed
        print("  A 를 이전 측정값 %.6f 로 고정 (시드 변경 시나리오)" % A, flush=True)
    else:
        A = (a.lb * U / 1e5 + V) / 2.0
    c_opt = A / V
    bss_now = 1e5 * (2 * A - V) / U
    bss_opt = 1e5 * A * A / (V * U)

    # target_rate 교체로 회수되는 몫 (평균 이동에 따른 벌점)
    gap = r_old - R_LB
    pen = K * gap * gap

    print()
    print("=" * 66)
    print("[LB 재보정]   실제 점수 %.1f" % a.lb)
    print("=" * 66)
    print("  로컬 실측    sd(p) %.5f   V %.6f" % (np.sqrt(V), V))
    print("  %s A %.6f" % ("고정값     " if a.a_fixed else "점수에서 역산", A))
    print()
    if a.a_fixed:
        # A 를 고정했으므로 bss_now 는 순환논법이 아닌 **독립 예측**이다.
        print("  이 구성(배율 1.0)의 예상 점수  %.1f   (기준 %.1f 대비 %+.1f)"
              % (bss_now, a.lb, bss_now - a.lb))
        print("    ↑ 시드 수 · 모델 변경으로 V 가 달라진 몫")
    else:
        print("  (검산) %.1f  ← A 를 점수에서 역산했으므로 항등식이다 (검증력 없음)"
              % bss_now)
    print()
    print("  [1] target_rate  %.5f → %.5f            회수 %+.1f점" % (r_old, R_LB, pen))
    print("  [2] 전역 배율    c = A/V = %.4f          회수 %+.1f점"
          % (c_opt, bss_opt - bss_now))
    print()
    print("  최종 예상 %.1f  (= 최적 배율에서의 BSS)" % (bss_opt + pen))
    print("    기준 %.1f 대비 %+.1f점" % (a.lb, bss_opt + pen - a.lb))
    print()
    print("  새 가중치 (기존 × %.4f)" % c_opt)
    print("    make_v13.py 는 params 를 params_v13.json 에서 읽는다.")
    print("    아래 값으로 model/params_v13.json 을 갱신하면 된다:")
    print("      target_rate  %.5f" % R_LB)
    print("      blend_w_cb   %.4f  (기존 %.4f)" % (wcb * c_opt, wcb))
    print("      blend_w_ft   %.4f  (기존 %.4f)" % (wft * c_opt, wft))
    print("      blend_w_mlp  %.4f  (기존 %.4f)" % (wmlp * c_opt, wmlp))
    print()
    print("  주의: V 는 2024행의 season 만 바꾼 모사 분포에서 잰 값이다.")
    print("        실제 2025 분포와 다소 다를 수 있어 회수량은 추정치다.")
    print("=" * 66)

    json.dump({"lb": a.lb, "V": V, "A": A, "c_opt": c_opt,
               "target_rate": R_LB,
               "blend_w_cb": wcb * c_opt, "blend_w_ft": wft * c_opt,
               "blend_w_mlp": wmlp * c_opt,
               "expected_gain": float(bss_opt + pen - a.lb)},
              open(os.path.join(HERE, "model", "lb_recalib.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장: model/lb_recalib.json")


if __name__ == "__main__":
    main()
