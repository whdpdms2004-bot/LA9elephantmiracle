# -*- coding: utf-8 -*-
"""v13 베이스율 검증 — 제출 직전 마지막 관문.

K≈401,000 이라 최종 평균이 0.01 어긋나면 40점이 날아간다.
v8 때 이 검사가 60점짜리 오차를 잡아냈다. 5행 스모크로는 절대 안 보인다.

2024 학습 행의 season 만 2025 로 바꿔 테스트 분포를 모사하고,
패키징된 모델(_v13/model)을 그대로 써서 세 계열을 돌린다.

실행:
    python verify_v13.py        # 약 2분
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))
STAGE = os.path.join(HERE, "_v13", "model")
sys.path.insert(0, HERE)

import script_v13 as S                              # noqa: E402

NROWS = 40000


def main():
    if not os.path.exists(os.path.join(STAGE, "cb.npz")):
        sys.exit("없음: _v13/model — 먼저 python make_v13.py 를 돌리세요.")
    params = json.load(open(os.path.join(STAGE, "params.json"), encoding="utf-8"))
    r = params["target_rate"]; K = 100000.0 / (r * (1 - r))
    wcb, wft, wmlp = (params["blend_w_cb"], params["blend_w_ft"], params["blend_w_mlp"])

    X = np.load(os.path.join(WORK, "X168.npy"), mmap_mode="r")
    season = np.load(os.path.join(WORK, "season.npy"))
    names = json.load(open(os.path.join(WORK, "meta.json")))["names80"]
    si = names.index("season")

    all24 = np.where(season == 2024)[0]
    idx = np.sort(np.random.default_rng(0).choice(all24, min(NROWS, len(all24)),
                                                  replace=False))
    Xs = np.asarray(X[idx], dtype=np.float32).copy()
    Xs[:, si] = 2025.0
    print("표본 %d행 (2024 전체 %d행에서 무작위, season→2025)" % (len(idx), len(all24)),
          flush=True)

    def load(n):
        with np.load(os.path.join(STAGE, n)) as z:
            return {k: z[k] for k in z.files}
    cbz = load("cb.npz")
    prep = load("prep.npz")
    bnds = [prep["b%d" % j] for j in range(int(prep["n_col"][0]))]

    print("CatBoost...", flush=True)
    acc = np.zeros(len(Xs))
    n_cb = int(cbz["n_seeds"][0])
    for i in range(n_cb):
        blob = {k[3:]: v for k, v in cbz.items() if k.startswith("s%d_" % i)}
        acc += S.cb_predict(Xs, blob)
    p_cb = S.apply_calibration(acc / n_cb, params["model_cb"])

    import torch
    print("FT...", flush=True)
    Zf = S.prep_apply(Xs, bnds, False)
    p_ft = S.apply_calibration(
        S.torch_predict(Zf, torch.load(os.path.join(STAGE, "ft.pt"),
                                       map_location="cpu", weights_only=False), "ft"),
        params["model_ft"])
    del Zf
    print("MLP...", flush=True)
    Zm = S.prep_apply(Xs, bnds, True)
    p_mlp = S.apply_calibration(
        S.torch_predict(Zm, torch.load(os.path.join(STAGE, "mlp.pt"),
                                       map_location="cpu", weights_only=False), "mlp"),
        params["model_mlp"])
    del Zm

    p = np.clip(r + wcb * (p_cb - r) + wft * (p_ft - r) + wmlp * (p_mlp - r),
                params["floor"], params["ceil"])

    print()
    print("=" * 64)
    print("[VERIFY v13]   목표 %.5f,  K=%.0f" % (r, K))
    print("=" * 64)
    print("%-14s %10s %11s %10s" % ("", "평균", "오차", "벌점"))
    for lb, q in (("p_cb", p_cb), ("p_ft", p_ft), ("p_mlp", p_mlp), ("최종 blend", p)):
        d = q.mean() - r
        print("%-14s %10.5f %+11.5f %10.1f" % (lb, q.mean(), d, K * d * d))
    print()
    print("  sd : cb %.5f  ft %.5f  mlp %.5f  최종 %.5f"
          % (p_cb.std(), p_ft.std(), p_mlp.std(), p.std()))
    print("  rho: cb-ft %.4f  cb-mlp %.4f  ft-mlp %.4f"
          % (np.corrcoef(p_cb, p_ft)[0, 1], np.corrcoef(p_cb, p_mlp)[0, 1],
             np.corrcoef(p_ft, p_mlp)[0, 1]))
    clipped = int(((p <= params["floor"]) | (p >= params["ceil"])).sum())
    print("  클리핑에 걸린 행: %d / %d" % (clipped, len(p)))

    d = p.mean() - r
    ok = abs(d) < 0.004 and p.std() > 0.005
    print()
    print("통과 — 제출해도 됩니다." if ok else "실패 — 제출하지 마세요.")
    if not ok and abs(d) >= 0.004:
        print("  베이스율 오차 %.5f → 벌점 %.0f점" % (d, K * d * d))
    print("=" * 64)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
