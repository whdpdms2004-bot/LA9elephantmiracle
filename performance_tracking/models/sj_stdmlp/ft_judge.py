# -*- coding: utf-8 -*-
"""[항목 F] `ft` arm 을 **배포 순서 그대로** 판정한다.

배포는 모델별 `apply_calibration`(로짓 아핀 + 목표성공률 중심이동 + 클리핑)을
적용한 **뒤** 확률 공간에서 섞는다. 로짓 아핀과 확률 결합은 교환되지 않으므로
원시 예측으로 적합한 가중과 다르다. 그래서 판정도 배포 순서로 한다.

기준은 현행 최고 구성(`sj_stdmlp`: cb2 + 현행 ft + std_mlp)이고,
`ft` 자리만 바꿔 넣어 비교한다. 가중은 각 구성마다 두 폴드 평균으로 재적합한다
(cw 내부는 합 제약을 걸지 않는다 — 팀 층 `center_shift` 가 스케일을 흡수한다).

`ft` 는 `id_freq` 를 넣는 것만으로 효과가 있을지, 아니면 [축2] 의 `mlp` 처럼
**표현**을 바꿔야 하는지가 이 실험의 질문이다.

    python ft_judge.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

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

RIDGE = 0.02
# 배포 계기의 환산율. 리더보드로 검증된 d5->d6 가 이 계기에서 val2024 +4.4 였고
# Public +3.706 였다.
RATE = 3.706 / 4.4
BASE_PUBLIC = 1080.425


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def fitw(M, y):
    r = y.mean()
    D = M - r
    A = D.T @ (y - r) / len(y)
    Q = D.T @ D / len(y)
    Q = Q + RIDGE * np.trace(Q) / len(Q) * np.eye(len(Q))
    return np.linalg.solve(Q, A)


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
    ap.add_argument("--arms", default="q64:168,q64:176,std:176")
    a = ap.parse_args()

    from common import load_labels
    P = FINAL / "preds"
    par = json.loads(zipfile.ZipFile(a.zip).read("model/cw/model/params.json"))
    Y = {f: load_labels(f)["y"].to_numpy(np.float64) for f in (2024, 2022)}

    def mats(ftfile):
        out = []
        for f in (2024, 2022):
            out.append((np.column_stack([
                cal(np.load(P / ("E2_var_cb2_a0.15_%d.npy" % f)), par["model_cb"]),
                cal(np.load(P / ("%s_%d.npy" % (ftfile, f))), par["model_ft"]),
                cal(np.load(P / ("PREP_std_176_%d.npy" % f)), par["model_mlp"])]), Y[f]))
        return out

    def score(ms, w=None):
        if w is None:
            w = np.mean([fitw(M, y) for M, y in ms], axis=0)
        return [bss(np.clip(y.mean() + (M - y.mean()) @ w, 1e-6, 1 - 1e-6), y)
                for M, y in ms], w

    print("=" * 96)
    print("[항목 F] ft arm 판정 — 배포 순서 재현 (apply_calibration 후 확률결합)")
    print("  기준 = sj_stdmlp (cb2 + 현행 ft + std_mlp), 가중은 구성마다 재적합")
    print("=" * 96)
    print("%-22s %10s %10s %10s   %9s %9s   %s"
          % ("ft 구성", "ft단독24", "val2024", "val2022", "Δ24", "Δ22", "Public 예상"))
    print("-" * 96)

    y24 = Y[2024]
    # ★ 기준선은 **원시** ft 여야 한다. `S1_base__ft` 는 `run_arm.calib` 가 평가 라벨로
    # 스케일을 맞춘 예측이라, 이 파이프라인에서 `apply_calibration` 을 또 걸면 이중이다.
    # 현행 구성을 원시로 다시 학습한 것이 곧 `FTX_q64_168` arm 이므로 그것을 기준으로 쓴다.
    if not (P / "FTX_q64_168_2024.npy").exists():
        sys.exit("[중단] 원시 기준선 FTX_q64_168 이 아직 없다")
    base, wb = score(mats("FTX_q64_168"))
    print("%-22s %10.1f %10.1f %10.1f   %9s %9s   %.1f   w %.3f/%.3f/%.3f"
          % ("q64:168 (현행·원시)", bss(np.load(P / "FTX_q64_168_2024.npy"), y24),
             base[0], base[1], "기준", "기준", BASE_PUBLIC, wb[0], wb[1], wb[2]))
    ref, _ = score(mats("S1_base__ft"))
    print("%-22s %10.1f %10.1f %10.1f   %9s %9s   (참고: 이중보정)"
          % ("  S1_base__ft (보정본)", bss(np.load(P / "S1_base__ft_2024.npy"), y24),
             ref[0], ref[1], "", ""))
    for tok in a.arms.split(","):
        k, c = tok.split(":")
        if tok == "q64:168":
            continue
        f = "FTX_%s_%s" % (k, c)
        if not (P / ("%s_2024.npy" % f)).exists():
            print("%-22s   (아직 없음)" % tok)
            continue
        s, w = score(mats(f))
        print("%-22s %10.1f %10.1f %10.1f   %+9.1f %+9.1f   %.1f   w %.3f/%.3f/%.3f"
              % (tok, bss(np.load(P / ("%s_2024.npy" % f)), y24), s[0], s[1],
                 s[0] - base[0], s[1] - base[1],
                 BASE_PUBLIC + (s[0] - base[0]) * RATE + 0.0, w[0], w[1], w[2]))
    print("\n(현행 ft 는 평가라벨 보정을 거친 예측이라 단독값이 유리하다 —")
    print(" 새 arm 은 원시다. 판정은 단독이 아니라 블렌드 Δ 로 한다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
