# -*- coding: utf-8 -*-
"""[항목 F] `ft` 를 **원시 예측**으로 다시 재고, `id_freq` 와 표현을 함께 본다.

## 왜 이렇게 바꿔 잡았나

원래 항목 F 는 "`id_freq` 완전 스택 2폴드 확정" 이었다 — `S1_idfreq` 를 FT 단계에서
중단시켜 cb 단독 기록만 남은 것을 메우는 일. 그런데 [축2] 에서 배운 것이 있다.

**`mlp` 에 `id_freq` 를 넣는 것만으로는 효과가 0 이었다** (`q64:176` 가중 0.00).
효과를 낸 것은 피처가 아니라 **표현**이었다 (`std:176` 가중 0.15, +12.3).

그래서 `ft` 도 같은 두 축으로 본다. 피처만 넣고 끝내면 [축2] 에서 이미 본 함정을
그대로 밟는다.

## arm

    q64:168    현행 (분위수 순위, 마스크 없음 — 토큰당 1열이라 열을 늘리면 attention 이 4배)
    q64:176    id_freq 만 추가
    std:176    id_freq + 로버스트 z 점수

`ft` 는 열 하나가 토큰 하나라 마스크 열을 붙이지 않는다 (`with_mask=False`).

## 판정

**원시 예측만 쓴다** (`run_arm.calib` 는 평가 라벨로 스케일을 맞춘다 — 금지).
기준은 현행 배포 구성이고, 배포 순서(모델별 `apply_calibration` 후 확률결합)로
재현해 판정한다.

    python ft_raw.py --arms q64:168,q64:176,std:176 --seeds 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
sys.path.insert(0, str(ROOT / "cowork" / "cw" / "v17" / "src"))
sys.path.insert(0, str(FINAL / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="q64:168,q64:176,std:176")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    arms = []
    for tok in a.arms.split(","):
        k, c = tok.split(":")
        arms.append((k.strip(), int(c)))

    import importlib.util as iu
    sp = iu.spec_from_file_location(
        "pt_common", ROOT / "performance_tracking" / "tools" / "common.py")
    mod = iu.module_from_spec(sp)
    sp.loader.exec_module(mod)
    load_labels = mod.load_labels

    from run_arm import fit_torch, load_base
    from prep_mlp import apply_prep, make_prep
    import atoms as A

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]

    for fold in (2024, 2022):
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        y = load_labels(fold)["y"].to_numpy(np.float64)
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        X176 = np.concatenate([np.asarray(X), E], axis=1)
        del E
        for kind, ncol in arms:
            fp = FINAL / "preds" / ("FTX_%s_%d_%d.npy" % (kind, ncol, fold))
            if fp.exists():
                print("  fold%d %-6s %d열  (이미 있음, 단독 %.1f)"
                      % (fold, kind, ncol, bss(np.load(fp), y)), flush=True)
                continue
            src = X176 if ncol == 176 else np.asarray(X)
            Xt_raw = np.ascontiguousarray(src[tri])
            Xv_raw = np.ascontiguousarray(src[va])
            t0 = time.time()
            prep = make_prep(Xt_raw, kind)
            # ft 는 토큰당 1열이라 마스크를 붙이지 않는다 (열을 2배로 늘리면
            # attention 이 4배 무거워진다 — cw README §27.2)
            Xt = apply_prep(Xt_raw, prep, False)
            Xv = apply_prep(Xv_raw, prep, False)
            del Xt_raw, Xv_raw
            acc = np.zeros(len(va))
            for sd in range(a.seeds):
                ts = time.time()
                acc += np.clip(fit_torch("ft", Xt, y_all[tri], Xv, sd), 1e-6, 1 - 1e-6)
                print("     seed%d %.0f초" % (sd, time.time() - ts), flush=True)
            p = acc / a.seeds
            np.save(fp, p)
            print("  fold%d %-6s %d열 -> 입력 %d  %.0f초  단독 %.1f"
                  % (fold, kind, ncol, Xt.shape[1], time.time() - t0, bss(p, y)),
                  flush=True)
            del Xt, Xv
        del X176
    print("\n원시 예측 저장 완료 — 판정은 ft_judge 로 따로 한다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
