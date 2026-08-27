# -*- coding: utf-8 -*-
"""v12 준비 — 시즌폼 피처 통합, 룩업 생성, 80피처 행렬 구축.

네 가지를 한다.
  [1] 회귀 검사   season_form.build_training_features 가 project_v12.py 의 실험 코드와
                 같은 값을 내는지 확인. 다르면 즉시 중단한다.
  [2] 행 독립성   apply_all 을 전체/한 행씩 돌려 결과가 같은지 확인 (규정 3번 기준).
  [3] 룩업 저장   model/season_lut.npz  (2024년 말 통산상태 — 2025 추론용)
  [4] 행렬 구축   _work/X80.npy = 기존 72피처 + 시즌폼 8피처

실행:
    python prep_v12.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PITCH_DATA_DIR", os.path.join(HERE, "..", "open", "data"))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))
MODEL_DIR = os.path.join(HERE, "model")
sys.path.insert(0, HERE)

import season_form as SF                      # noqa: E402

COLS = ["season", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate",
        "batter_id", "asof_batter_n", "asof_batter_success_rate", "control_success"]


def main():
    t0 = time.time()
    print("train.csv 읽는 중...", flush=True)
    df = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=COLS, encoding="utf-8-sig")
    prior = float(df.control_success.mean())

    print("\n[1] 학습 피처 생성 + 회귀 검사", flush=True)
    A = SF.build_training_features(df)
    print("    생성 완료  shape=%s  (%.0f초)" % (A.shape, time.time() - t0), flush=True)
    ref_path = os.path.join(WORK, "season_feats.npy")
    if os.path.exists(ref_path):
        B = np.load(ref_path)
        d = float(np.abs(A.astype(np.float64) - B.astype(np.float64)).max())
        print("    project_v12.py 결과와 최대오차 %.3e  →  %s"
              % (d, "일치" if d < 1e-5 else "★ 불일치"))
        if d >= 1e-5:
            sys.exit("실험 코드와 결과가 다릅니다. 중단합니다.")
    else:
        print("    (참조 파일 없음 — 회귀 검사 건너뜀)")

    print("\n[2] 행 독립성 자체 검사", flush=True)
    lut24 = SF.build_lookup(df, 2024)
    smp = df.sample(200, random_state=0)
    full = SF.apply_all(smp, lut24, prior)
    one = np.vstack([SF.apply_all(smp.iloc[[i]], lut24, prior) for i in range(len(smp))])
    d = float(np.abs(full - one).max())
    print("    200행 일괄 vs 한 행씩:  최대오차 %.3e  →  %s"
          % (d, "통과" if d == 0.0 else "★ 실패"))
    if d != 0.0:
        sys.exit("행 독립성 위반. 중단합니다.")

    print("\n[3] 추론용 룩업 저장 (2024년 말 통산상태)", flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(MODEL_DIR, "season_lut.npz"), **lut24)
    sz = os.path.getsize(os.path.join(MODEL_DIR, "season_lut.npz")) / 1024
    print("    투수 %d명 / 타자 %d명  →  model/season_lut.npz (%.0f KB)"
          % (len(lut24["p_key"]), len(lut24["b_key"]), sz))
    print("    prior = %.6f" % prior)

    print("\n[4] 80피처 행렬 구축", flush=True)
    X = np.load(os.path.join(WORK, "X.npy"), mmap_mode="r")
    assert len(X) == len(A), "행수 불일치"
    X80 = np.empty((len(X), X.shape[1] + SF.N_FEATURES), dtype=np.float32)
    step = 200000
    for s in range(0, len(X), step):
        e = min(len(X), s + step)
        X80[s:e, :X.shape[1]] = X[s:e]
        X80[s:e, X.shape[1]:] = A[s:e]
    np.save(os.path.join(WORK, "X80.npy"), X80)
    import json
    meta = json.load(open(os.path.join(WORK, "meta.json")))
    meta["names80"] = meta["names"] + SF.NAMES
    json.dump(meta, open(os.path.join(WORK, "meta.json"), "w"))
    print("    _work/X80.npy  shape=%s  (%.0f MB)"
          % (X80.shape, X80.nbytes / 1e6))

    print("\n" + "=" * 58)
    print("준비 완료.  다음: python train_v12.py --gpu")
    print("=" * 58)
    print("총 소요 %.1f분" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
