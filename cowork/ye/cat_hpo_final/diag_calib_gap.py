# -*- coding: utf-8 -*-
"""anchor(855.5)가 예상 벤치마크(900대)보다 낮은 게 calib() 유무 때문인지 빠르게 확인.

anchor 와 완전히 같은 하이퍼를 쓰되, lr x iterations = 60 근처(tune_cb.py 가
이미 확인한 "최적선")를 유지하면서 iterations 만 줄여 학습 시간을 단축한다
(3000/0.02=60 -> 500/0.12=60). fold2024 · 1시드만.

raw BSS 와, train_v13.py 의 calib()(평가 라벨로 로짓스케일 k 최적화 — 팀이 이미
run_arm.calib 을 오염으로 판정한 것과 같은 패턴)를 적용한 BSS 를 나란히 낸다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(m):
    print(m, flush=True)


def bss_raw(p, y):
    r = y.mean()
    return 100000.0 * max(0.0, (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r))))


def calib(p, yv):
    """train_v13.py 의 calib() 그대로. 평가 시즌 라벨(yv)로 로짓스케일 k 를 고른다."""
    r = yv.mean()
    U = r * (1 - r)
    c1 = np.log(r / (1 - r))
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    best = (-9e9, 1.0, None)
    for k in np.arange(0.2, 1.55, 0.05):
        q = 1 / (1 + np.exp(-(k * (z - z.mean()) + c1)))
        v = 1e5 * (1 - ((q - yv) ** 2).mean() / U)
        if v > best[0]:
            best = (v, k, q)
    return best


def main():
    t0 = time.time()
    from catboost import CatBoostRegressor

    X168 = np.load(WORK / "X168.npy", mmap_mode="r")
    y = np.load(WORK / "y.npy")
    season = np.load(WORK / "season.npy")
    names168 = json.load(open(WORK / "meta.json"))["names168"]
    idf = np.load(WORK / "idfreq8_2024.npy")
    idf_names = json.load(open(WORK / "meta.json"))["idfreq_names"]

    X176 = np.concatenate([X168, idf], axis=1)
    names176 = names168 + idf_names
    tr = season < 2024
    va = season == 2024
    Xt, yt = np.asarray(X176[tr]), y[tr]
    Xv, yv = np.asarray(X176[va]), y[va].astype(np.float64)
    log(f"학습 {len(Xt):,} 검증 {len(Xv):,}  ({time.time()-t0:.0f}s)")

    # lr x iterations = 60 유지 (tune_cb.py 최적선), 6배 빠르게
    params = dict(iterations=500, depth=6, learning_rate=0.12, l2_leaf_reg=10000.0,
                  loss_function="RMSE", random_seed=11, verbose=False,
                  allow_writing_files=False, thread_count=-1)
    log(f"학습 시작 (lr x it = {params['iterations']*params['learning_rate']:.1f}, anchor 는 60)")
    m = CatBoostRegressor(**params)
    m.fit(Xt, yt.astype(np.float64))
    p_raw = np.clip(m.predict(Xv), 1e-6, 1 - 1e-6)
    log(f"학습 완료 ({time.time()-t0:.0f}s)")

    raw_score = bss_raw(p_raw, yv)
    calib_score, best_k, p_calib = calib(p_raw, yv)

    log("\n===== 결과 =====")
    log(f"raw BSS        = {raw_score:.1f}   (pred_mean={p_raw.mean():.4f} true_mean={yv.mean():.4f})")
    log(f"calib() BSS    = {calib_score:.1f}  (best_k={best_k:.2f}, pred_mean={p_calib.mean():.4f})")
    log(f"델타(calib 효과) = {calib_score - raw_score:+.1f}")
    log(f"\n참고: anchor(3000it) 1시드 raw BSS = 855.5 (이미 완료된 느린 실행)")
    log(f"이 실험(500it, lr.12, 같은 lr x it=60)의 raw BSS = {raw_score:.1f} — 서로 비슷해야 유효한 비교")
    log(f"\n총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    raise SystemExit(main())
