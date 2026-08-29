# -*- coding: utf-8 -*-
"""챔피언 CB(176피처) 하이퍼파라미터 미탐색 축 screening.

anchor = 실제 배포값 (train_v13.py CB_P, depth 만 D2 갱신대로 5->6)
    iterations=3000, depth=6, learning_rate=0.02, l2_leaf_reg=10000.0,
    loss_function="RMSE", thread_count=-1  (seed 3개: 11,22,33)

이미 tune_cb.py(R1~R5)로 탐색된 것: loss함수, 시즌가중, depth(3~8), lr×iterations,
l2(6~30000), 오프셋, bootstrap+subsample(R2 1회, 후속 라운드에서 버려짐).
탐색 안 된 것(이번 대상): random_strength, border_count, bagging_temperature,
min_data_in_leaf.

정직 규칙: eval_set/조기종료/캘리브레이션에 평가 시즌 라벨 안 씀. 폴드마다 고정
iteration 으로 학습해 그 시즌 예측만 낸다 (팀 규칙 2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(m):
    print(m, flush=True)


ANCHOR = dict(iterations=3000, depth=6, learning_rate=0.02, l2_leaf_reg=10000.0,
              loss_function="RMSE", thread_count=-1, verbose=False,
              allow_writing_files=False)
SEEDS = [11, 22, 33]


def bss(p, y):
    r = y.mean()
    return 100000.0 * max(0.0, (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r))))


def load_fold(fold, X168, y, season, names168):
    idf = np.load(WORK / f"idfreq8_{fold}.npy")
    idf_names = json.load(open(WORK / "meta.json"))["idfreq_names"]
    X176 = np.concatenate([X168, idf], axis=1)
    names176 = names168 + idf_names
    tr = season < fold
    va = season == fold
    gt_col = names176.index("game_type=R")
    return X176[tr], y[tr], X176[va], y[va], X176[va][:, gt_col].astype(bool), names176


def fit_predict(Xt, yt, Xv, params, seed, dev=None):
    from catboost import CatBoostRegressor
    p = dict(ANCHOR)
    p.update(params)
    if dev:
        p.update(dev)
    p["random_seed"] = seed
    m = CatBoostRegressor(**p)
    m.fit(Xt, yt.astype(np.float64))
    pred = np.clip(m.predict(Xv), 1e-6, 1 - 1e-6)
    del m
    return pred


def run_candidate(name, params, folds_data, seeds=SEEDS, dev=None):
    row = {"name": name, "param": json.dumps(params)}
    t0 = time.time()
    for fold, (Xt, yt, Xv, yv, rmask, _) in folds_data.items():
        acc = np.zeros(len(yv))
        for sd in seeds:
            ts = time.time()
            p1 = fit_predict(Xt, yt, Xv, params, sd, dev=dev)
            acc += p1
            log(f"    {name} fold{fold} seed{sd}  단독BSS={bss(p1, yv):.1f}  "
                f"({time.time()-ts:.0f}s, 누적 {(time.time()-t0)/60:.1f}분)")
        pred = acc / len(seeds)
        all_bss = bss(pred, yv)
        r_bss = bss(pred[rmask], yv[rmask])
        row[f"val{fold}_all"] = all_bss
        row[f"val{fold}_R"] = r_bss
    row["train_minutes"] = (time.time() - t0) / 60
    log(f"  {name:20s} " + "  ".join(
        f"{k}={v:.1f}" for k, v in row.items() if k.startswith("val")) +
        f"  ({row['train_minutes']:.1f}분)")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-only", action="store_true", help="anchor 검증만")
    ap.add_argument("--group", default=None, help="A/B/C/D 중 하나만")
    ap.add_argument("--gpu", action="store_true", help="task_type=GPU devices=0 border_count=128")
    a = ap.parse_args()
    dev = dict(task_type="GPU", devices="0", border_count=128) if a.gpu else None

    t0 = time.time()
    log("데이터 로드 중...")
    X168 = np.load(WORK / "X168.npy", mmap_mode="r")
    y = np.load(WORK / "y.npy")
    season = np.load(WORK / "season.npy")
    names168 = json.load(open(WORK / "meta.json"))["names168"]

    folds_data = {}
    for fold in (2024, 2023, 2022):
        tf = time.time()
        folds_data[fold] = load_fold(fold, X168, y, season, names168)
        log(f"fold{fold} 준비: 학습 {len(folds_data[fold][1]):,} 검증 {len(folds_data[fold][3]):,}"
            f"  ({time.time()-tf:.0f}s)")

    log("\n===== ANCHOR 검증 =====")
    anchor_row = run_candidate("ANCHOR", {}, folds_data, dev=dev)
    rows = [anchor_row | {"status": "ANCHOR"}]
    pd.DataFrame(rows).to_csv(HERE / "results_hpo.csv", index=False)

    if a.anchor_only:
        log(f"\n총 {(time.time()-t0)/60:.1f}분 (anchor만)")
        return 0

    GROUPS = {
        "A": [  # random_strength (기본값 1.0)
            ("R1_rs0.5", dict(random_strength=0.5)),
            ("R2_rs2.0", dict(random_strength=2.0)),
            ("R3_rs5.0", dict(random_strength=5.0)),
        ],
        "B": [  # border_count (기본값 254)
            ("B1_bc32", dict(border_count=32)),
            ("B2_bc128", dict(border_count=128)),
        ],
        "C": [  # bootstrap (R2 에서 딱 한번 스치듯 나온 축, 재확인)
            ("C1_bernoulli.7", dict(bootstrap_type="Bernoulli", subsample=0.7)),
            ("C2_bayesian.5", dict(bootstrap_type="Bayesian", bagging_temperature=0.5)),
        ],
        "D": [  # min_data_in_leaf (기본값 1)
            ("D1_mdl10", dict(min_data_in_leaf=10)),
            ("D2_mdl50", dict(min_data_in_leaf=50)),
        ],
    }

    groups_to_run = [a.group] if a.group else list(GROUPS.keys())
    for g in groups_to_run:
        log(f"\n===== Group {g} =====")
        for name, params in GROUPS[g]:
            row = run_candidate(name, params, folds_data, dev=dev)
            row["status"] = "PENDING"
            rows.append(row)
            pd.DataFrame(rows).to_csv(HERE / "results_hpo.csv", index=False)

    log(f"\n총 {(time.time()-t0)/60:.1f}분")
    log("저장: results_hpo.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
