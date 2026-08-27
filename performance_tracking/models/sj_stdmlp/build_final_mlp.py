# -*- coding: utf-8 -*-
"""제출용 — cw 의 MLP 를 **로버스트 z 점수 전처리 + `id_freq` 176열**로 재학습한다.

## 왜 바꾸나

[축2] 실측: 현행 분위수-순위 전처리로는 **`mlp` 가 블렌드에서 가중 0.00** 이다
(원시 예측·월전방분할, cb depth 5~8 전부에서). 살아 있는 것처럼 보였던 것은
`run_arm.calib` 가 평가 라벨로 스케일을 맞춰준 착시였다.

전처리 4종을 교차해보니 **`std` 만 가중을 받는다.**

| mlp 전처리 | 단독2024 | 2024정 | 2024역 | w(mlp) |
|---|---|---|---|---|
| q64 (현행) | 635.1 | +0.0 | +0.0 | **0.00** |
| q256 | 635.4 | +0.0 | +0.0 | 0.00 |
| gauss | 576.2 | +0.0 | +0.0 | 0.00 |
| **std** | 668.3 | **+12.3** | **+6.1** | **0.15** |

**기전**: 분위수-순위 변환은 **순서만 남기고 크기를 버린다.** 트리도 순서만 쓰므로
순위 MLP 는 cb 와 정보가 겹친다. **크기를 보존하는 z 점수 MLP 만 새 정보를 낸다.**
단독은 `std`(668.3)가 현행보다 낮은데도 가중을 받는다 — 이번 캠페인 내내 확인된
"정확하지만 상관된 것은 쓸모없다"는 법칙 그대로다.

## 배포 계기 실측

    현 제출본 (Public 1080.43)              val2024 874.3 · val2022 2484.2
    cb2 + std_mlp + 내부가중 재적합           val2024 884.9 · val2022 2488.1   (+10.6 / +3.9)

## 보정 상수

배포는 `apply_calibration` 으로 로짓 축소 -> 목표 성공률 중심이동 -> 클리핑을 한다.
상수는 원본 `train_v13.py` 와 **같은 절차**로 새로 뽑는다.

    logit_scale     val2024 에서 최적화 (`run_arm.calib` 의 k)
    logit_center_C0 2024행의 season 을 2025 로 바꿔 예측한 뒤,
                    보정 후 평균이 target_rate 가 되도록 이분법으로 푼다
    target_rate     0.47469 — 학습 시즌 성공률의 선형 외삽 (기존 값 그대로)

    python build_final_mlp.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ROOT = FINAL.parents[2]
WORK = FINAL / "work"

sys.path.insert(0, str(ROOT / "cowork" / "cw" / "v17" / "src"))
sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import atoms as A                                                   # noqa: E402
from prep_mlp import apply_prep, make_prep                          # noqa: E402
from run_arm import CFG, MLP_BATCH, MLP_EPOCHS, calib, load_base    # noqa: E402


def log(m):
    print(m, flush=True)


def solve_c0(lg, s, c1, target):
    """보정 후 평균이 target 이 되는 중심. 원본 `train_v13.solve_c0` 와 같다."""
    lo, hi = -5.0, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if (1.0 / (1.0 + np.exp(-(s * (lg - mid) + c1)))).mean() > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def train_mlp(Xt, yt, Xq, seeds):
    """`run_arm.fit_torch` 와 같은 학습 루프. **state_dict 도 함께 돌려준다.**"""
    import torch
    import torch.nn as nn
    import dl as DL
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    states, preds = [], []
    for seed in range(seeds):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = DL.build_mlp(torch, nn, Xt.shape[1], CFG.width, CFG.depth, CFG.drop).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        n = len(Xt)
        nb = -(-n // MLP_BATCH)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, CFG.lr,
                                                  total_steps=nb * MLP_EPOCHS, pct_start=0.1)
        scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))
        Xtt = torch.from_numpy(Xt)
        ytt = torch.from_numpy(yt.astype(np.float32))
        on_gpu = False
        if dev == "cuda":
            need = Xtt.numel() * 4 + ytt.numel() * 4
            if need < torch.cuda.mem_get_info()[0] * 0.55:
                Xtt = Xtt.to(dev)
                ytt = ytt.to(dev)
                on_gpu = True
        for _ in range(MLP_EPOCHS):
            net.train()
            perm = torch.randperm(n, device=dev if on_gpu else "cpu")
            for s in range(0, n, MLP_BATCH):
                i = perm[s:s + MLP_BATCH]
                xb, yb = Xtt[i], ytt[i]
                if not on_gpu:
                    xb = xb.to(dev, non_blocking=True)
                    yb = yb.to(dev, non_blocking=True)
                with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                    loss = ((torch.sigmoid(net(xb).squeeze(-1)) - yb) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sch.step()
        net.eval()
        out = []
        with torch.no_grad():                   # 추론은 fp32 고정 (§27.2)
            Xqt = torch.from_numpy(Xq)
            for s in range(0, len(Xqt), 16384):
                out.append(torch.sigmoid(net(Xqt[s:s + 16384].to(dev)).squeeze(-1))
                           .double().cpu().numpy())
        preds.append(np.concatenate(out))
        states.append({k: v.detach().cpu() for k, v in net.state_dict().items()})
        log("  mlp seed%d  %.0f초" % (seed, time.time() - t0))
        del net, Xtt, ytt
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0), states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--prep", default="std")
    ap.add_argument("--target-rate", type=float, default=0.47469465355297163)
    ap.add_argument("--c1", type=float, default=-0.10130794309825776)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    t0 = time.time()
    X, y, season, row_id = load_base()
    names = json.load(open(WORK / "meta.json", encoding="utf-8"))["names"]
    n = len(y)
    log("=" * 78)
    log("제출용 MLP — 전체 %s행 · 전처리 %s · id_freq 포함 · %d시드"
        % (f"{n:,}", a.prep, a.seeds))
    log("=" * 78)

    # 전체 행이 학습행이다 (제출본은 시즌을 빼지 않는다)
    E, en = A.build(X, names, np.ones(n, bool), int(season.max()) + 1, ["id_freq"])
    X176 = np.ascontiguousarray(np.concatenate([np.asarray(X), E], axis=1))
    del E
    log("  원자 %d열 → 총 %d피처" % (len(en), X176.shape[1]))

    prep = make_prep(X176, a.prep)
    Z = apply_prep(X176, prep, True)
    log("  전처리 %s → 입력 %d열" % (a.prep, Z.shape[1]))

    # ── 2025 모사 — 2024행의 season 을 2025 로 바꾼다 (train_v13 §C 와 같다) ──
    si = names.index("season")
    m24 = np.asarray(season) == 2024
    X25 = X176[m24].copy()
    X25[:, si] = float(int(season.max()) + 1)
    Z25 = apply_prep(X25, prep, True)
    log("  2025 모사 %s행 (2024행의 season→%d)"
        % (f"{int(m24.sum()):,}", int(season.max()) + 1))
    del X25

    p25, states = train_mlp(Z, y, Z25, a.seeds)
    del Z, Z25

    # ── 보정 상수 ───────────────────────────────────────────────────────────
    # logit_scale 은 val2024 에서 최적화한다 — 원본 `train_v13.py` 와 같은 절차다.
    # 그 값은 [축2] 실험이 이미 낸 val 예측(`PREP_std_176_2024.npy`)에서 뽑는다.
    vp = FINAL / "preds" / ("PREP_%s_176_2024.npy" % a.prep)
    if not vp.exists():
        sys.exit("[중단] %s 가 없다 — prep_mlp.py 를 먼저 돌려야 한다" % vp.name)
    sys.path.insert(0, str(ROOT / "performance_tracking" / "tools"))
    import importlib.util as iu
    sp = iu.spec_from_file_location(
        "pt_common", ROOT / "performance_tracking" / "tools" / "common.py")
    mod = iu.module_from_spec(sp)
    sp.loader.exec_module(mod)
    y24 = mod.load_labels(2024)["y"].to_numpy(np.float64)
    s_bss, scale, _ = calib(np.load(vp), y24)
    lg = np.log(np.clip(p25, 1e-6, 1 - 1e-6) / (1 - np.clip(p25, 1e-6, 1 - 1e-6)))
    c0 = solve_c0(lg, scale, a.c1, a.target_rate)
    chk = float((1 / (1 + np.exp(-(scale * (lg - c0) + a.c1)))).mean())
    log("  보정  logit_scale %.4f (val2024 최적, BSS %.1f) · C0 %.6f → 모사평균 %.5f (목표 %.5f)"
        % (scale, s_bss, c0, chk, a.target_rate))

    # ── 저장 ────────────────────────────────────────────────────────────────
    ck = {"states": states, "d_in": int(2 * X176.shape[1]),
          "cfg": {"width": CFG.width, "depth": CFG.depth, "drop": CFG.drop,
                  "dtoken": CFG.dtoken, "layers": CFG.layers, "kind": "mlp"}}
    torch.save(ck, out / "mlp.pt")
    if a.prep == "std":
        np.savez_compressed(out / "stdprep.npz",
                            med=prep["med"].astype(np.float64),
                            iqr=prep["iqr"].astype(np.float64))
    log("  → mlp.pt %.1fMB · stdprep.npz"
        % ((out / "mlp.pt").stat().st_size / 1e6))
    json.dump({"prep": a.prep, "n_features": int(X176.shape[1]),
               "d_in": int(2 * X176.shape[1]), "seeds": a.seeds,
               "model_mlp": {"logit_scale": float(scale), "logit_center_C0": float(c0),
                             "cap": 0.20, "target_rate": a.target_rate,
                             "logit_target_C1": a.c1},
               "train_rows": int(n)},
              open(out / "mlp_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("\n총 %.1f분" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
