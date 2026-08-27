# -*- coding: utf-8 -*-
"""FT / MLP 하이퍼파라미터 스윕 — 한 번도 안 건드린 곳.

동기
    CatBoost 를 재조정하니 크게 움직였다.
        depth 6 -> 8   -11.5%       lr 0.06 -> 0.02   +3.9%       RMSE   +1.2%
    결론은 "모델이 과적합하고 있었다" 였다.

    그런데 FT 와 MLP 의 설정은 전부 처음에 찍은 값 그대로다.
        FT   d_token 64 / 3층 / 8헤드 / drop 0.1
        MLP  width 384 / 3층 / drop 0.3 / lr 2e-3 / wd 3e-4 / 15에폭
    FT 는 블렌드 가중치 0.416 으로 두 번째로 크다. 여기가 안 열려 있을 리 없다.

비용 비대칭
    MLP  약 15초/시드   →  3시드로 넉넉히 훑는다
    FT   약 520초/시드  →  1시드로 훑고 승자만 3시드로 확인한다

측정 규율
    val 2024 · 2022 두 해, min 규칙. 기준도 같은 시드로 매번 다시 잰다.
    점수는 **마지막 에폭** 값을 쓴다. 최고 에폭은 val 로 고른 값이라 낙관적이다.

실행
    python tune_dl.py --model mlp            # 약 20분
    python tune_dl.py --model ft             # 약 100분
    python tune_dl.py --model ft --confirm dtoken96   # 승자 3시드 확인
"""

import argparse
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))

import dl                                            # noqa: E402

# 현재 v13 설정
BASE_MLP = dict(width=384, depth=3, drop=0.3, lr=2e-3, wd=3e-4,
                epochs=15, batch=4096)
BASE_FT = dict(dtoken=64, layers=3, drop=0.1, lr=2e-3, wd=3e-4,
               epochs=15, batch=4096)

# CatBoost 의 교훈(과적합 억제가 이긴다)을 참고하되, 신경망은 반대일 수도 있으므로
# 양쪽으로 다 벌려서 훑는다.
SWEEP_MLP = [
    ("width768",  dict(width=768)),
    ("width192",  dict(width=192)),
    ("depth5",    dict(depth=5)),
    ("depth2",    dict(depth=2)),
    ("drop0.15",  dict(drop=0.15)),
    ("drop0.45",  dict(drop=0.45)),
    ("wd1e-3",    dict(wd=1e-3)),
    ("ep25",      dict(epochs=25)),
    ("ep8",       dict(epochs=8)),
    ("lr1e-3",    dict(lr=1e-3)),
    ("batch1024", dict(batch=1024)),
]
# MLP 2차. 1차 결과는 전부 "과적합하고 있었다" 였다.
#   ep25 -14.9%   drop0.15 -13.3%   |   batch1024 +8.0%   ep8 +6.6%   drop0.45 +5.3%
# 이긴 것들을 합친다. batch 를 줄이면 그래디언트 잡음이 늘어 규제처럼 작동한다.
SWEEP_MLP2 = [
    ("b1024+ep8",       dict(batch=1024, epochs=8)),
    ("b1024+d45",       dict(batch=1024, drop=0.45)),
    ("b1024+ep8+d45",   dict(batch=1024, epochs=8, drop=0.45)),
    ("b1024+ep8+w192",  dict(batch=1024, epochs=8, width=192)),
    ("b1024+ep8+d45+w192",
     dict(batch=1024, epochs=8, drop=0.45, width=192)),
    ("b512+ep8+d45",    dict(batch=512, epochs=8, drop=0.45)),
    ("b1024+ep5+d45",   dict(batch=1024, epochs=5, drop=0.45)),
]
SWEEP_FT = [
    ("dtoken32",  dict(dtoken=32)),          # 싸다
    ("layers2",   dict(layers=2)),           # 싸다
    ("dtoken96",  dict(dtoken=96)),
    ("layers4",   dict(layers=4)),
    ("drop0.25",  dict(drop=0.25)),
    ("ep25",      dict(epochs=25)),
]


def train_eval(torch, nn, Xtr, ytr, Xva, yva, model, cfg, seed):
    """조용한 학습 루프. dl.run_one 은 에폭마다 출력이 많아 스윕에 안 맞는다."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]
    net = (dl.build_ft(torch, nn, d, cfg["dtoken"], cfg["layers"], 8, cfg["drop"])
           if model == "ft" else
           dl.build_mlp(torch, nn, d, cfg["width"], cfg["depth"], cfg["drop"])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n = len(Xtr); bs = cfg["batch"]
    steps = math.ceil(n / bs) * cfg["epochs"]
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps,
                                              pct_start=0.1)
    scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr.astype(np.float32))
    Xv = torch.from_numpy(Xva)

    for _ in range(cfg["epochs"]):
        net.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = Xt[idx].to(dev, non_blocking=True)
            yb = yt[idx].to(dev, non_blocking=True)
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                p = torch.sigmoid(net(xb).squeeze(-1))
                loss = ((p - yb) ** 2).mean()          # Brier 직접
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sch.step()

    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(Xv), 8192):
            xb = Xv[s:s + 8192].to(dev)
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                out.append(torch.sigmoid(net(xb).squeeze(-1)).float().cpu().numpy())
    del net, Xt, Xv
    torch.cuda.empty_cache() if dev == "cuda" else None
    return dl.calib(np.concatenate(out), yva)[0]       # 마지막 에폭 값


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=["mlp", "ft"])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--confirm", nargs="*", default=None,
                    help="이 설정들만 시드를 늘려 확인")
    ap.add_argument("--round", type=int, default=1, help="MLP 스윕 라운드 (1 또는 2)")
    ap.add_argument("--then-confirm", type=int, default=0,
                    help="스윕이 끝나면 상위 N개를 3시드로 자동 재측정 (무인 실행용)")
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else (3 if a.model == "mlp" else 1)
    if a.confirm:
        seeds = a.seeds if a.seeds else 3

    import torch
    import torch.nn as nn
    print("torch %s | GPU %s" % (torch.__version__,
                                 torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else "없음"), flush=True)

    base = dict(BASE_FT if a.model == "ft" else BASE_MLP)
    if a.model == "ft":
        sweep = SWEEP_FT
    else:
        sweep = SWEEP_MLP2 if a.round == 2 else SWEEP_MLP
    if a.confirm:
        sweep = [(nm, c) for nm, c in sweep if nm in a.confirm]
        if not sweep:
            raise SystemExit("모르는 설정. 가능: %s" % [n for n, _ in
                                                    (SWEEP_FT if a.model == "ft"
                                                     else SWEEP_MLP)])

    t0 = time.time()
    X = np.load(os.path.join(WORK, "X168.npy"), mmap_mode="r")
    y = np.load(os.path.join(WORK, "y.npy"))
    season = np.load(os.path.join(WORK, "season.npy"))
    wm = (a.model != "ft")

    # 전처리는 조합마다 다시 하면 낭비다. val 연도별로 한 번만 만든다.
    data = {}
    for val in (2024, 2022):
        tri = np.where(season <= val - 1)[0]; va = np.where(season == val)[0]
        Xr = np.asarray(X[tri]); Xvr = np.asarray(X[va])
        prep = dl.make_prep(Xr)
        data[val] = (dl.apply_prep(Xr, prep, wm), y[tri],
                     dl.apply_prep(Xvr, prep, wm), y[va].astype(np.float64))
        del Xr, Xvr
        print("val%d 전처리 완료  학습 %d행 %d열  (%.1f분)"
              % (val, len(tri), data[val][0].shape[1], (time.time() - t0) / 60), flush=True)

    def measure(nm, cfg, ns):
        got = {}
        for val in (2024, 2022):
            Xtr, ytr, Xva, yva = data[val]
            ss = [train_eval(torch, nn, Xtr, ytr, Xva, yva, a.model, cfg, sd)
                  for sd in range(ns)]
            got[val] = (float(np.mean(ss)), float(np.std(ss, ddof=1)) if ns > 1 else 0.0)
            print("  %-11s val%d  %8.1f +- %4.1f  (%.1f분)"
                  % (nm, val, got[val][0], got[val][1], (time.time() - t0) / 60), flush=True)
        return got

    def phase(tag, sw, ns):
        """기준 + 조합들을 ns 시드로 재고 표를 찍는다. 정렬된 rows 를 돌려준다."""
        print("\n%s %s  (%d시드, 조합 %d개)"
              % (a.model.upper(), tag, ns, len(sw) + 1), flush=True)
        ref = measure("기준", base, ns)
        res = {nm: measure(nm, {**base, **chg}, ns) for nm, chg in sw}

        print()
        print("=" * 74)
        print("[%s %s]  기준 val2024 %.1f / val2022 %.1f"
              % (a.model.upper(), tag, ref[2024][0], ref[2022][0]))
        print("  기준설정  %s" % " ".join("%s=%s" % kv for kv in base.items()))
        print("=" * 74)
        print("%-11s %10s %8s %10s %8s %9s"
              % ("설정", "val2024", "%(t)", "val2022", "%(t)", "min%"))
        rows = []
        for nm, _ in sw:
            gs, ts, line = [], [], [nm]
            for val in (2024, 2022):
                b, sb = ref[val]; e, se = res[nm][val]
                sed = (np.sqrt(sb ** 2 + se ** 2) / np.sqrt(ns)) if ns > 1 else 0.0
                gs.append(e / b - 1)
                ts.append((e - b) / sed if sed > 0 else float("nan"))
                line += ["%.1f" % e,
                         "%+.2f%%" % (100 * (e / b - 1)) if ns == 1
                         else "%+.1f/%.1f" % (100 * (e / b - 1), ts[-1])]
            mn = min(gs)
            line += ["%+.2f%%" % (100 * mn)]
            rows.append((mn, nm, ts))
            print("%-11s %10s %8s %10s %8s %9s" % tuple(line))

        rows.sort(reverse=True)
        print()
        print("상위")
        for mn, nm, ts in rows[:4]:
            tt = " ".join("t=%.1f" % t if t == t else "t=-" for t in ts)
            ok = "채택" if (mn > 0 and all(t > 2 for t in ts if t == t)) else \
                 ("확인필요" if mn > 0 else "기각")
            print("  %-11s min %+.2f%%   %s   %s" % (nm, 100 * mn, tt, ok))
        json.dump({"model": a.model, "tag": tag, "seeds": ns, "base": base,
                   "ref": {str(k): v for k, v in ref.items()},
                   "res": {nm: {str(k): v for k, v in d.items()}
                           for nm, d in res.items()}},
                  open(os.path.join(WORK, "tune_dl_%s_%s.json" % (a.model, tag)), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=1)
        return rows

    rows = phase("스윕", sweep, seeds)

    # 1시드 스윕은 신뢰할 수 없다. 오늘 3시드 기준선이 2.2% 흔들려 승자를 통째로
    # 뒤집은 전례가 있다. 사람이 안 붙어 있어도 확인 단계가 자동으로 이어지게 한다.
    if a.then_confirm and seeds < 3 and not a.confirm:
        top = [nm for _, nm, _ in rows[:a.then_confirm]]
        print()
        print("=" * 74)
        print("자동 확인 단계 — 상위 %d개를 3시드로 다시 잰다: %s"
              % (len(top), ", ".join(top)))
        print("=" * 74, flush=True)
        sw2 = [(nm, chg) for nm, chg in sweep if nm in top]
        phase("확인", sw2, 3)

    print("=" * 74)
    print("총 %.1f분" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
