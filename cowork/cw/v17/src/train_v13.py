# -*- coding: utf-8 -*-
"""v13 최종 학습 — CatBoost(168) + FT-Transformer + MLP.

구성 변경
    v12 는 sklearn 트리 2계열(A·B) + CatBoost 였다.
    그런데 168피처 CatBoost 하나(826.1)가 v12 블렌드 전체(803.1)보다 낫다.
    A·B 를 빼면 추론이 크게 가벼워지고(10분 제한에 여유), 점수는 오히려 오른다.

    CB   CatBoost 168피처, 3시드 평균, numpy 로 내보냄
    FT   FT-Transformer, 3시드, torch state_dict (평가서버에 torch 2.7.1 + L4 GPU 기본설치)
    MLP  잔차 MLP, 3시드

검증 (val2024 / val2022)
    CB 826.1 / 2383.5   FT 811.4 / 2359.2   MLP 732.4 / 2215.7
    상관 CB-FT 0.910    CB-MLP 0.867    FT-MLP 0.874
    CB+FT+MLP  +4.41% / +2.64%   → LB 투영 975

3단계
  [A] 2019~2023 학습 → 2024·2022 로 계열별 로짓 스케일 + 블렌드 가중치
  [B] 2019~2024 전체 학습 → 내보내기
  [C] 2024행의 season→2025 분포에서 계열별 C0 (v8 때 이 단계를 빼먹어 60점 잃었다)

실행:
    python train_v13.py --gpu        # 약 25분
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))
MODEL_DIR = os.path.join(HERE, "model")
sys.path.insert(0, HERE)

from cb_export import export_catboost, cb_predict, verify as cb_verify   # noqa: E402
import dl as DL                                                          # noqa: E402

# tune_cb.py 5라운드 결과로 확정 (val 2024·2022, 8시드, min 규칙).
#   Logloss -> RMSE       평가지표가 Brier 라 손실을 일치시킨다
#   lr 0.06 -> 0.02       lr x 트리수 = 60 근처가 최적선
#   depth 6 -> 5          3 에서 꺾여 4~5 가 최적, 5 가 근소 우위
#   l2 6 -> 10000         20/50/150/400/1000/3000 내내 상승, 30000 에서 꺾였다
#
# 5라운드 최종  d5 / lr0.02 / 3000트리 / RMSE / l2 10000
#     val2024   800.1 -> 863.5  (+7.92%, t 16.2)
#     val2022  2328.0 -> 2479.8 (+6.52%, t 27.7)
# 부수 효과: 시드 표준편차가 ±10.7 -> ±2.9 로 줄었다. 시드 평균의 값어치가
# 거의 사라져 3시드로 충분하다.
CB_P = dict(iterations=3000, depth=5, learning_rate=0.02, l2_leaf_reg=10000.0,
            loss_function="RMSE", verbose=0, allow_writing_files=False,
            thread_count=-1)
CB_REGRESSION = CB_P["loss_function"] == "RMSE"
N_SEED = 3


def make_cb(seed, dev):
    """RMSE 는 회귀 손실이라 Regressor 를 써야 한다. raw 가 곧 확률이다."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    K = CatBoostRegressor if CB_REGRESSION else CatBoostClassifier
    return K(**CB_P, random_seed=seed, **dev)


def cb_fit_predict(m, Pool, Xt, yt, Xv=None):
    m.fit(Pool(Xt, yt.astype(np.float64) if CB_REGRESSION else yt))
    if Xv is None:
        return None
    return (np.clip(m.predict(Pool(Xv)), 1e-6, 1 - 1e-6) if CB_REGRESSION
            else m.predict_proba(Pool(Xv))[:, 1])


class A:      # dl.run_one 이 기대하는 설정 객체
    # tune_dl.py MLP 2라운드 결과: batch 4096 -> 1024, epochs 15 -> 8  (+5.97%)
    # drop 0.45 · width 192 를 더 얹으면 오히려 떨어졌다 — 셋 다 같은 병(과적합)을
    # 고치는 거라 겹친다.
    model = "ft"; epochs = 8; batch = 1024; lr = 2e-3; wd = 3e-4
    width = 384; depth = 3; drop = 0.3; dtoken = 64; layers = 3; eval_every = 99


def calib(p, yv):
    r = yv.mean(); U = r * (1 - r); c1 = np.log(r / (1 - r))
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    best = (-9e9, 1.0, None)
    for k in np.arange(0.2, 1.55, 0.05):
        q = 1 / (1 + np.exp(-(k * (z - z.mean()) + c1)))
        v = 1e5 * (1 - ((q - yv) ** 2).mean() / U)
        if v > best[0]:
            best = (v, k, q)
    return best


def blend_w(qs, yv, ridge=0.02):
    r = yv.mean(); U = r * (1 - r)
    D = np.column_stack([q - r for q in qs])
    Av = D.T @ (yv - r) / len(yv); M = D.T @ D / len(yv)
    w = np.linalg.solve(M + ridge * np.diag(np.diag(M)), Av)
    return w, float(2 * w @ Av - w @ M @ w) / U * 1e5


def solve_c0(lg, s, C1, target):
    lo, hi = -5.0, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if (1.0 / (1.0 + np.exp(-(s * (lg - mid) + C1)))).mean() > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def train_torch(torch, nn, kind, Xtr, ytr, Xq, seeds, epochs):
    """kind='ft'|'mlp'. 학습 후 (예측평균, [state_dict...]) 반환."""
    cfg = A(); cfg.model = kind; cfg.epochs = epochs
    if kind == "mlp":
        # tune_dl MLP 2라운드: batch 4096 -> 1024 가 min +5.97% (t 2.2 / 7.4).
        # 배치를 줄이면 그래디언트 잡음이 늘어 규제처럼 작동한다.
        cfg.batch = 1024
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    preds, states = [], []
    for sd in range(seeds):
        torch.manual_seed(sd); np.random.seed(sd)
        d = Xtr.shape[1]
        net = (DL.build_ft(torch, nn, d, cfg.dtoken, cfg.layers) if kind == "ft"
               else DL.build_mlp(torch, nn, d, cfg.width, cfg.depth, cfg.drop)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        n = len(Xtr); nb = math.ceil(n / cfg.batch)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=nb * cfg.epochs,
                                                  pct_start=0.1)
        scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))
        Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr.astype(np.float32))
        t0 = time.time()
        for ep in range(cfg.epochs):
            net.train(); perm = torch.randperm(n)
            for s in range(0, n, cfg.batch):
                i = perm[s:s + cfg.batch]
                xb = Xt[i].to(dev); yb = yt[i].to(dev)
                with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                    loss = ((torch.sigmoid(net(xb).squeeze(-1)) - yb) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        net.eval(); out = []
        with torch.no_grad():
            Xv = torch.from_numpy(Xq)
            for s in range(0, len(Xv), 8192):
                with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                    out.append(torch.sigmoid(net(Xv[s:s + 8192].to(dev)).squeeze(-1))
                               .float().cpu().numpy())
        preds.append(np.concatenate(out))
        states.append({k: v.cpu() for k, v in net.state_dict().items()})
        print("      %s seed%d 완료 (%.0f초)" % (kind, sd, time.time() - t0), flush=True)
        del net
    return np.mean(preds, axis=0), states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEED)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--mlp-epochs", type=int, default=8)     # tune_dl 2라운드 확정
    ap.add_argument("--skip-a", action="store_true",
                    help="[A] 결과를 stageA.json 에서 불러온다 (재실행 시 40분 절약)")
    a = ap.parse_args()
    import torch
    import torch.nn as nn
    from catboost import CatBoostClassifier, Pool
    dev = dict(task_type="GPU", devices="0", border_count=128) if a.gpu else {}

    cal = json.load(open(os.path.join(MODEL_DIR, "calib_train_only.json"), encoding="utf-8"))
    TARGET = float(cal["target_rate"]); C1 = float(cal["logit_target_C1"])
    print("target_rate %.5f (학습 데이터 외삽)" % TARGET, flush=True)

    t0 = time.time()
    X = np.load(os.path.join(WORK, "X168.npy"), mmap_mode="r")
    y = np.load(os.path.join(WORK, "y.npy"))
    season = np.load(os.path.join(WORK, "season.npy"))
    names = json.load(open(os.path.join(WORK, "meta.json")))["names80"]
    si = names.index("season")

    # ── [A] 스케일 + 가중치 ────────────────────────────────
    SA = os.path.join(MODEL_DIR, "v13_stageA.json")
    if a.skip_a and os.path.exists(SA):
        d = json.load(open(SA, encoding="utf-8"))
        scales = d["scales"]; w = np.array(d["w"])
        print("\n[A] 건너뜀 — %s 에서 불러옴" % os.path.basename(SA))
        print("  scales %s | w CB %.3f FT %.3f MLP %.3f"
              % (scales, w[0], w[1], w[2]), flush=True)
        years = ()
    else:
        print("\n[A] 2019~2023 → 검증", flush=True)
        scales, W, years = {}, {}, (2024, 2022)
    for val in years:
        tri = np.where(season <= val - 1)[0]; va = np.where(season == val)[0]
        yv = y[va].astype(np.float64)
        Xt_raw = np.asarray(X[tri]); Xv_raw = np.asarray(X[va])
        prep = DL.make_prep(Xt_raw)
        qs, labels = [], []
        # CatBoost
        acc = np.zeros(len(Xv_raw))
        for sd in range(a.seeds):
            m = make_cb(sd, dev)
            acc += cb_fit_predict(m, Pool, Xt_raw, y[tri], Xv_raw); del m
        b = calib(acc / a.seeds, yv); qs.append(b[2]); labels.append("CB")
        if val == 2024:
            scales["CB"] = b[1]
        print("  val%d CB  %8.1f  s=%.2f" % (val, b[0], b[1]), flush=True)
        # torch 2종
        for kind, ep in (("ft", a.ft_epochs), ("mlp", a.mlp_epochs)):
            wm = (kind != "ft")
            Xt = DL.apply_prep(Xt_raw, prep, wm); Xv = DL.apply_prep(Xv_raw, prep, wm)
            p, _ = train_torch(torch, nn, kind, Xt, y[tri], Xv, a.seeds, ep)
            b = calib(p, yv); qs.append(b[2]); labels.append(kind.upper())
            if val == 2024:
                scales[kind.upper()] = b[1]
            print("  val%d %-3s %8.1f  s=%.2f" % (val, kind.upper(), b[0], b[1]), flush=True)
            del Xt, Xv
        w, sc = blend_w(qs, yv)
        W[val] = w
        print("  val%d 블렌드 %8.1f   w = %s"
              % (val, sc, "  ".join("%s %.3f" % (l, v) for l, v in zip(labels, w))), flush=True)
        del Xt_raw, Xv_raw
    if years:
        w = (W[2024] + W[2022]) / 2.0
        json.dump({"scales": scales, "w": [float(v) for v in w]},
                  open(SA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("\n  두 연도 평균 가중치 CB %.3f  FT %.3f  MLP %.3f" % tuple(w), flush=True)
        print("  저장: model/v13_stageA.json  (다음부터 --skip-a 로 재사용)", flush=True)

    # ── [B] 전체 학습 + 내보내기 ───────────────────────────
    print("\n[B] 2019~2024 전체 학습", flush=True)
    Xa_raw = np.asarray(X)
    prep = DL.make_prep(Xa_raw)
    va = np.where(season == 2024)[0]
    X25_raw = np.asarray(X[va]).copy(); X25_raw[:, si] = 2025.0

    # CatBoost
    cbs = []
    for sd in range(a.seeds):
        t = time.time()
        m = make_cb(sd, dev)
        cb_fit_predict(m, Pool, Xa_raw, y)
        ok, blob = cb_verify(m, Xa_raw[va][:20000])
        if not ok:
            sys.exit("CatBoost 내보내기 불일치")
        cbs.append(blob); del m
        print("  CB seed%d (%.0f초)" % (sd, time.time() - t), flush=True)
    np.savez_compressed(os.path.join(MODEL_DIR, "v13_cb.npz"),
                        **{"s%d_%s" % (i, k): v for i, b in enumerate(cbs) for k, v in b.items()},
                        n_seeds=np.array([len(cbs)]))
    p25_cb = np.mean([cb_predict(X25_raw, b) for b in cbs], axis=0)

    # torch 2종
    states_all, p25 = {}, {"CB": p25_cb}
    for kind, ep in (("ft", a.ft_epochs), ("mlp", a.mlp_epochs)):
        wm = (kind != "ft")
        Xa = DL.apply_prep(Xa_raw, prep, wm); X25 = DL.apply_prep(X25_raw, prep, wm)
        p, st = train_torch(torch, nn, kind, Xa, y, X25, a.seeds, ep)
        states_all[kind] = st; p25[kind.upper()] = p
        # vars(A) 는 mappingproxy 라 pickle 이 안 된다. 필요한 값만 평범한 dict 로.
        cfg = {"width": A.width, "depth": A.depth, "drop": A.drop,
               "dtoken": A.dtoken, "layers": A.layers, "kind": kind}
        torch.save({"states": st, "d_in": int(Xa.shape[1]), "cfg": cfg},
                   os.path.join(MODEL_DIR, "v13_%s.pt" % kind))
        print("      저장: model/v13_%s.pt (%d시드)" % (kind, len(st)), flush=True)
        del Xa, X25
    np.savez_compressed(os.path.join(MODEL_DIR, "v13_prep.npz"),
                        **{"b%d" % j: b for j, b in enumerate(prep["bnds"])},
                        n_col=np.array([len(prep["bnds"])]))
    del Xa_raw

    # ── [C] C0 ─────────────────────────────────────────────
    print("\n[C] 베이스율 상수 (2024행 season→2025)", flush=True)
    C0 = {}
    for k in ("CB", "FT", "MLP"):
        p = np.clip(p25[k], 1e-6, 1 - 1e-6)
        lg = np.log(p / (1 - p))
        C0[k] = solve_c0(lg, scales[k], C1, TARGET)
        chk = (1 / (1 + np.exp(-(scales[k] * (lg - C0[k]) + C1)))).mean()
        print("  %-3s s=%.2f C0=%.6f → 평균 %.5f" % (k, scales[k], C0[k], chk), flush=True)

    out = {"target_rate": TARGET, "logit_target_C1": C1, "n_features": 168,
           "seeds": a.seeds, "ft_epochs": a.ft_epochs, "mlp_epochs": a.mlp_epochs,
           "blend_w_cb": float(w[0]), "blend_w_ft": float(w[1]), "blend_w_mlp": float(w[2]),
           "blend_source": "train-only: val2024·2022 에서 w*=M^-1A, 두 해 평균"}
    for k in ("CB", "FT", "MLP"):
        out["model_%s" % k.lower()] = {"logit_scale": float(scales[k]),
                                       "logit_center_C0": float(C0[k]), "cap": 0.20,
                                       "target_rate": TARGET, "logit_target_C1": C1}
    json.dump(out, open(os.path.join(MODEL_DIR, "params_v13.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n저장: v13_cb.npz, v13_ft.pt, v13_mlp.pt, v13_prep.npz, params_v13.json")
    print("총 소요 %.1f분" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
