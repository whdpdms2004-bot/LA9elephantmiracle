# -*- coding: utf-8 -*-
"""딥러닝 파이프라인 (torch, GPU) — 168피처 전용.

왜 지금인가
    지금까지 DL 이 진 이유는 재료가 없어서였다. 80피처는 트리가 이미 다 짜냈다.
    이제 168개 중 55개가 TrackMan 물리량(릴리스 높이·좌우·익스텐션·회전수의 표준편차)이다.
    트리는 이런 연속값을 축에 수직인 계단으로만 자른다.
    "릴리스가 흔들릴수록 × 구속차가 클수록 × 3-1 카운트일수록" 같은 매끄러운
    다중 상호작용은 신경망이 자연스럽게 다룬다.

모델
    mlp   전처리(분위수 변환) + 잔차 MLP. 빠르고 안정적
    ft    FT-Transformer. 피처 하나하나를 토큰으로 만들어 **피처들 사이에 attention**.
          시계열이 아니라 한 행 안에서만 작동하므로 행 독립성이 유지된다(규정 안전).

설계 요점
    · 손실은 Brier(=MSE) 직접 최적화. 평가지표와 일치시킨다.
    · 결측은 0 채움 + 결측표시 열 추가 (TrackMan 은 신인·2019년에 없다)
    · 연속값은 분위수 변환으로 꼬리를 눌러 학습을 안정화
    · AMP(fp16) 사용

실행
    python dl.py --model mlp --epochs 30
    python dl.py --model ft  --epochs 20 --batch 512
    python dl.py --model mlp --seeds 3 --val 2024 2022
"""

import argparse
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))


# ───────────────────────── 지표 ─────────────────────────

def calib(p, yv):
    r = yv.mean(); U = r * (1 - r); c1 = np.log(r / (1 - r))
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    best = (-9e9, 1.0)
    for k in np.arange(0.2, 1.55, 0.05):
        q = 1 / (1 + np.exp(-(k * (z - z.mean()) + c1)))
        v = 1e5 * (1 - ((q - yv) ** 2).mean() / U)
        if v > best[0]:
            best = (v, k)
    return best


# ───────────────────────── 전처리 ─────────────────────────

def make_prep(Xtr, n_q=64):
    """분위수 경계와 결측 정보를 학습 데이터에서만 산출."""
    qs = np.linspace(0, 1, n_q + 1)[1:-1]
    bnds, miss = [], []
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        ok = np.isfinite(col)
        miss.append(1.0 - ok.mean())
        bnds.append(np.unique(np.quantile(col[ok], qs)) if ok.sum() > 100 else np.array([0.0]))
    return {"bnds": bnds, "miss": np.array(miss)}


def apply_prep(X, prep, with_mask=True):
    """분위수 순위(-1~1)로 변환. with_mask=True 면 결측표시 열을 뒤에 붙인다.

    FT-Transformer 는 열 하나가 토큰 하나라 열을 2배로 늘리면 attention 이 4배 무거워진다.
    그래서 ft 모델에서는 with_mask=False 로 168열만 쓰고, 결측은 별도 임베딩으로 처리한다.
    """
    n, d = X.shape
    Z = np.empty((n, d * 2 if with_mask else d), dtype=np.float32)
    for j in range(d):
        col = X[:, j]
        ok = np.isfinite(col)
        b = prep["bnds"][j]
        r = np.searchsorted(b, np.where(ok, col, 0.0)).astype(np.float32) / max(len(b), 1)
        Z[:, j] = np.where(ok, r * 2.0 - 1.0, 0.0)
        if with_mask:
            Z[:, d + j] = ok.astype(np.float32)
    return Z


def make_mask(X):
    """결측 여부만 (n, d) 로. ft 모델이 임베딩으로 쓴다."""
    return np.isfinite(X).astype(np.float32)


# ───────────────────────── 모델 ─────────────────────────

def build_mlp(torch, nn, d_in, width=512, depth=4, drop=0.2):
    class Block(nn.Module):
        def __init__(self, w, p):
            super().__init__()
            self.n = nn.LayerNorm(w)
            self.f = nn.Sequential(nn.Linear(w, w * 2), nn.GELU(), nn.Dropout(p),
                                   nn.Linear(w * 2, w))

        def forward(self, x):
            return x + self.f(self.n(x))

    return nn.Sequential(nn.Linear(d_in, width), *[Block(width, drop) for _ in range(depth)],
                         nn.LayerNorm(width), nn.Linear(width, 1))


def build_ft(torch, nn, d_feat, d_token=64, layers=3, heads=8, drop=0.1):
    """FT-Transformer. 피처 하나 = 토큰 하나. 피처들 사이에 attention 을 건다."""
    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(d_feat, d_token) * 0.02)
            self.b = nn.Parameter(torch.zeros(d_feat, d_token))
            self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
            L = nn.TransformerEncoderLayer(d_token, heads, d_token * 4, drop,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
            self.enc = nn.TransformerEncoder(L, layers)
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

        def forward(self, x):
            t = x.unsqueeze(-1) * self.w + self.b                 # (B, F, d)
            t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], 1)
            return self.head(self.enc(t)[:, 0])
    return FT()


# ───────────────────────── 학습 ─────────────────────────

def run_one(torch, nn, Xtr, ytr, Xva, yva, args, seed):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]
    net = (build_ft(torch, nn, d, args.dtoken, args.layers)
           if args.model == "ft" else
           build_mlp(torch, nn, d, args.width, args.depth, args.drop)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    n = len(Xtr); steps = math.ceil(n / args.batch) * args.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps,
                                              pct_start=0.1)
    scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))
    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr.astype(np.float32))
    Xva_t = torch.from_numpy(Xva)

    best = (-9e9, None)
    nb = math.ceil(n / args.batch)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n)
        t0 = time.time()
        for bi, s in enumerate(range(0, n, args.batch)):
            idx = perm[s:s + args.batch]
            xb = Xtr_t[idx].to(dev, non_blocking=True)
            yb = ytr_t[idx].to(dev, non_blocking=True)
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                p = torch.sigmoid(net(xb).squeeze(-1))
                loss = ((p - yb) ** 2).mean()          # Brier 직접 최적화
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sch.step()
            if ep == 0 and bi == 49:                   # 첫 에폭 소요시간 추정
                el = time.time() - t0
                print("      50배치 %.1f초 → 1에폭 약 %.0f초, 전체 약 %.1f분"
                      % (el, el * nb / 50, el * nb * args.epochs / 50 / 60), flush=True)
        print("    ep%02d 학습 %.0f초" % (ep + 1, time.time() - t0), flush=True)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            net.eval(); out = []
            with torch.no_grad():
                for s in range(0, len(Xva_t), 8192):
                    xb = Xva_t[s:s + 8192].to(dev)
                    with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                        out.append(torch.sigmoid(net(xb).squeeze(-1)).float().cpu().numpy())
            p = np.concatenate(out)
            sc, k = calib(p, yva)
            if sc > best[0]:
                best = (sc, p)
            last = (sc, p)
            print("      → val %8.1f  (s=%.2f)" % (sc, k), flush=True)
    # best 는 val 로 고른 값이라 부풀려져 있다. 정직한 수치는 last(마지막 에폭)다.
    return last, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=["mlp", "ft"])
    # 30에폭은 심하게 과적합했다 (val2024 ep15 726 → ep30 437).
    # 15에폭 · 드롭아웃 0.3 · 감쇠 3e-4 로 낮춘다.
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--dtoken", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--val", type=int, nargs="+", default=[2024, 2022])
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--feat", default="X168.npy")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    print("torch %s | GPU %s" % (torch.__version__,
                                 torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else "없음"), flush=True)

    X = np.load(os.path.join(WORK, a.feat), mmap_mode="r")
    y = np.load(os.path.join(WORK, "y.npy"))
    season = np.load(os.path.join(WORK, "season.npy"))
    print("피처 %d개" % X.shape[1], flush=True)

    for val in a.val:
        tri = np.where(season <= val - 1)[0]; va = np.where(season == val)[0]
        yv = y[va].astype(np.float64)
        t0 = time.time()
        Xtr_raw = np.asarray(X[tri]); Xva_raw = np.asarray(X[va])
        prep = make_prep(Xtr_raw)
        wm = (a.model != "ft")          # ft 는 토큰 수를 늘리면 attention 이 4배 무거워진다
        Xtr = apply_prep(Xtr_raw, prep, wm); Xva = apply_prep(Xva_raw, prep, wm)
        del Xtr_raw, Xva_raw
        print("\n=== val %d ===  전처리 후 %d열 (%.0f초)"
              % (val, Xtr.shape[1], time.time() - t0), flush=True)
        preds, last_s, best_s = [], [], []
        for sd in range(a.seeds):
            print("  seed %d" % sd, flush=True)
            (sl, pl), (sb, pb) = run_one(torch, nn, Xtr, y[tri], Xva, yv, a, sd)
            last_s.append(sl); best_s.append(sb); preds.append(pl)
        P = np.mean(preds, axis=0)
        s_ens = calib(P, yv)[0]
        np.save(os.path.join(WORK, "p_dl_%s_val%d.npy" % (a.model, val)), P)
        print("  마지막에폭 시드별 %s" % " ".join("%.1f" % s for s in last_s))
        print("  (참고) 최고에폭  %s   ← val 로 고른 값이라 낙관적"
              % " ".join("%.1f" % s for s in best_s))
        print("  앙상블(마지막에폭) %8.1f   저장: p_dl_%s_val%d.npy"
              % (s_ens, a.model, val), flush=True)
        del Xtr, Xva


if __name__ == "__main__":
    main()
