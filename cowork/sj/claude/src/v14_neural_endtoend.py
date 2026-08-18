"""V14: 신경망 end-to-end 성분 분해 — 합성식을 통과해 학습한다.

원 아이디어 (12_IDEA_REGISTRY I3)
    "각 제구 실패 유형별 모델을 학습하고 그 확률들의 합사건으로 최종 확률을 구하면서
     그거 기반 오차로 학습을 계속한다"

    지금까지는 GBDT 5개를 각자 학습해 사후 합성만 했다. 합성식을 통과하는
    end-to-end 학습은 안 했다. GBDT 로는 합성식이 미분 불가라 자연스럽지 않다.

구조
    공유 인코더 -> 5개 헤드 (m, r, mr, ob, oz)
    P(success) = 1 - [p_m + p_r - p_mr + p_ob + p_oz]
    L = Brier(P, y_success) + lam_aux * mean_k BCE(head_k, y_k)

    보조 손실이 필수다. 최종 타깃만으로 학습하면 헤드의 정체성이 사라져
    단일 모델을 과다 파라미터로 쓰는 것과 같아진다. 이득은 멀티태스크 정규화에서 온다.

왜 지금인가
    현재 성분 라인은 프로덕션과 로짓 상관 0.858 이고 결합 가중치가 0.20 에서
    포화다. GBDT 끼리라 상관이 높다. 신경망은 계열이 완전히 달라 상관이 낮을 수
    있고, 그러면 가중치가 올라가 이득이 커진다.
    (팀통합 2-3: 알고리즘 선택보다 서로 다른 걸 섞는 게 이득)

arm
    N0  aux 없음 (합성 손실만)      <- 헤드 정체성이 사라지는지 확인
    N1  lam_aux = 0.3
    N2  lam_aux = 1.0
    N3  lam_aux = 3.0
    각각 3시드. 최종은 GBDT 성분 라인과 신경망을 함께 프로덕션에 섞는다.

판정: Val2024 전체 BSS, 프로덕션 836.503 대비, 균일 w.
출력: outputs/v15_neural_big.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, OUT, CACHE, load, metrics

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [11, 22, 33, 44, 55]
EPOCHS, BATCH, LR = 40, 4096, 3e-3
WS = [0.10, 0.20, 0.30, 0.40, 0.50]
EPS = 1e-6
COMPONENTS = ["m", "r", "mr", "ob", "oz"]

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float32)
ok = df["label_ok"].to_numpy() == 1
tr, va = season < 2024, season == 2024
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}

print("피처 생성 (V12 G4 구성, 105개)", flush=True)
train_df = df.loc[tr]
spec = CF.make_spec(train_df)
platoon = CF.make_platoon_table(train_df)
bat_platoon = CF.make_batter_platoon_table(train_df, {k: v[tr] for k, v in LAB.items()})
feat = CF.build(df[INPUT_COLS], spec, platoon, bat_platoon)
X = feat.to_numpy(np.float32)
print(f"  {X.shape}", flush=True)

# 표준화는 학습 fold 에서만 fit
mu = np.nanmean(X[tr], axis=0)
sd = np.nanstd(X[tr], axis=0)
sd[sd < 1e-8] = 1.0
Xs = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

Y = np.stack([np.nan_to_num(LAB[k], nan=0.0) for k in COMPONENTS], axis=1).astype(np.float32)
M = np.stack([~np.isnan(LAB[k]) for k in COMPONENTS], axis=1).astype(np.float32)


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.10),
        )
        self.head = nn.Linear(128, len(COMPONENTS))

    def forward(self, x):
        return self.head(self.enc(x))


def compose(logits):
    p = torch.sigmoid(logits)
    u = p[:, 0] + p[:, 1] - p[:, 2] + p[:, 3] + p[:, 4]     # m + r - mr + ob + oz
    return torch.clamp(1.0 - u, EPS, 1.0 - EPS)


def run(lam_aux, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(Xs.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    idx_tr = np.where(tr)[0]
    xb_all = torch.from_numpy(Xs).to(DEV)
    yb_all = torch.from_numpy(y_all).to(DEV)
    cb_all = torch.from_numpy(Y).to(DEV)
    mb_all = torch.from_numpy(M).to(DEV)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * (len(idx_tr) // BATCH + 1))
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(idx_tr)
        for i in range(0, len(perm), BATCH):
            b = torch.from_numpy(perm[i:i + BATCH]).to(DEV)
            lg = net(xb_all[b])
            loss = ((compose(lg) - yb_all[b]) ** 2).mean()
            if lam_aux > 0:
                aux = bce(lg, cb_all[b]) * mb_all[b]
                loss = loss + lam_aux * (aux.sum() / mb_all[b].sum().clamp(min=1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    out = []
    with torch.no_grad():
        vi = np.where(va)[0]
        for i in range(0, len(vi), 65536):
            b = torch.from_numpy(vi[i:i + 65536]).to(DEV)
            out.append(compose(net(xb_all[b])).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va].astype(np.float64)
gt = prod["game_type"].astype(str).to_numpy()
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())


def logit(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


t0, rows, preds = time.time(), [], {}
print(f"\n{'arm':<14}{'단독BSS':>10}{'corr':>8}   " + "".join(f"w{w:<6.2f}" for w in WS),
      flush=True)
for name, lam in [("N0_noaux", 0.0), ("N1_aux0.3", 0.3),
                  ("N3_aux3.0", 3.0)]:
    p_nn = np.clip(np.mean([run(lam, s) for s in SEEDS], axis=0), EPS, 1 - EPS)
    preds[name] = p_nn
    solo = metrics(y_va, p_nn)["bss_raw"]
    corr = float(np.corrcoef(logit(p_prod), logit(p_nn))[0, 1])
    line = f"{name:<14}{solo:>10.2f}{corr:>8.4f}   "
    for w in WS:
        q = np.clip(w * p_nn + (1 - w) * p_prod, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"arm": name, "lam_aux": lam, "solo_bss": solo, "corr": corr,
                     "w": w, "bss": mm["bss_raw"], "dbss": d, "se_row": se,
                     "t_row": d / se})
        line += f"{d:+7.2f}"
    print(line + f"   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v15_neural_big.csv", index=False)
best = res.sort_values("dbss", ascending=False).iloc[0]
print(f"\n신경망 단독 최고: {best.arm} w={best.w:.2f}  ΔBSS {best.dbss:+.3f}  "
      f"corr {best['corr']:.4f}  t_row {best.t_row:+.2f}")
print(f"  [GBDT 성분 라인(submit_027)은 corr 0.858, ΔBSS +23.99]")
print(f"\nsaved -> {OUT/'v14_neural.csv'}")
np.save(CACHE / "v15_neural_preds.npy",
        np.stack([preds[k] for k in preds]), allow_pickle=False)
print(f"saved -> {CACHE/'v14_neural_preds.npy'}  (3자 결합용)")
