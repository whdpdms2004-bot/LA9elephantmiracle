"""V68: 찬우의 방법을 가져와 재설계 — 작은 Brier-MLP 를 앙상블 멤버로.

가져온 방법 (cowork/cw/submit_v8_mlp.zip 의 model/params.json)
    kind: mlp   hidden: [24, 12]   drop: 0.05   epochs: 45   seeds: 5
    val2024_score: 702.1
    blend_source_v8: "GBDT단독 693.7, MLP[24,12] 702.1, rho 0.880 -> 결합 742.6 (+7.06%)"

    작은 MLP 를 Brier 직접 최적화로 만들어 '멤버로만' 썼고 GBDT 와 상관 0.880 에서
    +7% 를 냈다. 찬우의 검증셋 스케일은 우리와 다르지만 구조는 그대로 가져올 수 있다.

내가 앞서 신경망을 기각한 것과 무엇이 다른가
    V14/V15 는 '성분 분해를 종단학습'시킨 구성이었다. 5성분을 한 신경망에서 내고
    합사건 확률로 학습 — 단독 331 / 402 로 GBDT 748 을 절반도 못 따라갔다.
    여기서는 성공률을 직접, 작게(24-12), Brier 로, 그리고 '멤버로만' 쓴다.
    함수족이 틀렸던 게 아니라 쓰임이 틀렸다는 가설이다.

왜 지금 유망한가 (V65)
    멤버의 가치는 단독 정확도가 아니라 base 와의 비상관성이다.
    같은 111피처 GBDT 직접 모델은 단독 751.20 으로 성분(745.30)보다 좋은데
    결합은 +23.43 vs +41.04 로 크게 밀렸다. 차이는 상관뿐(0.8529 vs 0.8219).
    함수족이 다른 MLP 는 구조적으로 상관이 더 낮아야 한다. 찬우의 0.880 이 방증.

설계
    입력   성분 라인과 같은 111피처. 결측은 중앙값 대체 후 표준화(학습 시즌 통계만).
    구조   111 -> 24 -> 12 -> 1, tanh, dropout 0.05
    손실   Brier (MSE on probability)  <- 대회 지표와 동일
    출력   sigmoid(logit(base_rate_forecast) + net)  로 레벨을 외삽에 고정
    시드   5개 평균
    F행    학습 가중치 0.20, 짧은 등판 0.5 (V64)

arm
    M0  base + 성분 라인                    = submit_032 구성
    M1  base + MLP (w 격자)
    M2  base + 성분 + MLP (MLP 비중 격자)

판정: 두 fold 모두에서 M2 가 M0 이상이어야 채택.
출력: outputs/v68_mlp_member.csv
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load, metrics, forecast_base_rate

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
FOLDS = [2023, 2024]
MLP_SEEDS = [0, 1, 2, 3, 4]
HIDDEN, DROP, EPOCHS, BATCH, LR = (24, 12), 0.05, 45, 8192, 3e-3
WM = [0.05, 0.10, 0.15, 0.20, 0.30]
EPS = 1e-7
DEV = "cuda" if torch.cuda.is_available() else "cpu"

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
IS_F = df["game_type"].astype(str).to_numpy() == "F"
pid = df["pitcher_id"].to_numpy()
NVOL = df["asof_pitcher_n"].to_numpy()

o = np.argsort(pid.astype(np.int64) * 10_000_000 + NVOL, kind="stable")
pv = df["asof_pitcher_prev1_game_success_rate"].to_numpy()[o]
gp = pid[o]
chg = np.r_[True, (gp[1:] != gp[:-1]) | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
outing = np.empty(len(df), dtype=np.int64)
outing[o] = np.cumsum(chg) - 1
od = pd.DataFrame({"outing": outing, "pid": pid, "inn": df["inning"].to_numpy()})
agg = od.groupby("outing").agg(n=("outing", "size"), pid=("pid", "first"),
                               first_inn=("inn", "min"))
agg["start"] = (agg["first_inn"] == 1).astype(int)
agg = agg.join(agg.groupby(["pid", "start"])["n"].median().rename("med"),
               on=["pid", "start"])
SHORT = np.nan_to_num((agg["n"] / agg["med"].clip(lower=1)).reindex(outing).to_numpy(),
                      nan=1.0) < 0.5
ROW_W = np.where(IS_F, 0.20, 1.0) * np.where(SHORT, 0.5, 1.0)

models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
BASE_P = {}
for f in FOLDS:
    fid = df.loc[season == f, "row_id"].to_numpy()
    if f == 2024:
        pr = pd.read_parquet(PROD).set_index("row_id").reindex(fid)
        BASE_P[f] = np.clip(pr["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                            EPS, 1 - EPS)
    else:
        acc, c = None, 0
        for mn in models:
            p = OOF_DIR / f"{mn}_fold{f}.parquet"
            if p.exists():
                v = pd.read_parquet(p).set_index("row_id").reindex(fid)["prediction"].to_numpy()
                acc = v if acc is None else acc + v
                c += 1
        BASE_P[f] = np.clip(acc / c, EPS, 1 - EPS)


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, HIDDEN[0]), nn.Tanh(), nn.Dropout(DROP),
            nn.Linear(HIDDEN[0], HIDDEN[1]), nn.Tanh(), nn.Dropout(DROP),
            nn.Linear(HIDDEN[1], 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def train_mlp(Xtr, wtr, ytr, Xva, bias, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    wt = torch.tensor(wtr, dtype=torch.float32, device=DEV)
    n = len(yt)
    b0 = float(np.log(bias / (1 - bias)))
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEV)
        net.train()
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            p = torch.sigmoid(net(xt[idx]) + b0)
            loss = (wt[idx] * (p - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    out = []
    with torch.no_grad():
        xv = torch.tensor(Xva, dtype=torch.float32, device=DEV)
        for i in range(0, len(xv), 200000):
            out.append(torch.sigmoid(net(xv[i:i + 200000]) + b0).cpu().numpy())
    return np.concatenate(out)


t0, rows = time.time(), []
lg = lambda z: np.log(z / (1 - z))
for fold in FOLDS:
    tr, va = season < fold, season == fold
    comp = np.clip(np.load(CACHE / f"v65_comp_{fold}.npy"), EPS, 1 - EPS)
    F = pd.read_parquet(CACHE / "train.parquet") if False else None
    # 성분 라인과 같은 피처 행렬을 v65 와 동일하게 재구성하지 않고,
    # 캐시된 성분 예측만 쓰고 MLP 입력은 원본 asof/상황 피처로 만든다.
    num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in [TARGET] if c in df.columns], errors="ignore")
    Xall = num.to_numpy(np.float64)
    med = np.nanmedian(Xall[tr], axis=0)
    Xall = np.where(np.isnan(Xall), med, Xall)
    mu, sd = Xall[tr].mean(0), Xall[tr].std(0) + 1e-9
    Xall = ((Xall - mu) / sd).astype(np.float32)

    y, b = y_all[va], BASE_P[fold]
    null = y.mean() * (1 - y.mean())
    ref = metrics(y, b)["bss_raw"]
    wv = BW[bucket_all[va]]
    bias = forecast_base_rate(df, tr, fold)

    acc = np.zeros(int(va.sum()))
    for s in MLP_SEEDS:
        acc += train_mlp(Xall[tr], ROW_W[tr], y_all[tr], Xall[va], bias, s)
    mlp = np.clip(acc / len(MLP_SEEDS), EPS, 1 - EPS)
    np.save(CACHE / f"v68_mlp_{fold}.npy", mlp)

    print(f"{chr(10)}fold {fold}  base {ref:9.2f}  성분단독 "
          f"{metrics(y, comp)['bss_raw']:9.2f}  MLP단독 "
          f"{metrics(y, mlp)['bss_raw']:9.2f}   [{time.time()-t0:.0f}s]")
    print(f"  상관(logit)  base x 성분 {np.corrcoef(lg(b), lg(comp))[0,1]:.4f}   "
          f"base x MLP {np.corrcoef(lg(b), lg(mlp))[0,1]:.4f}   "
          f"성분 x MLP {np.corrcoef(lg(comp), lg(mlp))[0,1]:.4f}", flush=True)

    def rec(name, q):
        d = metrics(y, q)["bss_raw"] - ref
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": fold, "arm": name, "dbss": d, "t_row": d / se})
        return d

    q0 = np.clip(wv * comp + (1 - wv) * b, EPS, 1 - EPS)
    print(f"  {'M0 base+성분':<24}{rec('M0_comp', q0):>+9.2f}")
    for wm in WM:
        d = rec(f"M1_mlp_w{wm:.2f}",
                np.clip(wm * mlp + (1 - wm) * b, EPS, 1 - EPS))
        print(f"  {'M1 base+MLP w'+f'{wm:.2f}':<24}{d:>+9.2f}")
    for wm in WM:
        d = rec(f"M2_both_w{wm:.2f}",
                np.clip(wv * comp + wm * mlp + (1 - wv - wm) * b, EPS, 1 - EPS))
        print(f"  {'M2 base+성분+MLP w'+f'{wm:.2f}':<24}{d:>+9.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v68_mlp_member.csv", index=False)
piv = res.pivot_table(index="arm", columns="fold", values="dbss")
piv["최악"] = piv.min(axis=1)
print(f"{chr(10)}{'='*58}{chr(10)}fold 별 ΔBSS (base 대비){chr(10)}{'='*58}")
print(piv.round(2).sort_values("최악", ascending=False).to_string())
m0 = piv.loc["M0_comp"]
print(f"{chr(10)}현행 M0 ({m0[2023]:+.2f} / {m0[2024]:+.2f}) 를 두 fold 모두 "
      f"넘는 arm 만 채택한다.")
print(f"{chr(10)}saved -> {OUT/'v68_mlp_member.csv'}")
