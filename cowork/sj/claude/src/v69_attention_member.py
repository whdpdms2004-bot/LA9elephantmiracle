"""V69: attention 기반 성분 분해 모델을 앙상블 멤버로. (FT-Transformer 계열)

지시: "종합할 때 DL 과 attention 알고리즘 적극적으로 사용해봐"

base 에 이미 있는 DL 을 피한다
    submit_021 이름의 tabm 이 TabM 이다. R행 Val2024 699.57, F행 전문가 가중치 0.3.
    TabM 은 k=32 병렬 헤드 MLP 로 attention 이 없다.
    feature-tokenizer + self-attention 은 이 프로젝트에 없던 함수족이다.

왜 이 방향인가 (V65)
    앙상블 멤버의 가치는 단독 정확도가 아니라 base 와의 비상관성이다.
        같은 111피처 GBDT 직접 모델   단독 751.20   상관 0.8529   결합 +23.43
        성분 분해 GBDT               단독 745.30   상관 0.8219   결합 +41.04
    함수족이 다르면 상관이 더 낮아진다. 찬우가 MLP 로 rho 0.880 에서 +7% 를 얻은 것이
    방증이다(cw params.json blend_source_v8).

V14/V15 실패와 무엇이 다른가
    그때는 '평범한 MLP 로 성분을 종단학습'했고 단독 331 / 402 로 GBDT 748 을
    절반도 못 따라갔다. 세 가지를 바꾼다.
        1) 함수족: 피처 토큰 사이 self-attention (상호작용을 구조로 준다)
        2) 보조 손실: 합성 Brier 만이 아니라 성분별 Brier 도 함께 준다
           (순수 합성 손실은 5개 성분에 신호를 나눠주지 못해 학습이 불안했다)
        3) 쓰임: 주력이 아니라 '멤버'로만. 단독이 약해도 상관이 낮으면 값을 한다.

설계
    입력    성분 라인과 같은 111피처. 학습 시즌 통계로 중앙값 대체 후 표준화.
    토큰화  각 피처 i -> x_i * W_i + b_i  (d=24). CLS 토큰 1개 추가.
    인코더  2층 x 4헤드 self-attention, FFN 48, dropout 0.1
    헤드    CLS -> 5성분 로짓. 각 성분은 2025 외삽 base_score 를 bias 로 고정.
    합성    P(success) = 1 - (p_m + p_r - p_mr + p_ob + p_oz)
    손실    Brier(합성) + 0.3 * mean(성분별 Brier)
    가중    F행 0.20, 짧은 등판 0.5 (V64)
    시드    3개 평균

arm
    A0  base + 성분 GBDT                  = submit_032 구성
    A1  base + attention 모델 (w 격자)
    A2  base + 성분 GBDT + attention (attention 비중 격자)

판정: 두 fold 모두에서 A2 가 A0 이상이어야 채택.
출력: outputs/v69_attention_member.csv
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
import component_features as CF
from harness import TARGET, OUT, CACHE, load, metrics

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
SIGN = np.array([1.0, 1.0, -1.0, 1.0, 1.0])
FOLDS = [2023, 2024]
NN_SEEDS = [0, 1, 2]
D_TOK, N_LAYER, N_HEAD, D_FF, DROP = 24, 2, 4, 48, 0.1
EPOCHS, BATCH, LR, AUX = 12, 4096, 2e-3, 0.3
WM = [0.05, 0.10, 0.15, 0.20, 0.30]
K = 300
EPS = 1e-7
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {DEV}", flush=True)

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
IS_F = df["game_type"].astype(str).to_numpy() == "F"
NVOL = df["asof_pitcher_n"].to_numpy()
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])

o = np.argsort(pid.astype(np.int64) * 10_000_000 + NVOL, kind="stable")
pv = df["asof_pitcher_prev1_game_success_rate"].to_numpy()[o]
gp = pid[o]
chg = np.r_[True, (gp[1:] != gp[:-1]) | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
outing = np.empty(len(df), dtype=np.int64)
outing[o] = np.cumsum(chg) - 1
od = pd.DataFrame({"outing": outing, "pid": pid, "inn": df["inning"].to_numpy()})
ag = od.groupby("outing").agg(n=("outing", "size"), pid=("pid", "first"),
                              first_inn=("inn", "min"))
ag["start"] = (ag["first_inn"] == 1).astype(int)
ag = ag.join(ag.groupby(["pid", "start"])["n"].median().rename("med"),
             on=["pid", "start"])
SHORT = np.nan_to_num((ag["n"] / ag["med"].clip(lower=1)).reindex(outing).to_numpy(),
                      nan=1.0) < 0.5
ROW_W = np.where(IS_F, 0.20, 1.0) * np.where(SHORT, 0.5, 1.0)

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)


def AND(*a):
    m = np.ones(len(df), bool)
    for x in a:
        m &= (x == 1)
    return np.where(ok, m.astype(float), np.nan)


LAB = {"m": ym, "r": yr, "mr": AND(ym, yr), "ob": AND(yo, yb), "oz": AND(yo, 1 - yb)}

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


def features(fold):
    tr = season < fold
    td = df.loc[tr]
    F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                 CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
    pidx = pd.MultiIndex.from_arrays([pid, bhand])
    for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
        d2 = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": ax[tr], "y": y_all[tr]})
        l0 = float(d2["y"].mean())
        g2 = d2.groupby(["p", "h"])["y"].agg(["sum", "size"])
        g3 = d2.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
        e2 = (g2["sum"] + K * l0) / (g2["size"] + K)
        e3 = (g3["sum"] + K * l0) / (g3["size"] + K)
        i3 = pd.MultiIndex.from_arrays([pid, bhand, ax])
        v2 = np.where(np.isnan(e2.reindex(pidx).to_numpy()), l0,
                      e2.reindex(pidx).to_numpy())
        v3 = np.where(np.isnan(e3.reindex(i3).to_numpy()), l0,
                      e3.reindex(i3).to_numpy())
        sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
        F[f"{tag}_split"], F[f"{tag}_rel"] = v3 - v2, sz / (sz + K)
        F[f"{tag}_w"] = (v3 - v2) * sz / (sz + K)
    return F.to_numpy(np.float64)


class FTT(nn.Module):
    """피처 토크나이저 + self-attention. CLS -> 5성분 로짓."""

    def __init__(self, d_feat, bias):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_feat, D_TOK) * 0.02)
        self.b = nn.Parameter(torch.zeros(d_feat, D_TOK))
        self.cls = nn.Parameter(torch.randn(1, 1, D_TOK) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_TOK, nhead=N_HEAD, dim_feedforward=D_FF, dropout=DROP,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYER)
        self.head = nn.Linear(D_TOK, len(COMPONENTS))
        nn.init.zeros_(self.head.weight)
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        t = x.unsqueeze(-1) * self.w + self.b               # (B, F, D)
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], dim=1)
        h = self.enc(t)[:, 0]                               # CLS
        return torch.sigmoid(self.head(h) + self.bias)      # (B, 5)


def train_ftt(Xtr, Wtr, Ytr, Ycomp, Xva, bias, seed):
    torch.manual_seed(seed)
    net = FTT(Xtr.shape[1], bias).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    n = len(Ytr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * ((n + BATCH - 1) // BATCH))
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=DEV)
    wt = torch.tensor(Wtr, dtype=torch.float32, device=DEV)
    ct = torch.tensor(np.nan_to_num(Ycomp), dtype=torch.float32, device=DEV)
    cm = torch.tensor(~np.isnan(Ycomp), dtype=torch.float32, device=DEV)
    sg = torch.tensor(SIGN, dtype=torch.float32, device=DEV)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            p = net(xt[idx])
            succ = torch.clamp(1.0 - (p * sg).sum(1), 1e-6, 1 - 1e-6)
            loss = (wt[idx] * (succ - yt[idx]) ** 2).mean()
            aux = (cm[idx] * (p - ct[idx]) ** 2).sum() / cm[idx].sum().clamp(min=1)
            (loss + AUX * aux).backward()
            opt.step()
            opt.zero_grad()
            sched.step()
    net.eval()
    out = []
    with torch.no_grad():
        xv = torch.tensor(Xva, dtype=torch.float32, device=DEV)
        for i in range(0, len(xv), 100000):
            p = net(xv[i:i + 100000])
            out.append(torch.clamp(1.0 - (p * sg).sum(1), 1e-6, 1 - 1e-6).cpu().numpy())
    return np.concatenate(out)


t0, rows = time.time(), []
lg = lambda z: np.log(z / (1 - z))
for fold in FOLDS:
    tr, va = season < fold, season == fold
    Xr = features(fold)
    med = np.nanmedian(Xr[tr], axis=0)
    Xr = np.where(np.isnan(Xr), med, Xr)
    mu, sd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-9
    X = np.clip((Xr - mu) / sd, -8, 8).astype(np.float32)

    ycomp = np.column_stack([LAB[t] for t in COMPONENTS])
    bias = []
    for t in COMPONENTS:
        a = LAB[t]
        m_ = tr & ~np.isnan(a)
        s = pd.Series(a[m_]).groupby(pd.Series(season[m_])).mean().sort_index()
        last = float(s.iloc[-1])
        r = float(np.clip(last + (last - float(s.iloc[0]))
                          / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))
        bias.append(np.log(r / (1 - r)))

    acc = np.zeros(int(va.sum()))
    for s in NN_SEEDS:
        acc += train_ftt(X[tr], ROW_W[tr], y_all[tr], ycomp[tr], X[va],
                         np.array(bias), s)
        print(f"    seed {s} 완료  [{time.time()-t0:.0f}s]", flush=True)
    att = np.clip(acc / len(NN_SEEDS), EPS, 1 - EPS)
    np.save(CACHE / f"v69_att_{fold}.npy", att)

    comp = np.clip(np.load(CACHE / f"v65_comp_{fold}.npy"), EPS, 1 - EPS)
    y, b = y_all[va], BASE_P[fold]
    null = y.mean() * (1 - y.mean())
    ref = metrics(y, b)["bss_raw"]
    wv = BW[bucket_all[va]]
    print(f"{chr(10)}fold {fold}  base {ref:9.2f}  성분GBDT {metrics(y, comp)['bss_raw']:9.2f}"
          f"  attention {metrics(y, att)['bss_raw']:9.2f}")
    print(f"  상관(logit)  base x 성분 {np.corrcoef(lg(b), lg(comp))[0,1]:.4f}   "
          f"base x att {np.corrcoef(lg(b), lg(att))[0,1]:.4f}   "
          f"성분 x att {np.corrcoef(lg(comp), lg(att))[0,1]:.4f}", flush=True)

    def rec(name, q):
        d = metrics(y, q)["bss_raw"] - ref
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": fold, "arm": name, "dbss": d, "t_row": d / se})
        return d

    print(f"  {'A0 base+성분':<26}"
          f"{rec('A0_comp', np.clip(wv*comp+(1-wv)*b, EPS, 1-EPS)):>+9.2f}")
    for wm in WM:
        print(f"  {'A1 base+att w'+f'{wm:.2f}':<26}"
              f"{rec(f'A1_att_w{wm:.2f}', np.clip(wm*att+(1-wm)*b, EPS, 1-EPS)):>+9.2f}")
    for wm in WM:
        q = np.clip(wv * comp + wm * att + (1 - wv - wm) * b, EPS, 1 - EPS)
        print(f"  {'A2 base+성분+att w'+f'{wm:.2f}':<26}"
              f"{rec(f'A2_both_w{wm:.2f}', q):>+9.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v69_attention_member.csv", index=False)
piv = res.pivot_table(index="arm", columns="fold", values="dbss")
piv["최악"] = piv.min(axis=1)
print(f"{chr(10)}{'='*58}{chr(10)}fold 별 ΔBSS (base 대비){chr(10)}{'='*58}")
print(piv.round(2).sort_values("최악", ascending=False).to_string())
a0 = piv.loc["A0_comp"]
print(f"{chr(10)}현행 A0 ({a0[2023]:+.2f} / {a0[2024]:+.2f}) 를 두 fold 모두 "
      f"넘는 arm 만 채택한다.")
print(f"{chr(10)}saved -> {OUT/'v69_attention_member.csv'}")
