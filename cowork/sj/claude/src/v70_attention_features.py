"""V70: 피처 엔지니어링에 attention 을 쓴다 — 상황이 투수 프로파일을 조회한다.

지시: "DL 과 어텐션을 모델링 및 피처 엔지니어링 각 부분에 사용해보고 가장 적절한 설계"
    V69 = 모델링 축 (FT-Transformer 멤버)
    V70 = 피처 엔지니어링 축 (cross-attention 으로 피처를 만들어 GBDT 에 준다)

무엇을 대체하는가
    지금 성분 라인의 핵심 피처는 손으로 만든 계층 차감 스플릿이다.
        split(p,h,축) = EB(투수, 타자손, 축) − EB(투수, 타자손)
    축마다 사람이 정해서 넣었다 (손 / 카운트 / 이닝 / 타자).
    V33 에서 축을 6종 더 넣었더니 전부 실패했다 — 사람이 고를 축은 소진됐다.

    대신 투수의 조건부 프로파일 '전체'를 토큰으로 주고, 현재 상황이 그 위에서
    무엇을 볼지 attention 이 정하게 한다. 축 선택을 학습으로 넘긴다.

설계
    프로파일 토큰 (투수마다 12개 = 카운트6 x 타자손2)
        각 셀 = [성공, m, r, mr, ob, oz] 6종의 계층 차감값 + 신뢰도 n/(n+K)
        계층 차감: EB(p, cell) − EB(p) − EB(cell) + lg   (V19 원리, 단위 동일)
        학습 시즌만으로 만든다. 투수별 정적 테이블이라 추론 시 행 독립.

    쿼리 = 현재 행의 상황 (카운트버킷, 타자손, 이닝군, 아웃, 주자, 볼/스트라이크)
    cross-attention 1층 4헤드: 쿼리가 12개 프로파일 셀을 조회 -> d=16 벡터
    그 벡터를 5성분 로짓으로 보내 학습(합성 Brier + 성분별 보조)

    학습이 끝나면 attention 출력 d=16 을 '피처'로 뽑아 GBDT 성분 모델에 준다.

누수 처리 — 여기가 핵심이다
    attention 모듈이 학습에 쓴 행의 피처를 그대로 만들면 자기 라벨이 샌다.
    V1/V37/V59 에서 네 번 본 붕괴다. 그래서 순방향 내부 분할로 뽑는다.
        검증 fold V 행   -> season < V 로 학습한 모듈이 생성
        학습 행 (시즌 s) -> season < s 로 학습한 모듈이 생성
    즉 어떤 행의 피처도 그 행을 본 적 없는 모듈이 만든다.
    프로파일 토큰 자체도 해당 시점 이전 시즌만으로 만든다.

arm
    B0  현행 (손으로 만든 계층 차감 4축)
    B1  현행 + attention 피처 16
    B2  attention 피처만 (손 계층차감 스플릿 제거) — 학습이 사람을 대체하는가
출력: outputs/v70_attention_features.csv
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, OUT, CACHE, BASE_PARAMS, load, metrics

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, K = 400, 300
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
SIGN = np.array([1.0, 1.0, -1.0, 1.0, 1.0])
FOLDS = [2023, 2024]
D_ATT, N_HEAD, D_OUT = 16, 4, 16
EPOCHS, BATCH, LR, AUX = 8, 8192, 3e-3, 0.3
NN_SEED = 0
EPS = 1e-7
DEV = "cuda" if torch.cuda.is_available() else "cpu"

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
inning = df["inning"].to_numpy()
outs = df["outs_before"].to_numpy()
nrun = df["num_runners_on"].to_numpy()


def cnt6(b, s):
    ahead, even = s > b, s == b
    tb, ts = b == 3, s == 2
    o = np.zeros(len(b), dtype=np.int64)
    o[ahead & ~ts] = 1
    o[ts & ~tb] = 2
    o[tb & ~ts] = 3
    o[tb & ts] = 4
    o[(~ahead) & (~even) & (~tb)] = 5
    return o


CB6 = cnt6(balls, strikes)
CELL = CB6 * 2 + (bhand == 2).astype(np.int64)          # 12셀
inn_b = np.digitize(inning, [4, 7, 10])

o_ = np.argsort(pid.astype(np.int64) * 10_000_000 + NVOL, kind="stable")
pv = df["asof_pitcher_prev1_game_success_rate"].to_numpy()[o_]
gp = pid[o_]
chg = np.r_[True, (gp[1:] != gp[:-1]) | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
outing = np.empty(len(df), dtype=np.int64)
outing[o_] = np.cumsum(chg) - 1
ag = pd.DataFrame({"outing": outing, "pid": pid, "inn": inning}).groupby("outing").agg(
    n=("outing", "size"), pid=("pid", "first"), first_inn=("inn", "min"))
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
TARGETS = {"success": y_all, **LAB}
PIDS = np.unique(pid)
POS = {p: i for i, p in enumerate(PIDS)}
PROW = np.array([POS[p] for p in pid])

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


def profile_tokens(fit_mask):
    """(투수, 12셀) x [6타깃 계층차감 + 신뢰도].  fit_mask 시즌만 사용."""
    T = np.zeros((len(PIDS), 12, 7), dtype=np.float32)
    for j, (name, arr) in enumerate(TARGETS.items()):
        m = fit_mask & ~np.isnan(arr)
        d = pd.DataFrame({"p": PROW[m], "c": CELL[m], "y": arr[m]})
        lg = float(d["y"].mean())
        g = d.groupby(["p", "c"])["y"].agg(["sum", "size"])
        gp_ = d.groupby("p")["y"].agg(["sum", "size"])
        gc = d.groupby("c")["y"].agg(["sum", "size"])
        eb = ((g["sum"] + K * lg) / (g["size"] + K)).unstack(fill_value=np.nan)
        ep = (gp_["sum"] + K * lg) / (gp_["size"] + K)
        ec = (gc["sum"] + K * lg) / (gc["size"] + K)
        M = eb.reindex(index=range(len(PIDS)), columns=range(12)).to_numpy()
        M = M - ep.reindex(range(len(PIDS))).to_numpy()[:, None] \
              - ec.reindex(range(12)).to_numpy()[None, :] + lg
        T[:, :, j] = np.nan_to_num(M)
        if j == 0:
            sz = g["size"].unstack(fill_value=0.0).reindex(
                index=range(len(PIDS)), columns=range(12)).fillna(0.0).to_numpy()
            T[:, :, 6] = sz / (sz + K)
    return T


class CrossAtt(nn.Module):
    """상황 쿼리가 투수 프로파일 12셀을 조회한다."""

    def __init__(self, d_q, bias):
        super().__init__()
        self.tok = nn.Linear(7, D_ATT)
        self.cell = nn.Parameter(torch.randn(12, D_ATT) * 0.02)
        self.q = nn.Sequential(nn.Linear(d_q, D_ATT), nn.GELU(),
                               nn.Linear(D_ATT, D_ATT))
        self.att = nn.MultiheadAttention(D_ATT, N_HEAD, batch_first=True)
        self.proj = nn.Sequential(nn.LayerNorm(D_ATT), nn.Linear(D_ATT, D_OUT),
                                  nn.GELU())
        self.head = nn.Linear(D_OUT, len(COMPONENTS))
        nn.init.zeros_(self.head.weight)
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

    def embed(self, tok, q):
        kv = self.tok(tok) + self.cell                    # (B, 12, D)
        qq = self.q(q).unsqueeze(1)                       # (B, 1, D)
        z, _ = self.att(qq, kv, kv)
        return self.proj(z.squeeze(1))

    def forward(self, tok, q):
        e = self.embed(tok, q)
        return torch.sigmoid(self.head(e) + self.bias), e


def situation(rows):
    cb = np.eye(6, dtype=np.float32)[CB6[rows]]
    bh = np.eye(2, dtype=np.float32)[(bhand[rows] == 2).astype(int)]
    ib = np.eye(4, dtype=np.float32)[np.clip(inn_b[rows], 0, 3)]
    ob = np.eye(3, dtype=np.float32)[np.clip(outs[rows], 0, 2)]
    rn = np.eye(4, dtype=np.float32)[np.clip(nrun[rows], 0, 3)]
    extra = np.column_stack([
        balls[rows] / 3.0, strikes[rows] / 2.0, np.clip(inning[rows], 1, 12) / 12.0,
        np.log1p(NVOL[rows]) / 10.0, IS_F[rows].astype(np.float32)]).astype(np.float32)
    return np.hstack([cb, bh, ib, ob, rn, extra])


def train_extract(fit_mask, out_mask, bias):
    """fit_mask 로 학습해 out_mask 행의 attention 임베딩을 뽑는다."""
    torch.manual_seed(NN_SEED)
    T = torch.tensor(profile_tokens(fit_mask), device=DEV)
    fit = np.where(fit_mask)[0]
    Q = situation(fit)
    net = CrossAtt(Q.shape[1], bias).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    qt = torch.tensor(Q, device=DEV)
    pr = torch.tensor(PROW[fit], device=DEV, dtype=torch.long)
    yt = torch.tensor(y_all[fit], dtype=torch.float32, device=DEV)
    wt = torch.tensor(ROW_W[fit], dtype=torch.float32, device=DEV)
    ct = torch.tensor(np.nan_to_num(np.column_stack([LAB[c][fit] for c in COMPONENTS])),
                      dtype=torch.float32, device=DEV)
    cm = torch.tensor(~np.isnan(np.column_stack([LAB[c][fit] for c in COMPONENTS])),
                      dtype=torch.float32, device=DEV)
    sg = torch.tensor(SIGN, dtype=torch.float32, device=DEV)
    n = len(fit)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            p, _ = net(T[pr[idx]], qt[idx])
            succ = torch.clamp(1.0 - (p * sg).sum(1), 1e-6, 1 - 1e-6)
            loss = (wt[idx] * (succ - yt[idx]) ** 2).mean()
            aux = (cm[idx] * (p - ct[idx]) ** 2).sum() / cm[idx].sum().clamp(min=1)
            (loss + AUX * aux).backward()
            opt.step()
            opt.zero_grad()
    net.eval()
    outi = np.where(out_mask)[0]
    Qo = torch.tensor(situation(outi), device=DEV)
    pro = torch.tensor(PROW[outi], device=DEV, dtype=torch.long)
    res = np.zeros((len(outi), D_OUT), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(outi), 100000):
            res[i:i + 100000] = net.embed(T[pro[i:i + 100000]],
                                          Qo[i:i + 100000]).cpu().numpy()
    return res


def comp_bias(fit_mask):
    out = []
    for t in COMPONENTS:
        a = LAB[t]
        m_ = fit_mask & ~np.isnan(a)
        s = pd.Series(a[m_]).groupby(pd.Series(season[m_])).mean().sort_index()
        last = float(s.iloc[-1])
        r = float(np.clip(last + (last - float(s.iloc[0]))
                          / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))
        out.append(np.log(r / (1 - r)))
    return np.array(out)


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def gbdt_line(X, fold):
    tr, va = season < fold, season == fold
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & ~np.isnan(arr)
        s_ = pd.Series(arr[mm]).groupby(pd.Series(season[mm])).mean().sort_index()
        bs = float(np.clip(float(s_.iloc[-1]) + (float(s_.iloc[-1]) - float(s_.iloc[0]))
                           / (float(s_.index[-1]) - float(s_.index[0])), 0.005, 0.995))
        prm = {**BASE_PARAMS, "base_score": bs,
               **params_for(float(np.nanmean(arr[mm])))}
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=ROW_W[mm])
        d_va = xgb.DMatrix(X[va])
        p_tr = Pool(X[mm], arr[mm], weight=ROW_W[mm])
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(X[va])[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


t0, rows = time.time(), []
SPLIT_COLS = ["platoon_split", "platoon_rel", "platoon_w", "cnt_split", "cnt_rel",
              "cnt_w", "inn_split", "inn_rel", "inn_w"]
for fold in FOLDS:
    tr, va = season < fold, season == fold
    td = df.loc[tr]
    BF = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
    pidx = pd.MultiIndex.from_arrays([pid, bhand])
    for tag, ax in [("cnt", CB6), ("inn", inn_b)]:
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
        BF[f"{tag}_split"], BF[f"{tag}_rel"] = v3 - v2, sz / (sz + K)
        BF[f"{tag}_w"] = (v3 - v2) * sz / (sz + K)

    # 순방향 내부 분할로 attention 피처 추출 (자기 라벨 누수 차단)
    ATT = np.zeros((len(df), D_OUT), dtype=np.float32)
    seasons_tr = sorted(set(season[tr]))
    for s_ in seasons_tr[1:]:
        fit = season < s_
        ATT[season == s_] = train_extract(fit, season == s_, comp_bias(fit))
        print(f"    attention 피처: 시즌 {s_} 생성 (학습 season<{s_})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    ATT[va] = train_extract(tr, va, comp_bias(tr))
    print(f"    attention 피처: fold {fold} 생성 (학습 season<{fold})  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    first = seasons_tr[0]
    keep = ~(season == first)                      # 첫 시즌은 피처 없음 -> 학습 제외

    y, b = y_all[va], BASE_P[fold]
    null = y.mean() * (1 - y.mean())
    ref = metrics(y, b)["bss_raw"]
    wv = BW[bucket_all[va]]
    lgf = lambda z: np.log(z / (1 - z))
    print(f"{chr(10)}fold {fold}   base {ref:9.2f}")
    print(f"  {'arm':<10}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}{'t_row':>8}")
    for arm in ["B0", "B1", "B2"]:
        F = BF.copy()
        if arm == "B2":
            F = F.drop(columns=[c for c in SPLIT_COLS if c in F.columns])
        if arm in ("B1", "B2"):
            for i in range(D_OUT):
                F[f"att_{i:02d}"] = ATT[:, i]
        X = F.to_numpy(np.float32)
        Xf = np.where(keep[:, None], X, X)          # 첫 시즌 행은 att=0 이므로 그대로
        p_ie = gbdt_line(Xf, fold)
        np.save(CACHE / f"v70_{arm}_{fold}.npy", p_ie)
        solo = metrics(y, p_ie)["bss_raw"]
        corr = float(np.corrcoef(lgf(b), lgf(p_ie))[0, 1])
        q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
        d = metrics(y, q)["bss_raw"] - ref
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": fold, "arm": arm, "n_features": F.shape[1],
                     "solo_bss": solo, "corr": corr, "dbss": d, "t_row": d / se})
        print(f"  {arm:<10}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}"
              f"{d/se:>8.2f}   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v70_attention_features.csv", index=False)
piv = res.pivot_table(index="arm", columns="fold", values="dbss")
sol = res.pivot_table(index="arm", columns="fold", values="solo_bss")
print(f"{chr(10)}{'='*58}{chr(10)}B0 대비{chr(10)}{'='*58}")
print("ΔBSS")
print(piv.subtract(piv.loc["B0"], axis=1).round(2).to_string())
print(f"{chr(10)}성분단독")
print(sol.subtract(sol.loc["B0"], axis=1).round(2).to_string())
print(f"{chr(10)}B1 이 오르면 attention 이 사람이 만든 축에 정보를 더한 것이고,")
print(f"B2 가 B0 에 근접하면 attention 이 축 선택을 대체할 수 있다는 뜻이다.")
print(f"{chr(10)}saved -> {OUT/'v70_attention_features.csv'}")
