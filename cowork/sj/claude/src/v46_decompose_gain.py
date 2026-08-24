"""V46: V45 의 +53 이 어디서 오는가를 분해한다. (CPU 전용)

V45 결과
    base 변형만 NNLS         전체 839.05   held-out 리프트  −0.47
    +성분 라인 1종 (P2_msplit) 891.66              +53.73
    +2종                      894.10              +52.72
    +3종                      895.65              +52.14
    전부 31종                 895.87              +51.41

    라인을 늘려도 held-out 은 안 오른다. 이득은 앙상블이 아니라
    '가중치를 격자가 아니라 데이터로 푼 것'에서 나온다.

분해할 것
    A  현행 라인 + w=0.25 고정            = submit_031 근사
    B  현행 라인 + w 를 NNLS 로 적합       <- 가중치 효과만
    C  P2_msplit 라인 + w=0.25 고정        <- 라인 교체 효과만
    D  P2_msplit + NNLS                    <- 둘 다
    E  D + base 변형까지 NNLS              <- V45 K=1

    그리고 실제로 뽑힌 가중치를 찍는다. w 가 0.25 에서 얼마나 멀어졌는지가
    이 실험의 핵심 숫자다.

    fold 전이도 같이 본다 — 2022/2023 에 대해 R0/R1 라인이 캐시돼 있으므로
    2024 에서 적합한 w 를 그 두 fold 에 그대로 적용하면 어떻게 되는지 잰다.
    사용자가 세 fold 게이트를 쓰지 말라고 했으므로 채택 조건으로 쓰지 않고
    정보로만 보고한다.

출력: outputs/v46_decompose.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
NPZ = PROD.parent / "reverse20_submission_components.npz"
ALPHAS = [1.00, 1.25, 1.50]
SCALES = [0.10, 0.30, 0.475]
N_SPLIT = 20
EPS = 1e-7

df = load()
season = df["season"].to_numpy()
va = season == 2024
ids = df.loc[va, "row_id"].to_numpy()
y = df[TARGET].to_numpy(np.float64)[va]
n = len(y)

prod = pd.read_parquet(PROD).set_index("row_id").reindex(ids)
z = np.load(NPZ, allow_pickle=True)
order = pd.Index(z["row_id"]).get_indexer(ids)
rc = z["r_correction"].astype(np.float64)[order]
rev = z["reverse20"].astype(np.float64)[order]
p021 = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
BASE = {f"base_a{a:.2f}_s{s:.3f}":
        np.clip(p021 + (a - 1.0) * rc + (s - 0.40) * 0.6085 * rev, EPS, 1 - EPS)
        for a in ALPHAS for s in SCALES}
CUR = np.clip(np.load(CACHE / "v38_R0_current_2024.npy"), EPS, 1 - EPS)
MSP = np.clip(np.load(CACHE / "v35_P2_msplit.npy"), EPS, 1 - EPS)


def bss(mask, p, yy=None):
    yy = y if yy is None else yy
    a, b = yy[mask], np.clip(p[mask], EPS, 1 - EPS)
    nl = a.mean() * (1 - a.mean())
    return 100000 * (nl - ((b - a) ** 2).mean()) / nl


def fit(cols, mask, target=None, nn=None):
    t = y if target is None else target
    nn = n if nn is None else nn
    X = np.column_stack([c[mask] for c in cols]
                        + [np.ones(int(mask.sum())), -np.ones(int(mask.sum()))])
    w, _ = nnls(X, t[mask])
    return w


def apply(cols, w, size=None):
    q = np.zeros(n if size is None else size)
    for c, wi in zip(cols, w[:len(cols)]):
        q += wi * c
    return np.clip(q + w[-2] - w[-1], EPS, 1 - EPS)


splits = [np.random.default_rng(1000 + i).random(n) < 0.5 for i in range(N_SPLIT)]
ALL = np.ones(n, bool)
ref = bss(ALL, p021)


def run(label, cols, fixed=None):
    if fixed is not None:
        q = np.clip(fixed, EPS, 1 - EPS)
        full = bss(ALL, q)
        lift = float(np.median([bss(~sp, q) - bss(~sp, p021) for sp in splits]))
        return {"arm": label, "full_bss": full, "hold_lift": lift, "weights": "고정"}
    w = fit(cols, ALL)
    q = apply(cols, w)
    lift = float(np.median([bss(~sp, apply(cols, fit(cols, sp))) - bss(~sp, p021)
                            for sp in splits]))
    return {"arm": label, "full_bss": bss(ALL, q), "hold_lift": lift, "w": w,
            "weights": " ".join(f"{v:.3f}" for v in w)}


bkeys = sorted(BASE)
rows = [
    run("A 현행라인 w0.25 고정", None, fixed=0.25 * CUR + 0.75 * p021),
    run("B 현행라인 NNLS", [p021, CUR]),
    run("C msplit w0.25 고정", None, fixed=0.25 * MSP + 0.75 * p021),
    run("D msplit NNLS", [p021, MSP]),
    run("E D + base변형 NNLS", [BASE[k] for k in bkeys] + [MSP]),
    run("F 현행+msplit NNLS", [p021, CUR, MSP]),
]
print(f"기준 submit_021 {ref:.2f}{chr(10)}")
print(f"{'arm':<24}{'전체':>10}{'021 대비':>10}{'held-out 리프트':>16}")
for r in rows:
    print(f"{r['arm']:<24}{r['full_bss']:>10.2f}{r['full_bss']-ref:>+10.2f}"
          f"{r['hold_lift']:>16.2f}")

print(f"{chr(10)}{'='*80}{chr(10)}뽑힌 가중치{chr(10)}{'='*80}")
for r in rows:
    if "w" not in r:
        continue
    w = r["w"]
    if r["arm"].startswith("B"):
        print(f"  B  base {w[0]:.3f}   현행라인 {w[1]:.3f}   절편 {w[-2]-w[-1]:+.5f}"
              f"    -> 성분 비중 {w[1]/(w[0]+w[1]):.3f}")
    elif r["arm"].startswith("D"):
        print(f"  D  base {w[0]:.3f}   msplit   {w[1]:.3f}   절편 {w[-2]-w[-1]:+.5f}"
              f"    -> 성분 비중 {w[1]/(w[0]+w[1]):.3f}")
    elif r["arm"].startswith("F"):
        print(f"  F  base {w[0]:.3f}   현행 {w[1]:.3f}   msplit {w[2]:.3f}   "
              f"절편 {w[-2]-w[-1]:+.5f}")
    elif r["arm"].startswith("E"):
        nz = [(bkeys[i], w[i]) for i in range(len(bkeys)) if w[i] > 1e-4]
        print(f"  E  성분 {w[len(bkeys)]:.3f}   절편 {w[-2]-w[-1]:+.5f}")
        for k, v in nz:
            print(f"       {k:<22}{v:.3f}")

# ---------------------------------------------- 2022/2023 전이 (정보용)
print(f"{chr(10)}{'='*80}{chr(10)}2024 에서 적합한 가중치를 2022/2023 에 적용 (정보용)"
      f"{chr(10)}{'='*80}")
import re
models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
wD = [r for r in rows if r["arm"].startswith("D")][0]["w"]
wB = [r for r in rows if r["arm"].startswith("B")][0]["w"]
out = []
for fold in [2022, 2023]:
    fid = df.loc[season == fold, "row_id"].to_numpy()
    acc, cnt = None, 0
    for mn in models:
        f = OOF_DIR / f"{mn}_fold{fold}.parquet"
        if f.exists():
            v = pd.read_parquet(f).set_index("row_id").reindex(fid)["prediction"].to_numpy()
            acc = v if acc is None else acc + v
            cnt += 1
    b = np.clip(acc / cnt, EPS, 1 - EPS)
    yy = df[TARGET].to_numpy(np.float64)[season == fold]
    r0 = np.clip(np.load(CACHE / f"v38_R0_current_{fold}.npy"), EPS, 1 - EPS)
    r1 = np.clip(np.load(CACHE / f"v38_R1_msplit_{fold}.npy"), EPS, 1 - EPS)
    nl = yy.mean() * (1 - yy.mean())

    def sc(p):
        p = np.clip(p, EPS, 1 - EPS)
        return 100000 * (nl - ((p - yy) ** 2).mean()) / nl

    base_s = sc(b)
    for nm, line, w in [("현행 w0.25", r0, None), ("현행 NNLS-w", r0, wB),
                        ("msplit w0.25", r1, None), ("msplit NNLS-w", r1, wD)]:
        q = (0.25 * line + 0.75 * b) if w is None else \
            (w[0] * b + w[1] * line + w[-2] - w[-1])
        out.append({"fold": fold, "arm": nm, "bss": sc(q), "dbss": sc(q) - base_s})
    print(f"  fold {fold}  base {base_s:9.2f}")
    for o in out[-4:]:
        print(f"    {o['arm']:<16}{o['bss']:>10.2f}   Δ {o['dbss']:>+8.2f}")

pd.DataFrame([{k: v for k, v in r.items() if k != "w"} for r in rows]).to_csv(
    OUT / "v46_decompose.csv", index=False)
pd.DataFrame(out).to_csv(OUT / "v46_transfer.csv", index=False)
print(f"{chr(10)}saved -> {OUT/'v46_decompose.csv'}, {OUT/'v46_transfer.csv'}")
