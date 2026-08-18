"""V45: 조합을 '실을 수 있는 크기'로 제약해 다시 고른다 — 해석적 해. (CPU 전용)

V44 결과
    탐욕 앙상블 Val2024 890.27 (submit_021 836.50, submit_031 근사 875.58)
    held-out 반쪽 리프트가 fit 반쪽보다 크다(+63.9 vs +43.5). 잡음 적합이 아니다.
    기각했던 라인들이 멤버로는 뽑힌다 — P4_matchup(단독 259)이 8.3%.

문제 — 추론 시간
    성분 라인 하나가 모델 약 80~112개, 로컬 30초, 6 vCPU 환산 120초.
    600초 한도이므로 실을 수 있는 라인은 최대 4종이다.
    base 변형(a, s)은 프로덕션 metadata 상수라 공짜다.

푸는 방법
    Brier = mean((Xw − y)^2) 는 w 에 대해 이차식이다. 격자로 훑을 게 아니라
    최소제곱으로 정확히 푼다. X 는 253,507 x 39 이고 그람 행렬은 39x39 이라
    즉시 풀린다. 음수 가중치를 막으려면 NNLS 를 쓴다.

    K 제약은 orthogonal matching pursuit — 한 종씩 늘리면서 매번 NNLS 로
    전체 가중치를 다시 푼다.

    정직한 측정: 무작위 절반 분할 20회. fit 반쪽에서 가중치를 풀고 held-out
    반쪽에서 잰다. 중앙값 리프트로 줄 세운다.

출력: outputs/v45_constrained_combo.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
NPZ = PROD.parent / "reverse20_submission_components.npz"
ALPHAS = [1.00, 1.25, 1.50]
SCALES = [0.10, 0.30, 0.475]
KMAX = 6
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

BASE, LINE = {}, {}
for a in ALPHAS:
    for s in SCALES:
        BASE[f"base_a{a:.2f}_s{s:.3f}"] = np.clip(
            p021 + (a - 1.0) * rc + (s - 0.40) * 0.6085 * rev, EPS, 1 - EPS)
for f in sorted(CACHE.glob("*.npy")):
    if f.stem.startswith(("v44_", "v45_")):
        continue
    v = np.load(f)
    if v.ndim == 1 and v.shape[0] == n and np.isfinite(v).all():
        LINE[f.stem.replace("_current", "")] = np.clip(v, EPS, 1 - EPS)
bkeys, lkeys = sorted(BASE), sorted(LINE)
print(f"base 변형 {len(bkeys)}종 (공짜)   성분 라인 {len(lkeys)}종 (각 120초)")


def bss(mask, p):
    yy, pp = y[mask], np.clip(p[mask], EPS, 1 - EPS)
    nl = yy.mean() * (1 - yy.mean())
    return 100000 * (nl - ((pp - yy) ** 2).mean()) / nl


def solve(cols, mask):
    """절편 포함 NNLS. 절편은 부호 제약을 피하려 +1/-1 두 열로 넣는다."""
    X = np.column_stack([c[mask] for c in cols]
                        + [np.ones(mask.sum()), -np.ones(mask.sum())])
    w, _ = nnls(X, y[mask])
    return w


def apply(cols, w):
    q = np.zeros(n)
    for c, wi in zip(cols, w[:len(cols)]):
        q += wi * c
    return np.clip(q + w[-2] - w[-1], EPS, 1 - EPS)


splits = [np.random.default_rng(1000 + i).random(n) < 0.5 for i in range(N_SPLIT)]
ALL = np.ones(n, bool)


def score(cols):
    """전체 적합 점수와, 20회 분할의 held-out 중앙 리프트."""
    q = apply(cols, solve(cols, ALL))
    lifts = []
    for sp in splits:
        qs = apply(cols, solve(cols, sp))
        lifts.append(bss(~sp, qs) - bss(~sp, p021))
    return bss(ALL, q), float(np.median(lifts)), q


# ---------------------------------------------- base 만
print(f"{chr(10)}{'='*88}{chr(10)}1) base 변형만 (성분 라인 0종, 추론 0초 추가)"
      f"{chr(10)}{'='*88}")
full_b, lift_b, q_b = score([BASE[k] for k in bkeys])
print(f"  base {len(bkeys)}종 NNLS   전체 {full_b:8.2f}   held-out 중앙 리프트 {lift_b:+7.2f}")
print(f"  현행 submit_021          {bss(ALL, p021):8.2f}")

# ---------------------------------------------- 상한: 전부
print(f"{chr(10)}{'='*88}{chr(10)}2) 상한 — 성분 라인 전부 ({len(lkeys)}종, 실을 수 없음)"
      f"{chr(10)}{'='*88}")
allcols = [BASE[k] for k in bkeys] + [LINE[k] for k in lkeys]
full_a, lift_a, _ = score(allcols)
print(f"  전체 {full_a:8.2f}   held-out 중앙 리프트 {lift_a:+7.2f}")

# ---------------------------------------------- K 제약 OMP
print(f"{chr(10)}{'='*88}{chr(10)}3) K 제약 (성분 라인 K 종 + base 변형 전부)"
      f"{chr(10)}{'='*88}")
print(f"  {'K':>2} {'추가된 라인':<24}{'전체':>10}{'held-out 리프트':>16}"
      f"{'추론 추정':>11}")
rows, chosen = [], []
for K in range(1, KMAX + 1):
    best = None
    for k in lkeys:
        if k in chosen:
            continue
        cols = [BASE[b] for b in bkeys] + [LINE[c] for c in chosen + [k]]
        full, lift, q = score(cols)
        if best is None or full > best[1]:
            best = (k, full, lift, q)
    k, full, lift, q = best
    chosen.append(k)
    np.save(CACHE / f"v45_K{K}.npy", q)
    rows.append({"K": K, "added": k, "lines": "+".join(chosen), "full_bss": full,
                 "hold_lift": lift, "infer_sec_est": 120 * K})
    flag = "" if 120 * K <= 480 else "  <- 한도 초과"
    print(f"  {K:>2} {k:<24}{full:>10.2f}{lift:>16.2f}{f'{120*K}초':>11}{flag}",
          flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v45_constrained_combo.csv", index=False)

cur = bss(ALL, np.clip(0.25 * LINE["v25_029"] + 0.75 * p021, EPS, 1 - EPS))
print(f"{chr(10)}{'='*88}")
print(f"  {'submit_021 (base)':<34}{bss(ALL, p021):>10.2f}")
print(f"  {'submit_031 근사 (1종, w0.25)':<34}{cur:>10.2f}")
for _, r in res.iterrows():
    print(f"  K={r.K} {r.added[:28]:<30}{r.full_bss:>10.2f}"
          f"   031 대비 {r.full_bss-cur:+7.2f}   {r.infer_sec_est}초")
print(f"{chr(10)}  전부({len(lkeys)}종) 상한          {full_a:>10.2f}")
print(f"{chr(10)}saved -> {OUT/'v45_constrained_combo.csv'}")
