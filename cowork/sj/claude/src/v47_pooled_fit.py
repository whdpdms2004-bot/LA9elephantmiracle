"""V47: 가중치를 2024 만 보고 풀지 말고 세 fold 를 다 보여주고 푼다. (CPU 전용)

V46 이 밝힌 것
    이득의 대부분은 라인 교체가 아니라 '가중치를 데이터로 푼 것'이다(+11.11).
    뽑힌 해는 base 0.670 + 성분 0.414 − 0.0438 로 합이 1.084 다.
    수축이 아니라 '확대'다 — 로짓 폭을 넓히는 방향.

    그 해를 다른 fold 에 적용하면
        2022 +67.62   2024 +53.43   2023 −65.37
    2023 에서만 무너진다. V34(수축)와 정확히 반대편의 같은 트레이드오프다.

    2024 한 fold 만 보고 풀었으니 2023 을 못 본 것이다. 규칙으로 막을 게 아니라
    데이터를 더 보여주고 다시 풀면 된다.

arm
    G  볼록 제약 (합=1, 절편 없음), 2024 적합    <- '확대'를 뺀 순수 w
    H  자유 아핀, 2024 적합                      <- V46 D (재게시)
    I  볼록 제약, 세 fold 풀링 적합
    J  자유 아핀, 세 fold 풀링 적합
    K  볼록 제약, 세 fold 최악값 최대화(minimax)  <- 가장 나쁜 해를 가장 좋게

    풀링은 fold 를 행 수로 균등 가중한다(2023 이 묻히지 않게 fold 당 동일 가중).

주의 — base 시스템이 fold 마다 다르다
    2024 는 프로덕션 submit_021, 2022/2023 은 enhanced 25종 평균이다.
    절대 수준은 비교 불가지만 'base 대비 성분 비중'은 비교할 만하다.
    이 비대칭은 결과 해석에 그대로 남겨둔다.

출력: outputs/v47_pooled_fit.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
FOLDS = [2022, 2023, 2024]
EPS = 1e-7

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)

models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
B, L, Y = {}, {}, {}
for fold in FOLDS:
    fid = df.loc[season == fold, "row_id"].to_numpy()
    if fold == 2024:
        prod = pd.read_parquet(PROD).set_index("row_id").reindex(fid)
        B[fold] = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                          EPS, 1 - EPS)
    else:
        acc, cnt = None, 0
        for mn in models:
            f = OOF_DIR / f"{mn}_fold{fold}.parquet"
            if f.exists():
                v = pd.read_parquet(f).set_index("row_id").reindex(fid)["prediction"].to_numpy()
                acc = v if acc is None else acc + v
                cnt += 1
        B[fold] = np.clip(acc / cnt, EPS, 1 - EPS)
    L[fold] = np.clip(np.load(CACHE / f"v38_R1_msplit_{fold}.npy"), EPS, 1 - EPS)
    Y[fold] = y_all[season == fold]


def sc(fold, p):
    yy = Y[fold]
    nl = yy.mean() * (1 - yy.mean())
    p = np.clip(p, EPS, 1 - EPS)
    return 100000 * (nl - ((p - yy) ** 2).mean()) / nl


REF = {f: sc(f, B[f]) for f in FOLDS}
print("fold 별 base")
for f in FOLDS:
    print(f"  {f}  {REF[f]:10.2f}   성분단독 {sc(f, L[f]):10.2f}")


def pred(fold, th):
    wb, wl, c = th
    return np.clip(wb * B[fold] + wl * L[fold] + c, EPS, 1 - EPS)


def brier(fold, th):
    return float(((pred(fold, th) - Y[fold]) ** 2).mean())


def fit_affine(folds):
    Xs = [np.column_stack([B[f], L[f], np.ones(len(Y[f])), -np.ones(len(Y[f]))])
          for f in folds]
    ws = [1.0 / np.sqrt(len(Y[f]) * len(folds)) for f in folds]
    X = np.vstack([x * w for x, w in zip(Xs, ws)])
    t = np.concatenate([Y[f] * w for f, w in zip(folds, ws)])
    w, _ = nnls(X, t)
    return np.array([w[0], w[1], w[2] - w[3]])


def fit_convex(folds):
    def obj(u):
        wl = 1 / (1 + np.exp(-u[0]))
        return float(np.mean([brier(f, (1 - wl, wl, 0.0)) for f in folds]))
    r = minimize(obj, [-1.0], method="Nelder-Mead",
                 options={"xatol": 1e-5, "fatol": 1e-12})
    wl = 1 / (1 + np.exp(-r.x[0]))
    return np.array([1 - wl, wl, 0.0])


def fit_minimax(folds):
    def obj(u):
        wl = 1 / (1 + np.exp(-u[0]))
        return -min(sc(f, pred(f, (1 - wl, wl, 0.0))) - REF[f] for f in folds)
    r = minimize(obj, [-1.0], method="Nelder-Mead",
                 options={"xatol": 1e-5, "fatol": 1e-9})
    wl = 1 / (1 + np.exp(-r.x[0]))
    return np.array([1 - wl, wl, 0.0])


ARMS = [
    ("G 볼록 2024", fit_convex([2024])),
    ("H 아핀 2024", fit_affine([2024])),
    ("I 볼록 3fold", fit_convex(FOLDS)),
    ("J 아핀 3fold", fit_affine(FOLDS)),
    ("K 볼록 minimax", fit_minimax(FOLDS)),
    ("현행 w0.25", np.array([0.75, 0.25, 0.0])),
]

rows = []
print(f"{chr(10)}{'='*84}")
print(f"{'arm':<18}{'w_base':>8}{'w_성분':>8}{'절편':>10}"
      + "".join(f"{f'Δ{f}':>11}" for f in FOLDS) + f"{'최악':>10}")
print("=" * 84)
for name, th in ARMS:
    ds = [sc(f, pred(f, th)) - REF[f] for f in FOLDS]
    rows.append({"arm": name, "w_base": th[0], "w_line": th[1], "intercept": th[2],
                 **{f"d{f}": d for f, d in zip(FOLDS, ds)}, "worst": min(ds)})
    print(f"{name:<18}{th[0]:>8.3f}{th[1]:>8.3f}{th[2]:>+10.5f}"
          + "".join(f"{d:>+11.2f}" for d in ds) + f"{min(ds):>+10.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v47_pooled_fit.csv", index=False)
best_sum = res.assign(s=res[[f"d{f}" for f in FOLDS]].sum(axis=1)).sort_values(
    "s", ascending=False).iloc[0]
best_worst = res.sort_values("worst", ascending=False).iloc[0]
print(f"{chr(10)}세 fold 합 최고   {best_sum.arm}   합 "
      f"{best_sum[[f'd{f}' for f in FOLDS]].sum():+.2f}")
print(f"최악 fold 최고     {best_worst.arm}   최악 {best_worst.worst:+.2f}   "
      f"w_성분 {best_worst.w_line:.3f}")
print(f"{chr(10)}saved -> {OUT/'v47_pooled_fit.csv'}")
