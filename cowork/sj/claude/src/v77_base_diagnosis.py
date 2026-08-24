"""V77: fold 2023 base 가 왜 음수인가. 그리고 고칠 수 있는가.

배경
    harness.base_pred 는 fold 에 따라 완전히 다른 것을 준다.
        fold 2024 -> 프로덕션 제출 OOF 한 개 (submit021_reverse20_s040_tabm)
        fold 2023 -> enhanced_seed_oof_parts 에 파일이 있는 모델의 단순 평균
    2023 base 가 -140.20 이면 그 위에서 잰 ΔBSS 는 '깨진 기준선을 얼마나
    구조했나' 를 재게 된다. 2024 의 ΔBSS 와 같은 축에 놓을 수 없다.

이 스크립트가 답하는 것
    1. 2023 평균에 실제로 몇 개 모델이 들어갔나. 2024 와 같은 구성인가.
    2. 음수의 원인이 평균 오프셋인가, 분해능 부족인가.
       BSS 를 신뢰도(reliability) / 분해능(resolution) 으로 쪼개 본다.
    3. 재중심화하면 얼마나 회복되나. 세 가지를 시험한다.
         A 가산 이동      p + (ybar - pbar)
         B 로짓 이동      sigmoid(logit(p) + c),  c 는 평균을 맞추는 값
         C 로짓 선형보정  sigmoid(a*logit(p) + b),  a,b 는 이전 시즌에서 적합
       C 의 a,b 는 반드시 fold 이전 시즌(<2023)에서만 적합한다. 검증 시즌을
       보고 맞추면 그건 측정이 아니라 커닝이다.
    4. 개별 모델을 하나씩 재면 특정 모델이 평균을 망치고 있는가.

실행: python v77_base_diagnosis.py     (CPU/IO 전용, GPU 작업과 겹쳐도 안전)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, load, metrics

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
EPS = 1e-7

lgt = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
sig = lambda z: 1.0 / (1.0 + np.exp(-z))

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)

print("=" * 88)
print("1. fold 별로 base 에 무엇이 들어가는가")
print("=" * 88)

avail = {}
for p in sorted(OOF_DIR.glob("*.parquet")):
    m = re.match(r"(.+)_fold(\d{4})\.parquet", p.name)
    if m:
        avail.setdefault(int(m.group(2)), []).append(m.group(1))

for fold in sorted(avail):
    print(f"  fold {fold}   OOF 파트 {len(avail[fold]):>2}개   "
          f"{', '.join(sorted(avail[fold])[:4])}"
          f"{' ...' if len(avail[fold]) > 4 else ''}")
prod_ok = PROD.exists()
print(f"  프로덕션 OOF 파일 존재: {prod_ok}   ({PROD.name})")
if prod_ok:
    pc = [c for c in pd.read_parquet(PROD).columns if c != "row_id"]
    print(f"    열 {len(pc)}개: {', '.join(pc[:5])}{' ...' if len(pc) > 5 else ''}")

common = set.intersection(*(set(v) for v in avail.values())) if len(avail) > 1 else set()
print(f"{chr(10)}  모든 fold 에 공통으로 있는 모델: {len(common)}개")
if common:
    print(f"    {', '.join(sorted(common))}")


def base_of(fold, names=None):
    fid = df.loc[season == fold, "row_id"].to_numpy()
    acc, used = None, []
    for mn in sorted(avail.get(fold, [])):
        if names is not None and mn not in names:
            continue
        v = (pd.read_parquet(OOF_DIR / f"{mn}_fold{fold}.parquet")
             .set_index("row_id").reindex(fid)["prediction"].to_numpy(np.float64))
        acc = v if acc is None else acc + v
        used.append(mn)
    return (np.clip(acc / len(used), EPS, 1 - EPS) if used else None), used, fid


def prod_of(fold):
    fid = df.loc[season == fold, "row_id"].to_numpy()
    pr = pd.read_parquet(PROD).set_index("row_id")
    col = "submit021_reverse20_s040_tabm"
    if col not in pr.columns:
        return None
    v = pr.reindex(fid)[col].to_numpy(np.float64)
    return None if np.isnan(v).all() else np.clip(v, EPS, 1 - EPS)


print(f"{chr(10)}{'='*88}")
print("2. 음수의 원인 — 평균 오프셋인가 분해능인가")
print("=" * 88)
print(f"  {'fold':>5}{'구성':>12}{'n':>9}{'ybar':>9}{'pbar':>9}{'오프셋':>10}"
      f"{'BSS':>10}{'오프셋손실':>12}{'분해능잔여':>12}")

rows = []
for fold in sorted(avail):
    va = season == fold
    y = y_all[va]
    ybar, null = y.mean(), y.mean() * (1 - y.mean())
    cands = [("OOF 평균", base_of(fold)[0])]
    pv = prod_of(fold)
    if pv is not None:
        cands.append(("프로덕션", pv))
    for tag, p in cands:
        if p is None or np.isnan(p).all():
            continue
        bss = metrics(y, p)["bss_raw"]
        off = p.mean() - ybar
        # 오프셋만으로 잃는 BSS: 예측을 y 평균으로 이동했을 때의 회복분
        p_shift = np.clip(p - off, EPS, 1 - EPS)
        bss_shift = metrics(y, p_shift)["bss_raw"]
        rows.append((fold, tag, p, y, bss, bss_shift))
        print(f"  {fold:>5}{tag:>12}{len(y):>9,}{ybar:>9.4f}{p.mean():>9.4f}"
              f"{off:>+10.4f}{bss:>10.2f}{bss_shift-bss:>+12.2f}{bss_shift:>12.2f}")

print(f"{chr(10)}  오프셋손실 = 예측 평균을 실제 평균에 맞췄을 때 회복되는 BSS")
print(f"  분해능잔여 = 오프셋을 완전히 제거하고도 남는 BSS (모델의 실제 실력)")

print(f"{chr(10)}{'='*88}")
print("3. 재중심화로 얼마나 회복되나  (C 는 이전 시즌에서만 적합)")
print("=" * 88)
print(f"  {'fold':>5}{'구성':>12}{'원본':>10}{'A 가산':>10}{'B 로짓':>10}"
      f"{'C 선형(정직)':>14}{'C 계수':>18}")

for fold, tag, p, y, bss, _ in rows:
    ybar = y.mean()
    pA = np.clip(p - (p.mean() - ybar), EPS, 1 - EPS)
    z = lgt(p)
    r = minimize_scalar(lambda c: (sig(z + c).mean() - ybar) ** 2,
                        bounds=(-3, 3), method="bounded")
    pB = np.clip(sig(z + r.x), EPS, 1 - EPS)

    # C: 직전 시즌들에서 a,b 를 적합. 검증 시즌은 절대 보지 않는다.
    prev = [f for f in sorted(avail) if f < fold]
    pC, coef = None, "이전 fold 없음"
    if prev:
        zs, ys = [], []
        for f2 in prev:
            b2 = base_of(f2)[0] if tag == "OOF 평균" else prod_of(f2)
            if b2 is None or np.isnan(b2).all():
                continue
            zs.append(lgt(b2))
            ys.append(y_all[season == f2])
        if zs:
            zz, yy = np.concatenate(zs), np.concatenate(ys)
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=200, C=1e6).fit(zz.reshape(-1, 1), yy)
            a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
            pC = np.clip(sig(a * z + b), EPS, 1 - EPS)
            coef = f"a={a:.3f} b={b:+.3f}"
    print(f"  {fold:>5}{tag:>12}{bss:>10.2f}"
          f"{metrics(y, pA)['bss_raw']:>10.2f}"
          f"{metrics(y, pB)['bss_raw']:>10.2f}"
          f"{(metrics(y, pC)['bss_raw'] if pC is not None else float('nan')):>14.2f}"
          f"{coef:>18}")

print(f"{chr(10)}  A, B 는 검증 시즌 평균을 보고 맞춘 것이라 '가능한 최대치' 참고값이다.")
print(f"  C 만이 실제로 쓸 수 있는 보정이다. V36 에서 사후 보정은 이미 기각됐으므로")
print(f"  (한 해의 확률 사상이 다른 해에 틀림, 최악 -162) C 가 나빠도 놀랄 일이 아니다.")

print(f"{chr(10)}{'='*88}")
print("4. 평균을 망치는 개별 모델이 있는가  (fold 별 단독 BSS)")
print("=" * 88)
for fold in sorted(avail):
    va = season == fold
    y = y_all[va]
    fid = df.loc[va, "row_id"].to_numpy()
    print(f"{chr(10)}  fold {fold}   y평균 {y.mean():.4f}")
    print(f"    {'모델':<34}{'pbar':>9}{'오프셋':>10}{'단독 BSS':>12}")
    solos = []
    for mn in sorted(avail[fold]):
        v = (pd.read_parquet(OOF_DIR / f"{mn}_fold{fold}.parquet")
             .set_index("row_id").reindex(fid)["prediction"].to_numpy(np.float64))
        v = np.clip(v, EPS, 1 - EPS)
        s = metrics(y, v)["bss_raw"]
        solos.append((mn, s))
        print(f"    {mn:<34}{v.mean():>9.4f}{v.mean()-y.mean():>+10.4f}{s:>12.2f}")
    if len(solos) > 1:
        worst = min(solos, key=lambda t: t[1])
        keep = {m for m, _ in solos if m != worst[0]}
        b2, _, _ = base_of(fold, keep)
        print(f"    → 최악 1개({worst[0]}) 제외 시 평균 BSS "
              f"{metrics(y, b2)['bss_raw']:.2f}")

print(f"{chr(10)}{'='*88}")
print("결론이 향하는 곳")
print("=" * 88)
print("  2023 base 가 오프셋 때문에 음수라면, 그 위에서 잰 ΔBSS 는 다른 축의 값이다.")
print("  두 fold 를 나란히 쓰려면 base 구성을 통일하거나, ΔBSS 대신")
print("  '오프셋 제거 후 BSS' 를 비교 기준으로 삼아야 한다.")
