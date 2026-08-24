"""V78: 시즌 평균 이동으로 base 오프셋을 고칠 수 있는가.

질문
    V77 에서 fold 2023 base 의 음수(-140.20)가 전부 평균 오프셋(+0.0224)임을
    확인했다. 그러면 그 낙폭만큼 예측을 이동시키면 되지 않는가?

이 실험의 전부는 한 가지다
    이동폭을 '검증 시즌 라벨을 보지 않고' 정할 수 있는가.
    정할 수 있으면 제출에 쓸 수 있는 절차다. 없으면 진단값일 뿐이다.

시험하는 규칙 (전부 fold 이전 시즌만 사용)
    R0  무보정                     현행
    R1  직전 시즌 값               naive. 20/25 모델이 사실상 이걸 하고 있다
    R2  전 시즌 선형 외삽
    R3  최근 3시즌 선형 외삽
    R4  지수가중 추세 (감쇠 0.7)
    R5  차분의 중앙값만큼 하락
    R9  실제 시즌 평균             상한. 부정직 — 얼마나 손해보는지 재는 기준

이동 방식
    로짓 이동: sigmoid(logit(p) + c) 의 평균이 목표 평균이 되도록 c 를 푼다.
    확률 가산 이동은 [0,1] 을 벗어날 수 있어 쓰지 않는다.

주의
    RULES §2 — 리더보드 점수로 상수를 다시 맞추지 않는다. 여기의 모든 규칙은
    train 시즌만으로 정해진다.

실행: python v78_mean_shift.py     (CPU/IO 전용)
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
SEASONS = sorted(set(season.tolist()))
RATE = {s: float(y_all[season == s].mean()) for s in SEASONS}

print("=" * 92)
print("시즌별 실제 성공률 — 이동폭을 무엇으로 예측할 것인가")
print("=" * 92)
prev = None
for s in SEASONS:
    d = f"{RATE[s]-prev:+.4f}" if prev is not None else "   —   "
    print(f"  {s}   {RATE[s]:.4f}   전년대비 {d}")
    prev = RATE[s]


def rules(fold):
    """fold 이전 시즌만으로 fold 의 평균을 예측한다."""
    hist = [s for s in SEASONS if s < fold]
    v = np.array([RATE[s] for s in hist], float)
    x = np.array(hist, float)
    out = {"R1 직전 시즌": v[-1]}
    if len(v) >= 2:
        a, b = np.polyfit(x, v, 1)
        out["R2 전체 선형"] = a * fold + b
        k = min(3, len(v))
        a3, b3 = np.polyfit(x[-k:], v[-k:], 1)
        out[f"R3 최근{k} 선형"] = a3 * fold + b3
        d = np.diff(v)
        w = 0.7 ** np.arange(len(d) - 1, -1, -1)
        out["R4 지수가중 추세"] = v[-1] + float((d * w).sum() / w.sum())
        out["R5 차분 중앙값"] = v[-1] + float(np.median(d))
    return out


def shift_to(p, target):
    """평균이 target 이 되도록 로짓을 평행이동."""
    z = lgt(p)
    r = minimize_scalar(lambda c: (sig(z + c).mean() - target) ** 2,
                        bounds=(-4, 4), method="bounded")
    return np.clip(sig(z + r.x), EPS, 1 - EPS)


avail = {}
for p in sorted(OOF_DIR.glob("*.parquet")):
    m = re.match(r"(.+)_fold(\d{4})\.parquet", p.name)
    if m:
        avail.setdefault(int(m.group(2)), []).append(m.group(1))


def base_of(fold):
    fid = df.loc[season == fold, "row_id"].to_numpy()
    acc = None
    for mn in sorted(avail.get(fold, [])):
        v = (pd.read_parquet(OOF_DIR / f"{mn}_fold{fold}.parquet")
             .set_index("row_id").reindex(fid)["prediction"].to_numpy(np.float64))
        acc = v if acc is None else acc + v
    return np.clip(acc / len(avail[fold]), EPS, 1 - EPS)


def prod_of(fold):
    fid = df.loc[season == fold, "row_id"].to_numpy()
    pr = pd.read_parquet(PROD).set_index("row_id")
    c = "submit021_reverse20_s040_tabm"
    if c not in pr.columns:
        return None
    v = pr.reindex(fid)[c].to_numpy(np.float64)
    return None if np.isnan(v).all() else np.clip(v, EPS, 1 - EPS)


rows = []
for fold in sorted(avail):
    if fold == min(avail):
        continue
    y = y_all[season == fold]
    ybar = RATE[fold]
    cands = [("OOF 평균", base_of(fold))]
    pv = prod_of(fold)
    if pv is not None:
        cands.append(("프로덕션", pv))
    for tag, p in cands:
        m0 = metrics(y, p)
        print(f"{chr(10)}{'='*92}")
        print(f"fold {fold}  ·  {tag}   실제 평균 {ybar:.4f}   "
              f"현재 예측 평균 {p.mean():.4f}   오프셋 {p.mean()-ybar:+.4f}")
        print("=" * 92)
        print(f"  {'규칙':<18}{'예측 평균':>11}{'예측오차':>10}"
              f"{'이동 후 BSS':>13}{'ΔBSS':>10}")
        print(f"  {'R0 무보정':<18}{p.mean():>11.4f}"
              f"{p.mean()-ybar:>+10.4f}{m0['bss_raw']:>13.2f}{0.0:>+10.2f}")
        for name, tgt in rules(fold).items():
            ps = shift_to(p, tgt)
            b = metrics(y, ps)["bss_raw"]
            print(f"  {name:<18}{tgt:>11.4f}{tgt-ybar:>+10.4f}"
                  f"{b:>13.2f}{b-m0['bss_raw']:>+10.2f}")
            rows.append({"fold": fold, "base": tag, "rule": name,
                         "pred_mean": tgt, "err": tgt - ybar, "bss": b,
                         "dbss": b - m0["bss_raw"]})
        ps = shift_to(p, ybar)
        b = metrics(y, ps)["bss_raw"]
        print(f"  {'R9 실제값 (상한)':<18}{ybar:>11.4f}{0.0:>+10.4f}"
              f"{b:>13.2f}{b-m0['bss_raw']:>+10.2f}   <- 부정직. 도달 가능한 최대")
        rows.append({"fold": fold, "base": tag, "rule": "R9 실제값(상한)",
                     "pred_mean": ybar, "err": 0.0, "bss": b,
                     "dbss": b - m0["bss_raw"]})

res = pd.DataFrame(rows)
res.to_csv(OUT / "v78_mean_shift.csv", index=False)

print(f"{chr(10)}{'='*92}")
print("정직한 규칙만 놓고 fold 평균 ΔBSS")
print("=" * 92)
hon = res[~res.rule.str.startswith("R9")]
g = (hon.groupby("rule")
     .agg(fold수=("fold", "nunique"), 평균ΔBSS=("dbss", "mean"),
          최악ΔBSS=("dbss", "min"), 평균절대오차=("err", lambda s: s.abs().mean()))
     .sort_values("평균ΔBSS", ascending=False))
print(g.round(4).to_string())

print(f"{chr(10)}판단 기준")
print("  ㄱ 모든 fold 에서 ΔBSS 가 양수여야 한다. 한 곳이라도 음수면 도박이다.")
print("  ㄴ R9(상한) 대비 몇 %를 회수하는가. 회수율이 낮으면 예측이 안 되는 것이다.")
print(f"{chr(10)}saved -> {OUT / 'v78_mean_shift.csv'}")
