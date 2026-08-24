"""V44: 선택 기준을 다 내려놓고 조합을 넓게 탐색한다. (CPU 전용)

지시
    "상관계수 신경쓰지 말고 그냥 성능 높을 것 같은 조합으로 가자.
     지금 변수 선택 기준 다 버리고 여러 조합 시험해보고 결과로만 평가해."

    그대로 한다. 세 fold 규칙도, 단독 BSS 동반 상승 조건도, 상관 기반 판단도
    적용하지 않는다. Val2024 결과로만 줄 세운다.

재학습이 필요 없다
    V33/V35/V37/V38/V40/V43 에서 학습한 성분 라인 예측이 2024 fold 에 대해
    전부 캐시돼 있다. 기각했던 것들도 남아 있다. 그것들을 '대체재'가 아니라
    '앙상블 멤버'로 쓰면 완전히 다른 질문이 된다 — 단독으로 진 라인이
    섞이면 이길 수 있다.

    base 쪽도 npz 로 재구성 가능하다.
        p(a, s) = p021 + (a-1)*r_correction + (s-0.40)*0.6085*reverse20
    a(r_context 스케일)와 s(reverse 스케일)를 격자로 넣는다.
    hw 의 val2024_pred.csv 도 후보에 넣는다.

방법
    Caruana 탐욕 앙상블 선택(중복 허용). 최종 예측 공간에서 직접 고른다 —
    base 자체도 멤버라서 몇 번 뽑히느냐가 곧 가중치다. w 를 따로 정할 필요가 없다.

    정직한 측정을 위해 2024 를 무작위 반으로 나눈다.
        fit 반쪽에서 고르고, held-out 반쪽에서 잰다.
    이건 게이트가 아니다. 고른 조합이 2024 잡음에 맞춘 것인지 아닌지를
    보여주는 숫자일 뿐이고, 채택 여부는 사용자가 결과를 보고 정한다.

    확률 평균과 로짓 평균 둘 다 낸다. 마지막에 절편 보정도 fit 반쪽에서 적합한다.

출력: outputs/v44_combo_search.csv, outputs/v44_members.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load, metrics

CW = Path(__file__).resolve().parents[3]
MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
NPZ = PROD.parent / "reverse20_submission_components.npz"
N_ITER = 60
SEED = 20260815
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

M, names = [], []


def add(name, v):
    v = np.asarray(v, np.float64)
    if v.shape != (n,) or not np.isfinite(v).all():
        return
    M.append(np.clip(v, EPS, 1 - EPS))
    names.append(name)


# base 계열
for a in [0.75, 1.00, 1.25]:
    for s in [0.30, 0.40, 0.475, 0.55, 0.70]:
        add(f"base_a{a:.2f}_s{s:.3f}", p021 + (a - 1.0) * rc + (s - 0.40) * 0.6085 * rev)
for col in ["submit017_reconstructed", "submit019_reconstructed",
            "submit020_reverse20_s055_tabm"]:
    if col in prod.columns:
        add(f"prod_{col[:12]}", prod[col].to_numpy(np.float64))

# 성분 라인 — 기각한 것 포함 전부
for f in sorted(CACHE.glob("*.npy")):
    v = np.load(f)
    if v.ndim == 1 and v.shape[0] == n:
        add(f.stem.replace("_current", "").replace("_p_ie", ""), v)

# 팀
hw = CW / "hw" / "val2024_pred.csv"
if hw.exists():
    s_ = pd.read_csv(hw).set_index("row_id").reindex(ids)["control_success"]
    add("hw", s_.to_numpy(np.float64))

P = np.vstack(M)
print(f"후보 {len(names)}개 x {n:,}행{chr(10)}")
solo = np.array([metrics(y, p)["bss_raw"] for p in P])
o = np.argsort(-solo)
print(f"{'단독 BSS 상위':<28}{'':<6}{'하위':<28}")
for i in range(12):
    hi, lo = o[i], o[-1 - i]
    print(f"  {names[hi]:<26}{solo[hi]:>8.2f}   {names[lo]:<26}{solo[lo]:>8.2f}")
pd.DataFrame({"member": names, "solo_bss": solo}).sort_values(
    "solo_bss", ascending=False).to_csv(OUT / "v44_members.csv", index=False)

rng = np.random.default_rng(SEED)
fit = rng.random(n) < 0.5
hold = ~fit
print(f"{chr(10)}fit {fit.sum():,}행 / held-out {hold.sum():,}행")


def bss(mask, p):
    yy, pp = y[mask], np.clip(p[mask], EPS, 1 - EPS)
    null = yy.mean() * (1 - yy.mean())
    return 100000 * (null - ((pp - yy) ** 2).mean()) / null


def greedy(space):
    """Caruana 탐욕 선택(중복 허용). space in {prob, logit}."""
    X = P if space == "prob" else np.log(P / (1 - P))
    acc = np.zeros(n)
    picks, trail = [], []
    for it in range(1, N_ITER + 1):
        cand = (acc[None, :] + X) / it
        q = cand if space == "prob" else 1 / (1 + np.exp(-cand))
        yy = y[fit]
        null = yy.mean() * (1 - yy.mean())
        err = ((np.clip(q[:, fit], EPS, 1 - EPS) - yy) ** 2).mean(axis=1)
        k = int(np.argmin(err))
        acc += X[k]
        picks.append(names[k])
        cur = acc / it
        qq = cur if space == "prob" else 1 / (1 + np.exp(-cur))
        trail.append((it, names[k], bss(fit, qq), bss(hold, qq), bss(slice(None), qq)))
    return picks, trail, qq


rows = []
best = {}
for space in ["prob", "logit"]:
    picks, trail, final = greedy(space)
    print(f"{chr(10)}{'='*84}{chr(10)}{space} 공간 탐욕 선택{chr(10)}{'='*84}")
    print(f"  {'it':>3} {'추가된 멤버':<26}{'fit':>10}{'held-out':>11}{'전체':>10}")
    for it, nm, a, b_, c in trail:
        if it <= 12 or it % 10 == 0 or it == N_ITER:
            print(f"  {it:>3} {nm:<26}{a:>10.2f}{b_:>11.2f}{c:>10.2f}", flush=True)
    cnt = pd.Series(picks).value_counts()
    print(f"{chr(10)}  최종 구성 ({len(cnt)}종)")
    for nm, c in cnt.items():
        print(f"    {nm:<28}{c/N_ITER*100:>6.1f}%")
    for it, nm, a, b_, c in trail:
        rows.append({"space": space, "iter": it, "added": nm, "fit_bss": a,
                     "hold_bss": b_, "full_bss": c})
    best[space] = final

ref_full, ref_hold = bss(slice(None), p021), bss(hold, p021)
print(f"{chr(10)}{'='*84}{chr(10)}절편 보정 (fit 반쪽에서 적합){chr(10)}{'='*84}")
for space, q in best.items():
    c0 = float(np.mean(y[fit] - q[fit]))
    qc = np.clip(q + c0, EPS, 1 - EPS)
    print(f"  {space:<6} c0 {c0:+.5f}   전체 {bss(slice(None), qc):9.2f} "
          f"(보정 전 {bss(slice(None), q):9.2f})   held-out {bss(hold, qc):9.2f}")
    rows.append({"space": space + "_c0", "iter": N_ITER, "added": f"c0={c0:+.5f}",
                 "fit_bss": bss(fit, qc), "hold_bss": bss(hold, qc),
                 "full_bss": bss(slice(None), qc)})
    np.save(CACHE / f"v44_best_{space}.npy", qc)

pd.DataFrame(rows).to_csv(OUT / "v44_combo_search.csv", index=False)
print(f"{chr(10)}{'='*84}")
print(f"기준 submit_021          전체 {ref_full:9.2f}   held-out {ref_hold:9.2f}")
print(f"기준 submit_031 (구간w)  전체 {bss(slice(None), np.clip(0.25*np.load(CACHE/'v25_p_ie_029.npy')+0.75*p021, EPS, 1-EPS)):9.2f}  (근사)")
for space, q in best.items():
    print(f"{space:<8} 조합                전체 {bss(slice(None), q):9.2f}   "
          f"held-out {bss(hold, q):9.2f}   차이 {bss(hold,q)-ref_hold:+8.2f}")
print(f"{chr(10)}held-out 이 전체와 비슷하면 조합이 2024 잡음에 맞춘 게 아니다.")
print(f"{chr(10)}saved -> {OUT/'v44_combo_search.csv'}, {OUT/'v44_members.csv'}")
