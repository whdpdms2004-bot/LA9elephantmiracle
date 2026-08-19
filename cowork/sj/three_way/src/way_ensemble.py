"""Stage W2: way 내 앙상블 — 한 way 안에서 서로 다른 구성을 섞는다.

way 간 앙상블(Stage 5)과 다르다
    Stage 5   1WAY 라인 x 3WAY x 프로덕션 base 를 섞는다
    여기      middle 안에서 서로 다른 middle 모델을 섞는다

시드 배깅이 아니다
    보류 중인 시드 배깅은 "같은 구성 다른 시드" 다.
    여기는 **다른 전처리 / 다른 학습 방식 / 다른 구조** 의 결합이다.

왜 되는가 — 1WAY V65 가 확인한 것
    결합 이득은 정확도가 아니라 비상관성이 정한다.
    way 내에서도 같다. 그래서 단독 성능과 **서로의 상관** 을 같이 본다.
    분할(split_ball)과 단일 모델은 구조가 달라 상관이 낮을 것으로 본다.

가중치
    검증 라벨로 가중치를 맞추면 그 fold 에 과적합한다.
    그래서 **fold 2023 에서 가중치를 정하고 fold 2024 에서 평가** 한다.
    단순 평균도 함께 보고한다 — 1WAY 에서 자유 최적화는 전부 전이되지 않았다.

사용
    python way_ensemble.py --target middle
    python way_ensemble.py --target middle --list      # 쓸 수 있는 예측 목록
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness3 import AUX_FOLD, DECISION_FOLD, OUT, TARGETS, bss, load_labeled, seed_noise

EPS = 1e-7
lgt = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
sig = lambda z: 1.0 / (1.0 + np.exp(-z))


def find_preds(target: str, fold: int) -> dict[str, np.ndarray]:
    """저장된 그 타깃 그 fold 의 예측을 전부 모은다."""
    pats = [
        (re.compile(rf"^{target}__(.+)__{fold}\.npy$"), "screen"),
        (re.compile(rf"^tr_{target}__(.+)__{fold}\.npy$"), "arm"),
        (re.compile(rf"^mf_{target}__(.+)__{fold}\.npy$"), "focus"),
        (re.compile(rf"^mn_{target}__(.+)__{fold}\.npy$"), "next"),
        (re.compile(rf"^s3_{target}__(.+)__{fold}\.npy$"), "layer"),
        (re.compile(rf"^ms_{target}__(.+)__{fold}\.npy$"), "sweep"),
        (re.compile(rf"^mh_{target}__(.+)__{fold}\.npy$"), "hier"),
        (re.compile(rf"^wb_{target}__(.+)__{fold}\.npy$"), "wbase"),
    ]
    out = {}
    for p in sorted(OUT.glob("*.npy")):
        for rx, kind in pats:
            m = rx.match(p.name)
            if m:
                out[f"{kind}:{m.group(1)}"] = np.load(p).astype(np.float64)
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="middle")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--top", type=int, default=8, help="단독 상위 몇 개를 후보로")
    ap.add_argument("--max-k", type=int, default=4, help="최대 몇 개까지 섞을지")
    ap.add_argument("--only", default="", help="이 문자열이 든 구성만 후보로")
    ap.add_argument("--rank-by", default="centered", choices=("raw", "centered"),
                    help="fold 2023 에서 후보를 무엇으로 줄세울지. "
                         "middle 은 f23 raw 순위가 오프셋 때문에 f24 와 반대로 간다 — "
                         "전이되는 것은 판별력이므로 centered 가 기본이다")
    args = ap.parse_args()

    tg = args.target
    df = load_labeled()
    season = df["season"].to_numpy()
    yv = pd.to_numeric(df[TARGETS[tg]], errors="coerce").to_numpy(np.float64)
    ok = (df["label_ok"].to_numpy() == 1) & ~np.isnan(yv)
    sd = seed_noise(tg)

    Y, P = {}, {}
    for f in (AUX_FOLD, DECISION_FOLD):
        Y[f] = yv[(season == f) & ok].astype(np.float64)
        P[f] = find_preds(tg, f)
        P[f] = {k: v for k, v in P[f].items() if len(v) == len(Y[f])}
        print(f"fold {f}: 예측 {len(P[f])}개  (검증 {len(Y[f]):,}행)")

    common = sorted(set(P[AUX_FOLD]) & set(P[DECISION_FOLD]))
    print(f"두 fold 모두 있는 구성: {len(common)}개")
    if args.list:
        for k in common:
            print(f"  {k:<52}"
                  f"f23 {bss(Y[AUX_FOLD], P[AUX_FOLD][k])['bss_raw']:>9.1f}  "
                  f"f24 {bss(Y[DECISION_FOLD], P[DECISION_FOLD][k])['bss_raw']:>9.1f}")
        return
    if len(common) < 2:
        print("섞을 게 부족하다."); return

    # 후보 추리기 — 성능만으로 고르면 비슷한 변형이 뭉친다.
    # 실제로 split_ball 변형만 9개 뽑혀 824 로 떨어진 반면,
    # 다양성을 넣은 bayesian+interact 는 887 이었다.
    # 그래서 **탐욕 다양성 선택**: 성능 1위부터 시작해, 이미 뽑힌 것들과
    # 상관이 낮은 순으로 채운다 (V65 — 결합 가치는 비상관성이 정한다).
    key = "bss_centered" if args.rank_by == "centered" else "bss_raw"
    rank = sorted(common, key=lambda k: -bss(Y[AUX_FOLD], P[AUX_FOLD][k])[key])
    print(f"후보 줄세우기 기준: fold {AUX_FOLD} {key}")
    if args.only:
        cand = [k for k in rank if any(o in k for o in args.only.split(","))]
        cand = cand[:args.top]
    else:
        pool = rank[:max(args.top * 4, 20)]
        Zp = {k: lgt(P[AUX_FOLD][k]) for k in pool}
        cand = [pool[0]]
        while len(cand) < min(args.top, len(pool)):
            best, best_c = None, 2.0
            for k in pool:
                if k in cand:
                    continue
                c = max(abs(float(np.corrcoef(Zp[k], Zp[c0])[0, 1])) for c0 in cand)
                if c < best_c:
                    best, best_c = k, c
            if best is None:
                break
            cand.append(best)
    print(f"{chr(10)}후보 {len(cand)}개 (fold {AUX_FOLD} 단독 기준으로 추림)")
    for k in cand:
        a = bss(Y[AUX_FOLD], P[AUX_FOLD][k])
        b = bss(Y[DECISION_FOLD], P[DECISION_FOLD][k])
        print(f"  {k:<52}f23 {a['bss_raw']:>8.1f} (cen {a['bss_centered']:>7.1f})"
              f"   f24 {b['bss_raw']:>8.1f}")

    print(f"{chr(10)}{'=' * 96}")
    print("상관 행렬 (fold 2023 로짓) — 낮을수록 결합 가치가 있다")
    print("=" * 96)
    Z = np.column_stack([lgt(P[AUX_FOLD][k]) for k in cand])
    C = np.corrcoef(Z.T)
    print("      " + "".join(f"{i:>7}" for i in range(len(cand))))
    for i, k in enumerate(cand):
        print(f"  {i:<4}" + "".join(f"{C[i, j]:>7.3f}" for j in range(len(cand)))
              + f"   {k[:40]}")

    print(f"{chr(10)}{'=' * 96}")
    print(f"조합 — 가중치는 fold {AUX_FOLD} 에서 정하고 fold {DECISION_FOLD} 에서 평가")
    print("=" * 96)
    base24 = max(bss(Y[DECISION_FOLD], P[DECISION_FOLD][k])["bss_raw"] for k in cand)
    print(f"  단독 최고 (f24) {base24:.1f}")
    print(f"  {'조합':<54}{'f23':>10}{'f24':>10}{'Δ vs 단독최고':>14}")

    rows = []
    for k in range(2, min(args.max_k, len(cand)) + 1):
        for combo in itertools.combinations(cand, k):
            z23 = np.column_stack([lgt(P[AUX_FOLD][c]) for c in combo])
            z24 = np.column_stack([lgt(P[DECISION_FOLD][c]) for c in combo])
            # (a) 단순 평균
            for tag, w in (("mean", np.full(k, 1.0 / k)),):
                p23 = sig(z23 @ w)
                p24 = sig(z24 @ w)
                rows.append({"combo": "+".join(c.split(":")[-1][:16] for c in combo),
                             "k": k, "w": tag,
                             "f23": bss(Y[AUX_FOLD], p23)["bss_raw"],
                             "f24": bss(Y[DECISION_FOLD], p24)["bss_raw"]})
            # (b) fold 2023 에서 로지스틱으로 가중치 적합
            try:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(max_iter=400, C=1.0).fit(z23, Y[AUX_FOLD])
                p23 = lr.predict_proba(z23)[:, 1]
                p24 = lr.predict_proba(z24)[:, 1]
                rows.append({"combo": "+".join(c.split(":")[-1][:16] for c in combo),
                             "k": k, "w": "lr23",
                             "f23": bss(Y[AUX_FOLD], p23)["bss_raw"],
                             "f24": bss(Y[DECISION_FOLD], p24)["bss_raw"]})
            except Exception:                                     # noqa: BLE001
                pass

    t = pd.DataFrame(rows)
    t["d"] = t["f24"] - base24
    t = t.sort_values("f24", ascending=False)
    t.to_csv(OUT / f"way_ensemble_{tg}.csv", index=False)
    for r in t.head(12).itertuples():
        print(f"  {r.combo[:48]:<48}{r.w:<6}{r.f23:>10.1f}{r.f24:>10.1f}{r.d:>+14.1f}")

    best = t.iloc[0]
    print(f"{chr(10)}  최고 {best.f24:.1f} ({best.w})   단독 최고 대비 {best.d:+.1f}")
    print(f"  목표 1300 까지 {1300 - best.f24:+.0f}")
    print(f"{chr(10)}  주의: 단순 평균(mean)이 lr23 보다 나으면 가중치 적합이 전이되지 않는 것이다.")
    print(f"        1WAY V44~V48 에서 자유 최적화는 전부 다른 fold 에서 무너졌다.")
    print(f"{chr(10)}saved -> {OUT / f'way_ensemble_{tg}.csv'}")


if __name__ == "__main__":
    main()
