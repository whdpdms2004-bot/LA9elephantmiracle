"""way 별 확률 최적화 — 정직한 예측만 모아 시드 배깅 + 탐욕 앙상블.

결합(세 확률 -> 최종 제구 성공)은 다음 단계다. 여기서는 way 별 확률만 본다.

원칙 (전부 실측 근거)
    · 정직한 예측만 쓴다. 채점 fold 를 조기 종료에 쓴 예측(tr_/screen/mf_/s3_)은
      제외한다 — audit_earlystop.py 참고.
    · 같은 설정의 여러 시드는 로짓 평균으로 먼저 합친다. middle f24 는 시드
      표준편차가 79.3 이라 단일 시드 비교로는 79 점 이하 차이를 신뢰할 수 없다.
    · 가중치를 fold 2023 에서 적합하면 fold 2024 로 전이되지 않는다
      (middle 을 lr23 에 통과시키기만 해도 903.9 -> 819.6). 그래서 **단순 평균만**
      쓴다.
    · 목표가 두 fold 동시이므로 선택 기준은 min(f23, f24) 다.

사용
    python way_optimize.py --target middle
    python way_optimize.py --target middle,reverse,outside --max-k 5
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness3 import OUT, TARGETS, bss, load_labeled

EPS = 1e-7
lgt = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
sig = lambda z: 1.0 / (1.0 + np.exp(-z))
HONEST = ("ww_", "ms_", "mn_", "wb_", "mh_", "mt_")
FOLDS = (2023, 2024)


def collect(tg: str, Y: dict) -> dict:
    """정직한 예측을 모으고 같은 설정의 시드는 로짓 평균으로 합친다."""
    raw = {f: {} for f in FOLDS}
    for p in glob.glob(str(OUT / "*.npy")):
        b = Path(p).name
        if not b.startswith(HONEST):
            continue
        m = re.match(rf"(\w+?)_{tg}__(.+)__(\d{{4}})\.npy$", b)
        if not m:
            continue
        f = int(m.group(3))
        if f not in Y:
            continue
        v = np.load(p).astype(np.float64)
        if len(v) != len(Y[f]):
            continue
        key = f"{m.group(1)}:{re.sub(r'_s\d+$', '', m.group(2))}"
        raw[f].setdefault(key, []).append(v)
    out = {f: {k: sig(np.mean([lgt(x) for x in vs], 0)) for k, vs in raw[f].items()}
           for f in FOLDS}
    nseed = {k: len(vs) for k, vs in raw[FOLDS[0]].items()}
    return out, nseed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="middle,reverse,outside")
    ap.add_argument("--pool", type=int, default=14, help="후보 풀 크기")
    ap.add_argument("--max-k", type=int, default=5)
    args = ap.parse_args()

    d = load_labeled()
    s = d["season"].to_numpy()
    ok0 = d["label_ok"].to_numpy() == 1
    summary = []

    for tg in [t.strip() for t in args.target.split(",") if t.strip()]:
        y = pd.to_numeric(d[TARGETS[tg]], errors="coerce").to_numpy(np.float64)
        ok = ok0 & ~np.isnan(y)
        Y = {f: y[(s == f) & ok] for f in FOLDS}
        P, nseed = collect(tg, Y)
        common = sorted(set(P[2023]) & set(P[2024]))
        if len(common) < 2:
            print(f"{tg}: 두 fold 공통 예측 {len(common)}개 — 건너뛴다")
            continue

        sc = {k: {f: bss(Y[f], P[f][k]) for f in FOLDS} for k in common}
        mn = lambda k: min(sc[k][2023]["bss_raw"], sc[k][2024]["bss_raw"])
        rank = sorted(common, key=lambda k: -mn(k))

        print(f"{chr(10)}{'=' * 104}")
        print(f"{tg}   정직한 구성 {len(common)}개   목표 f24>1300 & f23>1000")
        print("=" * 104)
        print(f"  {'구성':<58}{'시드':>5}{'f23':>10}{'f24':>10}{'최소':>10}")
        for k in rank[:8]:
            print(f"  {k[:58]:<58}{nseed.get(k, 1):>5}"
                  f"{sc[k][2023]['bss_raw']:>10.1f}{sc[k][2024]['bss_raw']:>10.1f}"
                  f"{mn(k):>10.1f}")

        # 탐욕 앙상블 — min(f23,f24) 를 올리는 멤버를 하나씩 추가
        pool = rank[:args.pool]
        cur = [pool[0]]
        best = mn(pool[0])
        hist = [(1, sc[pool[0]][2023]["bss_raw"], sc[pool[0]][2024]["bss_raw"])]
        while len(cur) < args.max_k:
            gain, pick = None, None
            for k in pool:
                if k in cur:
                    continue
                c = cur + [k]
                a = bss(Y[2023], sig(np.mean([lgt(P[2023][x]) for x in c], 0)))["bss_raw"]
                b = bss(Y[2024], sig(np.mean([lgt(P[2024][x]) for x in c], 0)))["bss_raw"]
                if gain is None or min(a, b) > gain:
                    gain, pick, ab = min(a, b), k, (a, b)
            if pick is None or gain <= best:
                break
            cur.append(pick)
            best = gain
            hist.append((len(cur), ab[0], ab[1]))

        print(f"{chr(10)}  탐욕 앙상블 (단순평균, min(f23,f24) 기준)")
        print(f"  {'k':>3}{'f23':>10}{'f24':>10}{'최소':>10}   추가된 구성")
        for i, (k, a, b) in enumerate(hist):
            nm = cur[i][:52] if i < len(cur) else ""
            print(f"  {k:>3}{a:>10.1f}{b:>10.1f}{min(a, b):>10.1f}   {nm}")
        fa = hist[-1][1]
        fb = hist[-1][2]
        print(f"{chr(10)}  최종  f23 {fa:.1f} {'통과' if fa > 1000 else '미달'}   "
              f"f24 {fb:.1f} {'통과' if fb > 1300 else '미달'}"
              f"   단독 최고 대비 {min(fa, fb) - mn(rank[0]):+.1f}")
        summary.append({"way": tg, "k": len(cur), "f23": fa, "f24": fb,
                        "members": " + ".join(cur)})

    if summary:
        t = pd.DataFrame(summary)
        t.to_csv(OUT / "way_optimize.csv", index=False)
        print(f"{chr(10)}{'=' * 104}")
        print("way 별 최적 확률 (결합 전)")
        print("=" * 104)
        for r in t.itertuples():
            print(f"  {r.way:<9} k={r.k}   f23 {r.f23:>9.1f}   f24 {r.f24:>9.1f}")
        print(f"{chr(10)}saved -> {OUT / 'way_optimize.csv'}")


if __name__ == "__main__":
    main()
