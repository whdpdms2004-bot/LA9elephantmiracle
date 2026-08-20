"""way 별 상위 20 구성을 결합 단계용으로 고정 저장한다.

왜 20 개인가
    다음 단계는 세 way 를 결합해 최종 제구 성공 확률을 만드는 것이다.
    결합 최적화는 way 마다 여러 후보를 놓고 조합을 탐색하므로, 단일 최고 하나가
    아니라 **성격이 다른 후보 묶음** 이 필요하다.
    실측: 같은 레시피 변형끼리는 앙상블 이득이 0 이었고, 서로 다른 구조
    (split_ball vs split_bc) 를 섞을 때만 두 fold 가 같이 올랐다.
    그래서 점수 상위만 뽑지 않고 **비상관성** 을 섞어 뽑는다.

선정 방식
    1) f24 상위 5      — 판정 fold 기준 최고들
    2) f23 상위 5      — 강건성 쪽 최고들
    3) min(f23,f24) 상위 5 — 두 fold 균형
    4) 나머지는 이미 뽑힌 것들과 로짓 상관이 낮은 순으로 채운다
    중복은 제거하고 20 개를 채운다.

무엇을 저장하나
    outputs/top20/<way>__<rank>__<fold>.npy   시드 배깅까지 끝난 예측
    outputs/top20/manifest.csv                구성 이름, 두 fold 점수, 시드 수, 파일 경로
    결합 단계는 manifest 만 읽으면 된다.

정직성
    채점 fold 를 조기 종료에 쓴 예측(tr_/screen/mf_/s3_)은 제외한다.
    audit_earlystop.py 참고.

사용
    python way_top20.py
    python way_top20.py --target middle --n 20
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
DEST = OUT / "top20"


def collect(tg: str, Y: dict):
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
        raw[f].setdefault(f"{m.group(1)}:{re.sub(r'_s\d+$', '', m.group(2))}",
                          []).append(v)
    P = {f: {k: sig(np.mean([lgt(x) for x in vs], 0)) for k, vs in raw[f].items()}
         for f in FOLDS}
    nseed = {k: len(vs) for k, vs in raw[FOLDS[0]].items()}
    return P, nseed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="middle,reverse,outside")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    d = load_labeled()
    s = d["season"].to_numpy()
    ok0 = d["label_ok"].to_numpy() == 1
    rows = []

    for tg in [t.strip() for t in args.target.split(",") if t.strip()]:
        y = pd.to_numeric(d[TARGETS[tg]], errors="coerce").to_numpy(np.float64)
        ok = ok0 & ~np.isnan(y)
        Y = {f: y[(s == f) & ok] for f in FOLDS}
        P, nseed = collect(tg, Y)
        common = sorted(set(P[2023]) & set(P[2024]))
        if not common:
            print(f"{tg}: 두 fold 공통 예측 없음")
            continue
        sc = {k: {f: bss(Y[f], P[f][k]) for f in FOLDS} for k in common}
        f23 = lambda k: sc[k][2023]["bss_raw"]
        f24 = lambda k: sc[k][2024]["bss_raw"]
        mn = lambda k: min(f23(k), f24(k))

        pick = []
        for key, take in ((f24, 5), (f23, 5), (mn, 5)):
            added = 0
            for k in sorted(common, key=lambda x: -key(x)):
                if k in pick:
                    continue
                pick.append(k)
                added += 1
                if added >= take:
                    break
        # 나머지는 이미 뽑힌 것들과 상관이 가장 낮은 순으로 채운다
        Z = {k: lgt(P[2024][k]) for k in common}
        while len(pick) < min(args.n, len(common)):
            best, bc = None, 2.0
            for k in common:
                if k in pick:
                    continue
                c = max(abs(float(np.corrcoef(Z[k], Z[p])[0, 1])) for p in pick)
                if c < bc:
                    best, bc = k, c
            if best is None:
                break
            pick.append(best)
        pick = pick[:args.n]

        print(f"\n{'=' * 104}")
        print(f"{tg}   정직한 구성 {len(common)}개 중 상위 {len(pick)}개 저장")
        print("=" * 104)
        print(f"  {'#':>3}  {'구성':<58}{'시드':>4}{'f23':>10}{'f24':>10}")
        for i, k in enumerate(pick, 1):
            for f in FOLDS:
                np.save(DEST / f"{tg}__{i:02d}__{f}.npy", P[f][k])
            a, b = sc[k][2023], sc[k][2024]
            print(f"  {i:>3}  {k[:58]:<58}{nseed.get(k, 1):>4}"
                  f"{a['bss_raw']:>10.1f}{b['bss_raw']:>10.1f}")
            rows.append({
                "way": tg, "rank": i, "config": k, "seeds": nseed.get(k, 1),
                "f23_raw": a["bss_raw"], "f23_centered": a["bss_centered"],
                "f23_offset": a["offset"], "f24_raw": b["bss_raw"],
                "f24_centered": b["bss_centered"], "f24_offset": b["offset"],
                "path_2023": str(DEST / f"{tg}__{i:02d}__2023.npy"),
                "path_2024": str(DEST / f"{tg}__{i:02d}__2024.npy"),
            })

    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(DEST / "manifest.csv", index=False)
        print(f"\n{'=' * 104}")
        print(f"저장 완료 -> {DEST}")
        print(f"  manifest.csv  {len(t)}행 ({t.way.nunique()} way)")
        for w, g in t.groupby("way"):
            print(f"  {w:<9} {len(g)}개   f23 {g.f23_raw.min():.0f}~{g.f23_raw.max():.0f}"
                  f"   f24 {g.f24_raw.min():.0f}~{g.f24_raw.max():.0f}")
        print(f"\n결합 단계는 manifest.csv 만 읽으면 된다 "
              f"(way x 20 조합 탐색).")


if __name__ == "__main__":
    main()
