"""두 fold 동시 판정 보고서. CPU 전용.

왜 두 fold 인가
    fold 2024 단독으로 고르면 과적합한다. 1WAY 에서 단일 fold 선별이
    네 번 뒤집혔다 (V23·V38·V47·V64).

판정 규칙
    주 지표는 **bss_raw** 다. centered 는 참고만 한다 —
    Stage 1 에서 클래스 가중 arm 이 centered 로는 1위인데 raw 로는 -4,791 이었다.
    평가 라벨로 평균을 맞추는 것은 규정상 제출에 쓸 수 없으므로(조항 2),
    raw 가 실제로 얻는 값이다.

    채택 조건
      ㄱ 두 fold 모두 기준선 대비 양수
      ㄴ 두 fold 각각의 잡음 폭(타깃별 sd)을 넘을 것
      ㄷ 최악 fold 개선폭으로 순위를 매긴다 (보수적)

    "2024 만 양수" 는 따로 표시한다. 그게 과적합의 전형이다.

사용
    python report.py                      전체
    python report.py --target middle      한 타깃
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness3 import AUX_FOLD, DECISION_FOLD, OUT, seed_noise

GOAL = 1300.0


def collect() -> pd.DataFrame:
    """outputs 의 결과 CSV 를 한 표로 모은다."""
    frames = []
    for p in sorted(OUT.glob("*.csv")):
        try:
            d = pd.read_csv(p)
        except Exception:                                        # noqa: BLE001
            continue
        if "bss_raw" not in d.columns:
            continue
        key = "arm" if "arm" in d.columns else ("combo" if "combo" in d.columns else None)
        if key is None or "target" not in d.columns:
            continue
        if "fold" not in d.columns:
            m = re.search(r"(20\d\d)", p.stem)
            if not m:
                continue
            d = d.assign(fold=int(m.group(1)))
        d = d.rename(columns={key: "name"})
        d["source"] = p.stem
        d["kind"] = "arm" if key == "arm" else "combo"
        for c in ("bss_centered", "offset"):
            if c not in d.columns:
                d[c] = np.nan
        frames.append(d[["target", "name", "fold", "kind", "source",
                         "bss_raw", "bss_centered", "offset"]])
    if not frames:
        return pd.DataFrame()
    t = pd.concat(frames, ignore_index=True)
    # 같은 (타깃, 이름, fold) 가 여러 번이면 마지막 것
    return t.drop_duplicates(["target", "name", "fold", "kind"], keep="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="")
    ap.add_argument("--kind", default="", help="arm 또는 combo")
    args = ap.parse_args()

    t = collect()
    if t.empty:
        print("결과 CSV 가 없다."); return
    if args.target:
        t = t[t.target.isin([x.strip() for x in args.target.split(",")])]
    if args.kind:
        t = t[t.kind == args.kind]

    print("=" * 100)
    print(f"두 fold 동시 판정   주 지표 bss_raw   목표 {GOAL:.0f}")
    print("=" * 100)

    for kind in ("combo", "arm"):
        sub = t[t.kind == kind]
        if sub.empty:
            continue
        print(f"{chr(10)}### {'전처리 조합' if kind == 'combo' else '학습 방식'}")
        for tg in sorted(sub.target.unique()):
            d = sub[sub.target == tg]
            piv = d.pivot_table(index="name", columns="fold", values="bss_raw")
            base_name = "baseline" if "baseline" in piv.index else (
                "base" if "base" in piv.index else None)
            folds = [f for f in (AUX_FOLD, DECISION_FOLD) if f in piv.columns]
            sd = seed_noise(tg)
            print(f"{chr(10)}  [{tg}]  잡음 sd {sd:.2f}   "
                  f"fold {folds}   {len(piv)}개")
            if base_name is None or len(folds) == 0:
                print("    기준선 없음 — 건너뜀"); continue
            b = piv.loc[base_name]
            for f in folds:
                gap = GOAL - b[f]
                print(f"    기준선 fold {f}: {b[f]:8.1f}   목표까지 {gap:+.0f}")
            if len(folds) < 2:
                print(f"    ! fold {DECISION_FOLD} 만 있다 — 단일 fold 판정은 네 번 뒤집혔다")
                top = piv[folds[0]].nlargest(6)
                for n, v in top.items():
                    print(f"      {n:<30}{v:>9.1f}{v - b[folds[0]]:>+9.1f}")
                continue
            d23, d24 = piv[AUX_FOLD] - b[AUX_FOLD], piv[DECISION_FOLD] - b[DECISION_FOLD]
            both = pd.DataFrame({"f23": piv[AUX_FOLD], "f24": piv[DECISION_FOLD],
                                 "d23": d23, "d24": d24}).dropna()
            both["worst"] = both[["d23", "d24"]].min(axis=1)
            both["pass"] = (both.d23 > sd) & (both.d24 > sd)
            ok = both[both["pass"]].sort_values("worst", ascending=False)
            print(f"    {'이름':<30}{'f23':>9}{'Δ23':>8}{'f24':>9}{'Δ24':>8}   판정")
            for n, r in ok.head(6).iterrows():
                print(f"    {n:<30}{r.f23:>9.1f}{r.d23:>+8.1f}"
                      f"{r.f24:>9.1f}{r.d24:>+8.1f}   채택 후보")
            if ok.empty:
                print(f"    (두 fold 모두 잡음을 넘는 것 없음)")
            only24 = both[(both.d24 > sd) & (both.d23 <= sd)].sort_values(
                "d24", ascending=False)
            if len(only24):
                print(f"    -- 2024 만 양수 ({len(only24)}개) — 과적합 의심 --")
                for n, r in only24.head(4).iterrows():
                    print(f"    {n:<30}{r.f23:>9.1f}{r.d23:>+8.1f}"
                          f"{r.f24:>9.1f}{r.d24:>+8.1f}   보류")

    print(f"{chr(10)}{'=' * 100}")
    print("목표 대비 현재 최고 (두 fold 모두 있는 것 중 최악 fold 기준)")
    print("=" * 100)
    print(f"  {'타깃':<10}{'f23':>10}{'f24':>10}{'최악':>10}{'목표까지':>10}  구성")
    for tg in sorted(t.target.unique()):
        d = t[t.target == tg]
        piv = d.pivot_table(index=["kind", "name"], columns="fold", values="bss_raw")
        if not {AUX_FOLD, DECISION_FOLD} <= set(piv.columns):
            best = piv[DECISION_FOLD].max() if DECISION_FOLD in piv.columns else np.nan
            print(f"  {tg:<10}{'—':>10}{best:>10.1f}{'—':>10}"
                  f"{GOAL - best:>+10.0f}  (fold {DECISION_FOLD} 만)")
            continue
        piv = piv.dropna(subset=[AUX_FOLD, DECISION_FOLD])
        piv["worst"] = piv[[AUX_FOLD, DECISION_FOLD]].min(axis=1)
        r = piv.nlargest(1, "worst")
        if r.empty:
            continue
        idx = r.index[0]
        k, n = (idx if isinstance(idx, tuple) else ("?", idx))
        row = r.iloc[0]
        print(f"  {tg:<10}{float(row[AUX_FOLD]):>10.1f}"
              f"{float(row[DECISION_FOLD]):>10.1f}"
              f"{float(row['worst']):>10.1f}"
              f"{GOAL - float(row['worst']):>+10.0f}  {k}:{str(n)[:40]}")


if __name__ == "__main__":
    main()
