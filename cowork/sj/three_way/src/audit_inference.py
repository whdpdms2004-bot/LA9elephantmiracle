"""3WAY 추론 경로 실측 검증 — 부분 실행 == 전체 실행. CPU 전용.

왜 필요한가
    정적 검사(패턴 grep)는 통과해도 실제로 행이 서로 영향을 주는지는 못 잡는다.
    1WAY 에서 훅 중복 주입이 6단계 검증을 통과하고 제출돼 3번을 날린 전례가 있다.
    제출 경로는 한 번 틀리면 실격이므로 실측이 필요하다.

무엇을 재는가
    같은 행을 (ㄱ) 전체 프레임 안에서 (ㄴ) 잘라낸 작은 프레임 안에서
    각각 피처로 만들었을 때 값이 정확히 같은가.
    다르면 그 행의 피처가 다른 행에 의존한다는 뜻 = RULES 조항 1 위반.

검사 대상
    three_way_runtime.py 의 행 단위 변환 함수 전부.
    학습 자산(lookup, spec)은 고정값이므로 그대로 넘긴다.

사용
    python audit_inference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from harness3 import LAB, SUCCESS, load_labeled

FAIL, WARN = [], []


def check(name, ok, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}   {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def main() -> None:
    rt_path = SRC / "three_way_runtime.py"
    if not rt_path.exists():
        print(f"추론 런타임이 없다: {rt_path}")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("tw_runtime", rt_path)
    RT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RT)

    df = load_labeled()
    season = df["season"].to_numpy()
    tr = season < 2024
    rng = np.random.default_rng(11)
    va_idx = np.flatnonzero(season == 2024)
    pick = np.sort(rng.choice(va_idx, size=300, replace=False))

    print(f"평가 대상 행 {len(pick)}개 (fold 2024 무작위)")
    print(f"런타임 함수: {[f for f in dir(RT) if f.startswith('_') and callable(getattr(RT, f))]}")

    # 학습 자산 — 실제 제출에서 metadata 로 고정 전달되는 것들을 재현
    lookups = {}
    for c in getattr(RT, "ID_COLUMNS", []):
        if c in df.columns:
            lookups[c] = df.loc[tr, c].value_counts()
    prior = {}
    for name, rate_col, n_col in getattr(RT, "RATE_SPECS", []):
        if rate_col in df.columns:
            prior[name] = float(pd.to_numeric(df.loc[tr, rate_col],
                                              errors="coerce").mean())

    full = df
    part = df.loc[np.r_[np.flatnonzero(tr)[:5000], pick]].reset_index(drop=True)
    pos_full = pick
    pos_part = np.arange(len(part) - len(pick), len(part))

    print(f"{chr(10)}" + "=" * 88)
    print("행 독립성 — 전체 프레임 vs 잘라낸 프레임에서 같은 행의 피처")
    print("=" * 88)

    cases = []
    if hasattr(RT, "_id_frequency"):
        cases.append(("_id_frequency", lambda f: RT._id_frequency(f, lookups)))
    if hasattr(RT, "_count_multiscale"):
        cases.append(("_count_multiscale", RT._count_multiscale))
    if hasattr(RT, "_temporal_cyclic"):
        cases.append(("_temporal_cyclic", lambda f: RT._temporal_cyclic(f, 2025)))
    if hasattr(RT, "_rate_multiscale") and prior:
        cases.append(("_rate_multiscale", lambda f: RT._rate_multiscale(f, prior)))

    for name, fn in cases:
        try:
            a = fn(full)
            b = fn(part)
        except Exception as exc:                                  # noqa: BLE001
            check(name, False, f"{type(exc).__name__}: {str(exc)[:60]}")
            continue
        keys = sorted(set(a) & set(b))
        if not keys:
            check(name, False, "공통 출력 열 없음")
            continue
        worst, bad = 0.0, []
        for k in keys:
            va = np.asarray(a[k], np.float64)[pos_full]
            vb = np.asarray(b[k], np.float64)[pos_part]
            d = np.nanmax(np.abs(np.nan_to_num(va) - np.nan_to_num(vb)))
            if d > worst:
                worst = float(d)
            if d > 1e-9:
                bad.append(k)
        check(f"{name} ({len(keys)}열)", worst < 1e-9,
              f"최대 차이 {worst:.3e}" + (f"  문제열 {bad[:3]}" if bad else ""))

    print(f"{chr(10)}{'=' * 88}")
    print("단일 행 추론 — 한 행만 넣어도 같은 값이 나오는가")
    print("=" * 88)
    one_idx = int(pick[0])
    one = df.loc[[one_idx]].reset_index(drop=True)
    for name, fn in cases:
        try:
            a = fn(full)
            b = fn(one)
        except Exception as exc:                                  # noqa: BLE001
            check(f"{name} 단일행", False, f"{type(exc).__name__}: {str(exc)[:60]}")
            continue
        keys = sorted(set(a) & set(b))
        worst = 0.0
        for k in keys:
            va = float(np.nan_to_num(np.asarray(a[k], np.float64)[one_idx]))
            vb = float(np.nan_to_num(np.asarray(b[k], np.float64)[0]))
            worst = max(worst, abs(va - vb))
        check(f"{name} 단일행", worst < 1e-9, f"최대 차이 {worst:.3e}")

    print(f"{chr(10)}{'=' * 88}")
    print(f"결과: {'전 항목 통과' if not FAIL else 'FAIL ' + str(FAIL)}")
    print("=" * 88)
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
