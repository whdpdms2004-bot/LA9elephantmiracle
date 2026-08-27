# -*- coding: utf-8 -*-
"""[축1] 원자의 **행 독립성**과 **시간 인과**를 실증한다. GPU 불필요.

`preprocess_lab/README.md` 의 "반드시 지켜야 할 것 넷" 은 어기면 제출 실격이다.
구종 피처에 적용했던 것과 같은 기준으로 원자에도 건다 (PROGRESS.md 곁가지 절 참조).

**행 독립성 시험** — 검증행 하나의 피처가 *다른 검증행의 값*에 의존하면 안 된다.
    1. 원본 X 로 원자를 만든다 (기준)
    2. 표본 300행만 남기고 **나머지 검증행을 전부 망가뜨린다** (값 셔플)
    3. 다시 만들어 그 300행의 값이 **비트 단위로 같은지** 본다
망가뜨린 행이 통계에 섞였다면 값이 달라진다. 학습행은 건드리지 않으므로
학습에서 적합한 통계(빈도·중앙값·IQR)는 그대로여야 한다.

**시간 인과 시험** — 학습 마스크 밖(=검증 시즌) 값을 바꿔도 통계가 안 흔들리는지.
위 시험이 통과하면 자동으로 성립하지만, 통계 객체를 직접 비교해 한 번 더 못박는다.

    python check_atoms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
WORK = FINAL / "work"
sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import atoms as A                                                   # noqa: E402

FOLD = 2024
NSAMP = 300


def log(m):
    print(m, flush=True)


def main():
    import json
    X = np.asarray(np.load(WORK / "X168.npy", mmap_mode="r"))
    season = np.load(WORK / "season.npy")
    names = json.load(open(WORK / "meta.json", encoding="utf-8"))["names"]
    tr = season <= FOLD - 1
    va = np.where(season == FOLD)[0]
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(va, NSAMP, replace=False))
    keepset = set(keep.tolist())
    other = np.array([i for i in va if i not in keepset])

    log("=" * 78)
    log("원자 행 독립성 · 시간 인과 검증 | fold %d | 표본 %d행 | 오염 %s행"
        % (FOLD, NSAMP, f"{len(other):,}"))
    log("=" * 78)

    E_ref, en = A.build(X, names, tr, FOLD, list(A.ATOMS))
    log("  기준 생성 %d열" % len(en))

    # 검증행 중 표본 아닌 것들을 통째로 망가뜨린다 (열 단위 셔플 + 부호 반전)
    X2 = X.copy()
    sub = X2[other]
    for j in range(sub.shape[1]):
        sub[:, j] = rng.permutation(sub[:, j]) * -3.0 + 7.0
    X2[other] = sub
    del sub
    E_bad, en2 = A.build(X2, names, tr, FOLD, list(A.ATOMS))
    assert en == en2, "열 이름이 달라졌다"

    a = E_ref[keep]
    b = E_bad[keep]
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    log("\n[1] 행 독립성 — 표본 %d행 x %d열" % (NSAMP, len(en)))
    log("    최대 절대차 %.3e   %s" % (diff.max(), "통과" if diff.max() == 0 else "★실패"))

    bad_cols = [en[j] for j in range(len(en)) if diff[:, j].max() > 0]
    if bad_cols:
        log("    ★ 오염된 열 %d개: %s" % (len(bad_cols), ", ".join(bad_cols[:12])))

    # 원자별로도 쪼개 본다 — 어느 원자가 문제인지 바로 보이게
    log("\n[2] 원자별")
    for k in list(A.ATOMS):
        E1k, e1k = A.build(X, names, tr, FOLD, [k])
        E2k, _ = A.build(X2, names, tr, FOLD, [k])
        aa, bb = E1k[keep], E2k[keep]
        bn = np.isnan(aa) & np.isnan(bb)
        d = np.where(bn, 0.0, np.abs(aa - bb)).max()
        log("    %-12s %3d열   최대차 %.3e   %s"
            % (k, len(e1k), d, "통과" if d == 0 else "★실패"))
        del E1k, E2k

    # 시간 인과 — 학습 통계 자체가 같은지
    log("\n[3] 시간 인과 — 검증 시즌 값을 바꿔도 학습에서 적합한 통계가 흔들리면 안 된다")
    F1 = A.Frame(X, names, tr)
    F2 = A.Frame(X2, names, tr)
    ok = True
    for c in A.ID_COLS:
        u1, c1 = np.unique(F1.col(c)[tr], return_counts=True)
        u2, c2 = np.unique(F2.col(c)[tr], return_counts=True)
        same = np.array_equal(u1, u2) and np.array_equal(c1, c2)
        ok &= same
        log("    빈도표 %-18s %s" % (c, "동일" if same else "★다르다"))
    for _, rc, _ in A.RATE_SPECS[:3]:
        m1 = np.nanmedian(F1.col(rc)[tr]); m2 = np.nanmedian(F2.col(rc)[tr])
        same = (m1 == m2) or (np.isnan(m1) and np.isnan(m2))
        ok &= same
        log("    학습중앙값 %-30s %.8f vs %.8f  %s"
            % (rc, m1, m2, "동일" if same else "★다르다"))

    log("\n판정: %s" % ("전부 통과 — 행 독립성·시간 인과 위반 없음"
                       if diff.max() == 0 and ok else "★위반 있음. 원자를 고쳐야 한다"))
    return 0 if (diff.max() == 0 and ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
