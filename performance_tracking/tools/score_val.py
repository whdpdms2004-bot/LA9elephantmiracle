"""규칙 1 채점기 (2026-08-28 개정) — 2024 는 R+F, 2022·2023 은 R 만.

    python performance_tracking/tools/score_val.py <name>
    python performance_tracking/tools/score_val.py <name> --baseline <기준> --register

    주 판정   2024 all (R+F)          올라야 한다
    관문      2023 R · 2022 R         떨어지면 안 된다

2024 가 올라도 관문 시즌이 떨어지면 기각이다 — 한 시즌만 보고 고른 구성이
시즌 전이에서 뒤집힌 실측이 반복적으로 있었다.

**관문을 all 이 아니라 R 로 보는 이유**: 2022 의 F 기저율은 0.7087, R 은 0.5037 이다
(2024 는 F 0.4593 · R 0.4897 로 거의 같다). all 로 재면 2022 관문의 약 70% 가
성능이 아니라 game_type 구성을 재게 된다. 근거는 group_by_perform/RESULTS.md §1.

2023 예측이 아직 없는 모델은 그 관문을 **미확인**으로 두고 판정을 보류하지 않는다 —
확보되면 자동으로 관문에 들어온다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common import (DECISION_SEASON, DECISION_SUBGROUP, GUARD_SEASONS,
                    GUARD_SUBGROUP, MODELS_DIR, REQUIRED_SEASONS, RESULTS,
                    SEASONS, SpecViolation, load_labels, load_pred, render,
                    score, val_path)

# 규칙 1 판정 여유. 시드 산포보다 작은 차이를 상승으로 읽지 않는다.
TOL = 0.0


def check_layout(name: str) -> list[str]:
    """규칙 2·4 - md / zip / 학습 폴더 / val 예측 이름이 모두 같은지."""
    missing = []
    if not (MODELS_DIR / f"{name}.md").exists():
        missing.append(f"models/{name}.md (규칙 2)")
    if not (MODELS_DIR / f"{name}.zip").exists():
        missing.append(f"models/{name}.zip (규칙 4)")
    if not (MODELS_DIR / name).is_dir():
        missing.append(f"models/{name}/ 학습 스크립트 폴더 (규칙 4)")
    for s in REQUIRED_SEASONS:
        if not val_path(name, s).exists():
            missing.append(f"val/{name}_{s}.csv (규칙 3)")
    return missing


def optional_missing(name: str) -> list[int]:
    """정책상 필요하지만 아직 자산이 없는 val 시즌 (지금은 2023)."""
    return [s for s in SEASONS
            if s not in REQUIRED_SEASONS and not val_path(name, s).exists()]


def evaluate(name: str) -> dict[int, dict]:
    """있는 시즌만 채점한다. 없는 시즌은 키 자체가 없다."""
    out = {}
    for s in SEASONS:
        if not val_path(name, s).exists():
            continue
        lab = load_labels(s)
        out[s] = score(load_pred(name, s, lab), s, lab)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--baseline", help="비교 기준 모델 이름")
    ap.add_argument("--register", action="store_true", help="results.csv 에 한 행 append")
    ap.add_argument("--note", default="", help="results.csv note 칸")
    ap.add_argument("--date", default="", help="제출일 (기본: 오늘)")
    ap.add_argument("--skip-layout", action="store_true",
                    help="md/zip/폴더 점검 생략 (채점만 먼저 볼 때)")
    a = ap.parse_args()

    missing = check_layout(a.name)
    if missing:
        head = "규격 미비:\n" + "\n".join(f"  - {m}" for m in missing)
        if a.skip_layout and not any(m.startswith("val/") for m in missing):
            print(head + "\n  (--skip-layout: 채점만 진행)\n")
        else:
            print(head)
            print("\nval 예측이 없으면 채점 자체가 불가하다. README 5절 절차를 따른다.")
            return 1

    try:
        cur = evaluate(a.name)
    except SpecViolation as e:
        print(f"규격 위반: {e}")
        return 1

    print(f"== {a.name}")
    for s in SEASONS:
        if s in cur:
            print(render(cur[s]))
    gap = optional_missing(a.name)
    if gap:
        print("  " + " · ".join(f"val/{a.name}_{s}.csv 없음" for s in gap)
              + "  → 그 관문은 미확인으로 둔다 (규칙 1, 2026-08-28 개정)")

    if not a.baseline:
        print("\n기준 모델 없이는 채택 판정을 하지 않는다 (--baseline <name>).")
        verdict = "미판정"
    else:
        try:
            base = evaluate(a.baseline)
        except (SpecViolation, FileNotFoundError) as e:
            print(f"\n기준 모델 로드 실패: {e}")
            return 1
        d_dec = (cur[DECISION_SEASON][DECISION_SUBGROUP]
                 - base[DECISION_SEASON][DECISION_SUBGROUP])
        print(f"\n== vs {a.baseline}")
        print(f"  {DECISION_SEASON} {DECISION_SUBGROUP:<3} "
              f"{base[DECISION_SEASON][DECISION_SUBGROUP]:>10,.1f} -> "
              f"{cur[DECISION_SEASON][DECISION_SUBGROUP]:>10,.1f}  "
              f"({d_dec:+,.1f})   [주 판정]")

        up_dec = d_dec > TOL
        why, unknown = [], []
        if not up_dec:
            why.append(f"{DECISION_SEASON} 미상승")
        for gs in GUARD_SEASONS:
            if gs not in cur or gs not in base:
                unknown.append(str(gs))
                print(f"  {gs} {GUARD_SUBGROUP:<3} "
                      f"{'':>10} -> {'':>10}   (미확인)      [비하락 관문]")
                continue
            dg = cur[gs][GUARD_SUBGROUP] - base[gs][GUARD_SUBGROUP]
            print(f"  {gs} {GUARD_SUBGROUP:<3} "
                  f"{base[gs][GUARD_SUBGROUP]:>10,.1f} -> "
                  f"{cur[gs][GUARD_SUBGROUP]:>10,.1f}  ({dg:+,.1f})   [비하락 관문]")
            if dg < -TOL:
                why.append(f"{gs} R 하락")

        verdict = "채택" if not why else "기각"
        if verdict == "채택" and unknown:
            verdict = "조건부채택"
            why.append(f"{'·'.join(unknown)} 미확인")
        print(f"\n  판정: {verdict}" + (f" - {', '.join(why)}" if why else ""))

    if a.register:
        date = a.date or pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d")
        row = {
            "name": a.name, "date": date,
            "val2024_all": round(cur[2024]["all"], 1),
            "val2024_R": round(cur[2024]["R"], 1),
            "val2024_F": round(cur[2024]["F"], 1),
            "val2023_R": round(cur[2023]["R"], 1) if 2023 in cur else "",
            "val2022_all": round(cur[2022]["all"], 1),
            "val2022_R": round(cur[2022]["R"], 1),
            "val2022_F": round(cur[2022]["F"], 1),
            "baseline": a.baseline or "", "verdict": verdict,
            "public": "", "note": a.note,
        }
        df = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(columns=list(row))
        if (df["name"].astype(str) == a.name).any():
            print(f"\nresults.csv 에 {a.name} 이 이미 있다. 기존 행은 손대지 않는다 - "
                  "값이 바뀌었으면 그 행을 직접 수정한다.")
        else:
            pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(
                RESULTS, index=False)
            print(f"\nresults.csv 등록: {a.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
