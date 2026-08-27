# -*- coding: utf-8 -*-
"""각 제출본이 **어떤 피처를 썼는지** 뽑아 `features/` 에 남긴다.

`performance_tracking` 만 보고도 팀 전체를 파악할 수 있어야 한다. 지금은 모델 카드에
"57피처" "71피처" 같은 숫자만 있고 **이름 목록이 없다.** 그러면 서로 뭘 쓰는지,
누가 무엇을 안 쓰는지 비교할 수 없다.

## 출처

| 모델 | 어디서 뽑나 |
|---|---|
| `sj_stdmlp` · `cw_v17_base` | `cowork/sj/sj_final/work/meta.json` (X168) + `atoms.id_freq` 8열 |
| `hw_v12` | `models/hw_v12/build_val2024_pred_v12.py` 의 구성 규칙을 그대로 실행 |
| `yn_fa10c` | `models/yn_fa10c.zip` 의 `model/meta.json` |
| `sj3way` | 챔피언 zip 의 `model/sj/model/base_metadata.json` |
| `ye_hand` | `models/sj_stdmlp/ye_repro.py` 의 `build()` 결과 |

찾을 수 없으면 **없다고 기록한다.** 빈 값과 "확인 못 함" 은 다른 것이다.

    python extract_features.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

PT = Path(__file__).resolve().parents[1]
ROOT = PT.parent
SJF = ROOT / "cowork" / "sj" / "sj_final"
OUT = PT / "features"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def save(name, cols, src, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / ("%s.txt" % name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("# %s — %d개\n" % (name, len(cols)))
        f.write("# 출처: %s\n" % src)
        if note:
            f.write("# %s\n" % note)
        f.write("\n".join(cols) + "\n")
    print("  %-16s %4d개  → features/%s.txt" % (name, len(cols), name))
    return cols


def main():
    got = {}

    # ── sj / cw — X168 + id_freq 8 ─────────────────────────────────────────
    mj = SJF / "work" / "meta.json"
    if mj.exists():
        names = json.load(open(mj, encoding="utf-8"))["names"]
        got["cw_v17_base"] = save(
            "cw_v17_base", names,
            "cowork/sj/sj_final/work/meta.json (X168)",
            "cw v17 스택의 입력. sj 도 이 168 위에서 시작했다")
        idf = []
        for c in ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]:
            idf += ["pa_%s_logfreq" % c, "pa_%s_unseen" % c]
        got["sj_stdmlp"] = save(
            "sj_stdmlp", names + idf,
            "위 168 + atoms.id_freq 8열",
            "cb·mlp 는 176열, ft 는 168열을 쓴다 (표현이 서로 다르다)")
    else:
        print("  ★ X168 meta.json 없음")

    # ── hw_v12 — 코드의 구성 규칙을 그대로 실행 ────────────────────────────
    tc = ROOT / "data" / "test.csv"
    if tc.exists():
        base47 = [c for c in pd.read_csv(tc, nrows=0).columns if c != "row_id"]
        hw = (base47
              + ["trend_prev%d" % k for k in (1, 3, 5)]
              + ["trend_abs_prev%d" % k for k in (1, 3, 5)]
              + ["platoon_split", "platoon_n", "count_state", "handedness_matchup"])
        got["hw_v12"] = save(
            "hw_v12", hw,
            "models/hw_v12/build_val2024_pred_v12.py 의 구성 규칙",
            "범주 9개: top_bottom, game_type, base_state, pitcher_team_id, "
            "batter_team_id, count_state, pitcher_hand, batter_hand, handedness_matchup")

    # ── ye — ye_repro.build() 의 결과 ──────────────────────────────────────
    try:
        sys.path.insert(0, str(SJF / "src"))
        import ye_repro
        df = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig", nrows=20000)
        df.columns = [c.strip("﻿") for c in df.columns]
        _, feats = ye_repro.build(df, df["season"] <= 2023)
        got["ye_hand"] = save(
            "ye_hand", feats,
            "models/sj_stdmlp/ye_repro.py 의 build() (노트북 셀3 이식)",
            "범주 3개: top_bottom, base_state, score_situation")
    except Exception as e:
        print("  ★ ye 피처 추출 실패: %s" % e)

    # ── yn — zip 안 meta.json ──────────────────────────────────────────────
    zp = PT / "models" / "yn_fa10c.zip"
    if zp.exists():
        m = json.loads(zipfile.ZipFile(zp).read("model/meta.json"))
        cand = None
        for k in ("feature_cols", "features", "raw_features", "feature_names", "cols"):
            if isinstance(m.get(k), list):
                cand = m[k]
                break
        if cand is None:
            listy = [(k, v) for k, v in m.items()
                     if isinstance(v, list) and v and isinstance(v[0], str)]
            if listy:
                k, cand = max(listy, key=lambda t: len(t[1]))
                print("  yn: 키 '%s' 를 피처 목록으로 판단 (%d개)" % (k, len(cand)))
        # ★ yn 은 **부분만** 복원된다. meta.json 에 있는 것은 원시 입력(raw_features 47)과
        # 신규 3열뿐이고, SUBMISSION_LOG 가 말하는 71피처(68 파생 + 3)의 **파생 68개
        # 이름은 저장소에 없다** — 추론 스크립트가 만들어 쓰고 버린다.
        # 학습 코드가 없어(MISSING.md) 그 이름을 복원할 방법도 없다.
        newc = m.get("new_feature_cols", [])
        got["yn_fa10c"] = save(
            "yn_fa10c", list(cand) + list(newc),
            "models/yn_fa10c.zip 의 model/meta.json (raw_features + new_feature_cols)",
            "★불완전 — SUBMISSION_LOG 는 71피처(68 파생 + 3)라고 적었으나 "
            "파생 68개의 이름이 저장소에 없다. 학습 코드도 없어(MISSING.md) 복원 불가. "
            "여기 목록은 원시 입력 %d + 신규 %d 뿐이다" % (len(cand), len(newc)))

    # ── sj3way — 챔피언 zip 의 base_metadata.json ──────────────────────────
    cz = SJF / "submit" / "submit_sj_stdmlp.zip"
    if cz.exists():
        try:
            m = json.loads(zipfile.ZipFile(cz).read("model/sj/model/base_metadata.json"))
            cand = None
            for k in ("feature_cols", "features", "feature_names", "columns", "cols"):
                if isinstance(m.get(k), list):
                    cand = m[k]
                    break
            if cand is None:
                listy = [(k, v) for k, v in m.items()
                         if isinstance(v, list) and v and isinstance(v[0], str)]
                if listy:
                    k, cand = max(listy, key=lambda t: len(t[1]))
                    print("  sj3way: 키 '%s' 를 피처 목록으로 판단 (%d개)" % (k, len(cand)))
            if cand:
                got["sj3way"] = save(
                    "sj3way", cand,
                    "챔피언 zip 의 model/sj/model/base_metadata.json",
                    "sj 의 279피처 3WAY 트랙. 행렬 자체는 로컬에 없다(X168.npy 만 있음)")
            else:
                print("  ★ sj3way 피처 목록 못 찾음. 키: %s" % sorted(m)[:12])
        except KeyError as e:
            print("  ★ sj3way base_metadata.json 없음: %s" % e)

    # ── 요약 + 겹침 ───────────────────────────────────────────────────────
    print("\n피처 겹침 (교집합 / 합집합)")
    ks = list(got)
    print("        " + "".join("%14s" % k[:13] for k in ks))
    for a in ks:
        line = "%-8s" % a[:8]
        for b in ks:
            A, B = set(got[a]), set(got[b])
            line += "%14s" % ("%d/%d" % (len(A & B), len(A | B)))
        print(line)

    rows = [{"모델": k, "피처 수": len(v)} for k, v in got.items()]
    miss = [m for m in ("cw_v17_base", "sj_stdmlp", "hw_v12", "yn_fa10c",
                        "ye_hand", "sj3way") if m not in got]
    for m in miss:
        rows.append({"모델": m, "피처 수": None})
    pd.DataFrame(rows).to_csv(OUT / "_summary.csv", index=False, encoding="utf-8-sig")
    if miss:
        print("\n★ 목록을 못 만든 모델: %s" % miss)
    print("\n→ %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
