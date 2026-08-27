# -*- coding: utf-8 -*-
"""팀원 자산 등록부를 엑셀 한 권으로 만든다.

`results.csv` 는 실험 등록부라 행이 많고 기계용이다. 이 파일은 **사람이 보는 것** —
"누구 것인가 / 어디서 왔나 / val 은 어떻게 쟀나 / 결합에서 쓸모가 있나" 를 한눈에 본다.

## val 은 전부 sj 가 다시 계산했다

팀원이 보고한 값을 그대로 옮기지 않았다. **같은 폴드 · 같은 식 · 같은 행 정합**으로
다시 쟀다. 그래야 서로 비교가 되고 결합 가중을 적합할 수 있다.

    val2024   train.csv 의 season==2024   253,507행
    val2022   train.csv 의 season==2022   247,472행
    BSS = 100000 x (1 - Brier / (r(1-r))),  r = 그 폴드의 실제 성공률
    예측은 row_id 로 정합한다 (행 순서에 의존하지 않는다)

팀원 자체보고와 다른 경우가 있다 (예: yn 자체보고 880.79 vs 여기 815.6).
**두 숫자를 섞지 않는다** — 이 파일은 전부 sj 재계산 기준이다.

    python build_team_xlsx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PT = Path(__file__).resolve().parents[1]
ROOT = PT.parent
OUT = PT / "TEAM_ASSETS.xlsx"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def load_labels(fold):
    return pd.read_csv(PT / ".cache" / ("labels_%d.csv" % fold))


def val_score(name, fold):
    p = PT / "val" / ("%s_%d.csv" % (name, fold))
    if not p.exists():
        return None, None
    L = load_labels(fold)
    d = pd.read_csv(p)[["row_id", "pred"]]
    j = L[["row_id", "y"]].merge(d, on="row_id", how="inner")
    return round(bss(j["pred"].to_numpy(float), j["y"].to_numpy(float)), 1), len(j)


# ── 1. 자산 목록 ───────────────────────────────────────────────────────────
# (등록명, 팀원, 원 출처, val 예측 출처, Public, 비고)
ASSETS = [
    ("cw_v17_base", "cw", "cowork/cw/v17/ · final_submissions/cw/submit_v17_base.zip",
     "팀원 제공", 983.0,
     "168피처 CB(RMSE)+FT+MLP. sj 가 이 스택을 동결해 개선했으므로 sj 현행과 ρ0.99 — 사실상 같은 모델"),
    ("hw_v12", "hw", "cowork/hw/ → models/hw_v12/ (학습·추론 코드 전부 있음)",
     "팀원 제공 (build_val2024_pred_v12.py)", 912.1253278902,
     "★공유분이 eval_set=(x_val,y_val) 로 **검증 라벨을 조기종료에 썼다** — 오염. sj 가 정직본을 다시 만들었다"),
    ("hw_v12_honest", "hw → sj 재생산", "models/sj_stdmlp/hw_honest.py",
     "sj 재생산 (eval_set 제거 · league_avg 를 fit 에서만)", None,
     "공유분 808.9 vs 정직 720.6 (차 −88.3, ρ0.9488). val2022 2204.9 도 함께 생성 — "
     "규칙1 비하락 관문을 hw 에 적용할 수 있게 됐다"),
    ("yn_fa10c", "yn", "cowork/sj/last_week/work/fa10c/ → models/yn_fa10c/ (추론 코드만)",
     "팀원 제공", 964.2538081998,
     "★학습 코드 없음(MISSING.md) — 예측 출처를 검증할 수 없고 재현도 불가. val2022 미보유"),
    ("ye_hand", "ye → sj 재생산", "cowork/ye/champion_structural_improvement.ipynb",
     "sj 재현 (models/sj_stdmlp/ye_repro.py)", None,
     "노트북에 코드 전부 있었다(출력만 삭제·모델은 맥 로컬). platoon/hand CatBoost 60피처. ρ(cw) 0.7952 로 팀 최저"),
    ("sj3way", "sj", "cowork/sj/val2024_pred.csv",
     "sj 자체", None,
     "챔피언 zip 의 model/sj/ 멤버. 팀 결합의 두 축 중 하나"),
    ("sj_grid_w060", "sj", "cowork/sj/sj_final/",
     "sj 자체", 1080.4252194208, "현재 채점 최고 (팀 결합본)"),
    ("sj_stdmlp", "sj", "cowork/sj/sj_final/",
     "sj 자체", None, "미채점. val 기준 최고 — S8 3관문 통과"),
]

# 피처 목록 출처 — `features/` 폴더의 각 파일과 짝이다
FEATS = {
    "cw_v17_base": ("cowork/sj/sj_final/work/meta.json (X168)", "완전",
                    "이름 168개 중 고유 167 — `cnt_31_x_tend` 가 중복이다"),
    "sj_stdmlp": ("위 168 + atoms.id_freq 8", "완전",
                  "cb·mlp 는 176열, ft 는 168열 (표현이 서로 다르다)"),
    "hw_v12": ("models/hw_v12/build_val2024_pred_v12.py 의 구성 규칙", "완전",
               "baseline47 + trend6 + platoon2 + count_state + handedness_matchup"),
    "ye_hand": ("models/sj_stdmlp/ye_repro.py (노트북 셀3 이식)", "완전",
                "sj 가 재현하며 확정. 범주 3개"),
    "yn_fa10c": ("models/yn_fa10c.zip 의 model/meta.json", "★불완전",
                 "원시입력 47 + 신규 3 만 있다. SUBMISSION_LOG 의 71피처 중 "
                 "**파생 68개 이름이 저장소에 없다** — 학습 코드도 없어(MISSING.md) 복원 불가"),
    "sj3way": ("챔피언 zip 의 model/sj/model/base_metadata.json", "완전",
               "feature_columns 272개가 실제 배포 모델 입력. 문서의 '279' 는 변형 범위값"),
}


def main():
    rows = []
    for name, who, src, pred_src, pub, note in ASSETS:
        s24, n24 = val_score(name, 2024)
        s22, n22 = val_score(name, 2022)
        rows.append({
            "등록명": name, "팀원": who, "원 출처": src,
            "val 예측 출처": pred_src,
            "제출 Public": pub,
            "val2024 (sj 재계산)": s24, "val2024 행수": n24,
            "val2022 (sj 재계산)": s22, "val2022 행수": n22,
            "비고": note,
        })
    df = pd.DataFrame(rows)

    # ── 상관 · 결합 판정 ───────────────────────────────────────────────────
    L = load_labels(2024)
    y = L["y"].to_numpy(float)
    have = {}
    for name in df["등록명"]:
        p = PT / "val" / ("%s_2024.csv" % name)
        if p.exists():
            d = pd.read_csv(p)[["row_id", "pred"]]
            j = L[["row_id"]].merge(d, on="row_id", how="left")
            have[name] = j["pred"].to_numpy(float)
    names = list(have)
    C = np.corrcoef(np.column_stack([have[n] for n in names]).T)
    corr = pd.DataFrame(C.round(4), index=names, columns=names)

    # ── val 계산 방법 ──────────────────────────────────────────────────────
    method = pd.DataFrame([
        ("폴드 정의", "val2024 = train.csv 의 season==2024 · val2022 = season==2022"),
        ("행수", "val2024 253,507행 · val2022 247,472행 (train.csv 원본 순서)"),
        ("지표", "BSS = 100000 x (1 - Brier / (r(1-r))),  r = 그 폴드의 실제 성공률"),
        ("행 정합", "예측을 row_id 로 병합한다 — 행 순서에 의존하지 않는다"),
        ("학습 범위", "각 폴드의 검증 시즌은 학습에 **전혀 쓰지 않는다** (fit: season < fold)"),
        ("★ 재계산 이유",
         "팀원 자체보고를 그대로 옮기면 서로 비교가 안 된다. 같은 폴드·같은 식·같은 "
         "행 정합으로 다시 재야 결합 가중 w*=M^-1A 를 적합할 수 있다"),
        ("★ 자체보고와의 차이",
         "yn 자체보고 880.79 vs 여기 815.6 — 여기 값은 combine 규격 raw 판이다. "
         "두 숫자를 섞지 않는다"),
        ("★ 오염 주의 1 — eval_set",
         "hw 공유분은 eval_set=(x_val,y_val) + use_best_model 로 **검증 라벨을 "
         "조기종료에 썼다**. sj 가 그것을 뺀 정직본(hw_v12_honest)을 다시 만들었다"),
        ("★ 오염 주의 2 — calib",
         "sj 쪽 run_arm.calib 은 평가 폴드 라벨로 로짓 스케일을 고른다. 이 표의 "
         "sj 값은 그 경로를 거치지 않은 것이거나, 배포 순서(학습 유래 상수)로 낸 것이다"),
        ("결합 판정 규약",
         "합=1 · 비음수 · 월전방분할 양방향(월3~6 <-> 월7~10). "
         "§31 은 fit(2022)->동결->eval(2024) 도 요구한다"),
    ], columns=["항목", "내용"])

    # ── 결합 판정 요약 ─────────────────────────────────────────────────────
    verdict = pd.DataFrame([
        ("cw + sj3way", "0.70 / 0.30", "채택", "정방향 736.2 · 역방향 1076.7 — 최고"),
        ("+ hw", "0.00", "기각", "역방향 1068.1 로 하락. F 부분군에선 hw 가 +152 앞서지만 전이 안 됨"),
        ("+ yn", "0.00", "기각", "sj3way 와 ρ0.9572 — 정보 중복"),
        ("+ ye", "0.00", "기각", "ρ(cw) 0.7952 로 팀 최저인데도 0. 단독 574.7 로 신호 부족"),
        ("+ cw_v17_base", "—", "기각", "sj 현행과 ρ0.99 — 사실상 같은 모델"),
    ], columns=["조합", "가중", "판정", "근거"])

    frows = []
    for name, (src, comp, note) in FEATS.items():
        fp = PT / "features" / ("%s.txt" % name)
        n = None
        if fp.exists():
            n = sum(1 for ln in open(fp, encoding="utf-8")
                    if ln.strip() and not ln.startswith("#"))
        frows.append({"모델": name, "피처 수": n, "완전성": comp,
                      "출처": src, "비고": note,
                      "파일": "features/%s.txt" % name if fp.exists() else "없음"})
    fdf = pd.DataFrame(frows)

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="1_팀원자산", index=False)
        fdf.to_excel(w, sheet_name="5_피처목록", index=False)
        method.to_excel(w, sheet_name="2_val계산방법", index=False)
        corr.to_excel(w, sheet_name="3_상관행렬(val2024)")
        verdict.to_excel(w, sheet_name="4_결합판정", index=False)
        for sh, widths in (("1_팀원자산", [18, 14, 46, 34, 16, 18, 13, 18, 13, 70]),
                           ("2_val계산방법", [22, 95]),
                           ("4_결합판정", [18, 14, 10, 60]),
                           ("5_피처목록", [16, 10, 12, 52, 78, 26])):
            ws = w.sheets[sh]
            for i, wd in enumerate(widths, 1):
                ws.column_dimensions[ws.cell(1, i).column_letter].width = wd
            ws.freeze_panes = "A2"
        ws = w.sheets["3_상관행렬(val2024)"]
        ws.column_dimensions["A"].width = 20
        ws.freeze_panes = "B2"

    print("→ %s" % OUT)
    print(df[["등록명", "팀원", "제출 Public", "val2024 (sj 재계산)",
              "val2022 (sj 재계산)"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
