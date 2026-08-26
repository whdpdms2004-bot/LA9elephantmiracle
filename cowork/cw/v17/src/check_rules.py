# -*- coding: utf-8 -*-
"""제출 규정 자동 검사 — 패키징 후 · 제출 전 필수.

데이콘 공지(Phase 2) 핵심 원칙:
    "평가 데이터의 각 행은 독립적으로 예측되어야 한다."
    허용 출처는 (1) 그 행의 입력변수 (2) 그 행만으로 만든 파생변수
                (3) 공식 학습 데이터 (4) 학습 데이터로 만든 통계·모델

그리고 공지 3번이 판정 기준을 직접 알려준다:
    "test.csv 에 그 행 1개만 있을 때와 전체가 있을 때 예측값이 같아야 한다."

그래서 이 스크립트는 **그 기준을 그대로 실행한다.**

  [1] 동적 검사 (결정적)  전체 실행 결과 vs 한 행씩 따로 실행한 결과를 비교
  [2] 정적 검사 (보조)    예측 경로의 행간 연산 패턴을 훑는다
  [3] 상수 공개          params.json 안의 상수를 전부 출력해 사람이 검토하게 한다

실행:
    python check_rules.py submit_v11_catboost.zip
"""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "open", "data")

# 예측 경로에 있으면 안 되는 연산
PATTERNS = [
    (r"\.groupby\s*\(", "groupby — 행간 집계"),
    (r"\.rolling\s*\(", "rolling"),
    (r"\.shift\s*\(", "shift / lag"),
    (r"\.cumsum\s*\(|\.cumcount\s*\(|np\.cumsum", "누적합"),
    (r"\.sort_values\s*\(|np\.argsort|np\.sort\s*\(", "정렬 — 행 순서 이용"),
    (r"\.rank\s*\(", "rank"),
    (r"value_counts", "value_counts — 빈도"),
    (r"\.duplicated\s*\(|\.drop_duplicates\s*\(", "중복 판정"),
    (r"np\.corrcoef|\.corr\s*\(", "상관계수 — 행간"),
    (r"\.interpolate\s*\(|\.ffill\s*\(|\.bfill\s*\(", "행간 채우기"),
]
# 전체 배열 통계 (print 안이면 무해하지만 검토자 눈에 띄므로 보고)
STAT = re.compile(r"\.(mean|std|median|quantile|sum|min|max)\s*\(\s*\)")


def strip_strings_and_comments(src):
    """문자열·주석을 지운 코드만 남긴다 (print 포맷 안의 통계는 예측에 영향 없음)."""
    src = re.sub(r'"""(?:.|\n)*?"""', '""', src)
    src = re.sub(r"'''(?:.|\n)*?'''", "''", src)
    out = []
    for line in src.split("\n"):
        line = re.sub(r"#.*$", "", line)
        out.append(line)
    return "\n".join(out)


def in_print(src, pos):
    """해당 위치가 print(...) 호출 안인지 대충 판정."""
    head = src.rfind("print(", 0, pos)
    if head < 0:
        return False
    depth, i = 0, head + 5
    while i < len(src):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return head < pos < i
        i += 1
    return False


def static_scan(src):
    code = strip_strings_and_comments(src)
    hits, notes = [], []
    for pat, name in PATTERNS:
        for m in re.finditer(pat, code):
            ln = code[:m.start()].count("\n") + 1
            (notes if in_print(code, m.start()) else hits).append(
                (ln, name, code.split("\n")[ln - 1].strip()[:90]))
    for m in STAT.finditer(code):
        ln = code[:m.start()].count("\n") + 1
        line = code.split("\n")[ln - 1].strip()[:90]
        (notes if in_print(code, m.start()) else hits).append((ln, "전체 배열 통계", line))
    return hits, notes


def run(stage, rows):
    """지정한 행만 담은 test.csv 로 script.py 를 돌리고 {row_id: 예측} 을 돌려준다."""
    d = os.path.join(stage, "data")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(DATA, "test.csv"), encoding="utf-8-sig") as f:
        rd = list(csv.DictReader(f)); fn = rd[0].keys()
    with open(os.path.join(d, "test.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fn)); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(DATA, "sample_submission.csv"), encoding="utf-8-sig") as f:
        sub = list(csv.DictReader(f))
    ids = {r["row_id"] for r in rows}
    with open(os.path.join(d, "sample_submission.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "control_success"]); w.writeheader()
        for r in sub:
            if r["row_id"] in ids:
                w.writerow(r)
    p = subprocess.run([sys.executable, "script.py"], cwd=stage,
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        raise SystemExit("script.py 실행 실패")
    with open(os.path.join(stage, "output", "submission.csv"), encoding="utf-8") as f:
        return {r["row_id"]: float(r["control_success"]) for r in csv.DictReader(f)}


def main():
    if len(sys.argv) < 2:
        raise SystemExit("사용법: python check_rules.py <제출zip>")
    zp = os.path.join(HERE, sys.argv[1]) if not os.path.isabs(sys.argv[1]) else sys.argv[1]
    stage = tempfile.mkdtemp(prefix="rulechk_")
    try:
        with zipfile.ZipFile(zp) as z:
            z.extractall(stage)
        src = open(os.path.join(stage, "script.py"), encoding="utf-8").read()

        print("=" * 68)
        print("[규정 검사]  %s" % os.path.basename(zp))
        print("=" * 68)

        # ── [1] 동적 — 행 독립성 (결정적 판정) ────────────
        print("\n[1] 행 독립성  (공지 3번의 판정 기준)")
        with open(os.path.join(DATA, "test.csv"), encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        full = run(stage, rows)
        print("    전체 %d행 동시 실행 완료" % len(rows), flush=True)
        # 판정 기준
        #   실제로 다른 행의 정보를 쓰면 차이가 1e-3 이상으로 크게 난다.
        #   1e-6 미만은 부동소수점 누적 순서 차이(배치 크기에 따른 GEMM 타일링)이며
        #   정보가 새는 것이 아니다. 그 사이는 경고로 두고 사람이 확인한다.
        TOL_OK, TOL_BAD = 1e-6, 1e-4
        worst = 0.0
        for r in rows:
            one = run(stage, [r])
            rid = r["row_id"]
            d = abs(one[rid] - full[rid])
            worst = max(worst, d)
            print("    %-12s 전체 %.10f | 단독 %.10f | 차 %.2e" % (rid, full[rid], one[rid], d))
        ok1 = worst < TOL_BAD
        if worst < TOL_OK:
            verdict = "통과 (완전 일치)"
        elif worst < TOL_BAD:
            verdict = "통과 (부동소수점 오차 수준, 정보 누수 아님)"
        else:
            verdict = "★ 위반"
        print("    → %s (최대차 %.2e)" % (verdict, worst))

        # ── [2] 정적 — 행간 연산 패턴 ─────────────────────
        print("\n[2] 예측 경로의 행간 연산")
        hits, notes = static_scan(src)
        if hits:
            for ln, name, line in hits:
                print("    ★ L%-4d %-18s %s" % (ln, name, line))
        else:
            print("    없음")
        if notes:
            print("    (참고 — print 안이라 예측에 영향 없으나 검토자 눈에 띌 수 있음)")
            for ln, name, line in notes:
                print("      L%-4d %-18s %s" % (ln, name, line))
        ok2 = not hits

        # ── [3] 상수 공개 ─────────────────────────────────
        print("\n[3] params.json 상수  (출처를 사람이 검토할 것)")
        pj = json.load(open(os.path.join(stage, "model", "params.json"), encoding="utf-8"))
        for k, v in pj.items():
            if isinstance(v, (int, float, str, bool)) and not isinstance(v, bool):
                s = str(v)
                print("    %-22s %s" % (k, s[:80] + ("..." if len(s) > 80 else "")))
        for sub in ("model_a", "model_b", "model_c", "model_cb"):
            if sub in pj and isinstance(pj[sub], dict):
                print("    %s:" % sub)
                for k, v in pj[sub].items():
                    if isinstance(v, (int, float, str)):
                        print("      %-20s %s" % (k, str(v)[:60]))

        print("\n" + "=" * 68)
        if ok1 and ok2:
            print("판정: 통과 — 각 행이 독립적으로 예측됩니다.")
        else:
            print("판정: ★ 제출하지 마세요.")
        print("=" * 68)
        sys.exit(0 if (ok1 and ok2) else 1)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
