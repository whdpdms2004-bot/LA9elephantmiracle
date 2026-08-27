# -*- coding: utf-8 -*-
"""제출 zip 을 `cowork/RULES.md` §5 규격에 대조한다. **제출본을 만들 때마다 돌린다.**

규칙을 손으로 확인하면 빠뜨린다. 위반 하나가 **설치 오류(차감 없음)** 또는
**제출 오류(차감됨)** 로 이어지므로 자동화한다.

## 검사 항목 (RULES.md §5 · §0)

| # | 항목 | 위반 시 |
|---|---|---|
| 1 | 최상위가 `script.py` · `requirements.txt` · `model/` **뿐** | 설치 오류 |
| 2 | `script.py` 파일명 정확히 일치 (대소문자) | 설치 오류 |
| 3 | **하드코딩된 절대경로 없음** (`C:\\`, `/home/`, `/mnt/` …) | 제출 오류 |
| 4 | 파일명에 **공백·한글 없음** | 설치 오류 |
| 5 | `script.py` 에 **학습 코드 없음** (`.fit(` · `train(` · optimizer …) | 규정 위반 |
| 6 | `model/` 비어 있지 않음 · 빈 디렉터리 없음 | 설치 오류 |
| 7 | **인터넷 접근 없음** (`requests` · `urllib` · `from_pretrained` …) | 규정 위반 |
| 8 | `requirements.txt` 가 기본 설치 패키지를 덮어쓰지 않음 | 설치 오류 |
| 9 | 압축/해제 크기 (10GB / 32GB) | 설치 오류 |

★ **행 독립성(§2)은 이 도구가 판정하지 못한다.** 정적 검사로는 `groupby` 유무만
보이고, 그것이 test 행을 가로지르는지는 코드를 읽어야 안다. 별도로 확인한다.

    python check_submit.py --zip <submit.zip>
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ALLOWED_TOP = {"script.py", "requirements.txt", "model"}

# 평가 서버 기본 설치 (RULES.md §6). 이걸 requirements 에 넣으면 버전 충돌 위험.
PREINSTALLED = {"torch", "pandas", "numpy", "scipy", "scikit-learn", "sklearn",
                "joblib", "transformers", "tqdm", "pyyaml"}

ABS_PATH = re.compile(r"""(?:['"])(?:[A-Za-z]:[\\/]|/home/|/mnt/|/Users/|/root/)""")
NET = re.compile(r"\b(?:requests\.|urllib\.request|urlopen|from_pretrained|"
                 r"hf_hub_download|socket\.|boto3|http[s]?://)")
TRAIN = re.compile(r"\.fit\s*\(|\.train\s*\(|optimizer|loss\.backward|"
                   r"train_test_split|GridSearch|\.partial_fit\s*\(")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    a = ap.parse_args()
    z = zipfile.ZipFile(a.zip)
    names = [n for n in z.namelist() if not n.endswith("/")]
    dirs = [n for n in z.namelist() if n.endswith("/")]
    fails, warns = [], []

    print("=" * 74)
    print("제출 규격 검사 — %s" % Path(a.zip).name)
    print("=" * 74)

    # 1·2. 최상위 구조
    top = {n.split("/")[0] for n in z.namelist()}
    extra = top - ALLOWED_TOP
    print("\n[1] 최상위 구조")
    print("    %s" % sorted(top))
    if extra:
        fails.append("최상위에 규격 외 항목: %s" % sorted(extra))
    for req in ("script.py", "requirements.txt"):
        if req not in names:
            fails.append("%s 없음" % req)
    if not any(n.startswith("model/") for n in names):
        fails.append("model/ 이 비어 있다")
    print("    script.py %s · requirements.txt %s · model/ %d개 파일"
          % ("O" if "script.py" in names else "X",
             "O" if "requirements.txt" in names else "X",
             sum(1 for n in names if n.startswith("model/"))))

    # 3·4·5·7. 파이썬 소스 정적 검사
    pys = [n for n in names if n.endswith(".py")]
    print("\n[2] 파이썬 소스 %d개 정적 검사" % len(pys))
    for n in pys:
        src = z.read(n).decode("utf-8", "replace")
        # 주석·문자열 안의 예시까지 잡히면 시끄러우므로 라인 단위로 보고 주석은 뺀다
        lines = [ln.split("#")[0] for ln in src.splitlines()]
        body = "\n".join(lines)
        if ABS_PATH.search(body):
            hits = [ln.strip()[:90] for ln in lines if ABS_PATH.search(ln)][:2]
            fails.append("절대경로: %s → %s" % (n, hits))
        if NET.search(body):
            hits = [ln.strip()[:90] for ln in lines if NET.search(ln)][:2]
            fails.append("인터넷 접근: %s → %s" % (n, hits))
        if n == "script.py" and TRAIN.search(body):
            hits = [ln.strip()[:90] for ln in lines if TRAIN.search(ln)][:2]
            fails.append("최상위 script.py 에 학습 코드: %s" % hits)
        elif TRAIN.search(body):
            hits = [ln.strip()[:80] for ln in lines if TRAIN.search(ln)][:1]
            warns.append("학습 관련 표현 (멤버 스크립트): %s → %s" % (n, hits))
    print("    절대경로·인터넷·학습코드 검사 완료")

    # 4. 파일명
    print("\n[3] 파일명")
    bad = [n for n in z.namelist() if " " in n or re.search(r"[가-힣]", n)]
    if bad:
        fails.append("공백/한글 파일명: %s" % bad[:3])
    print("    공백·한글 %s" % ("없음" if not bad else "★있음"))

    # 6. 빈 디렉터리
    empty = [d for d in dirs if not any(n.startswith(d) and n != d for n in names)]
    if empty:
        fails.append("빈 디렉터리: %s" % empty[:3])
    print("\n[4] 빈 디렉터리 %s" % ("없음" if not empty else "★%s" % empty[:3]))

    # 8. requirements
    print("\n[5] requirements.txt")
    req = z.read("requirements.txt").decode("utf-8", "replace") if "requirements.txt" in names else ""
    for ln in req.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        pkg = re.split(r"[=<>!~\[]", ln)[0].strip().lower()
        mark = "  ← 기본 설치. 버전 충돌 위험" if pkg in PREINSTALLED else ""
        if mark:
            warns.append("requirements 에 기본 설치 패키지: %s" % ln)
        print("    %-30s%s" % (ln, mark))
    if not req.strip():
        warns.append("requirements.txt 가 비어 있다 (기본 패키지만 쓰면 정상)")

    # 9. 크기
    comp = Path(a.zip).stat().st_size / 1e9
    raw = sum(z.getinfo(n).file_size for n in names) / 1e9
    print("\n[6] 크기  압축 %.2fGB / 10GB · 해제 %.2fGB / 32GB" % (comp, raw))
    if comp > 10:
        fails.append("압축 크기 초과 %.1fGB" % comp)
    if raw > 32:
        fails.append("해제 크기 초과 %.1fGB" % raw)

    print("\n" + "=" * 74)
    for w in warns:
        print("  주의  %s" % w)
    if fails:
        for f in fails:
            print("  ★위반 %s" % f)
        print("\n판정: **위반 %d건** — 고치기 전에 제출하지 않는다" % len(fails))
        return 1
    print("판정: 규격 통과 (주의 %d건)" % len(warns))
    print("\n※ 행 독립성(RULES §2)은 정적 검사로 판정할 수 없다. 별도 확인할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
