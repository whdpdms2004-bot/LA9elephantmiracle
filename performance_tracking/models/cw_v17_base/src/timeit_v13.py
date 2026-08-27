# -*- coding: utf-8 -*-
"""추론 시간 실측 — 10분 제한을 넘기면 0점이다.

평가 서버는 245,789 행을 10분 안에 처리해야 한다. v13 은 처음으로 딥러닝이 들어가고
CatBoost 도 3시드라, 계산이 아니라 **실제로 24만 행을 돌려서** 재야 한다.

방법
    train.csv 에서 245,789 행을 뽑아 test.csv 모양으로 만들고,
    패키징된 _v13/script.py 를 그대로 실행한다.

평가 서버 보정
    GPU   L4 (121 TFLOPS fp16)  vs  RTX 4090 (~330)   → 약 2.7배 느림
    CPU   6 vCPU                vs  로컬 (보통 12~24) → 약 2~3배 느림
    안전하게 CPU 3.0배 / GPU 3.0배 를 곱해 판정한다.

실행:
    python timeit_v13.py
"""

import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PITCH_DATA_DIR", os.path.join(HERE, "..", "open", "data"))
STAGE = os.path.join(HERE, "_v13")
N_TEST = 245789
FACTOR = 3.0
LIMIT = 600.0


def main():
    if not os.path.exists(os.path.join(STAGE, "script.py")):
        sys.exit("없음: _v13/script.py — 먼저 python make_v13.py 를 돌리세요.")

    print("가짜 test.csv 생성 (%d행)..." % N_TEST, flush=True)
    head = pd.read_csv(os.path.join(DATA, "test.csv"), encoding="utf-8-sig", nrows=1)
    cols = [c for c in head.columns]
    tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                     usecols=[c for c in cols if c != "row_id"] + ["season"])
    tr = tr[tr.season == 2024]
    if len(tr) < N_TEST:
        tr = pd.concat([tr] * (N_TEST // len(tr) + 1), ignore_index=True)
    tr = tr.iloc[:N_TEST].copy()
    tr["season"] = 2025
    tr["row_id"] = ["TEST_%06d" % (i + 1) for i in range(len(tr))]
    tr = tr[cols]
    d = os.path.join(STAGE, "data")
    os.makedirs(d, exist_ok=True)
    tr.to_csv(os.path.join(d, "test.csv"), index=False, encoding="utf-8")
    pd.DataFrame({"row_id": tr.row_id, "control_success": 0.5}).to_csv(
        os.path.join(d, "sample_submission.csv"), index=False, encoding="utf-8")
    print("  완료", flush=True)

    print("\n실행 중... (script.py 를 그대로)", flush=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, "script.py"], cwd=STAGE,
                       capture_output=True, text=True)
    el = time.time() - t0
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr[-3000:])
        sys.exit("실행 실패")

    # 단계별 누적 시간 파싱
    marks = re.findall(r"(features=\d+|Inference \w+|Saved)[^\n]*?(\d+)초", p.stdout)
    print("=" * 62)
    print("[TIMEIT v13]   제한 600초 (10분), 245,789행")
    print("=" * 62)
    prev = 0.0
    for lab, sec in marks:
        s = float(sec)
        print("  %-22s 누적 %6.0f초   (구간 %5.0f초)" % (lab, s, s - prev))
        prev = s
    print("\n  로컬 실측        %6.0f초" % el)
    print("  평가서버 추정    %6.0f초   (×%.1f 보정)" % (el * FACTOR, FACTOR))
    print("  여유             %6.0f초   (%.0f%%)"
          % (LIMIT - el * FACTOR, 100 * (1 - el * FACTOR / LIMIT)))
    print()
    if el * FACTOR < LIMIT * 0.6:
        print("  판정: 충분히 안전. 그대로 제출 가능.")
    elif el * FACTOR < LIMIT * 0.85:
        print("  판정: 통과하지만 여유가 크지 않다. 시드를 줄일지 검토.")
    else:
        print("  판정: ★ 위험. FT 시드를 3→2 로 줄이거나 배치를 키워야 한다.")
    print("=" * 62)
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(os.path.join(STAGE, "output"), ignore_errors=True)


if __name__ == "__main__":
    main()
