# -*- coding: utf-8 -*-
"""[S8] 제출 전 3관문 — **전체 파이프라인을 실제로 돌려서** 판정한다.

계획서 §32 의 마지막 배치가 요구한 것:

    S8  08-31 밤   3관문 (timeit / check_rules / verify) + 행 독립성 + 제출

`check_submit.py` 는 zip 을 **정적으로** 검사할 뿐이고, 스스로
"행 독립성(RULES §2)은 정적 검사로 판정할 수 없다" 고 적어뒀다.
여기서 그것을 **실행해서** 판정한다.

## 세 관문

| # | 관문 | 무엇을 잡나 |
|---|---|---|
| 1 | **행 독립성** | test 에 그 행 1개만 있을 때와 전체가 있을 때 예측이 같은가 (규정 판정 기준 그대로) |
| 2 | **베이스율** | 최종 평균이 `target_rate` 근처인가. K≈401,000 이라 0.01 어긋나면 40점이 날아간다 |
| 3 | **시간** | 전체 파이프라인 실측. 서버 제한 10분 |

## test.csv 를 건드리지 않는다

`data/test.csv` 는 저장소에 커밋된 **5행 형식 샘플**이다 (실제 테스트셋은 평가
서버에 있고 규모는 비공개다). 여기서는 그 파일의 **열 이름만** 읽고, 본체는
`train.csv` 의 2024 행으로 만든다 — `timeit_v13.py` 와 같은 방식이다.

**행 수를 가정하지 않는다.** val2024 의 253,507 행을 쓰고 결과는 **비율**로 보고한다.

    python s8_gates.py --zip <submit.zip> [--rows 253507]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
DATA = ROOT / "data"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(m):
    print(m, flush=True)


def build_mock_test(stage: Path, n_rows: int):
    """`train.csv` 의 2024 행으로 test 모양 파일을 만든다.

    열 목록은 커밋된 5행 샘플의 **헤더에서만** 가져온다.
    """
    head = pd.read_csv(DATA / "test.csv", encoding="utf-8-sig", nrows=1)
    cols = [c.strip("﻿") for c in head.columns]
    log("  test 열 %d개 (5행 샘플 헤더에서)" % len(cols))
    it = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig", chunksize=400000)
    parts = []
    got = 0
    for ch in it:
        ch.columns = [c.strip("﻿") for c in ch.columns]
        sub = ch[ch["season"] == 2024]
        if len(sub):
            parts.append(sub)
            got += len(sub)
        if got >= n_rows:
            break
    df = pd.concat(parts).head(n_rows)
    miss = [c for c in cols if c not in df.columns]
    if miss:
        sys.exit("[중단] train 에 없는 test 열: %s" % miss[:5])
    out = df[cols].copy()
    d = stage / "data"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({cols[0]: out[cols[0]], "control_success": 0.5}).to_csv(
        d / "sample_submission.csv", index=False, encoding="utf-8")
    log("  모사 test %s행 · %d열 → %s" % (f"{len(out):,}", len(cols), d / "test.csv"))
    return out, cols


def run_pipeline(stage: Path):
    t0 = time.time()
    r = subprocess.run([sys.executable, "script.py"], cwd=str(stage),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    dt = time.time() - t0
    if r.returncode != 0:
        log(r.stdout[-3000:])
        log(r.stderr[-3000:])
        sys.exit("[중단] 파이프라인이 실패했다 (exit %d)" % r.returncode)
    sub = stage / "output" / "submission.csv"
    if not sub.exists():
        cand = list(stage.rglob("submission.csv"))
        if not cand:
            log(r.stdout[-2000:])
            sys.exit("[중단] submission.csv 가 안 나왔다")
        sub = cand[0]
    return pd.read_csv(sub), dt, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(FINAL / "submit" / "submit_sj_stdmlp.zip"))
    ap.add_argument("--rows", type=int, default=253507,
                    help="모사 test 행수. **실제 테스트 규모는 비공개다** — 비율로만 본다")
    ap.add_argument("--probe", type=int, default=5, help="행 독립성 표본 수")
    ap.add_argument("--work", default="")
    a = ap.parse_args()

    stage = Path(a.work) if a.work else (FINAL / "work" / "_s8")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    log("=" * 78)
    log("[S8] 제출 전 3관문 — %s" % Path(a.zip).name)
    log("=" * 78)
    log("\n[0] zip 전개")
    with zipfile.ZipFile(a.zip) as z:
        z.extractall(stage)
    log("  → %s" % stage)

    full, cols = build_mock_test(stage, a.rows)

    log("\n[3] 전체 실행 (시간 관문)")
    pred, dt, out = run_pipeline(stage)
    log("  %s행 %.0f초" % (f"{len(pred):,}", dt))
    idcol = pred.columns[0]
    valcol = pred.columns[1]
    log("  예측 %s행 · 평균 %.6f · 범위 [%.4f, %.4f]"
        % (f"{len(pred):,}", pred[valcol].mean(), pred[valcol].min(), pred[valcol].max()))

    # ── 관문 2. 베이스율 ────────────────────────────────────────────────────
    par = json.loads((stage / "model" / "cw" / "model" / "params.json").read_text("utf-8"))
    tgt = float(par.get("target_rate", 0.47469465355297163))
    bw = json.loads((stage / "model" / "blend_weights.json").read_text("utf-8"))
    log("\n[2] 베이스율 관문")
    log("  최종 평균 %.6f · 목표(target_rate) %.6f · 차 %+.6f"
        % (pred[valcol].mean(), tgt, pred[valcol].mean() - tgt))
    # K = 1e5/(r(1-r)) ~= 401,000. 평균이 d 어긋나면 대략 K*d^2 점이 날아간다.
    d = abs(pred[valcol].mean() - tgt)
    log("  평균 이탈이 만드는 최대 손실 ~= %.1f점 (K=401,000 x 차^2)" % (401000 * d * d))
    log("  판정: %s" % ("통과 (0.01 미만)" if d < 0.01 else "★주의 — 0.01 이상 어긋난다"))

    # ── 관문 1. 행 독립성 ───────────────────────────────────────────────────
    log("\n[1] 행 독립성 관문 — 규정 판정 기준 그대로")
    log("  'test.csv 에 그 행 1개만 있을 때와 전체가 있을 때 예측값이 같아야 한다'")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(full), a.probe, replace=False)
    worst = 0.0
    for k, i in enumerate(idx):
        one = full.iloc[[i]]
        one.to_csv(stage / "data" / "test.csv", index=False, encoding="utf-8")
        pd.DataFrame({cols[0]: one[cols[0]], "control_success": 0.5}).to_csv(
            stage / "data" / "sample_submission.csv", index=False, encoding="utf-8")
        p1, _, _ = run_pipeline(stage)
        rid = one[cols[0]].iloc[0]
        vfull = float(pred.loc[pred[idcol] == rid, valcol].iloc[0])
        vone = float(p1[valcol].iloc[0])
        worst = max(worst, abs(vfull - vone))
        log("  %-16s 전체 %.10f · 단독 %.10f · 차 %.3e" % (rid, vfull, vone, abs(vfull - vone)))
    log("  최대절대차 %.3e  →  %s"
        % (worst, "통과" if worst < 1e-9 else
           ("주의 (수치 오차 수준)" if worst < 1e-5 else "★위반")))

    # ── 관문 3. 시간 ────────────────────────────────────────────────────────
    log("\n[3] 시간 관문")
    log("  모사 %s행에서 %.0f초 (%.2f ms/행)" % (f"{a.rows:,}", dt, 1000 * dt / a.rows))
    log("  ★실제 테스트 규모는 비공개다. 서버 실측 기준점: 제출1 이 388초/600초였고")
    log("   그때 cb 는 depth5·3시드, mlp 는 336입력이었다. 지금은 depth6 · mlp 352입력이다.")
    log("  로컬/서버 하드웨어가 다르므로 절대값이 아니라 **구성 간 비율**로만 판단한다.")

    log("\n" + "=" * 78)
    ok = (d < 0.01) and (worst < 1e-5)
    log("종합: %s" % ("3관문 통과" if ok else "★재검토 필요"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
