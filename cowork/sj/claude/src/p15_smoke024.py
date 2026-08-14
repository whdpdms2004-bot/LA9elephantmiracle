"""P13-C: submit_024.zip 6단계 검증 (AGENTS.md B1).

  1 ZIP 구조    루트 3개, 파일명 30자 미만, forward-slash, CRC
  2 금지 패턴   pickle/joblib, 절대경로, 인터넷 접근, test 행간 집계, 미래 시즌
  3 오프라인    서버와 같은 경로 구조로 실제 실행
  4 규모        245,789행 (실제 2024 + cold-start), 시간·메모리
  5 출력        row_id/control_success 순서, 행 수, finite, [0,1]
  6 행 독립성   predict(단독 행) == predict(전체)[i]
"""
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ZIP = (Path(sys.argv[1]) if len(sys.argv) > 1
       else ROOT / "submit" / "2026-08-14" / "submit_024.zip")
WORK = Path(os.environ.get(
    "SMOKE_DIR", r"C:\Users\isj67\AppData\Local\Temp\claude\smoke")) / ZIP.stem
N_ROWS = 245_789
fails = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}   {detail}", flush=True)
    if not cond:
        fails.append(name)


print("=" * 84)
print("1. ZIP 구조")
print("=" * 84)
zf = zipfile.ZipFile(ZIP)
names = zf.namelist()
roots = sorted({n.split("/")[0] for n in names})
check("루트 3개만", roots == ["model", "requirements.txt", "script.py"], str(roots))
check("파일명 30자 미만", len(ZIP.name) < 30, f"{len(ZIP.name)}자")
check("backslash 없음", all("\\" not in n for n in names))
check("CRC 정상", zf.testzip() is None)
check("크기 10GB 이하", ZIP.stat().st_size < 10 * 2**30,
      f"{ZIP.stat().st_size/2**20:.1f}MB")

print("=" * 84)
print("2. 금지 패턴 스캔")
print("=" * 84)
script = zf.read("script.py").decode("utf-8")
BAD = {
    "pickle": r"\bpickle\b|\bjoblib\b|\.pkl\b",
    "절대경로": r"['\"](?:/app/|/mnt/|C:\\\\)",
    "인터넷": r"requests\.|urllib|from_pretrained|hf_hub|http://|https://",
    "학습코드": r"\.fit\(|xgb\.train\(|train_test_split",
    "test 행간 집계": r"test\.groupby|test\[.{0,40}\]\.(?:rolling|expanding|cumsum)",
    "미래 시즌": r"season\s*==\s*2025|season\s*>=\s*2025",
}
for tag, pat in BAD.items():
    hits = [m.group(0) for m in re.finditer(pat, script)]
    check(f"{tag} 없음", not hits, str(hits[:3]))
check("성분 라벨 미포함", "y_middle" not in script and "y_reverse" not in script,
      "추론 경로에 라벨 유도 코드 없음")
check("경로 기준 __file__", "Path(__file__).resolve().parent" in script)

print("=" * 84)
print("3~4. 오프라인 실행 (245,789행)")
print("=" * 84)
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
zf.extractall(WORK)
(WORK / "data").mkdir()
(WORK / "output").mkdir()

train = pd.read_csv(ROOT / "data" / "train.csv")
cols = [c for c in train.columns if c != "control_success"]
v24 = train[train["season"] == 2024][cols].copy()
cold = train[train["season"] == 2019][cols].sample(
    n=max(0, N_ROWS - len(v24)), random_state=20260814).copy()
cold["asof_pitcher_n"] = 0
for c in cols:
    if c.startswith("asof_") and c.endswith("_rate"):
        cold[c] = np.nan
test = pd.concat([v24, cold], ignore_index=True).head(N_ROWS).copy()
test["season"] = 2025
test["row_id"] = [f"TEST_{i:07d}" for i in range(len(test))]
test.to_csv(WORK / "data" / "test.csv", index=False)
del train, v24, cold
check("test 행 수", len(test) == N_ROWS, f"{len(test):,}")

t0 = time.perf_counter()
proc = subprocess.run([sys.executable, "script.py"], cwd=WORK,
                      capture_output=True, text=True, timeout=1800)
elapsed = time.perf_counter() - t0
if proc.returncode != 0:
    print(proc.stdout[-3000:])
    print(proc.stderr[-3000:])
check("실행 성공", proc.returncode == 0, f"exit {proc.returncode}")
check("추론 시간", elapsed < 600, f"{elapsed:.1f}초 (로컬 24코어)")
print(f"       서버 6 vCPU 환산 추정 {elapsed*24/6:.0f}초 — 600초 한도 대비 "
      f"{elapsed*24/6/600*100:.0f}%")

print("=" * 84)
print("5. 출력")
print("=" * 84)
sub = pd.read_csv(WORK / "output" / "submission.csv")
check("컬럼 순서", list(sub.columns) == ["row_id", "control_success"],
      str(list(sub.columns)))
check("행 수", len(sub) == N_ROWS, f"{len(sub):,}")
check("row_id 일치", (sub["row_id"].to_numpy() == test["row_id"].to_numpy()).all())
p = sub["control_success"].to_numpy(float)
check("finite", np.isfinite(p).all())
check("[0,1] 범위", bool((p >= 0).all() and (p <= 1).all()),
      f"[{p.min():.6f}, {p.max():.6f}]")
print(f"       예측 평균 {p.mean():.6f}  sd {p.std():.6f}")

print("=" * 84)
print("6. 행 독립성")
print("=" * 84)
rng = np.random.default_rng(7)
idx = rng.choice(len(test), size=40, replace=False)
solo_dir = WORK.parent / "smoke024_solo"
if solo_dir.exists():
    shutil.rmtree(solo_dir)
shutil.copytree(WORK, solo_dir, ignore=shutil.ignore_patterns("output", "data"))
(solo_dir / "data").mkdir(exist_ok=True)
(solo_dir / "output").mkdir(exist_ok=True)
test.iloc[idx].to_csv(solo_dir / "data" / "test.csv", index=False)
r2 = subprocess.run([sys.executable, "script.py"], cwd=solo_dir,
                    capture_output=True, text=True, timeout=900)
if r2.returncode != 0:
    print(r2.stderr[-2000:])
check("부분 실행 성공", r2.returncode == 0)
if r2.returncode == 0:
    solo = pd.read_csv(solo_dir / "output" / "submission.csv")
    full = sub.set_index("row_id").loc[solo["row_id"]]["control_success"].to_numpy()
    d = float(np.max(np.abs(solo["control_success"].to_numpy() - full)))
    check("predict(부분) == predict(전체)", d < 1e-9, f"최대 절대차 {d:.3e}")

print("=" * 84)
print(f"결과: {len(fails)}건 실패" if fails else "결과: 전 항목 통과")
if fails:
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
