"""남은 실험을 하나씩 순차 실행한다. GPU 작업은 절대 겹치지 않는다.

배경
    이 세션에서 GPU 작업을 겹쳐 돌려 두 번 손해를 봤다.
        1) V40 학습 중 스모크 테스트 -> 추론이 1800초 타임아웃으로 죽음
        2) V66 학습 중 V69 동시 실행 -> 15.8/16.4GB 포화, 둘 다 결과 없이 중단
    CatBoost GPU 학습이 메모리를 크게 잡는 것을 계산에 넣지 않은 것이 원인이다.
    규칙: GPU 작업은 한 번에 하나. 이 스크립트가 그것을 강제한다.

순서 (사용자 지정)
    V66  TrackMan 물리 요약을 성분 모델에      5 arm x 2 fold   ~50분
    V69  attention 멤버 (FT-Transformer)      3시드 x 2 fold   ~20분 (미측정)
    V70  attention 피처 (cross-attention)     att 9회 + GBDT 6회 ~40분 (미측정)
    V71  2차 확장 (제곱/곱/차비)               5 arm x 2 fold   ~47분
    V72  학습 방식 스윕 (재개)                 40 arm           ~6.2시간

각 단계는 실패해도 다음으로 넘어간다. 로그는 outputs/queue_<이름>.log 에 남는다.
"""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
LOGDIR = SRC.parent / "outputs"
LOGDIR.mkdir(exist_ok=True)
PY = sys.executable

JOBS = [
    ("v66_trackman", ["v66_trackman_components.py"]),
    ("v69_attention_member", ["v69_attention_member.py"]),
    ("v70_attention_features", ["v70_attention_features.py"]),
    ("v71_second_order", ["v71_second_order.py"]),
    ("v72_learning_sweep", ["v72_learning_sweep.py"]),
]

t0 = time.time()
for name, args in JOBS:
    log = LOGDIR / f"queue_{name}.log"
    print(f"{chr(10)}{'='*70}{chr(10)}[{time.strftime('%H:%M:%S')}] 시작 {name}"
          f"   (누적 {(time.time()-t0)/60:.0f}분){chr(10)}{'='*70}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run([PY] + [str(SRC / a) for a in args],
                           stdout=fh, stderr=subprocess.STDOUT, text=True)
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
    print(chr(10).join(tail), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {name} 종료 코드 {p.returncode}   "
          f"로그 {log}", flush=True)

print(f"{chr(10)}전체 완료   총 {(time.time()-t0)/60:.0f}분")
