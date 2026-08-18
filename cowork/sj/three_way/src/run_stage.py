"""3WAY 단계 실행기 — 한 번에 한 Stage 씩, GPU 작업은 순차.

각 Stage 는 2~3시간 분량으로 끊어 두었다. 돌리고 결과를 본 뒤 다음을 정한다.
중간에 끊겨도 .npy 캐시로 이어진다.

    python run_stage.py --list             전 단계 개요
    python run_stage.py --stage 1 --dry    무엇이 돌지 확인 (GPU 불필요)
    python run_stage.py --stage 1          실행

계획 문서: ../STAGES.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
TW = SRC.parent
PY = sys.executable

STAGES = {
    1: {
        "name": "학습 방식 이식",
        "why": "1WAY 에서 통했는데 3WAY 에 없는 것들 (F행 가중치·짧은 등판·트리 파라미터·2차 상호작용)",
        "est": "약 2시간 / 96 fit",
        "jobs": [
            ("outside 전처리 스크리닝 (fold 2024)",
             ["screen_target.py", "--target", "outside"]),
            ("학습 방식 arm × 4타깃 (fold 2024)",
             ["train_arms.py", "--target", "middle,reverse,ball,outside",
              "--fold", "2024"]),
            ("학습 방식 arm × 4타깃 (fold 2023)",
             ["train_arms.py", "--target", "middle,reverse,ball,outside",
              "--fold", "2023"]),
            ("결합 재평가 (CPU)", ["combine.py"]),
        ],
    },
    2: {
        "name": "모델 계열 확장",
        "why": "1WAY 는 성분당 XGB+CatBoost 16모델. 3WAY 는 CatBoost 1개다",
        "est": "약 2.5시간 / 110 fit",
        "jobs": [
            ("XGB 전처리 재스크리닝 (계열마다 최적 전처리가 다를 수 있다)",
             ["screen_target.py", "--target", "middle,reverse,ball,outside",
              "--family", "xgboost"]),
            ("계열별 최적 조합 두 fold",
             ["train_arms.py", "--target", "middle,reverse,ball,outside",
              "--family", "both"]),
            ("결합 재평가 (CPU)", ["combine.py"]),
        ],
        "todo": "screen_target.py / train_arms.py 에 --family 추가 필요",
    },
    3: {
        "name": "조합 재탐색 + fold 2023 확인",
        "why": "S2 는 학습 방식·계열이 고정된 상태의 최적. 조건이 바뀌었으니 이동했을 수 있다",
        "est": "약 2시간 / 90 fit",
        "jobs": [
            ("타깃별 빔 재탐색 (Stage1·2 최적 위에서)",
             ["screen_target.py", "--target", "middle,reverse,ball,outside",
              "--beam", "3", "--rounds", "2"]),
            ("fold 2023 확인",
             ["screen_target.py", "--fold", "2023",
              "--target", "middle,reverse,ball,outside"]),
            ("결합 재평가 (CPU)", ["combine.py"]),
        ],
    },
    4: {
        "name": "시드 배깅",
        "why": "1WAY 는 성분당 16모델, 3WAY 는 1개. 가장 큰 남은 격차일 수 있다",
        "est": "약 3시간",
        "jobs": [],
        "blocked": "배깅 보류 중 (2026-08-19 지시). 해제 후 진행",
    },
    5: {
        "name": "앙상블",
        "why": "모델링이 끝나면 조합별로 반드시 하는 단계. 취소가 아니라 순서의 문제",
        "est": "미정",
        "jobs": [],
        "blocked": "Stage 1~4 완료 후",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.list or not args.stage:
        print(f"3WAY 단계 (계획: {TW / 'STAGES.md'}){chr(10)}")
        for k, s in STAGES.items():
            flag = ""
            if s.get("blocked"):
                flag = f"   [보류] {s['blocked']}"
            elif s.get("todo"):
                flag = f"   [준비필요] {s['todo']}"
            print(f"  Stage {k}  {s['name']:<22}{s['est']:<20}{flag}")
            print(f"           {s['why']}")
        print(f"{chr(10)}  python run_stage.py --stage 1 --dry")
        return

    s = STAGES.get(args.stage)
    if not s:
        raise SystemExit(f"모르는 stage: {args.stage}")
    if s.get("blocked"):
        raise SystemExit(f"Stage {args.stage} 는 보류 중: {s['blocked']}")
    if s.get("todo") and not args.dry:
        raise SystemExit(f"Stage {args.stage} 준비 미완: {s['todo']}")

    print(f"{'=' * 96}")
    print(f"Stage {args.stage}  {s['name']}   ({s['est']})")
    print(f"  {s['why']}")
    print(f"{'=' * 96}{chr(10)}")
    if not s["jobs"]:
        print("  실행할 작업이 없다."); return

    t0 = time.time()
    for i, (label, argv) in enumerate(s["jobs"], 1):
        cmd = [PY, str(SRC / argv[0])] + argv[1:]
        if args.dry:
            cmd.append("--dry")
        print(f"{chr(10)}{'-' * 96}")
        print(f"[{i}/{len(s['jobs'])}] {label}   (누적 {(time.time()-t0)/60:.0f}분)")
        print(f"  $ {' '.join(cmd[1:])}")
        print(f"{'-' * 96}", flush=True)
        r = subprocess.run(cmd, cwd=str(SRC))
        print(f"  종료 코드 {r.returncode}", flush=True)
        if r.returncode != 0:
            print(f"  ! 실패 — 다음 작업으로 넘어간다 (캐시가 남아 재실행 가능)")
    print(f"{chr(10)}Stage {args.stage} 완료   총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
