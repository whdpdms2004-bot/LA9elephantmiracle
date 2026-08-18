"""공식 raw data에서 F1 최종 가중치와 lookup을 순서대로 재생성한다."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
SJ = HERE.parent
REPO = SJ.parents[1]
DATA = REPO / "data"
MODEL_OPT = SJ / "experiment" / "model_optimization"
PITCHER_EMBED = SJ / "experiment" / "pitcher_embedding"
CLAUDE_SRC = SJ / "claude" / "src"
DEFAULT_OUTPUT = HERE / "outputs" / "final_f1_cat_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--skip-precompute", action="store_true",
        help="이미 생성·검증한 전처리 캐시가 있을 때만 사용한다.")
    parser.add_argument("--force-final", action="store_true")
    parser.add_argument("--large-rows", type=int, default=245789)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    args = parse_args()
    required = [DATA / "train.csv", DATA / "trackman_history.csv", DATA / "test.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"official data missing: {missing}")
    python = sys.executable
    started = time.time()

    if not args.skip_precompute:
        run([python, str(MODEL_OPT / "v2_temporal_features.py")])
        run([
            python,
            str(PITCHER_EMBED / "trackman500_cutoff.py"),
            "--cutoffs", "2020,2021,2022,2023,2024,2025",
            "--min-season-pitches", "500",
        ])
        run([python, str(PITCHER_EMBED / "build_trackman500_asof_features.py")])
        run([python, str(CLAUDE_SRC / "p5_failure_labels.py")])

    output = Path(args.output_dir).resolve()
    train_command = [
        python,
        str(HERE / "train_final_f1_cat.py"),
        "--output-dir", str(output),
    ]
    if args.force_final:
        train_command.append("--force")
    run(train_command)

    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA / "test.csv", data_dir / "test.csv")
    run([
        python,
        str(HERE / "verify_final_f1.py"),
        "--candidate", str(output),
        "--large-rows", str(args.large_rows),
    ])
    run([python, str(HERE / "verify_feature_artifacts.py")])
    print(f"complete in {time.time() - started:.1f}s -> {output}", flush=True)


if __name__ == "__main__":
    main()
