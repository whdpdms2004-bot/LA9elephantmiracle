"""F1 최종 디렉터리의 행 독립성, 출력 계약, 대규모 추론 시간을 검증한다."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = HERE.parents[1] / "feature_campaign_1000"
DEFAULT_CANDIDATE = CAMPAIGN / "outputs" / "final_f1_cat_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--large-rows", type=int, default=0)
    return parser.parse_args()


def run(candidate: Path) -> tuple[pd.DataFrame, float]:
    started = time.time()
    subprocess.run([sys.executable, str(candidate / "script.py")],
                   cwd=candidate, check=True)
    elapsed = time.time() - started
    output = pd.read_csv(candidate / "output" / "submission.csv")
    return output, elapsed


def validate(test: pd.DataFrame, output: pd.DataFrame) -> None:
    if output.columns.tolist() != ["row_id", "control_success"]:
        raise AssertionError(output.columns.tolist())
    if not output["row_id"].equals(test["row_id"]):
        raise AssertionError("row_id order mismatch")
    probability = output["control_success"].to_numpy(np.float64)
    if not np.isfinite(probability).all():
        raise AssertionError("non-finite probability")
    if not ((probability >= 0.0) & (probability <= 1.0)).all():
        raise AssertionError("probability outside [0, 1]")


def main() -> None:
    args = parse_args()
    candidate = Path(args.candidate).resolve()
    test_path = candidate / "data" / "test.csv"
    original = pd.read_csv(test_path)
    report = {}
    try:
        original.to_csv(test_path, index=False)
        full, elapsed = run(candidate)
        validate(original, full)
        singles = []
        single_times = []
        for index in range(len(original)):
            one = original.iloc[[index]].copy()
            one.to_csv(test_path, index=False)
            output, seconds = run(candidate)
            validate(one.reset_index(drop=True), output.reset_index(drop=True))
            singles.append(float(output["control_success"].iloc[0]))
            single_times.append(seconds)
        difference = np.max(np.abs(
            np.asarray(singles) - full["control_success"].to_numpy(np.float64)))
        report["row_independence"] = {
            "rows": len(original),
            "max_abs_diff": float(difference),
            "passed": bool(difference < 1e-12),
            "full_elapsed_sec": elapsed,
            "single_elapsed_sec_mean": float(np.mean(single_times)),
        }
        if difference >= 1e-12:
            raise AssertionError(report["row_independence"])

        if args.large_rows:
            repeats = int(np.ceil(args.large_rows / len(original)))
            large = pd.concat([original] * repeats, ignore_index=True).iloc[:args.large_rows]
            large = large.copy()
            large["row_id"] = [f"BENCH_{index:06d}" for index in range(len(large))]
            large.to_csv(test_path, index=False)
            output, seconds = run(candidate)
            validate(large.reset_index(drop=True), output.reset_index(drop=True))
            report["large_benchmark"] = {
                "rows": len(large),
                "elapsed_sec": seconds,
                "limit_sec": 600,
                "passed": seconds < 600,
                "pred_mean": float(output["control_success"].mean()),
            }
            if seconds >= 600:
                raise AssertionError(report["large_benchmark"])
    finally:
        original.to_csv(test_path, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
