from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TARGET = "control_success"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--rows", type=int, default=245789)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.resolve()
    if not (source / "model").is_dir() or not (source / "script.py").is_file():
        raise FileNotFoundError(source)
    sample = pd.read_csv(ROOT / "data" / "test.csv")
    repeated = sample.iloc[np.arange(args.rows) % len(sample)].reset_index(drop=True)
    repeated["row_id"] = [f"SIM_{index:06d}" for index in range(args.rows)]

    with tempfile.TemporaryDirectory(prefix="lg_inference_benchmark_") as temp:
        stage = Path(temp)
        shutil.copytree(source / "model", stage / "model")
        shutil.copy2(source / "script.py", stage / "script.py")
        (stage / "data").mkdir()
        repeated.to_csv(stage / "data" / "test.csv", index=False)
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(stage / "script.py")],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - started
        prediction = pd.read_csv(stage / "output" / "submission.csv")
        if len(prediction) != args.rows:
            raise RuntimeError(f"Unexpected rows: {len(prediction)}")
        if not prediction[TARGET].between(0, 1).all():
            raise RuntimeError("Prediction outside [0, 1]")
        if not np.isfinite(prediction[TARGET]).all():
            raise RuntimeError("Non-finite prediction")
        report = {
            "source": str(source),
            "rows": args.rows,
            "elapsed_sec": elapsed,
            "rows_per_sec": args.rows / elapsed,
            "pred_min": float(prediction[TARGET].min()),
            "pred_max": float(prediction[TARGET].max()),
            "pred_mean": float(prediction[TARGET].mean()),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
