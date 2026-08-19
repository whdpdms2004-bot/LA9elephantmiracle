"""3WAY 디렉터리 또는 ZIP의 제출 규격, 행 독립성, 실행 시간을 검증한다."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TW = HERE.parent
SJ = TW.parent
REPO = SJ.parents[1]
DEFAULT_CANDIDATE = SJ / "submit" / "2026-08-19" / "submit_037.zip"
PATTERNS = {
    "cross_row": re.compile(
        r"groupby|rolling|cumsum|expanding|\.shift\(|\.rank\(|\.transform\("),
    "batch_statistics": re.compile(
        r"\.mean\(\)|\.std\(\)|\.median\(\)|quantile|normalize|distribution_match"),
    "network": re.compile(
        r"requests|urllib|httpx|socket|from_pretrained|hf_hub|download|api_key"),
    "training": re.compile(
        r"\.fit\(|\.train\(|\.partial_fit\(|backward\(|optimizer"),
    "absolute_path": re.compile(
        r"^\s*(/|[A-Za-z]:\\\\)|/home/|/workspace/|/app/"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--large-rows", type=int, default=245789)
    parser.add_argument("--keep-extracted", action="store_true")
    return parser.parse_args()


def run(candidate: Path, offline: bool = False) -> tuple[pd.DataFrame, float]:
    started = time.time()
    if offline:
        code = (
            "import runpy,socket,sys;"
            "deny=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network disabled'));"
            "socket.socket.connect=deny;socket.create_connection=deny;"
            "runpy.run_path(sys.argv[1],run_name='__main__')")
        command = [sys.executable, "-c", code, str(candidate / "script.py")]
    else:
        command = [sys.executable, str(candidate / "script.py")]
    subprocess.run(command, cwd=candidate, check=True)
    elapsed = time.time() - started
    output = pd.read_csv(candidate / "output" / "submission.csv")
    return output, elapsed


def validate(test: pd.DataFrame, output: pd.DataFrame) -> None:
    if output.columns.tolist() != ["row_id", "control_success"]:
        raise AssertionError(output.columns.tolist())
    if output["row_id"].astype(str).tolist() != test["row_id"].astype(str).tolist():
        raise AssertionError("row_id order mismatch")
    probability = output["control_success"].to_numpy(np.float64)
    if not np.isfinite(probability).all():
        raise AssertionError("non-finite probability")
    if not ((probability >= 0.0) & (probability <= 1.0)).all():
        raise AssertionError("probability outside [0, 1]")


def scan_sources(candidate: Path) -> dict:
    report = {}
    sources = [candidate / "script.py", *sorted((candidate / "model").glob("*.py"))]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        hits = {}
        for label, pattern in PATTERNS.items():
            rows = []
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    rows.append({"line": number, "text": line.strip()})
            hits[label] = rows
        report[str(path.relative_to(candidate)).replace("\\", "/")] = hits
    return report


def prepare_candidate(path: Path, root: Path) -> tuple[Path, dict]:
    if path.is_dir():
        return path, {"kind": "directory"}
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        top_level = sorted({name.split("/", 1)[0] for name in names})
        if top_level != ["model", "requirements.txt", "script.py"]:
            raise AssertionError(top_level)
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"CRC failure: {bad_file}")
        if any("\\" in name for name in names):
            raise AssertionError("backslash ZIP path")
        archive.extractall(root)
    return root, {
        "kind": "zip", "top_level": top_level, "crc_passed": True,
        "file_count": len(names), "compressed_bytes": path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate).resolve()
    temp_context = tempfile.TemporaryDirectory(prefix="three_way_verify_")
    temp_root = Path(temp_context.name)
    try:
        candidate, package_report = prepare_candidate(candidate_path, temp_root)
        data_dir = candidate / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "data" / "test.csv", data_dir / "test.csv")
        test_path = data_dir / "test.csv"
        original = pd.read_csv(test_path)
        report = {
            "candidate": str(candidate_path),
            "package": package_report,
            "source_scan": scan_sources(candidate),
        }
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
            difference = float(np.max(np.abs(
                np.asarray(singles) - full["control_success"].to_numpy(np.float64))))
            report["row_independence"] = {
                "rows": len(original),
                "max_abs_diff": difference,
                "passed": difference < 1e-12,
                "full_elapsed_sec": elapsed,
                "single_elapsed_sec_mean": float(np.mean(single_times)),
            }
            if difference >= 1e-12:
                raise AssertionError(report["row_independence"])

            original.to_csv(test_path, index=False)
            offline_output, offline_seconds = run(candidate, offline=True)
            validate(original, offline_output)
            report["offline_smoke"] = {
                "passed": True, "elapsed_sec": offline_seconds}

            if args.large_rows:
                repeats = int(np.ceil(args.large_rows / len(original)))
                large = pd.concat([original] * repeats, ignore_index=True).iloc[
                    :args.large_rows].copy()
                large["row_id"] = [f"BENCH_{index:06d}" for index in range(len(large))]
                large.to_csv(test_path, index=False)
                output, seconds = run(candidate)
                validate(large.reset_index(drop=True), output.reset_index(drop=True))
                report["large_benchmark"] = {
                    "rows": len(large),
                    "elapsed_sec": seconds,
                    "limit_sec": 600,
                    "passed": seconds < 600,
                    "prediction_mean": float(output["control_success"].mean()),
                }
                if seconds >= 600:
                    raise AssertionError(report["large_benchmark"])
        finally:
            original.to_csv(test_path, index=False)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.keep_extracted and candidate_path.is_file():
            destination = TW / "outputs" / "package_smoke_037"
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copytree(candidate, destination)
            print(f"kept extracted package at {destination}")
    finally:
        temp_context.cleanup()


if __name__ == "__main__":
    main()
