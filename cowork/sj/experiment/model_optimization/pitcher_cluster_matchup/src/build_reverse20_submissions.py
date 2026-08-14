"""Package and smoke-test the two reverse 20-seed submissions."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "submit" / "2026-08-13" / "submit_019.zip"
OUTPUT_DIR = ROOT / "submit" / "2026-08-13"
FINAL_DIR = (
    ROOT
    / "experiment"
    / "model_optimization"
    / "pitcher_cluster_matchup"
    / "final"
    / "reverse_seedbag20"
)
REPORT = (
    ROOT
    / "experiment"
    / "model_optimization"
    / "pitcher_cluster_matchup"
    / "reports"
    / "reverse20_submission_manifest.json"
)
TARGET = "control_success"
SEEDS = [
    17, 2026, 4099, 43, 97, 311, 503, 719, 887, 1237,
    1429, 1699, 1877, 2131, 2389, 2683, 3001, 3253, 3529, 3851,
]
SPECS = [
    {
        "number": 20,
        "scale": 0.55,
        "label": "reverse20_conservative_s055",
        "val2024_bss": 835.794539,
        "purpose": "20-seed effect with the previously deployed reverse scale",
    },
    {
        "number": 21,
        "scale": 0.40,
        "label": "reverse20_aggressive_s040",
        "val2024_bss": 836.502924,
        "purpose": "lower reverse scale matched to the downstream R/F corrections",
    },
]


def file_info(name: str, compress_type: int = ZIP_DEFLATED) -> ZipInfo:
    info = ZipInfo(name)
    info.date_time = (2026, 8, 13, 0, 0, 0)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    return info


def directory_info(name: str) -> ZipInfo:
    info = ZipInfo(name)
    info.date_time = (2026, 8, 13, 0, 0, 0)
    info.compress_type = ZIP_STORED
    info.external_attr = (0o755 << 16) | 0x10
    return info


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def reverse_specs() -> list[dict]:
    return [
        {
            "pitcher_lookup_file": "pitcher_lookup_2025.csv",
            "batter_lookup_file": f"reverse20_s{seed}_lookup.csv",
            "pair_table_file": f"reverse20_s{seed}_pair.csv",
            "ridge_file": f"reverse20_s{seed}_ridge.json",
            "seed": seed,
            "center_mode": "season x pitcher_hand x batter_hand x count_state",
            "batter_algorithm": "kmeans",
            "batter_k_by_hand": {"left": 4, "right": 6},
            "pair_smoothing": 1000,
            "recency_half_life": 1.0,
            "ridge_alpha": 1000.0,
        }
        for seed in SEEDS
    ]


def package(spec: dict) -> Path:
    destination = OUTPUT_DIR / f"submit_{spec['number']:03d}.zip"
    candidate = destination.with_suffix(".candidate.zip")
    if destination.exists() or candidate.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    if len(destination.name) >= 30:
        raise RuntimeError(f"Filename is too long: {destination.name}")
    with ZipFile(BASE) as archive:
        metadata = json.loads(archive.read("model/metadata.json"))
        old_reverse_files = set()
        for item in metadata["reverse_matchup_corrections"]:
            for key in ["batter_lookup_file", "pair_table_file", "ridge_file"]:
                old_reverse_files.add(f"model/{item[key]}")
        excluded = old_reverse_files | {"model/metadata.json"}
        members = {
            item.filename: (archive.read(item.filename), item.compress_type)
            for item in archive.infolist()
            if not item.is_dir() and item.filename not in excluded
        }
    metadata = deepcopy(metadata)
    metadata["version"] = int(metadata.get("version", 0)) + 1
    metadata["track"] = spec["label"]
    metadata["reverse_matchup_scale"] = float(spec["scale"])
    metadata["reverse_matchup_corrections"] = reverse_specs()
    metadata["reverse_seedbag20_validation"] = {
        "seeds": SEEDS,
        "seed_count": 20,
        "strict_validation": "2022->2023 and 2023->2024",
        "equal_weight": True,
        "reverse_scale": float(spec["scale"]),
        "exact_val2024_system_bss": float(spec["val2024_bss"]),
        "base_submit019_exact_bss": 835.8612351056282,
        "purpose": spec["purpose"],
        "trackman_rule": "strict as-of; pitcher-season >=500",
    }
    artifacts = []
    for seed in SEEDS:
        for suffix in ["lookup.csv", "pair.csv", "ridge.json"]:
            path = FINAL_DIR / f"reverse20_s{seed}_{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
            artifacts.append(path)
    with ZipFile(
        candidate, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        archive.writestr(directory_info("model/"), b"")
        archive.writestr(
            file_info("model/metadata.json"),
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        for name, (content, compress_type) in sorted(members.items()):
            archive.writestr(file_info(name, compress_type), content)
        for path in sorted(artifacts):
            archive.writestr(file_info(f"model/{path.name}"), path.read_bytes())
    with ZipFile(candidate) as archive:
        names = archive.namelist()
        if names[0] != "model/" or archive.testzip() is not None:
            raise RuntimeError(f"Invalid ZIP: {candidate}")
        required = {"model/metadata.json", "script.py", "requirements.txt"}
        if not required.issubset(names):
            raise RuntimeError(f"Missing entries: {required.difference(names)}")
        for item in reverse_specs():
            for key in ["batter_lookup_file", "pair_table_file", "ridge_file"]:
                if f"model/{item[key]}" not in names:
                    raise RuntimeError(f"Missing reverse artifact: {item[key]}")
    candidate.replace(destination)
    return destination


def run_one(path: Path, benchmark: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="lgaimers_reverse20_") as directory:
        stage = Path(directory)
        with ZipFile(path) as archive:
            archive.extractall(stage)
        (stage / "data").mkdir(exist_ok=True)
        benchmark.to_csv(stage / "data" / "test.csv", index=False)
        result = subprocess.run(
            [sys.executable, str(stage / "script.py")],
            cwd=stage,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Inference failed for {path.name}\nSTDOUT:\n{result.stdout[-4000:]}"
                f"\nSTDERR:\n{result.stderr[-8000:]}"
            )
        output = pd.read_csv(stage / "output" / "submission.csv")
    return output, time.time() - started, result.stderr[-4000:]


def smoke(paths: list[Path]) -> list[dict]:
    sample = pd.read_csv(ROOT / "data" / "test.csv")
    rows = 245_789
    repeats = int(np.ceil(rows / len(sample)))
    benchmark = pd.concat([sample] * repeats, ignore_index=True).iloc[:rows].copy()
    benchmark["row_id"] = [f"BENCH_{index:06d}" for index in range(rows)]
    benchmark.loc[np.arange(rows) % 2 == 0, "game_type"] = "R"
    benchmark.loc[np.arange(rows) % 2 == 1, "game_type"] = "F"
    records = []
    outputs = []
    for path in paths:
        output, seconds, stderr = run_one(path, benchmark)
        probability = output[TARGET].to_numpy(float)
        if len(output) != rows or not np.isfinite(probability).all():
            raise RuntimeError(f"Invalid output: {path}")
        if not output["row_id"].equals(benchmark["row_id"]):
            raise RuntimeError(f"Row order changed: {path}")
        if not ((probability > 0) & (probability < 1)).all():
            raise RuntimeError(f"Probability out of range: {path}")
        with ZipFile(path) as archive:
            uncompressed = sum(item.file_size for item in archive.infolist())
            member_count = len(archive.infolist())
        records.append(
            {
                "filename": path.name,
                "filename_length": len(path.name),
                "size_bytes": path.stat().st_size,
                "uncompressed_bytes": uncompressed,
                "member_count": member_count,
                "sha256": digest(path),
                "rows": rows,
                "inference_seconds": seconds,
                "prediction_min": float(probability.min()),
                "prediction_max": float(probability.max()),
                "prediction_mean": float(probability.mean()),
                "stderr_tail": stderr,
            }
        )
        outputs.append(probability)
    records[0]["mean_abs_diff_vs_other"] = float(np.mean(np.abs(outputs[0] - outputs[1])))
    records[1]["mean_abs_diff_vs_other"] = records[0]["mean_abs_diff_vs_other"]
    return records


def main() -> None:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    paths = [package(spec) for spec in SPECS]
    records = smoke(paths)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "base": str(BASE.relative_to(ROOT)),
        "seeds": SEEDS,
        "specs": SPECS,
        "records": records,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
