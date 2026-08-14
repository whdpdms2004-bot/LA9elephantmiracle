from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
OUTPUT_DIR = ROOT / "submit" / "2026-08-12"
SCRIPT = Path(__file__).resolve().parent / "submission_script_matchup.py"
LOOKUP = WORK / "artifacts" / "r_context_2025" / "r_context_recent.csv"
sys.path.insert(0, str(MODEL_DIR))

from build_final_two_submissions import smoke_test  # noqa: E402


SPECS = [
    {
        "number": 15,
        "base": OUTPUT_DIR / "submit_013.zip",
        "validation_bss_2024": 830.5234166493314,
        "validation_r_bss_2024": 832.8604276582885,
        "label": "submit013_plus_r_context",
    },
    {
        "number": 16,
        "base": OUTPUT_DIR / "submit_014.zip",
        "validation_bss_2024": 831.3205849446725,
        "validation_r_bss_2024": 833.4656864893542,
        "label": "submit014_plus_r_context",
    },
]


def file_info(name: str, compress_type: int = ZIP_DEFLATED) -> ZipInfo:
    info = ZipInfo(name)
    info.date_time = (2026, 8, 12, 0, 0, 0)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    return info


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def build(spec: dict) -> dict:
    source = Path(spec["base"])
    destination = OUTPUT_DIR / f"submit_{spec['number']:03d}.zip"
    candidate = destination.with_suffix(".candidate.zip")
    if candidate.exists():
        raise FileExistsError(f"Refusing to overwrite temporary candidate: {candidate}")
    if not source.is_file() or not LOOKUP.is_file() or not SCRIPT.is_file():
        raise FileNotFoundError("Base submission, lookup, or inference script is missing")
    with ZipFile(source) as archive:
        metadata = json.loads(archive.read("model/metadata.json"))
        members = {
            item.filename: (archive.read(item.filename), item.compress_type)
            for item in archive.infolist()
            if not item.is_dir()
            and item.filename not in {
                "model/metadata.json", "script.py", "model/r_context_recent.csv"
            }
        }
    metadata = deepcopy(metadata)
    metadata["version"] = int(metadata.get("version", 0)) + 1
    metadata["track"] = spec["label"]
    metadata["r_context_correction"] = {
        "lookup_file": "r_context_recent.csv",
        "keys": [
            "balls_before", "strikes_before", "inning_bucket",
            "pitcher_hand", "batter_hand",
        ],
        "correction_column": "scaled_correction",
        "game_type": "R",
        "source_oof_seasons": [2023, 2024],
        "history_weight": 1.0,
        "smoothing": 5000.0,
        "scale": 1.15,
        "validation_rule": "strict prior-season OOF residual lookup",
        "val2024_bss": spec["validation_bss_2024"],
        "val2024_r_bss": spec["validation_r_bss_2024"],
    }
    with ZipFile(
        candidate, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        directory = ZipInfo("model/")
        directory.date_time = (2026, 8, 12, 0, 0, 0)
        directory.compress_type = ZIP_STORED
        directory.external_attr = (0o755 << 16) | 0x10
        archive.writestr(directory, b"")
        archive.writestr(
            file_info("model/metadata.json"),
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        for name, (content, compress_type) in sorted(members.items()):
            archive.writestr(file_info(name, compress_type), content)
        archive.writestr(file_info("model/r_context_recent.csv"), LOOKUP.read_bytes())
        archive.writestr(file_info("script.py"), SCRIPT.read_bytes())
    with ZipFile(candidate) as archive:
        names = archive.namelist()
        if names[0] != "model/" or archive.testzip() is not None:
            raise RuntimeError(f"Invalid ZIP: {candidate}")
        required = {
            "model/metadata.json", "model/r_context_recent.csv",
            "script.py", "requirements.txt",
        }
        if not required.issubset(names):
            raise RuntimeError(f"Missing ZIP entries: {required.difference(names)}")
    inference_sec = smoke_test(candidate)
    candidate.replace(destination)
    return {
        "filename": destination.name,
        "filename_length": len(destination.name),
        "base": source.name,
        "size_bytes": destination.stat().st_size,
        "sha256": digest(destination),
        "inference_sec_245789": inference_sec,
        "validation_bss_2024": spec["validation_bss_2024"],
        "validation_r_bss_2024": spec["validation_r_bss_2024"],
    }


def main() -> None:
    records = [build(spec) for spec in SPECS]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "context": {
            "keys": [
                "balls_before", "strikes_before", "inning_bucket",
                "pitcher_hand", "batter_hand",
            ],
            "source_oof_seasons": [2023, 2024],
            "older_weight": 1.0,
            "smoothing": 5000.0,
            "scale": 1.15,
            "game_type": "R",
            "trackman_used": False,
        },
        "records": records,
    }
    path = WORK / "final" / "r_context_submission_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
