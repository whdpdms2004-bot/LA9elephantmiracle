from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import date
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ROOT_ENTRIES = {"model/", "script.py", "requirements.txt"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and archive a DACON submission ZIP by date and attempt number."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("number", type=int)
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def validate_zip(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with ZipFile(path) as archive:
        names = archive.namelist()
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        if not REQUIRED_ROOT_ENTRIES.issubset(set(names)):
            raise RuntimeError(f"Missing required entries: {names}")
        if names[0] != "model/":
            raise RuntimeError(f"model/ must be the first explicit directory entry: {names}")
        unexpected_roots = {
            name.split("/", 1)[0] + "/"
            for name in names
            if "/" in name and not name.startswith("model/")
        }
        if unexpected_roots:
            raise RuntimeError(f"Unexpected top-level directories: {unexpected_roots}")


def main():
    args = parse_args()
    if args.number < 1 or args.number > 999:
        raise ValueError("number must be between 1 and 999")
    filename = f"submit_{args.number:03d}.zip"
    if len(filename) >= 30:
        raise RuntimeError(f"Filename is not below 30 characters: {filename}")
    source = args.source.resolve()
    validate_zip(source)

    destination_dir = ROOT / "submit" / args.date
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite an archived attempt: {destination}"
        )
    shutil.copy2(source, destination)
    validate_zip(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest().upper()
    print(f"archived={destination}")
    print(f"filename_length={len(filename)}")
    print(f"size_bytes={destination.stat().st_size}")
    print(f"sha256={digest}")
    print(f"log={destination_dir / 'SUBMISSION_LOG.md'}")


if __name__ == "__main__":
    main()
