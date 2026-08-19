"""검증 가능한 3WAY 최종 산출물을 제출 ZIP으로 묶는다."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
TW = HERE.parent
SJ = TW.parent
DEFAULT_SOURCE = TW / "outputs" / "final_three_way_v1"
DEFAULT_OUTPUT = SJ / "submit" / "2026-08-19" / "submit_037.zip"
EXPECTED_MODELS = {
    "three_way_middle.cbm", "three_way_reverse.cbm",
    "three_way_outside.cbm", "three_way_mr.cbm",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    required = [source / "model", source / "script.py", source / "requirements.txt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    metadata_path = source / "model" / "three_way_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("final_ready"):
        raise RuntimeError("prepare-only or incomplete artifacts cannot be packaged")
    actual_models = {path.name for path in (source / "model").glob("*.cbm")}
    if actual_models != EXPECTED_MODELS:
        raise AssertionError({"expected": sorted(EXPECTED_MODELS),
                              "actual": sorted(actual_models)})
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path for path in (source / "model").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc")
    files.extend([source / "script.py", source / "requirements.txt"])
    with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(source).as_posix())

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        top_level = sorted({name.split("/", 1)[0] for name in names})
        if top_level != ["model", "requirements.txt", "script.py"]:
            raise AssertionError(top_level)
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"CRC failure: {bad_file}")
    print(json.dumps({
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "top_level": top_level,
        "files": names,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
