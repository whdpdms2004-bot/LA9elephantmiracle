"""submit_037 teacher와 OOF 사건 보정기를 묶어 submit_040을 만든다."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
META = HERE.parent
TW = META.parent
SJ = TW.parent
SOURCE = TW / "outputs" / "final_three_way_v1"
FUSION = META / "outputs" / "initial_event_calibration_model.json"
BUILD = META / "outputs" / "initial_submission_v2"
OUTPUT = SJ / "submit" / "2026-08-20" / "submit_040.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if BUILD.exists() or OUTPUT.exists():
        raise FileExistsError({"build": str(BUILD), "zip": str(OUTPUT)})
    if not SOURCE.is_dir() or not FUSION.is_file():
        raise FileNotFoundError({"source": str(SOURCE), "fusion": str(FUSION)})
    BUILD.mkdir(parents=True)
    shutil.copytree(
        SOURCE / "model",
        BUILD / "model",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    shutil.copy2(
        SOURCE / "script.py", BUILD / "model" / "three_way_teacher_inference.py")
    shutil.copy2(FUSION, BUILD / "model" / "event_calibration.json")
    shutil.copy2(HERE / "initial_fusion_inference.py", BUILD / "script.py")
    shutil.copy2(SOURCE / "requirements.txt", BUILD / "requirements.txt")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    files = [path for path in sorted(BUILD.rglob("*")) if path.is_file()]
    with zipfile.ZipFile(
            OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(BUILD).as_posix())
    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        top = sorted({name.split("/", 1)[0] for name in names})
        if top != ["model", "requirements.txt", "script.py"]:
            raise AssertionError(top)
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC failure")
    result = {
        "path": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256(OUTPUT),
        "file_count": len(names),
        "top_level": top,
    }
    (META / "outputs" / "initial_package_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
