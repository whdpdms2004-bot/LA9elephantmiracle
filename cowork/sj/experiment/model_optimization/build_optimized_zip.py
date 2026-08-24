from __future__ import annotations

import shutil
import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "submit_optimized"
MODEL_DIR = SOURCE / "model"
SCRIPT_SOURCE = HERE / "optimized_script.py"
OUTPUT = HERE / "submit_optimized_linux.zip"
FIXED_TIMESTAMP = (2026, 8, 6, 0, 0, 0)


def directory_info(name):
    info = ZipInfo(name.rstrip("/") + "/", FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = ZIP_STORED
    info.external_attr = (stat.S_IFDIR | 0o755) << 16
    return info


def file_info(name):
    info = ZipInfo(name, FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def main():
    if not MODEL_DIR.is_dir() or not (MODEL_DIR / "metadata.json").is_file():
        raise FileNotFoundError("Run train_final_submission.py first")
    shutil.copy2(SCRIPT_SOURCE, SOURCE / "script.py")
    (SOURCE / "requirements.txt").write_text(
        "xgboost==3.1.1\ncatboost==1.2.8\n", encoding="utf-8"
    )

    model_files = sorted(path for path in MODEL_DIR.iterdir() if path.is_file())
    if not model_files:
        raise RuntimeError("No model artifacts")
    with ZipFile(OUTPUT, "w", allowZip64=True) as archive:
        archive.writestr(directory_info("model/"), b"")
        for source in model_files:
            archive.writestr(file_info(f"model/{source.name}"), source.read_bytes())
        archive.writestr(file_info("script.py"), (SOURCE / "script.py").read_bytes())
        archive.writestr(
            file_info("requirements.txt"), (SOURCE / "requirements.txt").read_bytes()
        )

    with ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if names[0] != "model/" or "script.py" not in names or "requirements.txt" not in names:
            raise RuntimeError(f"Unexpected ZIP structure: {names}")
        if any(name.startswith("submit_optimized/") for name in names):
            raise RuntimeError("Extra top-level directory detected")
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
    print(f"created={OUTPUT} size_mb={OUTPUT.stat().st_size / 2**20:.2f}")


if __name__ == "__main__":
    main()
