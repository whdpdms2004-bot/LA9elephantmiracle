from __future__ import annotations

import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "submit_v1"
OUTPUTS = [HERE / "submit_v1.zip", HERE / "submit_fixed_linux.zip"]
FIXED_TIMESTAMP = (2026, 8, 5, 0, 0, 0)


def directory_info(name: str) -> ZipInfo:
    info = ZipInfo(name.rstrip("/") + "/", FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = ZIP_STORED
    info.external_attr = (stat.S_IFDIR | 0o755) << 16
    return info


def file_info(name: str) -> ZipInfo:
    info = ZipInfo(name, FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def build(output: Path) -> None:
    files = {
        "model/model.pt": SOURCE / "model" / "model.pt",
        "script.py": SOURCE / "script.py",
        "requirements.txt": SOURCE / "requirements.txt",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Submission source files are missing: {missing}")

    with ZipFile(output, "w", allowZip64=True) as archive:
        archive.writestr(directory_info("model/"), b"")
        for archive_name, source in files.items():
            archive.writestr(file_info(archive_name), source.read_bytes())

    with ZipFile(output) as archive:
        expected = ["model/", "model/model.pt", "script.py", "requirements.txt"]
        if archive.namelist() != expected:
            raise RuntimeError(f"Unexpected archive structure: {archive.namelist()}")
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        if archive.getinfo("model/model.pt").file_size == 0:
            raise RuntimeError("Packaged model is empty")


for destination in OUTPUTS:
    build(destination)
    print(destination)

