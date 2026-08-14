from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
WATCHER = WORK_DIR / "wait_and_build_ensemble.py"
flags = 0
if os.name == "nt":
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
process = subprocess.Popen(
    [sys.executable, str(WATCHER)], cwd=WORK_DIR.parents[1],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    close_fds=True, creationflags=flags,
)
(WORK_DIR / "ensemble_watcher.pid").write_text(str(process.pid), encoding="ascii")
print(process.pid)

