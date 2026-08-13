from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
QUEUE = WORK_DIR / "run_search_queue.py"
PID_FILE = WORK_DIR / "search_queue.pid"

flags = 0
if os.name == "nt":
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NO_WINDOW
    )

process = subprocess.Popen(
    [sys.executable, str(QUEUE)],
    cwd=WORK_DIR.parents[1],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
    creationflags=flags,
)
PID_FILE.write_text(str(process.pid), encoding="ascii")
print(process.pid)

