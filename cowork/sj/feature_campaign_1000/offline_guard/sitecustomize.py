"""제출 smoke test에서 외부 네트워크 연결을 강제로 실패시킨다."""
from __future__ import annotations

import socket


def _blocked(*args, **kwargs):
    raise RuntimeError("network access is disabled by submission smoke test")


socket.create_connection = _blocked
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
