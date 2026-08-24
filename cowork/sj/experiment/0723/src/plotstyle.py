"""matplotlib 한글 폰트 설정 — 모든 노트북에서 `import plotstyle` 한 줄로 적용.

한글이 □□□ 로 깨지는 것 방지. 우선순위대로 사용 가능한 폰트를 찾아 적용한다.
(샌드박스: Noto Sans CJK JP 사용 — CJK 통합 폰트라 한글 글리프 포함)
"""
from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic",
              "Noto Sans KR", "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Serif CJK JP",
              "DejaVu Sans"]


def apply(size: int = 10) -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in CANDIDATES if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False       # 마이너스 기호 깨짐 방지
    plt.rcParams["font.size"] = size
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    return chosen


FONT = apply()
