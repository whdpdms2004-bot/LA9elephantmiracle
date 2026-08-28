"""group_by_perform 공용 — 축 정의 · 지표 · 모델 탐색.

PLAN.md §1~§2 의 규격을 강제하는 유일한 구현체다. 축을 새로 만들거나 지표를
바꾸려면 여기만 고친다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace", encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PT = HERE.parent
ROOT = PT.parent
OUT = HERE / "out"
VAL = PT / "val"
TRAIN = ROOT / "data" / "train.csv"

sys.path.insert(0, str(PT / "tools"))

SEASONS = (2022, 2024)
DECISION, GUARD = 2024, 2022
EPS = 1e-7

# --------------------------------------------------------------------------- #
# 축 — PLAN.md §2. key: (라벨 컬럼명, 표시명, 구간 순서)
# 순서는 여기 적힌 그대로 보고서에 나온다. 사전순 정렬에 맡기지 않는다.
# --------------------------------------------------------------------------- #
CNT = [f"{b}-{s}" for b in range(4) for s in range(3)]
AXES: dict[str, tuple[str, list[str]]] = {
    "a1_count":   ("A1 볼카운트", CNT),
    "a1b_phase":  ("A1b 카운트 국면", ["초구 0-0", "투수우위 0-1", "평행 1-1",
                                      "타자우위 1-0/2-0/2-1", "2스트라이크", "3볼", "풀카운트"]),
    "a2_li":      ("A2 경기 중요도 li", ["li<0.10", "0.10-0.35", "0.35-0.70",
                                        "0.70-1.20", "1.20-2.00", "li>=2.00"]),
    "a3_load":    ("A3 전시즌 경기당 대비 등판 투구수", ["<0.25", "0.25-0.50", "0.50-0.75",
                                                      "0.75-1.00", "1.00-1.25", "1.25-1.50",
                                                      ">=1.50", "전시즌없음"]),
    "a3b_gp":     ("A3b 등판 내 투구수", ["1-10", "11-20", "21-35", "36-60",
                                        "61-80", "81+"]),
    "a4_prevp":   ("A4 전시즌 투구수", ["없음", "1-200", "201-600", "601-1200",
                                      "1201-2000", "2001-2600", "2601+"]),
    "a5_asofn":   ("A5 투수 누적 이력", ["0-99", "100-499", "500-1499", "1500-3999",
                                       "4000-7999", "8000+"]),
    "a6_psucc":   ("A6 투수 수준 5분위", ["Q1 최저", "Q2", "Q3", "Q4", "Q5 최고", "이력없음"]),
    "a7_inning":  ("A7 이닝", ["1-3회", "4-6회", "7-8회", "9회+"]),
    "a8_hand":    ("A8 손 매치업", []),          # 값에서 채운다
    "a9_batn":    ("A9 타자 누적 이력", ["0-99", "100-499", "500-1499", "1500-3999", "4000+"]),
    "a10_base":   ("A10 주자 상황", ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]),
    "a11_typemo": ("A11 game_type x 월블록", ["R|early", "R|mid", "R|late",
                                             "F|early", "F|mid", "F|late"]),
    "a12_role":   ("A12 투수 역할(전시즌)", ["불펜(<30구/경기)", "스윙(30-70)",
                                           "선발(>=70)", "전시즌없음"]),
    "a13_tens":   ("A13 이닝 x 점수차", ["초반|0-1점", "초반|2-3점", "초반|4점+",
                                       "중반|0-1점", "중반|2-3점", "중반|4점+",
                                       "후반|0-1점", "후반|2-3점", "후반|4점+"]),
}
PRIMARY = ["a1_count", "a2_li", "a3_load", "a4_prevp"]      # 지시받은 4축

# val/ 에는 실험 중간본까지 54개가 쌓여 있다. 기본 대상은 등록·현행 모델만 본다.
# 전부 보려면 --all. TEAM 은 group_score.py 가 §4 규약대로 합성한다.
MAIN = ["sj_stdmlp", "sj_grid_w060", "sj_e2var", "sj3way_nv",
        "cw_v17_base", "hw_v12_honest", "ye_hand", "sj_cb_ft_fonly"]
MAIN_2024_ONLY = ["yn_fa10c"]
TEAM_MEMBERS = ["sj_stdmlp", "cw_v17_base", "hw_v12_honest", "ye_hand", "sj3way_nv"]


# --------------------------------------------------------------------------- #
# 지표 — PLAN.md §1
# --------------------------------------------------------------------------- #
def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC. 순위 기반이라 tie 를 평균순위로 처리한다 (Mann-Whitney)."""
    n = len(y)
    n1 = float(y.sum())
    if n1 == 0 or n1 == n:
        return float("nan")
    r = pd.Series(p).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (n - n1)))


def bin_metrics(y: np.ndarray, p: np.ndarray, null_all: float, n_all: int) -> dict:
    """구간 하나의 전 지표. null_all·n_all 은 시즌 전체값 — deficit 을 합 보존시킨다."""
    n = len(y)
    if n == 0:
        return {}
    p = np.clip(p, EPS, 1 - EPS)
    r = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    pm = float(p.mean())
    bias = pm - r
    local_null = r * (1.0 - r)
    return {
        "n": n,
        "share": n / n_all,
        "rate": r,
        "pred_mean": pm,
        "bias": bias,
        "auc": auc(y, p),
        "brier": brier,
        "bss_local": 100000.0 * (1.0 - brier / local_null) if local_null > 0 else np.nan,
        # Sigma deficit = 100000 - BSS_all (항등식). 구간 비교의 유일한 기준.
        "deficit": 100000.0 * (n / n_all) * brier / null_all,
        # 상수 c=r-pred_mean 를 더하면 Brier 가 정확히 bias^2 만큼 준다.
        "shift_gain_self": 100000.0 * (n / n_all) * (bias ** 2) / null_all,
    }


# --------------------------------------------------------------------------- #
# 자료
# --------------------------------------------------------------------------- #
def load_axes(season: int) -> pd.DataFrame:
    p = OUT / f"axes_{season}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — build_axes.py 를 먼저 돌린다.")
    return pd.read_csv(p, dtype={"row_id": str})


def models(season: int) -> list[str]:
    """val/<name>_<season>.csv 가 있는 이름 전부. 진단은 등록 여부를 따지지 않는다."""
    return sorted({f.name[: -len(f"_{season}.csv")]
                   for f in VAL.glob(f"*_{season}.csv")})


def both_season_models() -> list[str]:
    return sorted(set(models(2022)) & set(models(2024)))
