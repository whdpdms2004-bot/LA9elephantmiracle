"""B2 예시 — 키 조인 룩업. 학습 시즌만으로 테이블을 만든다.

v76_feature_intake.py --spec featspec/example_b2.py --gate-only
"""
import numpy as np
import pandas as pd

CLASS = "B2"
KEYS = ["pitcher_id", "batter_hand"]
CELL_KEYS = ["pitcher_id", "batter_hand"]     # G3 셀 크기 검사에 사용
EB_K = 300
NOTE = "예시. 실제 피처로 교체할 것."


def make_table(df, tr_mask):
    """학습 행만으로 테이블 생성. 반환: KEYS + 피처열."""
    d = df.loc[tr_mask, KEYS + ["control_success"]]
    lg = float(d["control_success"].mean())
    g = d.groupby(KEYS)["control_success"].agg(["sum", "size"]).reset_index()
    g["ex_eb"] = (g["sum"] + EB_K * lg) / (g["size"] + EB_K) - lg
    g["ex_rel"] = g["size"] / (g["size"] + EB_K)
    return g[KEYS + ["ex_eb", "ex_rel"]]
