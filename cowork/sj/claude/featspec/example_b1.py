"""B1 예시 — 행 단위 파생. 새 피처를 받으면 이 형식으로 옮겨 담는다.

v76_feature_intake.py --spec featspec/example_b1.py --gate-only
"""
import numpy as np
import pandas as pd

CLASS = "B1"
NOTE = "예시. 실제 피처로 교체할 것."


def make(df, tr_mask):
    """df: 전체 프레임, tr_mask: 학습 행 마스크(테이블 생성용).
    반환: 행 수 == len(df) 인 DataFrame, 새 피처 열만."""
    out = pd.DataFrame(index=range(len(df)))
    p = pd.to_numeric(df["asof_pitcher_success_rate"]).to_numpy()
    b = pd.to_numeric(df["asof_batter_success_rate"]).to_numpy()
    out["ex_pitch_minus_bat"] = p - b
    out["ex_pitch_over_bat"] = p / np.clip(b, 1e-3, None)
    return out
