"""새 전처리를 추가하는 템플릿. 이 파일을 복사해서 이름만 바꾸면 된다.

이 폴더에 두면 자동으로 발견된다. 다른 파일은 건드릴 필요가 없다.

    cp example_template.py my_idea.py
    # NAME / TARGETS / NOTE / apply 를 채운다
    python <랩>/scripts/run_combo.py --combos my_idea --confirm-fold 2024

지켜야 할 것 (METHOD.md §6)
    1. 행 독립성   test 행끼리 집계하지 않는다. groupby/rolling/cumsum 전부 금지.
                   frame 전체를 받지만 통계는 train_mask 안에서만 계산할 것.
    2. 시간 인과   fold 이전 시즌만 사용. as-of 컬럼은 그대로 써도 된다.
    3. 라벨 금지   control_success 를 직접 읽지 않는다.
                   라벨 집계 테이블을 만들 거라면 셀 크기를 먼저 확인할 것 (METHOD.md §7).
    4. 결정적      난수를 쓰면 시드를 고정한다.

이 파일 자체는 DISABLED = True 라서 등록되지 않는다. 복사본에서 지울 것.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DISABLED = True                 # 복사본에서는 이 줄을 지운다

NAME = "example_template"       # 고유해야 한다. 파일명과 같게 두면 편하다
TARGETS = ["asof_pitcher_success_rate", "asof_pitcher_n"]   # 어느 피처를 건드리는가
NOTE = "왜 이게 더 나을 것 같은지 한 줄. 근거가 있으면 출처도."
CONFLICTS = []                  # 같이 켜면 안 되는 변환 이름들. 예: ["no_trackman"]


def apply(frame: pd.DataFrame, features: list[str], categorical: list[str],
          train_mask: pd.Series, fold: int):
    """전처리를 적용한다.

    입력
        frame        전체 데이터 (학습 + 검증). 통계는 train_mask 안에서만 낼 것
        features     현재 사용 중인 피처 이름 목록
        categorical  그중 범주형으로 다룰 것들
        train_mask   season < fold 인 행. 통계 fit 은 여기서만
        fold         검증 시즌 (예: 2024)

    반환 (셋 다 필수)
        extras       {새 열 이름: np.ndarray}   — 추가할 피처. 없으면 {}
        features     갱신된 피처 목록           — 제거하지 않으면 그대로 반환
        categorical  갱신된 범주형 목록

    주의: extras 의 배열 길이는 len(frame) 이어야 한다.
    """
    rate = pd.to_numeric(frame["asof_pitcher_success_rate"],
                         errors="coerce").to_numpy(np.float64)
    n = pd.to_numeric(frame["asof_pitcher_n"],
                      errors="coerce").to_numpy(np.float64)
    n = np.nan_to_num(n, nan=0.0).clip(min=0.0)

    # 사전확률은 학습 행에서만 낸다 (시간 인과)
    prior = float(pd.to_numeric(
        frame.loc[train_mask, "asof_pitcher_success_rate"],
        errors="coerce").mean())

    K = 150.0
    eb = (np.nan_to_num(rate, nan=prior) * n + prior * K) / (n + K)
    extras = {
        "ex_pitcher_eb150": eb.astype(np.float32),
        "ex_pitcher_eb150_gap": (eb - prior).astype(np.float32),
    }
    return extras, features, categorical
