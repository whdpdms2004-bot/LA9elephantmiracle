"""투수별 반응성 — "이 투수는 어떤 조건에서 자기 평균보다 무너지는가".

계층 차감 구조를 그대로 쓴다. 이 프로젝트 최대 발견이다.

    split(투수, 조건) = EB(투수, 조건) - EB(투수)

투수 주효과를 명시적으로 빼야 "남들보다 얼마나 다른가" 만 남는다.
1겹만 빼면(리그평균만) +0.16, 2겹 빼면 +8.44 였다 (V8 vs V19).

성립 조건 둘 (METHOD.md 참조)
    ㄱ. 중복이 실재해야 한다 -> 투수 주효과가 asof_pitcher_success_rate 로 이미 모델에 있다. 충족
    ㄴ. 두 항이 같은 단위여야 한다 -> 둘 다 성공률 EB 라 환산 계수가 정확히 1. 충족

이미 있는 것과 겹치지 않는다
    타자 손   EB(투수, 타자손)                        현행 +13.75
    볼카운트  EB(투수, 타자손, 카운트) - EB(투수,타자손)  현행 +8.44
    이닝      같은 구조                               현행 +2.98
    -> 이 파일은 위 셋과 다른 축만 만든다.

축 선택 근거
    workload  시즌 누적 부하. 완전 미탐색. 셀 중앙 400행으로 현행 타자손(393)과 동급
    li        경기 중요도. V33 이 (투수,타자손,li) 로 시도해 -2.72 로 실패했으나
              셀이 72행이었다. (투수,li) 는 130행으로 1.8배다.
              당시 실패 양상이 '신호 없음' 이 아니라 '단독 BSS 746.85 -> 721.30 하락'
              이라 잡음/누수 서명에 가깝다. 굵은 셀로 재시도할 가치가 있다.
    form      최근 등판이 시즌 평균에서 얼마나 벗어났는가에 대한 반응성

셀 크기 (학습 시즌, K=300 기준 한 행 기여분)
    (투수)                711셀  중앙 807행  0.090%
    (투수, 부하5군)      2,003셀  중앙 400행  0.143%   <- 현행 타자손과 동급
    (투수, li 5군)       3,198셀  중앙 130행  0.233%
    (투수, 이탈3군)        아래 계산
    전부 1% 미만 = 자기 라벨 누수 관문 통과 (METHOD.md 7)

행 독립성
    테이블은 학습 시즌으로만 만들고, 추론은 (투수, 구간) 키 조인이다.
    구간은 그 행의 값만으로 정해진다. test 행끼리 집계하지 않는다.

주의
    등판 내 투구 수는 데이터에 없다. 유도하려면 test 행을 묶어 세야 하고
    그건 행 독립성 위반이다. 그래서 '부하' 는 시즌 누적(asof_pitcher_n) 으로 잡았다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "pitcher_reactivity"
TARGETS = ["pitcher_id x asof_pitcher_n", "pitcher_id x li",
           "pitcher_id x recent-form-gap"]
NOTE = ("투수별 조건 반응성. EB(투수,조건) - EB(투수) 계층 차감. "
        "타자손/카운트/이닝은 이미 있으므로 부하·중요도·최근이탈 세 축만 만든다.")
CONFLICTS = []

K = 300.0
TARGET_COL = "control_success"


def _buckets(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    num = lambda c: pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
    n = np.nan_to_num(num("asof_pitcher_n"), nan=0.0)
    li = np.nan_to_num(num("li"), nan=1.0)
    prev = np.vstack([num(f"asof_pitcher_prev{k}_game_success_rate") for k in (1, 3, 5)])
    with np.errstate(invalid="ignore"):
        gap = np.nanmean(prev, axis=0) - num("asof_pitcher_success_rate")
    gap = np.nan_to_num(gap, nan=0.0)
    return {
        "workload": np.digitize(n, [100, 500, 2000, 4000]),
        "leverage": np.digitize(li, [0.7, 1.0, 1.5, 2.5]),
        "form": np.digitize(gap, [-0.04, 0.04]),
    }


def apply(frame: pd.DataFrame, features: list[str], categorical: list[str],
          train_mask: pd.Series, fold: int):
    y = pd.to_numeric(frame[TARGET_COL], errors="coerce").to_numpy(np.float64)
    pid = frame["pitcher_id"].to_numpy()
    buckets = _buckets(frame)
    tr = np.asarray(train_mask).astype(bool)
    league = float(np.nanmean(y[tr]))

    # 1겹: 투수 주효과. 학습 행에서만.
    d1 = pd.DataFrame({"p": pid[tr], "y": y[tr]}).groupby("p")["y"].agg(["sum", "size"])
    eb1 = ((d1["sum"] + K * league) / (d1["size"] + K)).rename("eb1")

    extras: dict[str, np.ndarray] = {}
    for axis, b in buckets.items():
        d2 = (pd.DataFrame({"p": pid[tr], "b": b[tr], "y": y[tr]})
              .groupby(["p", "b"])["y"].agg(["sum", "size"]))
        eb2 = (d2["sum"] + K * league) / (d2["size"] + K)
        tbl = eb2.rename("eb2").reset_index()
        tbl = tbl.merge(eb1.reset_index(), left_on="p", right_index=True, how="left")
        tbl["split"] = tbl["eb2"] - tbl["eb1"]
        tbl["rel"] = (d2["size"].reindex(
            pd.MultiIndex.from_arrays([tbl["p"], tbl["b"]])).to_numpy()
            / (d2["size"].reindex(
                pd.MultiIndex.from_arrays([tbl["p"], tbl["b"]])).to_numpy() + K))

        key = pd.DataFrame({"p": pid, "b": b})
        joined = key.merge(tbl[["p", "b", "split", "rel"]], on=["p", "b"], how="left")
        split = joined["split"].to_numpy(np.float64)
        rel = np.nan_to_num(joined["rel"].to_numpy(np.float64), nan=0.0)
        split = np.nan_to_num(split, nan=0.0)
        extras[f"react_{axis}_split"] = split.astype(np.float32)
        extras[f"react_{axis}_rel"] = rel.astype(np.float32)
        # 신뢰도로 감쇠한 형태. 표본이 적은 셀의 값을 자동으로 0 쪽으로 민다.
        extras[f"react_{axis}_split_w"] = (split * rel).astype(np.float32)

    return extras, features, categorical
