"""performance_tracking 공용 로더·채점기.

README 2~3절의 규격을 강제하는 유일한 구현체다. score_val.py / corr.py 둘 다
여기를 통과한다. 규격 위반은 조용히 통과시키지 않고 예외로 세운다 - 잘못된
행 정렬이나 way 확률이 섞여 들어오면 점수가 그럴듯하게 나오면서 틀린다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows 콘솔(cp949)에서 표에 없는 문자 하나로 스크립트가 죽지 않게 한다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[2]          # 저장소 루트
PT = ROOT / "performance_tracking"
VAL_DIR = PT / "val"
MODELS_DIR = PT / "models"
RESULTS = PT / "results.csv"
CACHE = PT / ".cache"

TRAIN = ROOT / "data" / "train.csv"
TARGET = "control_success"

SEASONS = (2024, 2022)          # 앞이 주 판정, 뒤가 비하락 조건 (규칙 1)
DECISION_SEASON = 2024
GUARD_SEASON = 2022
EPS = 1e-7

# README 2절 - 월 블록. 후반이 2025 에서 가장 위험한 구간이다.
BLOCKS = {"early": (3, 5), "mid": (6, 7), "late": (8, 10)}

# way 확률이 섞여 들어온 걸 잡는 창. 최종 success 평균은 0.47~0.53 이고
# way(middle 0.15 / reverse 0.23 / outside 0.13 / mr 0.03)는 겹치지 않는다.
PRED_MEAN_LO, PRED_MEAN_HI = 0.35, 0.65


class SpecViolation(ValueError):
    """val 예측 파일이 README 2절 규격을 어겼을 때."""


# --------------------------------------------------------------------------- #
# 라벨
# --------------------------------------------------------------------------- #
def load_labels(season: int) -> pd.DataFrame:
    """해당 시즌의 row_id / y / game_type / game_month. 첫 호출 때 캐시를 만든다."""
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"labels_{season}.csv"
    if cached.exists():
        return pd.read_csv(cached)
    if not TRAIN.exists():
        raise FileNotFoundError(
            f"{TRAIN} 가 없다. 대회 원본은 커밋하지 않으므로 직접 배치해야 한다."
        )
    cols = ["row_id", "season", "game_type", "game_month", TARGET]
    df = pd.read_csv(TRAIN, usecols=lambda c: c.strip("﻿") in cols)
    df.columns = [c.strip("﻿") for c in df.columns]
    df = df[(df["season"] == season) & df[TARGET].notna()].copy()
    out = pd.DataFrame({
        "row_id": df["row_id"].astype(str),
        "y": df[TARGET].astype(np.float64),
        "game_type": df["game_type"].astype(str),
        "game_month": pd.to_numeric(df["game_month"], errors="coerce"),
    }).sort_values("row_id").reset_index(drop=True)
    out.to_csv(cached, index=False)
    return out


# --------------------------------------------------------------------------- #
# 예측
# --------------------------------------------------------------------------- #
def val_path(name: str, season: int) -> Path:
    return VAL_DIR / f"{name}_{season}.csv"


def load_pred(name: str, season: int, labels: pd.DataFrame | None = None) -> np.ndarray:
    """val/<name>_<season>.csv 를 라벨 행 순서에 맞춰 정렬해 반환한다."""
    p = val_path(name, season)
    if not p.exists():
        raise SpecViolation(f"{p} 가 없다 (규칙 3 - val 예측을 남겨야 등록된다).")
    lab = load_labels(season) if labels is None else labels

    df = pd.read_csv(p)
    if list(df.columns[:2]) != ["row_id", "pred"]:
        raise SpecViolation(f"{p.name}: 컬럼은 row_id,pred 여야 한다 (받은 값 {list(df.columns)}).")
    df = df[["row_id", "pred"]].copy()
    df["row_id"] = df["row_id"].astype(str)
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")

    if df["row_id"].duplicated().any():
        n = int(df["row_id"].duplicated().sum())
        raise SpecViolation(f"{p.name}: row_id 중복 {n:,}개.")
    if not np.isfinite(df["pred"]).all():
        raise SpecViolation(f"{p.name}: pred 에 결측/비유한값 {int((~np.isfinite(df['pred'])).sum()):,}개.")

    want, got = set(lab["row_id"]), set(df["row_id"])
    if want != got:
        raise SpecViolation(
            f"{p.name}: 행 집합 불일치 - 누락 {len(want - got):,}, 초과 {len(got - want):,} "
            f"(기대 {len(want):,}행, {season} 시즌의 {TARGET} 비결측 전체)."
        )

    pred = df.set_index("row_id").loc[lab["row_id"], "pred"].to_numpy(np.float64)
    if pred.min() < 0 or pred.max() > 1:
        raise SpecViolation(f"{p.name}: pred 가 [0,1] 밖이다 (min {pred.min():.4f}, max {pred.max():.4f}). 로짓을 넣었나.")
    m = float(pred.mean())
    if not (PRED_MEAN_LO <= m <= PRED_MEAN_HI):
        raise SpecViolation(
            f"{p.name}: 예측 평균 {m:.4f} 가 최종 success 범위({PRED_MEAN_LO}~{PRED_MEAN_HI}) 밖이다. "
            "way 확률이나 미보정 출력을 넣은 것으로 본다 (README 2절)."
        )
    return pred


# --------------------------------------------------------------------------- #
# 채점
# --------------------------------------------------------------------------- #
def bss(y: np.ndarray, p: np.ndarray) -> float:
    """대회 공식. Score = 100000 x (1 - Brier / (r(1-r))), r 은 그 부분군 기저율."""
    if len(y) == 0:
        return float("nan")
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    y = np.asarray(y, np.float64)
    r = float(y.mean())
    null = r * (1.0 - r)
    if null <= 0:
        return float("nan")
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / null)


def score(pred: np.ndarray, season: int, labels: pd.DataFrame | None = None) -> dict:
    """한 시즌 전체 지표. all 이 판정값, R/F·월 블록은 착시 점검용이다."""
    lab = load_labels(season) if labels is None else labels
    y = lab["y"].to_numpy(np.float64)
    g = lab["game_type"].to_numpy()
    mth = lab["game_month"].to_numpy(np.float64)
    p = np.clip(pred, EPS, 1 - EPS)
    R, F = g == "R", g == "F"

    out = {
        "season": season, "n": len(y),
        "all": bss(y, p), "R": bss(y[R], p[R]), "F": bss(y[F], p[F]),
        "n_R": int(R.sum()), "n_F": int(F.sum()),
        "brier": float(np.mean((p - y) ** 2)),
        "pred_mean": float(p.mean()), "true_mean": float(y.mean()),
    }
    for k, (a, b) in BLOCKS.items():
        blk = (mth >= a) & (mth <= b)
        out[k] = bss(y[blk], p[blk])
        out[f"n_{k}"] = int(blk.sum())
    return out


def render(m: dict) -> str:
    return (
        f"  {m['season']}  n={m['n']:,}\n"
        f"    all {m['all']:>10,.1f}   R {m['R']:>10,.1f} (n={m['n_R']:,})   "
        f"F {m['F']:>10,.1f} (n={m['n_F']:,})\n"
        f"    early {m['early']:>9,.1f}   mid {m['mid']:>9,.1f}   late {m['late']:>9,.1f}\n"
        f"    pred_mean {m['pred_mean']:.4f}  true_mean {m['true_mean']:.4f}  "
        f"offset {m['pred_mean'] - m['true_mean']:+.4f}  brier {m['brier']:.6f}"
    )


# --------------------------------------------------------------------------- #
# 등록부
# --------------------------------------------------------------------------- #
def registered() -> list[str]:
    """results.csv 에 등록된 이름. val 예측이 실제로 있는 것만."""
    if not RESULTS.exists():
        return []
    df = pd.read_csv(RESULTS)
    if "name" not in df.columns:
        return []
    return [n for n in df["name"].astype(str).tolist()
            if all(val_path(n, s).exists() for s in SEASONS)]
