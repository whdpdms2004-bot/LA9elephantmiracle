"""SJ의 이미 로드된 CW 행렬에 YN A3 피처를 붙이는 독립 어댑터.

이 모듈은 ``train.csv``나 ``yn_fa10c.zip``을 읽지 않는다. SJ 코드의
``load_base()``가 반환한 ``X, season``과 ``work/meta.json``의 ``names``만 사용한다.
CW 168열에는 원본 ID와 ``game_type=F`` 원핫 열이 있으므로, 그 정보만으로 시즌별
walk-forward A3와 2025 배포 lookup을 정확히 재구성할 수 있다.

계약:
  * 시즌 S 행은 season < S 행만 lookup 산출에 사용한다.
  * 2025 lookup은 season <= 2024 행만 사용한다.
  * lookup 미등장 선수는 NaN이다.
  * 반환 열 순서는 A_COLS 고정이다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


A_COLS = [
    "fe_pitcher_futures_share",
    "fe_batter_futures_share",
    "fe_pitcher_prior_n_log",
]


def _column(X: np.ndarray, names: Sequence[str], name: str) -> np.ndarray:
    try:
        index = list(names).index(name)
    except ValueError as exc:
        raise ValueError(f"SJ base 피처에 {name!r} 열이 없음") from exc
    return np.asarray(X[:, index])


def _f_indicator(X: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """CW 원핫 또는 원본 문자열 game_type에서 F 표시를 얻는다."""
    names = list(names)
    for candidate in ("game_type=F", "game_type_F"):
        if candidate in names:
            values = np.asarray(X[:, names.index(candidate)], dtype=np.float64)
            return (values > 0.5).astype(np.float64)
    if "game_type" in names:
        values = np.asarray(X[:, names.index("game_type")])
        return (values.astype(str) == "F").astype(np.float64)
    raise ValueError("SJ base 피처에 game_type=F 원핫 또는 game_type 원본 열이 없음")


def _group_share_and_count(keys: np.ndarray, is_f: np.ndarray):
    keys = np.asarray(keys)
    is_f = np.asarray(is_f, dtype=np.float64)
    finite = np.isfinite(keys.astype(np.float64, copy=False))
    clean_keys = keys[finite]
    clean_f = is_f[finite]
    if not len(clean_keys):
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    unique, inverse = np.unique(clean_keys, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    f_count = np.bincount(inverse, weights=clean_f).astype(np.float64)
    return unique, f_count / count, count


def _lookup(keys: np.ndarray, table_keys: np.ndarray, table_values: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys)
    out = np.full(len(keys), np.nan, dtype=np.float64)
    if not len(table_keys):
        return out
    pos = np.searchsorted(table_keys, keys)
    valid = pos < len(table_keys)
    clipped = np.minimum(pos, len(table_keys) - 1)
    valid &= table_keys[clipped] == keys
    out[valid] = table_values[clipped[valid]]
    return out


def _base_columns(X: np.ndarray, names: Sequence[str]):
    pitcher_id = _column(X, names, "pitcher_id").astype(np.float64, copy=False)
    batter_id = _column(X, names, "batter_id").astype(np.float64, copy=False)
    is_f = _f_indicator(X, names)
    return pitcher_id, batter_id, is_f


def build_walkforward_a3(
    X: np.ndarray,
    names: Sequence[str],
    season: np.ndarray,
) -> np.ndarray:
    """SJ 학습행과 같은 순서로 누수 없는 A3 3열을 반환한다."""
    X = np.asarray(X)
    season = np.asarray(season)
    if len(X) != len(season):
        raise ValueError(f"행 수 불일치: X={len(X):,}, season={len(season):,}")
    pitcher_id, batter_id, is_f = _base_columns(X, names)
    out = np.full((len(X), len(A_COLS)), np.nan, dtype=np.float32)

    for current in np.unique(season[np.isfinite(season)]):
        history = season < current
        rows = season == current
        if not history.any() or not rows.any():
            continue
        p_key, p_share, p_count = _group_share_and_count(pitcher_id[history], is_f[history])
        b_key, b_share, _ = _group_share_and_count(batter_id[history], is_f[history])
        out[rows, 0] = _lookup(pitcher_id[rows], p_key, p_share)
        out[rows, 1] = _lookup(batter_id[rows], b_key, b_share)
        out[rows, 2] = np.log1p(_lookup(pitcher_id[rows], p_key, p_count))
    return out


def append_walkforward_a3(
    X: np.ndarray,
    names: Sequence[str],
    season: np.ndarray,
):
    """기존 열 뒤에 A3를 붙인 행렬과 새 이름을 반환한다."""
    a3 = build_walkforward_a3(X, names, season)
    joined = np.ascontiguousarray(np.concatenate([np.asarray(X), a3], axis=1))
    return joined, [*names, *A_COLS]


def build_inference_lookup(
    X: np.ndarray,
    names: Sequence[str],
    season: np.ndarray,
    cutoff: int = 2024,
) -> dict[str, np.ndarray]:
    """2025 추론 스크립트에 저장할 cutoff 고정 lookup을 만든다."""
    season = np.asarray(season)
    pitcher_id, batter_id, is_f = _base_columns(np.asarray(X), names)
    history = season <= cutoff
    if not history.any():
        raise ValueError(f"season <= {cutoff} 학습행이 없음")
    p_key, p_share, p_count = _group_share_and_count(pitcher_id[history], is_f[history])
    b_key, b_share, _ = _group_share_and_count(batter_id[history], is_f[history])
    return {
        "pitcher_key": p_key,
        "pitcher_f_share": p_share,
        "pitcher_n": p_count,
        "batter_key": b_key,
        "batter_f_share": b_share,
        "cutoff": np.asarray([cutoff], dtype=np.int16),
        "feature_names": np.asarray(A_COLS),
    }


def apply_inference_lookup(
    pitcher_id: np.ndarray,
    batter_id: np.ndarray,
    lookup: Mapping[str, np.ndarray],
) -> np.ndarray:
    """저장된 lookup을 평가행에 행별 적용한다. 다른 평가행은 참조하지 않는다."""
    pitcher_id = np.asarray(pitcher_id, dtype=np.float64)
    batter_id = np.asarray(batter_id, dtype=np.float64)
    if len(pitcher_id) != len(batter_id):
        raise ValueError("pitcher_id와 batter_id 행 수가 다름")
    out = np.empty((len(pitcher_id), len(A_COLS)), dtype=np.float32)
    out[:, 0] = _lookup(pitcher_id, lookup["pitcher_key"], lookup["pitcher_f_share"])
    out[:, 1] = _lookup(batter_id, lookup["batter_key"], lookup["batter_f_share"])
    out[:, 2] = np.log1p(_lookup(pitcher_id, lookup["pitcher_key"], lookup["pitcher_n"]))
    return out


def save_inference_lookup(
    path: str | Path,
    X: np.ndarray,
    names: Sequence[str],
    season: np.ndarray,
    cutoff: int = 2024,
) -> Path:
    """pickle 없는 NPZ로 lookup을 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **build_inference_lookup(X, names, season, cutoff))
    return path


def _self_test() -> None:
    names = ["pitcher_id", "batter_id", "game_type=F"]
    X = np.asarray([
        [10, 100, 0],
        [10, 101, 1],
        [20, 100, 1],
        [10, 100, 0],
        [30, 102, 1],
    ], dtype=np.float64)
    season = np.asarray([2019, 2019, 2020, 2020, 2021])
    got = build_walkforward_a3(X, names, season)
    assert np.isnan(got[:2]).all()
    np.testing.assert_allclose(got[2], [np.nan, 0.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(got[3], [0.5, 0.0, np.log(3.0)], equal_nan=True)
    assert np.isnan(got[4]).all()
    lut = build_inference_lookup(X, names, season, cutoff=2021)
    pred = apply_inference_lookup(np.asarray([10, 999]), np.asarray([100, 999]), lut)
    np.testing.assert_allclose(pred[0], [1 / 3, 1 / 3, np.log(4.0)], rtol=1e-6)
    assert np.isnan(pred[1]).all()
    print("sj_a3_adapter self-test 통과")


if __name__ == "__main__":
    _self_test()
