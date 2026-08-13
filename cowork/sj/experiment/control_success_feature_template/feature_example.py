"""협업용 control_success 피처 모듈 표준 예시.

핵심 계약
---------
1. 입력: 원본 train/test와 같은 행 단위 DataFrame.
2. 출력: row_id + 이 모듈이 만든 피처만 포함한 DataFrame.
3. 행 수와 row_id 순서를 절대 바꾸지 않는다.
4. 모든 피처명은 모듈 고유 prefix로 시작한다.
5. target, row_id 숫자, 다른 test 행의 통계를 피처 계산에 사용하지 않는다.
6. 학습으로 상태를 만드는 피처는 이 템플릿과 분리하고 fold 안에서 fit/transform한다.

이 파일은 실행 가능한 예시인 동시에 팀원이 복사해 사용하는 템플릿이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ID_COLUMN = "row_id"
TARGET_COLUMN = "control_success"


@dataclass(frozen=True)
class FeatureSpec:
    """한 피처 묶음의 소유권·입출력·누수 규칙을 명시한다."""

    name: str
    version: str
    owner: str
    prefix: str
    description: str
    required_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    stateful: bool
    uses_target: bool
    uses_test_aggregate: bool
    time_rule: str


SPEC = FeatureSpec(
    name="demo_situation_history",
    version="1.0.0",
    owner="CHANGE_ME",
    prefix="demo__",
    description="행 내부 경기상황과 제공된 asof 이력만 사용하는 누수 안전 예제",
    required_columns=(
        ID_COLUMN,
        "balls_before",
        "strikes_before",
        "inning",
        "score_diff_pitcher_team",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
        "li",
        "pitcher_hand",
        "batter_hand",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
    ),
    feature_columns=(
        "demo__balls_norm",
        "demo__strikes_norm",
        "demo__count_diff",
        "demo__is_full_count",
        "demo__is_two_strike",
        "demo__abs_score_diff",
        "demo__is_close_game",
        "demo__is_trailing",
        "demo__is_late_inning",
        "demo__late_close_game",
        "demo__runner_pressure",
        "demo__has_risp",
        "demo__is_bases_loaded",
        "demo__is_high_leverage",
        "demo__same_hand_matchup",
        "demo__pitcher_history_log1p",
        "demo__batter_history_log1p",
        "demo__pitcher_history_confidence",
        "demo__pitcher_rate_missing",
        "demo__batter_rate_missing",
        "demo__pitcher_success_shrunk",
        "demo__batter_success_shrunk",
        "demo__recent_pitcher_success",
        "demo__recent_career_success_gap",
        "demo__historical_control_risk",
    ),
    stateful=False,
    uses_target=False,
    uses_test_aggregate=False,
    time_rule="현재 행에 이미 제공된 투구 직전 정보(asof_*)와 행 내부 상황만 사용",
)


PARAMS = {
    "prior_success": 0.5,
    "pitcher_smoothing": 200.0,
    "batter_smoothing": 300.0,
    "high_leverage_threshold": 2.0,
    "close_game_run_threshold": 1,
    "late_inning_threshold": 7,
    "recent_weight": 0.6,
}


def _validate_spec(spec: FeatureSpec = SPEC) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*__", spec.prefix):
        raise ValueError(f"prefix는 'owner_topic__' 형태여야 합니다: {spec.prefix!r}")
    if spec.owner == "CHANGE_ME":
        # 예제 실행은 허용하되 manifest에 그대로 남겨 교체 필요성을 보이게 한다.
        pass
    if len(spec.feature_columns) != len(set(spec.feature_columns)):
        raise ValueError("feature_columns에 중복 이름이 있습니다.")
    bad = [c for c in spec.feature_columns if not c.startswith(spec.prefix)]
    if bad:
        raise ValueError(f"prefix 규칙을 위반한 피처: {bad}")
    if ID_COLUMN in spec.feature_columns or TARGET_COLUMN in spec.feature_columns:
        raise ValueError("row_id와 target은 feature_columns에 포함할 수 없습니다.")
    if spec.uses_target or spec.uses_test_aggregate:
        raise ValueError("이 예제는 target/test 집계를 사용하지 않아야 합니다.")


def _require_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"필수 입력 컬럼이 없습니다: {missing}")


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    """원본 dtype 차이에 흔들리지 않도록 수치형으로 읽는다."""

    return pd.to_numeric(df[column], errors="coerce")


def _shrunk_rate(
    rate: pd.Series,
    n: pd.Series,
    prior: float,
    smoothing: float,
) -> pd.Series:
    n_clean = n.fillna(0).clip(lower=0).astype("float64")
    rate_clean = rate.fillna(prior).clip(0, 1).astype("float64")
    return (n_clean * rate_clean + smoothing * prior) / (n_clean + smoothing)


def build_features(df: pd.DataFrame, spec: FeatureSpec = SPEC) -> pd.DataFrame:
    """한 행을 다른 행과 섞지 않고 누수 안전 파생 피처를 만든다.

    train과 test를 각각 독립적으로 호출해야 한다. 반환값은 반드시
    ``row_id + spec.feature_columns`` 계약을 만족한다.
    """

    _validate_spec(spec)
    _require_columns(df, spec.required_columns)
    # pandas 대입 정렬이 외부 index를 따라가지 않도록 현재 행 순서만 보존해 초기화한다.
    # 이 줄이 없으면 셔플·필터 후 남은 원본 index 때문에 피처가 다른 row_id에 붙을 수 있다.
    df = df.reset_index(drop=True)

    balls = _num(df, "balls_before")
    strikes = _num(df, "strikes_before")
    inning = _num(df, "inning")
    score_diff = _num(df, "score_diff_pitcher_team")
    li = _num(df, "li")
    on_1b = _num(df, "runner_on_1b").fillna(0)
    on_2b = _num(df, "runner_on_2b").fillna(0)
    on_3b = _num(df, "runner_on_3b").fillna(0)
    pitcher_hand = df["pitcher_hand"]
    batter_hand = df["batter_hand"]

    pitcher_n = _num(df, "asof_pitcher_n").fillna(0).clip(lower=0)
    batter_n = _num(df, "asof_batter_n").fillna(0).clip(lower=0)
    pitcher_rate = _num(df, "asof_pitcher_success_rate")
    batter_rate = _num(df, "asof_batter_success_rate")
    reverse_rate = _num(df, "asof_pitcher_reverse_rate")
    middle_rate = _num(df, "asof_pitcher_middle_rate")

    pitcher_shrunk = _shrunk_rate(
        pitcher_rate,
        pitcher_n,
        PARAMS["prior_success"],
        PARAMS["pitcher_smoothing"],
    )
    batter_shrunk = _shrunk_rate(
        batter_rate,
        batter_n,
        PARAMS["prior_success"],
        PARAMS["batter_smoothing"],
    )
    recent_rate = (
        _num(df, "asof_pitcher_prev5_game_success_rate")
        .combine_first(_num(df, "asof_pitcher_prev3_game_success_rate"))
        .combine_first(pitcher_rate)
        .fillna(PARAMS["prior_success"])
        .clip(0, 1)
    )
    recent_career_blend = (
        PARAMS["recent_weight"] * recent_rate
        + (1.0 - PARAMS["recent_weight"]) * pitcher_shrunk
    )

    out = pd.DataFrame({ID_COLUMN: df[ID_COLUMN].astype("string").to_numpy(copy=True)})
    out["demo__balls_norm"] = (balls / 3.0).astype("float32")
    out["demo__strikes_norm"] = (strikes / 2.0).astype("float32")
    out["demo__count_diff"] = (balls - strikes).astype("float32")
    out["demo__is_full_count"] = (balls.eq(3) & strikes.eq(2)).astype("int8")
    out["demo__is_two_strike"] = strikes.eq(2).astype("int8")

    out["demo__abs_score_diff"] = score_diff.abs().astype("float32")
    out["demo__is_close_game"] = score_diff.abs().le(
        PARAMS["close_game_run_threshold"]
    ).astype("int8")
    out["demo__is_trailing"] = score_diff.lt(0).astype("int8")
    out["demo__is_late_inning"] = inning.ge(PARAMS["late_inning_threshold"]).astype(
        "int8"
    )
    out["demo__late_close_game"] = (
        inning.ge(PARAMS["late_inning_threshold"])
        & score_diff.abs().le(PARAMS["close_game_run_threshold"])
    ).astype("int8")

    out["demo__runner_pressure"] = (on_1b + 2 * on_2b + 3 * on_3b).astype("float32")
    out["demo__has_risp"] = (on_2b.eq(1) | on_3b.eq(1)).astype("int8")
    out["demo__is_bases_loaded"] = (
        on_1b.eq(1) & on_2b.eq(1) & on_3b.eq(1)
    ).astype("int8")
    out["demo__is_high_leverage"] = li.ge(PARAMS["high_leverage_threshold"]).astype(
        "int8"
    )
    out["demo__same_hand_matchup"] = pitcher_hand.eq(batter_hand).astype("int8")

    out["demo__pitcher_history_log1p"] = np.log1p(pitcher_n).astype("float32")
    out["demo__batter_history_log1p"] = np.log1p(batter_n).astype("float32")
    out["demo__pitcher_history_confidence"] = (
        pitcher_n / (pitcher_n + PARAMS["pitcher_smoothing"])
    ).astype("float32")
    out["demo__pitcher_rate_missing"] = pitcher_rate.isna().astype("int8")
    out["demo__batter_rate_missing"] = batter_rate.isna().astype("int8")
    out["demo__pitcher_success_shrunk"] = pitcher_shrunk.astype("float32")
    out["demo__batter_success_shrunk"] = batter_shrunk.astype("float32")
    out["demo__recent_pitcher_success"] = recent_career_blend.astype("float32")
    out["demo__recent_career_success_gap"] = (recent_rate - pitcher_shrunk).astype(
        "float32"
    )
    out["demo__historical_control_risk"] = (
        reverse_rate.fillna(0) + middle_rate.fillna(0)
    ).astype("float32")

    return out[[ID_COLUMN, *spec.feature_columns]]


def validate_feature_block(
    source: pd.DataFrame,
    block: pd.DataFrame,
    spec: FeatureSpec = SPEC,
) -> dict:
    """모델 팀에 전달하기 전에 반드시 통과해야 하는 계약 검사."""

    _validate_spec(spec)
    expected = [ID_COLUMN, *spec.feature_columns]
    errors: list[str] = []

    if list(block.columns) != expected:
        errors.append("출력 컬럼 또는 순서가 SPEC과 다릅니다.")
    if len(source) != len(block):
        errors.append(f"행 수가 바뀌었습니다: source={len(source)}, block={len(block)}")
    if source[ID_COLUMN].isna().any() or not source[ID_COLUMN].is_unique:
        errors.append("입력 row_id가 결측이거나 중복입니다.")
    source_ids = source[ID_COLUMN].astype("string").reset_index(drop=True)
    block_ids = block[ID_COLUMN].astype("string").reset_index(drop=True)
    if not source_ids.equals(block_ids):
        errors.append("row_id 값 또는 순서가 바뀌었습니다.")
    if block.columns.duplicated().any():
        errors.append("출력 컬럼명이 중복됩니다.")
    if TARGET_COLUMN in block.columns:
        errors.append("출력에 target이 포함되어 있습니다.")

    feature_frame = block.drop(columns=ID_COLUMN, errors="ignore")
    non_numeric = [
        c for c in feature_frame.columns if not pd.api.types.is_numeric_dtype(feature_frame[c])
    ]
    if non_numeric:
        errors.append(f"숫자가 아닌 피처가 있습니다: {non_numeric}")
    numeric = feature_frame.select_dtypes(include=[np.number])
    infinity_n = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
    if infinity_n:
        errors.append(f"무한대 값이 {infinity_n}개 있습니다.")

    if errors:
        raise AssertionError("\n- " + "\n- ".join(errors))

    return {
        "rows": int(len(block)),
        "features": int(len(spec.feature_columns)),
        "row_id_unique": bool(block[ID_COLUMN].is_unique),
        "null_cells": int(feature_frame.isna().sum().sum()),
        "infinite_cells": infinity_n,
        "all_numeric": True,
        "prefix_ok": True,
        "uses_target": spec.uses_target,
        "uses_test_aggregate": spec.uses_test_aggregate,
    }


def assert_row_independence(
    source: pd.DataFrame,
    spec: FeatureSpec = SPEC,
    max_rows: int = 5_000,
) -> None:
    """행 순서·다른 test 행의 존재 여부에 피처가 의존하지 않는지 검사한다."""

    check = source.head(max_rows).copy()
    base = build_features(check, spec).sort_values(ID_COLUMN).reset_index(drop=True)

    shuffled_source = check.sample(frac=1.0, random_state=20260805)
    shuffled = (
        build_features(shuffled_source, spec)
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )
    assert_frame_equal(base, shuffled, check_dtype=True, check_exact=True)

    subset_source = check.iloc[::2].copy()
    subset = (
        build_features(subset_source, spec)
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )
    expected_subset = (
        base[base[ID_COLUMN].isin(subset_source[ID_COLUMN].astype("string"))]
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )
    assert_frame_equal(expected_subset, subset, check_dtype=True, check_exact=True)


def feature_summary(block: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in block.columns:
        if column == ID_COLUMN:
            continue
        series = block[column]
        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "missing_n": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100.0),
                "nunique": int(series.nunique(dropna=True)),
                "min": float(series.min()) if series.notna().any() else None,
                "median": float(series.median()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def merge_feature_blocks(
    base_ids: pd.DataFrame,
    blocks: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """여러 팀원의 피처 묶음을 row_id 기준으로 안전하게 조립한다."""

    if list(base_ids.columns) != [ID_COLUMN]:
        raise ValueError(f"base_ids는 [{ID_COLUMN!r}] 한 컬럼이어야 합니다.")
    if base_ids[ID_COLUMN].isna().any() or not base_ids[ID_COLUMN].is_unique:
        raise ValueError("base_ids의 row_id는 결측 없이 유일해야 합니다.")

    merged = base_ids.copy()
    seen = {ID_COLUMN}
    for index, block in enumerate(blocks, start=1):
        if ID_COLUMN not in block:
            raise ValueError(f"{index}번째 block에 row_id가 없습니다.")
        if block[ID_COLUMN].isna().any() or not block[ID_COLUMN].is_unique:
            raise ValueError(f"{index}번째 block의 row_id가 결측 또는 중복입니다.")
        duplicate_features = (set(block.columns) - {ID_COLUMN}) & seen
        if duplicate_features:
            raise ValueError(f"피처명이 충돌합니다: {sorted(duplicate_features)}")
        merged = merged.merge(
            block,
            on=ID_COLUMN,
            how="left",
            sort=False,
            validate="one_to_one",
        )
        seen.update(block.columns)

    if len(merged) != len(base_ids):
        raise AssertionError("피처 조립 중 행 수가 바뀌었습니다.")
    if not merged[ID_COLUMN].astype("string").equals(
        base_ids[ID_COLUMN].astype("string").reset_index(drop=True)
    ):
        raise AssertionError("피처 조립 중 row_id 순서가 바뀌었습니다.")
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    train_block: pd.DataFrame,
    test_block: pd.DataFrame,
    train_validation: dict,
    test_validation: dict,
    output_format: str,
) -> dict:
    script_path = Path(__file__).resolve()
    return {
        "feature_spec": asdict(SPEC),
        "parameters": PARAMS,
        "train_validation": train_validation,
        "test_validation": test_validation,
        "train_rows": int(len(train_block)),
        "test_rows": int(len(test_block)),
        "output_format": output_format,
        "script_path": str(script_path),
        "script_sha256": _sha256(script_path),
        "feature_schema_sha256": hashlib.sha256(
            "|".join(SPEC.feature_columns).encode("utf-8")
        ).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_checklist": {
            "row_order_preserved": True,
            "target_not_used": not SPEC.uses_target,
            "test_aggregate_not_used": not SPEC.uses_test_aggregate,
            "row_independence_tested": True,
            "stateful": SPEC.stateful,
        },
    }


def _write_block(block: pd.DataFrame, path_without_suffix: Path, output_format: str) -> Path:
    if output_format == "parquet":
        path = path_without_suffix.with_suffix(".parquet")
        block.to_parquet(path, index=False)
    elif output_format == "csv":
        path = path_without_suffix.with_suffix(".csv")
        block.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"지원하지 않는 출력 형식: {output_format}")
    return path


def run_cli(args: argparse.Namespace) -> dict:
    read_kwargs = {"usecols": list(SPEC.required_columns), "low_memory": False}
    if args.nrows is not None:
        read_kwargs["nrows"] = args.nrows

    train = pd.read_csv(args.train, **read_kwargs)
    test = pd.read_csv(args.test, **read_kwargs)

    train_block = build_features(train)
    test_block = build_features(test)
    train_validation = validate_feature_block(train, train_block)
    test_validation = validate_feature_block(test, test_block)
    assert_row_independence(train)
    assert_row_independence(test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{SPEC.name}_v{SPEC.version.replace('.', '_')}"
    train_path = _write_block(train_block, output_dir / f"{stem}__train", args.format)
    test_path = _write_block(test_block, output_dir / f"{stem}__test", args.format)

    summary = feature_summary(train_block)
    summary_path = output_dir / f"{stem}__feature_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    manifest = build_manifest(
        train_block,
        test_block,
        train_validation,
        test_validation,
        args.format,
    )
    manifest.update(
        {
            "train_output": str(train_path.resolve()),
            "test_output": str(test_path.resolve()),
            "feature_summary": str(summary_path.resolve()),
        }
    )
    manifest_path = output_dir / f"{stem}__manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"manifest": str(manifest_path), **manifest}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_root / "data" / "train.csv")
    parser.add_argument("--test", type=Path, default=project_root / "data" / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=here / "feature_outputs")
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="스모크 테스트용 행 수. 생략하면 전체 train/test를 처리합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run_cli(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
