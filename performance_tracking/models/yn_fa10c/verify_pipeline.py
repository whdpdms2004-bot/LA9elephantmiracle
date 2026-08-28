"""학습 코드의 피처·lookup 계약을 기존 FA10C 제출 ZIP과 대조한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import features
import pipeline as fa


def load_inference_module(path: Path):
    spec = importlib.util.spec_from_file_location("fa10c_inference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def compare_dict_float(left: dict, right: dict, label: str) -> None:
    if set(left) != set(right):
        raise AssertionError(f"{label} 키 불일치")
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, list):
            if not np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True):
                raise AssertionError(f"{label}[{key}] 불일치")
        elif not np.isclose(float(a), float(b), equal_nan=True):
            raise AssertionError(f"{label}[{key}] 불일치: {a} != {b}")


def run(args: argparse.Namespace) -> None:
    data_dir = fa.find_data_dir(args.data_dir)
    reference_zip = Path(args.reference_zip).resolve()
    with zipfile.ZipFile(reference_zip) as archive:
        names = {item.filename for item in archive.infolist() if not item.is_dir()}
        if len(names) != 63:
            raise AssertionError(f"참조 ZIP 엔트리가 63개가 아님: {len(names)}")
        meta = json.loads(archive.read("model/meta.json"))

    assert len(meta["raw_features"]) == 47
    assert meta["new_feature_cols"] == fa.A_COLS
    assert meta["lgb_seeds"] == fa.LGB_SEEDS
    assert meta["cb_seeds"] == fa.CB_SEEDS
    assert meta["team_cb_seeds"] == fa.CB_SEEDS
    assert meta["team_representation_alpha"] == fa.TEAM_ALPHA
    assert meta["ensemble_lgb_weight"] == fa.LGB_WEIGHT
    assert max(meta["iso_y"]) == fa.ISO_CAP

    raw_features = fa.raw_feature_list(data_dir)
    if raw_features != meta["raw_features"]:
        raise AssertionError("test.csv 헤더에서 얻은 raw feature 순서가 참조 meta와 다름")
    full = fa.load_train(data_dir, raw_features)
    local_stats = features.fit_stats(full)
    compare_dict_float(local_stats, meta["stats"], "stats")

    local_lookup = fa.build_inference_lookup(full, cutoff=2024)
    for key in ("pitcher_share", "pitcher_n", "batter_share"):
        compare_dict_float(local_lookup[key], meta["futures_lookup"][key], f"lookup.{key}")

    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    local_x, _ = fa.prepare(
        test,
        full,
        raw_features,
        local_stats,
        features.get_categorical_columns(),
        meta["category_levels"],
    )
    inference = load_inference_module(Path(__file__).resolve().parent / "script_fa10c_inference.py")
    reference_x = inference.build_features(
        test,
        meta["raw_features"],
        meta["stats"],
        meta["category_levels"],
        meta["futures_lookup"],
        meta["new_feature_cols"],
    )
    if list(local_x.columns) != list(reference_x.columns):
        raise AssertionError("학습/추론 피처 열 순서 불일치")
    for col in local_x:
        if str(local_x[col].dtype) == "category":
            if not np.array_equal(local_x[col].astype(str), reference_x[col].astype(str)):
                raise AssertionError(f"범주형 피처 불일치: {col}")
        else:
            if not np.allclose(
                local_x[col].to_numpy(dtype=float),
                reference_x[col].to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise AssertionError(f"수치 피처 불일치: {col}")

    print(
        "검증 통과: 참조 ZIP 63엔트리, raw 47열, 모델 71피처, "
        "stats/lookup 0오차, 학습-추론 피처 순서·값 일치"
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--reference-zip",
        default=str(here.parent.parent / "yn_fa10c.zip"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
