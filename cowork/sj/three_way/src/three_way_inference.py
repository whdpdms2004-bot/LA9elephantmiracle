"""2019~2024 전체 학습 3WAY 포함-배제 모델의 제출용 추론 스크립트."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "output"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_direct_products(frame: pd.DataFrame) -> None:
    added = {}
    pairs = [
        ("asof_pitcher_success_rate", "asof_batter_success_rate"),
        ("asof_pitcher_middle_rate", "asof_batter_middle_rate"),
        ("asof_pitcher_success_rate", "li"),
        ("asof_pitcher_reverse_rate", "asof_pitcher_fastball_rate"),
    ]
    for index, (left, right) in enumerate(pairs):
        added[f"sx_product_{index:02d}"] = (
            pd.to_numeric(frame[left], errors="coerce").to_numpy(np.float64)
            * pd.to_numeric(frame[right], errors="coerce").to_numpy(np.float64))
    count = np.log1p(np.nan_to_num(
        pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(np.float64),
        nan=0.0))
    added["sx_pitcher_success_x_logn"] = (
        frame["asof_pitcher_success_rate"].to_numpy(np.float64) * count)
    added["sx_pitcher_minus_batter_success"] = (
        frame["asof_pitcher_success_rate"].to_numpy(np.float64)
        - frame["asof_batter_success_rate"].to_numpy(np.float64))
    batter_success = np.clip(
        frame["asof_batter_success_rate"].to_numpy(np.float64), 1e-3, None)
    added["sx_pitcher_over_batter_success"] = np.divide(
        frame["asof_pitcher_success_rate"].to_numpy(np.float64), batter_success)
    balls = frame["balls_before"].to_numpy(np.float64)
    strikes = frame["strikes_before"].to_numpy(np.float64)
    inning = frame["inning"].to_numpy(np.float64)
    leverage = frame["li"].to_numpy(np.float64)
    score_diff = frame["score_diff_pitcher_team"].to_numpy(np.float64)
    close = (np.abs(score_diff) <= 1).astype(float)
    added.update({
        "sx_count_margin": balls - strikes,
        "sx_two_strike": (strikes == 2).astype(float),
        "sx_three_ball": (balls == 3).astype(float),
        "sx_full_count": ((balls == 3) & (strikes == 2)).astype(float),
        "sx_close_game": close,
        "sx_late_close": (inning >= 7).astype(float) * close,
        "sx_li_close": leverage * close,
        "sx_li_count_margin": leverage * (balls - strikes),
        "sx_reverse_two_strike": (
            frame["asof_pitcher_reverse_rate"].to_numpy(np.float64)
            * (strikes == 2)),
        "sx_ball_three_ball": (
            frame["asof_pitcher_ball_rate"].to_numpy(np.float64)
            * (balls == 3)),
        "sx_prev5_success_x_balls": (
            frame["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
            * balls),
    })
    for name, values in added.items():
        frame[name] = np.asarray(values, dtype=np.float32)


def build_base_features(test: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    base_meta = metadata["base_metadata"]
    base_runtime = load_module(
        "three_way_base_runtime", MODEL_DIR / base_meta["base_runtime_file"])
    base_runtime.MODEL_DIR = MODEL_DIR
    engineered = base_runtime.build_feature_frame(test, "enhanced", base_meta)
    add_direct_products(engineered)

    component = load_module(
        "three_way_component_runtime",
        MODEL_DIR / base_meta["component_runtime_file"])
    assets = base_meta["component_assets"]
    built = component.build(
        test[base_meta["raw_columns"]],
        base_meta["component_spec"],
        pd.read_csv(MODEL_DIR / assets["platoon"]),
        pd.read_csv(MODEL_DIR / assets["bat_platoon"]),
        pd.read_csv(MODEL_DIR / assets["count_platoon"]),
        pd.read_csv(MODEL_DIR / assets["inning_platoon"]),
    )
    renamed = built.rename(columns={column: f"sx_cf_{column}" for column in built})
    component_frame = renamed[base_meta["component_feature_columns"]].astype("float32")
    engineered = pd.concat(
        [engineered.reset_index(drop=True), component_frame.reset_index(drop=True)],
        axis=1)
    missing = [column for column in base_meta["feature_columns"]
               if column not in engineered]
    if missing:
        raise ValueError(f"missing base features: {missing}")
    return engineered


def load_id_lookups(metadata: dict) -> dict[str, pd.Series]:
    lookups = {}
    for column, filename in metadata["id_frequency_files"].items():
        table = pd.read_csv(MODEL_DIR / filename)
        lookups[column] = table.set_index(column)["frequency"]
    return lookups


def predict_component(
        frame: pd.DataFrame,
        target_spec: dict,
        metadata: dict,
        runtime,
        id_lookups: dict[str, pd.Series]) -> np.ndarray:
    enriched = runtime.add_target_features(
        frame,
        target_spec["combo"],
        metadata["transform_spec"],
        id_lookups,
        prediction_season=int(metadata["prediction_season"]),
    )
    features = target_spec["feature_columns"]
    missing = [column for column in features if column not in enriched]
    if missing:
        raise ValueError(f"missing {target_spec['target']} features: {missing}")
    model_frame = enriched[features].copy()
    categorical = target_spec["categorical_columns"]
    for column in categorical:
        model_frame[column] = model_frame[column].fillna("__MISSING__").astype(str)

    from catboost import CatBoostClassifier, Pool
    baseline = np.full(len(model_frame), float(target_spec["baseline_logit"]))
    pool = Pool(model_frame, cat_features=categorical, baseline=baseline)
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / target_spec["model_file"]))
    return model.predict_proba(pool, thread_count=6)[:, 1].astype(np.float64)


def main() -> None:
    metadata_path = MODEL_DIR / "three_way_metadata.json"
    test_path = DATA_DIR / "test.csv"
    if not metadata_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("model metadata or data/test.csv is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test = pd.read_csv(test_path)
    if "control_success" in test.columns:
        test = test.drop(columns=["control_success"])
    frame = build_base_features(test, metadata)
    runtime = load_module(
        "three_way_runtime", MODEL_DIR / metadata["three_way_runtime_file"])
    id_lookups = load_id_lookups(metadata)
    component = {
        spec["target"]: predict_component(
            frame, spec, metadata, runtime, id_lookups)
        for spec in metadata["models"]
    }
    probability = np.clip(
        1.0 - (
            component["middle"] + component["reverse"]
            - component["mr"] + component["outside"]),
        1e-7,
        1.0 - 1e-7,
    )
    if len(probability) != len(test) or not np.isfinite(probability).all():
        raise RuntimeError("invalid prediction output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_id": test["row_id"].to_numpy(),
        "control_success": probability,
    }).to_csv(OUTPUT_DIR / "submission.csv", index=False)


if __name__ == "__main__":
    main()
