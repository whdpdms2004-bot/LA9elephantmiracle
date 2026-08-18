"""엄격 forward-OOF F1 CatBoost 단독 모델의 제출용 추론 스크립트 원본."""
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


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def sigmoid(value):
    z = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_direct_products(frame: pd.DataFrame) -> list[str]:
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
    n = np.log1p(np.nan_to_num(
        pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(np.float64),
        nan=0.0))
    added["sx_pitcher_success_x_logn"] = (
        frame["asof_pitcher_success_rate"].to_numpy(np.float64) * n)
    added["sx_pitcher_minus_batter_success"] = (
        frame["asof_pitcher_success_rate"].to_numpy(np.float64)
        - frame["asof_batter_success_rate"].to_numpy(np.float64))
    added["sx_pitcher_over_batter_success"] = (
        frame["asof_pitcher_success_rate"].to_numpy(np.float64)
        / np.clip(frame["asof_batter_success_rate"].to_numpy(np.float64), 1e-3, None))
    balls = frame["balls_before"].to_numpy(np.float64)
    strikes = frame["strikes_before"].to_numpy(np.float64)
    inning = frame["inning"].to_numpy(np.float64)
    li = frame["li"].to_numpy(np.float64)
    diff = frame["score_diff_pitcher_team"].to_numpy(np.float64)
    close = (np.abs(diff) <= 1).astype(float)
    added.update({
        "sx_count_margin": balls - strikes,
        "sx_two_strike": (strikes == 2).astype(float),
        "sx_three_ball": (balls == 3).astype(float),
        "sx_full_count": ((balls == 3) & (strikes == 2)).astype(float),
        "sx_close_game": close,
        "sx_late_close": (inning >= 7).astype(float) * close,
        "sx_li_close": li * close,
        "sx_li_count_margin": li * (balls - strikes),
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
    return list(added)


def build_component_features(test: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    component = load_module(
        "f1_component_runtime", MODEL_DIR / metadata["component_runtime_file"])
    spec = metadata["component_spec"]
    assets = metadata["component_assets"]
    platoon = pd.read_csv(MODEL_DIR / assets["platoon"])
    bat_platoon = pd.read_csv(MODEL_DIR / assets["bat_platoon"])
    count_platoon = pd.read_csv(MODEL_DIR / assets["count_platoon"])
    inning_platoon = pd.read_csv(MODEL_DIR / assets["inning_platoon"])
    raw_columns = metadata["raw_columns"]
    built = component.build(
        test[raw_columns], spec, platoon, bat_platoon,
        count_platoon, inning_platoon)
    renamed = built.rename(columns={column: f"sx_cf_{column}" for column in built})
    return renamed[metadata["component_feature_columns"]].astype("float32")


def main() -> None:
    metadata_path = MODEL_DIR / "f1_metadata.json"
    test_path = DATA_DIR / "test.csv"
    if not metadata_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("model metadata or data/test.csv is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test = pd.read_csv(test_path)

    base_runtime = load_module(
        "f1_base_runtime", MODEL_DIR / metadata["base_runtime_file"])
    base_runtime.MODEL_DIR = MODEL_DIR
    engineered = base_runtime.build_feature_frame(test, "enhanced", metadata)
    add_direct_products(engineered)
    component = build_component_features(test, metadata)
    engineered = pd.concat(
        [engineered.reset_index(drop=True), component.reset_index(drop=True)], axis=1)

    features = metadata["feature_columns"]
    missing = [column for column in features if column not in engineered]
    if missing:
        raise ValueError(f"missing engineered features: {missing}")
    categorical = [
        column for column in metadata["categorical_columns"] if column in features]
    model_frame = engineered[features].copy()
    for column in categorical:
        model_frame[column] = model_frame[column].fillna("__MISSING__").astype(str)

    from catboost import CatBoostClassifier, Pool
    pool = Pool(model_frame, cat_features=categorical)
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / metadata["model_file"]))
    prediction = model.predict_proba(pool, thread_count=6)[:, 1]
    prediction = sigmoid(logit(prediction) + float(metadata["season_logit_offset"]))
    prediction = np.clip(prediction, 1e-7, 1.0 - 1e-7)
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise RuntimeError("invalid prediction output")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_id": test["row_id"].to_numpy(),
        "control_success": prediction,
    }).to_csv(OUTPUT_DIR / "submission.csv", index=False)


if __name__ == "__main__":
    main()
