"""사건별 OOF 보정 후 실패 합집합을 계산하는 제출용 추론 스크립트."""
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
EPS = 1e-7


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, np.float64), EPS, 1.0 - EPS)
    return np.log(probability / (1.0 - probability))


def sigmoid(score: np.ndarray) -> np.ndarray:
    score = np.clip(np.asarray(score, np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-score))


def calibrate(probability: np.ndarray, spec: dict, strength: float) -> np.ndarray:
    beta = np.asarray(spec["beta"], np.float64)
    original = logit(probability)
    fitted = beta[0] + beta[1] * original
    return sigmoid((1.0 - strength) * original + strength * fitted)


def main() -> None:
    teacher = load_module(
        "three_way_teacher_inference", MODEL_DIR / "three_way_teacher_inference.py")
    teacher.BASE = BASE
    teacher.MODEL_DIR = MODEL_DIR
    teacher.DATA_DIR = DATA_DIR
    teacher.OUTPUT_DIR = OUTPUT_DIR

    metadata = json.loads(
        (MODEL_DIR / "three_way_metadata.json").read_text(encoding="utf-8"))
    fusion = json.loads(
        (MODEL_DIR / "event_calibration.json").read_text(encoding="utf-8"))
    test = pd.read_csv(DATA_DIR / "test.csv")
    if "control_success" in test.columns:
        test = test.drop(columns=["control_success"])
    frame = teacher.build_base_features(test, metadata)
    runtime = teacher.load_module(
        "three_way_runtime", MODEL_DIR / metadata["three_way_runtime_file"])
    id_lookups = teacher.load_id_lookups(metadata)
    raw = {
        spec["target"]: teacher.predict_component(
            frame, spec, metadata, runtime, id_lookups)
        for spec in metadata["models"]
    }

    strength = float(fusion["strength"])
    component = {
        name: calibrate(raw[name], fusion["calibrators"][name], strength)
        for name in ("middle", "reverse", "outside", "mr")
    }
    lower = np.maximum(0.0, component["middle"] + component["reverse"] - 1.0)
    upper = np.minimum(component["middle"], component["reverse"])
    component["mr"] = np.clip(component["mr"], lower, upper)
    failure_union = (
        component["middle"] + component["reverse"]
        - component["mr"] + component["outside"])
    probability = np.clip(1.0 - failure_union, EPS, 1.0 - EPS)
    if len(probability) != len(test) or not np.isfinite(probability).all():
        raise RuntimeError("invalid prediction output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_id": test["row_id"].to_numpy(),
        "control_success": probability,
    }).to_csv(OUTPUT_DIR / "submission.csv", index=False)


if __name__ == "__main__":
    main()
