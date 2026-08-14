from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
FINAL_DIR = WORK / "final" / "robust_matchup_v1"
BASE_MODEL_DIR = WORK / "submit007_check" / "model"
MODEL_BANK = MODEL_DIR / "large_xgb_final_model_bank"
BASE_ZIP = ROOT / "submit" / "2026-08-12" / "submit_013.zip"
OUTPUT_DIR = ROOT / "submit" / "2026-08-12"
DESTINATION = OUTPUT_DIR / "submit_014.zip"
INFERENCE_SCRIPT = Path(__file__).resolve().parent / "submission_script_matchup.py"
TARGET = "control_success"
SEED = 2026
TRIAL_NUMBER = 93
ANCHOR_ITERATIONS = 2642
LARGE_ITERATIONS = 2398
LARGE_BASE_WEIGHT = 0.90
FINAL_BASE_WEIGHT = 0.3595515607106235
FINAL_CURRENT_WEIGHT = 0.25166569203957306
FINAL_LARGE_WEIGHT = 0.3887827472498035
SUCCESS_CONFIG = "m_com_gmm_kme_b3r4_l1000_h1_d72788b5"
SUCCESS_FEATURES = [
    "match_pair_delta",
    "match_pair_delta_reliability",
    "match_pair_delta_rate",
    "match_pair_known",
]
REVERSE_FEATURES = [
    "reverse_pair_delta",
    "reverse_pair_delta_reliability",
    "reverse_pair_rate",
    "reverse_pair_known",
]
REVERSE_SEEDS = [17, 2026, 4099]

sys.path.insert(0, str(MODEL_DIR))
from benchmark_insight_features import (  # noqa: E402
    add_calibration_features,
    build_past_only_lookups,
)
from build_final_two_submissions import (  # noqa: E402
    encode_xgboost_full,
    smoke_test,
    write_zip,
)
from run_optuna_enhanced import load_enhanced_frame  # noqa: E402
from run_optuna_family import recency_weights  # noqa: E402


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def load_trial93():
    study_name = "xgboost_v2r200_tm500_local_2024"
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{(MODEL_DIR / f'{study_name}.db').as_posix()}",
    )
    return next(item for item in study.trials if item.number == TRIAL_NUMBER)


def load_training_frame():
    frame, base_features = load_enhanced_frame()
    labels = pd.read_parquet(MODEL_DIR / "failure_component_labels.parquet")
    lookups, audit = build_past_only_lookups(frame, labels)
    if not all(
        item["source_season"] is None
        or item["source_season"] < item["target_season"]
        for item in audit
    ):
        raise RuntimeError("Past-only feature audit failed")
    frame, _, prior_columns = add_calibration_features(frame, lookups)
    adjusted = [
        column
        for column in prior_columns
        if column.startswith(("pitcher_success_", "batter_success_"))
        and "_adjusted_smoothed_" in column
    ]
    features = list(dict.fromkeys(base_features + adjusted))
    if len(adjusted) != 2 or len(features) != 211:
        raise RuntimeError(
            f"Unexpected adjusted feature set: adjusted={adjusted}, total={len(features)}"
        )
    return frame, features


def train_models():
    frame, features = load_training_frame()
    matrix, mappings = encode_xgboost_full(frame, features)
    trial = load_trial93()
    base_params = dict(trial.params)
    half_life = float(base_params.pop("half_life"))
    target = frame[TARGET].to_numpy("int8")
    weights = recency_weights(frame["season"], 2025, half_life)
    MODEL_BANK.mkdir(parents=True, exist_ok=True)

    large_params = deepcopy(base_params)
    large_params.update(
        {
            "learning_rate": float(base_params["learning_rate"]) / 1.10,
            "max_depth": 8,
            "max_leaves": 24,
            "reg_lambda": float(base_params["reg_lambda"]) * 1.10,
            "subsample": 0.90,
            "colsample_bytree": 0.70,
        }
    )
    specs = [
        {
            "name": "xgb_insight_anchor_t93",
            "filename": "large_anchor_t93.ubj",
            "params": base_params,
            "iterations": ANCHOR_ITERATIONS,
        },
        {
            "name": "xgb_insight_large_m24d",
            "filename": "large_m24d_t93.ubj",
            "params": large_params,
            "iterations": LARGE_ITERATIONS,
        },
    ]
    models = []
    paths = []
    for spec in specs:
        params = deepcopy(spec["params"])
        params["n_estimators"] = int(spec["iterations"])
        model = XGBClassifier(
            **params,
            grow_policy="lossguide",
            objective="binary:logistic",
            eval_metric="rmse",
            tree_method="hist",
            device="cuda",
            random_state=SEED + 2025,
            n_jobs=6,
        )
        started = time.time()
        model.fit(matrix, target, sample_weight=weights, verbose=False)
        path = MODEL_BANK / spec["filename"]
        model.save_model(str(path))
        elapsed = time.time() - started
        models.append(
            {
                "model_name": spec["name"],
                "family": "xgboost",
                "feature_version": "insight_adjusted",
                "trial": TRIAL_NUMBER,
                "study": "large_xgb_capacity_search",
                "filenames": [spec["filename"]],
                "seeds": [SEED + 2025],
                "iterations": [int(spec["iterations"])],
                "half_life": half_life,
                "weight": 1.0,
                "train_elapsed_sec": elapsed,
                "max_leaves": int(params["max_leaves"]),
                "max_depth": int(params["max_depth"]),
            }
        )
        paths.append(path)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del frame, matrix, target, weights
    gc.collect()
    return models, paths, features, mappings


def load_large_oof_2024():
    prediction_dir = MODEL_DIR / "large_xgb" / "predictions"
    anchor = pd.read_parquet(
        prediction_dir / "anchor_logloss_f2024_s0.parquet"
    ).sort_values("row_id").reset_index(drop=True)
    large = pd.read_parquet(
        prediction_dir / "moderate24_diverse_f2024_s0.parquet"
    ).sort_values("row_id").reset_index(drop=True)
    if not anchor[["row_id", "season", TARGET]].equals(
        large[["row_id", "season", TARGET]]
    ):
        raise RuntimeError("2024 OOF alignment failed")
    source = large["prediction"].to_numpy("float64")
    reference = anchor["prediction"].to_numpy("float64")
    prediction = reference + LARGE_BASE_WEIGHT * (source - reference)
    output = anchor[["row_id", "season", TARGET]].copy()
    output["prediction"] = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    return output


def ridge_artifact(base, cache_path, features, alpha):
    cache = pd.read_parquet(cache_path, columns=["row_id", "season", *features])
    frame = base.merge(cache, on=["row_id", "season"], validate="one_to_one")
    residual = frame[TARGET] - frame["prediction"]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    model.fit(frame[features], residual)
    imputer, scaler, ridge = model
    manual = (
        (imputer.transform(frame[features]) - scaler.mean_) / scaler.scale_
    ) @ ridge.coef_
    library = model.predict(frame[features])
    if not np.allclose(manual, library, atol=1e-8, rtol=1e-8):
        raise RuntimeError("Manual Ridge export mismatch")
    return {
        "feature_order": features,
        "imputer_statistics": imputer.statistics_.astype(float).tolist(),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "ridge_coef": ridge.coef_.astype(float).tolist(),
        "ridge_alpha": float(alpha),
        "correction_clip": [-0.05, 0.05],
        "train_season": 2024,
        "rows": int(len(frame)),
        "base_model": "anchor_plus_0.90_times_large_minus_anchor",
        "train_residual_mean": float(residual.mean()),
        "train_correction_mean": float(library.mean()),
        "train_correction_std": float(library.std()),
        "manual_max_abs_diff": float(np.max(np.abs(manual - library))),
    }


def build_ridge_artifacts():
    base = load_large_oof_2024()
    records = {}
    success_name = "large_success_ridge.json"
    success = ridge_artifact(
        base,
        WORK / "oof" / f"matchup_features_{SUCCESS_CONFIG}.parquet",
        SUCCESS_FEATURES,
        10.0,
    )
    (MODEL_BANK / success_name).write_text(
        json.dumps(success, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    records["success"] = success_name
    for seed in REVERSE_SEEDS:
        filename = f"large_reverse_s{seed}_ridge.json"
        artifact = ridge_artifact(
            base,
            WORK / "oof" / "reverse_batter_seed" / f"seed_{seed}.parquet",
            REVERSE_FEATURES,
            1000.0,
        )
        (MODEL_BANK / filename).write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records[str(seed)] = filename
    return records


def package(models, model_paths, features, mappings, ridge_files):
    with ZipFile(BASE_ZIP) as archive:
        metadata = json.loads(archive.read("model/metadata.json"))
        requirements = archive.read("requirements.txt").decode("utf-8")
    metadata = deepcopy(metadata)
    metadata["version"] = 7
    metadata["track"] = "large_xgb_multi_expert_cluster_probe"
    metadata["models"] = metadata["models"][:2] + models
    metadata["feature_sets"]["insight_adjusted"] = features
    metadata["category_mappings"]["insight_adjusted"] = mappings
    metadata.pop("outer_blend", None)
    large_success_spec = deepcopy(metadata["matchup_correction"])
    large_success_spec["ridge_file"] = ridge_files["success"]
    large_success_spec["correction_scale"] = 0.225
    large_reverse_specs = deepcopy(metadata["reverse_matchup_corrections"])
    for spec in large_reverse_specs:
        spec["ridge_file"] = ridge_files[str(spec["seed"])]
    metadata["multi_insight_blend"] = {
        "insight_anchor_model": "xgb_insight_anchor_t93",
        "insight_large_model": "xgb_insight_large_m24d",
        "large_base_weight": LARGE_BASE_WEIGHT,
        "current_success_correction": deepcopy(metadata["matchup_correction"]),
        "current_reverse_corrections": deepcopy(metadata["reverse_matchup_corrections"]),
        "current_success_scale": 0.25,
        "current_reverse_scale": 0.55,
        "large_success_correction": large_success_spec,
        "large_reverse_corrections": large_reverse_specs,
        "large_success_scale": 0.20,
        "large_reverse_scale": 0.575,
        "base_weight": FINAL_BASE_WEIGHT,
        "current_weight": FINAL_CURRENT_WEIGHT,
        "large_weight": FINAL_LARGE_WEIGHT,
    }
    metadata["large_xgb_validation"] = {
        "selection": "strict rolling 2022->2023 and 2023->2024",
        "feature_version": "INSIGHT_SUCCESS_ADJUSTED",
        "trackman_cutoff": "strictly before validation season",
        "trackman_min_pitcher_season_pitches": 500,
        "anchor_max_leaves": 18,
        "large_max_leaves": 24,
        "large_subsample": 0.90,
        "large_colsample_bytree": 0.70,
        "large_base_weight_fixed_across_folds": LARGE_BASE_WEIGHT,
        "large_success_scale": 0.20,
        "large_reverse_scale": 0.575,
        "reverse_seeds": REVERSE_SEEDS,
        "f23_correction_delta_brier": -1.4530092178904204e-05,
        "f24_correction_delta_brier": -4.443495203589887e-05,
        "single_corrected_bss_2024": 803.3732614326161,
        "threeway_weights": {
            "base": FINAL_BASE_WEIGHT,
            "current": FINAL_CURRENT_WEIGHT,
            "large": FINAL_LARGE_WEIGHT,
        },
        "outer_blend_bss_2024": 813.4317212822095,
        "previous_submit013_bss_2024": 812.704034,
        "risk_note": "F23 and F24 optimal outer weights differ materially; comparison probe, not automatic replacement.",
    }

    artifacts = {
        path
        for path in BASE_MODEL_DIR.iterdir()
        if path.is_file()
        and path.name not in {"metadata.json", "insight_adjusted_t93.ubj"}
    }
    artifacts.update(model_paths)
    artifacts.add(MODEL_BANK / ridge_files["success"])
    artifacts.update(MODEL_BANK / ridge_files[str(seed)] for seed in REVERSE_SEEDS)
    artifacts.update(
        {
            FINAL_DIR / "pitcher_lookup_2025.csv",
            FINAL_DIR / "batter_lookup_2025.csv",
            FINAL_DIR / "pair_table_2025.csv",
            FINAL_DIR / "ridge_correction.json",
        }
    )
    for seed in REVERSE_SEEDS:
        artifacts.update(
            {
                FINAL_DIR / f"reverse_batter_s{seed}_lookup_2025.csv",
                FINAL_DIR / f"reverse_batter_s{seed}_pair_2025.csv",
                FINAL_DIR / f"reverse_batter_s{seed}_ridge.json",
            }
        )
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package artifacts: {missing}")

    candidate = MODEL_DIR / f"candidate_large_xgb_{time.time_ns()}.zip"
    write_zip(
        candidate,
        metadata,
        artifacts,
        requirements,
        script_path=INFERENCE_SCRIPT,
    )
    inference_sec = smoke_test(candidate)
    candidate.replace(DESTINATION)
    return metadata, artifacts, inference_sec


def append_log(record):
    log_path = OUTPUT_DIR / "SUBMISSION_LOG.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.is_file() else "# 제출 기록\n"
    marker = "## 대형 XGBoost 다중 전문가 014"
    if marker in existing:
        raise RuntimeError("Submission log section already exists")
    section = f"""

{marker}

생성 시각: {datetime.now().astimezone().isoformat()}

| 파일 | 모델 확대 | 분포 정렬 | 성공 보정 | reverse 보정 | 2024 내부 BSS | 추론 |
|---|---|---|---:|---:|---:|---:|
| `{record['filename']}` | 18-leaf anchor + 24-leaf diverse expert | fixed large 0.90 | 0.200 | 0.575 | **813.432** | {record['inference_sec']:.1f}초 |

- 검증은 2022→2023, 2023→2024의 strict rolling 방식이다.
- TrackMan은 각 검증 시즌 직전까지만 사용했고 투수-시즌 500구 이상만 포함했다.
- large expert는 모든 fold와 2025에 동일한 anchor 0.10 + large 0.90 조합이며 test 전체 통계를 사용하지 않는다.
- 기존 013 corrected insight를 보존하고 large corrected insight를 별도 expert로 추가했다.
- large correction Ridge는 새 large expert의 2024 OOF 잔차 기준으로 다시 학습했다.
- 최종 두 XGBoost는 2019~2024 전체 학습 후 2025를 추론한다.
- ZIP 구조, CRC, 모델 존재, 245,789행 로컬 추론, 확률 범위를 검증했다.

SHA256: `{record['sha256']}`
"""
    log_path.write_text(existing.rstrip() + section + "\n", encoding="utf-8")


def main():
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite: {DESTINATION}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models, model_paths, features, mappings = train_models()
    ridge_files = build_ridge_artifacts()
    metadata, artifacts, inference_sec = package(
        models, model_paths, features, mappings, ridge_files
    )
    record = {
        "filename": DESTINATION.name,
        "filename_length": len(DESTINATION.name),
        "size_bytes": DESTINATION.stat().st_size,
        "sha256": digest(DESTINATION),
        "model_count": len(metadata["models"]),
        "artifact_count": len(artifacts),
        "inference_sec": inference_sec,
        "validation_bss_2024": 813.4317212822095,
        "previous_submit013_bss_2024": 812.704034,
    }
    append_log(record)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "base_submission": str(BASE_ZIP.relative_to(ROOT)),
        "record": record,
        "large_xgb_validation": metadata["large_xgb_validation"],
        "models": models,
        "ridge_files": ridge_files,
    }
    manifest_path = WORK / "final" / "large_xgb_submission_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
