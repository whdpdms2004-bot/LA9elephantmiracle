from __future__ import annotations

import gc
import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import optuna
import pandas as pd
import torch
from xgboost import XGBClassifier

from benchmark_insight_features import (
    WORK_DIR,
    add_calibration_features,
    build_past_only_lookups,
    logit,
    weighted_prior,
)
from build_final_two_submissions import (
    BANK_DIR as BASE_BANK_DIR,
    encode_xgboost_full,
    smoke_test,
    write_zip,
)
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import ROOT, SEED, TARGET, recency_weights


OUTPUT_DIR = ROOT / "submit" / "2026-08-12"
MODEL_BANK = WORK_DIR / "insight_final_model_bank"
INFERENCE_SCRIPT = WORK_DIR / "final_ensemble_script.py"
BASE_ZIPS = {
    "performance": ROOT / "submit" / "2026-08-06" / "submit_005.zip",
    "robust": ROOT / "submit" / "2026-08-06" / "submit_006.zip",
}
SUBMISSIONS = {
    "performance": {
        "number": 7,
        "version": "insight_adjusted",
        "mode": "adjusted",
        "feature_version": "INSIGHT_SUCCESS_ADJUSTED",
        "best_iteration": 2641,
        "outer_weight": 0.5284093304636978,
        "validation_bss": 801.1471125329517,
        "source_bss": 784.5568275342663,
    },
    "robust": {
        "number": 8,
        "version": "insight_success",
        "mode": "success_full",
        "feature_version": "INSIGHT_PRIOR_SUCCESS",
        "best_iteration": 2587,
        "outer_weight": 0.6089911948398207,
        "validation_bss": 799.6296600295549,
        "source_bss": 783.5233530519071,
    },
}


def read_zip_metadata(path: Path):
    with ZipFile(path) as archive:
        metadata = json.loads(archive.read("model/metadata.json"))
        requirements = archive.read("requirements.txt").decode("utf-8")
    return metadata, requirements


def load_trial93():
    study = optuna.load_study(
        study_name="xgboost_v2r200_tm500_local_2024",
        storage=f"sqlite:///{(WORK_DIR / 'xgboost_v2r200_tm500_local_2024.db').as_posix()}",
    )
    return next(trial for trial in study.trials if trial.number == 93)


def prior_constants_2025(frame):
    actual = frame.groupby("season")[TARGET].mean().to_dict()
    output = {}
    for prefix, rate_column, fixed_prior in [
        ("pitcher_success", "asof_pitcher_success_rate", 0.50),
        ("batter_success", "asof_batter_success_rate", 0.50),
    ]:
        mean_asof = frame.groupby("season")[rate_column].mean().to_dict()
        prior_last = float(actual[2024])
        gap = float(logit([prior_last])[0] - logit([float(mean_asof[2024])])[0])
        output[prefix] = {
            "source_season": 2024,
            "prior_last": prior_last,
            "prior_ewm1": weighted_prior(actual, 2025, 1.0, fixed_prior),
            "prior_ewm2": weighted_prior(actual, 2025, 2.0, fixed_prior),
            "gap_logit_last": float(np.clip(gap, -0.50, 0.50)),
        }
    return output


def feature_variants(frame, base_features, prior_columns):
    adjusted = [
        column
        for column in prior_columns
        if column.startswith(("pitcher_success_", "batter_success_"))
        and "_adjusted_smoothed_" in column
    ]
    success = [
        column
        for column in prior_columns
        if column.startswith(("pitcher_success_", "batter_success_"))
    ]
    variants = {
        "insight_adjusted": list(dict.fromkeys(base_features + adjusted)),
        "insight_success": list(dict.fromkeys(base_features + success)),
    }
    if len(adjusted) != 2 or len(success) != 12:
        raise RuntimeError(
            f"Unexpected insight columns: adjusted={len(adjusted)}, success={len(success)}"
        )
    for features in variants.values():
        missing = set(features).difference(frame.columns)
        if missing:
            raise RuntimeError(f"Missing training features: {sorted(missing)}")
    return variants


def train_insight_model(frame, features, spec, trial):
    version = spec["version"]
    matrix, mappings = encode_xgboost_full(frame, features)
    params = dict(trial.params)
    half_life = float(params.pop("half_life"))
    params["n_estimators"] = int(spec["best_iteration"]) + 1
    target = frame[TARGET].to_numpy("int8")
    weights = recency_weights(frame["season"], 2025, half_life)
    model = XGBClassifier(
        **params,
        grow_policy="lossguide",
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED + 2025,
        n_jobs=6,
    )
    started = time.time()
    model.fit(matrix, target, sample_weight=weights, verbose=False)
    MODEL_BANK.mkdir(parents=True, exist_ok=True)
    filename = f"{version}_t93.ubj"
    path = MODEL_BANK / filename
    model.save_model(str(path))
    elapsed = time.time() - started
    del model, matrix, target, weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "model_name": f"xgb_{version}_t93",
        "family": "xgboost",
        "feature_version": version,
        "trial": 93,
        "study": "xgboost_insight_success_local_2024",
        "filenames": [filename],
        "seeds": [SEED + 2025],
        "iterations": [int(spec["best_iteration"]) + 1],
        "half_life": half_life,
        "weight": 1.0,
        "train_elapsed_sec": elapsed,
    }, mappings, path


def package_one(track, spec, insight_model, mappings, model_path, features, constants):
    base_metadata, requirements = read_zip_metadata(BASE_ZIPS[track])
    metadata = deepcopy(base_metadata)
    metadata["version"] = 4
    metadata["track"] = f"{track}_insight"
    metadata["models"] = deepcopy(base_metadata["models"]) + [insight_model]
    metadata["feature_sets"][spec["version"]] = features
    metadata["category_mappings"][spec["version"]] = mappings
    metadata["insight_feature_mode"] = spec["mode"]
    metadata["insight_prior_constants"] = constants
    metadata["outer_blend"] = {
        "space": "probability",
        "insight_model": insight_model["model_name"],
        "insight_weight": spec["outer_weight"],
        "base_weight": 1.0 - spec["outer_weight"],
    }
    metadata["insight_validation"] = {
        "fold": 2024,
        "feature_version": spec["feature_version"],
        "single_bss": spec["source_bss"],
        "outer_blend_bss": spec["validation_bss"],
        "trackman_cutoff": "strictly before validation season",
        "min_trackman_season_pitches": 500,
    }
    artifact_paths = {model_path, BASE_BANK_DIR / metadata["trackman_lookup_file"]}
    for item in base_metadata["models"]:
        artifact_paths.update(BASE_BANK_DIR / filename for filename in item["filenames"])
    missing = [str(path) for path in artifact_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package artifacts: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"submit_{spec['number']:03d}.zip"
    if len(filename) >= 30:
        raise RuntimeError(f"Filename too long: {filename}")
    destination = OUTPUT_DIR / filename
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    candidate = WORK_DIR / f"candidate_insight_{track}_{time.time_ns()}.zip"
    write_zip(
        candidate,
        metadata,
        artifact_paths,
        requirements,
        script_path=INFERENCE_SCRIPT,
    )
    inference_sec = smoke_test(candidate)
    candidate.replace(destination)
    return {
        "track": track,
        "number": spec["number"],
        "path": str(destination.relative_to(ROOT)),
        "filename_length": len(filename),
        "feature_version": spec["feature_version"],
        "feature_count": len(features),
        "single_bss_2024": spec["source_bss"],
        "blend_bss_2024": spec["validation_bss"],
        "outer_insight_weight": spec["outer_weight"],
        "model_count": len(metadata["models"]),
        "artifact_count": sum(len(item["filenames"]) for item in metadata["models"]),
        "inference_sec": inference_sec,
        "size_bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest().upper(),
    }


def write_log(records):
    lines = [
        "# 제출 기록",
        "",
        f"생성 시각: {datetime.now().astimezone().isoformat()}",
        "",
        "| 파일 | 목적 | 핵심 피처 | 단일 BSS(2024) | 혼합 BSS(2024) | 새 모델 가중치 | 추론 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| `{Path(row['path']).name}` | {row['track']} | {row['feature_version']} "
            f"({row['feature_count']}열) | {row['single_bss_2024']:.3f} | "
            f"{row['blend_bss_2024']:.3f} | {row['outer_insight_weight']:.4f} | "
            f"{row['inference_sec']:.1f}초 |"
        )
    lines.extend(
        [
            "",
            "- 학습: 2019~2024 전체 train, 2025를 기준으로 recency weight 적용.",
            "- 검증: 2024 holdout에서는 train 2023 이하, TrackMan 2023 이하만 사용.",
            "- TrackMan 투수-시즌 조건: 500구 이상.",
            "- 경로: `script.py` 위치 기준 `model/`, `data/`, `output/` 상대경로.",
            "- 두 ZIP 모두 245,789행 로컬 추론 및 CRC 검사를 통과한 뒤 게시.",
        ]
    )
    (OUTPUT_DIR / "SUBMISSION_LOG.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    for spec in SUBMISSIONS.values():
        path = OUTPUT_DIR / f"submit_{spec['number']:03d}.zip"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    frame, base_features = load_enhanced_frame()
    labels = pd.read_parquet(WORK_DIR / "failure_component_labels.parquet")
    lookups, audit = build_past_only_lookups(frame, labels)
    if not all(
        item["source_season"] is None
        or item["source_season"] < item["target_season"]
        for item in audit
    ):
        raise RuntimeError("Past-only audit failed")
    constants = prior_constants_2025(frame)
    frame, _, prior_columns = add_calibration_features(frame, lookups)
    variants = feature_variants(frame, base_features, prior_columns)
    trial = load_trial93()
    trained = {}
    for track, spec in SUBMISSIONS.items():
        model, mappings, path = train_insight_model(
            frame, variants[spec["version"]], spec, trial
        )
        trained[track] = (model, mappings, path)

    records = []
    for track, spec in SUBMISSIONS.items():
        model, mappings, path = trained[track]
        records.append(
            package_one(
                track,
                spec,
                model,
                mappings,
                path,
                variants[spec["version"]],
                constants,
            )
        )
    write_log(records)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "records": records,
        "prior_constants_2025": constants,
    }
    manifest_path = WORK_DIR / "insight_submissions_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
