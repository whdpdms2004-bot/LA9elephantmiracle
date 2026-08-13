from __future__ import annotations

import gc
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    load_frame,
    recency_weights,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
BANK_DIR = WORK_DIR / "final_two_model_bank"
SELECTION_PATH = WORK_DIR / "enhanced_ensemble_selection.json"
SCRIPT_PATH = WORK_DIR / "final_ensemble_script.py"
LOOKUP_PATH = WORK_DIR / "trackman500_lookup_2025.parquet"
OUTPUT_DIR = ROOT / "submit" / "2026-08-06"
TRACK_NUMBERS = {"performance": 5, "robust": 6}
FIXED_TIMESTAMP = (2026, 8, 6, 0, 0, 0)
SEED_OFFSETS = [0, 100_000, 200_000]


def safe_name(value):
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value)


def encode_xgboost_full(frame, features):
    encoded = frame[features].copy()
    mappings = {}
    for column in CATEGORICAL_COLUMNS:
        values = encoded[column].fillna("__MISSING__").astype(str)
        unique = pd.unique(values)
        mapping = {value: int(index) for index, value in enumerate(unique)}
        mappings[column] = mapping
        encoded[column] = values.map(mapping).astype("int32")
    for column in features:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce").astype("float32")
    return encoded, mappings


def prepare_catboost_full(frame, features):
    output = frame[features].copy()
    for column in CATEGORICAL_COLUMNS:
        output[column] = output[column].fillna("__MISSING__").astype(str)
    return output


def load_model_specs():
    v1_selection = json.loads(
        (WORK_DIR / "ensemble_selection.json").read_text(encoding="utf-8")
    )
    enhanced_selection = json.loads(
        (WORK_DIR / "enhanced_seed_oof_selection.json").read_text(encoding="utf-8")
    )
    v1 = {f"v1__{safe_name(item['model_name'])}": item for item in v1_selection["selected_models"]}
    enhanced = {
        f"enh__{safe_name(item['model_name'])}": item for item in enhanced_selection
    }
    return {**v1, **enhanced}


def recent_iterations(logical_name, spec):
    if logical_name.startswith("v1__"):
        value = spec["best_iterations"].get("2024", spec["best_iterations"].get(2024))
        return [int(value) + 1]
    model_name = logical_name.split("__", 1)[1]
    metadata_path = WORK_DIR / "enhanced_seed_oof_parts" / f"{safe_name(model_name)}_fold2024.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return [int(value) + 1 for value in metadata["best_iterations"]]


def train_logical_model(
    logical_name,
    spec,
    frame,
    features,
    encoded_xgb,
    cat_frame,
):
    family = spec["family"]
    params = dict(spec["params"])
    half_life = float(params.pop("half_life"))
    iterations = recent_iterations(logical_name, spec)
    seed_count = 3 if logical_name.startswith("enh__") else 1
    if len(iterations) != seed_count:
        raise RuntimeError(f"seed/iteration mismatch: {logical_name}")
    target = frame[TARGET].to_numpy("int8")
    weights = recency_weights(frame["season"].to_numpy("int16"), 2025, half_life)
    filenames = []
    seeds = []
    for seed_index in range(seed_count):
        seed_offset = SEED_OFFSETS[seed_index] if seed_count == 3 else 0
        random_seed = SEED + int(spec["trial"]) + 2025 + seed_offset
        stem = safe_name(f"{logical_name}_s{seed_index}")
        if family == "xgboost":
            local_params = dict(params)
            local_params["n_estimators"] = iterations[seed_index]
            model = XGBClassifier(
                **local_params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=random_seed,
                n_jobs=6,
            )
            model.fit(encoded_xgb, target, sample_weight=weights, verbose=False)
            filename = f"{stem}.ubj"
            model.save_model(str(BANK_DIR / filename))
        elif family == "lightgbm":
            local_params = dict(params)
            local_params["n_estimators"] = iterations[seed_index]
            model = LGBMClassifier(
                **local_params,
                objective="binary",
                metric="None",
                random_state=random_seed,
                n_jobs=6,
                verbosity=-1,
                force_col_wise=True,
            )
            model.fit(encoded_xgb, target, sample_weight=weights)
            filename = f"{stem}.txt"
            model.booster_.save_model(str(BANK_DIR / filename))
        elif family == "catboost":
            local_params = dict(params)
            local_params["iterations"] = iterations[seed_index]
            train_pool = Pool(
                cat_frame,
                label=target,
                cat_features=CATEGORICAL_COLUMNS,
                weight=weights,
            )
            model = CatBoostClassifier(
                **local_params,
                loss_function="Logloss",
                eval_metric="Logloss",
                task_type="GPU",
                devices="0",
                random_seed=random_seed,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(train_pool)
            filename = f"{stem}.cbm"
            model.save_model(str(BANK_DIR / filename))
            del train_pool
        else:
            raise ValueError(family)
        filenames.append(filename)
        seeds.append(random_seed)
        print(
            json.dumps(
                {
                    "model": logical_name,
                    "family": family,
                    "seed": random_seed,
                    "iterations": iterations[seed_index],
                    "file": filename,
                }
            ),
            flush=True,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del target, weights
    return {
        "model_name": logical_name,
        "family": family,
        "feature_version": "enhanced" if logical_name.startswith("enh__") else "v1",
        "trial": int(spec["trial"]),
        "study": spec["study"],
        "filenames": filenames,
        "seeds": seeds,
        "iterations": iterations,
        "half_life": half_life,
    }


def directory_info(name):
    info = ZipInfo(name.rstrip("/") + "/", FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = ZIP_STORED
    info.external_attr = (stat.S_IFDIR | 0o755) << 16
    return info


def file_info(name):
    info = ZipInfo(name, FIXED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def write_zip(path, metadata, artifact_paths, requirements, script_path=SCRIPT_PATH):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    metadata_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
    with ZipFile(path, "w", allowZip64=True) as archive:
        archive.writestr(directory_info("model/"), b"")
        archive.writestr(file_info("model/metadata.json"), metadata_bytes)
        for artifact in sorted(artifact_paths, key=lambda item: item.name):
            archive.writestr(file_info(f"model/{artifact.name}"), artifact.read_bytes())
        archive.writestr(file_info("script.py"), Path(script_path).read_bytes())
        archive.writestr(file_info("requirements.txt"), requirements.encode("utf-8"))
    with ZipFile(path) as archive:
        names = archive.namelist()
        if names[0] != "model/" or archive.testzip() is not None:
            raise RuntimeError(f"Invalid ZIP: {path}")
        if "script.py" not in names or "requirements.txt" not in names:
            raise RuntimeError(f"Missing root files: {names}")
        expected = {f"model/{item.name}" for item in artifact_paths}
        if not expected.issubset(names):
            raise RuntimeError(f"Missing model artifacts: {expected.difference(names)}")


def smoke_test(zip_path):
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="lgaimers_submit_") as directory:
        stage = Path(directory)
        with ZipFile(zip_path) as archive:
            archive.extractall(stage)
        data_dir = stage / "data"
        data_dir.mkdir()
        sample = pd.read_csv(ROOT / "data" / "test.csv")
        repeats = int(np.ceil(245_789 / len(sample)))
        benchmark = pd.concat([sample] * repeats, ignore_index=True).iloc[:245_789].copy()
        benchmark["row_id"] = [f"BENCH_{index:06d}" for index in range(len(benchmark))]
        benchmark.to_csv(data_dir / "test.csv", index=False)
        result = subprocess.run(
            [sys.executable, str(stage / "script.py")],
            cwd=stage,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"smoke failed for {zip_path.name}: {result.stderr[-4000:]}"
            )
        submission = pd.read_csv(stage / "output" / "submission.csv")
        if len(submission) != len(benchmark):
            raise RuntimeError("smoke output row count mismatch")
        probability = submission[TARGET].to_numpy(float)
        if not np.isfinite(probability).all() or not ((probability > 0) & (probability < 1)).all():
            raise RuntimeError("smoke output probability validation failed")
    return time.time() - started


def append_log(records, selection):
    log_path = OUTPUT_DIR / "SUBMISSION_LOG.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.is_file() else "# 제출 기록\n"
    marker = "## 최종 강화 앙상블 005·006"
    if marker in existing:
        return
    lines = [
        "",
        marker,
        "",
        f"생성 시각: {datetime.now().astimezone().isoformat()}",
        "",
        "| 회수 | 목적 | 시간 OOF 설계 | 2024 BSS | 모델 수 | 245,789행 로컬 추론 | SHA256 |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        best = selection["best"][record["track"]]
        transition = best["transition_parameters"]["2023_to_2024"]
        lines.append(
            f"| {record['number']:03d} | {record['track']} | "
            f"2023 OOF 적합→2024 검증, 2024 OOF로 배포 재적합 | "
            f"{transition['bss']:.4f} | {record['model_count']} | "
            f"{record['inference_sec']:.1f}초 | `{record['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "- 두 모델 모두 2019~2024 전체 학습 데이터를 재학습했다.",
            "- Trackman은 2024년까지의 과거 로그 중 시즌 500구 이상 투수-시즌만 사용했다.",
            "- `script.py`는 실행 파일 위치를 기준으로 `model/`, `data/`, `output/`을 해석한다.",
            "- ZIP 내부 최상위 구조와 모든 모델 파일 존재 여부, CRC, 245,789행 추론을 검증했다.",
        ]
    )
    log_path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    for track, number in TRACK_NUMBERS.items():
        destination = OUTPUT_DIR / f"submit_{number:03d}.zip"
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing {track} submission: {destination}"
            )
    all_specs = load_model_specs()
    active_by_track = {}
    required_names = set()
    for track in ["performance", "robust"]:
        deployment = selection["deployment"][track]
        pairs = [
            (name, float(weight))
            for name, weight in zip(deployment["selected_models"], deployment["weights"])
            if float(weight) > 1e-6
        ]
        total = sum(weight for _, weight in pairs)
        active_by_track[track] = [(name, weight / total) for name, weight in pairs]
        required_names.update(name for name, _ in pairs)
    missing = required_names.difference(all_specs)
    if missing:
        raise KeyError(f"No deployment spec for: {sorted(missing)}")

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trained = {}
    feature_sets = {}
    category_mappings = {}
    for version in ["v1", "enhanced"]:
        version_names = sorted(
            name
            for name in required_names
            if (name.startswith("enh__")) == (version == "enhanced")
        )
        if not version_names:
            continue
        if version == "v1":
            frame, features = load_frame(0)
        else:
            frame, features = load_enhanced_frame()
        feature_sets[version] = features
        families = {all_specs[name]["family"] for name in version_names}
        encoded_xgb = None
        cat_frame = None
        if "xgboost" in families or "lightgbm" in families:
            encoded_xgb, category_mappings[version] = encode_xgboost_full(frame, features)
        else:
            category_mappings[version] = {}
        if "catboost" in families:
            cat_frame = prepare_catboost_full(frame, features)
        for name in version_names:
            trained[name] = train_logical_model(
                name,
                all_specs[name],
                frame,
                features,
                encoded_xgb,
                cat_frame,
            )
        del frame, encoded_xgb, cat_frame
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lookup = pd.read_parquet(LOOKUP_PATH)
    lookup_file = "trackman500_lookup_2025.csv"
    lookup.to_csv(BANK_DIR / lookup_file, index=False)
    tm_columns = [column for column in lookup if column != "pitcher_id"]
    records = []
    for track in ["performance", "robust"]:
        number = TRACK_NUMBERS[track]
        filename = f"submit_{number:03d}.zip"
        if len(filename) >= 30:
            raise RuntimeError(f"filename too long: {filename}")
        deployment = selection["deployment"][track]
        models = []
        artifact_paths = {BANK_DIR / lookup_file}
        for name, weight in active_by_track[track]:
            item = {**trained[name], "weight": weight}
            models.append(item)
            artifact_paths.update(BANK_DIR / value for value in item["filenames"])
        metadata = {
            "version": 3,
            "track": track,
            "target": TARGET,
            "models": models,
            "feature_sets": feature_sets,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "category_mappings": category_mappings,
            "trackman_lookup_file": lookup_file,
            "trackman_columns": tm_columns,
            "trackman_rule": "2019-2024 only; pitcher-season >=500",
            "blend_space": deployment["space"],
            "calibration": deployment["calibration"],
            "calibrator_params": deployment["calibrator_params"],
            "validation": selection["best"][track],
        }
        requirements = "xgboost==3.1.1\n"
        if any(item["family"] == "catboost" for item in models):
            requirements += "catboost==1.2.8\n"
        if any(item["family"] == "lightgbm" for item in models):
            requirements += "lightgbm==4.6.0\n"
        zip_path = OUTPUT_DIR / filename
        if zip_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing submission: {zip_path}")
        candidate_path = WORK_DIR / f"candidate_{number:03d}_{time.time_ns()}.zip"
        write_zip(candidate_path, metadata, artifact_paths, requirements)
        inference_sec = smoke_test(candidate_path)
        records.append(
            {
                "track": track,
                "number": number,
                "candidate_path": str(candidate_path),
                "destination_path": str(zip_path),
                "filename_length": len(filename),
                "model_count": len(models),
                "artifact_count": sum(len(item["filenames"]) for item in models),
                "inference_sec": inference_sec,
            }
        )
    # Publish neither file until both full-size inference tests pass.
    for record in records:
        candidate_path = Path(record.pop("candidate_path"))
        zip_path = Path(record.pop("destination_path"))
        candidate_path.replace(zip_path)
        record["path"] = str(zip_path.relative_to(ROOT))
        record["size_bytes"] = zip_path.stat().st_size
        record["sha256"] = hashlib.sha256(zip_path.read_bytes()).hexdigest().upper()
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "selection": str(SELECTION_PATH.relative_to(ROOT)),
        "records": records,
    }
    (WORK_DIR / "final_two_submissions_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    append_log(records, selection)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
