from __future__ import annotations

import gc
import json
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

from run_optuna_family import (
    CATEGORICAL_COLUMNS,
    ROOT,
    SEED,
    TARGET,
    load_frame,
    prepare_catboost_frame,
    recency_weights,
)


WORK_DIR = ROOT / "experiment" / "model_optimization"
INFERENCE_SCRIPT = WORK_DIR / "optimized_script.py"
FIXED_TIMESTAMP = (2026, 8, 6, 0, 0, 0)
PROBES = [
    {
        "slug": "01_catboost_robust",
        "family": "catboost",
        "study": "catboost_v1_full_2023_2024",
        "trial": 71,
        "reason": "two-fold robust objective winner",
    },
    {
        "slug": "02_xgboost_recent",
        "family": "xgboost",
        "study": "xgboost_v1_full_2023_2024",
        "trial": 24,
        "reason": "best 2024 BSS in XGBoost family",
    },
]


def load_trial(study_name, trial_number):
    database = WORK_DIR / f"{study_name}.db"
    study = optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{database.as_posix()}"
    )
    trial = next(item for item in study.trials if item.number == trial_number)
    if trial.state != optuna.trial.TrialState.COMPLETE:
        raise RuntimeError(f"Trial is not complete: {study_name}/{trial_number}")
    return trial


def encode_xgboost_full(frame, features):
    output = frame[features].copy()
    mappings = {}
    for column in CATEGORICAL_COLUMNS:
        values = output[column].fillna("__MISSING__").astype(str)
        mapping = {str(value): int(index) for index, value in enumerate(pd.unique(values))}
        mappings[column] = mapping
        output[column] = values.map(mapping).astype("int32")
    for column in features:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float32")
    return output, mappings


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


def write_common_files(destination, metadata, family):
    model_dir = destination / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(INFERENCE_SCRIPT, destination / "script.py")
    requirement = (
        "catboost==1.2.8\n" if family == "catboost" else "xgboost==3.1.1\n"
    )
    (destination / "requirements.txt").write_text(requirement, encoding="utf-8")


def local_smoke_test(source):
    with tempfile.TemporaryDirectory(prefix="lg_probe_") as temporary:
        stage = Path(temporary)
        shutil.copytree(source / "model", stage / "model")
        shutil.copy2(source / "script.py", stage / "script.py")
        (stage / "data").mkdir()
        shutil.copy2(ROOT / "data" / "test.csv", stage / "data" / "test.csv")
        completed = subprocess.run(
            [sys.executable, str(stage / "script.py")],
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = pd.read_csv(stage / "output" / "submission.csv")
        if list(output.columns) != ["row_id", TARGET] or len(output) != 5:
            raise RuntimeError(f"Invalid smoke output: {output.shape}, {output.columns.tolist()}")
        if not np.isfinite(output[TARGET]).all() or not output[TARGET].between(0, 1).all():
            raise RuntimeError("Smoke predictions are invalid")
        return {
            "rows": len(output),
            "pred_min": float(output[TARGET].min()),
            "pred_max": float(output[TARGET].max()),
            "stdout": completed.stdout.strip(),
        }


def build_zip(source, output):
    model_files = sorted(path for path in (source / "model").iterdir() if path.is_file())
    with ZipFile(output, "w", allowZip64=True) as archive:
        archive.writestr(directory_info("model/"), b"")
        for path in model_files:
            archive.writestr(file_info(f"model/{path.name}"), path.read_bytes())
        archive.writestr(file_info("script.py"), (source / "script.py").read_bytes())
        archive.writestr(
            file_info("requirements.txt"), (source / "requirements.txt").read_bytes()
        )
    with ZipFile(output) as archive:
        names = archive.namelist()
        if names[0] != "model/" or names[-2:] != ["script.py", "requirements.txt"]:
            raise RuntimeError(f"Unexpected ZIP layout: {names}")
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC check failed")
        for name in names:
            if name != "model/" and name.count("/") > 1:
                raise RuntimeError(f"Unexpected nested path: {name}")


def main():
    frame, features = load_frame(0)
    target = frame[TARGET].to_numpy("int8")
    seasons = frame["season"].to_numpy("int16")
    cat_frame = prepare_catboost_frame(frame, features)
    xgb_frame, xgb_mappings = encode_xgboost_full(frame, features)
    manifest = []

    for probe in PROBES:
        trial = load_trial(probe["study"], probe["trial"])
        params = dict(trial.params)
        half_life = float(params.pop("half_life"))
        iterations = int(trial.user_attrs["best_iteration_2024"]) + 1
        sample_weight = recency_weights(seasons, 2025, half_life)
        source = WORK_DIR / f"submit_{probe['slug']}"
        if source.exists():
            shutil.rmtree(source)
        model_dir = source / "model"
        model_dir.mkdir(parents=True)

        print(
            f"training {probe['slug']} trial={trial.number} iterations={iterations}",
            flush=True,
        )
        if probe["family"] == "catboost":
            params["iterations"] = iterations
            pool = Pool(
                cat_frame,
                label=target,
                cat_features=CATEGORICAL_COLUMNS,
                weight=sample_weight,
            )
            model = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                eval_metric="Logloss",
                task_type="GPU",
                devices="0",
                random_seed=SEED + trial.number,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(pool)
            filename = "model.cbm"
            model.save_model(str(model_dir / filename))
            mappings = {}
            del pool
        else:
            params["n_estimators"] = iterations
            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=SEED + trial.number,
                n_jobs=6,
            )
            model.fit(xgb_frame, target, sample_weight=sample_weight, verbose=False)
            filename = "model.ubj"
            model.save_model(str(model_dir / filename))
            mappings = xgb_mappings

        metadata = {
            "version": 2,
            "target": TARGET,
            "feature_columns": features,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "category_mappings": mappings,
            "models": [
                {
                    "model_name": probe["slug"],
                    "family": probe["family"],
                    "trial": trial.number,
                    "filename": filename,
                    "weight": 1.0,
                    "half_life": half_life,
                    "iterations": iterations,
                }
            ],
            "blend_space": "probability",
            "calibration": "none",
            "calibrator_params": {},
            "validation": {
                "reason": probe["reason"],
                "robust_objective": trial.value,
                "bss_2023": trial.user_attrs["fold_2023"]["bss"],
                "bss_2024": trial.user_attrs["fold_2024"]["bss"],
            },
        }
        write_common_files(source, metadata, probe["family"])
        smoke = local_smoke_test(source)
        zip_path = WORK_DIR / f"submit_{probe['slug']}_linux.zip"
        build_zip(source, zip_path)
        manifest.append(
            {
                "slug": probe["slug"],
                "zip": str(zip_path),
                "zip_size_mb": zip_path.stat().st_size / 2**20,
                "trial": trial.number,
                "family": probe["family"],
                "iterations": iterations,
                "validation": metadata["validation"],
                "smoke_test": smoke,
            }
        )
        print(f"created {zip_path} smoke={smoke}", flush=True)
        del model, sample_weight
        gc.collect()

    manifest_path = WORK_DIR / "two_probe_submissions_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
