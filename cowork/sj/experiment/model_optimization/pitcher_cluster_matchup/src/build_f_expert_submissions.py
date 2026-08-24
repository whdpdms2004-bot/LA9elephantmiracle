from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[4]
MODEL_OPT = ROOT / "experiment" / "model_optimization"
EXPERT_DIR = MODEL_OPT / "game_type_experts"
MODEL_BANK = EXPERT_DIR / "final_model_bank"
OUTPUT_DIR = ROOT / "submit" / "2026-08-12"
SCRIPT = Path(__file__).resolve().parent / "submission_script_matchup.py"
TARGET = "control_success"

sys.path.insert(0, str(MODEL_OPT))
from benchmark_game_type_experts import load_frame_and_features  # noqa: E402
from run_optuna_family import CATEGORICAL_COLUMNS, SEED  # noqa: E402


XGB_TRIAL = 31
CAT_TRIAL = 9
XGB_ITERATIONS = 97
CAT_ITERATIONS = 487
XGB_WEIGHT = 0.15
CAT_WEIGHT = 0.19

SPECS = [
    {
        "number": 17,
        "base": OUTPUT_DIR / "submit_015.zip",
        "reference_model": "xgb_insight_adjusted_t93",
        "base_val2024_bss": 830.5234166493314,
        "projected_val2024_bss": 833.329,
        "label": "submit015_plus_f_partial_pool",
    },
    {
        "number": 18,
        "base": OUTPUT_DIR / "submit_016.zip",
        "reference_model": "xgb_insight_anchor_t93",
        "base_val2024_bss": 831.3205849446725,
        "projected_val2024_bss": 834.127,
        "label": "submit016_plus_f_partial_pool",
    },
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def file_info(name: str, compress_type: int = ZIP_DEFLATED) -> ZipInfo:
    info = ZipInfo(name)
    info.date_time = (2026, 8, 12, 0, 0, 0)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    return info


def directory_info(name: str) -> ZipInfo:
    info = ZipInfo(name)
    info.date_time = (2026, 8, 12, 0, 0, 0)
    info.compress_type = ZIP_STORED
    info.external_attr = (0o755 << 16) | 0x10
    return info


def study_trial(study_name: str, trial_number: int):
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{(EXPERT_DIR / f'{study_name}.db').as_posix()}",
    )
    return next(trial for trial in study.trials if trial.number == trial_number)


def encode_xgboost(frame: pd.DataFrame, features: list[str]):
    output = frame[features].copy()
    mappings = {}
    for column in CATEGORICAL_COLUMNS:
        if column not in output:
            continue
        values = output[column].fillna("__MISSING__").astype(str)
        mapping = {
            value: int(index) for index, value in enumerate(pd.unique(values))
        }
        mappings[column] = mapping
        output[column] = values.map(mapping).astype("int32")
    for column in features:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(
            "float32"
        )
    return output, mappings


def train_final_experts():
    frame, _, local_features = load_frame_and_features()
    train_mask = (
        frame["season"].isin([2023, 2024])
        & frame["game_type"].astype(str).eq("F")
    )
    local = frame.loc[train_mask].reset_index(drop=True)
    target = local[TARGET].to_numpy("int8")
    if len(local) < 50_000 or set(local["season"].unique()) != {2023, 2024}:
        raise RuntimeError("Unexpected final F training population")

    xgb_trial = study_trial("xgb_game_type_f_postbreak", XGB_TRIAL)
    xgb_params = dict(xgb_trial.params)
    xgb_params["n_estimators"] = XGB_ITERATIONS
    xgb_matrix, mappings = encode_xgboost(local, local_features)
    xgb_model = XGBClassifier(
        **xgb_params,
        grow_policy="lossguide",
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=SEED + XGB_TRIAL,
        n_jobs=6,
    )
    MODEL_BANK.mkdir(parents=True, exist_ok=True)
    xgb_path = MODEL_BANK / "f_xgb_t31.ubj"
    cat_path = MODEL_BANK / "f_cat_t9.cbm"
    for path in [xgb_path, cat_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite final expert: {path}")

    started = time.time()
    xgb_model.fit(xgb_matrix, target, verbose=False)
    xgb_model.save_model(str(xgb_path))
    xgb_elapsed = time.time() - started
    del xgb_model, xgb_matrix
    gc.collect()

    categorical = [
        column for column in CATEGORICAL_COLUMNS if column in local_features
    ]
    cat_frame = local[local_features].copy()
    for column in categorical:
        cat_frame[column] = cat_frame[column].fillna("__MISSING__").astype(str)
    cat_trial = study_trial("cat_game_type_f_postbreak", CAT_TRIAL)
    cat_params = dict(cat_trial.params)
    cat_params["iterations"] = CAT_ITERATIONS
    cat_model = CatBoostClassifier(
        **cat_params,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="GPU",
        devices="0",
        bootstrap_type="Bayesian",
        random_seed=SEED + CAT_TRIAL,
        verbose=False,
        allow_writing_files=False,
    )
    started = time.time()
    cat_model.fit(Pool(cat_frame, label=target, cat_features=categorical))
    cat_model.save_model(str(cat_path))
    cat_elapsed = time.time() - started

    training = {
        "rows": int(len(local)),
        "seasons": [2023, 2024],
        "game_type": "F",
        "target_mean": float(target.mean()),
        "feature_count": len(local_features),
        "xgb_trial": XGB_TRIAL,
        "xgb_iterations": XGB_ITERATIONS,
        "xgb_seed": SEED + XGB_TRIAL,
        "xgb_train_sec": xgb_elapsed,
        "cat_trial": CAT_TRIAL,
        "cat_iterations": CAT_ITERATIONS,
        "cat_seed": SEED + CAT_TRIAL,
        "cat_train_sec": cat_elapsed,
        "trackman_rule": "row-wise strictly past; pitcher-season >=500",
    }
    del cat_model, cat_frame, local, frame, target
    gc.collect()
    return local_features, mappings, xgb_path, cat_path, training


def package(spec, features, mappings, xgb_path, cat_path, training):
    source = Path(spec["base"])
    destination = OUTPUT_DIR / f"submit_{spec['number']:03d}.zip"
    candidate = destination.with_suffix(".candidate.zip")
    if destination.exists() or candidate.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    if len(destination.name) >= 30:
        raise RuntimeError(f"Submission filename is too long: {destination.name}")

    excluded = {
        "model/metadata.json",
        "script.py",
        f"model/{xgb_path.name}",
        f"model/{cat_path.name}",
    }
    with ZipFile(source) as archive:
        metadata = json.loads(archive.read("model/metadata.json"))
        members = {
            item.filename: (archive.read(item.filename), item.compress_type)
            for item in archive.infolist()
            if not item.is_dir() and item.filename not in excluded
        }
    metadata = deepcopy(metadata)
    metadata["version"] = int(metadata.get("version", 0)) + 1
    metadata["track"] = spec["label"]
    metadata["feature_sets"]["f_insight_adjusted"] = features
    metadata["category_mappings"]["f_insight_adjusted"] = mappings
    metadata["models"].extend(
        [
            {
                "model_name": "f_xgb_t31",
                "family": "xgboost",
                "feature_version": "f_insight_adjusted",
                "route_game_type": "F",
                "filenames": [xgb_path.name],
                "seeds": [SEED + XGB_TRIAL],
                "iterations": [XGB_ITERATIONS],
                "weight": 0.0,
            },
            {
                "model_name": "f_cat_t9",
                "family": "catboost",
                "feature_version": "f_insight_adjusted",
                "route_game_type": "F",
                "filenames": [cat_path.name],
                "seeds": [SEED + CAT_TRIAL],
                "iterations": [CAT_ITERATIONS],
                "weight": 0.0,
            },
        ]
    )
    metadata["game_type_expert"] = {
        "game_type": "F",
        "reference_model": spec["reference_model"],
        "experts": [
            {"model_name": "f_xgb_t31", "weight": XGB_WEIGHT},
            {"model_name": "f_cat_t9", "weight": CAT_WEIGHT},
        ],
        "partial_pooling_reference_weight": 1.0 - XGB_WEIGHT - CAT_WEIGHT,
        "selection": "joint 2023 expanding-month and Val2024; weights shrunk 50%",
        "sequential_2023_bss": 128.24206504048473,
        "val2024_f_bss": 507.6399908947926,
        "base_val2024_bss": spec["base_val2024_bss"],
        "projected_val2024_bss": spec["projected_val2024_bss"],
        "training": training,
    }

    with ZipFile(
        candidate, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        archive.writestr(directory_info("model/"), b"")
        archive.writestr(
            file_info("model/metadata.json"),
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        for name, (content, compress_type) in sorted(members.items()):
            archive.writestr(file_info(name, compress_type), content)
        archive.writestr(file_info(f"model/{xgb_path.name}"), xgb_path.read_bytes())
        archive.writestr(file_info(f"model/{cat_path.name}"), cat_path.read_bytes())
        archive.writestr(file_info("script.py"), SCRIPT.read_bytes())

    with ZipFile(candidate) as archive:
        names = archive.namelist()
        if names[0] != "model/" or archive.testzip() is not None:
            raise RuntimeError(f"Invalid ZIP: {candidate}")
        required = {
            "model/metadata.json",
            f"model/{xgb_path.name}",
            f"model/{cat_path.name}",
            "script.py",
            "requirements.txt",
        }
        if not required.issubset(names):
            raise RuntimeError(f"Missing ZIP entries: {required.difference(names)}")
    candidate.replace(destination)
    return destination


def mixed_smoke_test(zip_path: Path, rows: int = 245_789):
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="lgaimers_f_expert_") as directory:
        stage = Path(directory)
        with ZipFile(zip_path) as archive:
            archive.extractall(stage)
        data_dir = stage / "data"
        data_dir.mkdir()
        sample = pd.read_csv(ROOT / "data" / "test.csv")
        repeats = int(np.ceil(rows / len(sample)))
        benchmark = pd.concat([sample] * repeats, ignore_index=True).iloc[:rows].copy()
        benchmark["row_id"] = [f"BENCH_{index:06d}" for index in range(rows)]
        benchmark.loc[np.arange(rows) % 2 == 1, "game_type"] = "F"
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
            raise RuntimeError(f"Mixed smoke failed: {result.stderr[-5000:]}")
        submission = pd.read_csv(stage / "output" / "submission.csv")
        probability = submission[TARGET].to_numpy(float)
        if len(submission) != rows or not np.isfinite(probability).all():
            raise RuntimeError("Invalid mixed smoke-test output")
        if not ((probability > 0.0) & (probability < 1.0)).all():
            raise RuntimeError("Mixed smoke-test probabilities out of range")
        diagnostics = {
            "rows": rows,
            "r_rows": int((benchmark["game_type"] == "R").sum()),
            "f_rows": int((benchmark["game_type"] == "F").sum()),
            "pred_min": float(probability.min()),
            "pred_max": float(probability.max()),
            "pred_mean": float(probability.mean()),
        }
    return time.time() - started, diagnostics


def append_log(records):
    log_path = OUTPUT_DIR / "SUBMISSION_LOG.md"
    existing = log_path.read_text(encoding="utf-8")
    marker = "## F 전용 부분 풀링 전문가 017·018"
    if marker in existing:
        raise RuntimeError("F expert log section already exists")
    lines = [
        "",
        marker,
        "",
        f"생성 시각: {datetime.now().astimezone().isoformat()}",
        "",
        "| 파일 | 기반 | F 분기 | 2023 순차 F BSS | Val2024 F BSS | Val2024 전체 BSS(추정) | 245,789행 혼합 추론 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| `{row['filename']}` | `{row['base']}` | XGB 0.15 + CAT 0.19 | "
            f"128.242 | 507.640 | **{row['projected_val2024_bss']:.3f}** | "
            f"{row['inference_sec']:.1f}초 |"
        )
    lines.extend(
        [
            "",
            "- R 행은 기존 모델과 R 컨텍스트 보정을 그대로 사용하고, F 행에서만 두 전문가를 호출한다.",
            "- F 전문가는 2023~2024 F 투구만 최종 학습했으며, 검증 시 TrackMan은 검증 시즌 이전 자료와 투수-시즌 500구 이상만 사용했다.",
            "- 전문가 잔차는 글로벌 앵커 대비 XGB 0.15, CatBoost 0.19만 반영한다. 나머지 0.66은 글로벌 예측을 보존하는 부분 풀링이다.",
            "- 전체 BSS는 F 잔차 개선분을 기존 015·016에 더한 추정치다. R/F 보정은 서로 다른 행에 적용된다.",
            "- ZIP 루트 구조, CRC, 모델 존재, F/R 혼합 245,789행 추론, 확률 유효 범위를 검증했다.",
        ]
    )
    log_path.write_text(
        existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    for spec in SPECS:
        destination = OUTPUT_DIR / f"submit_{spec['number']:03d}.zip"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite: {destination}")
    features, mappings, xgb_path, cat_path, training = train_final_experts()
    records = []
    for spec in SPECS:
        destination = package(
            spec, features, mappings, xgb_path, cat_path, training
        )
        inference_sec, diagnostics = mixed_smoke_test(destination)
        records.append(
            {
                "filename": destination.name,
                "filename_length": len(destination.name),
                "base": Path(spec["base"]).name,
                "size_bytes": destination.stat().st_size,
                "sha256": digest(destination),
                "base_val2024_bss": spec["base_val2024_bss"],
                "projected_val2024_bss": spec["projected_val2024_bss"],
                "inference_sec": inference_sec,
                "smoke": diagnostics,
            }
        )
    append_log(records)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "training": training,
        "weights": {"xgboost": XGB_WEIGHT, "catboost": CAT_WEIGHT},
        "validation": {
            "sequential_2023_f_bss": 128.24206504048473,
            "val2024_f_bss": 507.6399908947926,
            "selection_rule": "joint temporal validation, then 50% shrink",
        },
        "records": records,
    }
    manifest_path = EXPERT_DIR / "f_expert_submission_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
