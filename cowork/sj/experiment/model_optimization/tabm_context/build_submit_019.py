"""Train the frozen final F-TabM and package submit_019 from submit_017."""

from __future__ import annotations

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
import pandas as pd
import tabm
import torch

from run_tabm_temporal import (
    FoldPreprocessor,
    determine_base_features,
    seed_everything,
    training_loss,
)


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "submit" / "2026-08-12" / "submit_017.zip"
OUTPUT_DIR = ROOT / "submit" / "2026-08-13"
DESTINATION = OUTPUT_DIR / "submit_019.zip"
SCRIPT = (
    ROOT
    / "experiment/model_optimization/pitcher_cluster_matchup/src/submission_script_matchup.py"
)
MODEL_PATH = (
    ROOT
    / "experiment/model_optimization/tabm_context/final_model_bank/f_tabm_t0.pt"
)
REPORT = ROOT / "experiment/model_optimization/tabm_context/reports/submit_019_manifest.json"
TARGET = "control_success"
SEED = 20260813
TABM_WEIGHT = 0.30
REFERENCE_SCALE = 1.0 - TABM_WEIGHT
FINAL_EPOCHS = 6


def file_info(name: str, compress_type: int = ZIP_DEFLATED) -> ZipInfo:
    info = ZipInfo(name)
    info.date_time = (2026, 8, 13, 0, 0, 0)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    return info


def directory_info(name: str) -> ZipInfo:
    info = ZipInfo(name)
    info.date_time = (2026, 8, 13, 0, 0, 0)
    info.compress_type = ZIP_STORED
    info.external_attr = (0o755 << 16) | 0x10
    return info


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def train_final() -> tuple[list[str], dict]:
    if MODEL_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite final model: {MODEL_PATH}")
    seed_everything(SEED)
    frame = pd.read_csv(ROOT / "data/train.csv")
    mask = (
        frame["season"].isin([2023, 2024])
        & frame["game_type"].astype(str).eq("F")
    )
    train = frame.loc[mask].reset_index(drop=True)
    if len(train) != 55_696:
        raise RuntimeError(f"Unexpected final F rows: {len(train)}")
    base_features = determine_base_features(frame.columns.tolist())
    pre = FoldPreprocessor("t0", base_features).fit(train)
    x_num, x_cat = pre.transform(train)
    y = train[TARGET].to_numpy(dtype=np.float32)
    device = torch.device("cuda")
    x_num_t = torch.from_numpy(np.ascontiguousarray(x_num)).to(device)
    x_cat_t = torch.from_numpy(np.ascontiguousarray(x_cat)).to(device)
    y_t = torch.from_numpy(y).to(device)
    model_args = {
        "n_num_features": int(x_num.shape[1]),
        "cat_cardinalities": pre.cat_cardinalities,
        "d_out": 1,
        "n_blocks": 3,
        "d_block": 384,
        "dropout": 0.20,
        "k": 32,
        "arch_type": "tabm",
    }
    model = tabm.TabM.make(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    losses = []
    started = time.time()
    for epoch in range(1, FINAL_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(y_t), device=device)
        epoch_loss = []
        for start in range(0, len(y_t), 256):
            idx = permutation[start : start + 256]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(
                    x_num_t.index_select(0, idx), x_cat_t.index_select(0, idx)
                ).squeeze(-1)
                loss = training_loss(
                    logits, y_t.index_select(0, idx), "brier", 0.0
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss.append(float(loss.detach().item()))
        losses.append(float(np.mean(epoch_loss)))
        print(f"final F TabM epoch={epoch} loss={losses[-1]:.8f}", flush=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_args": model_args,
        "preprocessor": pre.to_dict(),
        "training": {
            "rows": int(len(train)),
            "seasons": [2023, 2024],
            "game_type": "F",
            "target_mean": float(y.mean()),
            "epochs": FINAL_EPOCHS,
            "seed": SEED,
            "loss": "brier",
            "lr": 0.0015,
            "batch_size": 256,
            "epoch_losses": losses,
            "elapsed_seconds": time.time() - started,
        },
    }
    torch.save(checkpoint, MODEL_PATH)
    return base_features, checkpoint["training"]


def package(base_features: list[str], training: dict) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate = DESTINATION.with_suffix(".candidate.zip")
    for path in [DESTINATION, candidate]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    if len(DESTINATION.name) >= 30:
        raise RuntimeError(f"Filename too long: {DESTINATION.name}")
    excluded = {
        "model/metadata.json",
        "script.py",
        "requirements.txt",
        f"model/{MODEL_PATH.name}",
    }
    with ZipFile(BASE) as archive:
        metadata = json.loads(archive.read("model/metadata.json"))
        requirements = archive.read("requirements.txt").decode("utf-8")
        members = {
            item.filename: (archive.read(item.filename), item.compress_type)
            for item in archive.infolist()
            if not item.is_dir() and item.filename not in excluded
        }
    metadata = deepcopy(metadata)
    metadata["version"] = int(metadata.get("version", 0)) + 1
    metadata["track"] = "submit017_plus_f_tabm30"
    metadata["feature_sets"]["tabm_t0"] = base_features
    metadata["models"].append(
        {
            "model_name": "f_tabm_t0",
            "family": "tabm",
            "feature_version": "tabm_t0",
            "route_game_type": "F",
            "filenames": [MODEL_PATH.name],
            "seeds": [SEED],
            "iterations": [FINAL_EPOCHS],
            "batch_size": 4096,
            "weight": 0.0,
        }
    )
    expert = metadata["game_type_expert"]
    for item in expert["experts"]:
        item["weight"] = float(item["weight"]) * REFERENCE_SCALE
    expert["experts"].append(
        {"model_name": "f_tabm_t0", "weight": TABM_WEIGHT}
    )
    expert["partial_pooling_reference_weight"] = (
        1.0 - sum(float(x["weight"]) for x in expert["experts"])
    )
    expert["tabm_extension"] = {
        "selection": "weight selected only on 2023 expanding-month OOF",
        "tabm_weight": TABM_WEIGHT,
        "val2024_used_for_weight_selection": False,
        "val2024_reference_f_bss": 507.6399908947926,
        "val2024_transferred_f_bss": 540.5434893668071,
        "val2024_delta_f_bss": 32.90349847201446,
        "final_training": training,
        "trackman_used_by_tabm": False,
    }
    requirements_lines = [x.strip() for x in requirements.splitlines() if x.strip()]
    for line in ["tabm==0.0.3", "rtdl-num-embeddings==0.0.12"]:
        if line not in requirements_lines:
            requirements_lines.append(line)
    requirements = "\n".join(requirements_lines) + "\n"

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
        archive.writestr(file_info(f"model/{MODEL_PATH.name}"), MODEL_PATH.read_bytes())
        archive.writestr(file_info("script.py"), SCRIPT.read_bytes())
        archive.writestr(file_info("requirements.txt"), requirements.encode("utf-8"))
    with ZipFile(candidate) as archive:
        names = archive.namelist()
        required = {
            "model/metadata.json",
            f"model/{MODEL_PATH.name}",
            "script.py",
            "requirements.txt",
        }
        if names[0] != "model/" or archive.testzip() is not None:
            raise RuntimeError("Invalid ZIP structure or CRC")
        if not required.issubset(names):
            raise RuntimeError(f"Missing entries: {required.difference(names)}")
        if any(name.startswith(("submit/", "open/")) for name in names):
            raise RuntimeError("Unexpected top-level wrapper")
    candidate.replace(DESTINATION)
    return {
        "filename": DESTINATION.name,
        "filename_length": len(DESTINATION.name),
        "size_bytes": DESTINATION.stat().st_size,
        "sha256": digest(DESTINATION),
        "requirements": requirements_lines,
    }


def run_one(zip_path: Path, benchmark: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="lgaimers_tabm_submit_") as directory:
        stage = Path(directory)
        with ZipFile(zip_path) as archive:
            archive.extractall(stage)
        (stage / "data").mkdir(exist_ok=True)
        benchmark.to_csv(stage / "data/test.csv", index=False)
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
                f"Inference failed ({result.returncode})\nSTDOUT:\n{result.stdout[-4000:]}"
                f"\nSTDERR:\n{result.stderr[-8000:]}"
            )
        output = pd.read_csv(stage / "output/submission.csv")
    return output, time.time() - started, result.stderr[-4000:]


def smoke(package_record: dict) -> dict:
    sample = pd.read_csv(ROOT / "data/test.csv")
    rows = 245_789
    repeats = int(np.ceil(rows / len(sample)))
    benchmark = pd.concat([sample] * repeats, ignore_index=True).iloc[:rows].copy()
    benchmark["row_id"] = [f"BENCH_{i:06d}" for i in range(rows)]
    benchmark.loc[np.arange(rows) % 2 == 0, "game_type"] = "R"
    benchmark.loc[np.arange(rows) % 2 == 1, "game_type"] = "F"
    new_output, new_seconds, stderr = run_one(DESTINATION, benchmark)
    base_output, base_seconds, _ = run_one(BASE, benchmark)
    if len(new_output) != rows or not np.isfinite(new_output[TARGET]).all():
        raise RuntimeError("Invalid new submission output")
    if not new_output["row_id"].equals(benchmark["row_id"]):
        raise RuntimeError("Output row order changed")
    probability = new_output[TARGET].to_numpy(dtype=np.float64)
    if not ((probability > 0.0) & (probability < 1.0)).all():
        raise RuntimeError("Probability out of bounds")
    is_r = benchmark["game_type"].eq("R").to_numpy()
    r_max_abs_diff = float(
        np.max(
            np.abs(
                new_output.loc[is_r, TARGET].to_numpy()
                - base_output.loc[is_r, TARGET].to_numpy()
            )
        )
    )
    if r_max_abs_diff > 1e-12:
        raise RuntimeError(f"R route changed unexpectedly: {r_max_abs_diff}")
    return {
        **package_record,
        "rows": rows,
        "r_rows": int(is_r.sum()),
        "f_rows": int((~is_r).sum()),
        "inference_seconds_new": new_seconds,
        "inference_seconds_base017": base_seconds,
        "r_max_abs_diff_vs_017": r_max_abs_diff,
        "f_mean_abs_diff_vs_017": float(
            np.mean(
                np.abs(
                    new_output.loc[~is_r, TARGET].to_numpy()
                    - base_output.loc[~is_r, TARGET].to_numpy()
                )
            )
        ),
        "prediction_min": float(probability.min()),
        "prediction_max": float(probability.max()),
        "prediction_mean": float(probability.mean()),
        "stderr_tail": stderr,
    }


def append_log(record: dict, training: dict) -> None:
    log = OUTPUT_DIR / "SUBMISSION_LOG.md"
    text = f"""# 2026-08-13 제출 기록

## submit_019 — 017 + F TabM 부분 풀링

| 항목 | 값 |
|---|---:|
| 기반 제출 | `submit_017.zip` (Public 895.404 계열) |
| F 혼합 | 기존 F 전문가 70% + TabM 30% |
| 가중치 선택 | 2023 F 월별 순차 OOF만 사용 |
| 2023 순차 F BSS | 128.242 → 145.656 |
| Val2024 F BSS | 507.640 → 540.543 |
| Val2024 F 개선 | +32.903 |
| 017 기준 전체 추정 | 약 837.214 BSS |
| 최종 TabM 학습 | 2023~2024 F {training['rows']:,}행, {training['epochs']} epoch |
| TrackMan | TabM에는 미사용; 기존 017 경로의 strict-as-of TM500만 유지 |
| 추론 시간(245,789행 혼합) | {record['inference_seconds_new']:.2f}초 |
| R 경로 차이 | {record['r_max_abs_diff_vs_017']:.3e} |
| 파일 크기 | {record['size_bytes'] / 1024**2:.1f} MiB |
| SHA256 | `{record['sha256']}` |

검증 시 2024는 혼합 가중치 선택에 사용하지 않았다. 2023의 네 개 월별 시계열
fold에서 30%를 선택한 뒤 해당 비율을 Val2024로 그대로 이전했다. ZIP 루트 구조,
CRC, 모델 존재, 245,789행 출력, 확률 범위, R 경로 불변성을 모두 확인했다.
"""
    log.write_text(text, encoding="utf-8")


def main() -> None:
    if not BASE.is_file() or not SCRIPT.is_file():
        raise FileNotFoundError("Base submit_017 or inference script is missing")
    base_features, training = train_final()
    package_record = package(base_features, training)
    record = smoke(package_record)
    record.update(
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "base": str(BASE.relative_to(ROOT)),
            "training": training,
            "validation": {
                "selection_2023_sequential_bss_before": 128.24199225889288,
                "selection_2023_sequential_bss_after": 145.6557396107616,
                "selected_tabm_weight": TABM_WEIGHT,
                "val2024_f_bss_before": 507.6399908947926,
                "val2024_f_bss_after": 540.5434893668071,
                "val2024_delta_f_bss": 32.90349847201446,
                "val2024_used_for_weight_selection": False,
                "submit017_exact_internal_bss": 833.3417,
                "projected_internal_bss": 837.2139522553058,
            },
        }
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(record, training)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
