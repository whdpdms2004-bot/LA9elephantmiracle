"""FA10C 최종 제출의 model/ 61개 파일과 ZIP을 처음부터 재생성한다.

기본 실행은 공식 train.csv 2019~2024 전체를 사용한다. 먼저 2019~2023으로
2024를 예측해 isotonic 좌표를 적합하고, 그 뒤 2019~2024 전체로 60개 모델을
재학습한다. 평가 데이터나 리더보드 정보는 읽지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

import features
import pipeline as fa


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def brier(target: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def write_zip(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in package_dir.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(package_dir).as_posix())


def check_feature_contract(full, raw_features) -> None:
    sample_parts = []
    for season in sorted(full["season"].unique()):
        part = full.loc[full["season"] == season]
        if not part.empty:
            sample_parts.append(part.head(200))
    sample = __import__("pandas").concat(sample_parts, ignore_index=True)
    train_for_stats = full.loc[full["season"] < 2024]
    cat_cols = features.get_categorical_columns()
    stats = features.fit_stats(train_for_stats)
    x, levels = fa.prepare(sample, full, raw_features, stats, cat_cols)
    assert x.shape[1] == 71
    assert list(x.columns[-3:]) == fa.A_COLS
    assert set(levels) == set(cat_cols)
    print(f"피처 계약 통과: {x.shape}, 마지막 3열={list(x.columns[-3:])}")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    data_dir = fa.find_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    package_dir = output_dir / "submission_package"
    model_dir = package_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    fa.log(f"data={data_dir}")
    raw_features = fa.raw_feature_list(data_dir)
    full = fa.load_train(data_dir, raw_features)
    fa.log(
        f"train={len(full):,}, seasons={sorted(full['season'].unique().tolist())}, "
        f"raw_features={len(raw_features)}"
    )
    check_feature_contract(full, raw_features)
    if args.check_only:
        print("check-only 완료: 모델 학습은 실행하지 않음")
        return

    # Stage 1: canonical 2024 holdout calibration.
    fa.log("Stage 1/4: 2019~2023 학습 -> 2024 예측 -> isotonic 좌표 적합")
    calibration_parts = fa.train_predictions_for_cutoff(
        full=full,
        raw_features=raw_features,
        cutoff=2023,
        valid_season=2024,
        checkpoint_dir=checkpoint_dir,
        seeds=fa.LGB_SEEDS,
        include_numeric_cb=True,
    )
    calibration_raw = fa.combine_raw(calibration_parts)
    calibration_rows = full.loc[full["season"] == 2024].reset_index(drop=True)
    calibration_target = calibration_rows[fa.TARGET].to_numpy(dtype=float)
    iso_x, iso_y = fa.fit_isotonic(calibration_raw, calibration_target)
    calibration_final = np.interp(calibration_raw, iso_x, iso_y)
    np.savez(
        checkpoint_dir / "calibration_2024.npz",
        row_id=calibration_rows[fa.ID_COL].astype(str).to_numpy(),
        lgb=calibration_parts["lgb"],
        numeric=calibration_parts["numeric"],
        team=calibration_parts["team"],
        raw=calibration_raw,
        final=calibration_final,
        target=calibration_target,
    )
    fa.log(
        f"calibration raw Brier={brier(calibration_target, calibration_raw):.9f}, "
        f"final Brier={brier(calibration_target, calibration_final):.9f}, "
        f"iso points={len(iso_x)}, max={max(iso_y):.3f}"
    )

    # Stage 2: all rows, same 71-feature ordering, 60 native models.
    fa.log("Stage 2/4: 2019~2024 전체로 LGB20 + numericCB20 + teamCB20 재학습")
    cat_cols = features.get_categorical_columns()
    stats = features.fit_stats(full)
    x_full, category_levels = fa.prepare(full, full, raw_features, stats, cat_cols)
    y_full = full[fa.TARGET].to_numpy(dtype=float)
    fa.train_family(
        "full", x_full, y_full, None, cat_cols, fa.LGB_SEEDS, "lgb", checkpoint_dir
    )
    fa.train_family(
        "full", x_full, y_full, None, cat_cols, fa.CB_SEEDS, "cb_num", checkpoint_dir
    )
    fa.train_family(
        "full", x_full, y_full, None, cat_cols, fa.CB_SEEDS, "cb_team", checkpoint_dir
    )

    # Stage 3: package with the exact inference contract.
    fa.log("Stage 3/4: cutoff=2024 lookup과 meta.json 패키징")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    model_dir.mkdir(parents=True)
    fa.copy_final_models(checkpoint_dir, model_dir)
    lookup = fa.build_inference_lookup(full, cutoff=2024)
    meta = fa.meta_payload(
        raw_features,
        cat_cols,
        category_levels,
        stats,
        iso_x,
        iso_y,
        lookup,
    )
    with (model_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)

    source_dir = Path(__file__).resolve().parent
    shutil.copy2(source_dir / "script_fa10c_inference.py", package_dir / "script.py")
    shutil.copy2(source_dir / "requirements.txt", package_dir / "requirements.txt")

    files = sorted(p.relative_to(package_dir).as_posix() for p in package_dir.rglob("*") if p.is_file())
    expected = {
        "script.py",
        "requirements.txt",
        "model/meta.json",
        *{f"model/lgb_booster_{seed}.txt" for seed in fa.LGB_SEEDS},
        *{f"model/cb_model_{seed}.cbm" for seed in fa.CB_SEEDS},
        *{f"model/cb_team_model_{seed}.cbm" for seed in fa.CB_SEEDS},
    }
    if set(files) != expected or len(files) != 63:
        raise RuntimeError(f"패키지 계약 불일치: files={len(files)}, missing={sorted(expected-set(files))}")

    # Stage 4: ZIP and reproducibility report.
    fa.log("Stage 4/4: ZIP 생성과 manifest 기록")
    zip_path = output_dir / "yn_fa10c_reproduced.zip"
    write_zip(package_dir, zip_path)
    report = {
        "data_dir": str(data_dir),
        "rows": len(full),
        "seasons": sorted(int(x) for x in full["season"].unique()),
        "raw_feature_count": len(raw_features),
        "model_feature_count": x_full.shape[1],
        "model_files": 60,
        "package_entries": 63,
        "calibration_raw_brier": brier(calibration_target, calibration_raw),
        "calibration_final_brier": brier(calibration_target, calibration_final),
        "calibration_pred_mean": float(calibration_final.mean()),
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "environment": fa.environment_versions(),
        "elapsed_seconds": time.time() - started,
    }
    with (output_dir / "reproduction_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    fa.log(f"완료: {zip_path} ({zip_path.stat().st_size:,} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="train.csv/test.csv가 있는 디렉터리")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "build"),
        help="체크포인트와 재현 ZIP 출력 경로",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="데이터·71피처 계약만 확인하고 학습하지 않음",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
