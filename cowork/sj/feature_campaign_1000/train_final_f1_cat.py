"""2019~2024 strict forward-OOF F1 CatBoost 단독 모델과 추론 자산을 생성한다.

학습행의 Target lookup은 각 행 시즌보다 이전 시즌으로만 만들고, 2025 추론
lookup은 2019~2024 전체 학습 데이터로 고정한다. 모델 산출물은 outputs/ 아래에
생성되며 저장소에 커밋하지 않는다.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


HERE = Path(__file__).resolve().parent
SJ = HERE.parent
MO = SJ / "experiment" / "model_optimization"
SRC = SJ / "claude" / "src"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC))

import component_features as CF
from evaluate_train_only_season_offsets import forecast_offset
from v77_single_xgb_screen import (
    CATEGORICAL_COLUMNS, TARGET, arm_features,
    build_component_unique_forward, load_enhanced_frame, recency_weights,
)


DEFAULT_OUT = HERE / "outputs" / "final_f1_cat_v1"
PARAMS_PATH = MO / "catboost_v2r200_tm500_robust_best.json"
TRACKMAN_LOOKUP = MO / "trackman500_lookup_2025.parquet"
BASE_RUNTIME = HERE / "f1_base_runtime.py"
COMPONENT_RUNTIME = HERE / "f1_component_runtime.py"
INFERENCE_SOURCE = HERE / "f1_inference.py"
ITERATIONS = 2595
RANDOM_SEED = 20262843


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_assets(frame: pd.DataFrame, raw_columns: list[str], model_dir: Path):
    labels = pd.read_parquet(SJ / "claude" / "cache" / "failure_labels.parquet")
    if not frame["row_id"].equals(labels["row_id"]):
        raise RuntimeError("failure label cache row order mismatch")
    train = frame[[*raw_columns, TARGET]]
    ok = labels["label_ok"].to_numpy() == 1
    middle = np.where(ok, labels["y_middle"].to_numpy(np.float64), np.nan)
    reverse = np.where(ok, labels["y_reverse"].to_numpy(np.float64), np.nan)
    outside = np.where(ok, labels["y_outside"].to_numpy(np.float64), np.nan)
    ball = np.where(ok, labels["y_ball"].to_numpy(np.float64), np.nan)
    components = {
        "m": middle,
        "r": reverse,
        "mr": np.where(ok, ((middle == 1) & (reverse == 1)).astype(float), np.nan),
        "ob": np.where(ok, ((outside == 1) & (ball == 1)).astype(float), np.nan),
        "oz": np.where(ok, ((outside == 1) & (ball == 0)).astype(float), np.nan),
    }
    spec = CF.make_spec(train)
    tables = {
        "platoon": CF.make_platoon_table(train),
        "bat_platoon": CF.make_batter_platoon_table(train, components),
        "count_platoon": CF.make_count_platoon_table(train),
        "inning_platoon": CF.make_inning_platoon_table(train),
    }
    files = {}
    for name, table in tables.items():
        filename = f"f1_{name}_2025.csv"
        table.to_csv(model_dir / filename, index=False)
        files[name] = filename
    built = CF.build(
        frame[raw_columns], spec, tables["platoon"], tables["bat_platoon"],
        tables["count_platoon"], tables["inning_platoon"])
    return spec, files, built


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    model_dir = output_dir / "model"
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"refusing to overwrite {output_dir}; pass --force intentionally")
        shutil.rmtree(output_dir)
    model_dir.mkdir(parents=True)

    started = time.time()
    frame, base_features = load_enhanced_frame()
    raw_columns = [
        column for column in frame.columns[:49] if column not in ("row_id", TARGET)]
    forward = build_component_unique_forward(frame, base_features, 2025)
    work = frame.copy(deep=False)
    features = arm_features(work, base_features, "F1", 2025, forward)
    if len(features) != 272:
        raise AssertionError(f"unexpected F1 feature count: {len(features)}")
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    model_frame = work[features].copy()
    for column in categorical:
        model_frame[column] = model_frame[column].fillna("__MISSING__").astype(str)
    target = frame[TARGET].to_numpy("int8")
    season = frame["season"].to_numpy("int16")

    params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(params.pop("half_life"))
    params["iterations"] = ITERATIONS
    weights = recency_weights(season, 2025, half_life)
    pool = Pool(
        model_frame, label=target, cat_features=categorical, weight=weights)
    model = CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="GPU",
        devices="0",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )
    print(
        f"training final CatBoost F1 rows={len(frame)} features={len(features)} "
        f"iterations={ITERATIONS}", flush=True)
    model.fit(pool)
    model_file = "f1_catboost.cbm"
    model.save_model(str(model_dir / model_file))
    del model, pool, weights, model_frame
    gc.collect()

    spec, asset_files, inference_component = component_assets(
        frame, raw_columns, model_dir)
    component_columns = forward.columns.tolist()
    expected_component = inference_component.rename(
        columns={column: f"sx_cf_{column}" for column in inference_component})
    missing_component = [
        column for column in component_columns if column not in expected_component]
    if missing_component:
        raise AssertionError(missing_component)

    trackman = pd.read_parquet(TRACKMAN_LOOKUP)
    trackman_file = "trackman500_lookup_2025.csv"
    trackman.to_csv(model_dir / trackman_file, index=False)
    base_runtime_file = "base_runtime.py"
    component_runtime_file = "component_runtime.py"
    shutil.copy2(BASE_RUNTIME, model_dir / base_runtime_file)
    shutil.copy2(COMPONENT_RUNTIME, model_dir / component_runtime_file)
    shutil.copy2(INFERENCE_SOURCE, output_dir / "script.py")

    rates = frame.groupby("season")[TARGET].mean()
    season_offset = forecast_offset(rates, 2025, window=None, damping=0.25)
    metadata = {
        "version": 1,
        "track": "strict_forward_oof_f1_catboost_single",
        "target": TARGET,
        "feature_columns": features,
        "feature_sets": {"enhanced": base_features},
        "categorical_columns": categorical,
        "model_file": model_file,
        "iterations": ITERATIONS,
        "random_seed": RANDOM_SEED,
        "half_life": half_life,
        "season_logit_offset": season_offset,
        "season_offset_source": (
            "25% of all-available-season linear logit trend extrapolation; "
            "training Target season means only; no test predictions or leaderboard"),
        "raw_columns": raw_columns,
        "component_feature_columns": component_columns,
        "component_spec": spec,
        "component_assets": asset_files,
        "base_runtime_file": base_runtime_file,
        "component_runtime_file": component_runtime_file,
        "trackman_lookup_file": trackman_file,
        "trackman_columns": [
            column for column in trackman if column != "pitcher_id"],
        "trackman_rule": "2019-2024 only; pitcher-season >=500; fixed 2025 lookup",
        "validation": {
            "cat_f1_all_linear_d025": {
                "2022": 2301.262255,
                "2023": 64.023919,
                "2024": 796.733927,
                "mean_bss": 1054.006700,
            },
            "training_feature_rule": (
                "2019 zero fallback; each 2020-2024 row uses only prior seasons"),
        },
    }
    metadata_path = model_dir / "f1_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "requirements.txt").write_text(
        "catboost==1.2.8\n", encoding="utf-8")
    manifest = {
        "elapsed_sec": time.time() - started,
        "files": {
            str(path.relative_to(output_dir)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(output_dir.rglob("*")) if path.is_file()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "season_logit_offset": season_offset,
        "feature_count": len(features),
        "elapsed_sec": manifest["elapsed_sec"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
