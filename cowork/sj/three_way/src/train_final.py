"""3WAY middle/reverse/outside/middle-intersection 모델을 2019~2024 전체로 학습한다.

최종 출력은 ``outputs/final_three_way_v1`` 아래에 생성한다. 학습 모델 네 개는
CatBoost GPU를 순차 사용하며, 추론 자산은 2019~2024 학습 데이터에서 고정한다.
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


HERE = Path(__file__).resolve().parent
TW = HERE.parent
SJ = TW.parent
CAMPAIGN = SJ / "feature_campaign_1000"
MODEL_OPT = SJ / "experiment" / "model_optimization"
CLAUDE_SRC = SJ / "claude" / "src"
LAB = SJ / "preprocess_lab"
for path in (HERE, CAMPAIGN, MODEL_OPT, CLAUDE_SRC, LAB):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guards import assert_features_clean, train_season_trend
from harness3 import SUCCESS, load_labeled
import three_way_runtime as RUNTIME


DEFAULT_OUTPUT = TW / "outputs" / "final_three_way_v1"
BASE_ASSET_DIR = CAMPAIGN / "outputs" / "final_f1_cat_v1" / "model"
PARAMS_PATH = MODEL_OPT / "catboost_v2r200_tm500_robust_best.json"
SEED = 20262844
PREDICTION_SEASON = 2025
TARGET_CONFIG = {
    "middle": {
        "label": "y_middle",
        "combo": ["id_frequency", "no_trackman", "temporal_cyclic"],
    },
    "reverse": {
        "label": "y_reverse",
        "combo": ["count_multiscale", "drop_ids", "trackman_quality"],
    },
    "outside": {
        "label": "y_outside",
        "combo": ["drop_ids", "no_trackman", "rate_multiscale"],
    },
    "mr": {
        "label": "y_mr",
        "combo": ["id_frequency", "no_trackman", "temporal_cyclic"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_base_assets(model_dir: Path) -> dict:
    metadata_path = BASE_ASSET_DIR / "f1_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "strict F1 2019~2024 inference assets are missing; run "
            "feature_campaign_1000/train_final_f1_cat.py first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    filenames = {
        metadata["base_runtime_file"],
        metadata["component_runtime_file"],
        metadata["trackman_lookup_file"],
        *metadata["component_assets"].values(),
    }
    for filename in sorted(filenames):
        source = BASE_ASSET_DIR / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, model_dir / filename)
    (model_dir / "base_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def build_transform_assets(
        frame: pd.DataFrame,
        train_mask: np.ndarray,
        model_dir: Path) -> tuple[dict, dict[str, str], dict[str, pd.Series]]:
    id_files = {}
    id_lookups = {}
    for column in RUNTIME.ID_COLUMNS:
        table = (frame.loc[train_mask, column].value_counts(dropna=False)
                 .rename_axis(column).reset_index(name="frequency"))
        filename = f"id_frequency_{column}_2025.csv"
        table.to_csv(model_dir / filename, index=False)
        id_files[column] = filename
        id_lookups[column] = table.set_index(column)["frequency"]

    rate_priors = {}
    for _, rate_column, _ in RUNTIME.RATE_SPECS:
        prior = float(pd.to_numeric(
            frame.loc[train_mask, rate_column], errors="coerce").median())
        if not np.isfinite(prior):
            raise RuntimeError(f"non-finite rate prior: {rate_column}")
        rate_priors[rate_column] = prior

    tm_columns = [column for column in frame
                  if column.startswith(("tm500_", "cw_"))]
    physical = [column for column in tm_columns
                if column.startswith("tm500_latest_") and column.endswith("_mean")]
    dispersion = [column for column in tm_columns
                  if column.startswith("tm500_latest_") and column.endswith("_std")]
    shifts = [column for column in tm_columns if column.endswith("_minus_recent")]
    robust_stats = {}
    for column in physical + dispersion + shifts:
        values = pd.to_numeric(
            frame.loc[train_mask, column], errors="coerce").to_numpy(np.float64)
        median = float(np.nanmedian(values))
        q25, q75 = np.nanpercentile(values, [25, 75])
        scale = max(float(q75 - q25), 1e-6)
        if not np.isfinite(median) or not np.isfinite(scale):
            raise RuntimeError(f"non-finite robust statistic: {column}")
        robust_stats[column] = {"median": median, "scale": scale}
    spec = {
        "source": "2019-2024 train rows with reconstructable component labels",
        "rate_priors": rate_priors,
        "trackman_quality": {
            "tm_columns": tm_columns,
            "physical": physical,
            "dispersion": dispersion,
            "shifts": shifts,
            "robust_stats": robust_stats,
        },
    }
    return spec, id_files, id_lookups


def assert_runtime_equivalence(
        reference: pd.DataFrame,
        runtime: pd.DataFrame,
        features: list[str],
        categorical: list[str],
        target: str) -> None:
    sample = np.linspace(
        0, len(reference) - 1, num=min(4096, len(reference)), dtype=int)
    if any(column not in runtime for column in features):
        missing = [column for column in features if column not in runtime]
        raise AssertionError(f"{target} runtime missing: {missing}")
    categorical_set = set(categorical)
    for column in features:
        left = reference.iloc[sample][column]
        right = runtime.iloc[sample][column]
        if column in categorical_set:
            a = left.fillna("__MISSING__").astype(str).to_numpy()
            b = right.fillna("__MISSING__").astype(str).to_numpy()
            if not np.array_equal(a, b):
                raise AssertionError(f"{target} categorical mismatch: {column}")
        else:
            a = pd.to_numeric(left, errors="coerce").to_numpy(np.float64)
            b = pd.to_numeric(right, errors="coerce").to_numpy(np.float64)
            if not np.allclose(a, b, rtol=0.0, atol=1e-12, equal_nan=True):
                delta = float(np.nanmax(np.abs(a - b)))
                raise AssertionError(
                    f"{target} numeric mismatch: {column}, max={delta}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    allowed_parent = (TW / "outputs").resolve()
    if output_dir.parent != allowed_parent:
        raise ValueError(f"output must be an immediate child of {allowed_parent}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True)
    started = time.time()

    base_metadata = copy_base_assets(model_dir)
    shutil.copy2(HERE / "three_way_runtime.py", model_dir / "three_way_runtime.py")
    shutil.copy2(HERE / "three_way_inference.py", output_dir / "script.py")
    (output_dir / "requirements.txt").write_text(
        "catboost==1.2.8\n", encoding="utf-8")

    from run_optuna_enhanced import load_enhanced_frame
    from run_optuna_family import CATEGORICAL_COLUMNS, recency_weights
    from v77_single_xgb_screen import (
        build_component_unique, build_component_unique_forward)
    from v80_single_catboost import make_features
    import transforms as T

    T.load_all()
    frame, enhanced = load_enhanced_frame()
    labeled = load_labeled()
    if not np.array_equal(frame["row_id"].to_numpy(), labeled["row_id"].to_numpy()):
        raise RuntimeError("failure label row order mismatch")
    if sorted(frame["season"].unique().tolist()) != [2019, 2020, 2021, 2022, 2023, 2024]:
        raise RuntimeError("final frame must contain exactly seasons 2019-2024")

    hierarchy = build_component_unique(frame, enhanced, PREDICTION_SEASON)
    forward = build_component_unique_forward(
        frame, enhanced, PREDICTION_SEASON, cache={PREDICTION_SEASON: hierarchy})
    base_frame, base_features = make_features(
        frame, enhanced, PREDICTION_SEASON, "F1", forward)
    for column in (SUCCESS, "season"):
        if column not in base_frame:
            base_frame[column] = frame[column].to_numpy()
    packaged_features = base_metadata["feature_columns"]
    if set(base_features) != set(packaged_features):
        missing = sorted(set(base_features) - set(packaged_features))
        extra = sorted(set(packaged_features) - set(base_features))
        raise AssertionError({
            "training_only_features": missing,
            "packaged_only_features": extra,
        })

    component_ok = labeled["label_ok"].to_numpy() == 1
    transform_spec, id_files, id_lookups = build_transform_assets(
        base_frame, component_ok, model_dir)
    base_categorical = [column for column in CATEGORICAL_COLUMNS
                        if column in base_features]
    train_series = pd.Series(component_ok, index=frame.index)
    params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(params.pop("half_life"))
    params.update({
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": SEED,
        "task_type": args.task_type,
        "verbose": False,
        "allow_writing_files": False,
    })
    if args.task_type == "GPU":
        params["devices"] = "0"

    model_specs = []
    from catboost import CatBoostClassifier, Pool
    for target, config in TARGET_CONFIG.items():
        label = config["label"]
        values = pd.to_numeric(labeled[label], errors="coerce").to_numpy(np.float64)
        train_mask = component_ok & np.isfinite(values)
        if not np.array_equal(train_mask, component_ok):
            raise RuntimeError(f"unexpected missing {target} labels")
        combo = sorted(config["combo"])
        reference, features, categorical = T.build(
            base_frame, base_features, base_categorical, combo,
            train_series, PREDICTION_SEASON)
        assert_features_clean(features, f"final {target}")
        runtime = RUNTIME.add_target_features(
            base_frame, combo, transform_spec, id_lookups, PREDICTION_SEASON)
        assert_runtime_equivalence(
            reference, runtime, features, categorical, target)

        season_values = frame.loc[train_mask, "season"].to_numpy()
        target_values = values[train_mask].astype("int8")
        prior = train_season_trend(
            target_values, season_values, PREDICTION_SEASON)
        baseline_logit = float(np.log(prior / (1.0 - prior)))
        season_rates = (pd.Series(target_values).groupby(pd.Series(season_values))
                        .mean().sort_index())
        model_file = f"three_way_{target}.cbm"
        spec = {
            "target": target,
            "label": label,
            "combo": combo,
            "feature_columns": features,
            "categorical_columns": categorical,
            "model_file": model_file,
            "n_features": len(features),
            "n_train": int(train_mask.sum()),
            "prior_2025": prior,
            "baseline_logit": baseline_logit,
            "season_target_rates": {
                str(int(season)): float(rate)
                for season, rate in season_rates.items()},
            "prior_source": (
                "linear probability extrapolation from 2019-2024 component "
                "target rates; no test values or leaderboard feedback"),
        }
        model_specs.append(spec)
        print(
            f"prepared {target}: rows={spec['n_train']:,} "
            f"features={len(features)} prior={prior:.8f}", flush=True)
        if args.prepare_only:
            del reference, runtime
            gc.collect()
            continue

        model_frame = reference.loc[train_mask, features].copy()
        for column in categorical:
            model_frame[column] = (
                model_frame[column].fillna("__MISSING__").astype(str))
        weights = np.asarray(recency_weights(
            frame.loc[train_mask, "season"], PREDICTION_SEASON, half_life),
            np.float64)
        pool = Pool(
            model_frame,
            label=target_values,
            cat_features=categorical,
            weight=weights,
            baseline=np.full(int(train_mask.sum()), baseline_logit),
        )
        model = CatBoostClassifier(**params)
        target_started = time.time()
        print(f"training {target} on {args.task_type}", flush=True)
        model.fit(pool)
        model.save_model(str(model_dir / model_file))
        print(f"saved {model_file} in {time.time() - target_started:.1f}s", flush=True)
        del reference, runtime, model_frame, weights, pool, model
        gc.collect()

    metadata = {
        "version": 1,
        "track": "three_way_mro_identity_full_2019_2024",
        "prediction_season": PREDICTION_SEASON,
        "training_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
        "training_rows_total": int(len(frame)),
        "training_rows_component_labels": int(component_ok.sum()),
        "random_seed": SEED,
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "half_life": half_life,
        "training_device": args.task_type,
        "final_ready": not args.prepare_only,
        "combination": "clip(1 - (middle + reverse - mr + outside))",
        "season_logit_offset": 0.0,
        "season_offset_source": "none; component training priors are documented per model",
        "component_label_source": (
            "train.csv official asof cumulative rates differenced only to create "
            "training targets; reconstructed labels never enter inference features"),
        "base_metadata": base_metadata,
        "three_way_runtime_file": "three_way_runtime.py",
        "id_frequency_files": id_files,
        "transform_spec": transform_spec,
        "models": model_specs,
        "validation_reference": {
            "fold": 2024,
            "rows_with_component_labels": 253035,
            "brier": 0.247736118950132,
            "bss_raw": 828.5897162937816,
            "bss_centered_diagnostic_only": 840.2271057839861,
            "prediction_mean": 0.4914628401567227,
            "target_mean": 0.4860710968838303,
        },
        "trackman_rule": (
            "fixed 2025 lookup built from 2019-2024 TrackMan only; same verified "
            "assets as submit_036"),
    }
    metadata_path = model_dir / "three_way_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    manifest = {
        "elapsed_sec": time.time() - started,
        "prepare_only": bool(args.prepare_only),
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
        "final_ready": metadata["final_ready"],
        "models": [item["model_file"] for item in model_specs],
        "elapsed_sec": manifest["elapsed_sec"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
