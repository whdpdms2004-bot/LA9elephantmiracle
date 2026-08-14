"""Exact Val2024 evaluation for the two reverse20 submission candidates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MODEL_OPT = ROOT / "experiment" / "model_optimization"
WORK = MODEL_OPT / "pitcher_cluster_matchup"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(MODEL_OPT))

from analyze_r_focus import load_fold_predictions  # noqa: E402
from integrate_r_middle_current_blend import attach_honest_references  # noqa: E402
from run_reverse_seedbag20 import (  # noqa: E402
    CACHE,
    FEATURES,
    SEEDS,
    SUCCESS_FEATURES,
    TARGET,
    correction,
    load_base,
)
from screen_r_context_history import history_correction, load_game  # noqa: E402
from screen_r_middle_preprocessing import (  # noqa: E402
    load_base as load_context_base,
    load_main as load_context_main,
)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    brier = float(np.mean(np.square(p - y)))
    null = float(y.mean() * (1 - y.mean()))
    return {
        "n": int(len(y)),
        "brier": brier,
        "bss": 100000.0 * (1 - brier / null),
        "target_mean": float(y.mean()),
        "prediction_mean": float(p.mean()),
    }


def r_context_2024(folds: dict[int, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    main = load_context_main()
    base = load_context_base().merge(
        main[["row_id", "game_type"]], on="row_id", validate="one_to_one"
    )
    base = attach_honest_references(base, folds)
    frame = base.merge(
        load_game().drop(columns=["game_type"]),
        on=["row_id", "season"], validate="one_to_one",
    )
    value, _ = history_correction(
        frame,
        ["balls_before", "strikes_before", "inning4", "pitcher_hand", "batter_hand"],
        2024,
        5000.0,
        1.0,
    )
    rows = frame.loc[frame["season"].eq(2024), "row_id"].to_numpy()
    return rows, 1.15 * value


def main() -> None:
    base = load_base()
    valid = base["season"].eq(2024)
    current = base.loc[valid, ["row_id", TARGET, "prediction"]].copy()
    row_id = current["row_id"].to_numpy()
    y = current[TARGET].to_numpy(float)
    adjusted = current["prediction"].to_numpy(float)
    success = correction(base, SUCCESS_FEATURES, 10.0, 2023, 2024)
    reverse = []
    for seed in SEEDS:
        features = base.merge(
            pd.read_parquet(CACHE / f"seed_{seed}.parquet"),
            on=["row_id", "season"], validate="one_to_one",
        )
        reverse.append(correction(features, FEATURES, 1000.0, 2023, 2024))
    reverse20 = np.mean(reverse, axis=0)
    reverse3 = np.mean(reverse[:3], axis=0)
    ensemble = pd.read_parquet(MODEL_OPT / "enhanced_ensemble_oof_predictions.parquet")
    ensemble = ensemble.loc[
        ensemble["season"].eq(2024) & ensemble["track"].eq("performance"),
        ["row_id", "prediction"],
    ].set_index("row_id").loc[row_id, "prediction"].to_numpy(float)

    folds = load_fold_predictions()
    old013 = folds[2024].set_index("row_id").loc[row_id, "current_blend"].to_numpy(float)
    r_rows, r_correction = r_context_2024(folds)
    r_correction = pd.Series(r_correction, index=r_rows).loc[row_id].to_numpy(float)

    expert = pd.read_parquet(MODEL_OPT / "game_type_experts" / "f_seedbag_oof.parquet")
    xgb = expert.loc[
        expert["family"].eq("xgboost") & expert["seed_index"].eq(0)
    ].set_index("row_id")["prediction"].reindex(row_id).to_numpy(float)
    cat = expert.loc[
        expert["family"].eq("catboost") & expert["seed_index"].eq(0)
    ].set_index("row_id")["prediction"].reindex(row_id).to_numpy(float)
    tabm = pd.read_parquet(
        MODEL_OPT / "tabm_context" / "outputs" / "gate24_f_post23_t0" / "oof_all.parquet"
    ).set_index("row_id").reindex(row_id)["prediction"].to_numpy(float)
    game = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=["row_id", "season", "game_type", "pitcher_id"],
    )
    game = game.loc[game["season"].eq(2024)].set_index("row_id").loc[row_id]
    is_f = game["game_type"].astype(str).eq("F").to_numpy()
    dx, dc, dt = xgb - adjusted, cat - adjusted, tabm - adjusted

    def base20(scale: float) -> np.ndarray:
        corrected = np.clip(adjusted + 0.25 * success + scale * reverse20, 1e-6, 1 - 1e-6)
        return 0.6085 * corrected + 0.3915 * ensemble

    def finish(base_prediction: np.ndarray, tabm_extension: bool) -> np.ndarray:
        prediction = np.clip(base_prediction + r_correction, 1e-6, 1 - 1e-6)
        f_delta = (
            0.105 * dx + 0.133 * dc + 0.30 * dt
            if tabm_extension
            else 0.15 * dx + 0.19 * dc
        )
        prediction[is_f] = np.clip(
            prediction[is_f] + f_delta[is_f], 1e-6, 1 - 1e-6
        )
        return prediction

    baselines = {
        "submit017_reconstructed": finish(old013.copy(), False),
        "submit019_reconstructed": finish(old013.copy(), True),
    }
    candidates = {
        "submit020_reverse20_s055_tabm": finish(base20(0.55), True),
        "submit021_reverse20_s040_tabm": finish(base20(0.40), True),
    }
    records = {}
    for name, prediction in {**baselines, **candidates}.items():
        records[name] = {
            "all": metrics(y, prediction),
            "R": metrics(y[~is_f], prediction[~is_f]),
            "F": metrics(y[is_f], prediction[is_f]),
        }
    records["parity"] = {
        "submit017_expected": 833.3416848376296,
        "submit019_expected": 835.8612351,
        "submit017_abs_error": abs(records["submit017_reconstructed"]["all"]["bss"] - 833.3416848376296),
        "submit019_abs_error": abs(records["submit019_reconstructed"]["all"]["bss"] - 835.8612351),
    }
    output = pd.DataFrame(
        {
            "row_id": row_id,
            TARGET: y.astype("int8"),
            "game_type": game["game_type"].to_numpy(),
            "pitcher_id": game["pitcher_id"].to_numpy(),
            **{name: value.astype("float32") for name, value in {**baselines, **candidates}.items()},
        }
    )
    output.to_parquet(WORK / "reports" / "reverse20_submission_oof.parquet", index=False)
    np.savez_compressed(
        WORK / "reports" / "reverse20_submission_components.npz",
        row_id=row_id,
        y=y,
        adjusted=adjusted,
        success=success,
        reverse3=reverse3,
        reverse20=reverse20,
        ensemble=ensemble,
        r_correction=r_correction,
        is_f=is_f,
        dx=dx,
        dc=dc,
        dt=dt,
    )
    (WORK / "reports" / "reverse20_submission_metrics.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
