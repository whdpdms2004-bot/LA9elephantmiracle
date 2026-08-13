from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_optuna_family import ROOT, TARGET, probability_metrics


WORK_DIR = ROOT / "experiment" / "model_optimization"
BINARY_MODEL = "xgboost_v2_row_selected_200"


def main():
    binary = pd.read_parquet(WORK_DIR / "v2_ablation_predictions.parquet")
    multi = pd.read_parquet(WORK_DIR / "failure_multiclass_predictions.parquet")
    binary = binary[binary["model"].eq(BINARY_MODEL)].rename(
        columns={"prediction": "binary_prediction"}
    )
    rows = []
    curves = []
    for model_name, model_frame in multi.groupby("model", sort=True):
        joined = binary.merge(
            model_frame[["row_id", "season", "prediction"]].rename(
                columns={"prediction": "multiclass_prediction"}
            ),
            on=["row_id", "season"],
            how="inner",
            validate="one_to_one",
        )
        if set(joined["season"].unique()) != {2023, 2024}:
            continue
        fold_best = {}
        for fold in [2023, 2024]:
            part = joined[joined["season"].eq(fold)]
            y = part[TARGET].to_numpy()
            p_binary = part["binary_prediction"].to_numpy(float)
            p_multi = part["multiclass_prediction"].to_numpy(float)
            correlation = float(np.corrcoef(p_binary, p_multi)[0, 1])
            direction = p_binary - p_multi
            binary_weight = float(
                np.clip(
                    -np.dot(p_multi - y, direction) / np.dot(direction, direction),
                    0.0,
                    1.0,
                )
            )
            prediction = binary_weight * p_binary + (1.0 - binary_weight) * p_multi
            metrics = probability_metrics(y, prediction)
            best = (metrics["brier"], binary_weight, metrics)
            fold_best[fold] = best
            rows.append(
                {
                    "experiment": "failure_same_fold_blend",
                    "multiclass_model": model_name,
                    "fold": fold,
                    "selection_fold": fold,
                    "binary_weight": best[1],
                    "prediction_correlation": correlation,
                    **best[2],
                }
            )

        # Honest transition: choose the weight on 2023, freeze it, score 2024.
        weight = fold_best[2023][1]
        part = joined[joined["season"].eq(2024)]
        prediction = (
            weight * part["binary_prediction"].to_numpy(float)
            + (1.0 - weight) * part["multiclass_prediction"].to_numpy(float)
        )
        metrics = probability_metrics(part[TARGET].to_numpy(), prediction)
        rows.append(
            {
                "experiment": "failure_cross_year_blend",
                "multiclass_model": model_name,
                "fold": 2024,
                "selection_fold": 2023,
                "binary_weight": weight,
                "prediction_correlation": float(
                    np.corrcoef(
                        part["binary_prediction"], part["multiclass_prediction"]
                    )[0, 1]
                ),
                **metrics,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["experiment", "fold", "brier", "multiclass_model"]
    )
    result.to_csv(WORK_DIR / "failure_blend_results.csv", index=False)
    summary = {
        "binary_model": BINARY_MODEL,
        "weight_solver": "closed_form_nonnegative_two_model_brier_optimum",
        "rows": result.to_dict("records"),
    }
    (WORK_DIR / "failure_blend_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
