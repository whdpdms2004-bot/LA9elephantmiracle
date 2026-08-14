from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
LARGE_DIR = MODEL_DIR / "large_xgb" / "predictions"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
REPORTS = WORK / "reports"
TARGET = "control_success"
SUCCESS_CONFIG = "m_com_gmm_kme_b3r4_l1000_h1_d72788b5"
SUCCESS_FEATURES = [
    "match_pair_delta", "match_pair_delta_reliability",
    "match_pair_delta_rate", "match_pair_known",
]
REVERSE_FEATURES = [
    "reverse_pair_delta", "reverse_pair_delta_reliability",
    "reverse_pair_rate", "reverse_pair_known",
]
REVERSE_SEEDS = [17, 2026, 4099]
EPS = 1e-6


def logit(values):
    p = np.clip(np.asarray(values, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(values):
    z = np.clip(np.asarray(values, dtype="float64"), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def metrics(y, prediction):
    y = np.asarray(y, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    brier = float(np.mean(np.square(p - y)))
    denominator = float(y.mean() * (1.0 - y.mean()))
    return {
        "brier": brier,
        "normalized_brier": brier / denominator,
        "bss": max(0.0, 100000.0 * (1.0 - brier / denominator)),
        "pred_mean": float(p.mean()),
        "pred_std": float(p.std()),
    }


def read_raw():
    output = {}
    for fold in [2022, 2023, 2024]:
        anchor = pd.read_parquet(
            LARGE_DIR / f"anchor_logloss_f{fold}_s0.parquet"
        ).sort_values("row_id").reset_index(drop=True)
        large = pd.read_parquet(
            LARGE_DIR / f"moderate24_diverse_f{fold}_s0.parquet"
        ).sort_values("row_id").reset_index(drop=True)
        if not anchor[["row_id", "season", TARGET]].equals(
            large[["row_id", "season", TARGET]]
        ):
            raise RuntimeError(f"Raw prediction alignment failed for {fold}")
        output[fold] = {
            "row_id": anchor["row_id"].to_numpy(),
            "y": anchor[TARGET].to_numpy("float64"),
            "anchor": anchor["prediction"].to_numpy("float64"),
            "large": large["prediction"].to_numpy("float64"),
        }
    return output


def optimal_pair_weight(y, left, right):
    direction = right - left
    denominator = float(np.dot(direction, direction))
    return 0.0 if denominator == 0 else float(
        np.clip(np.dot(y - left, direction) / denominator, 0.0, 1.0)
    )


def build_variants(raw):
    variants = {"anchor": {}, "large_raw": {}}
    names = [
        "prior_prob_mean",
        "prior_prob_mean_std",
        "prior_logit_mean",
        "prior_logit_mean_std",
        "prior_target_blend",
    ]
    variants.update({name: {} for name in names})
    parameters = {name: {} for name in names}
    fixed_weights = np.round(np.arange(0.1, 1.0, 0.1), 1)
    for weight in fixed_weights:
        name = f"fixed_large_w{int(round(weight * 10)):02d}"
        variants[name] = {}
        parameters[name] = {
            "large_weight": float(weight),
            "rule": "Same fixed anchor-large weight in every fold",
        }
    for fold in [2022, 2023, 2024]:
        source_fold = 2022 if fold == 2022 else fold - 1
        source = raw[source_fold]
        current = raw[fold]
        variants["anchor"][fold] = current["anchor"]
        variants["large_raw"][fold] = current["large"]

        prob_shift = float(source["anchor"].mean() - source["large"].mean())
        prob_scale = float(source["anchor"].std() / max(source["large"].std(), EPS))
        prob_intercept = float(
            source["anchor"].mean() - prob_scale * source["large"].mean()
        )
        source_anchor_logit = logit(source["anchor"])
        source_large_logit = logit(source["large"])
        logit_shift = float(source_anchor_logit.mean() - source_large_logit.mean())
        logit_scale = float(
            source_anchor_logit.std() / max(source_large_logit.std(), EPS)
        )
        logit_intercept = float(
            source_anchor_logit.mean() - logit_scale * source_large_logit.mean()
        )
        target_weight = optimal_pair_weight(
            source["y"], source["anchor"], source["large"]
        )

        variants["prior_prob_mean"][fold] = np.clip(
            current["large"] + prob_shift, EPS, 1.0 - EPS
        )
        variants["prior_prob_mean_std"][fold] = np.clip(
            prob_scale * current["large"] + prob_intercept, EPS, 1.0 - EPS
        )
        variants["prior_logit_mean"][fold] = sigmoid(
            logit(current["large"]) + logit_shift
        )
        variants["prior_logit_mean_std"][fold] = sigmoid(
            logit_scale * logit(current["large"]) + logit_intercept
        )
        variants["prior_target_blend"][fold] = (
            current["anchor"]
            + target_weight * (current["large"] - current["anchor"])
        )
        for weight in fixed_weights:
            name = f"fixed_large_w{int(round(weight * 10)):02d}"
            variants[name][fold] = (
                current["anchor"]
                + float(weight) * (current["large"] - current["anchor"])
            )
        parameters["prior_prob_mean"][fold] = {
            "source_fold": source_fold, "shift": prob_shift,
        }
        parameters["prior_prob_mean_std"][fold] = {
            "source_fold": source_fold, "scale": prob_scale,
            "intercept": prob_intercept,
        }
        parameters["prior_logit_mean"][fold] = {
            "source_fold": source_fold, "shift": logit_shift,
        }
        parameters["prior_logit_mean_std"][fold] = {
            "source_fold": source_fold, "scale": logit_scale,
            "intercept": logit_intercept,
        }
        parameters["prior_target_blend"][fold] = {
            "source_fold": source_fold, "large_weight": target_weight,
        }
    return variants, parameters


def attach_features(raw):
    ids = pd.concat(
        [
            pd.DataFrame({
                "row_id": raw[fold]["row_id"],
                "season": fold,
                TARGET: raw[fold]["y"],
            })
            for fold in [2022, 2023, 2024]
        ],
        ignore_index=True,
    )
    success = pd.read_parquet(
        WORK / "oof" / f"matchup_features_{SUCCESS_CONFIG}.parquet",
        columns=["row_id", "season", *SUCCESS_FEATURES],
    )
    attached = {"success": ids.merge(success, on=["row_id", "season"], validate="one_to_one")}
    for seed in REVERSE_SEEDS:
        reverse = pd.read_parquet(
            WORK / "oof" / "reverse_batter_seed" / f"seed_{seed}.parquet"
        )
        attached[str(seed)] = ids.merge(
            reverse, on=["row_id", "season"], validate="one_to_one"
        )
    return attached


def fit_correction(frame, base_prediction, features, alpha, train_year, valid_year):
    train = frame["season"].eq(train_year).to_numpy()
    valid = frame["season"].eq(valid_year).to_numpy()
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=False),
    )
    residual = frame.loc[train, TARGET].to_numpy("float64") - base_prediction[train]
    model.fit(frame.loc[train, features], residual)
    return np.clip(model.predict(frame.loc[valid, features]), -0.05, 0.05)


def correction_components(raw, variants, attached):
    output = {}
    row_counts = {fold: len(raw[fold]["y"]) for fold in [2022, 2023, 2024]}
    offsets = {
        2022: 0,
        2023: row_counts[2022],
        2024: row_counts[2022] + row_counts[2023],
    }
    for name, fold_predictions in variants.items():
        concatenated = np.concatenate(
            [fold_predictions[fold] for fold in [2022, 2023, 2024]]
        )
        output[name] = {}
        for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
            success = fit_correction(
                attached["success"], concatenated, SUCCESS_FEATURES,
                10.0, train_year, valid_year,
            )
            reverse = np.mean(
                [
                    fit_correction(
                        attached[str(seed)], concatenated, REVERSE_FEATURES,
                        1000.0, train_year, valid_year,
                    )
                    for seed in REVERSE_SEEDS
                ],
                axis=0,
            )
            output[name][valid_year] = {"success": success, "reverse": reverse}
    return output


def brier_delta(error, correction):
    return float(np.mean(2.0 * error * correction + np.square(correction)))


def tune_corrections(raw, variants, components):
    denominators = {
        fold: float(raw[fold]["y"].mean() * (1.0 - raw[fold]["y"].mean()))
        for fold in [2023, 2024]
    }
    results = []
    corrected = {}
    grid = np.round(np.arange(0.0, 0.801, 0.025), 3)
    for name in variants:
        rows = []
        for success_scale in grid:
            for reverse_scale in grid:
                deltas = {}
                for fold in [2023, 2024]:
                    error = variants[name][fold] - raw[fold]["y"]
                    correction = (
                        success_scale * components[name][fold]["success"]
                        + reverse_scale * components[name][fold]["reverse"]
                    )
                    deltas[fold] = brier_delta(error, correction)
                n23 = deltas[2023] / denominators[2023]
                n24 = deltas[2024] / denominators[2024]
                rows.append({
                    "variant": name,
                    "success_scale": float(success_scale),
                    "reverse_scale": float(reverse_scale),
                    "f23_delta_brier": deltas[2023],
                    "f24_delta_brier": deltas[2024],
                    "both_improve": deltas[2023] < 0 and deltas[2024] < 0,
                    "robust_objective": 0.30 * n23 + 0.70 * n24
                    + 0.50 * max(n23, n24, 0.0),
                })
        best = pd.DataFrame(rows).sort_values("robust_objective").iloc[0].to_dict()
        for fold in [2023, 2024]:
            prediction = np.clip(
                variants[name][fold]
                + best["success_scale"] * components[name][fold]["success"]
                + best["reverse_scale"] * components[name][fold]["reverse"],
                EPS, 1.0 - EPS,
            )
            corrected[(name, fold)] = prediction
            best.update({
                f"f{str(fold)[-2:]}_brier": metrics(raw[fold]["y"], prediction)["brier"],
                f"f{str(fold)[-2:]}_bss": metrics(raw[fold]["y"], prediction)["bss"],
            })
        results.append(best)
    return pd.DataFrame(results).sort_values("robust_objective"), corrected


def simplex_weights(y, matrix, l2=1e-6):
    n_models = matrix.shape[1]
    def objective(weight):
        prediction = matrix @ weight
        return float(np.mean(np.square(prediction - y)) + l2 * np.sum(np.square(weight)))
    result = minimize(
        objective,
        np.repeat(1.0 / n_models, n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda weight: weight.sum() - 1.0}],
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weight = np.clip(result.x, 0.0, 1.0)
    return weight / weight.sum()


def outer_search(raw, corrected, correction_results):
    ensemble = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    base_predictions = {}
    for fold, track in [(2023, "robust"), (2024, "performance")]:
        base_predictions[fold] = (
            ensemble[ensemble["season"].eq(fold) & ensemble["track"].eq(track)]
            .set_index("row_id")["prediction"]
            .reindex(raw[fold]["row_id"])
            .to_numpy("float64")
        )

    # Reproduce submit_013 current insight exactly from the anchor correction components.
    # correction_results already contains tuned anchor; current uses the frozen 0.25/0.55 scales.
    attached = None
    rows = []
    current = {}
    return base_predictions


def main():
    raw = read_raw()
    variants, parameters = build_variants(raw)
    attached = attach_features(raw)
    components = correction_components(raw, variants, attached)
    correction_results, corrected = tune_corrections(raw, variants, components)

    # Current submit_013 insight and a three-way mixture with each large candidate.
    current = {}
    for fold in [2023, 2024]:
        current[fold] = np.clip(
            raw[fold]["anchor"]
            + 0.25 * components["anchor"][fold]["success"]
            + 0.55 * components["anchor"][fold]["reverse"],
            EPS, 1.0 - EPS,
        )
    ensemble = pd.read_parquet(MODEL_DIR / "enhanced_ensemble_oof_predictions.parquet")
    base_predictions = {}
    for fold, track in [(2023, "robust"), (2024, "performance")]:
        base_predictions[fold] = (
            ensemble[ensemble["season"].eq(fold) & ensemble["track"].eq(track)]
            .set_index("row_id")["prediction"]
            .reindex(raw[fold]["row_id"])
            .to_numpy("float64")
        )

    mixture_rows = []
    for name in variants:
        for fold in [2023, 2024]:
            matrix = np.column_stack(
                [base_predictions[fold], current[fold], corrected[(name, fold)]]
            )
            weight = simplex_weights(raw[fold]["y"], matrix)
            prediction = matrix @ weight
            mixture_rows.append({
                "variant": name,
                "fold": fold,
                "base_weight": float(weight[0]),
                "current_weight": float(weight[1]),
                "large_weight": float(weight[2]),
                **metrics(raw[fold]["y"], prediction),
            })
    mixtures = pd.DataFrame(mixture_rows)

    # Apply weights selected on F23 to F24 as an additional temporal stability audit.
    transfer_rows = []
    for name in variants:
        selected = mixtures[
            mixtures["variant"].eq(name) & mixtures["fold"].eq(2023)
        ].iloc[0]
        weight = selected[["base_weight", "current_weight", "large_weight"]].to_numpy(float)
        matrix = np.column_stack(
            [base_predictions[2024], current[2024], corrected[(name, 2024)]]
        )
        transfer_rows.append({
            "variant": name,
            "source_fold": 2023,
            "target_fold": 2024,
            "base_weight": weight[0],
            "current_weight": weight[1],
            "large_weight": weight[2],
            **metrics(raw[2024]["y"], matrix @ weight),
        })
    transfer = pd.DataFrame(transfer_rows)

    REPORTS.mkdir(parents=True, exist_ok=True)
    correction_results.to_csv(REPORTS / "causal_large_corrections.csv", index=False)
    mixtures.to_csv(REPORTS / "causal_large_threeway.csv", index=False)
    transfer.to_csv(REPORTS / "causal_large_transfer.csv", index=False)
    summary = {
        "rule": "Every F23 transform uses 2022 only; every F24 transform uses 2023 only. Correction Ridge is fit on the immediately previous season.",
        "transform_parameters": parameters,
        "correction_results": correction_results.to_dict(orient="records"),
        "threeway_results": mixtures.to_dict(orient="records"),
        "f23_to_f24_transfer": transfer.to_dict(orient="records"),
        "artifacts": {
            "corrections": str(REPORTS / "causal_large_corrections.csv"),
            "threeway": str(REPORTS / "causal_large_threeway.csv"),
            "transfer": str(REPORTS / "causal_large_transfer.csv"),
        },
    }
    (REPORTS / "causal_large_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
