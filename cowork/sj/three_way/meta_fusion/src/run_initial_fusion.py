"""초기 probability-union fusion 실험.

세 실패 사건 M/R/O 중 하나라도 발생하면 제구 실패다. O는 M/R과 배타적이고
M과 R만 겹치므로, 세 주변확률에서 빠진 P(M AND R)를 시간순 OOF로 추정한다.
최종 성공확률은 언제나 포함-배제 합집합의 여집합으로 계산한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
META = HERE.parent
TW = META.parent
SRC = TW / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from harness3 import OUT, SUCCESS, bss, load_labeled


EPS = 1e-7
COMPONENT_TAGS = {
    "middle": "middle__id_frequency+no_trackman+temporal_cyclic__{fold}.npy",
    "reverse": "reverse__count_multiscale+drop_ids+trackman_quality__{fold}.npy",
    "outside": "outside__drop_ids+no_trackman+rate_multiscale__{fold}.npy",
    "mr": "mr__id_frequency+no_trackman+temporal_cyclic__{fold}.npy",
}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def raw_features(pm: np.ndarray, pr: np.ndarray, po: np.ndarray) -> np.ndarray:
    return np.column_stack([
        logit(pm),
        logit(pr),
        logit(po),
        pm * pr,
        np.minimum(pm, pr),
        np.abs(pm - pr),
        pm + pr + po,
    ])


def bounds(pm: np.ndarray, pr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(0.0, pm + pr - 1.0), np.minimum(pm, pr)


def predict_overlap(model: dict, pm: np.ndarray, pr: np.ndarray,
                    po: np.ndarray) -> np.ndarray:
    x = raw_features(pm, pr, po)
    mean = np.asarray(model["feature_mean"], np.float64)
    scale = np.asarray(model["feature_scale"], np.float64)
    beta = np.asarray(model["beta"], np.float64)
    xs = (x - mean) / scale
    design = np.column_stack([np.ones(len(xs)), xs])
    low, high = bounds(pm, pr)
    return low + (high - low) * sigmoid(design @ beta)


def fit_overlap(pm: np.ndarray, pr: np.ndarray, po: np.ndarray,
                y_mr: np.ndarray, l2: float) -> dict:
    x = raw_features(pm, pr, po)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    xs = (x - mean) / scale
    design = np.column_stack([np.ones(len(xs)), xs])
    low, high = bounds(pm, pr)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        s = sigmoid(design @ beta)
        pred = low + (high - low) * s
        error = pred - y_mr
        loss = float(np.mean(error ** 2) + l2 * np.sum(beta[1:] ** 2))
        grad_factor = 2.0 * error * (high - low) * s * (1.0 - s)
        grad = design.T @ grad_factor / len(design)
        grad[1:] += 2.0 * l2 * beta[1:]
        return loss, grad

    initial_rate = float(np.clip(y_mr.mean(), 1e-4, 1.0 - 1e-4))
    initial = np.zeros(design.shape[1], np.float64)
    initial[0] = np.log(initial_rate / (1.0 - initial_rate))
    result = minimize(
        objective, initial, method="L-BFGS-B", jac=True,
        options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"overlap optimization failed: {result.message}")
    return {
        "kind": "bounded_mr_brier",
        "feature_names": [
            "logit_pm", "logit_pr", "logit_po", "pm_x_pr",
            "min_pm_pr", "abs_pm_minus_pr", "pm_plus_pr_plus_po",
        ],
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "beta": result.x.tolist(),
        "l2": float(l2),
        "fit_objective": float(result.fun),
        "fit_rows": int(len(y_mr)),
    }


def fit_event_calibrator(p: np.ndarray, y: np.ndarray, l2: float) -> dict:
    """Identity-anchored logit calibration for one event probability."""
    z = logit(p)
    design = np.column_stack([np.ones(len(z)), z])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        pred = sigmoid(design @ beta)
        error = pred - y
        loss = float(
            np.mean(error ** 2)
            + l2 * (beta[0] ** 2 + (beta[1] - 1.0) ** 2))
        grad_factor = 2.0 * error * pred * (1.0 - pred)
        grad = design.T @ grad_factor / len(design)
        grad += 2.0 * l2 * np.array([beta[0], beta[1] - 1.0])
        return loss, grad

    result = minimize(
        objective, np.array([0.0, 1.0]), method="L-BFGS-B", jac=True,
        options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"event calibration failed: {result.message}")
    return {"beta": result.x.tolist(), "l2": float(l2),
            "fit_objective": float(result.fun), "fit_rows": int(len(y))}


def apply_event_calibrator(model: dict, p: np.ndarray,
                           strength: float) -> np.ndarray:
    beta = np.asarray(model["beta"], np.float64)
    original = logit(p)
    calibrated = beta[0] + beta[1] * original
    return sigmoid((1.0 - strength) * original + strength * calibrated)


def success_from_union(pm: np.ndarray, pr: np.ndarray, po: np.ndarray,
                       pmr: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - (pm + pr + po - pmr), EPS, 1.0 - EPS)


def simplex_success(pm: np.ndarray, pr: np.ndarray, po: np.ndarray,
                    pmr: np.ndarray) -> np.ndarray:
    q = np.column_stack([
        1.0 - (pm + pr + po - pmr),
        pm - pmr,
        pr - pmr,
        pmr,
        po,
    ])
    # Euclidean projection onto the probability simplex, independently per row.
    ordered = np.sort(q, axis=1)[:, ::-1]
    cssv = np.cumsum(ordered, axis=1) - 1.0
    ind = np.arange(1, q.shape[1] + 1, dtype=np.float64)
    cond = ordered - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(len(q)), rho] / (rho + 1.0)
    projected = np.maximum(q - theta[:, None], 0.0)
    return np.clip(projected[:, 0], EPS, 1.0 - EPS)


def load_fold(frame: pd.DataFrame, fold: int) -> dict:
    mask = ((frame["season"].to_numpy() == fold)
            & (frame["label_ok"].to_numpy() == 1))
    data = {
        key: np.load(OUT / pattern.format(fold=fold)).astype(np.float64)
        for key, pattern in COMPONENT_TAGS.items()
    }
    expected = int(mask.sum())
    if any(len(values) != expected for values in data.values()):
        raise RuntimeError(f"fold {fold} prediction length mismatch")
    data.update({
        "y_success": frame.loc[mask, SUCCESS].to_numpy(np.float64),
        "y_mr": frame.loc[mask, "y_mr"].to_numpy(np.float64),
        "game_type": frame.loc[mask, "game_type"].astype(str).to_numpy(),
        "game_month": frame.loc[mask, "game_month"].to_numpy(),
    })
    return data


def evaluate(data: dict, prediction: np.ndarray) -> dict:
    metrics = bss(data["y_success"], prediction)
    metrics["invalid_union_rows"] = int(
        ((prediction <= EPS) | (prediction >= 1.0 - EPS)).sum())
    return metrics


def main() -> None:
    output_dir = META / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_labeled()
    folds = {fold: load_fold(frame, fold) for fold in (2023, 2024)}
    fit_data, eval_data = folds[2023], folds[2024]

    rows = []
    baseline_by_fold = {}
    for fold, data in folds.items():
        raw = success_from_union(
            data["middle"], data["reverse"], data["outside"], data["mr"])
        structural_mr = np.clip(
            data["mr"], *bounds(data["middle"], data["reverse"]))
        structural = success_from_union(
            data["middle"], data["reverse"], data["outside"], structural_mr)
        projected = simplex_success(
            data["middle"], data["reverse"], data["outside"], data["mr"])
        baseline_by_fold[fold] = evaluate(data, raw)
        for name, prediction in (
                ("identity_raw", raw),
                ("identity_bounded_mr", structural),
                ("simplex_projection", projected)):
            rows.append({"fit_fold": None, "eval_fold": fold,
                         "method": name, "l2": None, "blend": None,
                         **evaluate(data, prediction)})

    candidates = []
    for l2 in (1e-4, 1e-3, 1e-2):
        model = fit_overlap(
            fit_data["middle"], fit_data["reverse"], fit_data["outside"],
            fit_data["y_mr"], l2)
        learned = predict_overlap(
            model, eval_data["middle"], eval_data["reverse"],
            eval_data["outside"])
        current = np.clip(
            eval_data["mr"],
            *bounds(eval_data["middle"], eval_data["reverse"]))
        for blend in (0.25, 0.50, 0.75, 1.0):
            overlap = (1.0 - blend) * current + blend * learned
            prediction = success_from_union(
                eval_data["middle"], eval_data["reverse"],
                eval_data["outside"], overlap)
            metrics = evaluate(eval_data, prediction)
            row = {"fit_fold": 2023, "eval_fold": 2024,
                   "method": "bounded_overlap_blend", "l2": l2,
                   "blend": blend, **metrics}
            rows.append(row)
            candidates.append((metrics["brier"], l2, blend, model, metrics))

    calibration_candidates = []
    target_names = {
        "middle": "y_middle", "reverse": "y_reverse",
        "outside": "y_outside", "mr": "y_mr",
    }
    fit_mask = ((frame["season"].to_numpy() == 2023)
                & (frame["label_ok"].to_numpy() == 1))
    eval_mask = ((frame["season"].to_numpy() == 2024)
                 & (frame["label_ok"].to_numpy() == 1))
    fit_targets = {
        key: frame.loc[fit_mask, column].to_numpy(np.float64)
        for key, column in target_names.items()
    }
    # A small, predeclared regularization grid. Selection uses only strict
    # 2023 -> 2024 forward transfer and the final recipe is then refit.
    for l2 in (1e-4, 1e-3, 1e-2, 1e-1):
        calibrators = {
            key: fit_event_calibrator(fit_data[key], fit_targets[key], l2)
            for key in target_names
        }
        for strength in (0.25, 0.50, 0.75, 1.0):
            calibrated = {
                key: apply_event_calibrator(calibrators[key], eval_data[key], strength)
                for key in target_names
            }
            calibrated["mr"] = np.clip(
                calibrated["mr"],
                *bounds(calibrated["middle"], calibrated["reverse"]))
            prediction = success_from_union(
                calibrated["middle"], calibrated["reverse"],
                calibrated["outside"], calibrated["mr"])
            metrics = evaluate(eval_data, prediction)
            row = {"fit_fold": 2023, "eval_fold": 2024,
                   "method": "event_logit_calibration", "l2": l2,
                   "blend": strength, **metrics}
            rows.append(row)
            calibration_candidates.append(
                (metrics["brier"], l2, strength, calibrators, metrics))

    candidates.sort(key=lambda item: item[0])
    calibration_candidates.sort(key=lambda item: item[0])
    _, selected_l2, selected_blend, _, selected_metrics = candidates[0]
    (cal_brier, cal_l2, cal_strength, _, cal_metrics) = calibration_candidates[0]

    # Selection is made on strict 2023 -> 2024 transfer. Refit the same fixed
    # recipe on both available OOF seasons for the 2025 submission asset.
    merged = {
        key: np.concatenate([folds[2023][key], folds[2024][key]])
        for key in ("middle", "reverse", "outside", "mr", "y_mr")
    }
    final_model = fit_overlap(
        merged["middle"], merged["reverse"], merged["outside"],
        merged["y_mr"], selected_l2)
    final_model.update({
        "blend_with_component_mr": float(selected_blend),
        "selection_protocol": "fit overlap on OOF2023, select on OOF2024 raw Brier",
        "refit_protocol": "same fixed recipe refit on OOF2023+OOF2024",
        "training_seasons": [2023, 2024],
        "component_probability_source": "strict forward OOF teachers",
        "union_formula": "1 - (p_middle + p_reverse + p_outside - p_mr_fused)",
        "leaderboard_used_for_fit": False,
        "selected_eval_metrics": selected_metrics,
        "identity_eval_metrics": baseline_by_fold[2024],
    })

    merged_targets = {
        key: frame.loc[
            ((frame["season"].isin([2023, 2024]))
             & (frame["label_ok"] == 1)), column].to_numpy(np.float64)
        for key, column in target_names.items()
    }
    final_calibrators = {
        key: fit_event_calibrator(merged[key], merged_targets[key], cal_l2)
        for key in target_names
    }
    calibration_model = {
        "kind": "event_logit_calibration_then_probability_union",
        "calibrators": final_calibrators,
        "strength": float(cal_strength),
        "selection_protocol": (
            "fit event calibrators on strict OOF2023 and select on OOF2024 raw Brier"),
        "refit_protocol": "same fixed recipe refit on strict OOF2023+OOF2024",
        "training_seasons": [2023, 2024],
        "component_probability_source": "strict forward OOF teachers",
        "union_formula": "1 - (p_middle + p_reverse + p_outside - p_mr)",
        "mr_constraint": "Frechet bounds after event calibration",
        "leaderboard_used_for_fit": False,
        "selected_eval_metrics": cal_metrics,
        "identity_eval_metrics": baseline_by_fold[2024],
    }

    table = pd.DataFrame(rows).sort_values(
        ["eval_fold", "brier", "method"], ascending=[True, True, True])
    table.to_csv(output_dir / "initial_fusion_results.csv", index=False)
    (output_dir / "initial_overlap_model.json").write_text(
        json.dumps(final_model, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    (output_dir / "initial_event_calibration_model.json").write_text(
        json.dumps(calibration_model, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    report = {
        "selected_l2": selected_l2,
        "selected_blend": selected_blend,
        "selected_eval_2024": selected_metrics,
        "identity_eval_2024": baseline_by_fold[2024],
        "delta_bss_2024": (
            selected_metrics["bss_raw"] - baseline_by_fold[2024]["bss_raw"]),
        "event_calibration": {
            "selected_l2": cal_l2,
            "selected_strength": cal_strength,
            "selected_eval_2024": cal_metrics,
            "delta_bss_2024": (
                cal_metrics["bss_raw"] - baseline_by_fold[2024]["bss_raw"]),
        },
        "top_results": json.loads(
            table.head(12).to_json(orient="records")),
    }
    (output_dir / "initial_fusion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
