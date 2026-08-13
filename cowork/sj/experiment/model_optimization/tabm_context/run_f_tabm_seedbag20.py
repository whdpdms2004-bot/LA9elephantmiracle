"""Build a resumable 20-seed F-TabM validation bank.

Selection is performed on 2023 expanding-month folds.  Val2024 is evaluated
only after the candidate rules have been frozen.  The Val2024 evaluator uses
the same residual formula as the submission script, including the actual
submit_015 F base rather than the older global-anchor approximation.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tabm
import torch

from run_tabm_f_sequential import fixed_f_reference_2023
from run_tabm_temporal import (
    FoldPreprocessor,
    brier_metrics,
    determine_base_features,
    move_dataset,
    predict,
    seed_everything,
    take_batch,
    training_loss,
)


ROOT = Path(__file__).resolve().parents[3]
MODEL_OPT = ROOT / "experiment" / "model_optimization"
WORK = MODEL_OPT / "tabm_context" / "seedbag20"
PRED_DIR = WORK / "predictions"
REPORT_DIR = WORK / "reports"
TARGET = "control_success"
SEEDS = [20260813 + 1009 * index for index in range(20)]
CUTOFFS = [5, 6, 7, 8]
EPOCHS = 6
BATCH_SIZE = 256
XGB_WEIGHT = 0.15
CAT_WEIGHT = 0.19

sys.path.insert(0, str(MODEL_OPT))
sys.path.insert(0, str(MODEL_OPT / "pitcher_cluster_matchup" / "src"))
from analyze_r_focus import load_fold_predictions  # noqa: E402


def model_args(pre: FoldPreprocessor, n_num_features: int) -> dict:
    return {
        "n_num_features": int(n_num_features),
        "cat_cardinalities": pre.cat_cardinalities,
        "d_out": 1,
        "n_blocks": 3,
        "d_block": 384,
        "dropout": 0.20,
        "k": 32,
        "arch_type": "tabm",
    }


def train_fixed(
    prepared: dict,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    seed_everything(seed)
    model = tabm.TabM.make(**prepared["model_args"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    losses = []
    started = time.time()
    ytr = prepared["ytr"]
    for _ in range(EPOCHS):
        model.train()
        permutation = torch.randperm(len(ytr), device=device)
        local_losses = []
        for start in range(0, len(ytr), BATCH_SIZE):
            idx = permutation[start : start + BATCH_SIZE]
            xb_num = take_batch(prepared["xtr_num"], idx, device)
            xb_cat = take_batch(prepared["xtr_cat"], idx, device)
            yb = take_batch(ytr, idx, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb_num, xb_cat).squeeze(-1)
                loss = training_loss(logits, yb, "brier", 0.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            local_losses.append(float(loss.detach().item()))
        losses.append(float(np.mean(local_losses)))
    prediction, member_std = predict(
        model,
        prepared["xva_num"],
        prepared["xva_cat"],
        4096,
        device,
        True,
    )
    record = {
        "seed": seed,
        "epochs": EPOCHS,
        "elapsed_seconds": time.time() - started,
        "epoch_losses": losses,
        "member_probability_std": member_std,
        **brier_metrics(prepared["yva"], prediction),
    }
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return prediction.astype("float32"), record


def prepare(
    frame: pd.DataFrame,
    base_features: list[str],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    device: torch.device,
) -> dict:
    pre = FoldPreprocessor("t0", base_features).fit(frame.iloc[train_idx])
    xtr_num, xtr_cat = pre.transform(frame.iloc[train_idx])
    xva_num, xva_cat = pre.transform(frame.iloc[valid_idx])
    ytr = frame.iloc[train_idx][TARGET].to_numpy(dtype=np.float32)
    yva = frame.iloc[valid_idx][TARGET].to_numpy(dtype=np.float32)
    xtr_num, xtr_cat, ytr_t = move_dataset(
        xtr_num, xtr_cat, ytr, device, False
    )
    xva_num, xva_cat, _ = move_dataset(
        xva_num, xva_cat, yva, device, False
    )
    return {
        "pre": pre,
        "model_args": model_args(pre, xtr_num.shape[1]),
        "xtr_num": xtr_num,
        "xtr_cat": xtr_cat,
        "ytr": ytr_t,
        "xva_num": xva_num,
        "xva_cat": xva_cat,
        "yva": yva,
        "valid_idx": valid_idx,
    }


def train_validation_bank(frame: pd.DataFrame, base_features: list[str]) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    season = frame["season"].to_numpy()
    month = frame["game_month"].to_numpy()
    game_type = frame["game_type"].astype(str).to_numpy()
    folds = [
        (
            f"seq_m{cutoff}",
            np.flatnonzero((season == 2023) & (month <= cutoff) & (game_type == "F")),
            np.flatnonzero((season == 2023) & (month == cutoff + 1) & (game_type == "F")),
            cutoff,
        )
        for cutoff in CUTOFFS
    ]
    folds.append(
        (
            "gate24",
            np.flatnonzero((season == 2023) & (game_type == "F")),
            np.flatnonzero((season == 2024) & (game_type == "F")),
            None,
        )
    )
    metric_rows = []
    for fold_name, train_idx, valid_idx, cutoff in folds:
        fold_dir = PRED_DIR / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        index_file = fold_dir / "index.parquet"
        if not index_file.exists():
            index = frame.iloc[valid_idx][["row_id", TARGET]].copy()
            if cutoff is not None:
                index["cutoff_month"] = cutoff
            index.to_parquet(index_file, index=False)
        prepared = prepare(frame, base_features, train_idx, valid_idx, device)
        for seed_index, seed in enumerate(SEEDS):
            pred_file = fold_dir / f"seed_{seed_index:02d}.npy"
            metric_file = fold_dir / f"seed_{seed_index:02d}.json"
            if pred_file.exists() and metric_file.exists():
                record = json.loads(metric_file.read_text(encoding="utf-8"))
            else:
                prediction, record = train_fixed(prepared, seed, device)
                np.save(pred_file, prediction)
                record.update(
                    {
                        "fold": fold_name,
                        "seed_index": seed_index,
                        "train_rows": int(len(train_idx)),
                        "valid_rows": int(len(valid_idx)),
                    }
                )
                metric_file.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            metric_rows.append(record)
            print(
                f"{fold_name} seed={seed_index + 1:02d}/20 "
                f"bss={record['bss']:.3f} sec={record['elapsed_seconds']:.1f}",
                flush=True,
            )
        del prepared
        gc.collect()
        torch.cuda.empty_cache()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(REPORT_DIR / "individual_metrics.csv", index=False)


def load_bag(fold_name: str, size: int) -> np.ndarray:
    return np.mean(
        [np.load(PRED_DIR / fold_name / f"seed_{index:02d}.npy") for index in range(size)],
        axis=0,
    ).astype("float64")


def selection_2023() -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    for cutoff in CUTOFFS:
        index = pd.read_parquet(PRED_DIR / f"seq_m{cutoff}" / "index.parquet")
        for bag_size in range(1, 21):
            part = index.copy()
            part["cutoff_month"] = cutoff
            part["bag_size"] = bag_size
            part["tabm_prediction"] = load_bag(f"seq_m{cutoff}", bag_size)
            pieces.append(part)
    tabm = pd.concat(pieces, ignore_index=True)
    reference_parts = []
    for bag_size in range(1, 21):
        local = fixed_f_reference_2023(tabm.loc[tabm["bag_size"].eq(bag_size)])
        local["bag_size"] = bag_size
        reference_parts.append(local)
    data = pd.concat(reference_parts, ignore_index=True)
    rows = []
    for bag_size in range(1, 21):
        bag = data.loc[data["bag_size"].eq(bag_size)]
        for mode in ["shrink", "additive"]:
            for weight in np.round(np.arange(0.0, 0.501, 0.025), 3):
                normalized = []
                all_y, all_p = [], []
                for cutoff, part in bag.groupby("cutoff_month"):
                    y = part[TARGET].to_numpy("float64")
                    global_p = part["global_prediction"].to_numpy("float64")
                    dx = part["xgb_prediction"].to_numpy("float64") - global_p
                    dc = part["cat_prediction"].to_numpy("float64") - global_p
                    dt = part["tabm_prediction"].to_numpy("float64") - global_p
                    expert_scale = 1.0 - weight if mode == "shrink" else 1.0
                    prediction = np.clip(
                        global_p
                        + expert_scale * (XGB_WEIGHT * dx + CAT_WEIGHT * dc)
                        + weight * dt,
                        1e-6,
                        1.0 - 1e-6,
                    )
                    metric = brier_metrics(y, prediction)
                    normalized.append(metric["brier"] / metric["null_brier"])
                    all_y.append(y)
                    all_p.append(prediction)
                pooled = brier_metrics(np.concatenate(all_y), np.concatenate(all_p))
                rows.append(
                    {
                        "bag_size": bag_size,
                        "mode": mode,
                        "tabm_weight": float(weight),
                        "robust_objective": float(
                            np.mean(normalized) + 0.25 * np.std(normalized)
                        ),
                        "folds_improved_vs_w0": 0,
                        **pooled,
                    }
                )
    grid = pd.DataFrame(rows)
    for (bag_size, mode), group in grid.groupby(["bag_size", "mode"]):
        baseline_brier = float(group.loc[group["tabm_weight"].eq(0), "brier"].iloc[0])
        grid.loc[group.index, "pooled_delta_brier_vs_w0"] = group["brier"] - baseline_brier
    selected = (
        grid.sort_values(["robust_objective", "brier"])
        .groupby(["bag_size", "mode"], as_index=False)
        .first()
    )
    return grid, selected


def gate_2024(selected: pd.DataFrame) -> pd.DataFrame:
    folds = load_fold_predictions()
    fold = folds[2024]
    is_f = fold["game_type"].astype(str).eq("F")
    f = fold.loc[is_f].copy()
    index = pd.read_parquet(PRED_DIR / "gate24" / "index.parquet")
    if not index["row_id"].equals(f["row_id"].reset_index(drop=True)):
        f = index[["row_id", TARGET]].merge(
            f.drop(columns=[TARGET]), on="row_id", validate="one_to_one"
        )
    expert_path = MODEL_OPT / "game_type_experts" / "f_seedbag_oof.parquet"
    expert = pd.read_parquet(expert_path)
    xgb = expert.loc[
        expert["family"].eq("xgboost") & expert["seed_index"].eq(0),
        ["row_id", "prediction"],
    ].rename(columns={"prediction": "xgb_prediction"})
    cat = expert.loc[
        expert["family"].eq("catboost") & expert["seed_index"].eq(0),
        ["row_id", "prediction"],
    ].rename(columns={"prediction": "cat_prediction"})
    f = f.merge(xgb, on="row_id", validate="one_to_one").merge(
        cat, on="row_id", validate="one_to_one"
    )
    y = f[TARGET].to_numpy("float64")
    p015 = f["current_blend"].to_numpy("float64")
    anchor = f["adjusted_base"].to_numpy("float64")
    dx = f["xgb_prediction"].to_numpy("float64") - anchor
    dc = f["cat_prediction"].to_numpy("float64") - anchor

    all_2024 = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=["season", "game_type", TARGET],
    )
    all_2024 = all_2024.loc[all_2024["season"].eq(2024)]
    r_y = all_2024.loc[all_2024["game_type"].astype(str).eq("R"), TARGET].to_numpy(float)
    r_null = float(r_y.mean() * (1.0 - r_y.mean()))
    r_brier = r_null * (1.0 - 832.8604276582885 / 100000.0)
    overall_y = all_2024[TARGET].to_numpy(float)
    overall_null = float(overall_y.mean() * (1.0 - overall_y.mean()))
    n_r = len(r_y)
    n_f = len(y)

    rows = []
    for _, choice in selected.iterrows():
        bag_size = int(choice["bag_size"])
        weight = float(choice["tabm_weight"])
        mode = str(choice["mode"])
        tabm_p = load_bag("gate24", bag_size)
        dt = tabm_p - anchor
        expert_scale = 1.0 - weight if mode == "shrink" else 1.0
        prediction = np.clip(
            p015
            + expert_scale * (XGB_WEIGHT * dx + CAT_WEIGHT * dc)
            + weight * dt,
            1e-6,
            1.0 - 1e-6,
        )
        metric = brier_metrics(y, prediction)
        whole_brier = (n_r * r_brier + n_f * metric["brier"]) / (n_r + n_f)
        whole_bss = 100000.0 * (1.0 - whole_brier / overall_null)
        rows.append(
            {
                **choice.to_dict(),
                "val2024_f_brier": metric["brier"],
                "val2024_f_bss": metric["bss"],
                "val2024_overall_brier": whole_brier,
                "val2024_overall_bss": whole_bss,
                "val2024_f_prediction_mean": metric["prediction_mean"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.time()
    frame = pd.read_csv(ROOT / "data" / "train.csv")
    base_features = determine_base_features(frame.columns.tolist())
    train_validation_bank(frame, base_features)
    grid, selected = selection_2023()
    gate = gate_2024(selected)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(REPORT_DIR / "selection_grid_2023.csv", index=False)
    selected.to_csv(REPORT_DIR / "selected_by_bag_mode_2023.csv", index=False)
    gate.to_csv(REPORT_DIR / "gate_2024.csv", index=False)
    full = gate.loc[gate["bag_size"].eq(20)].sort_values(
        "val2024_overall_brier"
    )
    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "selection_rule": "2023 expanding-month mean normalized Brier + 0.25*std",
        "gate_rule": "Val2024 evaluated once after 2023 selection",
        "actual_submission_formula": True,
        "full_bag_candidates": full.to_dict(orient="records"),
        "elapsed_seconds": time.time() - started,
    }
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
