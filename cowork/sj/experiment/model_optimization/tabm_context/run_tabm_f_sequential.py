"""Frozen TabM F expert on 2023 expanding-month validation.

The architecture was frozen before this script was run on Val2024.  This file
selects only a blend scale from the four 2023 month-ahead folds, then reports
the untouched transfer of that scale to the already-produced Val2024 OOF.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tabm
import torch

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
OUTPUT = ROOT / "experiment" / "model_optimization" / "tabm_context" / "outputs" / "f_seq_t0"
REPORT = ROOT / "experiment" / "model_optimization" / "tabm_context" / "reports"
EXPERT = ROOT / "experiment" / "model_optimization" / "game_type_experts"
TARGET = "control_success"
CUTOFFS = [5, 6, 7, 8]
SEED = 20260813


def train_one(
    frame: pd.DataFrame,
    base_features: list[str],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    cutoff: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    seed_everything(SEED + cutoff)
    pre = FoldPreprocessor("t0", base_features).fit(frame.iloc[train_idx])
    xtr_num, xtr_cat = pre.transform(frame.iloc[train_idx])
    xva_num, xva_cat = pre.transform(frame.iloc[valid_idx])
    ytr = frame.iloc[train_idx][TARGET].to_numpy(dtype=np.float32)
    yva = frame.iloc[valid_idx][TARGET].to_numpy(dtype=np.float32)
    xtr_num, xtr_cat, ytr_t = move_dataset(xtr_num, xtr_cat, ytr, device, False)
    xva_num, xva_cat, _ = move_dataset(xva_num, xva_cat, yva, device, False)

    model = tabm.TabM.make(
        n_num_features=xtr_num.shape[1],
        cat_cardinalities=pre.cat_cardinalities,
        d_out=1,
        n_blocks=3,
        d_block=384,
        dropout=0.20,
        k=32,
        arch_type="tabm",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_brier = np.inf
    best_epoch = 0
    best_prediction = None
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, 31):
        model.train()
        permutation = torch.randperm(len(ytr_t), device=device)
        losses = []
        for start in range(0, len(ytr_t), 256):
            idx = permutation[start : start + 256]
            xb_num = take_batch(xtr_num, idx, device)
            xb_cat = take_batch(xtr_cat, idx, device)
            yb = take_batch(ytr_t, idx, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb_num, xb_cat).squeeze(-1)
                loss = training_loss(logits, yb, "brier", 0.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
        prediction, member_std = predict(model, xva_num, xva_cat, 4096, device, True)
        metric = brier_metrics(yva, prediction)
        scheduler.step(metric["brier"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "member_probability_std": member_std,
                **metric,
            }
        )
        print(
            f"cutoff={cutoff} epoch={epoch:02d} bss={metric['bss']:.3f} "
            f"brier={metric['brier']:.8f} mean={metric['prediction_mean']:.5f}/"
            f"{metric['target_mean']:.5f}",
            flush=True,
        )
        if metric["brier"] < best_brier - 1e-8:
            best_brier = metric["brier"]
            best_epoch = epoch
            best_prediction = prediction.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 5 and stale >= 6:
            break
    if best_prediction is None:
        raise RuntimeError("No best prediction")
    checkpoint = {
        "state_dict": best_state,
        "preprocessor": pre.to_dict(),
        "best_epoch": best_epoch,
        "cutoff_month": cutoff,
        "config": {
            "feature_set": "t0",
            "k": 32,
            "d_block": 384,
            "n_blocks": 3,
            "dropout": 0.20,
            "loss": "brier",
            "lr": 0.0015,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, OUTPUT / f"cutoff_{cutoff}.pt")
    return best_prediction, {
        "cutoff_month": cutoff,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "best_epoch": best_epoch,
        "best": brier_metrics(yva, best_prediction),
        "history": history,
    }


def fixed_f_reference_2023(tabm_oof: pd.DataFrame) -> pd.DataFrame:
    seq = pd.read_parquet(EXPERT / "f_sequential_oof.parquet")
    global_prediction = seq.loc[
        seq["model"].eq("global_anchor"),
        ["row_id", "cutoff_month", TARGET, "prediction"],
    ].rename(columns={"prediction": "global_prediction"})
    xgb = seq.loc[
        seq["trial"].eq(31), ["row_id", "cutoff_month", "prediction"]
    ].rename(columns={"prediction": "xgb_prediction"})
    cat = pd.read_parquet(EXPERT / "f_cat_sequential_oof.parquet")[
        ["row_id", "cutoff_month", "prediction"]
    ].rename(columns={"prediction": "cat_prediction"})
    out = (
        global_prediction.merge(xgb, on=["row_id", "cutoff_month"], validate="one_to_one")
        .merge(cat, on=["row_id", "cutoff_month"], validate="one_to_one")
        .merge(tabm_oof, on=["row_id", "cutoff_month", TARGET], validate="one_to_one")
    )
    out["reference_prediction"] = (
        out["global_prediction"]
        + 0.15 * (out["xgb_prediction"] - out["global_prediction"])
        + 0.19 * (out["cat_prediction"] - out["global_prediction"])
    )
    return out


def f_reference_2024() -> pd.DataFrame:
    global_prediction = pd.read_parquet(EXPERT / "expert_oof_predictions.parquet")
    global_prediction = global_prediction.loc[
        global_prediction["season"].eq(2024)
        & global_prediction["game_type"].eq("F")
        & global_prediction["model"].eq("global_anchor"),
        ["row_id", TARGET, "prediction"],
    ].rename(columns={"prediction": "global_prediction"})
    seed = pd.read_parquet(EXPERT / "f_seedbag_oof.parquet")
    xgb = seed.loc[
        seed["family"].eq("xgboost") & seed["seed_index"].eq(0),
        ["row_id", "prediction"],
    ].rename(columns={"prediction": "xgb_prediction"})
    cat = seed.loc[
        seed["family"].eq("catboost") & seed["seed_index"].eq(0),
        ["row_id", "prediction"],
    ].rename(columns={"prediction": "cat_prediction"})
    tabm24 = pd.read_parquet(
        ROOT
        / "experiment/model_optimization/tabm_context/outputs/gate24_f_post23_t0/oof_all.parquet"
    )[["row_id", "prediction"]].rename(columns={"prediction": "tabm_prediction"})
    out = (
        global_prediction.merge(xgb, on="row_id", validate="one_to_one")
        .merge(cat, on="row_id", validate="one_to_one")
        .merge(tabm24, on="row_id", validate="one_to_one")
    )
    out["reference_prediction"] = (
        out["global_prediction"]
        + 0.15 * (out["xgb_prediction"] - out["global_prediction"])
        + 0.19 * (out["cat_prediction"] - out["global_prediction"])
    )
    return out


def select_and_transfer(seq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for weight in np.round(np.arange(0.0, 0.51, 0.05), 2):
        fold_normalized = []
        all_y = []
        all_p = []
        for cutoff, part in seq.groupby("cutoff_month"):
            y = part[TARGET].to_numpy(dtype=np.float64)
            p = (1.0 - weight) * part["reference_prediction"].to_numpy() + weight * part[
                "tabm_prediction"
            ].to_numpy()
            metric = brier_metrics(y, p)
            normalized = metric["brier"] / metric["null_brier"]
            fold_normalized.append(normalized)
            all_y.append(y)
            all_p.append(p)
            rows.append(
                {
                    "scope": "fold",
                    "cutoff_month": int(cutoff),
                    "tabm_weight": float(weight),
                    "normalized_brier": normalized,
                    **metric,
                }
            )
        robust = float(np.mean(fold_normalized) + 0.25 * np.std(fold_normalized))
        pooled = brier_metrics(np.concatenate(all_y), np.concatenate(all_p))
        rows.append(
            {
                "scope": "selection_summary",
                "cutoff_month": -1,
                "tabm_weight": float(weight),
                "normalized_brier": pooled["brier"] / pooled["null_brier"],
                "robust_objective": robust,
                **pooled,
            }
        )
    selection = pd.DataFrame(rows)
    summaries = selection[selection["scope"].eq("selection_summary")].sort_values(
        ["robust_objective", "brier"]
    )
    selected_weight = float(summaries.iloc[0]["tabm_weight"])

    transfer = f_reference_2024()
    y = transfer[TARGET].to_numpy(dtype=np.float64)
    transfer["prediction"] = (
        (1.0 - selected_weight) * transfer["reference_prediction"]
        + selected_weight * transfer["tabm_prediction"]
    )
    reference_metric = brier_metrics(y, transfer["reference_prediction"].to_numpy())
    transfer_metric = brier_metrics(y, transfer["prediction"].to_numpy())
    summary = {
        "selection_source": "2023 F expanding-month folds only",
        "selected_tabm_weight": selected_weight,
        "selection": summaries.to_dict(orient="records"),
        "val2024_reference": reference_metric,
        "val2024_transferred": transfer_metric,
        "val2024_delta_bss": transfer_metric["bss"] - reference_metric["bss"],
        "val2024_was_used_for_weight_selection": False,
    }
    return selection, transfer, summary


def main() -> None:
    started = time.time()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    frame = pd.read_csv(ROOT / "data/train.csv")
    base_features = determine_base_features(frame.columns.tolist())
    game_type = frame["game_type"].astype(str).to_numpy()
    season = frame["season"].to_numpy()
    month = frame["game_month"].to_numpy()
    predictions = []
    folds = []
    for cutoff in CUTOFFS:
        train_idx = np.flatnonzero((season == 2023) & (month <= cutoff) & (game_type == "F"))
        valid_idx = np.flatnonzero((season == 2023) & (month == cutoff + 1) & (game_type == "F"))
        prediction, summary = train_one(
            frame, base_features, train_idx, valid_idx, cutoff, device
        )
        part = frame.iloc[valid_idx][["row_id", TARGET]].copy()
        part["cutoff_month"] = cutoff
        part["tabm_prediction"] = prediction
        predictions.append(part)
        folds.append(summary)
        torch.cuda.empty_cache()
    oof = pd.concat(predictions, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OUTPUT / "oof_all.parquet", index=False)
    seq = fixed_f_reference_2023(oof)
    selection, transfer, summary = select_and_transfer(seq)
    selection.to_csv(REPORT / "f_tabm_sequential_blend.csv", index=False)
    transfer.to_parquet(REPORT / "f_tabm_gate24_transfer.parquet", index=False)
    summary["folds"] = folds
    summary["elapsed_seconds"] = time.time() - started
    (REPORT / "f_tabm_sequential_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
