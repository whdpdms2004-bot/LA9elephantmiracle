"""Evaluate fixed epochs 1..6 for the 20-seed F-TabM bank.

This corrects the weakness of using epoch 6 for every temporal fold.  Epoch
and blend weight are selected on 2023 expanding-month folds only; 2024 is a
single transfer gate using the exact submission residual formula.
"""

from __future__ import annotations

import gc
import json
import time

import numpy as np
import pandas as pd
import tabm
import torch

from run_f_tabm_seedbag20 import (
    BATCH_SIZE,
    CAT_WEIGHT,
    CUTOFFS,
    EPOCHS,
    MODEL_OPT,
    PRED_DIR,
    REPORT_DIR,
    ROOT,
    SEEDS,
    TARGET,
    XGB_WEIGHT,
    load_fold_predictions,
    model_args,
    prepare,
)
from run_tabm_f_sequential import fixed_f_reference_2023
from run_tabm_temporal import (
    brier_metrics,
    determine_base_features,
    predict,
    seed_everything,
    take_batch,
    training_loss,
)


EPOCH_DIR = PRED_DIR / "all_epochs"


def train_all_epochs(prepared: dict, seed: int, device: torch.device) -> tuple[np.ndarray, dict]:
    seed_everything(seed)
    model = tabm.TabM.make(**prepared["model_args"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    predictions = []
    history = []
    started = time.time()
    ytr = prepared["ytr"]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(ytr), device=device)
        losses = []
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
            losses.append(float(loss.detach().item()))
        prediction, member_std = predict(
            model, prepared["xva_num"], prepared["xva_cat"], 4096, device, True
        )
        predictions.append(prediction.astype("float32"))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "member_probability_std": member_std,
                **brier_metrics(prepared["yva"], prediction),
            }
        )
    record = {
        "seed": seed,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(predictions), record


def train_bank(frame: pd.DataFrame, base_features: list[str]) -> None:
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
        )
        for cutoff in CUTOFFS
    ]
    folds.append(
        (
            "gate24",
            np.flatnonzero((season == 2023) & (game_type == "F")),
            np.flatnonzero((season == 2024) & (game_type == "F")),
        )
    )
    for fold_name, train_idx, valid_idx in folds:
        out = EPOCH_DIR / fold_name
        out.mkdir(parents=True, exist_ok=True)
        prepared = prepare(frame, base_features, train_idx, valid_idx, device)
        for seed_index, seed in enumerate(SEEDS):
            npz = out / f"seed_{seed_index:02d}.npz"
            meta = out / f"seed_{seed_index:02d}.json"
            if not (npz.exists() and meta.exists()):
                predictions, record = train_all_epochs(prepared, seed, device)
                np.savez_compressed(npz, predictions=predictions)
                record.update(
                    {
                        "fold": fold_name,
                        "seed_index": seed_index,
                        "train_rows": int(len(train_idx)),
                        "valid_rows": int(len(valid_idx)),
                    }
                )
                meta.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            else:
                record = json.loads(meta.read_text(encoding="utf-8"))
            print(
                f"{fold_name} seed={seed_index + 1:02d}/20 "
                f"e1={record['history'][0]['bss']:.1f} "
                f"e6={record['history'][-1]['bss']:.1f}",
                flush=True,
            )
        del prepared
        gc.collect()
        torch.cuda.empty_cache()


def bag_prediction(fold_name: str, epoch: int, bag_size: int = 20) -> np.ndarray:
    values = []
    for index in range(bag_size):
        with np.load(EPOCH_DIR / fold_name / f"seed_{index:02d}.npz") as data:
            values.append(data["predictions"][epoch - 1])
    return np.mean(values, axis=0).astype("float64")


def selection_grid() -> pd.DataFrame:
    base_parts = []
    for cutoff in CUTOFFS:
        index = pd.read_parquet(PRED_DIR / f"seq_m{cutoff}" / "index.parquet")
        for epoch in range(1, EPOCHS + 1):
            part = index.copy()
            part["cutoff_month"] = cutoff
            part["epoch"] = epoch
            part["tabm_prediction"] = bag_prediction(f"seq_m{cutoff}", epoch)
            base_parts.append(part)
    raw = pd.concat(base_parts, ignore_index=True)
    rows = []
    for epoch in range(1, EPOCHS + 1):
        data = fixed_f_reference_2023(raw.loc[raw["epoch"].eq(epoch)])
        for mode in ["shrink", "additive"]:
            for weight in np.round(np.arange(0.0, 0.501, 0.025), 3):
                normalized, fold_briers = [], []
                all_y, all_p = [], []
                for _, part in data.groupby("cutoff_month"):
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
                    fold_briers.append(metric["brier"])
                    all_y.append(y)
                    all_p.append(prediction)
                pooled = brier_metrics(np.concatenate(all_y), np.concatenate(all_p))
                rows.append(
                    {
                        "epoch": epoch,
                        "mode": mode,
                        "tabm_weight": float(weight),
                        "robust_objective": float(
                            np.mean(normalized) + 0.25 * np.std(normalized)
                        ),
                        "fold_briers": json.dumps(fold_briers),
                        **pooled,
                    }
                )
    grid = pd.DataFrame(rows)
    baseline = grid.loc[grid["tabm_weight"].eq(0)].groupby("mode")["brier"].first()
    grid["pooled_delta_brier_vs_w0"] = [
        row.brier - baseline[row.mode] for row in grid.itertuples()
    ]
    return grid


def gate_rows(choices: pd.DataFrame) -> pd.DataFrame:
    fold = load_fold_predictions()[2024]
    f = fold.loc[fold["game_type"].astype(str).eq("F")].copy()
    expert = pd.read_parquet(MODEL_OPT / "game_type_experts" / "f_seedbag_oof.parquet")
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
    y = f[TARGET].to_numpy(float)
    p015 = f["current_blend"].to_numpy(float)
    anchor = f["adjusted_base"].to_numpy(float)
    dx = f["xgb_prediction"].to_numpy(float) - anchor
    dc = f["cat_prediction"].to_numpy(float) - anchor

    all24 = pd.read_csv(ROOT / "data" / "train.csv", usecols=["season", "game_type", TARGET])
    all24 = all24.loc[all24["season"].eq(2024)]
    r_y = all24.loc[all24["game_type"].astype(str).eq("R"), TARGET].to_numpy(float)
    r_null = float(r_y.mean() * (1 - r_y.mean()))
    r_brier = r_null * (1 - 832.8604276582885 / 100000)
    overall_y = all24[TARGET].to_numpy(float)
    overall_null = float(overall_y.mean() * (1 - overall_y.mean()))

    rows = []
    for choice in choices.itertuples():
        tabm_p = bag_prediction("gate24", int(choice.epoch))
        expert_scale = 1 - choice.tabm_weight if choice.mode == "shrink" else 1.0
        prediction = np.clip(
            p015
            + expert_scale * (XGB_WEIGHT * dx + CAT_WEIGHT * dc)
            + choice.tabm_weight * (tabm_p - anchor),
            1e-6,
            1 - 1e-6,
        )
        metric = brier_metrics(y, prediction)
        whole_brier = (len(r_y) * r_brier + len(y) * metric["brier"]) / len(all24)
        rows.append(
            {
                **choice._asdict(),
                "val2024_f_brier": metric["brier"],
                "val2024_f_bss": metric["bss"],
                "val2024_overall_brier": whole_brier,
                "val2024_overall_bss": 100000 * (1 - whole_brier / overall_null),
                "val2024_tabm_alone_bss": brier_metrics(y, tabm_p)["bss"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.time()
    frame = pd.read_csv(ROOT / "data" / "train.csv")
    base_features = determine_base_features(frame.columns.tolist())
    train_bank(frame, base_features)
    grid = selection_grid()
    selected = (
        grid.sort_values(["robust_objective", "brier"])
        .groupby("mode", as_index=False)
        .first()
    )
    # Include each fixed epoch's 2023-selected weight for diagnostic transfer.
    by_epoch = (
        grid.sort_values(["robust_objective", "brier"])
        .groupby(["epoch", "mode"], as_index=False)
        .first()
    )
    gate = gate_rows(by_epoch)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(REPORT_DIR / "epoch_selection_grid_2023.csv", index=False)
    selected.to_csv(REPORT_DIR / "epoch_selected_2023.csv", index=False)
    gate.to_csv(REPORT_DIR / "epoch_gate_2024.csv", index=False)
    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "seeds": SEEDS,
        "selection": selected.to_dict(orient="records"),
        "gate_for_each_epoch": gate.to_dict(orient="records"),
        "elapsed_seconds": time.time() - started,
    }
    (REPORT_DIR / "epoch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
