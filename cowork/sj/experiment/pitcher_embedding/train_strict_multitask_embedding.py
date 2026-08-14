from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "trackman500_multitask"
TARGET = "control_success"
SEED = 2026
HISTORY_COLUMNS = [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
LABEL_COLUMNS = [TARGET, "reverse", "middle", "outside_only"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", type=int, choices=[16, 32, 64], required=True)
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    parser.add_argument(
        "--main-loss",
        choices=["brier", "bce", "soft_f1", "macro_soft_f1", "brier_macro_f1"],
        default="brier",
    )
    parser.add_argument(
        "--f1-weight",
        type=float,
        default=0.10,
        help="Weight of macro soft-F1 regularization for brier_macro_f1.",
    )
    parser.add_argument("--half-life", type=float, default=1.0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def fit_scaler(values: np.ndarray):
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(values), values, median)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale > 1e-7, scale, 1.0)
    return median.astype("float32"), mean.astype("float32"), scale.astype("float32")


def transform(values: np.ndarray, scaler):
    median, mean, scale = scaler
    values = np.where(np.isfinite(values), values, median)
    return ((values - mean) / scale).astype("float32")


def cohort_index(asof_n: np.ndarray):
    return np.select(
        [asof_n <= 0, asof_n <= 100, asof_n <= 500, asof_n <= 2000],
        [0, 1, 2, 3],
        default=4,
    ).astype("int64")


class PitcherMultiTaskNet(nn.Module):
    def __init__(self, history_dim, trackman_dim, n_pitchers, embedding_dim):
        super().__init__()
        hidden = max(64, embedding_dim * 2)
        self.pitcher_embedding = nn.Embedding(n_pitchers + 1, embedding_dim, padding_idx=0)
        self.cohort_embedding = nn.Embedding(5, embedding_dim)
        self.history_tower = nn.Sequential(
            nn.Linear(history_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, embedding_dim),
            nn.SiLU(),
        )
        self.trackman_tower = nn.Sequential(
            nn.Linear(trackman_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, embedding_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 1, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, embedding_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(embedding_dim, 4)

    def representation(self, history, trackman, pitcher, cohort, asof_n, tm_available):
        history_h = self.history_tower(history)
        trackman_h = self.trackman_tower(trackman) * tm_available.unsqueeze(1)
        individual = self.pitcher_embedding(pitcher)
        fallback = self.cohort_embedding(cohort)
        reliability = (asof_n / (asof_n + 500.0)).clamp(0.0, 1.0).unsqueeze(1)
        reliability = reliability * pitcher.ne(0).float().unsqueeze(1)
        pitcher_h = reliability * individual + (1.0 - reliability) * fallback
        return self.fusion(
            torch.cat([history_h, trackman_h, pitcher_h, tm_available.unsqueeze(1)], dim=1)
        )

    def forward(self, history, trackman, pitcher, cohort, asof_n, tm_available):
        representation = self.representation(
            history, trackman, pitcher, cohort, asof_n, tm_available
        )
        return self.head(representation), representation


def brier_metrics(target: np.ndarray, probability: np.ndarray):
    target = target.astype("float64")
    probability = np.clip(probability.astype("float64"), 1e-7, 1 - 1e-7)
    brier = float(np.mean((probability - target) ** 2))
    rate = float(target.mean())
    ratio = brier / (rate * (1 - rate))
    prediction = probability >= 0.5
    threshold_grid = np.linspace(0.20, 0.80, 121)
    threshold_scores = np.asarray(
        [f1_score(target, probability >= threshold, zero_division=0) for threshold in threshold_grid]
    )
    best_index = int(np.argmax(threshold_scores))
    return {
        "brier": brier,
        "normalized_brier": ratio,
        "bss": max(0.0, 100000 * (1 - ratio)),
        "logloss": float(log_loss(target, probability, labels=[0, 1])),
        "auc": float(roc_auc_score(target, probability)),
        "target_mean": rate,
        "pred_mean": float(probability.mean()),
        "mean_gap": float(probability.mean() - rate),
        "f1_at_050": float(f1_score(target, prediction, zero_division=0)),
        "precision_at_050": float(precision_score(target, prediction, zero_division=0)),
        "recall_at_050": float(recall_score(target, prediction, zero_division=0)),
        "best_f1_diagnostic": float(threshold_scores[best_index]),
        "best_f1_threshold_diagnostic": float(threshold_grid[best_index]),
    }


def weighted_mean(value: torch.Tensor, weight: torch.Tensor):
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def soft_f1_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    macro: bool,
):
    eps = 1e-7
    true_positive = (weight * probability * target).sum()
    false_positive = (weight * probability * (1.0 - target)).sum()
    false_negative = (weight * (1.0 - probability) * target).sum()
    positive_f1 = (2.0 * true_positive + eps) / (
        2.0 * true_positive + false_positive + false_negative + eps
    )
    if not macro:
        return 1.0 - positive_f1
    negative_probability = 1.0 - probability
    negative_target = 1.0 - target
    true_negative = (weight * negative_probability * negative_target).sum()
    negative_false_positive = (weight * negative_probability * target).sum()
    negative_false_negative = (weight * probability * negative_target).sum()
    negative_f1 = (2.0 * true_negative + eps) / (
        2.0 * true_negative
        + negative_false_positive
        + negative_false_negative
        + eps
    )
    return 1.0 - 0.5 * (positive_f1 + negative_f1)


def main_objective(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    name: str,
    f1_weight: float,
):
    probability = torch.sigmoid(logits)
    brier = weighted_mean((probability - target).square(), weight)
    if name == "brier":
        return brier
    if name == "bce":
        return weighted_mean(
            nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none"),
            weight,
        )
    if name == "soft_f1":
        return soft_f1_loss(probability, target, weight, macro=False)
    macro_f1 = soft_f1_loss(probability, target, weight, macro=True)
    if name == "macro_soft_f1":
        return macro_f1
    if name == "brier_macro_f1":
        return brier + f1_weight * macro_f1
    raise ValueError(f"Unknown main loss: {name}")


def artifact_stem(args):
    base = f"multitask_dim{args.embedding_dim}"
    if args.main_loss == "brier":
        return base
    if args.main_loss == "brier_macro_f1":
        weight = f"{args.f1_weight:g}".replace(".", "p")
        return f"{base}_{args.main_loss}_w{weight}"
    return f"{base}_{args.main_loss}"


def load_data():
    main = pd.read_csv(ROOT / "data" / "train.csv")
    labels = pd.read_parquet(MODEL_DIR / "failure_component_labels.parquet")
    tm = pd.read_parquet(MODEL_DIR / "trackman500_asof_train.parquet")
    if not main["row_id"].equals(labels["row_id"]) or not main["row_id"].equals(tm["row_id"]):
        raise RuntimeError("Strict embedding inputs are not row-aligned")
    tm_columns = [column for column in tm if column not in {"row_id", "season"}]
    frame = pd.concat(
        [
            main,
            labels[["reverse", "middle", "outside_only"]],
            tm[tm_columns],
        ],
        axis=1,
    )
    if frame[tm_columns].drop(columns=["tm500_available", "tm500_unavailable"]).empty:
        raise RuntimeError("Trackman columns are missing")
    return frame, tm_columns


def make_tensors(frame, indices, history_scaler, tm_scaler, tm_columns, pitcher_map):
    part = frame.loc[indices]
    history = transform(part[HISTORY_COLUMNS].to_numpy("float32"), history_scaler)
    trackman = transform(part[tm_columns].to_numpy("float32"), tm_scaler)
    pitcher = part["pitcher_id"].map(pitcher_map).fillna(0).to_numpy("int64")
    asof_n = part["asof_pitcher_n"].fillna(0).to_numpy("float32")
    cohort = cohort_index(asof_n)
    available = part["tm500_available"].fillna(0).to_numpy("float32")
    return history, trackman, pitcher, cohort, asof_n, available


def predict(model, tensors, batch_size, device):
    dataset = TensorDataset(*(torch.from_numpy(value) for value in tensors))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    representations = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = [value.to(device) for value in batch]
            logits, representation = model(*batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            representations.append(representation.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(representations)


def run_fold(frame, tm_columns, fold, args, device):
    started = time.time()
    train_indices = frame.index[
        frame["season"].lt(fold) & frame[LABEL_COLUMNS].notna().all(axis=1)
    ].to_numpy()
    if args.max_train_rows and len(train_indices) > args.max_train_rows:
        rng = np.random.default_rng(SEED + fold + args.embedding_dim)
        train_indices = np.sort(
            rng.choice(train_indices, args.max_train_rows, replace=False)
        )
    valid_indices = frame.index[frame["season"].eq(fold)].to_numpy()
    history_scaler = fit_scaler(frame.loc[train_indices, HISTORY_COLUMNS].to_numpy("float32"))
    tm_scaler = fit_scaler(frame.loc[train_indices, tm_columns].to_numpy("float32"))
    known_pitchers = np.sort(frame.loc[frame["season"].lt(fold), "pitcher_id"].unique())
    pitcher_map = {int(value): index + 1 for index, value in enumerate(known_pitchers)}
    tensors = make_tensors(
        frame, train_indices, history_scaler, tm_scaler, tm_columns, pitcher_map
    )
    labels = frame.loc[train_indices, LABEL_COLUMNS].to_numpy("float32")
    age = fold - frame.loc[train_indices, "season"].to_numpy("float32")
    weights = np.power(0.5, age / args.half_life).astype("float32")
    weights /= weights.mean()
    dataset = TensorDataset(
        *(torch.from_numpy(value) for value in tensors),
        torch.from_numpy(labels),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(SEED + fold + args.embedding_dim)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    torch.manual_seed(SEED + fold + args.embedding_dim)
    model = PitcherMultiTaskNet(
        len(HISTORY_COLUMNS), len(tm_columns), len(pitcher_map), args.embedding_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=2e-4
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    epoch_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = main_sum = brier_sum = seen = 0.0
        for batch in loader:
            *inputs, target, weight = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*inputs)
            probability = torch.sigmoid(logits[:, 0])
            brier = (probability - target[:, 0]).square()
            auxiliary = bce(logits[:, 1:], target[:, 1:]).mean(dim=1)
            main_loss = main_objective(
                logits[:, 0], target[:, 0], weight, args.main_loss, args.f1_weight
            )
            auxiliary_loss = weighted_mean(auxiliary, weight)
            loss = main_loss + args.aux_weight * auxiliary_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)
            main_sum += float(main_loss.item()) * len(target)
            brier_sum += float(weighted_mean(brier, weight).item()) * len(target)
            seen += len(target)
        epoch_rows.append(
            {
                "fold": fold,
                "epoch": epoch,
                "loss": loss_sum / seen,
                "main_loss": main_sum / seen,
                "weighted_brier": brier_sum / seen,
                "main_loss_name": args.main_loss,
            }
        )
        print(json.dumps(epoch_rows[-1]), flush=True)

    valid_tensors = make_tensors(
        frame, valid_indices, history_scaler, tm_scaler, tm_columns, pitcher_map
    )
    valid_probability, _ = predict(model, valid_tensors, args.batch_size, device)
    metrics = brier_metrics(frame.loc[valid_indices, TARGET].to_numpy(), valid_probability[:, 0])
    metrics.update(
        {
            "fold": fold,
            "train_through": fold - 1,
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "train_rows": len(train_indices),
            "valid_rows": len(valid_indices),
            "trackman": True,
            "trackman_cutoff": fold,
            "min_trackman_season_pitches": 500,
            "elapsed_sec": time.time() - started,
            "main_loss": args.main_loss,
            "f1_weight": args.f1_weight if args.main_loss == "brier_macro_f1" else 0.0,
        }
    )

    first = (
        frame.loc[valid_indices]
        .sort_values(["pitcher_id", "asof_pitcher_n", "row_id"])
        .groupby("pitcher_id", sort=False)
        .head(1)
    )
    start_tensors = make_tensors(
        frame, first.index.to_numpy(), history_scaler, tm_scaler, tm_columns, pitcher_map
    )
    _, representation = predict(model, start_tensors, args.batch_size, device)
    embedding = first[["pitcher_id", "season", "asof_pitcher_n", "tm500_available"]].reset_index(drop=True)
    vector_columns = [f"tm500_mt{args.embedding_dim}_{index:02d}" for index in range(args.embedding_dim)]
    embedding = pd.concat(
        [embedding, pd.DataFrame(representation, columns=vector_columns)], axis=1
    )
    embedding["embedding_known_pitcher"] = first["pitcher_id"].isin(pitcher_map).to_numpy().astype("int8")
    embedding["trained_through_season"] = fold - 1
    prediction = pd.DataFrame(
        {
            "row_id": frame.loc[valid_indices, "row_id"].to_numpy(),
            "season": fold,
            TARGET: frame.loc[valid_indices, TARGET].to_numpy("int8"),
            "prediction": valid_probability[:, 0].astype("float32"),
        }
    )
    artifact = {
        "state_dict": model.state_dict(),
        "pitcher_map": pitcher_map,
        "history_columns": HISTORY_COLUMNS,
        "trackman_columns": tm_columns,
        "history_scaler": history_scaler,
        "trackman_scaler": tm_scaler,
        "embedding_dim": args.embedding_dim,
        "fold": fold,
        "trained_through_season": fold - 1,
        "min_trackman_season_pitches": 500,
        "main_loss": args.main_loss,
        "f1_weight": args.f1_weight,
    }
    del model, dataset, loader, tensors, valid_tensors, start_tensors
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, epoch_rows, embedding, prediction, artifact


def main():
    global OUTPUT_DIR
    args = parse_args()
    OUTPUT_DIR = Path(args.output_dir).resolve()
    folds = [int(value) for value in args.folds.split(",") if value]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame, tm_columns = load_data()
    results = []
    epochs = []
    embeddings = []
    predictions = []
    for fold in folds:
        metrics, epoch_rows, embedding, prediction, artifact = run_fold(
            frame, tm_columns, fold, args, device
        )
        results.append(metrics)
        epochs.extend(epoch_rows)
        embeddings.append(embedding)
        predictions.append(prediction)
        stem = artifact_stem(args)
        torch.save(
            artifact,
            OUTPUT_DIR / f"{stem}_fold{fold}.pt",
        )
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    stem = artifact_stem(args)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / f"{stem}_validation.csv", index=False)
    pd.DataFrame(epochs).to_csv(OUTPUT_DIR / f"{stem}_epochs.csv", index=False)
    pd.concat(embeddings, ignore_index=True).to_parquet(
        OUTPUT_DIR / f"{stem}_oof_embeddings.parquet", index=False
    )
    pd.concat(predictions, ignore_index=True).to_parquet(
        OUTPUT_DIR / f"{stem}_oof_predictions.parquet", index=False
    )
    manifest = {
        "model": "strict_trackman500_multitask_pitcher_embedding",
        "embedding_dim": args.embedding_dim,
        "folds": folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "aux_weight": args.aux_weight,
        "half_life": args.half_life,
        "main_loss": args.main_loss,
        "f1_weight": args.f1_weight,
        "history_columns": HISTORY_COLUMNS,
        "trackman_columns": tm_columns,
        "targets": LABEL_COLUMNS,
        "results": results,
        "rules": {
            "trackman_min_pitcher_season_pitches": 500,
            "fold_trackman_max_season": "fold - 1",
            "trackman_tower_for_unavailable": "zero gated",
            "rookie_cohorts": ["n=0", "1-100", "101-500", "501-2000", ">2000"],
        },
    }
    (OUTPUT_DIR / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
