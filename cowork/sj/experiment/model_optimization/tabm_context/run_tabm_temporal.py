"""Time-safe TabM experiments for pitch-level control probability.

Selection policy
----------------
* Tune/select only on validation seasons 2022 and 2023.
* Open 2024 exactly once after the configuration is frozen.
* A fold uses rows with season < validation season for every fitted transform.

Feature sets
------------
T0: the established BASE43 (24 situation + 19 as-of history features).
T1: T0 plus deterministic, target-free interactions and trend features.
T2: T1 plus raw pitcher/batter/team identities as categorical inputs.  TabM's
    one-hot input followed by its first linear layer acts as a learned embedding.

This file deliberately does not use TrackMan.  TrackMan is reserved for the
final 2025 model after the 2024 gate, and must be joined strictly as-of.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tabm
import torch
import torch.nn.functional as F
from rtdl_num_embeddings import LinearReLUEmbeddings


TARGET = "control_success"
ID_COLUMNS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
BASE_CATEGORICAL = [
    "top_bottom",
    "game_type",
    "base_state",
    "game_month",
    "game_dayofweek",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
]
LOG1P_COLUMNS = {
    "asof_pitcher_n",
    "asof_batter_n",
    "asof_pitcher_pitchmix_n",
}
PROFILE_DIR = Path(
    "experiment/model_optimization/pitcher_cluster_matchup/profiles"
)
_PROFILE_CACHE: dict[int, pd.DataFrame] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment/model_optimization/tabm_context/outputs"),
    )
    parser.add_argument("--name", default="t0_default")
    parser.add_argument(
        "--feature-set",
        choices=["t0", "t1", "t2", "t3", "t3r"],
        default="t0",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument(
        "--game-type",
        choices=["all", "R", "F"],
        default="all",
        help="Train and validate one regime only. Use separate R/F runs for dispatch.",
    )
    parser.add_argument(
        "--min-train-season",
        type=int,
        default=None,
        help="Optional recent-regime cutoff, mainly for the post-break F expert.",
    )
    parser.add_argument("--seed", type=int, default=20260813)

    parser.add_argument("--d-block", type=int, default=384)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--arch-type", choices=["tabm", "tabm-mini"], default="tabm")
    parser.add_argument(
        "--num-embedding",
        choices=["none", "linear_relu"],
        default="none",
    )
    parser.add_argument("--num-embedding-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=["bce", "brier", "hybrid"], default="hybrid")
    parser.add_argument(
        "--brier-weight",
        type=float,
        default=0.25,
        help="Only used by hybrid: (1-w)*BCE + w*Brier.",
    )

    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--data-on-cpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def brier_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    rate = float(y.mean())
    brier = float(np.mean(np.square(p - y)))
    null_brier = rate * (1.0 - rate)
    bss = float(100000.0 * (1.0 - brier / null_brier)) if null_brier > 0 else float("nan")
    return {
        "n": int(len(y)),
        "target_mean": rate,
        "prediction_mean": float(p.mean()),
        "brier": brier,
        "null_brier": float(null_brier),
        "bss": bss,
    }


def determine_base_features(columns: list[str]) -> list[str]:
    asof = [c for c in columns if c.startswith("asof_")]
    situation = [
        c
        for c in columns
        if c not in asof + ID_COLUMNS + ["row_id", TARGET]
    ]
    base = situation + asof
    if len(base) != 43:
        raise ValueError(f"Expected BASE43, found {len(base)} columns: {base}")
    return base


def add_t1_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic interactions; no label or future row is referenced."""
    out = frame.copy()
    eps = 1e-6

    out["fx_count_state"] = (
        out["balls_before"].astype(np.float32) * 3.0
        + out["strikes_before"].astype(np.float32)
    )
    out["fx_same_hand"] = (
        out["pitcher_hand"].astype(str) == out["batter_hand"].astype(str)
    ).astype(np.float32)
    out["fx_abs_score_diff"] = out["score_diff_pitcher_team"].abs().astype(np.float32)
    out["fx_late_inning"] = (out["inning"] >= 7).astype(np.float32)
    out["fx_high_leverage"] = (out["li"].fillna(0) >= 2.0).astype(np.float32)
    out["fx_li_x_late"] = out["li"].astype(np.float32) * out["fx_late_inning"]
    out["fx_li_x_runners"] = out["li"].astype(np.float32) * out["num_runners_on"].astype(np.float32)

    out["fx_pitcher_minus_batter_success"] = (
        out["asof_pitcher_success_rate"] - out["asof_batter_success_rate"]
    )
    out["fx_pitcher_failure_rate"] = 1.0 - out["asof_pitcher_success_rate"]
    out["fx_known_middle_reverse_sum"] = (
        out["asof_pitcher_middle_rate"] + out["asof_pitcher_reverse_rate"]
    )
    out["fx_middle_reverse_balance"] = (
        out["asof_pitcher_middle_rate"] - out["asof_pitcher_reverse_rate"]
    )
    out["fx_ball_minus_strike"] = (
        out["asof_pitcher_ball_rate"] - out["asof_pitcher_strike_rate"]
    )
    out["fx_fastball_minus_breaking"] = (
        out["asof_pitcher_fastball_rate"] - out["asof_pitcher_breaking_rate"]
    )
    out["fx_offspeed_share"] = out["asof_pitcher_offspeed_rate"]

    out["fx_success_trend_1v5"] = (
        out["asof_pitcher_prev1_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    )
    out["fx_success_trend_3v5"] = (
        out["asof_pitcher_prev3_game_success_rate"]
        - out["asof_pitcher_prev5_game_success_rate"]
    )
    out["fx_middle_trend_1v5"] = (
        out["asof_pitcher_prev1_game_middle_rate"]
        - out["asof_pitcher_prev5_game_middle_rate"]
    )
    out["fx_middle_trend_3v5"] = (
        out["asof_pitcher_prev3_game_middle_rate"]
        - out["asof_pitcher_prev5_game_middle_rate"]
    )

    pitcher_n = out["asof_pitcher_n"].astype(np.float64).clip(lower=0)
    batter_n = out["asof_batter_n"].astype(np.float64).clip(lower=0)
    out["fx_pitcher_reliability"] = pitcher_n / (pitcher_n + 100.0 + eps)
    out["fx_batter_reliability"] = batter_n / (batter_n + 100.0 + eps)
    out["fx_joint_reliability"] = np.sqrt(
        out["fx_pitcher_reliability"] * out["fx_batter_reliability"]
    )
    return out


def load_control_profile(cutoff: int) -> pd.DataFrame:
    """Load only strict-prior control profiles; all TrackMan columns are excluded."""
    if cutoff not in _PROFILE_CACHE:
        path = PROFILE_DIR / f"pitcher_profile_cutoff_{cutoff}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing strict-as-of pitcher profile for cutoff {cutoff}: {path}"
            )
        profile = pd.read_parquet(path)
        safe_columns = ["pitcher_id"] + [c for c in profile.columns if c.startswith("ctl_")]
        profile = profile[safe_columns].copy()
        if profile["pitcher_id"].duplicated().any():
            raise ValueError(f"Duplicate pitcher_id in {path}")
        _PROFILE_CACHE[cutoff] = profile
    return _PROFILE_CACHE[cutoff]


def attach_t3_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach pitcher residual/component history using seasons strictly before each row."""
    chunks: list[pd.DataFrame] = []
    minimum_profile_cutoff = 2020
    for season, part in frame.groupby("season", sort=False):
        season = int(season)
        part = part.copy()
        part["__source_index"] = part.index
        if season >= minimum_profile_cutoff:
            profile = load_control_profile(season)
            part = part.merge(profile, on="pitcher_id", how="left", validate="many_to_one")
        chunks.append(part)
    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values("__source_index", kind="stable").set_index("__source_index")
    out.index.name = frame.index.name

    hand1 = out.get("ctl_split_batter_hand1_resid", pd.Series(np.nan, index=out.index))
    hand2 = out.get("ctl_split_batter_hand2_resid", pd.Series(np.nan, index=out.index))
    out["fx_ctl_current_batter_hand_resid"] = np.where(
        out["batter_hand"].eq(1), hand1, hand2
    )
    full_count = out["balls_before"].eq(3) & out["strikes_before"].eq(2)
    two_strike = out["strikes_before"].eq(2)
    full_resid = out.get("ctl_split_full_count_resid", pd.Series(np.nan, index=out.index))
    two_resid = out.get("ctl_split_two_strike_resid", pd.Series(np.nan, index=out.index))
    out["fx_ctl_current_count_resid"] = np.where(
        full_count, full_resid, np.where(two_strike, two_resid, 0.0)
    )
    high_li_resid = out.get("ctl_split_high_li_resid", pd.Series(np.nan, index=out.index))
    out["fx_ctl_current_li_resid"] = np.where(out["li"].ge(2.0), high_li_resid, 0.0)
    return out


@dataclass
class NumericTransform:
    name: str
    median: float
    lower: float
    upper: float
    mean: float
    std: float
    log1p: bool
    add_missing: bool


class FoldPreprocessor:
    def __init__(self, feature_set: str, base_features: list[str]):
        self.feature_set = feature_set
        self.base_features = list(base_features)
        self.feature_columns: list[str] = []
        self.cat_columns: list[str] = []
        self.num_columns: list[str] = []
        self.cat_maps: dict[str, dict[str, int]] = {}
        self.cat_cardinalities: list[int] = []
        self.num_transforms: list[NumericTransform] = []
        self.num_output_names: list[str] = []

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns = list(self.base_features)
        if self.feature_set in {"t1", "t2"}:
            frame = add_t1_features(frame)
            columns.extend([c for c in frame.columns if c.startswith("fx_")])
        if self.feature_set == "t2":
            columns.extend(ID_COLUMNS)
        if self.feature_set in {"t3", "t3r"}:
            frame = attach_t3_profiles(frame)
            profile_columns = [c for c in frame.columns if c.startswith("ctl_")]
            if self.feature_set == "t3r":
                meta = {
                    "ctl_total_n",
                    "ctl_last_n",
                    "ctl_last_season",
                    "ctl_season_gap",
                    "ctl_history_seasons",
                    "ctl_rookie",
                }
                profile_columns = [
                    c
                    for c in profile_columns
                    if c in meta
                    or c.endswith("_resid")
                    or c.endswith("_between_std")
                    or (c.startswith("ctl_split_") and c.endswith("_n"))
                ]
            columns.extend(profile_columns)
            columns.extend([c for c in frame.columns if c.startswith("fx_ctl_")])
        # Preserve order and prevent accidental duplicates.
        columns = list(dict.fromkeys(columns))
        return frame.loc[:, columns]

    def fit(self, train: pd.DataFrame) -> "FoldPreprocessor":
        x = self._feature_frame(train)
        self.feature_columns = x.columns.tolist()
        identity_cats = ID_COLUMNS if self.feature_set == "t2" else []
        self.cat_columns = [c for c in BASE_CATEGORICAL + identity_cats if c in x.columns]
        self.num_columns = [c for c in x.columns if c not in self.cat_columns]

        self.cat_maps = {}
        self.cat_cardinalities = []
        for column in self.cat_columns:
            values = x[column].astype("string").fillna("<NA>")
            unique = sorted(values.unique().tolist())
            # 0 is reserved for categories unseen in the training history.
            mapping = {value: i + 1 for i, value in enumerate(unique)}
            self.cat_maps[column] = mapping
            self.cat_cardinalities.append(len(mapping) + 1)

        self.num_transforms = []
        self.num_output_names = []
        for column in self.num_columns:
            raw = pd.to_numeric(x[column], errors="coerce").to_numpy(dtype=np.float64)
            raw[~np.isfinite(raw)] = np.nan
            use_log = column in LOG1P_COLUMNS or (
                column.startswith("ctl_") and column.endswith("_n")
            )
            if use_log:
                raw = np.log1p(np.clip(raw, 0.0, None))
            missing = np.isnan(raw)
            if missing.all():
                median = lower = upper = mean = 0.0
                std = 1.0
            else:
                median = float(np.nanmedian(raw))
                filled = np.where(missing, median, raw)
                lower, upper = np.quantile(filled, [0.001, 0.999]).astype(float)
                clipped = np.clip(filled, lower, upper)
                mean = float(clipped.mean())
                std = float(clipped.std())
                if not np.isfinite(std) or std < 1e-6:
                    std = 1.0
            transform = NumericTransform(
                name=column,
                median=median,
                lower=float(lower),
                upper=float(upper),
                mean=mean,
                std=std,
                log1p=use_log,
                add_missing=bool(missing.any()),
            )
            self.num_transforms.append(transform)
            self.num_output_names.append(column)
            if transform.add_missing:
                self.num_output_names.append(f"{column}__missing")
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self._feature_frame(frame)
        cat_parts: list[np.ndarray] = []
        for column in self.cat_columns:
            mapping = self.cat_maps[column]
            values = x[column].astype("string").fillna("<NA>")
            cat_parts.append(
                values.map(mapping).fillna(0).to_numpy(dtype=np.int64, copy=False)
            )
        x_cat = (
            np.column_stack(cat_parts).astype(np.int64, copy=False)
            if cat_parts
            else np.empty((len(x), 0), dtype=np.int64)
        )

        num_parts: list[np.ndarray] = []
        for transform in self.num_transforms:
            raw = pd.to_numeric(x[transform.name], errors="coerce").to_numpy(dtype=np.float64)
            raw[~np.isfinite(raw)] = np.nan
            if transform.log1p:
                raw = np.log1p(np.clip(raw, 0.0, None))
            missing = np.isnan(raw)
            filled = np.where(missing, transform.median, raw)
            scaled = (
                np.clip(filled, transform.lower, transform.upper) - transform.mean
            ) / transform.std
            num_parts.append(scaled.astype(np.float32, copy=False))
            if transform.add_missing:
                num_parts.append(missing.astype(np.float32, copy=False))
        x_num = (
            np.column_stack(num_parts).astype(np.float32, copy=False)
            if num_parts
            else np.empty((len(x), 0), dtype=np.float32)
        )
        return x_num, x_cat

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set,
            "base_features": self.base_features,
            "feature_columns": self.feature_columns,
            "cat_columns": self.cat_columns,
            "num_columns": self.num_columns,
            "cat_maps": self.cat_maps,
            "cat_cardinalities": self.cat_cardinalities,
            "num_transforms": [asdict(x) for x in self.num_transforms],
            "num_output_names": self.num_output_names,
        }


def deterministic_subsample(indices: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=limit, replace=False)
    return np.sort(selected)


def move_dataset(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    data_on_cpu: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    storage = torch.device("cpu") if data_on_cpu else device
    return (
        torch.from_numpy(np.ascontiguousarray(x_num)).to(storage),
        torch.from_numpy(np.ascontiguousarray(x_cat)).to(storage),
        torch.from_numpy(np.asarray(y, dtype=np.float32)).to(storage),
    )


def take_batch(x: torch.Tensor, idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.device.type == device.type:
        return x.index_select(0, idx)
    return x.index_select(0, idx.cpu()).to(device, non_blocking=True)


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    member_std_sum = 0.0
    n_rows = len(x_num)
    for start in range(0, n_rows, batch_size):
        stop = min(n_rows, start + batch_size)
        idx = torch.arange(start, stop, device=device)
        xb_num = take_batch(x_num, idx, device)
        xb_cat = take_batch(x_cat, idx, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(
                xb_num if xb_num.shape[1] else None,
                xb_cat if xb_cat.shape[1] else None,
            ).squeeze(-1)
            member_prob = logits.float().sigmoid()
            ensemble_prob = member_prob.mean(dim=1)
        predictions.append(ensemble_prob.cpu().numpy())
        member_std_sum += float(member_prob.std(dim=1, unbiased=False).sum().item())
    return np.concatenate(predictions), member_std_sum / max(n_rows, 1)


def training_loss(logits: torch.Tensor, y: torch.Tensor, kind: str, brier_weight: float) -> torch.Tensor:
    target = y[:, None].expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if kind == "bce":
        return bce
    brier = torch.square(logits.sigmoid() - target).mean()
    if kind == "brier":
        return brier
    return (1.0 - brier_weight) * bce + brier_weight * brier


def fit_fold(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    base_features: list[str],
    fold: int,
    run_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    seasons = frame["season"].to_numpy()
    train_mask = seasons < fold
    valid_mask = seasons == fold
    if args.min_train_season is not None:
        train_mask &= seasons >= args.min_train_season
    if args.game_type != "all":
        game_type = frame["game_type"].astype(str).to_numpy()
        train_mask &= game_type == args.game_type
        valid_mask &= game_type == args.game_type
    train_idx = np.flatnonzero(train_mask)
    valid_idx = np.flatnonzero(valid_mask)
    train_idx = deterministic_subsample(train_idx, args.max_train_rows, fold_seed)
    valid_idx = deterministic_subsample(valid_idx, args.max_valid_rows, fold_seed + 1)
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise ValueError(f"Fold {fold} has train={len(train_idx)} valid={len(valid_idx)}")

    print(
        f"\n[FOLD {fold}] train={len(train_idx):,} valid={len(valid_idx):,} "
        f"train seasons={sorted(frame.iloc[train_idx]['season'].unique().tolist())}",
        flush=True,
    )
    preprocessing_started = time.time()
    preprocessor = FoldPreprocessor(args.feature_set, base_features).fit(frame.iloc[train_idx])
    xtr_num, xtr_cat = preprocessor.transform(frame.iloc[train_idx])
    xva_num, xva_cat = preprocessor.transform(frame.iloc[valid_idx])
    ytr = frame.iloc[train_idx][TARGET].to_numpy(dtype=np.float32)
    yva = frame.iloc[valid_idx][TARGET].to_numpy(dtype=np.float32)
    preprocessing_seconds = time.time() - preprocessing_started

    print(
        f"  prepared num={xtr_num.shape[1]} cat={xtr_cat.shape[1]} "
        f"onehot={sum(preprocessor.cat_cardinalities)} in {preprocessing_seconds:.1f}s",
        flush=True,
    )

    xtr_num_t, xtr_cat_t, ytr_t = move_dataset(
        xtr_num, xtr_cat, ytr, device, args.data_on_cpu or device.type == "cpu"
    )
    xva_num_t, xva_cat_t, _ = move_dataset(
        xva_num, xva_cat, yva, device, args.data_on_cpu or device.type == "cpu"
    )
    del xtr_num, xtr_cat, xva_num, xva_cat

    num_embeddings = None
    if args.num_embedding == "linear_relu":
        num_embeddings = LinearReLUEmbeddings(
            n_features=xtr_num_t.shape[1],
            d_embedding=args.num_embedding_dim,
        )
    model = tabm.TabM.make(
        n_num_features=xtr_num_t.shape[1],
        cat_cardinalities=preprocessor.cat_cardinalities,
        d_out=1,
        num_embeddings=num_embeddings,
        n_blocks=args.n_blocks,
        d_block=args.d_block,
        dropout=args.dropout,
        k=args.k,
        arch_type=args.arch_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, threshold=1e-6, min_lr=1e-5
    )
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_brier = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_prediction: np.ndarray | None = None
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    training_started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.time()
        permutation = torch.randperm(len(ytr_t), device=device)
        losses: list[float] = []
        steps = 0
        for start in range(0, len(ytr_t), args.batch_size):
            if args.max_steps_per_epoch is not None and steps >= args.max_steps_per_epoch:
                break
            idx = permutation[start : start + args.batch_size]
            xb_num = take_batch(xtr_num_t, idx, device)
            xb_cat = take_batch(xtr_cat_t, idx, device)
            yb = take_batch(ytr_t, idx, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(
                    xb_num if xb_num.shape[1] else None,
                    xb_cat if xb_cat.shape[1] else None,
                ).squeeze(-1)
                loss = training_loss(logits, yb, args.loss, args.brier_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
            steps += 1

        pva, member_std = predict(
            model, xva_num_t, xva_cat_t, args.eval_batch_size, device, amp
        )
        metrics = brier_metrics(yva, pva)
        scheduler.step(metrics["brier"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "steps": steps,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "member_probability_std": member_std,
            "seconds": time.time() - epoch_started,
            **metrics,
        }
        history.append(epoch_record)
        print(
            f"  epoch={epoch:02d} loss={epoch_record['train_loss']:.6f} "
            f"brier={metrics['brier']:.8f} bss={metrics['bss']:.3f} "
            f"mean={metrics['prediction_mean']:.5f}/{metrics['target_mean']:.5f} "
            f"member_std={member_std:.5f} lr={epoch_record['lr']:.2e} "
            f"time={epoch_record['seconds']:.1f}s",
            flush=True,
        )

        if metrics["brier"] < best_brier - 1e-8:
            best_brier = metrics["brier"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_prediction = pva.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(f"  early stop: best epoch={best_epoch}", flush=True)
            break

    if best_state is None or best_prediction is None:
        raise RuntimeError("Training produced no valid checkpoint")

    fold_dir = run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "model_args": {
            "n_num_features": int(xtr_num_t.shape[1]),
            "cat_cardinalities": preprocessor.cat_cardinalities,
            "d_out": 1,
            "n_blocks": args.n_blocks,
            "d_block": args.d_block,
            "dropout": args.dropout,
            "k": args.k,
            "arch_type": args.arch_type,
            "num_embedding": args.num_embedding,
            "num_embedding_dim": args.num_embedding_dim,
        },
        "preprocessor": preprocessor.to_dict(),
        "fold": fold,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, fold_dir / "model.pt")

    validation = frame.iloc[valid_idx][["row_id", "season", "game_type", TARGET]].copy()
    validation["prediction"] = best_prediction
    validation["fold"] = fold
    validation["feature_set"] = args.feature_set
    validation.to_parquet(fold_dir / "oof.parquet", index=False)

    overall = brier_metrics(yva, best_prediction)
    group_metrics: dict[str, Any] = {}
    game_type = validation["game_type"].astype(str).to_numpy()
    for value in sorted(np.unique(game_type)):
        mask = game_type == value
        group_metrics[value] = brier_metrics(yva[mask], best_prediction[mask])
    summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "overall": overall,
        "by_game_type": group_metrics,
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": time.time() - training_started,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "train_seasons": sorted(frame.iloc[train_idx]["season"].unique().astype(int).tolist()),
        "n_num_features_after_preprocess": int(xtr_num_t.shape[1]),
        "n_cat_features": int(xtr_cat_t.shape[1]),
        "cat_cardinalities": preprocessor.cat_cardinalities,
        "history": history,
    }
    (fold_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return validation, summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available; pass --cpu only for a small smoke test")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = args.output_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} torch={torch.__version__} tabm={getattr(tabm, '__version__', '?')}")
    print(f"loading {args.data}", flush=True)
    frame = pd.read_csv(args.data)
    base_features = determine_base_features(frame.columns.tolist())
    print(f"rows={len(frame):,} BASE={len(base_features)} target={frame[TARGET].mean():.6f}")
    print(f"selection folds={args.folds}; 2024 is forbidden unless explicitly passed", flush=True)

    all_oof: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in args.folds:
        validation, summary = fit_fold(args, frame, base_features, fold, run_dir, device)
        all_oof.append(validation)
        fold_summaries.append(summary)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(all_oof, ignore_index=True)
    oof.to_parquet(run_dir / "oof_all.parquet", index=False)
    weighted_brier = float(
        np.average(
            [x["overall"]["brier"] for x in fold_summaries],
            weights=[x["overall"]["n"] for x in fold_summaries],
        )
    )
    run_summary = {
        "name": args.name,
        "selection_policy": "select on 2022/2023; gate on 2024 once after freeze",
        "trackman_used": False,
        "game_type_dispatch": args.game_type,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "base_features": base_features,
        "folds": fold_summaries,
        "weighted_brier": weighted_brier,
        "mean_fold_bss": float(np.mean([x["overall"]["bss"] for x in fold_summaries])),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nDONE")
    for x in fold_summaries:
        print(
            f"fold={x['fold']} epoch={x['best_epoch']} "
            f"brier={x['overall']['brier']:.8f} bss={x['overall']['bss']:.3f}"
        )
    print(f"mean fold BSS={run_summary['mean_fold_bss']:.3f}")


if __name__ == "__main__":
    main()
