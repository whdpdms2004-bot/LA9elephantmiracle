from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn


ID_COL = "row_id"
TARGET_COL = "control_success"
# 평가 서버가 어떤 working directory에서 실행하더라도 script.py가 있는
# 제출물 루트를 기준으로 모든 경로를 절대경로로 해석한다.
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
PREFERRED_MODEL_PATH = APP_DIR / "model" / "model.pt"
OUTPUT_PATH = APP_DIR / "output" / "submission.csv"
BATCH_SIZE = 8192


def resolve_model_path():
    """Resolve the packaged model without depending on the process cwd."""
    if PREFERRED_MODEL_PATH.is_file():
        return PREFERRED_MODEL_PATH

    # Some archive handlers may preserve an unexpected single wrapper folder.
    # Search only below the absolute application directory and accept one match.
    matches = sorted(p.resolve() for p in APP_DIR.rglob("model.pt") if p.is_file())
    if len(matches) == 1:
        return matches[0]

    visible = sorted(
        str(p.relative_to(APP_DIR))
        for p in APP_DIR.rglob("*")
        if p.is_file() and "data" not in p.relative_to(APP_DIR).parts
    )
    raise FileNotFoundError(
        f"Packaged model is missing. expected={PREFERRED_MODEL_PATH}; "
        f"model_matches={matches}; app_files={visible[:50]}"
    )


class PitcherBrierNet(nn.Module):
    def __init__(self, hist_dim, tm_dim, n_pitchers, n_cohorts, mode, embedding_dim=32):
        super().__init__()
        self.mode = mode
        self.pitcher_embedding = nn.Embedding(n_pitchers + 1, 16, padding_idx=0)
        self.cohort_embedding = nn.Embedding(n_cohorts, 8)
        self.history_tower = nn.Sequential(
            nn.Linear(hist_dim, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(96, 48), nn.SiLU(),
        )
        self.trackman_tower = nn.Sequential(
            nn.Linear(tm_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(64, 24), nn.SiLU(),
        )
        self.individual_projection = nn.Linear(16, 16)
        self.cohort_projection = nn.Linear(8, 16)
        self.fusion = nn.Sequential(
            nn.Linear(48 + 24 + 16 + 8, 64), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(64, embedding_dim), nn.LayerNorm(embedding_dim), nn.SiLU(),
        )
        self.direct_head = nn.Linear(embedding_dim, 1)
        self.reverse_head = nn.Linear(embedding_dim, 1)
        self.middle_head = nn.Linear(embedding_dim, 1)
        self.far_head = nn.Linear(embedding_dim, 1)

    def forward(self, history_x, trackman_x, pitcher_idx, cohort_idx, asof_n):
        history_h = self.history_tower(history_x)
        trackman_h = self.trackman_tower(trackman_x)
        individual = self.individual_projection(self.pitcher_embedding(pitcher_idx))
        cohort_raw = self.cohort_embedding(cohort_idx)
        cohort = self.cohort_projection(cohort_raw)
        alpha = (asof_n / (asof_n + 100.0)).clamp(0, 1).unsqueeze(1)
        alpha = alpha * pitcher_idx.ne(0).float().unsqueeze(1)
        pitcher_h = alpha * individual + (1 - alpha) * cohort
        embedding = self.fusion(torch.cat([history_h, trackman_h, pitcher_h, cohort_raw], dim=1))
        direct_logit = self.direct_head(embedding).squeeze(1)
        component_logits = torch.cat([
            self.reverse_head(embedding), self.middle_head(embedding), self.far_head(embedding)
        ], dim=1)
        return direct_logit, component_logits

    def success_probability(self, direct_logit, component_logits):
        if self.mode == "direct":
            return torch.sigmoid(direct_logit)
        return torch.prod(1 - torch.sigmoid(component_logits), dim=1)


def add_legal_row_features(test, checkpoint):
    """Create features from the current row and fixed official-history lookup only."""
    frame = test.copy()
    if not frame["season"].eq(2025).all():
        raise ValueError("This artifact contains a Trackman cutoff fixed for season 2025.")

    for source in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        frame[f"log1p_{source}"] = np.log1p(frame[source].clip(lower=0))

    lookup = checkpoint["track_lookup"]
    records = [lookup.get(int(pid), {}) for pid in frame["pitcher_id"].to_numpy()]
    track = pd.DataFrame.from_records(records, index=frame.index)
    for column in checkpoint["trackman_base_features"]:
        if column not in track:
            track[column] = np.nan

    for column in checkpoint["trackman_count_raw"]:
        track[column] = pd.to_numeric(track[column], errors="coerce").fillna(0.0)
    track["tm_available"] = track["tm_prior_n"].gt(0).astype("float32")
    for source in checkpoint["trackman_count_raw"]:
        track[f"log1p_{source}"] = np.log1p(track[source].clip(lower=0))

    established = track["tm_prior_max_season_n"].gt(100).to_numpy()
    previous_missing = track["tm_prev_season_n"].eq(0).to_numpy()
    prior_missing = track["tm_prior_n"].eq(0).to_numpy()
    asof_n = frame["asof_pitcher_n"].to_numpy()
    cohort = np.select(
        [
            (asof_n == 0) & prior_missing,
            (~established) & (asof_n <= 25),
            (~established) & (asof_n <= 100),
            established & previous_missing,
        ],
        ["UNSEEN", "ROOKIE_1_25", "ROOKIE_26_100", "RETURNING"],
        default="VETERAN",
    )
    return frame, track, cohort


def transform_numeric(frame, columns, preprocessor):
    x = frame[columns].to_numpy(dtype=np.float32)
    median = np.asarray(preprocessor["median"], dtype=np.float32)
    mean = np.asarray(preprocessor["mean"], dtype=np.float32)
    scale = np.asarray(preprocessor["scale"], dtype=np.float32)
    x = np.where(np.isfinite(x), x, median)
    return ((x - mean) / scale).astype("float32")


@torch.inference_mode()
def predict(test, checkpoint, device):
    history, trackman, cohort = add_legal_row_features(test, checkpoint)
    history_x = transform_numeric(
        history, checkpoint["history_features"], checkpoint["history_preprocessor"]
    )
    trackman_x = transform_numeric(
        trackman, checkpoint["trackman_features"], checkpoint["trackman_preprocessor"]
    )
    pitcher_idx = (
        history["pitcher_id"].map(checkpoint["pitcher_to_index"])
        .fillna(0).astype("int64").to_numpy()
    )
    cohort_idx = pd.Series(cohort).map(checkpoint["cohort_to_index"]).astype("int64").to_numpy()
    asof_n = history["asof_pitcher_n"].astype("float32").to_numpy()

    model = PitcherBrierNet(
        hist_dim=len(checkpoint["history_features"]),
        tm_dim=len(checkpoint["trackman_features"]),
        n_pitchers=len(checkpoint["pitcher_to_index"]),
        n_cohorts=len(checkpoint["cohort_to_index"]),
        mode=checkpoint["mode"],
        embedding_dim=checkpoint["embedding_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    result = []
    for start in range(0, len(test), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(test))
        direct_logit, component_logits = model(
            torch.from_numpy(history_x[start:stop]).to(device),
            torch.from_numpy(trackman_x[start:stop]).to(device),
            torch.from_numpy(pitcher_idx[start:stop]).to(device),
            torch.from_numpy(cohort_idx[start:stop]).to(device),
            torch.from_numpy(asof_n[start:stop]).to(device),
        )
        result.append(model.success_probability(direct_logit, component_logits).cpu().numpy())

    probability = np.concatenate(result) if result else np.empty(0, dtype=np.float32)
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    a, b = checkpoint["calibrator"]
    logit = np.log(probability / (1 - probability))
    probability = 1 / (1 + np.exp(-(a * logit + b)))
    return np.clip(probability, 1e-6, 1 - 1e-6)


def build_submission(test, sample_submission, probability):
    if len(test) != len(probability):
        raise ValueError("Prediction length mismatch.")
    if test[ID_COL].duplicated().any() or sample_submission[ID_COL].duplicated().any():
        raise ValueError("row_id must be unique.")
    prediction = pd.Series(probability, index=test[ID_COL].astype(str))
    result = sample_submission[[ID_COL]].copy()
    result[TARGET_COL] = result[ID_COL].astype(str).map(prediction)
    if result[TARGET_COL].isna().any():
        raise ValueError("sample_submission contains a row_id missing from test.csv.")
    if not np.isfinite(result[TARGET_COL]).all():
        raise ValueError("Non-finite prediction detected.")
    return result


def main():
    started = time.perf_counter()
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    model_path = resolve_model_path()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probability = predict(test, checkpoint, device)
    submission = build_submission(test, sample_submission, probability)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"rows={len(test):,}, device={device}, mode={checkpoint['mode']}, elapsed={elapsed:.3f}s")
    print(f"model={model_path}")
    print(f"saved={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
