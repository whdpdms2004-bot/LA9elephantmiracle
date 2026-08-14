from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
CHECKPOINT_PATH = HERE / "submit_v1" / "model" / "model.pt"
OUTPUT_DIR = HERE / "outputs"


class TrackmanTower(nn.Module):
    def __init__(self, tm_dim):
        super().__init__()
        self.trackman_tower = nn.Sequential(
            nn.Linear(tm_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(64, 24), nn.SiLU(),
        )
        self.individual_projection = nn.Linear(16, 16)


def transform_numeric(frame, columns, preprocessor):
    x = frame[columns].to_numpy(dtype=np.float32)
    median = np.asarray(preprocessor["median"], dtype=np.float32)
    mean = np.asarray(preprocessor["mean"], dtype=np.float32)
    scale = np.asarray(preprocessor["scale"], dtype=np.float32)
    x = np.where(np.isfinite(x), x, median)
    return ((x - mean) / scale).astype("float32")


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    pitcher_to_index = checkpoint["pitcher_to_index"]
    pitcher_ids = sorted(pitcher_to_index)
    lookup = checkpoint["track_lookup"]

    records = []
    for pitcher_id in pitcher_ids:
        record = {"pitcher_id": int(pitcher_id)}
        record.update(lookup.get(int(pitcher_id), {}))
        records.append(record)
    table = pd.DataFrame(records)

    for column in checkpoint["trackman_base_features"]:
        if column not in table:
            table[column] = np.nan
    for column in checkpoint["trackman_count_raw"]:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    table["tm_available"] = table["tm_prior_n"].gt(0).astype("float32")
    for source in checkpoint["trackman_count_raw"]:
        table[f"log1p_{source}"] = np.log1p(table[source].clip(lower=0))

    established = table["tm_prior_max_season_n"].gt(100)
    table["cohort_at_2025_start"] = np.select(
        [
            table.tm_prior_n.eq(0),
            ~established,
            established & table.tm_prev_season_n.eq(0),
        ],
        ["UNSEEN", "ROOKIE_1_25", "RETURNING"],
        default="VETERAN",
    )

    state = checkpoint["state_dict"]
    tower = TrackmanTower(len(checkpoint["trackman_features"]))
    tower.trackman_tower.load_state_dict({
        key.removeprefix("trackman_tower."): value
        for key, value in state.items() if key.startswith("trackman_tower.")
    })
    tower.individual_projection.load_state_dict({
        key.removeprefix("individual_projection."): value
        for key, value in state.items() if key.startswith("individual_projection.")
    })
    tower.eval()

    pitcher_weight = state["pitcher_embedding.weight"]
    pitcher_indices = torch.tensor([pitcher_to_index[pid] for pid in pitcher_ids], dtype=torch.long)
    with torch.inference_mode():
        individual = tower.individual_projection(pitcher_weight[pitcher_indices]).numpy()
        trackman_x = transform_numeric(
            table, checkpoint["trackman_features"], checkpoint["trackman_preprocessor"]
        )
        trackman_embedding = tower.trackman_tower(torch.from_numpy(trackman_x)).numpy()
        cohort_indices = torch.tensor([
            checkpoint["cohort_to_index"][name] for name in table["cohort_at_2025_start"]
        ], dtype=torch.long)
        cohort_embedding = state["cohort_embedding.weight"][cohort_indices].numpy()

    individual_columns = [f"pitcher_embedding_{i:02d}" for i in range(individual.shape[1])]
    trackman_columns = [f"trackman_embedding_{i:02d}" for i in range(trackman_embedding.shape[1])]
    cohort_columns = [f"cohort_embedding_{i:02d}" for i in range(cohort_embedding.shape[1])]
    result = table[[
        "pitcher_id", "cohort_at_2025_start", "tm_available",
        "tm_prior_n", "tm_prev_season_n", "tm_prior_max_season_n",
    ]].copy()
    result["valid_for_season"] = 2025
    result["trained_through_season"] = 2024
    result["safe_for_same_training_rows"] = False
    result = pd.concat([
        result,
        pd.DataFrame(individual, columns=individual_columns),
        pd.DataFrame(trackman_embedding, columns=trackman_columns),
        pd.DataFrame(cohort_embedding, columns=cohort_columns),
    ], axis=1)

    parquet_path = OUTPUT_DIR / "pitcher_embedding_lookup_2025.parquet"
    csv_path = OUTPUT_DIR / "pitcher_embedding_lookup_2025.csv"
    result.to_parquet(parquet_path, index=False)
    result.to_csv(csv_path, index=False)
    print(f"rows={len(result)}, embedding_dim={len(individual_columns) + len(trackman_columns) + len(cohort_columns)}")
    print(parquet_path)


if __name__ == "__main__":
    main()
