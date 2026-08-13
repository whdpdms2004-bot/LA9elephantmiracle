from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "experiment" / "model_optimization" / "pitcher_cluster_matchup"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    return parser.parse_args()


def config_spec(config_id):
    registry = pd.read_csv(WORK / "reports" / "cluster_registry.csv")
    selected = registry.loc[registry["config_id"].eq(config_id)]
    if selected.empty:
        raise KeyError(config_id)
    row = selected.iloc[0]
    return int(row["k_left"]), int(row["k_right"])


def build_config(main, config_id):
    k_left, k_right = config_spec(config_id)
    pieces = []
    for season, part in main.groupby("season", sort=True):
        base = part[["row_id", "season", "pitcher_id", "pitcher_hand"]].copy()
        lookup_path = WORK / "clusters" / config_id / f"pitcher_lookup_{int(season)}.parquet"
        if lookup_path.is_file():
            lookup = pd.read_parquet(lookup_path)
            keep = [
                "pitcher_id", "cohort", "cluster_index", "cluster_distance",
                "cluster_entropy", "cluster_top_gap", "cluster_size_pitchers",
            ]
            keep += [column for column in lookup if column.startswith("cluster_q_rank")]
            keep += [column for column in lookup if column.startswith("cluster_style_")]
            base = base.merge(lookup[keep], on="pitcher_id", how="left", validate="many_to_one")
        else:
            base["cohort"] = "new"
            base["cluster_index"] = -1

        base["cohort"] = base["cohort"].fillna("new")
        base["cluster_index"] = base["cluster_index"].fillna(-1).astype("int16")
        available = base["cluster_index"].ge(0)
        base["pcm_available"] = available.astype("int8")
        base["pcm_cluster_id"] = np.where(
            available,
            base["pitcher_hand"].astype("int16") * 100 + base["cluster_index"],
            base["pitcher_hand"].astype("int16") * 100 + 99,
        ).astype("int16")
        for hand, k in [(1, k_left), (2, k_right)]:
            for cluster in range(k):
                base[f"pcm_h{hand}_c{cluster:02d}"] = (
                    base["pitcher_hand"].eq(hand)
                    & base["cluster_index"].eq(cluster)
                ).astype("int8")
        for cohort in ["tm_eligible", "control_only", "rookie", "new"]:
            base[f"pcm_cohort_{cohort}"] = base["cohort"].eq(cohort).astype("int8")
        if "cluster_size_pitchers" in base:
            base["cluster_log_size"] = np.log1p(base["cluster_size_pitchers"])
        drop = ["pitcher_id", "pitcher_hand", "cohort", "cluster_index"]
        base = base.drop(columns=[column for column in drop if column in base])
        rename = {
            column: f"pcm_{column.removeprefix('cluster_')}"
            for column in base
            if column.startswith("cluster_")
        }
        base = base.rename(columns=rename)
        for column in base.columns:
            if column not in {"row_id", "season"}:
                base[column] = pd.to_numeric(base[column], errors="coerce").astype("float32")
        pieces.append(base)
    output = pd.concat(pieces, ignore_index=True)
    if not output["row_id"].equals(main["row_id"]):
        raise RuntimeError(f"Row order mismatch for {config_id}")
    path = WORK / "oof" / f"cluster_features_{config_id}.parquet"
    output.to_parquet(path, index=False)
    record = {
        "config_id": config_id,
        "rows": int(len(output)),
        "features": int(len(output.columns) - 2),
        "path": str(path.relative_to(ROOT)),
        "season_coverage": {
            str(int(season)): float(output.loc[output["season"].eq(season), "pcm_available"].mean())
            for season in sorted(output["season"].unique())
        },
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main():
    args = parse_args()
    configs = [value.strip() for value in args.configs.split(",") if value.strip()]
    main = pd.read_csv(
        ROOT / "data" / "train.csv",
        usecols=["row_id", "season", "pitcher_id", "pitcher_hand"],
    )
    records = [build_config(main, config_id) for config_id in configs]
    path = WORK / "reports" / "row_feature_manifest.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
