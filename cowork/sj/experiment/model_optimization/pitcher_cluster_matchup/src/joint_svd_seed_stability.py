from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))

from deep_pitcher_cluster_search import CUTOFFS, REVERSE_FEATURES, load_base  # noqa: E402
from joint_svd_cluster_search import DIRECT_FEATURES, pair_features, prepare_cutoff  # noqa: E402
from joint_svd_outer_search import correction, optimize_candidate, prepare_fixed_folds  # noqa: E402
from screen_reverse_batter_clusters import load_main  # noqa: E402


SEEDS = [17, 43, 97, 2026, 4099]
CANDIDATES = [
    {
        "name": "aggressive",
        "matrix_lambda": 100.0, "svd_dim": 4,
        "pitcher_k": (4, 8), "batter_k": (6, 8),
        "cluster_mode": "unit", "feature_variant": "cluster",
    },
    {
        "name": "safe",
        "matrix_lambda": 100.0, "svd_dim": 4,
        "pitcher_k": (3, 6), "batter_k": (3, 4),
        "cluster_mode": "unit", "feature_variant": "cluster_direct",
    },
]


def build_seed_feature(main, candidate, seed):
    pieces = []
    audits = []
    for cutoff in CUTOFFS:
        prepared = prepare_cutoff(
            main, cutoff, candidate["matrix_lambda"], candidate["svd_dim"], seed
        )
        feature, audit = pair_features(
            prepared, candidate["pitcher_k"], candidate["batter_k"],
            candidate["cluster_mode"], seed,
        )
        pieces.append(feature)
        audits.append({"seed": seed, "cutoff": cutoff, **audit})
    return pd.concat(pieces, ignore_index=True), audits


def subset_names(seeds):
    output = []
    for size in range(1, len(seeds) + 1):
        for subset in itertools.combinations(seeds, size):
            output.append(subset)
    return output


def main():
    main_frame = load_main()
    base = load_base()
    folds = prepare_fixed_folds(base)
    all_rows = []
    audit_rows = []
    correlation_rows = []
    for candidate in CANDIDATES:
        feature_columns = (
            REVERSE_FEATURES
            if candidate["feature_variant"] == "cluster"
            else REVERSE_FEATURES + DIRECT_FEATURES
        )
        corrections = {seed: {} for seed in SEEDS}
        for seed in SEEDS:
            feature, audits = build_seed_feature(main_frame, candidate, seed)
            audit_rows.extend({"candidate": candidate["name"], **item} for item in audits)
            frame = base.merge(feature, on=["row_id", "season"], validate="one_to_one")
            for train_year, valid_year in [(2022, 2023), (2023, 2024)]:
                corrections[seed][valid_year] = correction(
                    frame, feature_columns, 100.0, train_year, valid_year
                )
            print(json.dumps({
                "candidate": candidate["name"], "completed_seed": seed
            }, ensure_ascii=False), flush=True)

        for year in [2023, 2024]:
            matrix = np.column_stack([corrections[seed][year] for seed in SEEDS])
            corr = np.corrcoef(matrix, rowvar=False)
            for i, left in enumerate(SEEDS):
                for j in range(i + 1, len(SEEDS)):
                    correlation_rows.append({
                        "candidate": candidate["name"], "season": year,
                        "left_seed": left, "right_seed": SEEDS[j],
                        "correlation": float(corr[i, j]),
                    })

        for subset in subset_names(SEEDS):
            averaged = {
                year: np.mean([corrections[seed][year] for seed in subset], axis=0)
                for year in [2023, 2024]
            }
            metadata = {
                **candidate,
                "pitcher_k": str(candidate["pitcher_k"]),
                "batter_k": str(candidate["batter_k"]),
                "seeds": "-".join(map(str, subset)),
                "seed_count": len(subset),
            }
            all_rows.extend(optimize_candidate(folds, averaged, metadata))

    result = pd.DataFrame(all_rows)
    reports = WORK / "reports"
    result.to_csv(reports / "joint_svd_seed_stability.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(
        reports / "joint_svd_seed_stability_audit.csv", index=False
    )
    pd.DataFrame(correlation_rows).to_csv(
        reports / "joint_svd_seed_stability_correlation.csv", index=False
    )
    outer = result.loc[result["criterion"].eq("outer")].sort_values(
        "outer_brier_2024"
    )
    robust = result.loc[result["criterion"].eq("robust")].sort_values(
        "robust_objective"
    )
    summary = {
        "best_outer": outer.iloc[0].to_dict(),
        "best_robust": robust.iloc[0].to_dict(),
        "best_by_candidate": (
            outer.groupby("name", as_index=False).first()
            .sort_values("outer_brier_2024").to_dict("records")
        ),
        "mean_seed_correlation": (
            pd.DataFrame(correlation_rows)
            .groupby(["candidate", "season"])["correlation"].mean()
            .reset_index().to_dict("records")
        ),
    }
    (reports / "joint_svd_seed_stability.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTOP OUTER 20")
    print(outer.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
