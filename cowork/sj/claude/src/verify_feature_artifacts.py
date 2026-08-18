"""V76 산출물의 시점/키/유한값/행 독립성 계약을 검증한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = HERE.parents[1] / "feature_campaign_1000"
ROOT = HERE.parents[3]
ART = CAMPAIGN / "outputs" / "trackman_release"
OUT = CAMPAIGN / "outputs" / "audit" / "artifact_verification.json"


def mapped(frame: pd.DataFrame, lookup: pd.DataFrame, columns: list[str]):
    table = lookup.set_index("pitcher_id")
    return pd.DataFrame({c: frame["pitcher_id"].map(table[c]).to_numpy()
                         for c in columns}, index=frame.index)


def main():
    checks = []
    train = pd.read_csv(ROOT / "data" / "train.csv",
                        usecols=["row_id", "season", "pitcher_id"])
    for cutoff in (2023, 2024, 2025):
        folder = ART / f"cutoff_{cutoff}"
        lookup = pd.read_parquet(folder / "main_pitcher_release.parquet")
        summary = pd.read_parquet(folder / "trackman_release_summary.parquet")
        spec = json.loads((folder / "pca_spec.json").read_text(encoding="utf-8"))
        residual_spec = json.loads(
            (folder / "residual_pca_spec.json").read_text(encoding="utf-8"))
        assert not lookup["pitcher_id"].duplicated().any()
        assert not lookup["pitcher_trackman_id"].duplicated().any()
        assert int(lookup["evidence_max_season"].max()) < cutoff
        assert int(summary["tr_last_season"].max()) < cutoff
        assert not any("target" in c.lower() or "control_success" in c.lower()
                       for c in lookup.columns)
        pca_cols = [c for c in lookup if c.startswith("tr_pca_")]
        residual_cols = [c for c in lookup if c.startswith("tr_resid_pca_")]
        assert len(pca_cols) == spec["n_components"]
        assert np.isfinite(lookup[pca_cols].to_numpy()).all()
        assert len(spec["components"]) == len(pca_cols)
        assert len(spec["components"][0]) == len(spec["input_columns"])
        assert len(residual_cols) == residual_spec["n_components"]
        assert np.isfinite(lookup[residual_cols].to_numpy()).all()

        season_rows = train[train["season"].eq(min(cutoff, 2024))].head(100)
        cols = pca_cols + residual_cols + ["tr_latest_all_release_trace",
                           "tr_recent_all_release_area", "cw_mean_sim"]
        full = mapped(season_rows, lookup, cols)
        singles = pd.concat(
            [mapped(season_rows.iloc[[i]], lookup, cols)
             for i in range(len(season_rows))]
        ).sort_index()
        a, b = full.sort_index().to_numpy(), singles.to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        max_diff = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert max_diff == 0.0
        checks.append({
            "cutoff": cutoff,
            "max_evidence_season": int(lookup["evidence_max_season"].max()),
            "max_trackman_season": int(summary["tr_last_season"].max()),
            "main_pitchers": int(len(lookup)),
            "pca_features": len(pca_cols),
            "residual_pca_features": len(residual_cols),
            "single_vs_batch_max_abs_diff": max_diff,
            "status": "pass",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
