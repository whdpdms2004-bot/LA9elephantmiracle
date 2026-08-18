"""새 TrackMan PCA가 기존 tm500 요약과 얼마나 중복되는지 투수 단위로 측정한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
SJ = HERE.parent
NEW = HERE / "outputs" / "trackman_release" / "cutoff_2025" / "main_pitcher_release.parquet"
OLD = SJ / "experiment" / "model_optimization" / "trackman500_lookup_2025.parquet"
OUT = HERE / "outputs" / "audit"


def main():
    new = pd.read_parquet(NEW)
    old = pd.read_parquet(OLD)
    merged = new.merge(old, on="pitcher_id", how="inner", suffixes=("", "_old"),
                       validate="one_to_one")
    old_features = [c for c in old.columns if c != "pitcher_id"]
    corr_features = [c for c in old_features if merged[c].nunique(dropna=True) > 1]
    targets = [c for c in new.columns
               if c.startswith("tr_pca_") or c.startswith("tr_resid_pca_")]
    x = merged[old_features].replace([np.inf, -np.inf], np.nan)
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    rows = []
    for target in targets:
        y = merged[target].to_numpy(np.float64)
        model = TransformedTargetRegressor(
            regressor=make_pipeline(SimpleImputer(strategy="median"),
                                    StandardScaler(), Ridge(alpha=100.0)),
            transformer=StandardScaler(),
        )
        pred = cross_val_predict(model, x, y, cv=cv)
        corr = merged[corr_features].corrwith(merged[target]).abs().max()
        rows.append({
            "feature": target,
            "cv_r2_from_existing_tm500": float(r2_score(y, pred)),
            "max_abs_pairwise_corr": float(corr),
        })
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "trackman_pca_novelty.csv", index=False)
    raw_mask = result["feature"].str.startswith("tr_pca_")
    resid_mask = result["feature"].str.startswith("tr_resid_pca_")
    summary = {
        "matched_pitchers": int(len(merged)),
        "existing_features": len(old_features),
        "new_pca_features": len([c for c in targets if c.startswith("tr_pca_")]),
        "new_residual_pca_features": len(
            [c for c in targets if c.startswith("tr_resid_pca_")]),
        "mean_cv_r2": float(result["cv_r2_from_existing_tm500"].mean()),
        "median_cv_r2": float(result["cv_r2_from_existing_tm500"].median()),
        "raw_pca_mean_cv_r2": float(
            result.loc[raw_mask, "cv_r2_from_existing_tm500"].mean()),
        "residual_pca_mean_cv_r2": float(
            result.loc[resid_mask, "cv_r2_from_existing_tm500"].mean()),
        "raw_pca_dims_below_0_5": int(
            (result.loc[raw_mask, "cv_r2_from_existing_tm500"] < 0.5).sum()),
        "residual_pca_dims_below_0_5": int(
            (result.loc[resid_mask, "cv_r2_from_existing_tm500"] < 0.5).sum()),
        "all_dims_with_cv_r2_below_0_5": int(
            (result["cv_r2_from_existing_tm500"] < 0.5).sum()),
        "mean_max_abs_corr": float(result["max_abs_pairwise_corr"].mean()),
    }
    (OUT / "trackman_pca_novelty_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
