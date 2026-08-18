"""계층 피처로 production base의 순방향 OOF 오차 모양만 학습한다.

Val2023: 2022 OOF residual 학습 -> 2023
Val2024: 2022~2023 OOF residual 학습 -> 2024

각 학습 시즌 residual의 평균을 제거해 시즌 레벨/offset을 학습하지 않는다.
피처도 해당 행 시즌 이전 데이터로 만든 component hierarchy만 사용한다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
SJ = HERE.parent
SRC = SJ / "claude" / "src"
MO = SJ / "experiment" / "model_optimization"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MO))
from harness import CACHE, TARGET, load as load_compact, metrics
from run_optuna_enhanced import load_enhanced_frame
from v77_single_xgb_screen import add_direct_products, build_component_unique

OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
OUT = HERE / "outputs" / "oof_residual"
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
EPS = 1e-7


def base_prediction(df: pd.DataFrame, fold: int) -> np.ndarray:
    ids = df.loc[df["season"].eq(fold), "row_id"].to_numpy()
    if fold == 2024:
        return (pd.read_parquet(PROD).set_index("row_id").reindex(ids)
                ["submit021_reverse20_s040_tabm"].to_numpy(np.float64))
    names = sorted({
        re.match(r"(.+)_fold\d{4}\.parquet", path.name).group(1)
        for path in OOF_DIR.glob("*.parquet")
    })
    values = []
    for name in names:
        path = OOF_DIR / f"{name}_fold{fold}.parquet"
        if path.exists():
            values.append(pd.read_parquet(path).set_index("row_id").reindex(ids)
                          ["prediction"].to_numpy(np.float64))
    if not values:
        raise FileNotFoundError(f"base OOF missing for {fold}")
    return np.mean(values, axis=0)


def prepare_parts(frame: pd.DataFrame, base_features: list[str]):
    direct_frame = frame.copy(deep=False)
    direct_names = add_direct_products(direct_frame)
    parts = {}
    for season in (2022, 2023, 2024):
        mask = frame["season"].eq(season).to_numpy()
        hierarchy = build_component_unique(frame, base_features, season).loc[mask]
        hierarchy = hierarchy.reset_index(drop=True)
        direct = direct_frame.loc[mask, direct_names].reset_index(drop=True)
        y = frame.loc[mask, TARGET].to_numpy(np.float64)
        base = np.clip(base_prediction(frame, season), EPS, 1 - EPS)
        residual = y - base
        parts[season] = {
            "C0": hierarchy.to_numpy(np.float32),
            "C1": np.hstack([hierarchy.to_numpy(np.float32),
                              direct.to_numpy(np.float32)]),
            "y": y,
            "base": base,
            "residual_centered": residual - residual.mean(),
            "row_id": frame.loc[mask, "row_id"].to_numpy(),
        }
        print(f"prepared {season}: rows={mask.sum()} C0={parts[season]['C0'].shape[1]} "
              f"C1={parts[season]['C1'].shape[1]} residual_mean={residual.mean():+.6f}",
              flush=True)
    return parts


def main():
    compact = load_compact()
    frame, base_features = load_enhanced_frame()
    if not compact["row_id"].equals(frame["row_id"]):
        raise RuntimeError("frame row order mismatch")
    parts = prepare_parts(frame, base_features)
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for fold in (2023, 2024):
        train_seasons = [season for season in (2022, 2023) if season < fold]
        valid_mask = compact["season"].eq(fold).to_numpy()
        game_type = compact.loc[valid_mask, "game_type"].astype(str).to_numpy()
        bucket = np.digitize(
            compact.loc[valid_mask, "asof_pitcher_n"].to_numpy(np.float64), CUTS)
        weight = BW[bucket]
        component_p0 = np.load(CACHE / f"v75_confirm_P0_{fold}.npy")
        component_p1 = np.load(CACHE / f"v75_confirm_P1_{fold}.npy")
        base = parts[fold]["base"]
        y = parts[fold]["y"]
        current = np.clip((1 - weight) * base + weight * component_p0, EPS, 1 - EPS)
        p1 = np.clip((1 - weight) * base + weight * component_p1, EPS, 1 - EPS)
        ref_base = metrics(y, base)["bss_raw"]
        ref_current = metrics(y, current)["bss_raw"]
        for arm in ("C0", "C1"):
            train_x = np.vstack([parts[season][arm] for season in train_seasons])
            train_y = np.concatenate([
                parts[season]["residual_centered"] for season in train_seasons])
            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=256,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=2.0,
                reg_lambda=100.0,
                objective="reg:squarederror",
                tree_method="hist",
                device="cuda",
                random_state=20260818 + fold,
                n_jobs=6,
            )
            model.fit(train_x, train_y, verbose=False)
            correction = model.predict(parts[fold][arm]).astype(np.float64)
            np.save(OUT / f"{arm}_correction_{fold}.npy", correction)
            for scale in (0.10, 0.25, 0.50, 0.75, 1.00):
                for anchor_name, anchor, ref in (
                    ("base", base, ref_base),
                    ("current_p0", current, ref_current),
                    ("p1", p1, ref_current),
                ):
                    pred = np.clip(anchor + scale * correction, EPS, 1 - EPS)
                    score = metrics(y, pred)
                    rows.append({
                        "fold": fold,
                        "arm": arm,
                        "anchor": anchor_name,
                        "scale": scale,
                        "bss": score["bss_raw"],
                        "dbss": score["bss_raw"] - ref,
                        "pred_mean": float(pred.mean()),
                        "correction_mean": float(correction.mean()),
                        "correction_sd": float(correction.std()),
                        "r_brier": float(np.mean(
                            (pred[game_type == "R"] - y[game_type == "R"]) ** 2)),
                        "f_brier": float(np.mean(
                            (pred[game_type == "F"] - y[game_type == "F"]) ** 2)),
                    })
            print(f"fold={fold} arm={arm} correction mean={correction.mean():+.6f} "
                  f"sd={correction.std():.6f}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "residual_model_grid.csv", index=False)
    pivot = result.pivot_table(
        index=["arm", "anchor", "scale"], columns="fold", values="dbss")
    pivot["worst"] = pivot.min(axis=1)
    pivot["sum"] = pivot[[2023, 2024]].sum(axis=1)
    pivot = pivot.sort_values(["worst", "sum"], ascending=False)
    print(pivot.head(30).round(3).to_string())
    best = pivot.reset_index().iloc[0].to_dict()
    (OUT / "summary.json").write_text(json.dumps({
        "selection": "maximize worst-fold Delta BSS",
        "residual_target": "y - base, centered within each OOF training season",
        "best": best,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
