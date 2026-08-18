"""새 단일-model 행 파생 피처의 test 스키마와 행 독립성을 검증한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = HERE.parents[1] / "feature_campaign_1000"
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CAMPAIGN))
from v77_single_xgb_screen import add_direct_products, add_trackman_residual


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    names = add_direct_products(out)
    names.extend(add_trackman_residual(out, 2025))
    return out[names]


def main():
    test = pd.read_csv(REPO / "data" / "test.csv")
    batch = derive(test)
    pieces = []
    for index in range(len(test)):
        one = derive(test.iloc[[index]].copy())
        one.index = [index]
        pieces.append(one)
    separate = pd.concat(pieces).loc[batch.index, batch.columns]
    left = batch.to_numpy(np.float64)
    right = separate.to_numpy(np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    diff = np.abs(left[finite] - right[finite])
    same_nan = np.array_equal(np.isnan(left), np.isnan(right))
    max_diff = float(diff.max()) if diff.size else 0.0
    report = {
        "sample_rows": len(test),
        "derived_features": batch.shape[1],
        "same_nan_pattern": same_nan,
        "max_abs_diff_single_vs_batch": max_diff,
        "passed": bool(same_nan and max_diff == 0.0),
    }
    output = CAMPAIGN / "outputs" / "verification"
    output.mkdir(parents=True, exist_ok=True)
    (output / "row_derived_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise AssertionError(report)


if __name__ == "__main__":
    main()
