from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiment" / "pitcher_embedding" / "outputs" / "trackman500"
WORK_DIR = ROOT / "experiment" / "model_optimization"
MIN_PITCHES = 500


def predictive_columns(reference: pd.DataFrame) -> list[str]:
    excluded = {
        "pitcher_id",
        "pitcher_trackman_id",
        "cutoff",
        "trained_through_season",
        "min_trackman_season_pitches",
        "evidence_max_season",
        "tm500_cutoff",
        "tm500_trained_through_season",
        "tm500_min_season_pitches",
    }
    return [column for column in reference.columns if column not in excluded]


def load_cutoff(cutoff: int, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    directory = BASE / f"cutoff_{cutoff}"
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if int(audit["min_season_pitches"]) != MIN_PITCHES:
        raise AssertionError(f"Wrong Trackman threshold for cutoff {cutoff}")
    if int(audit["crosswalk_evidence_max_season"]) >= cutoff:
        raise AssertionError(f"Future Trackman evidence in cutoff {cutoff}")
    lookup = pd.read_parquet(directory / "main_pitcher_trackman500.parquet")
    return lookup[["pitcher_id"] + columns], audit


def main():
    train = pd.read_csv(
        ROOT / "data" / "train.csv", usecols=["row_id", "season", "pitcher_id"]
    )
    reference = pd.read_parquet(
        BASE / "cutoff_2025" / "main_pitcher_trackman500.parquet"
    )
    columns = predictive_columns(reference)
    parts = []
    season_audit = []

    for season, target in train.groupby("season", sort=True):
        season = int(season)
        base = target[["row_id", "season", "pitcher_id"]].copy()
        if season == 2019:
            for column in columns:
                base[column] = np.nan
            audit = {
                "cutoff": 2019,
                "crosswalk_pitchers": 0,
                "crosswalk_evidence_max_season": None,
            }
        else:
            lookup, audit = load_cutoff(season, columns)
            base = base.merge(lookup, on="pitcher_id", how="left", validate="many_to_one")
        base["tm500_available"] = base["tm500_total_pitches"].notna().astype("int8")
        base["tm500_unavailable"] = (1 - base["tm500_available"]).astype("int8")
        season_audit.append(
            {
                "season": season,
                "rows": len(base),
                "available_rows": int(base["tm500_available"].sum()),
                "row_coverage": float(base["tm500_available"].mean()),
                "available_pitchers": int(
                    base.loc[base["tm500_available"].eq(1), "pitcher_id"].nunique()
                ),
                "evidence_max_season": audit.get("crosswalk_evidence_max_season"),
            }
        )
        parts.append(base.drop(columns="pitcher_id"))

    cache = pd.concat(parts, ignore_index=True).sort_values("row_id")
    order = train[["row_id"]].copy()
    cache = order.merge(cache, on="row_id", how="left", validate="one_to_one")
    if not cache["season"].equals(train["season"]):
        raise AssertionError("Trackman cache season mismatch")
    cache_path = WORK_DIR / "trackman500_asof_train.parquet"
    cache.to_parquet(cache_path, index=False)

    final_lookup, final_audit = load_cutoff(2025, columns)
    final_lookup["tm500_available"] = 1
    final_lookup["tm500_unavailable"] = 0
    final_lookup.to_parquet(WORK_DIR / "trackman500_lookup_2025.parquet", index=False)

    manifest = {
        "rows": len(cache),
        "columns": len(cache.columns),
        "feature_columns": columns + ["tm500_available", "tm500_unavailable"],
        "min_trackman_season_pitches": MIN_PITCHES,
        "strict_asof_rule": "main row season S uses Trackman/crosswalk evidence from season < S",
        "season_audit": season_audit,
        "final_2025_audit": final_audit,
    }
    (WORK_DIR / "trackman500_asof_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Trackman 500구 strict as-of 감사",
        "",
        "- 각 메인 행의 시즌 S에는 S보다 이전 시즌 Trackman만 사용한다.",
        "- 투수-시즌별 Trackman 투구 수가 500개 이상인 시즌만 통계에 포함한다.",
        "- 2019 학습행은 이전 Trackman이 없으므로 전부 미사용 처리한다.",
        "- 2024 검증행의 최대 증거 시즌은 2023이다.",
        "- 최종 2025 추론 lookup은 2019~2024 Trackman을 사용한다.",
        "",
        pd.DataFrame(season_audit).to_markdown(index=False, floatfmt=".6f"),
        "",
        f"피처 수: {len(columns) + 2}",
        "",
        "원본 수치와 crosswalk 품질값은 결측을 유지하며, 별도 available 플래그로 구분한다.",
    ]
    (WORK_DIR / "TRACKMAN500_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(cache_path)


if __name__ == "__main__":
    main()
