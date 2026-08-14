from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import sys


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "experiment" / "model_optimization"
WORK = MODEL_DIR / "pitcher_cluster_matchup"
BASE_ZIP = ROOT / "submit" / "2026-08-12" / "submit_007.zip"
BASE_MODEL_DIR = WORK / "submit007_check" / "model"
FINAL_DIR = WORK / "final" / "robust_matchup_v1"
INFERENCE_SCRIPT = Path(__file__).resolve().parent / "submission_script_matchup.py"
OUTPUT_DIR = ROOT / "submit" / "2026-08-12"
DESTINATION = OUTPUT_DIR / "submit_012.zip"

sys.path.insert(0, str(MODEL_DIR))
from build_final_two_submissions import smoke_test, write_zip  # noqa: E402


def digest(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def main():
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite: {DESTINATION}")
    with ZipFile(BASE_ZIP) as archive:
        base_metadata = json.loads(archive.read("model/metadata.json"))
        requirements = archive.read("requirements.txt").decode("utf-8")
    metadata = deepcopy(base_metadata)
    metadata["version"] = 6
    metadata["track"] = "success_reverse_batter_matchup_robust"
    metadata["outer_blend"]["insight_weight"] = 0.598
    metadata["outer_blend"]["base_weight"] = 0.402
    metadata["matchup_correction"] = {
        "pitcher_lookup_file": "pitcher_lookup_2025.csv",
        "batter_lookup_file": "batter_lookup_2025.csv",
        "pair_table_file": "pair_table_2025.csv",
        "ridge_file": "ridge_correction.json",
        "correction_scale": 0.25,
        "pair_smoothing": 1000,
        "recency_half_life": 1.0,
        "ridge_alpha": 10.0,
    }
    metadata["reverse_matchup_correction"] = {
        "pitcher_lookup_file": "pitcher_lookup_2025.csv",
        "batter_lookup_file": "reverse_batter_lookup_2025.csv",
        "pair_table_file": "reverse_batter_pair_table_2025.csv",
        "ridge_file": "reverse_batter_ridge_correction.json",
        "correction_scale": 0.65,
        "center_mode": "season x pitcher_hand x batter_hand x count_state",
        "batter_algorithm": "kmeans",
        "batter_k_by_hand": {"left": 4, "right": 6},
        "pair_smoothing": 1000,
        "recency_half_life": 1.0,
        "ridge_alpha": 10000.0,
    }
    metadata["dual_matchup_validation"] = {
        "selection": "strict rolling: 2022->2023 and 2023->2024",
        "trackman_min_season_pitches": 500,
        "success_scale": 0.25,
        "reverse_scale": 0.65,
        "f23_delta_brier": -1.0459047631750096e-05,
        "f24_delta_brier": -3.0296951689046114e-05,
        "single_bss_2024": 796.6849746927073,
        "outer_insight_weight": 0.598,
        "outer_blend_bss_2024": 810.2571506239276,
    }
    artifacts = {
        path for path in BASE_MODEL_DIR.iterdir()
        if path.is_file() and path.name != "metadata.json"
    }
    artifacts.update({
        FINAL_DIR / "pitcher_lookup_2025.csv",
        FINAL_DIR / "batter_lookup_2025.csv",
        FINAL_DIR / "pair_table_2025.csv",
        FINAL_DIR / "ridge_correction.json",
        FINAL_DIR / "reverse_batter_lookup_2025.csv",
        FINAL_DIR / "reverse_batter_pair_table_2025.csv",
        FINAL_DIR / "reverse_batter_ridge_correction.json",
    })
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")
    write_zip(
        DESTINATION,
        metadata,
        artifacts,
        requirements,
        script_path=INFERENCE_SCRIPT,
    )
    elapsed = smoke_test(DESTINATION)
    with ZipFile(DESTINATION) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        names = archive.namelist()
        if names[0] != "model/":
            raise RuntimeError("model/ is not the first ZIP entry")
    record = {
        "filename": DESTINATION.name,
        "filename_length": len(DESTINATION.name),
        "size_bytes": DESTINATION.stat().st_size,
        "sha256": digest(DESTINATION),
        "smoke_rows": 245789,
        "inference_sec": elapsed,
        "f23_delta_brier": -1.0459047631750096e-05,
        "f24_delta_brier": -3.0296951689046114e-05,
        "single_bss_2024": 796.6849746927073,
        "blend_bss_2024": 810.2571506239276,
    }
    log_path = OUTPUT_DIR / "SUBMISSION_LOG.md"
    existing = log_path.read_text(encoding="utf-8")
    marker = "## reverse 전용 타자 군집 상성 012"
    if marker in existing:
        raise RuntimeError("Submission log section already exists")
    section = [
        "",
        marker,
        "",
        f"생성 시각: {datetime.now().astimezone().isoformat()}",
        "",
        "| 파일 | 성공 보정 | reverse 보정 | 제구 가중치 | 2023 ΔBrier | 2024 ΔBrier | 2024 혼합 BSS | 추론 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| `{record['filename']}` | 0.25 | 0.65 | 0.598 | "
        f"{record['f23_delta_brier']:+.8f} | {record['f24_delta_brier']:+.8f} | "
        f"{record['blend_bss_2024']:.3f} | {elapsed:.1f}초 |",
        "",
        "- reverse는 시즌·투수손·타자손·볼카운트별 평균을 제거한 뒤 투수유형×타자유형으로 집계.",
        "- reverse 전용 타자 군집은 좌 4/우 6 KMeans, smoothing=1000, half-life=1, Ridge alpha=10000.",
        "- 성공 상성과 reverse 상성은 별도 Ridge로 학습해 각각 0.25/0.65만 반영.",
        "- 2024 검증에는 2023년 이하만, 2025 최종 lookup에는 2024년 이하만 사용.",
        "- ZIP 루트 구조, CRC, 245,789행 로컬 추론, 확률 범위 검사 통과.",
        "",
        f"SHA256: `{record['sha256']}`",
    ]
    log_path.write_text(existing.rstrip() + "\n" + "\n".join(section) + "\n", encoding="utf-8")
    manifest_path = WORK / "final" / "reverse_batter_submission_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "created_at": datetime.now().astimezone().isoformat(),
            "base_submission": str(BASE_ZIP.relative_to(ROOT)),
            "record": record,
            "metadata": {
                "matchup_correction": metadata["matchup_correction"],
                "reverse_matchup_correction": metadata["reverse_matchup_correction"],
                "dual_matchup_validation": metadata["dual_matchup_validation"],
            },
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
