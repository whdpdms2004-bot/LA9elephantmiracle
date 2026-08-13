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

sys.path.insert(0, str(MODEL_DIR))
from build_final_two_submissions import smoke_test, write_zip  # noqa: E402


VARIANTS = [
    {
        "number": 9,
        "name": "matchup_conservative",
        "correction_scale": 0.60,
        "insight_weight": 0.558,
        "single_bss_2024": 789.6837620973929,
        "blend_bss_2024": 805.563750399807,
        "f23_delta_brier": -1.2164894861832476e-05,
        "f24_delta_brier": -1.2807437668987953e-05,
    },
    {
        "number": 10,
        "name": "matchup_aggressive",
        "correction_scale": 1.00,
        "insight_weight": 0.532,
        "single_bss_2024": 786.2753773,
        "blend_bss_2024": 806.4875295791473,
        "f23_delta_brier": -9.960e-06,
        "f24_delta_brier": -4.107e-06,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_base():
    if not BASE_ZIP.is_file():
        raise FileNotFoundError(BASE_ZIP)
    with ZipFile(BASE_ZIP) as archive:
        metadata = json.loads(archive.read("model/metadata.json"))
        requirements = archive.read("requirements.txt").decode("utf-8")
    return metadata, requirements


def package_variant(base_metadata, requirements, variant):
    metadata = deepcopy(base_metadata)
    metadata["version"] = 5
    metadata["track"] = variant["name"]
    metadata["outer_blend"]["insight_weight"] = variant["insight_weight"]
    metadata["outer_blend"]["base_weight"] = 1.0 - variant["insight_weight"]
    metadata["matchup_correction"] = {
        "pitcher_lookup_file": "pitcher_lookup_2025.csv",
        "batter_lookup_file": "batter_lookup_2025.csv",
        "pair_table_file": "pair_table_2025.csv",
        "ridge_file": "ridge_correction.json",
        "correction_scale": variant["correction_scale"],
        "pitcher_representation": "combined_physical_control",
        "pitcher_algorithm": "diag_gmm",
        "pitcher_k_by_hand": {"left": 2, "right": 4},
        "batter_algorithm": "kmeans",
        "batter_k_by_hand": {"left": 3, "right": 4},
        "pair_smoothing": 1000,
        "recency_half_life": 1.0,
        "ridge_alpha": 10.0,
        "trackman_min_season_pitches": 500,
        "frozen_evidence_end_season": 2024,
    }
    metadata["matchup_validation"] = {
        "selection": "strict rolling: 2022->2023 and 2023->2024",
        "f23_delta_brier": variant["f23_delta_brier"],
        "f24_delta_brier": variant["f24_delta_brier"],
        "single_bss_2024": variant["single_bss_2024"],
        "outer_blend_bss_2024": variant["blend_bss_2024"],
        "correction_scale": variant["correction_scale"],
        "insight_weight": variant["insight_weight"],
    }

    artifact_paths = {
        path for path in BASE_MODEL_DIR.iterdir() if path.is_file() and path.name != "metadata.json"
    }
    artifact_paths.update(
        {
            FINAL_DIR / "pitcher_lookup_2025.csv",
            FINAL_DIR / "batter_lookup_2025.csv",
            FINAL_DIR / "pair_table_2025.csv",
            FINAL_DIR / "ridge_correction.json",
        }
    )
    missing = [str(path) for path in artifact_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")

    filename = f"submit_{variant['number']:03d}.zip"
    if len(filename) >= 30:
        raise RuntimeError(f"Filename must be below 30 characters: {filename}")
    destination = OUTPUT_DIR / filename
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    write_zip(
        destination,
        metadata,
        artifact_paths,
        requirements,
        script_path=INFERENCE_SCRIPT,
    )
    inference_sec = smoke_test(destination)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"CRC failed: {destination}")
        names = archive.namelist()
        if names[0] != "model/":
            raise RuntimeError("model/ is not the first ZIP entry")
        top = {name.split("/", 1)[0] for name in names}
        if top != {"model", "script.py", "requirements.txt"}:
            raise RuntimeError(f"Unexpected ZIP roots: {sorted(top)}")
    return {
        **variant,
        "filename": filename,
        "filename_length": len(filename),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "smoke_rows": 245789,
        "inference_sec": inference_sec,
    }


def append_log(records):
    path = OUTPUT_DIR / "SUBMISSION_LOG.md"
    existing = path.read_text(encoding="utf-8") if path.is_file() else "# 제출 기록\n"
    marker = "## 투수 유형 × 타자 유형 매치업 보정 009·010"
    if marker in existing:
        raise RuntimeError("Submission log section already exists")
    lines = [
        "",
        marker,
        "",
        f"생성 시각: {datetime.now().astimezone().isoformat()}",
        "",
        "| 파일 | 성격 | 보정 강도 | 제구 가중치 | 2023 ΔBrier | 2024 ΔBrier | 2024 혼합 BSS | 추론 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| `{row['filename']}` | {row['name']} | {row['correction_scale']:.2f} | "
            f"{row['insight_weight']:.3f} | {row['f23_delta_brier']:+.8f} | "
            f"{row['f24_delta_brier']:+.8f} | {row['blend_bss_2024']:.3f} | "
            f"{row['inference_sec']:.1f}초 |"
        )
    lines.extend(
        [
            "",
            "- 투수 유형: 좌·우 분리, combined(구위+제구) PCA 8차원, diagonal GMM, 좌 2/우 4군집.",
            "- 타자 유형: 타석 손 방향 분리 KMeans, 좌 3/우 4군집. 스위치 타자는 실제 타석 손 방향별로 분리.",
            "- 매치업: 과거 성공률 잔차를 유형 쌍 단위로 집계하고 smoothing=1000, half-life=1 적용.",
            "- 보정: 4개 매치업 변수 Ridge(alpha=10), 2025용 계수는 2024 OOF 잔차로 재학습.",
            "- 누수 방지: 2024 검증은 2023년 이하만 사용. 2025 최종 산출물은 2024년 이하만 사용.",
            "- TrackMan 임베딩: 투수-시즌 500구 이상만 사용. 미충족자는 rookie/control-only 별도 유형.",
            "- 두 ZIP 모두 루트 구조, CRC, 245,789행 로컬 추론, 확률 범위를 통과함.",
            "",
            "### 파일 해시",
            "",
        ]
    )
    for row in records:
        lines.append(f"- `{row['filename']}`: `{row['sha256']}`")
    path.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        destination = OUTPUT_DIR / f"submit_{variant['number']:03d}.zip"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite: {destination}")
    base_metadata, requirements = load_base()
    records = [package_variant(base_metadata, requirements, item) for item in VARIANTS]
    append_log(records)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "base_submission": str(BASE_ZIP.relative_to(ROOT)),
        "records": records,
    }
    path = WORK / "final" / "matchup_submissions_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
