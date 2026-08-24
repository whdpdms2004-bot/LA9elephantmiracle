"""팀 결합용 구간별 블렌드 가중치 계산 -- team_blend_build/model/blend_weights.json 산출 코드.

방법: Val2024(train.csv season==2024) 실제 라벨로 w* = M^-1 A를 투수표본수
3구간(asof_pitcher_n 기준)별로 각각 계산한다.

    D = [p_hw - r, p_sj - r, p_yn - r]   (r = 그 구간의 실제 성공률 평균)
    M = D.T @ D / n                       (예측 간 공분산)
    A = D.T @ (y - r) / n                 (예측과 오차의 공분산)
    w* = M^-1 A                           (닫힌 형태 최적해)

구간 경계 [-1, 200, 2000, inf]는 hw 라인 오프셋에서 이미 실LB +8.84로
검증된 경계를 재사용 (submission_v9_bucketoffset).

리더보드 미참조 (RULES.md §2 준수) -- Val2024 실제 라벨만 사용.

입력: cowork/hw/val2024_pred.csv, cowork/sj/val2024_pred.csv,
      cowork/yn/val2024_pred.csv, data/train.csv
출력: blend_weights.json (team_blend_build/model/ 에 넣는 것과 동일 포맷)
      val2024_oof_hw_sj_yn.csv (row_id, asof_pitcher_n, y, p_hw, p_sj, p_yn)

실행 (저장소 어디서나, __file__ 기준 상대경로라 clone 위치 무관):
    py cowork/hw/compute_blend_weights.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]  # cowork/hw -> cowork -> 저장소 루트
DATA_DIR = REPO / "data"
TARGET = "control_success"

MEMBERS = ["hw", "sj", "yn"]
BUCKET_EDGES = [-1, 200, 2000, float("inf")]
BUCKET_LABELS = ["low", "mid", "high"]


def solve_weights(sub: pd.DataFrame, cols: list[str]):
    """닫힌 형태 w* = M^-1 A. r은 이 구간(sub)의 실제 성공률 평균."""
    y = sub[TARGET].to_numpy()
    r = y.mean()
    D = sub[cols].to_numpy() - r
    M = D.T @ D / len(sub)
    A = D.T @ (y - r) / len(sub)
    try:
        w = np.linalg.solve(M, A)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(M, A, rcond=None)[0]
    return w, r


def main():
    raw = pd.read_csv(DATA_DIR / "train.csv")
    val = raw[raw.season == 2024][["row_id", "asof_pitcher_n", TARGET]].copy()

    df = val
    for m in MEMBERS:
        pred = pd.read_csv(REPO / "cowork" / m / "val2024_pred.csv")
        pred = pred.rename(columns={TARGET: f"p_{m}"})[["row_id", f"p_{m}"]]
        df = df.merge(pred, on="row_id", how="inner")
    print(f"병합된 행수: {len(df)} (Val2024 전체 {len(val)})")

    cols = [f"p_{m}" for m in MEMBERS]

    # ---- OOF 파일 저장 (재현·검증용) ----
    oof_cols = ["row_id", "asof_pitcher_n", TARGET] + cols
    oof = df[oof_cols].rename(columns={TARGET: "y"})
    oof_path = Path(__file__).resolve().parent / "val2024_oof_hw_sj_yn.csv"
    oof.to_csv(oof_path, index=False)
    print(f"저장: {oof_path} ({len(oof)}행, 컬럼: {list(oof.columns)})")

    # ---- 구간별 가중치 계산 ----
    df["bucket"] = pd.cut(df["asof_pitcher_n"].fillna(0), BUCKET_EDGES, labels=BUCKET_LABELS)

    out = {
        "members": MEMBERS,
        "bucket_edges": [None if e == float("inf") else e for e in BUCKET_EDGES],
        "bucket_labels": BUCKET_LABELS,
        "buckets": {},
    }
    for label in BUCKET_LABELS:
        sub = df[df["bucket"] == label]
        w, r = solve_weights(sub, cols)
        out["buckets"][label] = {
            "n": int(len(sub)),
            "r": float(r),
            "w": {m: float(x) for m, x in zip(MEMBERS, w)},
        }
        print(f"  {label:5s} n={len(sub):6d} r={r:.4f}  "
              + "  ".join(f"{m}{x:+.4f}" for m, x in zip(MEMBERS, w)))

    weights_path = Path(__file__).resolve().parent / "blend_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {weights_path}")

    # ---- 검증용 지표: 단일 블렌드 vs 구간별 블렌드 ----
    def bss(y, p):
        y = np.asarray(y)
        p = np.clip(np.asarray(p), 0, 1)
        r = y.mean()
        u = r * (1 - r)
        return max(0.0, 100000 * (1 - np.mean((y - p) ** 2) / u))

    w_all, r_all = solve_weights(df, cols)
    pred_single = r_all + (df[cols].to_numpy() - r_all) @ w_all
    pred_seg = np.zeros(len(df))
    for label in BUCKET_LABELS:
        mask = (df["bucket"] == label).to_numpy()
        spec = out["buckets"][label]
        r = spec["r"]
        w = np.array([spec["w"][m] for m in MEMBERS])
        pred_seg[mask] = r + (df.loc[mask, cols].to_numpy() - r) @ w

    print(f"\n최고 단독: {max(bss(df[TARGET], df[c]) for c in cols):.2f}")
    print(f"단일 블렌드: {bss(df[TARGET], pred_single):.2f}")
    print(f"구간별 블렌드: {bss(df[TARGET], pred_seg):.2f}")


if __name__ == "__main__":
    main()
