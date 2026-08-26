"""팀 결합 공용 -- 구간별 블렌드 가중치 계산.

cowork/hw/compute_blend_weights.py (3-way 때 처음 작성)의 후속 버전. 하드코딩된
MEMBERS 대신 cowork/*/val2024_pred.csv 가 있는 폴더를 자동으로 찾는다 -- 새
멤버가 파일을 추가하면 다음 실행부터 자동으로 들어간다.

방법: Val2024(train.csv season==2024) 실제 라벨로 w* = M^-1 A 를 투수표본수
3구간(asof_pitcher_n 기준)별로 각각 계산한다.

    D = [p_m - r for m in members]   (r = 그 구간의 실제 성공률 평균)
    M = D.T @ D / n                   (예측 간 공분산)
    A = D.T @ (y - r) / n             (예측과 오차의 공분산)
    w* = M^-1 A                       (닫힌 형태 최적해)

구간 경계 [-1, 200, 2000, inf] 는 hw 라인 오프셋에서 실LB +8.84로 검증된 경계를
재사용.

리더보드 미참조 (RULES.md §2 준수) -- Val2024 실제 라벨만 사용.

★ 주의 (4-way에서 확인된 함정): 멤버 수 x 구간 수만큼 파라미터가 늘어난다.
멤버 간 상관이 0.9 이상으로 높으면 구간별 무제약 최적화가 Val2024 노이즈에
과적합될 수 있다 (실측: 4-way 구간별 Val2024 909.18 -> 실LB 1068.42 로, cw+sj
2-way 전역 블렌드의 실LB 1072보다 낮았음). 멤버가 5명(hw/sj/yn/cw/ye)이 되면
이 위험이 더 커지므로, 아래 두 가지를 항상 같이 본다:
    1. NNLS(음수 미허용) 버전과 비교 -- 무제약과 크게 다르면 과적합 의심
    2. val2022_pred.csv 가 있으면 그 가중치를 2022에 그대로 적용해(재학습 아님)
       방향이 맞는지 확인

출력: blend_weights.json, val2024_oof_<members>.csv

실행 (저장소 어디서나):
    python cowork/combine/compute_blend_weights.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import nnls
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
OUT_DIR = Path(__file__).resolve().parent
TARGET = "control_success"

BUCKET_EDGES = [-1, 200, 2000, float("inf")]
BUCKET_LABELS = ["low", "mid", "high"]


def discover_members():
    members = []
    for d in sorted((REPO / "cowork").iterdir()):
        if d.is_dir() and (d / "val2024_pred.csv").exists():
            members.append(d.name)
    return members


def solve_weights(sub: pd.DataFrame, cols: list[str]):
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


def solve_weights_nnls(sub: pd.DataFrame, cols: list[str]):
    if not HAS_SCIPY:
        return None, None
    y = sub[TARGET].to_numpy()
    r = y.mean()
    D = sub[cols].to_numpy() - r
    w, _ = nnls(D, y - r)
    return w, r


def bss(y, p):
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 0, 1)
    r = y.mean()
    u = r * (1 - r)
    return max(0.0, 100000 * (1 - np.mean((y - p) ** 2) / u)) if u > 0 else 0.0


def main():
    members = discover_members()
    print(f"발견된 멤버 ({len(members)}명): {members}")
    if len(members) < 2:
        print("val2024_pred.csv 가 2개 미만이라 계산 불가. cowork/<initial>/val2024_pred.csv 확인.")
        return

    raw = pd.read_csv(DATA_DIR / "train.csv")
    val = raw[raw.season == 2024][["row_id", "asof_pitcher_n", TARGET]].copy()

    df = val
    for m in members:
        pred = pd.read_csv(REPO / "cowork" / m / "val2024_pred.csv")
        pred = pred.rename(columns={TARGET: f"p_{m}"})[["row_id", f"p_{m}"]]
        df = df.merge(pred, on="row_id", how="inner")
    print(f"병합된 행수: {len(df)} (Val2024 전체 {len(val)})")

    cols = [f"p_{m}" for m in members]

    oof_cols = ["row_id", "asof_pitcher_n", TARGET] + cols
    oof = df[oof_cols].rename(columns={TARGET: "y"})
    oof_name = f"val2024_oof_{'_'.join(members)}.csv"
    oof.to_csv(OUT_DIR / oof_name, index=False)
    print(f"저장: {oof_name} ({len(oof)}행)")

    df["bucket"] = pd.cut(df["asof_pitcher_n"].fillna(0), BUCKET_EDGES, labels=BUCKET_LABELS)

    out = {
        "members": members,
        "bucket_edges": [None if e == float("inf") else e for e in BUCKET_EDGES],
        "bucket_labels": BUCKET_LABELS,
        "buckets": {},
    }
    print("\n구간별 가중치 (무제약 w*=M^-1A):")
    for label in BUCKET_LABELS:
        sub = df[df["bucket"] == label]
        w, r = solve_weights(sub, cols)
        out["buckets"][label] = {
            "n": int(len(sub)), "r": float(r),
            "w": {m: float(x) for m, x in zip(members, w)},
        }
        print(f"  {label:5s} n={len(sub):6d} r={r:.4f}  "
              + "  ".join(f"{m}{x:+.4f}" for m, x in zip(members, w)))

    if HAS_SCIPY:
        print("\n구간별 가중치 (NNLS, 음수 미허용 -- 과적합 비교용):")
        for label in BUCKET_LABELS:
            sub = df[df["bucket"] == label]
            w, r = solve_weights_nnls(sub, cols)
            print(f"  {label:5s} " + "  ".join(f"{m}{x:+.4f}" for m, x in zip(members, w)))
    else:
        print("\n(scipy 없음 -- NNLS 비교 스킵. `pip install scipy` 후 재실행 권장)")

    weights_path = OUT_DIR / "blend_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {weights_path}")

    w_all, r_all = solve_weights(df, cols)
    pred_single = r_all + (df[cols].to_numpy() - r_all) @ w_all
    pred_seg = np.zeros(len(df))
    for label in BUCKET_LABELS:
        mask = (df["bucket"] == label).to_numpy()
        spec = out["buckets"][label]
        r = spec["r"]
        w = np.array([spec["w"][m] for m in members])
        pred_seg[mask] = r + (df.loc[mask, cols].to_numpy() - r) @ w

    print(f"\n최고 단독: {max(bss(df[TARGET], df[c]) for c in cols):.2f}")
    print(f"단일 블렌드: {bss(df[TARGET], pred_single):.2f}")
    print(f"구간별 블렌드: {bss(df[TARGET], pred_seg):.2f}")


if __name__ == "__main__":
    main()
