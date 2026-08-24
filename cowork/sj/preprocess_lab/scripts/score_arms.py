"""저장된 예측(.npy)을 채점한다. CPU 전용 — GPU 작업과 겹쳐도 안전하다.

순위는 bss_centered 로 매긴다. 이유는 METHOD.md §2.
    bss_raw − bss_centered = "평균 정렬로 번 점수" 이고, 그건 다른 fold 나
    리더보드로 따라가지 않는다.

사용
    python score_arms.py                          # 랩 + 캠페인 산출물 전부
    python score_arms.py --dir <경로> --fold 2024
    python score_arms.py --baseline baseline       # 기준 arm 이름
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
SJ = LAB.parent
REPO = SJ.parents[1]
CAMPAIGN = SJ / "feature_campaign_1000"
MODEL_OPT = SJ / "experiment" / "model_optimization"
for p in (MODEL_OPT, CAMPAIGN, SJ / "claude" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_DIRS = [LAB / "outputs",
                CAMPAIGN / "outputs" / "preprocess_screen",
                CAMPAIGN / "outputs" / "prep_beam"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=None,
                    help="예측 .npy 폴더. 여러 번 줄 수 있다")
    ap.add_argument("--fold", type=int, default=2024)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--out", default=str(LAB / "outputs" / "scores.csv"))
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dir] if args.dir else DEFAULT_DIRS
    print(f"랩       {LAB}")
    print(f"저장소   {REPO}")
    for d in dirs:
        print(f"  탐색   {d}  {'(없음)' if not d.exists() else ''}")

    from run_optuna_enhanced import load_enhanced_frame
    from run_optuna_family import TARGET

    frame, _ = load_enhanced_frame()
    valid = frame["season"].eq(args.fold)
    y = frame.loc[valid, TARGET].to_numpy(np.float64)
    ybar, null = float(y.mean()), float(y.mean() * (1 - y.mean()))
    print(f"{chr(10)}fold {args.fold}   n={len(y):,}   실제 평균 {ybar:.4f}")

    rows, seen = [], set()
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.glob(f"*_{args.fold}.npy")):
            arm = path.name[: -len(f"_{args.fold}.npy")]
            if arm in seen:
                continue
            pred = np.load(path).astype(np.float64)
            if len(pred) != len(y):
                print(f"  ! {arm} 길이 불일치 {len(pred)} vs {len(y)} — 건너뜀")
                continue
            seen.add(arm)
            pred = np.clip(pred, 1e-7, 1 - 1e-7)
            off = float(pred.mean()) - ybar
            ctr = np.clip(pred - off, 1e-7, 1 - 1e-7)
            rows.append({
                "arm": arm,
                "n_atoms": arm.count("+") + 1 if arm != "baseline" else 0,
                "offset": off,
                "bss_raw": 100000 * (1 - float(np.mean((pred - y) ** 2)) / null),
                "bss_centered": 100000 * (1 - float(np.mean((ctr - y) ** 2)) / null),
                "source": d.name,
            })

    if not rows:
        print("채점할 .npy 가 없다.")
        return

    t = pd.DataFrame(rows)
    t["mean_gain"] = t["bss_raw"] - t["bss_centered"]
    if (t["arm"] == args.baseline).any():
        b = t.loc[t["arm"] == args.baseline].iloc[0]
        t["d_centered"] = t["bss_centered"] - b["bss_centered"]
        t["d_raw"] = t["bss_raw"] - b["bss_raw"]
    t = t.sort_values("bss_centered", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)

    print(f"{chr(10)}{'=' * 100}")
    print(f"순위 — bss_centered 기준 (METHOD.md §2)")
    print("=" * 100)
    cols = ["arm", "n_atoms", "bss_centered", "bss_raw", "mean_gain", "offset"]
    if "d_centered" in t:
        cols = ["arm", "n_atoms", "d_centered", "d_raw", "mean_gain", "offset"]
    show = t[cols].copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(4 if c == "offset" else 2)
    print(show.to_string(index=False))

    if "d_centered" in t:
        flip = t[(t["d_centered"] > 0) & (t["d_raw"] < 0)]
        if len(flip):
            print(f"{chr(10)}주의 — raw 로 보면 탈락인데 신호는 양수인 것 "
                  f"({len(flip)}개). METHOD.md §2 의 함정이다.")
            print(flip[["arm", "d_raw", "d_centered", "offset"]]
                  .round(3).to_string(index=False))
    print(f"{chr(10)}saved -> {args.out}")


if __name__ == "__main__":
    main()
