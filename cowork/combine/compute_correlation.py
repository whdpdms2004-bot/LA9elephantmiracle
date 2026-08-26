"""팀 결합 공용 -- cowork/*/val2024_pred.csv (있으면 val2022_pred.csv도) 를
자동으로 찾아서 상관행렬 + 멤버별 단독 BSS를 출력한다.

새 멤버가 cowork/<initial>/val2024_pred.csv 를 추가하면 다음 실행부터 자동으로
잡힌다 (MEMBERS 하드코딩 없음).

스펙: cowork/combine/README.md 참고 (fit<val_season, val==val_season, 오프셋
미적용, 리더보드 미참조).

실행 (저장소 어디서나):
    python cowork/combine/compute_correlation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
TARGET = "control_success"
YEARS = {2024: "val2024_pred.csv", 2022: "val2022_pred.csv"}


def bss(y, p):
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 0, 1)
    r = y.mean()
    u = r * (1 - r)
    return max(0.0, 100000 * (1 - np.mean((y - p) ** 2) / u)) if u > 0 else 0.0


def discover_members(filename):
    members = []
    for d in sorted((REPO / "cowork").iterdir()):
        if d.is_dir() and (d / filename).exists():
            members.append(d.name)
    return members


def main():
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["row_id", "season", TARGET])

    for year, filename in YEARS.items():
        members = discover_members(filename)
        if len(members) < 2:
            print(f"\n=== {year} ===  파일 {len(members)}개뿐 (2개 이상 필요) -- 스킵: {members}")
            continue

        val = train[train.season == year][["row_id", TARGET]].copy()
        df = val
        for m in members:
            pred = pd.read_csv(REPO / "cowork" / m / filename)
            pred = pred.rename(columns={TARGET: f"p_{m}"})[["row_id", f"p_{m}"]]
            df = df.merge(pred, on="row_id", how="inner")

        cols = [f"p_{m}" for m in members]
        print(f"\n=== {year} (병합 {len(df)}행, 멤버: {members}) ===")
        print("단독 BSS:")
        for m, c in zip(members, cols):
            print(f"  {m:4s} {bss(df[TARGET], df[c]):8.2f}")
        print("상관행렬:")
        print(df[cols].corr().round(4).to_string())


if __name__ == "__main__":
    main()
