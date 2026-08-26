# -*- coding: utf-8 -*-
"""추론용 룩업 테이블 생성 — TrackMan 프로파일 + 역할.

build_v13.py 는 행별 피처만 저장했고 룩업 자체는 안 남겼다.
추론 때는 그 행의 pitcher_id 로 조회해야 하므로 테이블이 필요하다.

만드는 것
    model/domain_lut.npz
        t2_key (n,)      pitcher_id
        t2_val (n, 55)   TrackMan 프로파일 (2019~2024 누적 평균)
        r_key  (m,)
        r_val  (m, 5)    역할 프로파일
        cols_t2 / cols_r 컬럼 이름 (순서 검증용)

검증
    "2023년까지" 룩업을 만들어 2024 행에 적용한 결과가
    X168.npy 의 해당 열과 정확히 같은지 확인한다.
    같으면 추론 경로가 학습 경로와 일치한다는 뜻이다.

규정
    학습 데이터로만 만들고, 추론 때는 그 행의 pitcher_id 로 조회만 한다.
    평가 데이터의 다른 행을 보지 않는다.

실행:
    python build_luts_v13.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PITCH_DATA_DIR", os.path.join(HERE, "..", "open", "data"))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))
MODEL_DIR = os.path.join(HERE, "model")
sys.path.insert(0, HERE)

from build_v13 import build_trackman, lag_lookup, build_count, TR_COLS   # noqa: E402


def role_table(df):
    g = df.groupby(["pitcher_id", "season"])
    t = pd.DataFrame({"inn_mean": g.inning.mean(), "inn_std": g.inning.std(),
                      "inn_min": g.inning.min(),
                      "p1_ratio": g.inning.apply(lambda v: (v <= 2).mean()),
                      "p_season": g.size()}).reset_index().rename(
        columns={"pitcher_id": "pid"})
    return t, ["inn_mean", "inn_std", "inn_min", "p1_ratio", "p_season"]


def collapse(tab, upto, cols):
    """upto 시즌 이하를 투수별 평균으로 접는다 (추론용 단일 룩업)."""
    h = tab[tab.season <= upto]
    g = h.groupby("pid")[cols].mean()
    k = g.index.to_numpy(np.int64)
    o = np.argsort(k)
    return k[o], g.to_numpy(np.float64)[o]


def lookup(ids, keys, vals):
    out = np.full((len(ids), vals.shape[1]), np.nan)
    if len(keys) == 0:
        return out
    pos = np.clip(np.searchsorted(keys, ids), 0, len(keys) - 1)
    hit = keys[pos] == ids
    v = vals[pos].copy(); v[~hit] = np.nan
    return v


def main():
    print("train.csv 읽는 중...", flush=True)
    df = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=TR_COLS, encoding="utf-8-sig")
    ids = df.pitcher_id.to_numpy(np.int64); sea = df.season.to_numpy()

    print("TrackMan 프로파일 표 생성...", flush=True)
    t2_tab, t2_cols = build_trackman("pitcher_id_map2.csv")
    r_tab, r_cols = role_table(df)

    # ── 검증: 2023년까지 룩업을 2024 행에 적용 → X168 과 대조 ──
    print("\n[검증] 2023까지 룩업 → 2024 행 → X168 대조", flush=True)
    X = np.load(os.path.join(WORK, "X168.npy"), mmap_mode="r")
    m24 = sea == 2024
    ref_t2 = np.asarray(X[np.where(m24)[0]][:, 80:80 + 55], dtype=np.float64)
    ref_r = np.asarray(X[np.where(m24)[0]][:, 80 + 55 + 27:80 + 55 + 27 + 5], dtype=np.float64)

    k, v = collapse(t2_tab, 2023, t2_cols)
    got_t2 = lookup(ids[m24], k, v)
    k2, v2 = collapse(r_tab, 2023, r_cols)
    got_r = lookup(ids[m24], k2, v2)

    # X168.npy 는 float32 다. 회전수처럼 2000 대 값은 최소 단위가 2.6e-4 이므로
    # 절대오차 기준은 쓸 수 없다. 상대오차로 본다.
    for nm, a, b in (("T2", ref_t2, got_t2), ("R", ref_r, got_r)):
        both = np.isfinite(a) & np.isfinite(b)
        nan_ok = (np.isnan(a) == np.isnan(b)).all()
        if both.any():
            rel = np.abs(a[both] - b[both]) / (np.abs(a[both]) + 1e-6)
            dr = float(rel.max()); da = float(np.abs(a[both] - b[both]).max())
        else:
            dr = da = 0.0
        print("  %-3s 상대오차 %.2e (절대 %.2e) | 결측패턴 일치 %s | 유효 %.1f%%"
              % (nm, dr, da, "예" if nan_ok else "★아니오", 100 * both.mean()), flush=True)
        if dr > 1e-5 or not nan_ok:
            sys.exit("★ 룩업 적용 결과가 학습 피처와 다릅니다. 중단합니다.")

    # ── 추론용(2024까지) 저장 ──────────────────────────────
    kt, vt = collapse(t2_tab, 2024, t2_cols)
    kr, vr = collapse(r_tab, 2024, r_cols)
    np.savez_compressed(os.path.join(MODEL_DIR, "domain_lut.npz"),
                        t2_key=kt, t2_val=vt, r_key=kr, r_val=vr,
                        cols_t2=np.array(t2_cols), cols_r=np.array(r_cols))
    sz = os.path.getsize(os.path.join(MODEL_DIR, "domain_lut.npz")) / 1024
    cov = np.isfinite(lookup(ids[m24], kt, vt)[:, 0]).mean()
    print("\n저장: model/domain_lut.npz (%.0f KB)" % sz)
    print("  TrackMan 투수 %d명 / 역할 투수 %d명" % (len(kt), len(kr)))
    print("  2024 행 기준 TrackMan 커버리지 %.1f%%  (2025 는 더 높다)" % (100 * cov))
    print("\n다음: script_v13.py 작성 → make_v13.py")


if __name__ == "__main__":
    main()
