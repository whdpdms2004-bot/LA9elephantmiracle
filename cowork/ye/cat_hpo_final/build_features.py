# -*- coding: utf-8 -*-
"""챔피언 cb 의 176피처(X176)를 원본 train.csv 에서 처음부터 재구성한다.

`run_arm.py` 가 저장소에 없어서(sj 로컬에만 있음) 그 파이프라인을 직접 못 돌린다.
대신 실제 존재하는 조각들을 그대로 이어붙인다 — 새로 설계한 게 아니라 전부 기존
코드다:

  cowork/cw/v17/src/common.py          build_features (X72) + build_encodings/encode_rows (+4)
  cowork/cw/v17/src/season_form.py     build_training_features (+8)  -> X80 (엄밀히는 X~79/80)
  cowork/cw/v17/src/build_v13.py       TrackMan(55)+count(27)+role(5)+missing(1)=88  -> X168
  performance_tracking/models/sj_stdmlp/atoms.py   id_freq (+8, fold 마다 다시 계산) -> X176

전부 as-of(그 행 시즌 이전 데이터만) 라서 시즌 걸치는 정보 유입이 없다. id_freq 만
fold(학습 컷오프)에 따라 값이 달라지므로 fold 별로 따로 만든다.

산출물: cowork/ye/cat_hpo_final/work/X168.npy · y.npy · season.npy · meta.json
        + fold 별 idfreq8_<fold>.npy (fold in 2022,2023,2024)

실행 (오래 걸림 — TrackMan 3.5억행 처리):
    python build_features.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CW_SRC = REPO / "cowork" / "cw" / "v17" / "src"
CW_ASSETS = REPO / "cowork" / "cw" / "v17" / "assets"
ATOMS_DIR = REPO / "performance_tracking" / "models" / "sj_stdmlp"
DATA = REPO / "data"
WORK = HERE / "work"
WORK.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CW_SRC))
sys.path.insert(0, str(ATOMS_DIR))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(m):
    print(m, flush=True)


def main():
    import common as C
    import season_form as SF
    import atoms as A

    t0 = time.time()
    log("train.csv 읽는 중...")
    df = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    df.columns = [c.strip("﻿") for c in df.columns]
    season = df["season"].to_numpy()
    y = df["control_success"].to_numpy(dtype=np.float64)
    log(f"  {len(df):,}행  ({time.time()-t0:.0f}s)")

    # ── X72 ──────────────────────────────────────────────────────────
    log("\n[1/5] common.build_features -> X72")
    X72, names72 = C.build_features(df)
    log(f"  shape={X72.shape}  ({time.time()-t0:.0f}s)")

    # ── + enc_* 4열 (walk-forward 플래툰 스플릿) ────────────────────
    log("\n[2/5] 플래툰 스플릿 인코딩 (시즌별 walk-forward)")
    pid = df["pitcher_id"].to_numpy(np.int64)
    bid = df["batter_id"].to_numpy(np.int64)
    phand = df["pitcher_hand"].to_numpy(np.int64)
    bhand = df["batter_hand"].to_numpy(np.int64)
    enc4 = np.zeros((len(df), 4), dtype=np.float32)
    for s in sorted(set(season.tolist())):
        hist = season < s
        cur = season == s
        if hist.sum() == 0:
            continue
        enc = C.build_encodings(pid[hist], bid[hist], phand[hist], bhand[hist], y[hist])
        enc4[cur] = C.encode_rows(pid[cur], bid[cur], phand[cur], bhand[cur], enc)
    X_enc = np.concatenate([X72, enc4], axis=1).astype(np.float32)
    names_enc = names72 + C.ENC_NAMES
    log(f"  shape={X_enc.shape}  ({time.time()-t0:.0f}s)")

    # ── + season_form 8열 -> X80 ─────────────────────────────────────
    log("\n[3/5] season_form (시즌폼 8열)")
    sf8 = SF.build_training_features(df)
    X80 = np.concatenate([X_enc, sf8], axis=1).astype(np.float32)
    names80 = names_enc + SF.NAMES
    log(f"  shape={X80.shape}  ({time.time()-t0:.0f}s)")
    del X72, X_enc, enc4, sf8

    # ── + TrackMan/count/role/missing 88열 -> X168 ───────────────────
    log("\n[4/5] 도메인 블록 (TrackMan 55 + count 27 + role 5 + missing 1)")
    import build_v13 as BV

    TM_COLS = ["pitcher_trackman_id", "season", "pitch_type_group", "rel_speed", "spin_rate",
               "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side"]
    mp = pd.read_csv(CW_ASSETS / "pitcher_id_map2.csv")
    m = dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))
    tm = pd.read_csv(DATA / "trackman_history.csv", usecols=TM_COLS)
    tm["pid"] = tm.pitcher_trackman_id.map(m)
    GROUPS = ["fastball", "breaking", "offspeed"]
    tm = tm[tm.pid.notna() & tm.pitch_type_group.isin(GROUPS)]
    tm["pid"] = tm.pid.astype(np.int64)
    PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
            "extension", "rel_height", "rel_side"]
    parts = []
    for g in GROUPS:
        sub = tm[tm.pitch_type_group == g]
        a = sub.groupby(["pid", "season"])[PHYS].agg(["mean", "std"])
        a.columns = ["%s_%s_%s" % (g[:2], c, s) for c, s in a.columns]
        a["%s_n" % g[:2]] = sub.groupby(["pid", "season"]).size()
        parts.append(a)
    T = pd.concat(parts, axis=1)
    for c in ("rel_height", "rel_side", "extension", "spin_rate", "rel_speed"):
        T["d_fb_bk_%s" % c] = T["fa_%s_mean" % c] - T["br_%s_mean" % c]
        T["d_fb_of_%s" % c] = T["fa_%s_mean" % c] - T["of_%s_mean" % c]
    T = T.reset_index()
    tm_cols = [c for c in T.columns if c not in ("pid", "season")]
    ids = df.pitcher_id.to_numpy(np.int64)
    T2 = BV.lag_lookup(T, ids, season, tm_cols).astype(np.float32)
    have = np.isfinite(T2[:, 0])
    log(f"  TrackMan {len(tm_cols)}피처 | 전체 커버 {100*have.mean():.1f}%  ({time.time()-t0:.0f}s)")

    C2 = BV.build_count(df)
    R = BV.build_role(df, ids, season)
    M = have.astype(np.float32)[:, None]
    ADD = np.concatenate([T2, C2, R, M], axis=1).astype(np.float32)
    names_add = tm_cols + [f"cnt_{i}" for i in range(C2.shape[1])] + \
        ["inn_mean", "inn_std", "inn_min", "p1_ratio", "p_season"] + ["tm_missing"]
    log(f"  도메인 블록 {ADD.shape[1]}피처 (T2 {T2.shape[1]} + C2 {C2.shape[1]} + R {R.shape[1]} + M 1)")

    X168 = np.concatenate([X80, ADD], axis=1).astype(np.float32)
    names168 = names80 + names_add
    log(f"  shape={X168.shape}  ({time.time()-t0:.0f}s)")
    del X80, ADD, T2, C2, R, M

    np.save(WORK / "X168.npy", X168)
    np.save(WORK / "y.npy", y)
    np.save(WORK / "season.npy", season)
    json.dump({"names168": names168}, open(WORK / "meta.json", "w"), ensure_ascii=False)
    log(f"  저장: work/X168.npy {X168.nbytes/1e6:.0f}MB")

    # ── id_freq 8열 — fold(2022/2023/2024) 별로 따로 ─────────────────
    log("\n[5/5] id_freq (fold별)")
    for fold in (2022, 2023, 2024):
        tr = season < fold
        F = A.Frame(X168, names168, tr)
        idf = A.id_freq(F, fold)
        cols = np.column_stack([idf[k] for k in sorted(idf.keys())]).astype(np.float32)
        np.save(WORK / f"idfreq8_{fold}.npy", cols)
        log(f"  fold{fold}  열={sorted(idf.keys())}  shape={cols.shape}  ({time.time()-t0:.0f}s)")
    json.dump({"names168": names168, "idfreq_names": sorted(A.id_freq(
        A.Frame(X168, names168, season < 2024), 2024).keys())},
        open(WORK / "meta.json", "w"), ensure_ascii=False)

    log(f"\n총 {(time.time()-t0)/60:.1f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
