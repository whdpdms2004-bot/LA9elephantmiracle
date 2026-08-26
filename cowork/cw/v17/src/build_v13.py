# -*- coding: utf-8 -*-
"""v13 통합 피처셋 구축 + 5시드 검증.

80피처(원본 72 + 시즌폼 8) 에 도메인 3블록을 붙여 168피처를 만든다.

  T2  55  TrackMan 구종별 릴리스 일관성 (직구/변화구/체인지업 각각의 표준편차)
          + 구종 간 릴리스 차이.  pitcher_id_map2.csv(605명) 사용
  C2  27  볼카운트 12종 개별 + 투수 볼성향과의 곱 (3-1 배팅찬스 등)
  R    5  선발/불펜/스윙맨 (등판 이닝 분포)
  M    1  TrackMan 프로파일 결측 표시

프로파일은 전부 **직전 시즌까지** 로만 만든다 (s년 행 ← s년 미만).
2025 추론 때 2024년까지로 만드는 것과 규칙이 같다.
학습 데이터만 쓰고, 각 행은 자기 pitcher_id 로 조회만 한다. 규정 안전.

검증 결과 (5시드, verify_domain2.py)
    val2024  759.2 → 784.0  (+3.3%, t=3.1)
    val2022 2271.7 → 2315.9 (+2.0%, t=3.2)

실행:
    python build_v13.py --gpu        # 약 6분
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PITCH_DATA_DIR", os.path.join(HERE, "..", "open", "data"))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))

TM_COLS = ["pitcher_trackman_id", "season", "pitch_type_group", "rel_speed", "spin_rate",
           "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side"]
TR_COLS = ["season", "pitcher_id", "inning", "balls_before", "strikes_before",
           "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "control_success"]
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side"]
GROUPS = ["fastball", "breaking", "offspeed"]


def build_trackman(mapfile):
    mp = pd.read_csv(os.path.join(HERE, mapfile))
    m = dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))
    tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), usecols=TM_COLS)
    tm["pid"] = tm.pitcher_trackman_id.map(m)
    tm = tm[tm.pid.notna() & tm.pitch_type_group.isin(GROUPS)]
    tm["pid"] = tm.pid.astype(np.int64)
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
    return T, [c for c in T.columns if c not in ("pid", "season")]


def lag_lookup(tab, ids, seasons, cols):
    """s년 행 ← s년 미만 시즌의 프로파일 (누적 평균)."""
    out = np.full((len(ids), len(cols)), np.nan)
    for s in sorted(set(seasons.tolist())):
        h = tab[tab.season < s]
        if len(h) == 0:
            continue
        g = h.groupby("pid")[cols].mean()
        idx = g.index.to_numpy(); v = g.to_numpy()
        m = seasons == s
        pos = np.clip(np.searchsorted(idx, ids[m]), 0, max(len(idx) - 1, 0))
        hit = idx[pos] == ids[m]
        vv = v[pos].copy(); vv[~hit] = np.nan
        out[m] = vv
    return out


def build_count(df):
    b = df.balls_before.to_numpy(); s = df.strikes_before.to_numpy()
    br = np.nan_to_num(df.asof_pitcher_ball_rate.to_numpy(dtype=float), nan=0.0)
    sr = np.nan_to_num(df.asof_pitcher_strike_rate.to_numpy(dtype=float), nan=0.0)
    tend = br - sr
    cols = []
    for bb in range(4):
        for ss in range(3):
            k = ((b == bb) & (s == ss)).astype(np.float32)
            cols += [k, k * tend]
    adv = (b - s).astype(np.float32)
    cols += [adv, adv * tend, ((b == 3) & (s == 1)).astype(np.float32) * tend]
    return np.column_stack(cols).astype(np.float32)


def build_role(df, ids, sea):
    g = df.groupby(["pitcher_id", "season"])
    t = pd.DataFrame({"inn_mean": g.inning.mean(), "inn_std": g.inning.std(),
                      "inn_min": g.inning.min(),
                      "p1_ratio": g.inning.apply(lambda v: (v <= 2).mean()),
                      "p_season": g.size()}).reset_index().rename(
        columns={"pitcher_id": "pid"})
    cols = ["inn_mean", "inn_std", "inn_min", "p1_ratio", "p_season"]
    return lag_lookup(t, ids, sea, cols).astype(np.float32)


def calib(p, yv):
    r = yv.mean(); U = r * (1 - r); c1 = np.log(r / (1 - r))
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return max(1e5 * (1 - ((1 / (1 + np.exp(-(k * (z - z.mean()) + c1))) - yv) ** 2).mean() / U)
               for k in np.arange(0.2, 1.55, 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--map", default="pitcher_id_map2.csv")
    a = ap.parse_args()
    from catboost import CatBoostClassifier, Pool
    dev = dict(task_type="GPU", devices="0", border_count=128) if a.gpu else {}

    t0 = time.time()
    print("train.csv 읽는 중...", flush=True)
    df = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=TR_COLS, encoding="utf-8-sig")
    ids = df.pitcher_id.to_numpy(np.int64); sea = df.season.to_numpy()

    print("TrackMan 프로파일 (%s)..." % a.map, flush=True)
    tab, cols = build_trackman(a.map)
    T2 = lag_lookup(tab, ids, sea, cols).astype(np.float32)
    have = np.isfinite(T2[:, 0])
    print("  %d피처 | 전체 커버 %.1f%% | 2024년 행만 보면 %.1f%%"
          % (len(cols), 100 * have.mean(), 100 * have[sea == 2024].mean()), flush=True)
    C2 = build_count(df)
    R = build_role(df, ids, sea)
    M = have.astype(np.float32)[:, None]
    ADD = np.concatenate([T2, C2, R, M], axis=1).astype(np.float32)
    print("  도메인 블록 %d피처 (T2 %d + C2 %d + R %d + 결측표시 1)"
          % (ADD.shape[1], T2.shape[1], C2.shape[1], R.shape[1]), flush=True)

    X80 = np.load(os.path.join(WORK, "X80.npy"), mmap_mode="r")
    y = np.load(os.path.join(WORK, "y.npy"))
    season = np.load(os.path.join(WORK, "season.npy"))
    assert len(X80) == len(ADD)

    out = os.path.join(WORK, "X168.npy")
    Z = np.empty((len(X80), X80.shape[1] + ADD.shape[1]), dtype=np.float32)
    for s in range(0, len(X80), 200000):
        e = min(len(X80), s + 200000)
        Z[s:e, :X80.shape[1]] = X80[s:e]; Z[s:e, X80.shape[1]:] = ADD[s:e]
    np.save(out, Z)
    np.save(os.path.join(WORK, "domain_v13.npy"), ADD)
    print("  저장: _work/X168.npy  shape=%s (%.0f MB)" % (Z.shape, Z.nbytes / 1e6), flush=True)

    def fit(Xt, ytr, Xv, yv, sd):
        m = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.06, l2_leaf_reg=6.0,
                               loss_function="Logloss", random_seed=sd, verbose=0,
                               allow_writing_files=False, thread_count=-1, **dev)
        m.fit(Pool(Xt, ytr)); s = calib(m.predict_proba(Pool(Xv))[:, 1], yv); del m
        return s

    print("\n5시드 검증", flush=True)
    res = {}
    for lb, XX in (("80피처", X80), ("168피처", Z)):
        for val in (2024, 2022):
            tri = np.where(season <= val - 1)[0]; va = np.where(season == val)[0]
            yv = y[va].astype(float)
            Xt = np.asarray(XX[tri]); Xv = np.asarray(XX[va])
            ss = [fit(Xt, y[tri], Xv, yv, sd) for sd in range(a.seeds)]
            res[(lb, val)] = (float(np.mean(ss)), float(np.std(ss, ddof=1)))
            print("  %-8s val%d  %8.1f ± %4.1f  (%.0f초)"
                  % (lb, val, res[(lb, val)][0], res[(lb, val)][1], time.time() - t0),
                  flush=True)
            del Xt, Xv

    print()
    print("=" * 64)
    print("[BUILD v13]")
    print("=" * 64)
    gains = []
    for val in (2024, 2022):
        b, sb = res[("80피처", val)]; e, se = res[("168피처", val)]
        se_diff = np.sqrt(sb ** 2 + se ** 2) / np.sqrt(a.seeds)
        g = e / b - 1; t = (e - b) / se_diff
        gains.append(g)
        print("  val %d : %8.1f → %8.1f   %+6.2f%%   t = %.1f %s"
              % (val, b, e, g * 100, t, "★" if t > 2 else ""))
    mn = min(gains)
    print()
    print("  min 이득 %+.2f%%  →  LB 투영 %.0f  (950 기준)" % (mn * 100, 950 * (1 + mn)))
    print("=" * 64)
    print("총 소요 %.1f분" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
