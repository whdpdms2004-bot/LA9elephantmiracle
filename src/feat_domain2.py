# -*- coding: utf-8 -*-
"""도메인 피처 2차 — 1차의 두 가지 오류를 수정.

[T2] 구종별 릴리스 일관성  ← 1차의 결정적 오류 수정
    1차는 구종 구분 없이 rel_height/rel_side 의 표준편차를 냈다. 그런데
        직구      회전 높음, 릴리스 앞
        커브·슬라  회전 높음, 릴리스 뒤
        체인지업   회전 낮음, 릴리스 앞
    이라 구종을 섞으면 표준편차가 커진다. 즉 1차 지표는 "제구 일관성" 이 아니라
    **"구종을 몇 개 던지느냐"** 를 쟀다. 변화구를 잘 섞는 좋은 투수일수록 나쁘게 나온다.

    수정: 구종 그룹 안에서 따로 잰다. 직구의 릴리스가 흔들리는 것 — 그게 커맨드다.
    추가로 구종 간 릴리스 차이도 낸다 (너무 작으면 구종 구분이 안 되고,
    너무 크면 타자가 읽는다).

[C2] 볼카운트 세분화  ← 1차는 "3볼" 로 뭉뚱그렸다
    3-1 은 배팅찬스다. 타자가 노리고 들어오니 투수가 가장 신경 쓴다.
    3-0 은 거의 지켜보므로 존에 넣는다. 3-2 는 승부다. 심리가 전부 다르다.
    12가지 카운트를 개별로 주고, 투수의 볼 성향과 곱한다.

실행:
    python feat_domain2.py --gpu        # 약 6분
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
TR_COLS = ["season", "pitcher_id", "balls_before", "strikes_before", "num_runners_on",
           "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
           "asof_pitcher_success_rate", "control_success"]
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side"]
GROUPS = ["fastball", "breaking", "offspeed"]


def build_trackman2():
    """구종 그룹 안에서 잰 일관성 + 구종 간 릴리스 차이."""
    mp = pd.read_csv(os.path.join(HERE, "pitcher_id_map.csv"))
    m = dict(zip(mp.pitcher_trackman_id, mp.pitcher_id))
    tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), usecols=TM_COLS)
    tm["pid"] = tm.pitcher_trackman_id.map(m)
    tm = tm[tm.pid.notna() & tm.pitch_type_group.isin(GROUPS)]
    tm["pid"] = tm.pid.astype(np.int64)

    parts, names = [], []
    for g in GROUPS:
        sub = tm[tm.pitch_type_group == g]
        a = sub.groupby(["pid", "season"])[PHYS].agg(["mean", "std"])
        a.columns = ["%s_%s_%s" % (g[:2], c, s) for c, s in a.columns]
        a["%s_n" % g[:2]] = sub.groupby(["pid", "season"]).size()
        parts.append(a)
    T = pd.concat(parts, axis=1)

    # 구종 간 릴리스 차이 — 너무 작으면 구종 구분 실패, 너무 크면 타자가 읽는다
    for c in ("rel_height", "rel_side", "extension", "spin_rate", "rel_speed"):
        T["d_fb_bk_%s" % c] = T["fa_%s_mean" % c] - T["br_%s_mean" % c]
        T["d_fb_of_%s" % c] = T["fa_%s_mean" % c] - T["of_%s_mean" % c]
    T = T.reset_index()
    return T, [c for c in T.columns if c not in ("pid", "season")]


def lag_lookup(tab, ids, seasons, cols):
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
        vv = v[pos]; vv[~hit] = np.nan
        out[m] = vv
    return out


def build_count2(df):
    """12가지 볼카운트를 개별로. 3-1(배팅찬스)과 3-0, 3-2 는 심리가 다르다."""
    b = df.balls_before.to_numpy(); s = df.strikes_before.to_numpy()
    br = np.nan_to_num(df.asof_pitcher_ball_rate.to_numpy(dtype=float), nan=0.0)
    sr = np.nan_to_num(df.asof_pitcher_strike_rate.to_numpy(dtype=float), nan=0.0)
    tend = br - sr                                        # 볼 성향
    cols, names = [], []
    for bb in range(4):
        for ss in range(3):
            k = ((b == bb) & (s == ss)).astype(np.float32)
            cols.append(k); names.append("c%d%d" % (bb, ss))
            cols.append(k * tend); names.append("c%d%d_x_tend" % (bb, ss))
    # 타자 유리 / 투수 유리 정도
    adv = (b - s).astype(np.float32)
    cols += [adv, adv * tend, ((b == 3) & (s == 1)).astype(np.float32) * tend]
    return np.column_stack(cols).astype(np.float32)


def calib(p, yv):
    r = yv.mean(); U = r * (1 - r); c1 = np.log(r / (1 - r))
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return max(1e5 * (1 - ((1 / (1 + np.exp(-(k * (z - z.mean()) + c1))) - yv) ** 2).mean() / U)
               for k in np.arange(0.2, 1.55, 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    from catboost import CatBoostClassifier, Pool
    dev = dict(task_type="GPU", devices="0", border_count=128) if a.gpu else {}
    P = dict(iterations=900, depth=6, learning_rate=0.06, l2_leaf_reg=6.0,
             loss_function="Logloss", random_seed=42, verbose=0,
             allow_writing_files=False, thread_count=-1, **dev)

    t0 = time.time()
    print("train.csv 읽는 중...", flush=True)
    df = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=TR_COLS, encoding="utf-8-sig")
    ids = df.pitcher_id.to_numpy(np.int64); sea = df.season.to_numpy()

    print("TrackMan 구종별 프로파일 생성 중...", flush=True)
    tab, cols = build_trackman2()
    T2 = lag_lookup(tab, ids, sea, cols).astype(np.float32)
    print("  %d피처, 매칭 %.1f%% (%.0f초)"
          % (len(cols), 100 * np.isfinite(T2[:, 0]).mean(), time.time() - t0), flush=True)
    C2 = build_count2(df)
    print("카운트 세분화 %d피처" % C2.shape[1], flush=True)
    R = np.load(os.path.join(WORK, "domain_R.npy")) if os.path.exists(
        os.path.join(WORK, "domain_R.npy")) else None

    X = np.load(os.path.join(WORK, "X80.npy"), mmap_mode="r")
    y = np.load(os.path.join(WORK, "y.npy"))
    season = np.load(os.path.join(WORK, "season.npy"))

    G = {"T2": T2, "C2": C2}
    if R is not None:
        G["R"] = R
    combos = [("T2", ["T2"]), ("C2", ["C2"])]
    if R is not None:
        combos.append(("T2+C2+R", ["T2", "C2", "R"]))
    else:
        combos.append(("T2+C2", ["T2", "C2"]))

    base = {}
    for val in (2024, 2022):
        tri = np.where(season <= val - 1)[0]; va = np.where(season == val)[0]
        m = CatBoostClassifier(**P); m.fit(Pool(np.asarray(X[tri]), y[tri]))
        base[val] = calib(m.predict_proba(Pool(np.asarray(X[va])))[:, 1], y[va].astype(float))
        del m
        print("기준선 val%d %8.1f" % (val, base[val]), flush=True)

    print()
    print("%-10s %10s %10s %10s %10s" % ("구성", "val2024", "val2022", "min이득", "LB투영"))
    res = []
    for lb, keys in combos:
        ADD = np.concatenate([G[k] for k in keys], axis=1)
        g = []; t = time.time()
        for val in (2024, 2022):
            tri = np.where(season <= val - 1)[0]; va = np.where(season == val)[0]
            Xt = np.column_stack([np.asarray(X[tri]), ADD[tri]])
            Xv = np.column_stack([np.asarray(X[va]), ADD[va]])
            m = CatBoostClassifier(**P); m.fit(Pool(Xt, y[tri]))
            s = calib(m.predict_proba(Pool(Xv))[:, 1], y[va].astype(float)); del m, Xt, Xv
            g.append(s / base[val] - 1)
        mn = min(g); res.append((mn, lb))
        np.save(os.path.join(WORK, "dom2_%s.npy" % lb.replace("+", "")), ADD)
        print("%-10s %9.2f%% %9.2f%% %9.2f%% %10.0f   %.0f초"
              % (lb, g[0] * 100, g[1] * 100, mn * 100, 950 * (1 + mn), time.time() - t),
              flush=True)

    print()
    print("=" * 62)
    for mn, lb in sorted(res, reverse=True):
        print("  %-10s min %+.2f%%  →  LB %.0f %s"
              % (lb, mn * 100, 950 * (1 + mn), "★ 채택" if mn >= 0.015 else ""))
    print("=" * 62)
    print("총 소요 %.1f분" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
