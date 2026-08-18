# -*- coding: utf-8 -*-
"""TrackMan 투수 매칭 확장 — 커버리지 67.6% → 최대화.

기존 trackman_match.py 의 병목:
    match_pitchers 가 "한 명만 던진 하프이닝" 만 근거로 썼다.
        b.groupby(["gid","hi"]).filter(lambda x: len(x) == 1)
    투수 교체가 있는 이닝을 통째로 버리므로, **이닝 중간에 등판하는 불펜**은
    근거가 거의 안 쌓인다. 792명 중 578명만 붙은 이유다.

확장 방법:
    경기 매칭은 이미 유사도 중앙값 1.0 으로 정확하다. 그렇다면 매칭된 경기 안에서
    **하프이닝별 투구 수가 양쪽이 같으면 순서대로 1:1 대응**시킬 수 있다.
    투수 교체가 있어도 순서는 보존되므로, 교체 이닝도 근거로 쓸 수 있다.

    추가로 argmax 대신 **Hungarian 1:1 할당**을 쓴다. 기존은 여러 pitcher_id 가
    같은 trackman id 로 붙을 수 있었다.

검증:
    매칭에 전혀 쓰지 않은 **투수 좌우 손** 일치율로 확인한다 (기존 576/578).

실행:
    python trackman_match2.py        # 약 3분
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PITCH_DATA_DIR", os.path.join(HERE, "..", "open", "data"))
TEAM = {12: "DOO_BEA", 13: "LG_TWI", 14: "KIW_HER", 15: "LOT_GIA", 16: "KIA_TIG",
        17: "HAN_EAG", 18: "SAM_LIO", 19: "NC_DIN", 20: "KT_WIZ", 21: None}


def load():
    tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                     usecols=["season", "game_month", "game_dayofweek", "trackman_game_id",
                              "pitch_no", "inning", "top_bottom", "pitcher_trackman_id",
                              "pitcher_team", "batter_team", "pitcher_hand",
                              "balls_before", "strikes_before"])
    tm = tm[~tm.pitcher_team.str.startswith("MIN_")
            & ~tm.batter_team.str.startswith("MIN_")].copy()
    inv = {v: k for k, v in TEAM.items() if v}
    inv.update({"SK_WYV": 21, "SSG_LAN": 21})
    tm["t1"] = np.minimum(tm.pitcher_team.map(inv), tm.batter_team.map(inv))
    tm["t2"] = np.maximum(tm.pitcher_team.map(inv), tm.batter_team.map(inv))
    tm["tb"] = tm.top_bottom.astype(str).str[0].str.upper()

    tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                     usecols=["season", "game_month", "game_dayofweek", "inning",
                              "top_bottom", "balls_before", "strikes_before",
                              "pitcher_team_id", "batter_team_id", "game_type",
                              "pitcher_id", "pitcher_hand"])
    tr["tb"] = tr.top_bottom.astype(str).str[0].str.upper()
    return tm, tr


def reconstruct_games(tr):
    tr = tr.copy()
    tr["t1"] = np.minimum(tr.pitcher_team_id, tr.batter_team_id)
    tr["t2"] = np.maximum(tr.pitcher_team_id, tr.batter_team_id)
    key = tr[["season", "game_month", "game_dayofweek", "t1", "t2", "game_type"]]
    changed = (key != key.shift(1)).any(axis=1)
    tr["gid"] = changed.cumsum()
    return tr


def half_vec(df, key):
    d = df.copy()
    d["hi"] = np.clip(d["inning"], 1, 13) * 2 + (d["tb"] == "B").astype(int)
    return d.pivot_table(index=key, columns="hi", aggfunc="size", fill_value=0)


def match_games(tm, trR):
    HA = half_vec(tm, "trackman_game_id"); HB = half_vec(trR, "gid")
    g = trR.groupby("gid")
    gt = pd.DataFrame({"season": g.season.first(), "m": g.game_month.first(),
                       "d": g.game_dayofweek.first(), "t1": g.t1.first(),
                       "t2": g.t2.first()}).reset_index()
    gi = tm.groupby("trackman_game_id").agg(
        season=("season", "first"), m=("game_month", "first"),
        d=("game_dayofweek", "first"), t1=("t1", "first"), t2=("t2", "first")).reset_index()
    pairs = []
    for k, B in gt.groupby(["season", "m", "d", "t1", "t2"]):
        A = gi[(gi.season == k[0]) & (gi.m == k[1]) & (gi.d == k[2])
               & (gi.t1 == k[3]) & (gi.t2 == k[4])]
        if not len(A):
            continue
        cols = HA.columns.union(HB.columns)
        Xa = HA.reindex(index=A.trackman_game_id, columns=cols, fill_value=0).astype(float).values
        Yb = HB.reindex(index=B.gid, columns=cols, fill_value=0).astype(float).values
        Xa = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-9)
        Yb = Yb / (np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-9)
        S = Yb @ Xa.T
        r, c = linear_sum_assignment(-S)
        for i, j in zip(r, c):
            pairs.append((int(B.gid.iloc[i]), A.trackman_game_id.iloc[j], float(S[i, j])))
    return pd.DataFrame(pairs, columns=["gid", "tgid", "sim"])


def match_pitchers_seq(tm, trR, P, sim_min=0.95):
    """매칭된 경기 안에서 하프이닝별 투구를 순서대로 1:1 대응시킨다.

    양쪽 투구 수가 같은 하프이닝만 쓴다. 교체가 있어도 순서는 보존되므로
    기존(한 명만 던진 이닝)보다 훨씬 많은 근거가 모인다.
    """
    P = P[P.sim > sim_min]
    a = tm[tm.trackman_game_id.isin(set(P.tgid))].copy()
    b = trR[trR.gid.isin(set(P.gid))].copy()
    for d, ic in ((a, "inning"), (b, "inning")):
        d["hi"] = np.clip(d[ic], 1, 13) * 2 + (d["tb"] == "B").astype(int)
    a = a.sort_values(["trackman_game_id", "hi", "pitch_no"])
    a["k"] = a.groupby(["trackman_game_id", "hi"]).cumcount()
    b = b.merge(P[["gid", "tgid"]], on="gid")
    b["k"] = b.groupby(["gid", "hi"]).cumcount()

    na = a.groupby(["trackman_game_id", "hi"]).size().rename("na")
    nb = b.groupby(["tgid", "hi"]).size().rename("nb")
    ok = pd.concat([na, nb], axis=1).dropna()
    ok = ok[ok.na == ok.nb].reset_index()
    ok.columns = ["tgid", "hi", "na", "nb"]
    print("  투구수 일치 하프이닝 %d개 / 전체 %d개 (%.1f%%)"
          % (len(ok), len(na), 100 * len(ok) / max(len(na), 1)), flush=True)

    key = set(zip(ok.tgid, ok.hi))
    a = a[[(t, h) in key for t, h in zip(a.trackman_game_id, a.hi)]]
    b = b[[(t, h) in key for t, h in zip(b.tgid, b.hi)]]
    mm = b.merge(a, left_on=["tgid", "hi", "k"],
                 right_on=["trackman_game_id", "hi", "k"], suffixes=("_b", "_a"))
    print("  정렬된 투구 %d개" % len(mm), flush=True)

    pids = np.sort(mm.pitcher_id.unique()); tids = np.sort(mm.pitcher_trackman_id.unique())
    pi = {v: i for i, v in enumerate(pids)}; ti = {v: i for i, v in enumerate(tids)}
    M = np.zeros((len(pids), len(tids)))
    np.add.at(M, (mm.pitcher_id.map(pi).values, mm.pitcher_trackman_id.map(ti).values), 1.0)
    share = M / (M.sum(1, keepdims=True) + 1e-9)

    # Hungarian 1:1 할당 (기존 argmax 는 중복 배정이 가능했다)
    n = min(len(pids), len(tids))
    r, c = linear_sum_assignment(-share)
    res = pd.DataFrame({"pitcher_id": pids[r], "pitcher_trackman_id": tids[c],
                        "conf": share[r, c], "evidence_pitches": M[r, c]})
    return res[res.evidence_pitches > 0].reset_index(drop=True), mm


def main():
    print("데이터 로드...", flush=True)
    tm, tr = load()
    tr = reconstruct_games(tr)
    trR = tr[tr.game_type == "R"].copy()
    print("  train 복원 경기 %d개 | trackman 경기 %d개"
          % (trR.gid.nunique(), tm.trackman_game_id.nunique()), flush=True)

    print("경기 매칭...", flush=True)
    P = match_games(tm, trR)
    print("  %d건, 유사도 중앙값 %.4f, 0.95 초과 %d건"
          % (len(P), P.sim.median(), (P.sim > 0.95).sum()), flush=True)

    print("투수 매칭 (순서 정렬 방식)...", flush=True)
    mp, mm = match_pitchers_seq(tm, trR, P)

    # 검증: 매칭에 안 쓴 투수 좌우 손
    hb = trR.groupby("pitcher_id").pitcher_hand.agg(lambda s: s.mode().iloc[0])
    ha = tm.groupby("pitcher_trackman_id").pitcher_hand.agg(lambda s: s.mode().iloc[0])
    v = mp.copy()
    v["hb"] = v.pitcher_id.map(hb); v["ha"] = v.pitcher_trackman_id.map(ha)
    v = v.dropna(subset=["hb", "ha"])
    # train 은 0/1 정수, trackman 은 문자열일 수 있으므로 순위로 비교
    hb_bin = (v.hb.astype(str).str[0].str.upper() == v.hb.astype(str).str[0].str.upper())
    same = (v.ha.astype(str).str[0].str.upper().map({"R": 1, "L": 0})
            == pd.to_numeric(v.hb, errors="coerce").map({1: 1, 0: 0}))
    agree = float(np.nanmean(same.astype(float))) if same.notna().any() else float("nan")

    old = pd.read_csv(os.path.join(HERE, "pitcher_id_map.csv"))
    n_tr = len(trR)
    cov_old = trR.pitcher_id.isin(set(old.pitcher_id)).mean()
    cov_new = trR.pitcher_id.isin(set(mp.pitcher_id)).mean()

    print()
    print("=" * 62)
    print("[MATCH 2]")
    print("=" * 62)
    print("  기존  투수 %3d명   행 커버리지 %.1f%%" % (len(old), 100 * cov_old))
    print("  신규  투수 %3d명   행 커버리지 %.1f%%" % (len(mp), 100 * cov_new))
    print("  신뢰도 0.9 이상 %d명 / 0.99 이상 %d명"
          % ((mp.conf >= 0.9).sum(), (mp.conf >= 0.99).sum()))
    print("  좌우손 일치율 %.3f  (검증용, 매칭에 미사용)" % agree)
    ov = old.merge(mp, on="pitcher_id", suffixes=("_old", "_new"))
    if len(ov):
        print("  기존과 겹치는 %d명 중 동일 배정 %d명 (%.1f%%)"
              % (len(ov), (ov.pitcher_trackman_id_old == ov.pitcher_trackman_id_new).sum(),
                 100 * (ov.pitcher_trackman_id_old == ov.pitcher_trackman_id_new).mean()))
    mp.to_csv(os.path.join(HERE, "pitcher_id_map2.csv"), index=False)
    print("\n저장: pitcher_id_map2.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
