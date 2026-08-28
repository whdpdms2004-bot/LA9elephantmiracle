"""PLAN.md §2 의 13축을 파생해 out/axes_<season>.csv 로 굽는다.

핵심은 `game_uid` 복원이다 — train.csv 에 경기 ID 가 없어서 투수 등판 단위 축
(A3·A3b·A12)을 만들 수 없다. 복원 근거와 검증값은 PLAN.md §2 를 본다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gbp_common import AXES, OUT, SEASONS, TRAIN

USE = ["row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "game_type", "balls_before", "strikes_before", "outs_before",
       "score_diff_pitcher_team", "base_state", "li",
       "pitcher_id", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
       "asof_pitcher_n", "asof_pitcher_success_rate", "asof_batter_n", "control_success"]


def cut(x, edges, labels):
    """오른쪽 열린 구간 [a,b). 결측은 NaN 라벨로 남긴다 (호출부가 채운다)."""
    return pd.cut(x, bins=edges, labels=labels, right=False, ordered=False).astype(object)


def build_game_uid(d: pd.DataFrame) -> np.ndarray:
    """파일 시간순 + 팀쌍 전환 + 이닝 감소로 경기를 끊는다 (PLAN.md §2)."""
    a = d[["pitcher_team_id", "batter_team_id"]].to_numpy()
    lo, hi = np.minimum(a[:, 0], a[:, 1]), np.maximum(a[:, 0], a[:, 1])
    key = (d["season"].astype(str) + "|" + d["game_month"].astype(str) + "|"
           + d["game_dayofweek"].astype(str) + "|" + pd.Series(lo, index=d.index).astype(str)
           + "|" + pd.Series(hi, index=d.index).astype(str))
    new_key = (key != key.shift()).to_numpy()
    inn = d["inning"].to_numpy()
    inning_drop = np.r_[False, inn[1:] < inn[:-1]]
    return (new_key | inning_drop).cumsum()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print(f"read {TRAIN} ...")
    d = pd.read_csv(TRAIN, usecols=lambda c: c.strip("﻿") in USE)
    d.columns = [c.strip("﻿") for c in d.columns]
    d["row_id"] = d["row_id"].astype(str)

    # --- 경기·등판 복원 (전 시즌에서 해야 전시즌 집계가 나온다) ------------- #
    d["game_uid"] = build_game_uid(d)
    d["game_pitch_no"] = d.groupby(["game_uid", "pitcher_id"]).cumcount() + 1
    gs = d.groupby("game_uid").size()
    ap = d.groupby(["game_uid", "pitcher_id"]).size()
    print(f"  game_uid {d.game_uid.nunique():,}개 · 시즌별 "
          f"{d.groupby('season').game_uid.nunique().to_dict()}")
    print(f"  경기당 투구 중앙 {gs.median():.0f} · 등판당 투구 중앙 {ap.median():.0f} "
          f"p90 {ap.quantile(.9):.0f} max {ap.max()}")

    # --- 전시즌 집계 (season-1 의 그 투수) --------------------------------- #
    prev = (d.groupby(["pitcher_id", "season"])
              .agg(prev_pitches=("row_id", "size"), prev_games=("game_uid", "nunique"))
              .reset_index())
    prev["season"] += 1
    d = d.merge(prev, on=["pitcher_id", "season"], how="left")
    d["prev_ppg"] = d["prev_pitches"] / d["prev_games"]
    d["load"] = d["game_pitch_no"] / d["prev_ppg"]

    d = d[d["season"].isin(SEASONS) & d["control_success"].notna()].copy()

    # --- A6 은 두 시즌 풀에서 5분위 절단을 한 번만 정해 공유한다 ------------ #
    q = d["asof_pitcher_success_rate"].quantile([.2, .4, .6, .8]).tolist()
    (OUT / "bins.json").write_text(json.dumps({"a6_psucc_edges": q}, indent=2), encoding="utf-8")

    # --- 축 ---------------------------------------------------------------- #
    d["a1_count"] = d.balls_before.astype(str) + "-" + d.strikes_before.astype(str)

    b, s = d.balls_before, d.strikes_before
    d["a1b_phase"] = np.select(
        [(b == 3) & (s == 2), b == 3, s == 2, (b == 0) & (s == 0), (b == 0) & (s == 1), (b == 1) & (s == 1)],
        ["풀카운트", "3볼", "2스트라이크", "초구 0-0", "투수우위 0-1", "평행 1-1"],
        default="타자우위 1-0/2-0/2-1")

    d["a2_li"] = cut(d.li, [0, .10, .35, .70, 1.20, 2.00, np.inf], AXES["a2_li"][1])

    d["a3_load"] = cut(d.load, [0, .25, .50, .75, 1.00, 1.25, 1.50, np.inf],
                       AXES["a3_load"][1][:-1])
    d.loc[d.prev_pitches.isna(), "a3_load"] = "전시즌없음"

    d["a3b_gp"] = cut(d.game_pitch_no, [1, 11, 21, 36, 61, 81, np.inf], AXES["a3b_gp"][1])

    d["a4_prevp"] = cut(d.prev_pitches, [1, 201, 601, 1201, 2001, 2601, np.inf],
                        AXES["a4_prevp"][1][1:])
    d.loc[d.prev_pitches.isna(), "a4_prevp"] = "없음"

    d["a5_asofn"] = cut(d.asof_pitcher_n, [0, 100, 500, 1500, 4000, 8000, np.inf],
                        AXES["a5_asofn"][1])

    d["a6_psucc"] = cut(d.asof_pitcher_success_rate, [-np.inf] + q + [np.inf],
                        AXES["a6_psucc"][1][:-1])
    d.loc[d.asof_pitcher_success_rate.isna(), "a6_psucc"] = "이력없음"

    d["a7_inning"] = cut(d.inning, [1, 4, 7, 9, np.inf], AXES["a7_inning"][1])
    d["a8_hand"] = d.pitcher_hand.astype(str) + "v" + d.batter_hand.astype(str)
    d["a9_batn"] = cut(d.asof_batter_n, [0, 100, 500, 1500, 4000, np.inf], AXES["a9_batn"][1])
    d["a10_base"] = d.base_state.astype(str)

    mo = np.select([d.game_month.between(3, 5), d.game_month.between(6, 7)],
                   ["early", "mid"], default="late")
    d["a11_typemo"] = d.game_type.astype(str) + "|" + mo

    d["a12_role"] = np.select(
        [d.prev_ppg.isna(), d.prev_ppg < 30, d.prev_ppg < 70],
        ["전시즌없음", "불펜(<30구/경기)", "스윙(30-70)"], default="선발(>=70)")

    ig = np.select([d.inning <= 3, d.inning <= 6], ["초반", "중반"], default="후반")
    ad = d.score_diff_pitcher_team.abs()
    dg = np.select([ad <= 1, ad <= 3], ["0-1점", "2-3점"], default="4점+")
    d["a13_tens"] = pd.Series(ig, index=d.index) + "|" + pd.Series(dg, index=d.index)

    cols = ["row_id"] + list(AXES)
    for season in SEASONS:
        o = d[d.season == season][cols].sort_values("row_id").reset_index(drop=True)
        assert o.notna().all().all(), f"{season}: 축에 결측이 남았다"
        o.to_csv(OUT / f"axes_{season}.csv", index=False)
        print(f"\n=== {season}  n={len(o):,} ===")
        for k, (title, order) in AXES.items():
            vc = o[k].value_counts()
            small = vc[vc < 5000]
            miss = [v for v in order if v not in vc.index] if order else []
            flag = ""
            if len(small):
                flag = f"  ⚠ n<5000: {dict(small)}"
            if miss:
                flag += f"  ⚠ 빈 구간 {miss}"
            print(f"  {title:<38} {len(vc):>2}구간 min_n={vc.min():>7,}{flag}")


if __name__ == "__main__":
    main()
