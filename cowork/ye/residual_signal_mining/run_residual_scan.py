# -*- coding: utf-8 -*-
"""챔피언 residual 에서 안정적인 interaction 신호를 찾는다 (CPU-only 진단).

새 모델·재학습·calibration·blend 전부 안 한다. pandas/numpy 만 쓴다.

OOF 예측 우선순위(지시서 §3):
  1순위 sj_stdmlp  -- 2022/2024 만 등록돼 있음 (val/sj_stdmlp_2023.csv 없음)
  2023 은 sj3way_nv 로 대체한다 (다음으로 공식적인 챔피언 계열 OOF).
  ★ 계기가 다르다는 걸 결과에 항상 명시한다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data"
VAL = REPO / "performance_tracking" / "val"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(m):
    print(m, flush=True)


OOF_SOURCE = {2022: "sj_stdmlp", 2023: "sj3way_nv", 2024: "sj_stdmlp"}


def brier(p, y):
    return float(((p - y) ** 2).mean())


def bss(p, y):
    r = y.mean()
    return 100000.0 * max(0.0, 1.0 - brier(p, y) / (r * (1 - r)))


def load_oof():
    frames = []
    for season, src in OOF_SOURCE.items():
        f = pd.read_csv(VAL / f"{src}_{season}.csv").rename(columns={"pred": "pred"})
        f["season_key"] = season
        f["oof_source"] = src
        frames.append(f[["row_id", "pred", "season_key", "oof_source"]])
    return pd.concat(frames, ignore_index=True)


def build_role(df):
    """cowork/cw/v17/src/build_v13.py 의 build_role 과 동일 로직(재사용, 새로 설계 안 함).
    이닝 분포로 선발/불펜/스윙맨 근사. as-of(직전 시즌까지)."""
    g = df.groupby(["pitcher_id", "season"])
    t = pd.DataFrame({
        "inn_mean": g["inning"].mean(), "inn_std": g["inning"].std(),
        "p1_ratio": g["inning"].apply(lambda v: (v <= 2).mean()),
        "p_season": g.size(),
    }).reset_index().rename(columns={"pitcher_id": "pid"})

    ids = df["pitcher_id"].to_numpy(np.int64)
    sea = df["season"].to_numpy()
    out = np.full((len(df), 2), np.nan)  # inn_mean, p1_ratio
    for s in sorted(set(sea.tolist())):
        h = t[t.season < s]
        if len(h) == 0:
            continue
        gg = h.groupby("pid")[["inn_mean", "p1_ratio"]].mean()
        idx = gg.index.to_numpy()
        v = gg.to_numpy()
        m = sea == s
        pos = np.clip(np.searchsorted(idx, ids[m]), 0, max(len(idx) - 1, 0))
        hit = idx[pos] == ids[m]
        vv = v[pos].copy()
        vv[~hit] = np.nan
        out[m] = vv
    return out[:, 0], out[:, 1]  # inn_mean(높을수록 선발형), p1_ratio(높을수록 초반등판=마무리/원포인트형)


def main():
    t0 = time.time()
    log("데이터 로드 중...")
    cols = ["row_id", "season", "game_month", "balls_before", "strikes_before",
            "pitcher_id", "batter_id", "pitcher_hand", "batter_hand", "inning",
            "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
            "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_batter_n", "asof_batter_success_rate", "control_success", "li"]
    df = pd.read_csv(DATA / "train.csv", usecols=cols, encoding="utf-8-sig")
    df.columns = [c.strip("﻿") for c in df.columns]
    log(f"  {len(df):,}행  ({time.time()-t0:.0f}s)")

    oof = load_oof()
    df = df.merge(oof, on="row_id", how="inner")
    log(f"  OOF 조인 후 {len(df):,}행 (시즌별: "
        f"{dict(df.groupby('season_key').size())})  ({time.time()-t0:.0f}s)")

    y = df["control_success"].to_numpy(float)
    pred = df["pred"].to_numpy(float)
    df["residual"] = y - pred
    df["signed_error"] = df["residual"]
    df["abs_error"] = np.abs(df["residual"])
    df["squared_error"] = df["residual"] ** 2

    # ── §6 sanity check ──────────────────────────────────────────
    log("\n===== 1단계: sanity check =====")
    rows = []
    for season in (2022, 2023, 2024):
        sub = df[df.season == season]
        rows.append({"season": season, "n": len(sub), "actual_rate": sub.control_success.mean(),
                     "pred_mean": sub.pred.mean(), "mean_residual": sub.residual.mean(),
                     "mean_abs_error": sub.abs_error.mean(), "brier": brier(sub.pred, sub.control_success)})
    sanity = pd.DataFrame(rows)
    print(sanity.to_string(index=False))
    sanity.to_csv(HERE / "sanity_check.csv", index=False)

    # ── 파생 피처 (기존 raw 조합, 새 정보 아님 — 재료용) ──────────
    log("\n파생 피처 생성...")
    df["form_delta1"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["form_delta3"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["form_delta5"] = df["asof_pitcher_prev5_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["two_strike"] = (df["strikes_before"] == 2).astype(int)
    df["late_season"] = (df["game_month"] >= 8).astype(int)
    df["matchup_gap"] = df["asof_batter_success_rate"] - df["asof_pitcher_success_rate"]
    df["platoon_same"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["pitcher_experience"] = np.log1p(df["asof_pitcher_n"].fillna(0))
    df["reliability"] = df["asof_pitcher_n"] / (df["asof_pitcher_n"] + 200.0)
    inn_mean, p1_ratio = build_role(df)
    df["role_starter_ness"] = inn_mean
    df["role_short_ness"] = p1_ratio
    df["count_phase"] = df["balls_before"] * 3 + df["strikes_before"]

    rng = np.random.default_rng(42)
    df["random_noise"] = rng.normal(size=len(df))
    df["row_id_hash"] = df["row_id"].astype(str).apply(lambda s: hash(s) % 10007)

    # ── §9 interaction 후보 (A~H + negative control) ────────────
    CANDS = {}
    # A. Count x Pitcher
    CANDS["A_2strike_x_pitcherlvl"] = df.two_strike * df.asof_pitcher_success_rate
    CANDS["A_2strike_x_form3"] = df.two_strike * df.form_delta3
    CANDS["A_2strike_x_middlerate"] = df.two_strike * df.asof_pitcher_middle_rate
    CANDS["A_2strike_x_experience"] = df.two_strike * df.pitcher_experience
    CANDS["A_balls_x_pitcherlvl"] = df.balls_before * df.asof_pitcher_success_rate
    # B. Count x Batter
    CANDS["B_2strike_x_batterlvl"] = df.two_strike * df.asof_batter_success_rate
    CANDS["B_2strike_x_battern"] = df.two_strike * np.log1p(df.asof_batter_n.fillna(0))
    CANDS["B_2strike_x_matchup"] = df.two_strike * df.matchup_gap
    CANDS["B_countphase_x_matchup"] = df.count_phase * df.matchup_gap
    # C. Recent form x career
    CANDS["C_form1_x_pitcherlvl"] = df.form_delta1 * df.asof_pitcher_success_rate
    CANDS["C_form3_x_experience"] = df.form_delta3 * df.pitcher_experience
    CANDS["C_form5_x_pitcherlvl"] = df.form_delta5 * df.asof_pitcher_success_rate
    CANDS["C_form3_x_countphase"] = df.form_delta3 * df.count_phase
    # D. Late season
    CANDS["D_late_x_form3"] = df.late_season * df.form_delta3
    CANDS["D_late_x_form1"] = df.late_season * df.form_delta1
    CANDS["D_late_x_pitcherlvl"] = df.late_season * df.asof_pitcher_success_rate
    CANDS["D_late_x_experience"] = df.late_season * df.pitcher_experience
    CANDS["D_late_x_matchup"] = df.late_season * df.matchup_gap
    CANDS["D_late_x_2strike"] = df.late_season * df.two_strike
    CANDS["D_late_x_platoon"] = df.late_season * df.platoon_same
    # E. Pitcher level x reliability
    CANDS["E_pitcherlvl_x_experience"] = df.asof_pitcher_success_rate * df.pitcher_experience
    CANDS["E_pitcherlvl_x_reliability"] = df.asof_pitcher_success_rate * df.reliability
    # F. role
    CANDS["F_role_starter_x_pitcherlvl"] = df.role_starter_ness * df.asof_pitcher_success_rate
    CANDS["F_role_short_x_pitcherlvl"] = df.role_short_ness * df.asof_pitcher_success_rate
    CANDS["F_role_short_x_2strike"] = df.role_short_ness * df.two_strike
    # G. matchup
    CANDS["G_matchup_x_2strike"] = df.matchup_gap * df.two_strike
    CANDS["G_matchup_x_late"] = df.matchup_gap * df.late_season
    CANDS["G_matchup_x_experience"] = df.matchup_gap * df.pitcher_experience
    # H. platoon
    CANDS["H_platoon_x_pitcherlvl"] = df.platoon_same * df.asof_pitcher_success_rate
    CANDS["H_platoon_x_batterlvl"] = df.platoon_same * df.asof_batter_success_rate
    CANDS["H_platoon_x_form3"] = df.platoon_same * df.form_delta3
    CANDS["H_platoon_x_2strike"] = df.platoon_same * df.two_strike
    # negative controls
    CANDS["NC_random_noise"] = df.random_noise
    CANDS["NC_rowid_hash"] = df.row_id_hash
    CANDS["NC_pitcherlvl_x_noise"] = df.asof_pitcher_success_rate * df.random_noise

    log(f"\n총 후보 수: {len(CANDS)}  ({time.time()-t0:.0f}s)")

    # ── §11 후보 평가: 5분위 버킷, high-low residual gap, 방향, 시즌 안정성 ──
    def eval_candidate(name, series):
        row = {"candidate": name, "sample_n": int(series.notna().sum()),
               "missing_rate": float(series.isna().mean())}
        for season, gate in ((2022, "R"), (2023, "R"), (2024, "all")):
            if gate == "R":
                mask = (df.season == season) & (df.game_type == "R") if "game_type" in df.columns else (df.season == season)
            else:
                mask = df.season == season
            s = series[mask]
            r = df.loc[mask, "residual"]
            valid = s.notna()
            s, r = s[valid], r[valid]
            if len(s) < 500 or s.std() == 0:
                row[f"effect_{season}{gate}"] = np.nan
                row[f"direction_{season}"] = 0
                continue
            try:
                q = pd.qcut(s, 5, labels=False, duplicates="drop")
            except ValueError:
                row[f"effect_{season}{gate}"] = np.nan
                row[f"direction_{season}"] = 0
                continue
            g = r.groupby(q).mean()
            if len(g) < 2:
                row[f"effect_{season}{gate}"] = np.nan
                row[f"direction_{season}"] = 0
                continue
            effect = float(g.iloc[-1] - g.iloc[0])
            row[f"effect_{season}{gate}"] = effect
            row[f"direction_{season}"] = 1 if effect > 0 else (-1 if effect < 0 else 0)
        dirs = [row["direction_2022"], row["direction_2023"], row["direction_2024"]]
        nonzero = [d for d in dirs if d != 0]
        row["direction_consistency"] = (sum(1 for d in nonzero if d == nonzero[0]) if nonzero else 0)
        effects = [row.get("effect_2022R"), row.get("effect_2023R"), row.get("effect_2024all")]
        effects = [abs(e) for e in effects if e is not None and not np.isnan(e)]
        row["median_abs_effect"] = float(np.median(effects)) if effects else 0.0
        row["sample_share"] = row["sample_n"] / len(df)
        return row

    # game_type 없으면 R 게이트를 season 전체로 근사
    has_gt = False
    if "game_type" not in df.columns:
        gt = pd.read_csv(DATA / "train.csv", usecols=["row_id", "game_type"], encoding="utf-8-sig")
        df = df.merge(gt, on="row_id", how="left")
        has_gt = True
    log(f"game_type 확보: {has_gt or 'already had'}")

    results = []
    for name, series in CANDS.items():
        results.append(eval_candidate(name, series))
    rank_df = pd.DataFrame(results)
    rank_df["score"] = (rank_df["direction_consistency"].clip(lower=0) *
                        rank_df["median_abs_effect"] * np.sqrt(rank_df["sample_share"].clip(lower=1e-6)))
    rank_df = rank_df.sort_values("score", ascending=False).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)

    log("\n===== 후보 랭킹 (상위 15) =====")
    show_cols = ["rank", "candidate", "direction_2022", "direction_2023", "direction_2024",
                 "direction_consistency", "median_abs_effect", "sample_share", "score"]
    print(rank_df[show_cols].head(15).to_string(index=False))

    rank_df.to_csv(HERE / "candidate_ranking.csv", index=False)
    log(f"\n저장: candidate_ranking.csv")
    log(f"총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    raise SystemExit(main())
