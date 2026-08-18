"""V75: production 조건에서 행 단위 파생 피처 family를 선별한다.

이 실험의 첫 목적은 submit_035의 2차 피처가 최종 학습 조건에서도 재현되는지
확인하는 것이다. V71은 F행 0.20과 짧은 등판 0.50을 함께 썼지만 submit_035는
F행 0.20만 사용한다. 여기서는 최종 구성과 동일하게 F행만 가중한다.

stage=screen
    Val2024, XGB+CatBoost 2 seeds, 250 rounds. 여러 family를 빠르게 비교한다.
stage=confirm
    Val2023/Val2024, XGB+CatBoost 8 seeds, 400 rounds. --arms로 지정한 후보만
    production 수준으로 재확인한다. P0은 항상 포함된다.

모든 후보는 현재 행과 fold 학습 데이터에서 고정한 lookup만 사용한다.
test 행간 집계, test 분포, 미래 시즌 Target은 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import component_features as CF
from harness import BASE_PARAMS, CACHE, OUT, TARGET, load, metrics

SJ = Path(__file__).resolve().parents[2]
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = SJ / "feature_campaign_1000"
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
TM_RELEASE = CAMPAIGN / "outputs" / "trackman_release"

ALL_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
K = 300
EPS = 1e-7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["screen", "confirm"], default="screen")
    p.add_argument("--arms", default="P1,P2,P3,P4,P5,P6,P7,P8,P9")
    p.add_argument(
        "--reuse-cache",
        action="store_true",
        help=("같은 stage/arm/fold 설정으로 저장된 예측을 재사용한다. "
              "피처 코드나 학습 설정을 바꾼 뒤에는 사용하지 않는다."),
    )
    p.add_argument("--folds", default=None,
                   help="선택 실행용 검증 시즌 목록. 예: 2023 또는 2023,2024")
    p.add_argument("--seed-count", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    return p.parse_args()


ARGS = parse_args()
if ARGS.stage == "screen":
    FOLDS, SEEDS, N_ROUNDS = [2024], ALL_SEEDS[:2], 250
else:
    FOLDS, SEEDS, N_ROUNDS = [2023, 2024], ALL_SEEDS, 400
if ARGS.folds:
    FOLDS = [int(value) for value in ARGS.folds.split(",") if value]
if ARGS.seed_count is not None:
    SEEDS = ALL_SEEDS[:ARGS.seed_count]
if ARGS.rounds is not None:
    N_ROUNDS = ARGS.rounds
REQUESTED = [a.strip() for a in ARGS.arms.split(",") if a.strip()]
ARMS = ["P0"] + [a for a in REQUESTED if a != "P0"]

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c not in ("label_ok", TARGET)]
pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
balls = df["balls_before"].to_numpy()
strikes = df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
ROW_W = np.where(df["game_type"].astype(str).to_numpy() == "F", 0.20, 1.0)

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)


def and_label(*arrs):
    m = np.ones(len(df), dtype=bool)
    for arr in arrs:
        m &= arr == 1
    return np.where(ok, m.astype(float), np.nan)


LAB = {
    "m": ym,
    "r": yr,
    "mr": and_label(ym, yr),
    "ob": and_label(yo, yb),
    "oz": and_label(yo, 1 - yb),
}


def load_base_predictions():
    models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                     for p in OOF_DIR.glob("*.parquet")})
    out = {}
    for fold in FOLDS:
        ids = df.loc[season == fold, "row_id"].to_numpy()
        if fold == 2024:
            pr = pd.read_parquet(PROD).set_index("row_id").reindex(ids)
            out[fold] = np.clip(
                pr["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                EPS, 1 - EPS)
        else:
            acc, count = None, 0
            for name in models:
                path = OOF_DIR / f"{name}_fold{fold}.parquet"
                if path.exists():
                    pred = (pd.read_parquet(path).set_index("row_id")
                            .reindex(ids)["prediction"].to_numpy(np.float64))
                    acc = pred if acc is None else acc + pred
                    count += 1
            if count == 0:
                raise FileNotFoundError(f"base OOF missing for fold {fold}")
            out[fold] = np.clip(acc / count, EPS, 1 - EPS)
    return out


BASE_P = load_base_predictions()


def base_features(fold: int) -> pd.DataFrame:
    tr = season < fold
    train = df.loc[tr]
    feat = CF.build(
        df[INPUT_COLS],
        CF.make_spec(train),
        CF.make_platoon_table(train),
        CF.make_batter_platoon_table(train, {k: v[tr] for k, v in LAB.items()}),
    )
    pidx = pd.MultiIndex.from_arrays([pid, bhand])
    for tag, axis in [("cnt", cnt_b), ("inn", inn_b)]:
        d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": axis[tr],
                          "y": y_all[tr]})
        league = float(d["y"].mean())
        g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
        g3 = d.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
        eb2 = (g2["sum"] + K * league) / (g2["size"] + K)
        eb3 = (g3["sum"] + K * league) / (g3["size"] + K)
        i3 = pd.MultiIndex.from_arrays([pid, bhand, axis])
        v2 = eb2.reindex(pidx).fillna(league).to_numpy()
        v3 = eb3.reindex(i3).fillna(league).to_numpy()
        size = g3["size"].reindex(i3).fillna(0.0).to_numpy()
        feat[f"{tag}_split"] = v3 - v2
        feat[f"{tag}_rel"] = size / (size + K)
        feat[f"{tag}_w"] = (v3 - v2) * size / (size + K)
    return feat


def safe_values(frame: pd.DataFrame, names: list[str], fallback: float = 0.0):
    return [np.nan_to_num(frame[n].to_numpy(np.float64), nan=fallback,
                          posinf=fallback, neginf=fallback) for n in names]


def add_p1_second_order(frame: pd.DataFrame) -> pd.DataFrame:
    """submit_035에서 실제 생성된 12열을 충돌 없는 이름으로 재현한다."""
    out = frame.copy()
    products = [
        ("platoon_split", "cnt_split"),
        ("platoon_split", "inn_split"),
        ("cnt_split", "inn_split"),
        ("asof_pitcher_success_rate", "asof_batter_success_rate"),
        ("asof_pitcher_middle_rate", "asof_batter_middle_rate"),
        ("asof_pitcher_success_rate", "li"),
        ("asof_pitcher_reverse_rate", "asof_pitcher_fastball_rate"),
    ]
    for i, (a, b) in enumerate(products):
        out[f"p1_product_{i:02d}"] = out[a].to_numpy() * out[b].to_numpy()
    logn = np.log1p(np.nan_to_num(out["asof_pitcher_n"].to_numpy(), nan=0.0))
    out["p1_psuccess_x_logn"] = out["asof_pitcher_success_rate"].to_numpy() * logn
    out["p1_platoon_x_logn"] = out["platoon_split"].to_numpy() * logn
    # v71/v73의 실제 결과는 이름 충돌 때문에 아래 두 중복 delta와 비율만 남았다.
    out["p1_duplicate_prev1_middle_delta"] = (
        out["asof_pitcher_prev1_game_middle_rate"].to_numpy()
        - out["asof_pitcher_middle_rate"].to_numpy())
    out["p1_pitcher_minus_batter_success"] = (
        out["asof_pitcher_success_rate"].to_numpy()
        - out["asof_batter_success_rate"].to_numpy())
    out["p1_pitcher_over_batter_success"] = (
        out["asof_pitcher_success_rate"].to_numpy()
        / np.clip(out["asof_batter_success_rate"].to_numpy(), 1e-3, None))
    return out


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    b, s, inning, li, diff = safe_values(
        out, ["balls_before", "strikes_before", "inning", "li",
              "score_diff_pitcher_team"])
    close = (np.abs(diff) <= 1).astype(float)
    late = (inning >= 7).astype(float)
    out["ctx_count_margin"] = b - s
    out["ctx_two_strike"] = (s == 2).astype(float)
    out["ctx_three_ball"] = (b == 3).astype(float)
    out["ctx_full_count"] = ((b == 3) & (s == 2)).astype(float)
    out["ctx_tie_game"] = (diff == 0).astype(float)
    out["ctx_close_game"] = close
    out["ctx_late_close"] = late * close
    out["ctx_li_close"] = li * close
    out["ctx_li_count_margin"] = li * (b - s)
    out["ctx_score_per_inning"] = out["run_total_before"].to_numpy() / (inning + 1.0)
    out["ctx_scoring_position"] = np.maximum(
        out["runner_on_2b"].to_numpy(), out["runner_on_3b"].to_numpy())
    out["ctx_winexp_extremity"] = np.abs(
        out["home_win_expectancy"].to_numpy() - 50.0) / 50.0
    return out


def add_form(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    sn = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
    mn = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]
    s1, s3, s5 = safe_values(out, sn, fallback=np.nan)
    m1, m3, m5 = safe_values(out, mn, fallback=np.nan)
    career_s = out["asof_pitcher_success_rate"].to_numpy(np.float64)
    career_m = out["asof_pitcher_middle_rate"].to_numpy(np.float64)
    for arr, fallback in [(s1, career_s), (s3, career_s), (s5, career_s),
                          (m1, career_m), (m3, career_m), (m5, career_m)]:
        miss = np.isnan(arr)
        arr[miss] = fallback[miss]
    rel = out["rel200"].to_numpy(np.float64)
    out["form_s_short_slope"] = s1 - s3
    out["form_s_long_slope"] = s3 - s5
    out["form_s_acceleration"] = s1 - 2.0 * s3 + s5
    out["form_m_short_slope"] = m1 - m3
    out["form_m_long_slope"] = m3 - m5
    out["form_m_acceleration"] = m1 - 2.0 * m3 + m5
    out["form_s_abs_shock"] = np.abs(s1 - career_s)
    out["form_m_abs_shock"] = np.abs(m1 - career_m)
    out["form_s_weighted"] = 0.50 * s1 + 0.30 * s3 + 0.20 * s5
    out["form_m_weighted"] = 0.50 * m1 + 0.30 * m3 + 0.20 * m5
    out["form_s_delta_x_rel"] = (out["form_s_weighted"].to_numpy() - career_s) * rel
    out["form_m_delta_x_rel"] = (out["form_m_weighted"].to_numpy() - career_m) * rel
    out["form_prev1_success_x_middle"] = s1 * m1
    return out


def entropy3(a, b, c):
    vals = np.column_stack([a, b, c]).astype(np.float64)
    vals = np.clip(np.nan_to_num(vals, nan=0.0), 0.0, None)
    total = vals.sum(axis=1, keepdims=True)
    share = vals / np.clip(total, 1e-8, None)
    ent = -(share * np.log(np.clip(share, 1e-8, None))).sum(axis=1) / np.log(3.0)
    ent[total[:, 0] <= 1e-8] = 0.0
    return ent, share.max(axis=1), total[:, 0]


def add_profile_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    m, r, ball = safe_values(out, ["asof_pitcher_middle_rate",
                                    "asof_pitcher_reverse_rate",
                                    "asof_pitcher_ball_rate"])
    fast, breaking, off = safe_values(out, ["asof_pitcher_fastball_rate",
                                             "asof_pitcher_breaking_rate",
                                             "asof_pitcher_offspeed_rate"])
    fent, fmax, fsum = entropy3(m, r, ball)
    pent, pmax, psum = entropy3(fast, breaking, off)
    out["prof_failure_entropy"] = fent
    out["prof_failure_concentration"] = fmax
    out["prof_failure_mass"] = fsum
    out["prof_pitchmix_entropy"] = pent
    out["prof_pitchmix_concentration"] = pmax
    out["prof_pitchmix_sum"] = psum
    out["prof_pitchmix_effective_n"] = np.exp(pent * np.log(3.0))
    out["prof_middle_share"] = m / np.clip(fsum, 1e-6, None)
    out["prof_reverse_share"] = r / np.clip(fsum, 1e-6, None)
    out["prof_ball_share"] = ball / np.clip(fsum, 1e-6, None)
    out["prof_fast_minus_breaking"] = fast - breaking
    out["prof_fast_over_offspeed"] = fast / np.clip(off, 1e-3, None)
    return out


def add_explicit_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    b = out["balls_before"].to_numpy(np.float64)
    s = out["strikes_before"].to_numpy(np.float64)
    two = (s == 2).astype(float)
    three = (b == 3).astype(float)
    margin = b - s
    out["ix_reverse_two_strike"] = out["asof_pitcher_reverse_rate"].to_numpy() * two
    out["ix_ball_three_ball"] = out["asof_pitcher_ball_rate"].to_numpy() * three
    out["ix_strike_two_strike"] = out["asof_pitcher_strike_rate"].to_numpy() * two
    out["ix_middle_li"] = out["asof_pitcher_middle_rate"].to_numpy() * out["li"].to_numpy()
    out["ix_success_count_margin"] = out["asof_pitcher_success_rate"].to_numpy() * margin
    out["ix_fastball_count_margin"] = out["asof_pitcher_fastball_rate"].to_numpy() * margin
    out["ix_breaking_two_strike"] = out["asof_pitcher_breaking_rate"].to_numpy() * two
    out["ix_offspeed_two_strike"] = out["asof_pitcher_offspeed_rate"].to_numpy() * two
    out["ix_batter_success_count"] = out["asof_batter_success_rate"].to_numpy() * margin
    out["ix_platoon_weighted_count"] = out["platoon_split_w"].to_numpy() * margin
    return out


def add_audit_signal_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """단변량 감사 상위 신호를 balls/strikes와 명시적으로 교차한다."""
    out = frame.copy()
    balls_ = out["balls_before"].to_numpy(np.float64)
    strikes_ = out["strikes_before"].to_numpy(np.float64)
    signals = [
        "asof_pitcher_ball_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_offspeed_rate",
        "asof_pitcher_breaking_rate",
    ]
    for i, col in enumerate(signals):
        values = out[col].to_numpy(np.float64)
        out[f"audit_ix_{i:02d}_balls"] = values * balls_
        out[f"audit_ix_{i:02d}_strikes"] = values * strikes_
    prev5 = out["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
    career = out["asof_pitcher_success_rate"].to_numpy(np.float64)
    shock = prev5 - career
    out["audit_ix_prev5_shock_balls"] = shock * balls_
    out["audit_ix_prev5_shock_strikes"] = shock * strikes_
    out["audit_ix_ball_reverse"] = (
        out["asof_pitcher_ball_rate"].to_numpy(np.float64)
        * out["asof_pitcher_reverse_rate"].to_numpy(np.float64))
    out["audit_ix_success_offspeed"] = (
        career * out["asof_pitcher_offspeed_rate"].to_numpy(np.float64))
    return out


TM_LOOKUP_CACHE: dict[int, pd.DataFrame] = {}


def trackman_release_features(fold: int, kind: str) -> pd.DataFrame:
    """cutoff별 target-free TrackMan 투수 lookup을 전체 행에 단건 매핑한다."""
    if fold not in TM_LOOKUP_CACHE:
        path = TM_RELEASE / f"cutoff_{fold}" / "main_pitcher_release.parquet"
        lookup = pd.read_parquet(path).set_index("pitcher_id")
        if lookup.index.duplicated().any():
            raise AssertionError(f"duplicate pitcher_id in {path}")
        TM_LOOKUP_CACHE[fold] = lookup
    lookup = TM_LOOKUP_CACHE[fold]
    common = ["cw_mean_sim", "cw_min_margin", "tr_eligible_seasons",
              "tr_total_pitches", "tr_season_gap"]
    if kind == "pca":
        selected = common + [f"tr_pca_{i:02d}" for i in range(12)]
    elif kind == "resid_pca":
        selected = common + [f"tr_resid_pca_{i:02d}" for i in range(12)]
    elif kind == "core":
        selected = list(common)
        for prefix in ("tr_latest_", "tr_recent_"):
            for group in ("all", "fastball", "breaking", "offspeed"):
                for metric in ("release_trace", "release_area",
                               "release_eigen_ratio"):
                    selected.append(f"{prefix}{group}_{metric}")
            for metric in ("corr_rel_height_rel_side",
                           "corr_rel_speed_rel_height",
                           "corr_rel_speed_rel_side",
                           "corr_rel_speed_extension",
                           "corr_rel_speed_induced_vert_break",
                           "corr_rel_speed_horz_break"):
                selected.append(f"{prefix}all_{metric}")
            for pair in ("fastball_breaking", "fastball_offspeed",
                         "breaking_offspeed"):
                selected.append(f"{prefix}release_sep_{pair}")
    else:
        raise ValueError(kind)
    missing = [c for c in selected if c not in lookup.columns]
    if missing:
        raise KeyError(f"TrackMan columns missing: {missing}")
    mapped = pd.DataFrame(index=np.arange(len(df)))
    for col in selected:
        values = df["pitcher_id"].map(lookup[col]).to_numpy(np.float64)
        if col == "tr_total_pitches":
            values = np.log1p(np.nan_to_num(values, nan=0.0))
        mapped[f"tmr_{kind}_{col}"] = values
    mapped[f"tmr_{kind}_available"] = df["pitcher_id"].isin(lookup.index).to_numpy(float)
    return mapped


def make_arm(base: pd.DataFrame, arm: str, fold: int) -> pd.DataFrame:
    if arm == "P0":
        return base
    out = base
    if arm in ("P1", "P6", "P7", "P8", "P9", "P11", "P12",
               "N1", "N2", "N3", "N4",
               "R3", "R4"):
        out = add_p1_second_order(out)
    if arm in ("P2", "P6", "P9", "P12", "N2", "N3", "N4",
               "R3", "R4"):
        out = add_context(out)
    if arm in ("P3", "P7", "P9"):
        out = add_form(out)
    if arm in ("P4", "P8", "P9"):
        out = add_profile_geometry(out)
    if arm in ("P5", "P9"):
        out = add_explicit_interactions(out)
    if arm in ("P10", "P11", "P12", "N4"):
        out = add_audit_signal_interactions(out)
    if arm in ("T3", "T4", "T5"):
        out = add_p1_second_order(out)
    if arm in ("T7", "T8"):
        out = add_p1_second_order(out)
    if arm in ("T5",):
        out = add_context(out)
    if arm in ("T8",):
        out = add_context(out)
    if arm in ("T1", "T3", "T5"):
        out = pd.concat([out, trackman_release_features(fold, "pca")], axis=1)
    if arm in ("T2", "T4"):
        out = pd.concat([out, trackman_release_features(fold, "core")], axis=1)
    if arm in ("T6", "T7", "T8", "T9", "R4"):
        out = pd.concat([out, trackman_release_features(fold, "resid_pca")], axis=1)
    if arm == "T9":
        out = add_p1_second_order(out)
        out = add_context(out)
        out = add_audit_signal_interactions(out)
    if arm in ("N3",):
        out = pd.concat([out, trackman_release_features(fold, "resid_pca")], axis=1)
    if arm in ("N0", "N1", "N2", "N3", "N4", "R0", "R1", "R2", "R3", "R4"):
        out = out.drop(columns=["season"])
    if arm in ("R0", "R2", "R3", "R4"):
        out = out.drop(columns=["game_type"])
    if arm in ("R1", "R2", "R3", "R4"):
        out = out.drop(columns=[
            "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
        ])
    return out


ARM_NAME = {
    "P0": "production 111",
    "P1": "submit035 second-order",
    "P2": "context pressure",
    "P3": "form curvature",
    "P4": "profile geometry",
    "P5": "explicit interactions",
    "P6": "P1+context",
    "P7": "P1+form",
    "P8": "P1+profile",
    "P9": "all",
    "P10": "audit top-signal x count",
    "P11": "P1+audit interactions",
    "P12": "P1+context+audit interactions",
    "T1": "TrackMan release PCA12",
    "T2": "TrackMan release core",
    "T3": "P1+TrackMan PCA12",
    "T4": "P1+TrackMan core",
    "T5": "P1+context+TrackMan PCA12",
    "T6": "TrackMan residual PCA12",
    "T7": "P1+TrackMan residual PCA12",
    "T8": "P1+context+TrackMan residual PCA12",
    "T9": "P1+context+audit interactions+TrackMan residual PCA12",
    "N0": "remove season",
    "N1": "remove season+P1",
    "N2": "remove season+P1+context",
    "N3": "remove season+P1+context+residual PCA12",
    "N4": "remove season+P1+context+audit interactions",
    "R0": "remove season+game_type",
    "R1": "remove season+raw IDs",
    "R2": "remove season+game_type+raw IDs",
    "R3": "R2+P1+context",
    "R4": "R3+TrackMan residual PCA12",
}


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def component_line(x: np.ndarray, fold: int) -> np.ndarray:
    tr, va = season < fold, season == fold
    pred = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        fit = tr & ~np.isnan(arr)
        yearly = pd.Series(arr[fit]).groupby(pd.Series(season[fit])).mean().sort_index()
        base_score = float(np.clip(
            float(yearly.iloc[-1])
            + (float(yearly.iloc[-1]) - float(yearly.iloc[0]))
            / (float(yearly.index[-1]) - float(yearly.index[0])),
            0.005, 0.995))
        prm = {**BASE_PARAMS, "base_score": base_score,
               **params_for(float(np.nanmean(arr[fit])))}
        dtr = xgb.DMatrix(x[fit], label=arr[fit], weight=ROW_W[fit])
        dva = xgb.DMatrix(x[va])
        pool = Pool(x[fit], arr[fit], weight=ROW_W[fit])
        acc = np.zeros(int(va.sum()), dtype=np.float64)
        for seed in SEEDS:
            xb = xgb.train({**prm, "seed": seed}, dtr,
                           num_boost_round=N_ROUNDS, verbose_eval=False)
            acc += 0.5 * xb.predict(dva)
            cb = CatBoostClassifier(
                iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                l2_leaf_reg=6.0, loss_function="Logloss", random_seed=seed,
                task_type="GPU", verbose=0)
            cb.fit(pool)
            acc += 0.5 * cb.predict_proba(x[va])[:, 1]
        pred[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1.0 - (pred["m"] + pred["r"] - pred["mr"]
                          + pred["ob"] + pred["oz"]), EPS, 1 - EPS)


rows = []
t0 = time.time()
for fold in FOLDS:
    va = season == fold
    base_feat = base_features(fold)
    y = y_all[va]
    base_pred = BASE_P[fold]
    ref_bss = metrics(y, base_pred)["bss_raw"]
    null = y.mean() * (1 - y.mean())
    blend_w = BW[bucket_all[va]]
    logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS)
                              / (1 - np.clip(p, EPS, 1 - EPS)))
    print(f"\nfold {fold} base={ref_bss:.3f} stage={ARGS.stage} "
          f"seeds={len(SEEDS)} rounds={N_ROUNDS}", flush=True)
    for arm in ARMS:
        if arm not in ARM_NAME:
            raise ValueError(f"unknown arm {arm}")
        feat = make_arm(base_feat, arm, fold)
        if feat.columns.duplicated().any():
            dup = feat.columns[feat.columns.duplicated()].tolist()
            raise AssertionError(f"duplicate columns in {arm}: {dup}")
        cache_path = CACHE / f"v75_{ARGS.stage}_{arm}_{fold}.npy"
        if ARGS.reuse_cache and cache_path.exists():
            pred = np.load(cache_path)
            if len(pred) != int(va.sum()):
                raise AssertionError(
                    f"cache row mismatch for {arm}/{fold}: "
                    f"{len(pred)} != {int(va.sum())}")
            print(f"  {arm:<3} cache 재사용: {cache_path.name}", flush=True)
        else:
            pred = component_line(feat.to_numpy(np.float32), fold)
        solo = metrics(y, pred)["bss_raw"]
        corr = float(np.corrcoef(logit(base_pred), logit(pred))[0, 1])
        blended = np.clip(blend_w * pred + (1 - blend_w) * base_pred,
                          EPS, 1 - EPS)
        dbss = metrics(y, blended)["bss_raw"] - ref_bss
        row_gain = (base_pred - y) ** 2 - (blended - y) ** 2
        se = 100000 * float(row_gain.std(ddof=1) / np.sqrt(len(row_gain))) / null
        rows.append({
            "stage": ARGS.stage,
            "fold": fold,
            "arm": arm,
            "name": ARM_NAME[arm],
            "n_features": feat.shape[1],
            "solo_bss": solo,
            "corr": corr,
            "dbss": dbss,
            "t_row": dbss / se,
            "elapsed_sec": round(time.time() - t0, 1),
        })
        np.save(cache_path, pred)
        print(f"  {arm:<3} {ARM_NAME[arm]:<25} f={feat.shape[1]:3d} "
              f"solo={solo:9.2f} corr={corr:.4f} dBSS={dbss:+7.2f} "
              f"t={dbss/se:5.2f} [{time.time()-t0:.0f}s]", flush=True)

result = pd.DataFrame(rows)
path = OUT / f"v75_feature_family_{ARGS.stage}.csv"
result.to_csv(path, index=False)
print(f"\nsaved -> {path}")
for value in ("dbss", "solo_bss"):
    pivot = result.pivot_table(index="arm", columns="fold", values=value)
    if "P0" in pivot.index:
        pivot = pivot.subtract(pivot.loc["P0"], axis=1)
    print(f"\n{value} minus P0\n{pivot.round(3).to_string()}")
