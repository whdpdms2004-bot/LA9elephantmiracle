"""추론 스크립트 — 평가 서버가 이 파일을 그대로 실행한다.

model/lgb_booster_{0..19}.txt + model/cb_model_{0..19}.cbm(numeric CatBoost) +
model/cb_team_model_{0..19}.cbm(team-ID 범주형 CatBoost) + model/meta.json 을 읽어
data/test.csv 를 예측하고 output/submission.csv 를 생성한다.

예측 순서(학습 src 쪽과 반드시 동일해야 함):
  CB_mix = (1-alpha)*numericCB평균 + alpha*teamcatCB평균
  raw    = w_lgb*LGB평균 + (1-w_lgb)*CB_mix
  final  = isotonic(raw) -> clip(0,1)

피처는 71개다 = 기존 68개 + 아래 3개(1군/2군 오염 보정):
  fe_pitcher_futures_share : 그 투수의 과거 기록 중 2군(game_type=='F') 비중
  fe_batter_futures_share  : 타자쪽 동일
  fe_pitcher_prior_n_log   : log1p(그 투수의 과거 기록 행 수)

이 3개는 meta.json에 미리 구워둔 lookup(cutoff=2024, 공식 train.csv 기반)을
pitcher_id / batter_id 로 조회해서 붙인다 — 평가 서버에 train.csv가 있다는 보장이
없으므로 원본을 다시 읽지 않는다. lookup에 없는 신규 선수는 NaN이며 LightGBM/
CatBoost가 결측을 네이티브로 처리한다.

행 독립성: 모든 피처가 해당 행의 컬럼값 + 사전 저장된 상수 lookup만으로 계산되므로
predict(단독 행) == predict(전체 test)[i] 를 만족한다. test.csv의 다른 행을 쓰는
집계/rolling/분포 보정은 일절 없다.

모델은 sklearn pickle 없이 네이티브 포맷(LightGBM 텍스트, CatBoost .cbm)으로 저장하고
isotonic은 좌표 + np.interp로 재현한다 — 로컬/평가서버 라이브러리 버전 차이로 언피클이
깨질 위험을 없애기 위함이다.
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ID_COL = "row_id"
TARGET_COL = "control_success"

SMOOTH_GROUPS = {
    "asof_pitcher_n": [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
    ],
    "asof_batter_n": [
        "asof_batter_success_rate", "asof_batter_middle_rate",
    ],
    "asof_pitcher_pitchmix_n": [
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ],
}
NO_N_RATE_COLS = [
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]
SMOOTHING_K = 200


# =======================
# 피처 생성 (src/features.py의 add_derived_features와 동일 로직)
# =======================
def add_derived_features(df, stats):
    df = df.copy()

    df["count_state"] = df["balls_before"].astype(str) + df["strikes_before"].astype(str)
    df["is_two_strikes"] = (df["strikes_before"] == 2).astype(int)
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
    df["is_pitcher_favor"] = (df["strikes_before"] > df["balls_before"]).astype(int)
    df["is_batter_favor"] = (df["balls_before"] > df["strikes_before"]).astype(int)

    df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)

    df["inning_group"] = pd.cut(
        df["inning"], bins=[0, 3, 6, 99], labels=["1-3", "4-6", "7+"]
    ).astype(str)
    df["scoring_position"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype(int)
    df["score_diff_abs"] = df["score_diff_pitcher_team"].abs()
    df["li_bucket"] = pd.cut(df["li"], bins=stats["li_bin_edges"], labels=False).astype(float)

    for n_col, rate_cols in SMOOTH_GROUPS.items():
        n = df[n_col].fillna(0)
        for col in rate_cols:
            league_avg = stats[f"league_avg__{col}"]
            smoothed = (df[col].fillna(0) * n + SMOOTHING_K * league_avg) / (n + SMOOTHING_K)
            df[f"smooth_{col}"] = smoothed
    for col in NO_N_RATE_COLS:
        df[col] = df[col].fillna(stats[f"league_avg__{col}"])

    return df


def add_futures_features(X, test_df, lookup, new_cols):
    """1군/2군 오염 보정 3피처를 **기존 블록 맨 뒤에** 붙인다.

    meta.json에 저장된 cutoff=2024 고정 lookup을 pitcher_id/batter_id로 조회할 뿐이며,
    test.csv의 다른 행을 참조하지 않는다(행 독립성 유지)."""
    p_share = {int(k): v for k, v in lookup["pitcher_share"].items()}
    p_n = {int(k): v for k, v in lookup["pitcher_n"].items()}
    b_share = {int(k): v for k, v in lookup["batter_share"].items()}

    pid = test_df["pitcher_id"]
    bid = test_df["batter_id"]
    add = pd.DataFrame(index=X.index)
    add["fe_pitcher_futures_share"] = pid.map(p_share).astype(float).to_numpy()
    add["fe_batter_futures_share"] = bid.map(b_share).astype(float).to_numpy()
    add["fe_pitcher_prior_n_log"] = np.log1p(pid.map(p_n).astype(float).to_numpy())
    return pd.concat([X, add[new_cols]], axis=1)


def apply_category_levels(df, category_levels):
    df = df.copy()
    for col, levels in category_levels.items():
        df[col] = pd.Categorical(df[col].astype(str), categories=levels)
    return df


def build_features(df, raw_features, stats, category_levels, lookup, new_cols):
    X = df[raw_features].copy()
    X = add_derived_features(X, stats)
    X = add_futures_features(X, df, lookup, new_cols)
    X = apply_category_levels(X, category_levels)
    return X


def predict_lgb_bagged(boosters, X):
    if len(X) == 0:
        return np.array([])
    preds = np.zeros(len(X))
    for booster in boosters:
        preds += np.asarray(booster.predict(X, num_iteration=booster.best_iteration))
    return preds / len(boosters)


def predict_cb_bagged(models, X, cat_cols):
    """numeric CatBoost — 학습(Pool 생성) 때와 동일하게 문자열 변환."""
    if len(X) == 0:
        return np.array([])
    X2 = X.copy()
    for c in cat_cols:
        X2[c] = X2[c].astype(str)
    pool = Pool(X2, cat_features=cat_cols)
    preds = np.zeros(len(X))
    for m in models:
        preds += m.predict_proba(pool)[:, 1]
    return preds / len(models)


def predict_cb_team_bagged(models, X, cat_cols):
    """team-ID 범주형 CatBoost — 학습 때와 같은 결측 문자열 처리."""
    if len(X) == 0:
        return np.array([])
    X2 = X.copy()
    for c in cat_cols:
        values = X2[c].astype(object)
        X2[c] = values.where(pd.notna(values), "__MISSING__").astype(str)
    pool = Pool(X2, cat_features=cat_cols)
    preds = np.zeros(len(X))
    for m in models:
        preds += m.predict_proba(pool)[:, 1]
    return preds / len(models)


def apply_isotonic(preds, iso_x, iso_y):
    return np.interp(preds, iso_x, iso_y)


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}")
    return df


def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


def main():
    TEST_DIR, MODEL_DIR, OUT_DIR = "./data", "./model", "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load model...")
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    raw_features = meta["raw_features"]
    cat_cols = meta["cat_cols"]
    stats = meta["stats"]
    category_levels = meta["category_levels"]
    iso_x, iso_y = meta["iso_x"], meta["iso_y"]
    lgb_seeds, cb_seeds = meta["lgb_seeds"], meta["cb_seeds"]
    team_cb_seeds, team_cat_cols = meta["team_cb_seeds"], meta["team_cat_cols"]
    team_alpha = float(meta["team_representation_alpha"])
    lookup, new_cols = meta["futures_lookup"], meta["new_feature_cols"]

    lgb_boosters = [lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lgb_booster_{s}.txt"))
                    for s in lgb_seeds]
    cb_models = []
    for s in cb_seeds:
        m = CatBoostClassifier()
        m.load_model(os.path.join(MODEL_DIR, f"cb_model_{s}.cbm"))
        cb_models.append(m)
    team_cb_models = []
    for s in team_cb_seeds:
        m = CatBoostClassifier()
        m.load_model(os.path.join(MODEL_DIR, f"cb_team_model_{s}.cbm"))
        team_cb_models.append(m)
    print(f" OK. LightGBM {len(lgb_boosters)}개, numeric CatBoost {len(cb_models)}개, "
          f"team CatBoost {len(team_cb_models)}개, num_feature={lgb_boosters[0].num_feature()}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, raw_features, stats, category_levels, lookup, new_cols)
    print(f" features={X.shape[1]}")

    print("Inference model...")
    lgb_pred = predict_lgb_bagged(lgb_boosters, X)
    cb_numeric_pred = predict_cb_bagged(cb_models, X, cat_cols)
    cb_team_pred = predict_cb_team_bagged(team_cb_models, X, team_cat_cols)
    cb_pred = (1.0 - team_alpha) * cb_numeric_pred + team_alpha * cb_team_pred
    lgb_weight = float(meta.get("ensemble_lgb_weight", 0.5))
    cb_weight = float(meta.get("ensemble_cb_weight", 1.0 - lgb_weight))
    if not np.isclose(lgb_weight + cb_weight, 1.0, atol=1e-12):
        raise ValueError(f"ensemble weights must sum to 1: {lgb_weight}, {cb_weight}")
    preds_raw = (lgb_weight * lgb_pred + cb_weight * cb_pred if len(X) else np.array([]))
    preds = apply_isotonic(preds_raw, iso_x, iso_y) if len(preds_raw) else preds_raw
    preds = np.clip(preds, 0.0, 1.0)
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
