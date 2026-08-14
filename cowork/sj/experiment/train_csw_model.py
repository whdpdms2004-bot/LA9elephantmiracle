"""CSW(콜드스트라이크+헛스윙) 예측 모델 — 엄격한 투구 전 시점.

라벨: is_csw = description ∈ {called_strike, swinging_strike, swinging_strike_blocked}
      (data/statcast_2017_2019_raw_csw.parquet에 사전 계산됨)

예측 시점 규칙 (팀 결정 2026-07-16):
  투수가 공을 던지기 **전**에 알 수 있는 정보만 사용한다.
  - 제외: 현재 투구의 구종(pitch_type), 모든 물리값/궤적/위치, 릴리스 자세,
          당일 존 측정치(sz_top/sz_bot), 결과·WPA 계열 전부
  - 허용: 경기 상황, 과거 투구의 물리값(lag), 과거 누적/rolling 통계,
          투수의 최근 포심 커맨드 상태(kirby cmd_ff_*), 수비 배치(투구 전 확정)

분할: train 2017–2018 / test 2019 (시간순)

실행:
  python train_csw_model.py            # 전체 (로컬 권장, 수 분 소요)
  python train_csw_model.py --demo     # 2018만 사용한 빠른 검증 (train 3–7월 / test 8월~)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_statcast_strike_dataset import (
    SEQUENCE_COLUMNS, BuildConfig, build_features, make_model_dataset,
)
from kirby_index import add_prepitch_ff_command_features

RAW_CSW_FILE = Path("data/statcast_2017_2019_raw_csw.parquet")
REPORT_DIR = Path("reports/csw_model")

# 엄격한 투구 전 시점에서 추가로 제거해야 하는 열
STRICT_PREPITCH_DROP = {
    "sz_top", "sz_bot",          # 현재 투구 중 측정되는 존 상·하단
    "is_strike",                 # 이전 타깃(현재 투구 결과)
    "pitch_type", "pitch_name",  # 현재 구종 (구종 결정 전 예측)
}

EXCLUDE_AS_ID = {
    "game_pk", "game_date", "game_year", "at_bat_number", "pitch_number",
    "pitcher", "batter", "fielder_2", "player_name", "home_team", "away_team",
    "game_type", "split", "is_csw", "_source_row", "_sequence_in_chunk",
}

LOAD_COLUMNS = [
    "game_pk", "game_date", "game_year", "game_type", "home_team", "away_team",
    "at_bat_number", "pitch_number", "pitcher", "player_name", "batter", "fielder_2",
    "type", "description", "pitch_type", "balls", "strikes",
    "outs_when_up", "inning", "inning_topbot", "on_1b", "on_2b", "on_3b",
    "home_score", "away_score", "bat_score", "fld_score", "stand", "p_throws",
    "if_fielding_alignment", "of_fielding_alignment",
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x", "plate_z",
    "release_pos_x", "release_pos_y", "release_pos_z", "release_extension",
    "vx0", "vy0", "vz0", "ax", "ay", "az", "delta_home_win_exp",
    "is_csw",
]


def load_dataset(years: tuple[int, ...]) -> pd.DataFrame:
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(RAW_CSW_FILE).schema_arrow.names)
    cols = [c for c in LOAD_COLUMNS if c in available]
    df = pd.read_parquet(RAW_CSW_FILE, columns=cols)
    df = df[df["game_year"].isin(years)]
    return df.reset_index(drop=True)


def add_csw_history(df: pd.DataFrame, window: int = 100, min_periods: int = 20) -> pd.DataFrame:
    """투수/타자의 과거 CSW 성향 (현재 행 제외: shift(1))."""
    out = df.sort_values(SEQUENCE_COLUMNS, kind="stable").copy()
    lbl = out["is_csw"].astype("float32")
    for side in ["pitcher", "batter"]:
        g = lbl.groupby(out[side], sort=False)
        out[f"{side}_csw_rate_before"] = g.transform(
            lambda s: s.shift(1).expanding(min_periods=min_periods).mean()
        ).astype("float32")
        out[f"{side}_csw_rate_last{window}"] = g.transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
        ).astype("float32")
    return out


def build_model_frame(raw: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """기존 파이프라인 피처 + CSW 이력 + 투구 전 커맨드 피처 → 엄격 모드 정리."""
    raw = add_prepitch_ff_command_features(raw)          # 과거 FF 각도 SD (누수 없음)
    raw = add_csw_history(raw)

    enriched = build_features(raw, config)               # 상황+lag+rolling+LI (누수 방지 내장)
    model = make_model_dataset(enriched, config)         # 현재 투구 물리/결과 열 제거
    from build_statcast_strike_dataset import CURRENT_PITCH_LEAKAGE
    from kirby_index import COMMAND_FEATURE_LEAKAGE
    model = model.drop(columns=[
        c for c in (STRICT_PREPITCH_DROP | COMMAND_FEATURE_LEAKAGE) if c in model.columns
    ])

    # 안전망: 현재 투구에서만 알 수 있는 열이 남아있으면 즉시 실패
    leaked = (CURRENT_PITCH_LEAKAGE | COMMAND_FEATURE_LEAKAGE | STRICT_PREPITCH_DROP
              ) & set(model.columns)
    assert not leaked, f"leakage columns remain: {leaked}"
    assert "is_csw" in model.columns
    return model


def feature_columns(model: pd.DataFrame) -> list[str]:
    return [c for c in model.columns if c not in EXCLUDE_AS_ID]


def to_matrix(model: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = model[features].copy()
    for c in X.columns:
        if X[c].dtype == object or isinstance(X[c].dtype, pd.CategoricalDtype) \
                or pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X


def evaluate(y_true, prob) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def train_and_eval(model: pd.DataFrame, features: list[str], *, train_mask, test_mask,
                   max_iter: int = 300, sample_train: int | None = None, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    train = model[train_mask]
    if sample_train and len(train) > sample_train:
        train = train.sample(sample_train, random_state=seed)
    X_tr = to_matrix(train, features)
    X_te = to_matrix(model[test_mask], features)
    # train 기준으로 카테고리 정렬 통일
    for c in X_tr.columns:
        if isinstance(X_tr[c].dtype, pd.CategoricalDtype):
            cats = X_tr[c].cat.categories
            X_te[c] = pd.Categorical(X_te[c], categories=cats)

    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=63,
        min_samples_leaf=100, l2_regularization=1.0,
        categorical_features="from_dtype", early_stopping=False, random_state=seed,
    )
    clf.fit(X_tr, train["is_csw"].astype("int8"))
    prob = clf.predict_proba(X_te)[:, 1]
    metrics = evaluate(model.loc[test_mask, "is_csw"].astype("int8"), prob)
    metrics["n_train"] = int(len(X_tr))
    metrics["n_features"] = len(features)
    return {"model": clf, "metrics": metrics}


def permutation_top_features(clf, X_te, y_te, features, n=15, seed=0) -> pd.Series:
    from sklearn.inspection import permutation_importance
    sub = X_te.sample(min(50_000, len(X_te)), random_state=seed)
    r = permutation_importance(clf, sub, y_te.loc[sub.index], n_repeats=3,
                               random_state=seed, scoring="roc_auc")
    return pd.Series(r.importances_mean, index=features).sort_values(ascending=False).head(n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="2018년만 사용한 빠른 검증")
    parser.add_argument("--sample-train", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()

    config = BuildConfig()
    if args.demo:
        raw = load_dataset((2018,))
    else:
        raw = load_dataset((2017, 2018, 2019))
    print(f"[load] {len(raw):,} rows")

    model = build_model_frame(raw, config)
    if args.demo:
        dates = pd.to_datetime(model["game_date"])
        train_mask = dates < "2018-08-01"
        test_mask = ~train_mask
        print("[demo split] 2018-03~07 train / 2018-08~ test")
    else:
        train_mask = model["split"].eq("train").to_numpy()
        test_mask = model["split"].eq("test").to_numpy()

    features = feature_columns(model)
    cmd_features = [c for c in features if c.startswith("cmd_") or c.endswith("_csw_rate_before")
                    or "_csw_rate_last" in c]
    base_features = [c for c in features if c not in set(cmd_features)]
    print(f"[features] total {len(features)} (base {len(base_features)} + cmd/csw {len(cmd_features)})")

    results = {}
    for name, feats in [("base", base_features), ("full", features)]:
        out = train_and_eval(model, feats, train_mask=train_mask, test_mask=test_mask,
                             max_iter=args.max_iter, sample_train=args.sample_train)
        results[name] = out
        print(f"[{name}] " + json.dumps(out["metrics"], indent=None))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": "is_csw (called_strike + swinging_strike + swinging_strike_blocked)",
        "prediction_time": "strict pre-pitch (no current pitch_type/physics/zone measurements)",
        "split": "demo 2018 within-season" if args.demo else "train 2017-2018 / test 2019",
        "metrics": {k: v["metrics"] for k, v in results.items()},
        "ablation_delta_auc": results["full"]["metrics"]["roc_auc"] - results["base"]["metrics"]["roc_auc"],
        "features": {"base": base_features, "cmd_csw": cmd_features},
    }
    tag = "demo" if args.demo else "full"
    (REPORT_DIR / f"metrics_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {REPORT_DIR / f'metrics_{tag}.json'}")


if __name__ == "__main__":
    main()
