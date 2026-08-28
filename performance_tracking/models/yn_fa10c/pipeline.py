"""FA10C 공용 학습 파이프라인.

최종 제출과 walk-forward 검증이 같은 피처 생성·모델 학습 코드를 공유한다.
최종 구성은 71피처(기존 68 + A 오염보정 3), LGB20 + numeric-CB20 +
team-CB20, 0.10/0.90 결합, isotonic, 상한 0.80이다.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import features

ID_COL = "row_id"
TARGET = "control_success"
A_COLS = [
    "fe_pitcher_futures_share",
    "fe_batter_futures_share",
    "fe_pitcher_prior_n_log",
]
TEAM_ID_COLS = ["pitcher_team_id", "batter_team_id"]
LGB_SEEDS = list(range(20))
CB_SEEDS = list(range(20))
LGB_WEIGHT = 0.10
TEAM_ALPHA = 1.00
ISO_CAP = 0.80

LGB_PARAMS = dict(
    num_leaves=63,
    learning_rate=0.02,
    n_estimators=200,
    min_child_samples=300,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_split_gain=0,
    feature_fraction=0.8,
    subsample=0.8,
    n_jobs=-1,
    verbosity=-1,
)
CB_PARAMS = dict(
    iterations=500,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    bagging_temperature=2,
    random_strength=2,
    min_data_in_leaf=1,
    verbose=False,
    thread_count=-1,
)

CAP_NOTE = (
    "isotonic 출력 상한을 0.8로 클램프. 근거: 검증(2024) 최상단 bin이 단 2행(둘 다 성공)으로 "
    "적합돼 1.0으로 매핑됐는데 인접 구간 실제 성공률은 0.632로, 2개 표본에서 나온 과신이다. "
    "cap 적용 시 검증 R-only Brier 손실은 0.000000이나, 미적용 시 최악(해당 29행 전부 실패) "
    "손실은 +0.000114로 이번 변경의 전체 이득(+0.000072)을 초과한다. 학습데이터만으로 결정한 "
    "상수이며 test.csv 값·분포·평균과 리더보드 점수를 사용하지 않았다. 전 행 동일 적용."
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def find_data_dir(explicit: Optional[str] = None) -> Path:
    """train.csv와 test.csv가 모두 있는 데이터 디렉터리를 찾는다."""
    if explicit:
        candidates = [Path(explicit)]
    else:
        here = Path(__file__).resolve()
        candidates = [
            Path.cwd() / "data",
            here.parents[3] / "data",  # LA9elephantmiracle/data
            here.parents[4] / "data",  # open/data (현재 yn 로컬 배치)
        ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "train.csv").is_file() and (candidate / "test.csv").is_file():
            return candidate
    shown = ", ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"train.csv와 test.csv가 함께 있는 data 디렉터리를 찾지 못함: {shown}")


def raw_feature_list(data_dir: Path) -> list[str]:
    cols = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig", nrows=0).columns
    return [c for c in cols if c != ID_COL]


def load_train(data_dir: Path, raw_features: Sequence[str]) -> pd.DataFrame:
    usecols = list(dict.fromkeys([ID_COL, *raw_features, TARGET]))
    frame = pd.read_csv(
        data_dir / "train.csv",
        encoding="utf-8-sig",
        usecols=usecols,
    )
    if frame[TARGET].isna().any():
        frame = frame.loc[frame[TARGET].notna()].reset_index(drop=True)
    return frame


def build_futures_lookup(history: pd.DataFrame) -> Tuple[dict, dict, dict]:
    """과거 공식 train 행에서 선수별 F 비중과 투수 표본수를 계산한다."""
    is_f = (history["game_type"] == "F").astype(float)
    pitcher = history.assign(_is_f=is_f).groupby("pitcher_id")["_is_f"].agg(["mean", "size"])
    batter = history.assign(_is_f=is_f).groupby("batter_id")["_is_f"].mean()
    return pitcher["mean"].to_dict(), pitcher["size"].to_dict(), batter.to_dict()


def build_a_features(rows: pd.DataFrame, full_history: pd.DataFrame) -> pd.DataFrame:
    """시즌 S 행에는 season < S 이력만 사용해 A피처를 붙인다."""
    out = pd.DataFrame(index=rows.index, columns=A_COLS, dtype=float)
    for season in sorted(rows["season"].dropna().unique()):
        hist = full_history.loc[full_history["season"] < season]
        if hist.empty:
            continue
        pitcher_share, pitcher_n, batter_share = build_futures_lookup(hist)
        mask = rows["season"] == season
        part = rows.loc[mask]
        out.loc[mask, A_COLS[0]] = part["pitcher_id"].map(pitcher_share).to_numpy()
        out.loc[mask, A_COLS[1]] = part["batter_id"].map(batter_share).to_numpy()
        out.loc[mask, A_COLS[2]] = np.log1p(part["pitcher_id"].map(pitcher_n).to_numpy(dtype=float))
    return out


def prepare(
    rows: pd.DataFrame,
    full_history: pd.DataFrame,
    raw_features: Sequence[str],
    stats: dict,
    cat_cols: Sequence[str],
    category_levels: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """기존 68피처 뒤에 A피처 3개를 순서 고정해 추가한다."""
    x = features.add_derived_features(rows[list(raw_features)].copy(), stats)
    a = build_a_features(rows, full_history)
    x = pd.concat([x, a[A_COLS]], axis=1)
    if category_levels is None:
        category_levels = features.get_category_levels(x, cat_cols)
    x = features.apply_category_levels(x, category_levels)
    if x.shape[1] != 71:
        raise ValueError(f"FA10C 피처 수가 71이 아님: {x.shape[1]}")
    return x, category_levels


def build_cb(seed: int):
    from catboost import CatBoostClassifier

    params = dict(CB_PARAMS)
    if os.environ.get("FA10C_THREAD_COUNT"):
        params["thread_count"] = int(os.environ["FA10C_THREAD_COUNT"])
    return CatBoostClassifier(random_seed=seed, **params)


def cb_pool(x: pd.DataFrame, y, cat_cols: Sequence[str], team_mode: bool):
    from catboost import Pool

    x2 = x.copy()
    for col in cat_cols:
        if team_mode:
            values = x2[col].astype(object)
            x2[col] = values.where(pd.notna(values), "__MISSING__").astype(str)
        else:
            x2[col] = x2[col].astype(str)
    return Pool(x2, y, cat_features=list(cat_cols)) if y is not None else Pool(
        x2, cat_features=list(cat_cols)
    )


def train_family(
    tag: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: Optional[pd.DataFrame],
    cat_cols: Sequence[str],
    seeds: Sequence[int],
    kind: str,
    checkpoint_dir: Path,
) -> Optional[np.ndarray]:
    """모델 단위 체크포인트를 사용해 중단 후 재개 가능한 학습을 수행한다."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preds = np.zeros(len(x_valid), dtype=float) if x_valid is not None else None
    train_pool = valid_pool = None
    if kind != "lgb":
        team_mode = kind == "cb_team"
        pool_cats = list(cat_cols) + (TEAM_ID_COLS if team_mode else [])
        train_pool = cb_pool(x_train, y_train, pool_cats, team_mode)
        valid_pool = cb_pool(x_valid, None, pool_cats, team_mode) if x_valid is not None else None

    for seed in seeds:
        model_path = checkpoint_dir / f"{tag}_{kind}_{seed}.model"
        pred_path = checkpoint_dir / f"{tag}_{kind}_{seed}.npy"
        if model_path.exists() and (x_valid is None or pred_path.exists()):
            if preds is not None:
                preds += np.load(pred_path)
            continue

        started = time.time()
        if kind == "lgb":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)
            model.fit(x_train, y_train, categorical_feature=list(cat_cols))
            pred = model.predict_proba(x_valid)[:, 1] if x_valid is not None else None
            model.booster_.save_model(str(model_path))
        else:
            model = build_cb(seed)
            model.fit(train_pool)
            pred = model.predict_proba(valid_pool)[:, 1] if x_valid is not None else None
            model.save_model(str(model_path))

        if pred is not None:
            np.save(pred_path, pred)
            preds += pred
        log(f"{tag}/{kind} seed={seed} 완료 ({time.time() - started:.1f}s)")

    return preds / len(seeds) if preds is not None else None


def train_predictions_for_cutoff(
    full: pd.DataFrame,
    raw_features: Sequence[str],
    cutoff: int,
    valid_season: int,
    checkpoint_dir: Path,
    seeds: Sequence[int] = tuple(LGB_SEEDS),
    include_numeric_cb: bool = False,
) -> Dict[str, np.ndarray]:
    """season<=cutoff 학습 후 valid_season raw 성분 예측을 만든다."""
    if valid_season <= cutoff:
        raise ValueError("검증 시즌은 학습 cutoff보다 뒤여야 함")
    train_rows = full.loc[full["season"] <= cutoff].reset_index(drop=True)
    valid_rows = full.loc[full["season"] == valid_season].reset_index(drop=True)
    if train_rows.empty or valid_rows.empty:
        raise ValueError(f"빈 fold: cutoff={cutoff}, valid={valid_season}")

    cat_cols = features.get_categorical_columns()
    stats = features.fit_stats(train_rows)
    x_train, levels = prepare(train_rows, full, raw_features, stats, cat_cols)
    x_valid, _ = prepare(valid_rows, full, raw_features, stats, cat_cols, levels)
    y_train = train_rows[TARGET].to_numpy(dtype=float)
    tag = f"cutoff{cutoff}_valid{valid_season}"
    log(f"{tag}: train={len(train_rows):,}, valid={len(valid_rows):,}, features={x_train.shape[1]}")

    pred_lgb = train_family(
        tag, x_train, y_train, x_valid, cat_cols, seeds, "lgb", checkpoint_dir
    )
    pred_team = train_family(
        tag, x_train, y_train, x_valid, cat_cols, seeds, "cb_team", checkpoint_dir
    )
    result = {"lgb": pred_lgb, "team": pred_team}
    if include_numeric_cb:
        result["numeric"] = train_family(
            tag, x_train, y_train, x_valid, cat_cols, seeds, "cb_num", checkpoint_dir
        )
    return result


def combine_raw(parts: Dict[str, np.ndarray]) -> np.ndarray:
    numeric = parts.get("numeric")
    cb_mix = parts["team"] if numeric is None else (
        (1.0 - TEAM_ALPHA) * numeric + TEAM_ALPHA * parts["team"]
    )
    return LGB_WEIGHT * parts["lgb"] + (1.0 - LGB_WEIGHT) * cb_mix


def fit_isotonic(raw_pred: np.ndarray, target: np.ndarray) -> Tuple[list, list]:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_pred, target)
    iso_y = [min(float(value), ISO_CAP) for value in iso.y_thresholds_]
    return iso.X_thresholds_.tolist(), iso_y


def build_inference_lookup(full: pd.DataFrame, cutoff: int = 2024) -> dict:
    history = full.loc[full["season"] <= cutoff]
    pitcher_share, pitcher_n, batter_share = build_futures_lookup(history)
    return {
        "pitcher_share": {str(k): float(v) for k, v in pitcher_share.items()},
        "pitcher_n": {str(k): float(v) for k, v in pitcher_n.items()},
        "batter_share": {str(k): float(v) for k, v in batter_share.items()},
    }


def copy_final_models(checkpoint_dir: Path, package_model_dir: Path) -> None:
    package_model_dir.mkdir(parents=True, exist_ok=True)
    for seed in LGB_SEEDS:
        shutil.copy2(
            checkpoint_dir / f"full_lgb_{seed}.model",
            package_model_dir / f"lgb_booster_{seed}.txt",
        )
    for seed in CB_SEEDS:
        shutil.copy2(
            checkpoint_dir / f"full_cb_num_{seed}.model",
            package_model_dir / f"cb_model_{seed}.cbm",
        )
        shutil.copy2(
            checkpoint_dir / f"full_cb_team_{seed}.model",
            package_model_dir / f"cb_team_model_{seed}.cbm",
        )


def meta_payload(
    raw_features: Sequence[str],
    cat_cols: Sequence[str],
    category_levels: dict,
    stats: dict,
    iso_x: list,
    iso_y: list,
    lookup: dict,
) -> dict:
    return {
        "raw_features": list(raw_features),
        "cat_cols": list(cat_cols),
        "category_levels": category_levels,
        "stats": stats,
        "iso_x": iso_x,
        "iso_y": iso_y,
        "lgb_seeds": LGB_SEEDS,
        "cb_seeds": CB_SEEDS,
        "team_cb_seeds": CB_SEEDS,
        "team_cat_cols": list(cat_cols) + TEAM_ID_COLS,
        "ensemble_lgb_weight": LGB_WEIGHT,
        "ensemble_cb_weight": 1.0 - LGB_WEIGHT,
        "team_representation_alpha": TEAM_ALPHA,
        "team_representation_numeric_weight": 1.0 - TEAM_ALPHA,
        "team_representation_teamcat_weight": TEAM_ALPHA,
        "calibration_method": "in-sample isotonic on 2024 validation predictions of this exact structure",
        "calibration_source": "official train.csv 2019-2023 -> 2024 validation; no test/LB information",
        "new_feature_cols": A_COLS,
        "futures_lookup": lookup,
        "feature_note": (
            "A(1군/2군 오염보정) 3피처를 기존 68피처 블록 맨 뒤에 추가해 71피처. "
            "학습 시에는 시즌 S 행에 cutoff=S-1 walk-forward lookup을, 추론 시에는 "
            "cutoff=2024 고정 lookup을 pitcher_id/batter_id로 merge한다. 공식 train.csv "
            "라벨만 사용했고 test.csv 값/분포/평균과 리더보드 점수를 일절 쓰지 않았다."
        ),
        "calibration_cap_note": CAP_NOTE,
    }


def environment_versions() -> dict:
    import catboost
    import lightgbm
    import sklearn

    return {
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
    }
