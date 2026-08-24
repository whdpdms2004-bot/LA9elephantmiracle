from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
DEST = HERE / "optuna_max_performance_plan.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# 제구 성공 확률 — 성능 최우선 모델·Optuna 계획

이 노트북은 팀 전체가 같은 검증 규칙으로 모델을 나눠 튜닝하기 위한 실행 골격이다.

최종 구조는 다음과 같다.

1. 시간 순방향 CV와 정규화 Brier 목적함수 고정
2. CatBoost, LightGBM, XGBoost를 **서로 다른 Optuna study**로 탐색
3. 안전 기본 피처 → OOF 투수 임베딩 → 과거 Trackman 요약 순으로 ablation
4. 상위 모델을 여러 seed로 재학습
5. 시간 OOF 예측을 비음수 블렌딩하고 cross-fitted calibration

현재 투수 임베딩 단독 최고 2024 BSS는 363.23이고 공식 베이스라인 기준은 549.51이다. 임베딩은 단독 최종 모델이 아니라 부스팅 피처와 앙상블 다양성에 우선 사용한다.

> 보조 reverse/middle/far 라벨 모델은 운영진 허용 답변 전까지 `experimental`로 격리한다. 이 노트북의 기본 target은 공개된 `control_success` 하나다.
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 1. 실행 설정

- 기본값은 `RUN_STUDY=False`라서 셀을 실행해도 수백 trial이 시작되지 않는다.
- 팀원은 서로 다른 `MODEL_FAMILY`와 `STUDY_NAME`을 정해 병렬로 작업한다.
- study DB와 결과물은 이 폴더 아래에 저장한다.
- 실제 대규모 탐색 전 `QUICK_RUN=True`로 전체 파이프라인을 1~2 trial 확인한다.
"""))

cells.append(nbf.v4.new_code_cell(r"""from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from IPython.display import display
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

import catboost
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path.cwd()
if not (ROOT / "data" / "train.csv").exists():
    candidates = [Path.cwd().parent, Path.cwd().parent.parent, Path.cwd().parent.parent.parent]
    ROOT = next((p for p in candidates if (p / "data" / "train.csv").exists()), ROOT)

WORK_DIR = ROOT / "experiment" / "model_optimization"
WORK_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB = WORK_DIR / "optuna_studies.db"
STORAGE_URL = os.environ.get("OPTUNA_STORAGE_URL", f"sqlite:///{DEFAULT_DB.as_posix()}")

RUN_STUDY = False
QUICK_RUN = True
MODEL_FAMILY = "catboost"       # catboost / lightgbm / xgboost
FEATURE_VERSION = "V0"          # V0 / V1 / V3
N_TRIALS = 2 if QUICK_RUN else 300
ACTIVE_FOLDS = [2023, 2024] if QUICK_RUN else [2022, 2023, 2024]

versions = {
    "python": sys.version.split()[0], "platform": platform.platform(),
    "optuna": optuna.__version__, "catboost": catboost.__version__,
    "lightgbm": lgb.__version__, "xgboost": xgb.__version__,
}
print(json.dumps(versions, indent=2, ensure_ascii=False))
print("ROOT:", ROOT)
print("storage:", STORAGE_URL)
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 2. 평가 지표와 robust objective

공식 BSS를 바로 최대화하지 않고 각 fold의 `Brier / [r(1-r)]`를 최소화한다. 점수가 0에서 잘리는 구간에서도 trial 간 차이가 보존되기 때문이다.

```text
NB_f = Brier_f / (r_f * (1-r_f))
weighted_mean = 0.15*NB_2022 + 0.30*NB_2023 + 0.55*NB_2024
robust = 0.75*weighted_mean + 0.25*max(NB_f)
```

빠른 모드처럼 일부 fold만 사용할 때는 해당 연도 가중치를 다시 정규화한다.
"""))

cells.append(nbf.v4.new_code_cell(r"""FOLD_WEIGHTS = {2022: 0.15, 2023: 0.30, 2024: 0.55}

def probability_metrics(y_true, probability):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    brier = float(np.mean((p - y) ** 2))
    rate = float(y.mean())
    reference = rate * (1.0 - rate)
    normalized_brier = brier / reference if reference > 0 else np.inf
    bss = max(0.0, 100000.0 * (1.0 - normalized_brier))
    return {
        "brier": brier,
        "normalized_brier": normalized_brier,
        "bss": bss,
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
        "target_mean": rate,
        "pred_mean": float(p.mean()),
        "mean_gap": float(p.mean() - rate),
    }

def robust_temporal_objective(fold_metrics, mean_ratio=0.75):
    available = sorted(fold_metrics)
    weights = np.array([FOLD_WEIGHTS[y] for y in available], dtype=float)
    weights /= weights.sum()
    ratios = np.array([fold_metrics[y]["normalized_brier"] for y in available])
    weighted_mean = float(np.dot(weights, ratios))
    worst = float(ratios.max())
    return mean_ratio * weighted_mean + (1.0 - mean_ratio) * worst

# metric sanity check: perfect < useful model < constant
_y = np.array([0, 0, 1, 1])
display(pd.DataFrame({
    "perfect": probability_metrics(_y, _y)["normalized_brier"],
    "constant": probability_metrics(_y, np.repeat(_y.mean(), len(_y)))["normalized_brier"],
}, index=[0]))
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 3. 시간 fold와 데이터

| 검증 시즌 | 학습 시즌 |
|---:|---|
| 2022 | 2019~2021 |
| 2023 | 2019~2022 |
| 2024 | 2019~2023 |

무작위 K-fold는 사용하지 않는다. 최근성은 `history_window`와 `half_life`로 trial마다 튜닝하되, 각 fold의 검증 시즌을 기준으로만 계산한다.
"""))

cells.append(nbf.v4.new_code_cell(r"""TARGET = "control_success"
DROP_COLUMNS = ["row_id", TARGET]

CATEGORICAL_COLUMNS = [
    # season은 2025로 외삽해야 하므로 수치형으로 유지한다.
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]

def load_training_frame(quick_run=False):
    train = pd.read_csv(ROOT / "data" / "train.csv")
    if quick_run:
        # 시간 분포를 유지한 결정적 표본. 실제 정밀 탐색에는 사용하지 않는다.
        train = (train.groupby("season", group_keys=False)
                      .apply(lambda x: x.sample(min(len(x), 50_000), random_state=SEED))
                      .sort_index()
                      .reset_index(drop=True))
    return train

train_df = load_training_frame(QUICK_RUN)
print(train_df.shape)
display(train_df.groupby("season")[TARGET].agg(["size", "mean"]))
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 4. 피처 버전

- `V0`: 공식 입력 피처 그대로. `row_id`만 제외.
- `V1`: 안전한 행 단위 교호작용, 표본 수 로그, 장·단기 rate 차이.
- `V3`: V1 + 시즌 순방향 OOF 투수 임베딩. 2019~2020은 0 fallback과 availability 표시.

Trackman 원시 로그의 season-forward 집계(V4)는 별도 팀 피처 모듈이 확정된 뒤 같은 인터페이스로 추가한다. test 전체 빈도나 분포는 어떤 버전에서도 사용하지 않는다.
"""))

cells.append(nbf.v4.new_code_cell(r"""def add_rowwise_features(frame):
    x = frame.copy()
    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["runner_out_state"] = x["base_state"].astype(str) + "_o" + x["outs_before"].astype(str)
    x["handedness_matchup"] = x["pitcher_hand"].astype(str) + "_" + x["batter_hand"].astype(str)
    x["score_abs"] = x["score_diff_pitcher_team"].abs()
    x["late_inning"] = (x["inning"] >= 7).astype("int8")
    x["high_leverage"] = (x["li"] >= 2.0).astype("int8")

    for col in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        x[f"log1p_{col}"] = np.log1p(x[col].clip(lower=0))

    long_rate = x["asof_pitcher_success_rate"]
    for n in [1, 3, 5]:
        x[f"pitcher_success_delta_prev{n}"] = x[f"asof_pitcher_prev{n}_game_success_rate"] - long_rate
        x[f"pitcher_middle_delta_prev{n}"] = (
            x[f"asof_pitcher_prev{n}_game_middle_rate"] - x["asof_pitcher_middle_rate"]
        )
    x["ball_strike_rate_sum_gap"] = x["asof_pitcher_ball_rate"] + x["asof_pitcher_strike_rate"] - 1.0
    return x

def add_oof_pitcher_embedding(frame):
    path = ROOT / "experiment" / "pitcher_embedding" / "outputs" / "pitcher_season_embedding_oof.parquet"
    emb = pd.read_parquet(path)
    keep = ["pitcher_id", "season", "oof_available", "pitcher_known_before_season"] + [
        c for c in emb.columns if c.startswith(("pitcher_embedding_", "trackman_embedding_", "cohort_embedding_"))
    ]
    if emb[["pitcher_id", "season"]].duplicated().any():
        raise ValueError("OOF embedding key is not unique")
    out = frame.merge(emb[keep], on=["pitcher_id", "season"], how="left", validate="many_to_one")
    emb_cols = [c for c in keep if c not in {"pitcher_id", "season"}]
    out[emb_cols] = out[emb_cols].fillna(0)
    return out

def build_feature_frame(frame, version="V0"):
    if version not in {"V0", "V1", "V3"}:
        raise ValueError(version)
    out = frame.copy()
    cats = list(CATEGORICAL_COLUMNS)
    if version in {"V1", "V3"}:
        out = add_rowwise_features(out)
        cats += ["count_state", "runner_out_state", "handedness_matchup"]
    if version == "V3":
        out = add_oof_pitcher_embedding(out)
    cats = [c for c in cats if c in out.columns]
    features = [c for c in out.columns if c not in DROP_COLUMNS]
    return out, features, cats

model_df, FEATURE_COLUMNS, CAT_COLUMNS = build_feature_frame(train_df, FEATURE_VERSION)
print("feature version:", FEATURE_VERSION, "features:", len(FEATURE_COLUMNS), "categorical:", len(CAT_COLUMNS))
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 5. 모델 후보와 권장 순서

| 단계 | 후보 | 역할 | 권장 trial |
|---|---|---|---:|
| 기준선 | Logistic/ElasticNet, 공식 RF, ExtraTrees | 확률 anchor·오류 다양성 | 50~200 |
| 주력 1 | CatBoost | 고유값 큰 선수/팀 범주형 | 400~700 |
| 주력 2 | LightGBM | 수치·rate·임베딩, 빠른 광역 탐색 | 400~700 |
| 주력 3 | XGBoost hist | 강한 정규화와 앙상블 다양성 | 300~600 |
| 표현 | Direct MLP, two-tower, FT-Transformer | OOF embedding·비선형 상호작용 | 200~400 |
| 최종 | 비음수 blend + Platt/Beta/Isotonic | Brier 직접 개선 | 1000~5000 |

모델군을 하나의 조건부 study에 섞지 않는다. 각 family가 가진 최적화 공간이 너무 다르므로 TPE가 따로 학습하는 편이 효율적이다.
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 6. Optuna sampler·pruner·공통 탐색 축

- TPE의 multivariate/group 모드로 파라미터 상호작용을 함께 본다.
- 병렬 worker에서는 `constant_liar=True`로 같은 구간을 중복 탐색하는 현상을 줄인다.
- 부스팅은 native early stopping + fold 단위 pruning.
- 신경망은 epoch 단위 Hyperband pruning.
- SQLite DB로 중단·재개하고 팀원별 study 이름을 분리한다.
"""))

cells.append(nbf.v4.new_code_cell(r"""def make_sampler(seed=SEED):
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=40,
        n_ei_candidates=48,
        multivariate=True,
        group=True,
        constant_liar=True,
    )

def make_pruner():
    base = optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=1, interval_steps=1)
    return optuna.pruners.PatientPruner(base, patience=1)

def suggest_common(trial):
    return {
        "history_window": trial.suggest_categorical("history_window", [1, 2, 3, 4, 99]),
        "half_life": trial.suggest_float("half_life", 0.35, 8.0, log=True),
    }

def temporal_training_mask(frame, valid_year, history_window):
    lower = int(frame["season"].min()) if history_window == 99 else valid_year - int(history_window)
    return frame["season"].between(lower, valid_year - 1)

def recency_weights(seasons, valid_year, half_life):
    age = np.maximum(valid_year - np.asarray(seasons, dtype=float), 1.0)
    w = np.power(0.5, age / float(half_life))
    return (w / np.mean(w)).astype("float32")

def suggest_catboost(trial):
    bootstrap = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
    p = {
        "iterations": trial.suggest_int("iterations", 800, 8000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.12, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 300.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-4, 30.0, log=True),
        "bootstrap_type": bootstrap,
        "rsm": trial.suggest_float("rsm", 0.55, 1.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", [2, 8, 32, 128]),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
        "od_wait": trial.suggest_int("od_wait", 100, 500),
    }
    if bootstrap == "Bayesian":
        p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    else:
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    return p

def suggest_lightgbm(trial):
    max_depth = trial.suggest_categorical("max_depth", [-1, 4, 5, 6, 7, 8, 9, 10, 12])
    max_leaves = 511 if max_depth == -1 else min(511, 2 ** max_depth)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 12000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, max(15, max_leaves), log=True),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 10000, log=True),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 300.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255, 511]),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        "cat_l2": trial.suggest_float("cat_l2", 1e-3, 100.0, log=True),
    }

def suggest_xgboost(trial):
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 12000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.10, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 2000.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 300.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [64, 128, 256, 512]),
        "grow_policy": grow_policy,
    }
    if grow_policy == "lossguide":
        p["max_leaves"] = trial.suggest_int("max_leaves", 16, 256, log=True)
    return p
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 7. fold별 전처리와 모델 학습

숫자 기반 LightGBM/XGBoost의 범주 코드는 **각 fold 학습 데이터에서만** 만든다. 검증에만 등장하는 값은 `-1`이다. CatBoost는 문자열 범주와 `__MISSING__`을 사용한다.

기본 early stopping은 CatBoost Brier, LightGBM custom Brier, XGBoost logloss를 감시한다. 최종 Optuna 평가는 세 모델 모두 동일한 robust normalized Brier다.
"""))

cells.append(nbf.v4.new_code_cell(r"""def prepare_catboost(train_x, valid_x, cat_columns):
    a, b = train_x.copy(), valid_x.copy()
    for c in cat_columns:
        a[c] = a[c].fillna("__MISSING__").astype(str)
        b[c] = b[c].fillna("__MISSING__").astype(str)
    return a, b

def prepare_numeric_booster(train_x, valid_x, cat_columns):
    a, b = train_x.copy(), valid_x.copy()
    for c in cat_columns:
        train_values = a[c].fillna("__MISSING__").astype(str)
        valid_values = b[c].fillna("__MISSING__").astype(str)
        mapping = {v: i for i, v in enumerate(pd.unique(train_values))}
        a[c] = train_values.map(mapping).astype("int32")
        b[c] = valid_values.map(mapping).fillna(-1).astype("int32")
    for c in a.columns:
        if c not in cat_columns:
            a[c] = pd.to_numeric(a[c], errors="coerce")
            b[c] = pd.to_numeric(b[c], errors="coerce")
    return a, b

def lgb_brier_eval(y_true, y_pred):
    return "brier", float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)), False

def fit_one_fold(family, params, train_x, train_y, valid_x, valid_y, cat_columns, weights, seed):
    if family == "catboost":
        tx, vx = prepare_catboost(train_x, valid_x, cat_columns)
        train_pool = Pool(tx, label=train_y, cat_features=cat_columns, weight=weights)
        valid_pool = Pool(vx, label=valid_y, cat_features=cat_columns)
        model = CatBoostClassifier(
            **params, loss_function="Logloss", eval_metric="BrierScore",
            grow_policy="SymmetricTree", random_seed=seed,
            task_type="GPU" if os.environ.get("USE_CATBOOST_GPU", "0") == "1" else "CPU",
            verbose=False, allow_writing_files=False,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=params["od_wait"])
        pred = model.predict_proba(valid_pool)[:, 1]
        best_iteration = model.get_best_iteration()

    elif family == "lightgbm":
        tx, vx = prepare_numeric_booster(train_x, valid_x, cat_columns)
        model = LGBMClassifier(
            **params, objective="binary", metric="None", random_state=seed,
            n_jobs=6, verbosity=-1, device_type="cpu",
        )
        model.fit(
            tx, train_y, sample_weight=weights,
            eval_set=[(vx, valid_y)], eval_metric=lgb_brier_eval,
            callbacks=[lgb.early_stopping(300, first_metric_only=True, verbose=False)],
            categorical_feature=cat_columns,
        )
        pred = model.predict_proba(vx, num_iteration=model.best_iteration_)[:, 1]
        best_iteration = model.best_iteration_

    elif family == "xgboost":
        tx, vx = prepare_numeric_booster(train_x, valid_x, cat_columns)
        model = XGBClassifier(
            **params, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device="cuda" if os.environ.get("USE_XGB_GPU", "0") == "1" else "cpu",
            random_state=seed, n_jobs=6, early_stopping_rounds=300,
        )
        model.fit(tx, train_y, sample_weight=weights, eval_set=[(vx, valid_y)], verbose=False)
        pred = model.predict_proba(vx)[:, 1]
        best_iteration = getattr(model, "best_iteration", None)
    else:
        raise ValueError(family)

    return model, pred, best_iteration
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 8. 모델별 objective와 study 실행

fold가 하나 끝날 때마다 현재 robust objective를 report한다. 첫 fold 전에는 pruning하지 않는다. 모든 부스팅 family가 같은 objective를 반환하므로 study 간 값을 바로 비교할 수 있다.

실전에서는 팀원별 예시처럼 이름을 분리한다.

- `cat_v1_worker_a_20260805`
- `lgb_v3_worker_b_20260805`
- `xgb_v3_worker_c_20260805`
"""))

cells.append(nbf.v4.new_code_cell(r"""def build_objective(family, frame, feature_columns, cat_columns, active_folds):
    if family not in {"catboost", "lightgbm", "xgboost"}:
        raise ValueError(family)

    def objective(trial):
        common = suggest_common(trial)
        if family == "catboost":
            params = suggest_catboost(trial)
        elif family == "lightgbm":
            params = suggest_lightgbm(trial)
        else:
            params = suggest_xgboost(trial)

        fold_metrics = {}
        fold_best_iterations = {}
        started = time.time()

        for fold_idx, valid_year in enumerate(sorted(active_folds)):
            train_mask = temporal_training_mask(frame, valid_year, common["history_window"])
            valid_mask = frame["season"].eq(valid_year)
            if train_mask.sum() == 0 or valid_mask.sum() == 0:
                raise optuna.TrialPruned(f"empty temporal fold {valid_year}")

            train_part = frame.loc[train_mask]
            valid_part = frame.loc[valid_mask]
            train_x = train_part[feature_columns]
            valid_x = valid_part[feature_columns]
            train_y = train_part[TARGET].to_numpy("int8")
            valid_y = valid_part[TARGET].to_numpy("int8")
            weights = recency_weights(train_part["season"], valid_year, common["half_life"])

            model, pred, best_iter = fit_one_fold(
                family, params, train_x, train_y, valid_x, valid_y,
                cat_columns, weights, seed=SEED + trial.number * 17 + fold_idx,
            )
            fold_metrics[valid_year] = probability_metrics(valid_y, pred)
            fold_best_iterations[valid_year] = best_iter

            running = robust_temporal_objective(fold_metrics)
            trial.report(running, step=fold_idx)
            trial.set_user_attr(f"fold_{valid_year}", fold_metrics[valid_year])
            trial.set_user_attr(f"best_iteration_{valid_year}", best_iter)
            del model, pred, train_x, valid_x, train_part, valid_part
            gc.collect()

            if fold_idx >= 1 and trial.should_prune():
                raise optuna.TrialPruned()

        value = robust_temporal_objective(fold_metrics)
        trial.set_user_attr("elapsed_sec", time.time() - started)
        trial.set_user_attr("fold_best_iterations", fold_best_iterations)
        return value

    return objective

STUDY_NAME = f"{MODEL_FAMILY}_{FEATURE_VERSION.lower()}_seed{SEED}"
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_URL,
    direction="minimize",
    sampler=make_sampler(),
    pruner=make_pruner(),
    load_if_exists=True,
)

if RUN_STUDY:
    study.optimize(
        build_objective(MODEL_FAMILY, model_df, FEATURE_COLUMNS, CAT_COLUMNS, ACTIVE_FOLDS),
        n_trials=N_TRIALS,
        gc_after_trial=True,
        show_progress_bar=True,
    )
else:
    print("RUN_STUDY=False — 설정과 데이터만 검증했습니다.")
    print("study:", STUDY_NAME, "existing trials:", len(study.trials))
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 9. 결과표와 상위 trial 재평가

`best_trial` 하나를 바로 채택하지 않는다. 상위 20개 설정을 전체 fold에서 3개 seed로 다시 학습하고 평균·표준편차·최악 fold를 비교한다. family별 최종 3~5개만 OOF pool에 남긴다.
"""))

cells.append(nbf.v4.new_code_cell(r"""def study_leaderboard(study, top_n=20):
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rows = []
    for t in sorted(complete, key=lambda z: z.value)[:top_n]:
        row = {"trial": t.number, "objective": t.value, "elapsed_sec": t.user_attrs.get("elapsed_sec")}
        for year in [2022, 2023, 2024]:
            metric = t.user_attrs.get(f"fold_{year}")
            if metric:
                row[f"bss_{year}"] = metric["bss"]
                row[f"nb_{year}"] = metric["normalized_brier"]
                row[f"gap_{year}"] = metric["mean_gap"]
        row.update({f"param__{k}": v for k, v in t.params.items()})
        rows.append(row)
    return pd.DataFrame(rows)

leaderboard = study_leaderboard(study)
display(leaderboard)
if not leaderboard.empty:
    leaderboard.to_csv(WORK_DIR / f"{STUDY_NAME}_leaderboard.csv", index=False)
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 10. OOF 블렌딩과 확률 보정

각 모델은 동일한 시간 OOF 행에 대해 `row_id, season, y, pred, model_name`을 저장해야 한다. 블렌딩 study는 모델을 다시 학습하지 않으므로 1000~5000 trial도 빠르다.

권장 순서:

1. probability 공간 비음수 가중 평균
2. logit 공간 비음수 가중 평균
3. 모델군 weight cap
4. cross-fitted Platt 또는 beta calibration
5. isotonic은 모든 최근 fold에서 안정적으로 개선될 때만 사용
"""))

cells.append(nbf.v4.new_code_cell(r"""def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)

def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))

def normalized_weights(raw):
    w = np.asarray(raw, dtype=float)
    return w / max(w.sum(), 1e-12)

def blend_predictions(matrix, weights, space="probability"):
    w = normalized_weights(weights)
    if space == "probability":
        return np.asarray(matrix) @ w
    if space == "logit":
        return sigmoid(logit(matrix) @ w)
    raise ValueError(space)

def fit_calibrator(mode, fit_y, fit_p):
    fit_y = np.asarray(fit_y)
    fit_p = np.clip(np.asarray(fit_p), 1e-6, 1 - 1e-6)
    if mode == "none":
        return lambda p: np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    if mode == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(fit_p).reshape(-1, 1), fit_y)
        return lambda p: model.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    if mode == "beta":
        z = np.column_stack([np.log(fit_p), np.log1p(-fit_p)])
        model = LogisticRegression(C=1e6, solver="lbfgs").fit(z, fit_y)
        return lambda p: model.predict_proba(np.column_stack([
            np.log(np.clip(p, 1e-6, 1 - 1e-6)),
            np.log1p(-np.clip(p, 1e-6, 1 - 1e-6)),
        ]))[:, 1]
    if mode == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(fit_p, fit_y)
        return lambda p: model.predict(np.asarray(p))
    raise ValueError(mode)

def make_blend_objective(oof_wide, model_columns, calibration_year=2023, eval_year=2024):
    # calibration_year OOF로 보정기를 적합하고 미래 eval_year에서 평가한다.
    def objective(trial):
        raw = [trial.suggest_float(f"w__{c}", 0.0, 1.0) for c in model_columns]
        if sum(raw) < 1e-8:
            raise optuna.TrialPruned("all-zero blend")
        space = trial.suggest_categorical("space", ["probability", "logit"])
        mode = trial.suggest_categorical("calibration", ["none", "platt", "beta", "isotonic"])

        cal = oof_wide[oof_wide.season.eq(calibration_year)]
        val = oof_wide[oof_wide.season.eq(eval_year)]
        cal_raw = blend_predictions(cal[model_columns].to_numpy(), raw, space)
        val_raw = blend_predictions(val[model_columns].to_numpy(), raw, space)
        calibrator = fit_calibrator(mode, cal[TARGET].to_numpy(), cal_raw)
        val_pred = calibrator(val_raw)
        metrics = probability_metrics(val[TARGET].to_numpy(), val_pred)
        trial.set_user_attr(f"fold_{eval_year}", metrics)
        return metrics["normalized_brier"]
    return objective

print("OOF 파일이 준비되면 make_blend_objective(...)로 별도 blend study를 실행합니다.")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 11. 탐색 운영표

| 단계 | 데이터 | fold | 모델당 trial | 목적 |
|---|---:|---|---:|---|
| S0 | 시즌별 5만 이하 | 2023, 2024 | 10~20 | 코드·메모리·속도 확인 |
| S1 | 시간층화 25~35% | 2023, 2024 | 150~300 | 넓은 탐색 |
| S2 | 전체 147만 | 2022~2024 | 250~500 | 정밀 탐색 |
| S3 | 전체 | 2022~2024, 3 seeds | 상위 20 | seed 안정성 |
| S4 | 전체 | 2022~2024, 5 seeds | family별 3~5 | OOF pool |
| S5 | 캐시 OOF | 시간 순방향 | 1000~5000 | blend·calibration |

### 최종 선택 체크리스트

- 모든 시간 fold에서 상수 확률보다 개선
- 공식 RF 재현 모델보다 2024 BSS 개선
- 3~5 seed 평균이 좋고 편차가 작음
- 2024 예측 평균과 실제 성공률 차이가 작음
- 앙상블 후보끼리 잔차 상관이 충분히 다름
- 현재 행 정보와 frozen 과거 lookup만 사용
- 245,789행, 10분, 28GB RAM 제출 환경 검증
- reverse/middle/far 보조 라벨은 운영진 허용 시에만 활성화
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 12. 다음 실행 순서

1. `V0 + 공식 RF`를 같은 fold로 재현한다.
2. `V0 CatBoost/LightGBM/XGBoost` S1을 병렬 수행한다.
3. 상위 부스팅 설정으로 V1, V3 ablation을 먼저 끝낸다.
4. 확정 피처로 S2~S4를 수행한다.
5. Direct MLP와 투수 임베딩을 재튜닝해 OOF pool에 추가한다.
6. OOF 블렌딩·보정 study를 실행한다.
7. 전체 2019~2024 재학습 후 2025 frozen lookup과 제출 코드를 만든다.

세부 근거와 전체 탐색 범위는 같은 폴더의 `MODEL_OPTUNA_PLAN.md`에 정리되어 있다.
"""))

nb.cells = cells
nbf.write(nb, DEST)
print(DEST)
