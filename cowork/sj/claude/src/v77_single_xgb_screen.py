"""단일 XGBoost에서 전처리/행 파생/TrackMan family를 비교한다.

기준은 기존 V2R200 + strict-as-of TM500 209피처와 local-2024 trial 93이다.
하이퍼파라미터를 고정하고 피처만 바꾸므로 후보 간 차이를 피처 효과로 해석한다.

screen: Val2024 한 seed. confirm: Val2023/Val2024를 같은 설정으로 재학습한다.
모든 lookup은 fold 이전 데이터 또는 예측 시즌 이전 TrackMan으로 고정된다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


SJ = Path(__file__).resolve().parents[2]
# 2026-08-18 feature_campaign_1000 -> claude/src 이관.
# 데이터/산출물은 캠페인 폴더에 그대로 있으므로 CAMPAIGN 으로 가리킨다.
CAMPAIGN = SJ / "feature_campaign_1000"
MO = SJ / "experiment" / "model_optimization"
sys.path.insert(0, str(MO))
sys.path.insert(0, str(SJ / "claude" / "src"))

import component_features as CF
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import CATEGORICAL_COLUMNS, TARGET, probability_metrics, recency_weights

OUT = CAMPAIGN / "outputs" / "single_xgb"
TM_RELEASE = CAMPAIGN / "outputs" / "trackman_release"
PARAMS_PATH = MO / "xgboost_v2r200_tm500_local_2024_best.json"
RAW_IDS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "confirm"], default="screen")
    parser.add_argument(
        "--arms",
        default="I0,I1,I2,D0,T0,S1,D1,T1",
        help="쉼표로 구분한 arm. B0은 항상 포함한다.",
    )
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--max-estimators", type=int, default=None)
    parser.add_argument("--early-stopping-rounds", type=int, default=220)
    parser.add_argument(
        "--folds", default=None,
        help="선택 실행용 검증 시즌 목록. 예: 2023 또는 2023,2024",
    )
    parser.add_argument(
        "--params-path", default=str(PARAMS_PATH),
        help="기존 Optuna best JSON. best_params를 고정해 사용한다.",
    )
    parser.add_argument("--no-baseline", action="store_true",
                        help="요청 arm만 실행한다. 기준 예측이 이미 있을 때 사용한다.")
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args()


def add_direct_products(frame: pd.DataFrame) -> list[str]:
    """한 행의 원본/기존 파생만 이용한 선택적 2차항을 추가한다."""
    added: dict[str, np.ndarray] = {}
    pairs = [
        ("asof_pitcher_success_rate", "asof_batter_success_rate"),
        ("asof_pitcher_middle_rate", "asof_batter_middle_rate"),
        ("asof_pitcher_success_rate", "li"),
        ("asof_pitcher_reverse_rate", "asof_pitcher_fastball_rate"),
    ]
    for i, (left, right) in enumerate(pairs):
        added[f"sx_product_{i:02d}"] = (
            pd.to_numeric(frame[left], errors="coerce").to_numpy(np.float64)
            * pd.to_numeric(frame[right], errors="coerce").to_numpy(np.float64)
        )
    n = np.log1p(np.nan_to_num(
        pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(np.float64),
        nan=0.0,
    ))
    added["sx_pitcher_success_x_logn"] = (
        frame["asof_pitcher_success_rate"].to_numpy(np.float64) * n)
    added["sx_pitcher_minus_batter_success"] = (
        frame["asof_pitcher_success_rate"].to_numpy(np.float64)
        - frame["asof_batter_success_rate"].to_numpy(np.float64))
    added["sx_pitcher_over_batter_success"] = (
        frame["asof_pitcher_success_rate"].to_numpy(np.float64)
        / np.clip(frame["asof_batter_success_rate"].to_numpy(np.float64), 1e-3, None))

    balls = frame["balls_before"].to_numpy(np.float64)
    strikes = frame["strikes_before"].to_numpy(np.float64)
    inning = frame["inning"].to_numpy(np.float64)
    li = frame["li"].to_numpy(np.float64)
    diff = frame["score_diff_pitcher_team"].to_numpy(np.float64)
    close = (np.abs(diff) <= 1).astype(float)
    added.update({
        "sx_count_margin": balls - strikes,
        "sx_two_strike": (strikes == 2).astype(float),
        "sx_three_ball": (balls == 3).astype(float),
        "sx_full_count": ((balls == 3) & (strikes == 2)).astype(float),
        "sx_close_game": close,
        "sx_late_close": (inning >= 7).astype(float) * close,
        "sx_li_close": li * close,
        "sx_li_count_margin": li * (balls - strikes),
        "sx_reverse_two_strike": (
            frame["asof_pitcher_reverse_rate"].to_numpy(np.float64)
            * (strikes == 2)),
        "sx_ball_three_ball": (
            frame["asof_pitcher_ball_rate"].to_numpy(np.float64)
            * (balls == 3)),
        "sx_prev5_success_x_balls": (
            frame["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
            * balls),
    })
    names = list(added)
    for name, values in added.items():
        frame[name] = values.astype(np.float32)
    return names


def add_trackman_residual(frame: pd.DataFrame, fold: int) -> list[str]:
    lookup_path = TM_RELEASE / f"cutoff_{fold}" / "main_pitcher_release.parquet"
    lookup = pd.read_parquet(lookup_path).set_index("pitcher_id")
    if lookup.index.duplicated().any():
        raise AssertionError(f"duplicate pitcher_id in {lookup_path}")
    source = ["cw_mean_sim", "cw_min_margin", "tr_eligible_seasons",
              "tr_total_pitches", "tr_season_gap"]
    source += [f"tr_resid_pca_{i:02d}" for i in range(12)]
    names = []
    for column in source:
        name = f"sx_{column}"
        values = frame["pitcher_id"].map(lookup[column]).to_numpy(np.float64)
        if column == "tr_total_pitches":
            values = np.log1p(np.nan_to_num(values, nan=0.0))
        frame[name] = values.astype(np.float32)
        names.append(name)
    frame["sx_trackman_release_available"] = (
        frame["pitcher_id"].isin(lookup.index).to_numpy(np.float32))
    names.append("sx_trackman_release_available")
    return names


def build_component_unique(frame: pd.DataFrame, base_features: list[str],
                           fold: int, platoon_k: int = CF.PLATOON_K,
                           count_k: int | None = None,
                           inning_k: int | None = None) -> pd.DataFrame:
    """기존 성분 라인의 fold-safe 계층 차감 피처 중 base에 없는 열만 만든다."""
    labels_path = SJ / "claude" / "cache" / "failure_labels.parquet"
    labels = pd.read_parquet(labels_path)
    if not frame["row_id"].equals(labels["row_id"]):
        raise RuntimeError("failure label cache row order mismatch")
    raw_columns = [column for column in frame.columns[:49]
                   if column not in ("row_id", TARGET)]
    train_mask = frame["season"].lt(fold).to_numpy()
    train = frame.loc[train_mask, [*raw_columns, TARGET]]
    ok = labels["label_ok"].to_numpy() == 1
    middle = np.where(ok, labels["y_middle"].to_numpy(np.float64), np.nan)
    reverse = np.where(ok, labels["y_reverse"].to_numpy(np.float64), np.nan)
    outside = np.where(ok, labels["y_outside"].to_numpy(np.float64), np.nan)
    ball = np.where(ok, labels["y_ball"].to_numpy(np.float64), np.nan)
    component_labels = {
        "m": middle,
        "r": reverse,
        "mr": np.where(ok, ((middle == 1) & (reverse == 1)).astype(float), np.nan),
        "ob": np.where(ok, ((outside == 1) & (ball == 1)).astype(float), np.nan),
        "oz": np.where(ok, ((outside == 1) & (ball == 0)).astype(float), np.nan),
    }
    count_k = platoon_k if count_k is None else count_k
    inning_k = platoon_k if inning_k is None else inning_k
    built = CF.build(
        frame[raw_columns],
        CF.make_spec(train),
        CF.make_platoon_table(train, K=platoon_k),
        CF.make_batter_platoon_table(
            train, {name: values[train_mask]
                    for name, values in component_labels.items()}, K=platoon_k),
        CF.make_count_platoon_table(train, K=count_k),
        CF.make_inning_platoon_table(train, K=inning_k),
    )
    unique = [column for column in built.columns if column not in base_features]
    return built[unique].astype(np.float32).rename(
        columns={column: f"sx_cf_{column}" for column in unique})


def build_component_unique_forward(
        frame: pd.DataFrame, base_features: list[str], fold: int,
        cache: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    """학습행도 시즌 순방향 OOF, 검증행은 fold 이전 lookup으로 만든다.

    2019는 이전 시즌이 없으므로 신규 45열을 0으로 둔다. 2020년 이후 학습행은
    자신의 시즌보다 이전 데이터로만 만든 값을 받고, season==fold 검증행은
    모든 season<fold 데이터로 만든 값을 받는다.
    """
    cache = {} if cache is None else cache

    def cutoff_features(cutoff: int) -> pd.DataFrame:
        if cutoff not in cache:
            cache[cutoff] = build_component_unique(
                frame, base_features, cutoff)
        return cache[cutoff]

    template = cutoff_features(fold)
    output = pd.DataFrame(
        np.zeros(template.shape, dtype=np.float32),
        columns=template.columns,
        index=frame.index,
    )
    seasons = sorted(
        int(value) for value in frame.loc[frame["season"].le(fold), "season"].unique()
        if int(value) >= 2020)
    for season in seasons:
        source = cutoff_features(season)
        mask = frame["season"].eq(season)
        output.loc[mask, :] = source.loc[mask, :].to_numpy(np.float32)
    return output


ARM_NAMES = {
    "B0": "existing 209 features",
    "S1": "remove season",
    "S2": "remove season+game_type",
    "S3": "remove season+raw IDs",
    "S4": "remove season+game_type+raw IDs",
    "I0": "keep season, remove game_type",
    "I1": "keep season, remove raw IDs",
    "I2": "keep season, remove game_type+raw IDs",
    "DP": "keep season+direct product terms",
    "DC": "keep season+context/count terms",
    "D0": "keep season+direct products/context",
    "C0": "keep season+component hierarchy",
    "C1": "component hierarchy+D0",
    "C2": "component hierarchy+D0+TrackMan residual PCA12",
    "C3": "matchup/count/inning hierarchy only",
    "C4": "matchup/count/inning hierarchy+D0",
    "K1": "component hierarchy+D0 with K=100",
    "K6": "component hierarchy+D0 with K=600",
    "K12": "component hierarchy+D0 with K=1200",
    "H6": "K=600 for count+inning hierarchy only",
    "P6": "K=600 for pitcher+batter hierarchy only",
    "CT6": "K=600 for count hierarchy only",
    "IN6": "K=600 for inning hierarchy only",
    "F0": "strict forward-OOF component hierarchy",
    "F1": "strict forward-OOF component hierarchy+D0",
    "F2": "F1 + base_margin = logit(EB(asof_pitcher_success_rate))",
    "D1": "remove season+direct products/context",
    "D2": "remove season+game_type+raw IDs+direct products/context",
    "T0": "D0+TrackMan residual PCA12",
    "T1": "D1+TrackMan residual PCA12",
    "T2": "D2+TrackMan residual PCA12",
}


def arm_features(frame: pd.DataFrame, base: list[str], arm: str, fold: int,
                 component_unique: pd.DataFrame | None = None) -> list[str]:
    features = list(base)
    if arm in ("S1", "S2", "S3", "S4", "D1", "D2", "T1", "T2"):
        features.remove("season")
    if arm in ("S2", "S4", "I0", "I2", "D2", "T2"):
        features.remove("game_type")
    if arm in ("S3", "S4", "I1", "I2", "D2", "T2"):
        features = [column for column in features if column not in RAW_IDS]
    if arm in ("DP", "DC", "D0", "D1", "D2", "T0", "T1", "T2",
               "C1", "C2", "C4", "K1", "K6", "K12",
               "H6", "P6", "CT6", "IN6", "F1", "F2"):
        derived = add_direct_products(frame)
        # add_direct_products는 product/difference 7열 뒤 context/count 11열 순서다.
        if arm == "DP":
            derived = derived[:7]
        elif arm == "DC":
            derived = derived[7:]
        features.extend(derived)
    if arm in ("T0", "T1", "T2", "C2"):
        features.extend(add_trackman_residual(frame, fold))
    if arm in ("C0", "C1", "C2", "C3", "C4", "K1", "K6", "K12",
               "H6", "P6", "CT6", "IN6", "F0", "F1", "F2"):
        if component_unique is None:
            raise ValueError("component_unique is required")
        selected = component_unique
        if arm in ("C3", "C4"):
            suffixes = (
                "platoon_split", "platoon_rel", "platoon_split_w",
                "bat_platoon_split", "bat_platoon_rel", "bat_platoon_split_w",
                "count_platoon_split", "count_platoon_rel", "count_platoon_w",
                "inning_platoon_split", "inning_platoon_rel", "inning_platoon_w",
            )
            selected = component_unique[
                [column for column in component_unique
                 if column.endswith(suffixes) or "_bat_pl_" in column]
            ]
        for column in selected:
            frame[column] = selected[column].to_numpy()
        features.extend(selected.columns.tolist())
    return list(dict.fromkeys(features))


def encode(frame: pd.DataFrame, features: list[str], fold: int):
    train_mask = frame["season"].lt(fold)
    valid_mask = frame["season"].eq(fold)
    train_x = frame.loc[train_mask, features].copy()
    valid_x = frame.loc[valid_mask, features].copy()
    for column in [c for c in CATEGORICAL_COLUMNS if c in features]:
        values = train_x[column].fillna("__MISSING__").astype(str)
        mapping = {value: index for index, value in enumerate(pd.unique(values))}
        train_x[column] = values.map(mapping).astype("int32")
        valid_x[column] = (valid_x[column].fillna("__MISSING__").astype(str)
                           .map(mapping).fillna(-1).astype("int32"))
    return (train_mask, valid_mask,
            train_x.apply(pd.to_numeric, errors="coerce").astype("float32"),
            valid_x.apply(pd.to_numeric, errors="coerce").astype("float32"))


EB_PRIOR_K = 200.0


def pitcher_base_margin(frame: pd.DataFrame, mask, fold: int) -> np.ndarray:
    """logit(EB(투수 as-of 성공률)) - logit(리그 사전확률).

    행 자체의 asof_pitcher_success_rate / asof_pitcher_n 만 쓴다.
    test 행에도 그대로 있으므로 행 독립성과 시간 인과가 모두 유지된다.
    리그 사전확률은 season < fold 의 Target 평균으로 고정한다.
    """
    prior = float(frame.loc[frame["season"].lt(fold), TARGET].mean())
    rate = pd.to_numeric(frame.loc[mask, "asof_pitcher_success_rate"],
                         errors="coerce").to_numpy(np.float64)
    n = pd.to_numeric(frame.loc[mask, "asof_pitcher_n"],
                      errors="coerce").to_numpy(np.float64)
    n = np.nan_to_num(n, nan=0.0).clip(min=0.0)
    eb = (np.nan_to_num(rate, nan=prior) * n + prior * EB_PRIOR_K) / (n + EB_PRIOR_K)
    eb = np.clip(eb, 1e-4, 1 - 1e-4)
    return (np.log(eb / (1 - eb)) - np.log(prior / (1 - prior))).astype(np.float64)


def fit_predict(frame: pd.DataFrame, features: list[str], fold: int,
                params: dict, device: str, n_jobs: int,
                early_stopping_rounds: int,
                random_state: int | None,
                arm: str = "") -> tuple[np.ndarray, int, float]:
    started = time.time()
    train_mask, valid_mask, train_x, valid_x = encode(frame, features, fold)
    train_y = frame.loc[train_mask, TARGET].to_numpy("int8")
    valid_y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    model_params = dict(params)
    half_life = float(model_params.pop("half_life"))
    weights = recency_weights(frame.loc[train_mask, "season"], fold, half_life)
    model = XGBClassifier(
        **model_params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=(random_state if random_state is not None else 20260818 + fold),
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
    )
    if arm == "F2":
        # 학습·검증 모두에 같은 규칙으로 건다. 모델은 잔차만 배운다.
        model.set_params(base_score=0.5)
        bm_tr = pitcher_base_margin(frame, train_mask, fold)
        bm_va = pitcher_base_margin(frame, valid_mask, fold)
        model.fit(train_x, train_y, sample_weight=weights,
                  base_margin=bm_tr,
                  eval_set=[(valid_x, valid_y)],
                  base_margin_eval_set=[bm_va], verbose=False)
        prediction = model.predict_proba(valid_x,
                                         base_margin=bm_va)[:, 1].astype(np.float64)
    else:
        model.fit(train_x, train_y, sample_weight=weights,
                  eval_set=[(valid_x, valid_y)], verbose=False)
        prediction = model.predict_proba(valid_x)[:, 1].astype(np.float64)
    return prediction, int(model.best_iteration), time.time() - started


def main():
    args = parse_args()
    folds = ([int(value) for value in args.folds.split(",") if value]
             if args.folds else
             ([2024] if args.stage == "screen" else [2023, 2024]))
    requested = [value.strip() for value in args.arms.split(",") if value.strip()]
    arms = ([value for value in requested if value != "B0"]
            if args.no_baseline else
            ["B0"] + [value for value in requested if value != "B0"])
    unknown = [value for value in arms if value not in ARM_NAMES]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    params_path = Path(args.params_path)
    params = json.loads(params_path.read_text(encoding="utf-8"))["best_params"]
    if args.max_estimators is not None:
        params["n_estimators"] = args.max_estimators
    frame, base_features = load_enhanced_frame()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in folds:
        valid_mask = frame["season"].eq(fold)
        y = frame.loc[valid_mask, TARGET].to_numpy("int8")
        print(f"\nfold={fold} rows={len(y)} base_features={len(base_features)}", flush=True)
        component_unique = None
        hierarchy_arms = (
            "C0", "C1", "C2", "C3", "C4", "K1", "K6", "K12",
            "H6", "P6", "CT6", "IN6", "F0", "F1", "F2")
        if any(arm in hierarchy_arms for arm in arms):
            component_unique = build_component_unique(frame, base_features, fold)
            print(f"  component unique features={component_unique.shape[1]}", flush=True)
        forward_unique = None
        if any(arm in ("F0", "F1") for arm in arms):
            forward_unique = build_component_unique_forward(
                frame, base_features, fold, cache={fold: component_unique})
            print("  strict forward-OOF hierarchy ready", flush=True)
        k_variants = {}
        for arm, k in (("K1", 100), ("K6", 600), ("K12", 1200)):
            if arm in arms:
                k_variants[arm] = build_component_unique(
                    frame, base_features, fold, platoon_k=k)
        split_k_specs = {
            "H6": {"platoon_k": 300, "count_k": 600, "inning_k": 600},
            "P6": {"platoon_k": 600, "count_k": 300, "inning_k": 300},
            "CT6": {"platoon_k": 300, "count_k": 600, "inning_k": 300},
            "IN6": {"platoon_k": 300, "count_k": 300, "inning_k": 600},
        }
        for arm, spec in split_k_specs.items():
            if arm in arms:
                k_variants[arm] = build_component_unique(
                    frame, base_features, fold, **spec)
        for arm in arms:
            arm_frame = frame.copy(deep=False)
            arm_component = (
                forward_unique if arm in ("F0", "F1")
                else k_variants.get(arm, component_unique))
            features = arm_features(
                arm_frame, base_features, arm, fold,
                arm_component)
            estimator_tag = args.max_estimators or "full"
            params_tag = params_path.stem.replace("_best", "")
            seed_tag = args.random_state if args.random_state is not None else "campaign"
            cache = OUT / (
                f"{args.stage}_{params_tag}_{args.device}_e{estimator_tag}_"
                f"s{seed_tag}_{arm}_{fold}.npy")
            if args.reuse_cache and cache.exists():
                pred = np.load(cache)
                if len(pred) != len(y):
                    raise AssertionError(f"cache row mismatch: {cache}")
                best_iteration, elapsed = -1, 0.0
                marker = "cache"
            else:
                pred, best_iteration, elapsed = fit_predict(
                    arm_frame, features, fold, params, args.device, args.n_jobs,
                    args.early_stopping_rounds, args.random_state, arm)
                np.save(cache, pred)
                marker = "fit"
            score = probability_metrics(y, pred)
            bss_raw = 100000.0 * (1.0 - score["normalized_brier"])
            rows.append({
                "stage": args.stage,
                "fold": fold,
                "arm": arm,
                "name": ARM_NAMES[arm],
                "n_features": len(features),
                "best_iteration": best_iteration,
                "elapsed_sec": elapsed,
                "bss_raw": bss_raw,
                **score,
            })
            print(f"  {arm:<2} {marker:<5} f={len(features):3d} "
                  f"BSSraw={bss_raw:8.2f} mean={score['pred_mean']:.5f} "
                  f"iter={best_iteration:4d} t={elapsed:.1f}s", flush=True)
    result = pd.DataFrame(rows)
    path = OUT / f"single_xgb_{args.stage}.csv"
    result.to_csv(path, index=False)
    pivot = result.pivot_table(index="arm", columns="fold", values="bss_raw")
    if "B0" in pivot.index:
        pivot = pivot.subtract(pivot.loc["B0"], axis=1)
    print(f"\nBSS minus B0\n{pivot.round(3).to_string()}\nsaved -> {path}")


if __name__ == "__main__":
    main()
