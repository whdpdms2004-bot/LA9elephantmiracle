"""submission_v8 최종 모델 빌드 -- 다른 팀원 컴퓨터에서 대신 돌려주실 때를 위한 버전.

정희원 로컬(catboost_local/train_best_model_v8.py)과 로직은 100% 동일하고,
데이터 경로만 저장소 루트(data/) 기준으로 바꿨습니다.

## 실행 전 확인
- 이 저장소의 `data/train.csv`, `data/test.csv`가 이미 자리에 있어야 합니다
  (AGENTS.md A6 참고 -- git엔 없고 대회 페이지에서 각자 받아 `data/`에 넣는 파일).
- 필요 패키지: `pip install catboost==1.2.8 lightgbm` (lightgbm 없으면 자동으로
  안내 메시지가 뜹니다).

## 실행
    cd cowork/hw
    python train_best_model_v8.py

CatBoost 5시드 + LightGBM 10시드, family-weighted 50:50 앙상블입니다.
정희원 로컬(12코어) 기준 CatBoost 1개당 6~7분 걸렸습니다 -- 더 좋은 사양이면
훨씬 빠를 것으로 예상합니다.

## 중간에 꺼져도 괜찮습니다 (체크포인트 지원)
`phase1_progress.jsonl`에 멤버별 결과가 즉시 저장되고, `model/`에 이미 저장된
파일은 건너뜁니다. 중단됐으면 그냥 다시 `python train_best_model_v8.py` 실행하면
끊긴 지점부터 이어집니다.

## 끝나면
`cowork/hw/submission_v8/model/` 안에 CatBoost .cbm 5개 + LightGBM .txt 10개 +
medians.json + feature_cols.json + requirements.txt가 생깁니다. 이 폴더 그대로
`cowork/hw/`에 커밋해서 올려주시면 정희원이 이어받습니다 (자기 폴더라 자유 푸시
가능 -- AGENTS.md A2).

끝나면 꼭 `cowork/task.jsonl`에 한 줄 추가해주세요 (author는 실행해주신 분
이니셜로):
    echo '{"id":"<타임스탬프>Z-<이니셜>","author":"<이니셜>","ts":"<ISO8601+09:00>","type":"exp","title":"v8 CatBoost+LightGBM 앙상블 대신 실행","detail":"정희원 노트북이 느려서 대신 학습, 결과는 cowork/hw/submission_v8/에 저장","paths":["cowork/hw/submission_v8/"],"result":"<Phase1 출력의 delta 값 적기>","next":"정희원이 script.py 붙여서 제출 검증"}' >> cowork/task.jsonl

---

원본(정희원 실험 로그) 설계 배경:

팀 공유 리포트(903.37점, LightGBM 20-seed + CatBoost 10-seed, family-weighted
50:50 앙상블)에서 확인된 핵심 교훈을 우리 anchor에 적용한다:
    1. 하이퍼파라미터를 한 번도 제대로 튜닝 안 했다 -- 리포트 팀도 동일한
       상태였는데, 정규화를 덜 하고(L2 낮춤) depth를 늘리고 개별모델의
       분산을 배깅으로 상쇄하는 방향으로 튜닝해서 큰 개선(+151점)을 얻음.
    2. 앙상블을 "전체 N개 단순평균"이 아니라 "LightGBM 평균":"CatBoost 평균"
       = 50:50으로 모델 종류별 가중을 맞춰야 한다(개수가 다르면 단순평균은
       개수 많은 쪽에 암묵적으로 쏠림).
    3. feature 추가보다 모델링(튜닝+배깅+앙상블)이 훨씬 안정적으로 리더보드에
       반영됐다 -- 이건 우리 프로젝트의 반복된 경험(tm_pred_8, v6)과도
       정확히 일치.

feature set = 순수 anchor(baseline47 + trend6, 53개)만 사용. v7(구 하이퍼파라미터,
30개 단순평균)과 원인을 분리하기 위해 "하이퍼파라미터 + 앙상블 결합방식"이라는
변수만 바꿨다.

하이퍼파라미터 (리포트 값을 참고해 조정, 정확한 숫자를 맹신하지 않고 우리
데이터로 Phase1 early stopping을 통해 iteration 수는 직접 확인):
    CatBoost: depth=8(6->8), l2_leaf_reg=3(25->3), random_strength=2(0.6->2),
              bootstrap_type=Bayesian+bagging_temperature=2(Bernoulli+
              subsample=0.7 대체), min_data_in_leaf=1
    LightGBM: min_data_in_leaf=300(20->300), lambda_l1=1/lambda_l2=1(25->1),
              feature_fraction/bagging_fraction=0.8(0.7->0.8)

시드: LightGBM 10개(2026~2035), CatBoost 5개(2026~2030) -- 정희원 로컬 실측 결과
CatBoost가 production 1개당 6~7분으로 가장 느려서(20개면 그것만 2시간+), 전체
소요시간을 줄이려고 리포트 원안(20/10)보다 세트를 더 줄임. 비율(LightGBM >
CatBoost)은 유지.

★ 규정 준수: 이 스크립트가 쓰는 모든 상수(하이퍼파라미터, 결합비율 50:50)는
외부 공유 리포트와 우리 자체 홀드아웃(Val2024) 검증에서만 나온 값입니다.
리더보드 점수를 역산하거나 참조한 값은 하나도 없습니다 (cowork/RULES.md §2).

PHASE 1 (fit<2024, val==2024): 전체 멤버 1회만 학습, family-weighted 50:50
앙상블 BSS를 anchor(650.54) 대비 확인 + best_iteration 확보.
PHASE 2 (fit=season<2025): production 재학습, model/ 저장.

실행:
    py train_best_model_v8.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("lightgbm이 설치되어 있지 않습니다. `py -m pip install lightgbm` 실행 후 다시 시도하세요.") from e

BASE_DIR = Path(__file__).resolve().parent          # cowork/hw
REPO_DIR = Path(__file__).resolve().parents[2]       # cowork/hw -> cowork -> 저장소 루트
DATA_DIR = REPO_DIR / "data"                         # 저장소 루트의 data/ (AGENTS.md A6)
OUT_DIR = BASE_DIR / "submission_v8"
MODEL_DIR = OUT_DIR / "model"
PROGRESS_FILE = OUT_DIR / "phase1_progress.jsonl"  # 체크포인트: 멤버 하나 끝날 때마다 즉시 append

# 체크포인트 사용법:
#   - 중간에 꺼졌다가 다시 python train_best_model_v8.py 실행하면
#     phase1_progress.jsonl에 이미 기록된 멤버는 재학습 없이 best_iteration만 재사용하고,
#     model/ 안에 이미 저장된 .cbm/.txt 파일이 있는 멤버는 Phase 2도 건너뛴다.
#   - 즉 "중간에 끄면 처음부터 다시" 문제가 사라진다 -- 끊긴 지점부터 이어서 진행.

ID = "row_id"
TARGET = "control_success"
BASELINE_CATS = ["top_bottom", "game_type", "base_state"]
PREV_PAIRS = (1, 3, 5)
SANITY_VAL_SEASON = 2024
KNOWN_ANCHOR_BSS_2024 = 650.54
SANITY_TOL = 30.0

CAT_SEEDS = list(range(2026, 2026 + 5))    # 5개 (기존 10개 -- CatBoost가 시드당 6~7분으로 제일 느려서 축소)
LGB_SEEDS = list(range(2026, 2026 + 10))   # 10개 (기존 20개 -- 전체 소요시간을 줄이기 위해 절반으로)
MAX_ITER = 1000
EARLY_STOP = 100

CB_BASE_PARAMS = dict(
    loss_function="Logloss", eval_metric="BrierScore", depth=8,
    learning_rate=0.03, l2_leaf_reg=3, random_strength=2, min_data_in_leaf=1,
    border_count=128, thread_count=-1, grow_policy="Depthwise", boosting_type="Plain",
    bootstrap_type="Bayesian", bagging_temperature=2,
    verbose=False, od_type="Iter", od_wait=EARLY_STOP, allow_writing_files=False,
)

LGB_BASE_PARAMS = dict(
    objective="binary", metric="None", num_leaves=63, learning_rate=0.02,
    lambda_l1=1.0, lambda_l2=1.0, min_data_in_leaf=300, min_gain_to_split=0.0,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
)


def log(msg, t0):
    print(f"[{time.time()-t0:7.1f}s] {msg}")


def score(y, p):
    y = np.asarray(y)
    brier = float(np.mean((y - p) ** 2))
    r = y.mean()
    base = r * (1 - r)
    return brier, (max(0.0, 100000 * (1 - brier / base)) if base > 0 else 0.0)


def pred_stats(p):
    return {
        "pred_mean": round(float(np.mean(p)), 5), "pred_std": round(float(np.std(p)), 5),
        "pred_min": round(float(np.min(p)), 5), "pred_max": round(float(np.max(p)), 5),
    }


def add_trend(df):
    x = df.copy()
    for k in PREV_PAIRS:
        recent = f"asof_pitcher_prev{k}_game_success_rate"
        x[f"trend_prev{k}"] = x[recent] - x["asof_pitcher_success_rate"]
        x[f"trend_abs_prev{k}"] = x[f"trend_prev{k}"].abs()
    return x


def build_cat_maps(fit_df, cat_cols):
    maps = {}
    for c in cat_cols:
        vals = fit_df[c].fillna("__NA__").astype(str)
        uniq = sorted(vals.unique().tolist())
        if "__NA__" not in uniq:
            uniq.append("__NA__")
        maps[c] = uniq
    return maps


def matrix_catboost(df, cols, num_cols, med):
    x = df[cols].copy()
    x[num_cols] = x[num_cols].fillna(med)
    for c in BASELINE_CATS:
        if c in x.columns:
            x[c] = x[c].fillna("__NA__").astype(str)
    return x


def matrix_lightgbm(df, cols, num_cols, med, cat_maps):
    x = df[cols].copy()
    x[num_cols] = x[num_cols].fillna(med)
    for c in BASELINE_CATS:
        if c in x.columns:
            vals = x[c].fillna("__NA__").astype(str)
            x[c] = pd.Categorical(vals, categories=cat_maps[c]).codes
    return x


def brier_feval(preds, train_data):
    y = train_data.get_label()
    return "brier", float(np.mean((y - preds) ** 2)), False


def train_cb_member(seed, x_fit, y_fit, x_val=None, y_val=None, fixed_iter=None):
    params = dict(CB_BASE_PARAMS)
    params["random_seed"] = seed
    if fixed_iter is not None:
        params["iterations"] = fixed_iter
        model = CatBoostClassifier(**params)
        model.fit(x_fit, y_fit, cat_features=BASELINE_CATS)
        return model, fixed_iter
    params["iterations"] = MAX_ITER
    model = CatBoostClassifier(**params)
    model.fit(x_fit, y_fit, cat_features=BASELINE_CATS,
              eval_set=(x_val, y_val), use_best_model=True)
    return model, model.get_best_iteration()


def train_lgb_member(seed, x_fit, y_fit, x_val=None, y_val=None, fixed_iter=None):
    params = dict(LGB_BASE_PARAMS)
    params["seed"] = seed
    train_set = lgb.Dataset(x_fit, label=y_fit, categorical_feature=BASELINE_CATS, free_raw_data=False)
    if fixed_iter is not None:
        model = lgb.train(params, train_set, num_boost_round=fixed_iter)
        return model, fixed_iter
    valid_set = lgb.Dataset(x_val, label=y_val, reference=train_set, categorical_feature=BASELINE_CATS)
    model = lgb.train(
        params, train_set, num_boost_round=MAX_ITER, valid_sets=[valid_set],
        feval=brier_feval,
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    return model, model.best_iteration


def load_progress():
    """이미 완료된 Phase1 멤버의 best_iteration을 체크포인트 파일에서 읽어온다."""
    done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["key"]] = rec
    return done


def append_progress(key, seed, best_iter, bss):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "seed": seed, "best_iter": best_iter, "bss": bss}, ensure_ascii=False) + "\n")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)  # 파일로 리다이렉트해도 한 줄씩 바로 flush
    except AttributeError:
        pass
    t0 = time.time()
    if not (DATA_DIR / "train.csv").exists():
        raise FileNotFoundError(
            f"{DATA_DIR / 'train.csv'} 이 없습니다. AGENTS.md A6 참고해서 대회 페이지에서 "
            f"train.csv를 받아 저장소 루트의 data/ 폴더에 넣어주세요 (git엔 없는 파일입니다)."
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    done = load_progress()
    if done:
        print(f"[체크포인트 발견] 이미 완료된 Phase1 멤버 {len(done)}개 -- 이어서 실행합니다.")

    raw = pd.read_csv(DATA_DIR / "train.csv")
    test_cols = pd.read_csv(DATA_DIR / "test.csv", nrows=0).columns.tolist()
    baseline_47 = [c for c in test_cols if c != ID]
    trend_cols = [f"trend_prev{k}" for k in PREV_PAIRS] + [f"trend_abs_prev{k}" for k in PREV_PAIRS]
    anchor_cols = baseline_47 + trend_cols
    num_cols = [c for c in anchor_cols if c not in BASELINE_CATS]
    log(f"loaded train={raw.shape}, anchor_cols={len(anchor_cols)}  "
        f"(cat_seeds={len(CAT_SEEDS)}, lgb_seeds={len(LGB_SEEDS)})", t0)

    full = add_trend(raw)

    # ================================================================
    # PHASE 1: sanity + best_iteration 확보 (fit<2024, val==2024) -- 1회만
    # ================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 1: 신규 하이퍼파라미터 + family-weighted 50:50 앙상블 1회 검증")
    print(f"(fit<{SANITY_VAL_SEASON}, val=={SANITY_VAL_SEASON})")
    print("=" * 70)
    fit_p1 = full[full.season < SANITY_VAL_SEASON].copy()
    val_p1 = full[full.season == SANITY_VAL_SEASON].copy()
    med_p1 = fit_p1[num_cols].median(numeric_only=True)
    cat_maps_p1 = build_cat_maps(fit_p1, BASELINE_CATS)

    x_fit_cb_p1 = matrix_catboost(fit_p1, anchor_cols, num_cols, med_p1)
    x_val_cb_p1 = matrix_catboost(val_p1, anchor_cols, num_cols, med_p1)
    x_fit_lgb_p1 = matrix_lightgbm(fit_p1, anchor_cols, num_cols, med_p1, cat_maps_p1)
    x_val_lgb_p1 = matrix_lightgbm(val_p1, anchor_cols, num_cols, med_p1, cat_maps_p1)

    y_fit_p1 = fit_p1[TARGET]
    y_val_p1 = val_p1[TARGET]

    cb_preds_p1, lgb_preds_p1 = [], []
    best_iters = {}
    bss_list_cb, bss_list_lgb = [], []
    skipped = 0
    for i, seed in enumerate(CAT_SEEDS, 1):
        key = f"cat_{seed}"
        if key in done:
            best_iters[key] = done[key]["best_iter"]
            skipped += 1
            print(f"  [CatBoost {i}/{len(CAT_SEEDS)} seed={seed}] (체크포인트에서 이어받음) "
                  f"best_iter={done[key]['best_iter']} BSS={done[key]['bss']:8.2f}")
            continue
        model, best_iter = train_cb_member(seed, x_fit_cb_p1, y_fit_p1, x_val_cb_p1, y_val_p1)
        p = model.predict_proba(x_val_cb_p1)[:, 1]
        brier, bss = score(y_val_p1, p)
        print(f"  [CatBoost {i}/{len(CAT_SEEDS)} seed={seed}] best_iter={best_iter} BSS={bss:8.2f}")
        cb_preds_p1.append(p)
        best_iters[key] = best_iter
        bss_list_cb.append(bss)
        append_progress(key, seed, best_iter, bss)

    for i, seed in enumerate(LGB_SEEDS, 1):
        key = f"lgb_{seed}"
        if key in done:
            best_iters[key] = done[key]["best_iter"]
            skipped += 1
            print(f"  [LightGBM {i}/{len(LGB_SEEDS)} seed={seed}] (체크포인트에서 이어받음) "
                  f"best_iter={done[key]['best_iter']} BSS={done[key]['bss']:8.2f}")
            continue
        model, best_iter = train_lgb_member(seed, x_fit_lgb_p1, y_fit_p1, x_val_lgb_p1, y_val_p1)
        p = model.predict(x_val_lgb_p1)
        brier, bss = score(y_val_p1, p)
        print(f"  [LightGBM {i}/{len(LGB_SEEDS)} seed={seed}] best_iter={best_iter} BSS={bss:8.2f}")
        lgb_preds_p1.append(p)
        best_iters[key] = best_iter
        bss_list_lgb.append(bss)
        append_progress(key, seed, best_iter, bss)

    if skipped:
        print(f"\n  주의: {skipped}개 멤버는 체크포인트에서 이어받아 이번 실행에서 재예측하지 않음 "
              f"-- 아래 앙상블 검증 통계는 이번에 실제로 돌린 멤버만 반영한 '부분' 수치일 수 있음 "
              f"(Phase 2 production 저장에는 영향 없음, best_iteration은 정상 재사용됨).")

    cb_mean_p1 = np.mean(np.vstack(cb_preds_p1), axis=0)
    lgb_mean_p1 = np.mean(np.vstack(lgb_preds_p1), axis=0)
    brier_cb, bss_cb = score(y_val_p1, cb_mean_p1)
    brier_lgb, bss_lgb = score(y_val_p1, lgb_mean_p1)
    ens_p1 = 0.5 * cb_mean_p1 + 0.5 * lgb_mean_p1
    brier_ens, bss_ens = score(y_val_p1, ens_p1)
    delta = bss_ens - KNOWN_ANCHOR_BSS_2024

    print(f"\n  CatBoost {len(CAT_SEEDS)}개 BSS 범위: {min(bss_list_cb):.2f} ~ {max(bss_list_cb):.2f} "
          f"(평균 {np.mean(bss_list_cb):.2f}) | CatBoost-mean ensemble BSS={bss_cb:.2f}")
    print(f"  LightGBM {len(LGB_SEEDS)}개 BSS 범위: {min(bss_list_lgb):.2f} ~ {max(bss_list_lgb):.2f} "
          f"(평균 {np.mean(bss_list_lgb):.2f}) | LightGBM-mean ensemble BSS={bss_lgb:.2f}")
    print(f"\n  [FINAL 50:50 family-weighted ensemble] BSS={bss_ens:8.2f}  "
          f"delta vs anchor(650.54)={delta:+.2f}  {pred_stats(ens_p1)}")
    print("  판정: 개선/중립/악화 중 기록. 결과와 무관하게 production 진행(보류 원칙).")
    print(f"  best_iterations 확보 완료 (총 {len(best_iters)}개)")

    # ================================================================
    # PHASE 2: production (fit = season<2025 = train.csv 전체)
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: production 학습 (fit = season<2025, 즉 train.csv 전체)")
    print("=" * 70)
    fit_full = full.copy()
    med_full = fit_full[num_cols].median(numeric_only=True)
    cat_maps_full = build_cat_maps(fit_full, BASELINE_CATS)

    x_fit_cb_full = matrix_catboost(fit_full, anchor_cols, num_cols, med_full)
    x_fit_lgb_full = matrix_lightgbm(fit_full, anchor_cols, num_cols, med_full, cat_maps_full)
    y_fit_full = fit_full[TARGET]

    manifest = {"catboost": [], "lightgbm": [], "combine": {"catboost_weight": 0.5, "lightgbm_weight": 0.5}}
    for i, seed in enumerate(CAT_SEEDS, 1):
        fixed_iter = best_iters[f"cat_{seed}"]
        fname = f"cat_seed{seed}.cbm"
        if (MODEL_DIR / fname).exists():
            log(f"skip {fname} ({i}/{len(CAT_SEEDS)}, 이미 저장돼 있음 -- 체크포인트에서 이어감)", t0)
        else:
            model, _ = train_cb_member(seed, x_fit_cb_full, y_fit_full, fixed_iter=fixed_iter)
            model.save_model(str(MODEL_DIR / fname))
            log(f"saved {fname} ({i}/{len(CAT_SEEDS)}, iterations={fixed_iter})", t0)
        manifest["catboost"].append({"file": fname, "seed": seed, "iterations": fixed_iter})

    for i, seed in enumerate(LGB_SEEDS, 1):
        fixed_iter = best_iters[f"lgb_{seed}"]
        fname = f"lgb_seed{seed}.txt"
        if (MODEL_DIR / fname).exists():
            log(f"skip {fname} ({i}/{len(LGB_SEEDS)}, 이미 저장돼 있음 -- 체크포인트에서 이어감)", t0)
        else:
            model, _ = train_lgb_member(seed, x_fit_lgb_full, y_fit_full, fixed_iter=fixed_iter)
            model.save_model(str(MODEL_DIR / fname))
            log(f"saved {fname} ({i}/{len(LGB_SEEDS)}, iterations={fixed_iter})", t0)
        manifest["lightgbm"].append({"file": fname, "seed": seed, "iterations": fixed_iter})

    with open(MODEL_DIR / "medians.json", "w") as f:
        json.dump({k: float(v) for k, v in med_full.items()}, f, indent=2)
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump({
            "feature_cols": anchor_cols, "cat_features": BASELINE_CATS,
            "num_cols": num_cols, "cat_maps": cat_maps_full, "manifest": manifest,
        }, f, indent=2)

    with open(OUT_DIR / "requirements.txt", "w") as f:
        f.write("catboost==1.2.8\nlightgbm==4.5.0\n")

    print("\n" + "=" * 70)
    print("PHASE 2 완료 -- 저장된 아티팩트")
    print("=" * 70)
    print(f"  model/ 안에 CatBoost {len(CAT_SEEDS)}개 + LightGBM {len(LGB_SEEDS)}개 + medians.json + feature_cols.json")
    print("  requirements.txt")

    print(f"\n총 소요시간 {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}분)")
    print("다음 단계: 이 폴더(cowork/hw/)를 그대로 커밋/푸시해서 정희원에게 공유해주세요.")


if __name__ == "__main__":
    main()
