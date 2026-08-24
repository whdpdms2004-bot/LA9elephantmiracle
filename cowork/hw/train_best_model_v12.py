"""submission_v12 -- v11(실LB 895.9753214633 확인) 위에 좌우(hand) 인코딩 변경.

바뀌는 것 (validate_hand_encoding.py, 결정 fold 2024 단일시드):
    baseline(v11)   772.20
    A hand 범주형만  745.44  (-26.75, 단독으론 오히려 나쁨)
    B 매치업만       789.17  (+16.97)
    C 둘 다         800.32  (+28.12)  <- 채택
pitcher_hand/batter_hand를 범주형으로 등록 + handedness_matchup(좌우 매치업
4레벨) 추가. A 단독은 나쁜데 B와 같이 넣으면 최고 -- CatBoost가 hand를
범주형으로 잡아야 매치업 조합과 엮어서 쓸 수 있게 되는 것으로 보임.
("단독으로 나쁜 피처도 조합에선 산다" -- sj님 preprocess_lab 원리가 내 라인에서
처음 재현된 사례.)

왜 이 유형인가 -- 이번 세션 전이배수 관측:
    플래툰(피처 추가)      로컬 +82    실LB +0.7   0.01x
    count_state(피처 추가) 로컬 +14.43 실LB +3.85  0.27x
    team_id 범주형(인코딩) 로컬 +33.92 실LB +34.72 1.02x  <- 인코딩은 그대로 전이
이번 변경은 "있는 정보의 인코딩"이라 세 번째 유형에 해당한다.

같이 시험했다가 기각한 것:
    pitcher_id/batter_id 범주형 (-53.88 / -19.05 / -53.72)
    -- 카디널리티 792명/830명으로 team_id(10~13)와 규모가 달라 과적합.
       yn님의 기각 결과가 그대로 재현됨. team_id가 통한 건 저카디널리티라서.

Feature set: baseline47 + trend6 + platoon2 + count_state + handedness_matchup = 57개
BASELINE_CATS: top_bottom, game_type, base_state, pitcher_team_id, batter_team_id,
               count_state, pitcher_hand, batter_hand, handedness_matchup
Model: CatBoost 16-seed, v10/v11과 동일 하이퍼파라미터. 체크포인트 지원.
(시드 32개는 검토 후 기각 -- validate_platt_ensemble_size.py에서 시드-BSS 곡선을
 직접 재보니 12개에서 이미 포화, 13~16 구간이 +-0.5 노이즈였음.)

PHASE 1 (fit<2024, val==2024, lookup은 fit에서만 생성): 정직한 검증 + best_iteration
          + 오프셋 재계산용 per-bucket pred_mean 산출 (이 모델로, 정직하게).
PHASE 2 (fit=season<2025 전체): production 재학습 + lookup 전체 데이터로 재생성 + model/ 저장.

실행:
    py train_best_model_v12.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parents[2]  # cowork/hw -> cowork -> 저장소 루트
DATA_DIR = BASE_DIR / "data"
OUT_DIR = Path(__file__).resolve().parent / "submission_v12"
MODEL_DIR = OUT_DIR / "model"
PROGRESS_FILE = OUT_DIR / "phase1_progress.jsonl"

ID = "row_id"
TARGET = "control_success"
BASELINE_CATS = ["top_bottom", "game_type", "base_state",
                 "pitcher_team_id", "batter_team_id", "count_state",
                 "pitcher_hand", "batter_hand", "handedness_matchup"]
PREV_PAIRS = (1, 3, 5)
K_PLATOON = 300.0
SANITY_VAL_SEASON = 2024
KNOWN_V11_BSS_2024 = 772.20  # 참고용 -- validate_hand_encoding.py 단일시드 기준선(v11 구성)

# 오프셋 재계산용: 기존 season logit offset의 target(2019~2024 추세외삽, 리더보드 미참조,
# submission_v9와 동일 값 재사용 -- 이건 데이터 속성이지 모델 속성이 아니므로 바뀔 이유 없음)
TARGET_RATE = 0.4792
BUCKET_EDGES = [-1, 200, 2000, float("inf")]  # v9_bucketoffset과 동일 경계(이미 실LB +8.84 검증됨)

CAT_SEEDS = list(range(2026, 2026 + 16))  # v9와 동일 개수
MAX_ITER = 1500
EARLY_STOP = 100

CB_PARAMS = dict(  # v9와 완전히 동일 (하이퍼파라미터는 이번 실험 변수가 아님)
    loss_function="Logloss", eval_metric="BrierScore", depth=6,
    learning_rate=0.03, l2_leaf_reg=25, random_strength=0.6, border_count=128,
    thread_count=-1, grow_policy="Depthwise", boosting_type="Plain",
    bootstrap_type="Bernoulli", subsample=0.7, rsm=0.7,
    verbose=False, od_type="Iter", od_wait=EARLY_STOP, allow_writing_files=False,
)


def log(msg, t0):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def score(y, p):
    y = np.asarray(y)
    brier = float(np.mean((y - p) ** 2))
    r = y.mean()
    base = r * (1 - r)
    return brier, (max(0.0, 100000 * (1 - brier / base)) if base > 0 else 0.0)


def add_trend(df):
    x = df.copy()
    for k in PREV_PAIRS:
        recent = f"asof_pitcher_prev{k}_game_success_rate"
        x[f"trend_prev{k}"] = x[recent] - x["asof_pitcher_success_rate"]
        x[f"trend_abs_prev{k}"] = x[f"trend_prev{k}"].abs()
    return x


def add_count_state(df):
    """볼카운트 상태를 "balls-strikes" 범주형으로. 원본 컬럼(balls_before,
    strikes_before)의 재조합일 뿐 새 정보 없음 -- 투구 이전 정보만 사용."""
    x = df.copy()
    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    return x


def add_matchup(df):
    """좌우 매치업 4레벨. pitcher_hand/batter_hand 재조합일 뿐 새 정보 없음.
    platoon_split이 '이 투수가 좌타/우타 상대로 어떤가'(투수별 EB)를 잡는다면
    이건 '좌투-좌타' 같은 매치업 조합 자체의 리그 전반 효과를 잡는다."""
    x = df.copy()
    x["handedness_matchup"] = (x["pitcher_hand"].astype(str) + "_"
                               + x["batter_hand"].astype(str))
    return x


def add_platoon_prior_cumulative(df, league_avg, K=K_PLATOON):
    x = df.copy()
    cum_success_ph = x.groupby(["pitcher_id", "batter_hand"])[TARGET].cumsum()
    cum_n_ph = x.groupby(["pitcher_id", "batter_hand"]).cumcount() + 1
    prior_success_ph = cum_success_ph - x[TARGET]
    prior_n_ph = cum_n_ph - 1
    eb_ph = (prior_success_ph + K * league_avg) / (prior_n_ph + K)

    cum_success_p = x.groupby("pitcher_id")[TARGET].cumsum()
    cum_n_p = x.groupby("pitcher_id").cumcount() + 1
    prior_success_p = cum_success_p - x[TARGET]
    prior_n_p = cum_n_p - 1
    eb_p = (prior_success_p + K * league_avg) / (prior_n_p + K)

    x["platoon_split"] = eb_ph - eb_p
    x["platoon_n"] = np.log1p(prior_n_ph)
    return x


def build_platoon_lookup(source_df, league_avg, K=K_PLATOON):
    ph = (source_df.groupby(["pitcher_id", "batter_hand"])[TARGET]
          .agg(n="count", s="sum").reset_index())
    p = (source_df.groupby("pitcher_id")[TARGET]
         .agg(p_n="count", p_s="sum").reset_index())
    ph = ph.merge(p, on="pitcher_id", how="left")
    ph["eb_ph"] = (ph["s"] + K * league_avg) / (ph["n"] + K)
    ph["eb_p"] = (ph["p_s"] + K * league_avg) / (ph["p_n"] + K)
    ph["platoon_split"] = ph["eb_ph"] - ph["eb_p"]
    ph["platoon_n"] = np.log1p(ph["n"])
    return ph[["pitcher_id", "batter_hand", "platoon_split", "platoon_n"]]


def apply_platoon_lookup(df, lookup):
    x = df.merge(lookup, on=["pitcher_id", "batter_hand"], how="left")
    x["platoon_split"] = x["platoon_split"].fillna(0.0)
    x["platoon_n"] = x["platoon_n"].fillna(0.0)
    return x


def matrix(df, cols, num_cols, med):
    x = df[cols].copy()
    x[num_cols] = x[num_cols].fillna(med)
    for c in BASELINE_CATS:
        if c in x.columns:
            x[c] = x[c].fillna("__NA__").astype(str)
    return x


def load_progress():
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
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    done = load_progress()
    if done:
        print(f"[체크포인트 발견] 이미 완료된 Phase1 멤버 {len(done)}개 -- 이어서 실행합니다.", flush=True)

    raw = pd.read_csv(DATA_DIR / "train.csv")
    test_cols = pd.read_csv(DATA_DIR / "test.csv", nrows=0).columns.tolist()
    baseline_47 = [c for c in test_cols if c != ID]
    trend_cols = [f"trend_prev{k}" for k in PREV_PAIRS] + [f"trend_abs_prev{k}" for k in PREV_PAIRS]
    platoon_cols = ["platoon_split", "platoon_n"]
    feature_cols = baseline_47 + trend_cols + platoon_cols + ["count_state", "handedness_matchup"]
    num_cols = [c for c in feature_cols if c not in BASELINE_CATS]
    league_avg = raw[TARGET].mean()
    log(f"loaded train={raw.shape}, league_avg={league_avg:.6f}, feature_cols={len(feature_cols)}, "
        f"cat_features={BASELINE_CATS}", t0)

    full = add_trend(raw)
    full = add_count_state(full)
    full = add_matchup(full)

    # ================================================================
    # PHASE 1: 정직한 검증 (fit<2024, val==2024, lookup은 fit에서만 생성) -- 1회
    # ================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 1: anchor+platoon+team_id범주형, CatBoost {len(CAT_SEEDS)}시드 검증 "
          f"(fit<{SANITY_VAL_SEASON}, val=={SANITY_VAL_SEASON})")
    print("=" * 70, flush=True)
    fit_p1_raw = full[full.season < SANITY_VAL_SEASON].copy()
    val_p1_raw = full[full.season == SANITY_VAL_SEASON].copy()

    fit_p1 = add_platoon_prior_cumulative(fit_p1_raw, league_avg)
    lookup_p1 = build_platoon_lookup(fit_p1_raw, league_avg)
    val_p1 = apply_platoon_lookup(val_p1_raw, lookup_p1)

    med_p1 = fit_p1[num_cols].median(numeric_only=True)
    x_fit_p1 = matrix(fit_p1, feature_cols, num_cols, med_p1)
    x_val_p1 = matrix(val_p1, feature_cols, num_cols, med_p1)
    y_fit_p1, y_val_p1 = fit_p1[TARGET], val_p1[TARGET]

    preds_p1 = []
    best_iters = {}
    bss_list = []
    skipped = 0
    for i, seed in enumerate(CAT_SEEDS, 1):
        key = f"cat_{seed}"
        if key in done:
            best_iters[key] = done[key]["best_iter"]
            skipped += 1
            print(f"  [{i}/{len(CAT_SEEDS)} seed={seed}] (체크포인트에서 이어받음) "
                  f"best_iter={done[key]['best_iter']} BSS={done[key]['bss']:8.2f}", flush=True)
            continue
        params = dict(CB_PARAMS)
        params["random_seed"] = seed
        params["iterations"] = MAX_ITER
        model = CatBoostClassifier(**params)
        model.fit(x_fit_p1, y_fit_p1, cat_features=BASELINE_CATS,
                  eval_set=(x_val_p1, y_val_p1), use_best_model=True)
        p = model.predict_proba(x_val_p1)[:, 1]
        brier, bss = score(y_val_p1, p)
        best_iter = model.get_best_iteration()
        print(f"  [{i}/{len(CAT_SEEDS)} seed={seed}] BSS={bss:8.2f} best_iter={best_iter} "
              f"(v11 hand변경없음 단일시드 기준선 {KNOWN_V11_BSS_2024})", flush=True)
        preds_p1.append(p)
        best_iters[key] = best_iter
        bss_list.append(bss)
        append_progress(key, seed, best_iter, bss)

    print(f"\n  개별 시드 평균 BSS={np.mean(bss_list) if bss_list else float('nan'):.2f} "
          f"(이번 실행에서 새로 돈 {len(bss_list)}개 기준, 스킵 {skipped}개)")

    bucket_offsets = None
    if len(preds_p1) == len(CAT_SEEDS) or skipped == len(CAT_SEEDS):
        # 전체 앙상블 예측을 다시 만들어야 하므로, 스킵된 것들도 포함해 재예측
        # (체크포인트는 best_iter만 저장하고 예측 배열은 저장 안 함 -- 재실행 시 이 블록에서 재계산)
        if len(preds_p1) < len(CAT_SEEDS):
            print("  [오프셋 계산을 위해 전체 16개 시드로 val2024 재예측 -- 체크포인트 iterations 재사용]")
            preds_p1 = []
            for seed in CAT_SEEDS:
                key = f"cat_{seed}"
                params = dict(CB_PARAMS)
                params["random_seed"] = seed
                params["iterations"] = best_iters[key]
                m = CatBoostClassifier(**params)
                m.fit(x_fit_p1, y_fit_p1, cat_features=BASELINE_CATS)
                preds_p1.append(m.predict_proba(x_val_p1)[:, 1])

        ens_p1 = np.mean(np.vstack(preds_p1), axis=0)
        _, ens_bss = score(y_val_p1, ens_p1)
        pred_mean = ens_p1.mean()
        actual_mean = y_val_p1.mean()
        print(f"  [{len(CAT_SEEDS)}-seed 앙상블] BSS={ens_bss:8.2f}  "
              f"v11(hand변경 없음, 실LB 895.9753214633 확인) 대비 참고 비교용")
        print(f"  예측평균={pred_mean:.6f}  실제평균={actual_mean:.6f}  편향={100*(pred_mean-actual_mean):+.3f}%p")

        # ---- 오프셋 재계산: 정직한(fit<2024) 16-seed 앙상블 예측으로, 구간별 pred_mean 산출 ----
        print("\n  [오프셋 재계산] target=0.4792(기존과 동일, 데이터 속성) 기준, "
              "구간별 pred_mean은 이 모델(team_id범주형)로 정직하게(fit<2024) 재산출")
        pitcher_n_val = val_p1_raw["asof_pitcher_n"].fillna(0).to_numpy()
        bucket_idx_val = np.digitize(pitcher_n_val, BUCKET_EDGES[1:-1])
        bucket_offsets = []
        for b in range(len(BUCKET_EDGES) - 1):
            mask = bucket_idx_val == b
            bpred_mean = ens_p1[mask].mean()
            off = float(logit(TARGET_RATE) - logit(bpred_mean))
            bucket_offsets.append(off)
            print(f"    구간{b} n={mask.sum():7d} pred_mean={bpred_mean:.4f} -> offset={off:+.6f}")
        with open(OUT_DIR / "bucket_offsets.json", "w") as f:
            json.dump({"bucket_edges": [e if e != float("inf") else None for e in BUCKET_EDGES],
                       "bucket_offsets": bucket_offsets, "target_rate": TARGET_RATE}, f, indent=2)
        print(f"  저장: {OUT_DIR / 'bucket_offsets.json'}")
    else:
        print("  (일부 멤버만 체크포인트에서 이어받아져서 이번 실행만으론 오프셋 재계산을 못 함 -- "
              "다음 재실행에서 전체가 모이면 자동으로 계산됨)")
    print(f"  best_iterations 확보 완료 (총 {len(best_iters)}개)")

    # ================================================================
    # PHASE 2: production (fit = season<2025 전체, lookup도 전체 데이터로 재생성)
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: production 학습 (fit = season<2025, lookup은 train.csv 전체로 재생성)")
    print("=" * 70, flush=True)
    fit_full_raw = full.copy()
    fit_full = add_platoon_prior_cumulative(fit_full_raw, league_avg)
    lookup_full = build_platoon_lookup(fit_full_raw, league_avg)

    med_full = fit_full[num_cols].median(numeric_only=True)
    x_fit_full = matrix(fit_full, feature_cols, num_cols, med_full)
    y_fit_full = fit_full[TARGET]

    manifest = {"catboost": []}
    for i, seed in enumerate(CAT_SEEDS, 1):
        fixed_iter = best_iters[f"cat_{seed}"]
        fname = f"cat_seed{seed}.cbm"
        if (MODEL_DIR / fname).exists():
            log(f"skip {fname} ({i}/{len(CAT_SEEDS)}, 이미 저장돼 있음)", t0)
        else:
            params = dict(CB_PARAMS)
            params["random_seed"] = seed
            params["iterations"] = fixed_iter
            model = CatBoostClassifier(**params)
            model.fit(x_fit_full, y_fit_full, cat_features=BASELINE_CATS)
            model.save_model(str(MODEL_DIR / fname))
            log(f"saved {fname} ({i}/{len(CAT_SEEDS)}, iterations={fixed_iter})", t0)
        manifest["catboost"].append({"file": fname, "seed": seed, "iterations": fixed_iter})

    lookup_full.to_csv(MODEL_DIR / "platoon_lookup.csv", index=False)
    with open(MODEL_DIR / "medians.json", "w") as f:
        json.dump({k: float(v) for k, v in med_full.items()}, f, indent=2)
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump({
            "feature_cols": feature_cols, "cat_features": BASELINE_CATS,
            "num_cols": num_cols, "manifest": manifest,
            "league_avg": float(league_avg), "K_platoon": K_PLATOON,
        }, f, indent=2)
    with open(OUT_DIR / "requirements.txt", "w") as f:
        f.write("catboost==1.2.8\n")

    print("\n" + "=" * 70)
    print("PHASE 2 완료")
    print("=" * 70)
    print(f"  model/ 안에 CatBoost {len(CAT_SEEDS)}개 + platoon_lookup.csv + medians.json + feature_cols.json")
    if bucket_offsets:
        print(f"  bucket_offsets.json 에 오프셋 저장됨 -- script.py 작성 시 반영할 것: {bucket_offsets}")
    print(f"\n총 소요시간 {(time.time()-t0)/60:.1f}분")
    print("다음 단계: submission_v12/script.py 작성 -> 체크리스트 -> zip")


if __name__ == "__main__":
    main()
