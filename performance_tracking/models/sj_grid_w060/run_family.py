# -*- coding: utf-8 -*-
"""sj_final [축7] 모델 계열 — LightGBM · XGBoost 를 **네 번째 멤버**로.

계획(§29.2)에 없던 축이다. way 분해가 닫히면서 "곁가지가 본선" 이 됐으므로 넓힌다.

**왜 이 축인가 — §31 이 보상하는 방향 그대로다.**

```text
피처 이득     val +5.3%  ->  실제 +3.26%   전이율 0.62
앙상블 이득   val +9.19% ->  실제 +9.05%   전이율 0.99
```

같은 val 이득이면 **상관에서 온 쪽이 1.6배 남는다.** 다른 모델 계열은 상관을 낮추는
가장 직접적인 수단이고, 팀 코드에 **실측 선례가 있다** —
`new_val/package/src/merge_families.py` 의 Cat+LGB 계열 블렌드가
`w_lgb = 0.20` 에서 **850.3 → 860.9 (+10.6)** 였다.

**기각 목록에 없다.** §29.1 이 버린 것은 자유 학습형 **결합기**(logistic/GBDT meta)와
`node`(경사하강) 멤버지, 기저 학습기의 계열 추가가 아니다. §29.3 의 동결도
CB/MLP 의 **격자**를 얼린 것이지 계열을 늘리지 말라는 게 아니다.

★ **CPU 로 돈다.** 그래서 GPU 잡과 병행할 수 있다 (24코어 중 일부만 쓴다).
`preprocess_lab/README` 의 "GPU 작업은 한 번에 하나" 를 어기지 않는다.

판정: P-6. 단독이 낮아도 블렌드가 오르면 채택이다. 반대도 성립한다.

    python run_family.py --name FAM_r1 --arms lgb31,lgb63,xgb6 --seeds 3 --threads 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ROOT = FINAL.parents[2]
WORK = FINAL / "work"
PREDS = FINAL / "preds"
RESULTS = FINAL / "results"
PT_VAL = ROOT / "performance_tracking" / "val"

sys.path.insert(0, str(ROOT / "performance_tracking" / "tools"))
sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from run_arm import blend_w, calib, load_base, save_json            # noqa: E402

FROZEN = "S1_base"
EPS = 1e-6

# CB 동결값(depth5 · l2 10000 · lr 0.02 x 3000)과 규제 강도를 맞춘 자리에서 출발한다.
# 시드가 의미를 가지려면 부분표집이 있어야 하므로 feature/bagging 0.8 을 건다.
LGB_BASE = dict(objective="regression", metric="l2", learning_rate=0.02,
                n_estimators=3000, num_leaves=31, min_child_samples=200,
                lambda_l2=1000.0, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, verbose=-1)
XGB_BASE = dict(objective="reg:squarederror", learning_rate=0.02, n_estimators=3000,
                max_depth=6, min_child_weight=200, reg_lambda=1000.0,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist", verbosity=0)

ARMS = {
    # ── 라운드 1: 용량·규제만 바꾼다 (같은 함수 형태) ──
    "lgb31":  dict(lib="lgb", num_leaves=31),
    "lgb63":  dict(lib="lgb", num_leaves=63),
    "lgb127": dict(lib="lgb", num_leaves=127),
    "lgb31_l0": dict(lib="lgb", num_leaves=31, lambda_l2=10.0),     # 규제 약하게
    "xgb6":   dict(lib="xgb", max_depth=6),
    "xgb8":   dict(lib="xgb", max_depth=8),
    # ── 라운드 2: **함수 형태 자체를 바꾼다** ──
    # 라운드 1 이 전부 "축에 수직인 계단" 이라는 같은 가설을 공유한다. ρ 를 정말
    # 낮추려면 가설을 바꿔야 한다. 아래 넷은 각각 다른 방식으로 그렇게 한다.
    "lgb_extra":  dict(lib="lgb", num_leaves=31, extra_trees=True),   # 분할점을 무작위로
    "lgb_linear": dict(lib="lgb", num_leaves=31, linear_tree=True),   # 잎이 선형모형
    "lgb_dart":   dict(lib="lgb", num_leaves=31, boosting_type="dart",
                       drop_rate=0.1),                                # 부스팅 절차가 다르다
    # ★ `objective="huber"` 는 이 데이터에서 **제곱오차와 비트단위로 같다.**
    # LightGBM huber 의 기본 alpha=0.9 는 |잔차| < 0.9 에서 정확히 제곱오차이고,
    # 0/1 타깃에 예측이 0.49 근처라 잔차가 0.49~0.51 로 한 번도 0.9 를 넘지 않는다.
    # 실측 — lgb31 과 예측 최대절대차 0.000e+00. 무효 arm 이라 남겨만 둔다.
    "lgb_huber":  dict(lib="lgb", num_leaves=31, objective="huber"),
    # 진짜로 손실을 바꾸는 것들
    "lgb_hub03":  dict(lib="lgb", num_leaves=31, objective="huber", alpha=0.3),
    "lgb_mae":    dict(lib="lgb", num_leaves=31, objective="regression_l1"),
    "lgb_binary": dict(lib="lgb", num_leaves=31, objective="binary"),  # 링크까지 다르다
    "xgb_dart":   dict(lib="xgb", max_depth=6, booster="dart", rate_drop=0.1),
    # ── 라운드 3: **CatBoost 자신의 다양성 멤버** ──
    # §29.3 의 동결은 "가장 좋은 CB" 를 찾는 격자를 얼린 것이다. 여기서 만드는 것은
    # 좋은 CB 가 아니라 **덜 닮은 CB** 다. lgb63 이 lgb31 보다 단독은 나쁜데
    # (706.1 < 779.0) 블렌드 기여는 컸던(+5.1 > +1.8) 그 자리를 CB 안에서 찾는다.
    "cb_deep":    dict(lib="cb", depth=8, l2_leaf_reg=100.0),
    "cb_shallow": dict(lib="cb", depth=3, l2_leaf_reg=10000.0),
    "cb_logloss": dict(lib="cb", loss_function="Logloss"),
    "cb_fast":    dict(lib="cb", depth=6, learning_rate=0.08, iterations=800),
    # ── [축8] F 전문가 — **헤드룸이 F 에 있다는 실측에서 나온 축** ──
    # fit(2022·R) → eval(2024) 로 재면 멤버 간 차이가 R 에서는 폭 1.9(901.7~903.6)인데
    # F 에서는 폭 68(368.7~436.9)이다. R 은 포화했고 전장은 F(11.8%)다.
    # 그런데 지금 멤버 중 **F 를 겨냥해 만든 것이 하나도 없다.**
    #
    # `game_type` 단절(§28.1)이 이 축의 설계를 정한다 — F 성공률이 2022 0.7087 →
    # 2023 0.4729 로 무너지므로, F 를 배우려면 **단절 이후 시즌만** 써야 한다.
    # 그래서 `f_post`(2023 F 만)와 `post`(2023 전체)를 같이 둔다.
    "cb_f_only":  dict(lib="cb", rows="F"),         # F 행만 (전 시즌)
    "cb_f_last1": dict(lib="cb", rows="F_last1"),   # 직전 시즌 F 만
    "cb_last1":   dict(lib="cb", rows="last1"),     # 직전 시즌 전체 (레짐 전문가)
    "cb_last2":   dict(lib="cb", rows="last2"),
    "cb_f_up5":   dict(lib="cb", wF=5.0),           # 전체 학습, F 가중 5배
    "cb_f_up20":  dict(lib="cb", wF=20.0),
    # ── [축8] 라운드2 — `cb_last1`(Δ+5.7) 이 왜 이겼는지 가른다 ──
    # `cb_last2`(2시즌)는 Δ−1.1 로 무너졌다. 데이터가 늘수록 전체 cb 로 수렴한다
    # (ρ 0.9409 → 0.9726). 그러면 이득의 출처는 "한 시즌" 인가 "최근" 인가?
    "cb_prev1":   dict(lib="cb", rows="prev1"),     # 직전의 직전 시즌 — 최근성만 뺀다
    "cb_prev2":   dict(lib="cb", rows="prev2"),
    "cb_oldest":  dict(lib="cb", rows="oldest"),    # 가장 먼 시즌 — 최근성의 반대 극
    # 승자 주변 — 용량을 바꿔 최적점이 어디인지
    "cb_last1_d3": dict(lib="cb", rows="last1", depth=3),
    "cb_last1_d8": dict(lib="cb", rows="last1", depth=8),
    "cb_last1_f5": dict(lib="cb", rows="last1", wF=5.0),   # 최근성 + F 강조를 같이
    # ── [축8] 라운드2 개정 — **채택된 방향(`cb_f_only`)을 판다** ──
    # `cb_last1` 은 val2022 −5.4 로 P-2 관문에서 벤치됐다 (메커니즘이 레짐 단절
    # 포착이라 단절이 없는 2022 에서는 오히려 해가 된다). 통과한 것은 `cb_f_only`
    # (val2024 +2.7 / val2022 +2.3) 하나뿐이므로 그 주변을 판다.
    #
    # F 행은 131K 로 전체의 10.7% 다. 동결값(depth 5 · l2 10000)은 1.22M 행에서
    # 고른 것이라 **이 크기에는 규제가 과할 수 있다.** 용량·규제를 양쪽으로 연다.
    "cb_f_d3":    dict(lib="cb", rows="F", depth=3),
    "cb_f_d8":    dict(lib="cb", rows="F", depth=8),
    "cb_f_l2lo":  dict(lib="cb", rows="F", l2_leaf_reg=1000.0),
    "cb_f_l2hi":  dict(lib="cb", rows="F", l2_leaf_reg=30000.0),
    "cb_f_prev":  dict(lib="cb", rows="F_prev"),    # 직전 시즌을 뺀 F — 최근성 대조
    # ── §29.2 조건부 재탐색 — **입력 피처셋이 바뀌면 CB 최적점이 이동한다** ──
    # §29.3 의 CB 동결(depth 5 · l2 10000)은 **X168 입력 기준**이다. §29.2 가
    # "입력 피처셋이 실제로 168 에서 바뀌면 l2 {3000,10000,30000} × depth {4,5,6},
    # 3시드만 좁게 확인한다" 고 조건을 달아뒀고, `id_freq` 8열이 들어가 176 이 되면서
    # 그 조건이 발동했다. 새 8열은 ID 신뢰도라는 **성격이 다른 정보**라
    # (표본 크기 + 미출현 플래그) 최적 깊이·규제가 움직였을 수 있다.
    # `g_d5_l10k` 가 동결값 그대로이므로 대조군이다 (P-5).
    **{f"g_d{d}_l{lab}": dict(lib="cb", depth=d, l2_leaf_reg=float(v))
       for d in (4, 5, 6)
       for lab, v in (("3k", 3000), ("10k", 10000), ("30k", 30000))},
    # ── 모서리 연장 ──────────────────────────────────────────────────────────
    # §29.2 의 격자에서 **최적이 모서리(d6 · l2 3000)에 붙었다.** 모서리 해는
    # 격자 밖에 더 있다는 신호이므로 한 칸 더 본다. 깊이는 추론 시간을 늘리므로
    # (오블리비어스 트리는 깊이만큼 경계 비교를 한다) 이득이 멎는 지점을 확인해야
    # 서버 600초 예산 안에서 고를 수 있다.
    **{f"g_d{d}_l{lab}": dict(lib="cb", depth=d, l2_leaf_reg=float(v))
       for d, lab, v in ((6, "1k", 1000), (7, "3k", 3000), (7, "1k", 1000),
                         (8, "3k", 3000))},
}

# 행 필터 — (season, game_type) 만 본다. 라벨도 다른 행도 보지 않는다.
#
# ★ **폴드 상대적으로 정의한다.** 처음에 `season >= 2023`("단절 이후")으로 썼는데,
# fold 2023·2022 는 학습이 전부 단절 이전이라 **0행**이 된다 — fold 2024 에서만
# 만들 수 있는 멤버는 `fit(다른 폴드) → eval(2024)` 판정이 불가능하다.
# `s.max()` 는 그 폴드의 직전 시즌이므로 어느 폴드에서든 성립한다.
ROW_FILTERS = {
    "F":       lambda s, g: g == "F",
    "F_last1": lambda s, g: (g == "F") & (s == s.max()),
    "last1":   lambda s, g: s == s.max(),
    "last2":   lambda s, g: s >= s.max() - 1,
    # ★ 대조군 — `last1` 의 이득이 **최근성** 때문인지 **작고 다른 부분집합**이기
    # 때문인지 가른다. `prev1` 은 시즌 하나만 쓰는 것은 같고 최근성만 없다.
    # last1 이 이기면 최근성, 비슷하면 그냥 다양성이다.
    "prev1":   lambda s, g: s == s.max() - 1,
    "prev2":   lambda s, g: s == s.max() - 2,
    "oldest":  lambda s, g: s == s.min(),
    "F_prev":  lambda s, g: (g == "F") & (s < s.max()),   # 직전 시즌을 뺀 F
}


def log(m):
    print(m, flush=True)


def fit_lgb(cfg, Xt, yt, Xv, seed, threads):
    import lightgbm as L
    p = {k: v for k, v in {**LGB_BASE, **cfg}.items() if k != "lib"}
    if p.get("objective") == "binary":
        # 로지스틱 링크 + logloss — 회귀기와 링크 자체가 다르다. 진짜 다른 함수족이다.
        p.pop("metric", None)
        m = L.LGBMClassifier(**p, random_state=seed, n_jobs=threads)
        m.fit(Xt, yt.astype(int))
        out = m.predict_proba(Xv)[:, 1]
    else:
        m = L.LGBMRegressor(**p, random_state=seed, n_jobs=threads)
        m.fit(Xt, yt.astype(np.float64))
        out = m.predict(Xv)
    out = np.clip(out, EPS, 1 - EPS)
    del m
    return out


def fit_xgb(cfg, Xt, yt, Xv, seed, threads):
    import xgboost as X
    p = {k: v for k, v in {**XGB_BASE, **cfg}.items() if k != "lib"}
    m = X.XGBRegressor(**p, random_state=seed, n_jobs=threads)
    m.fit(Xt, yt.astype(np.float64))
    out = np.clip(m.predict(Xv), EPS, 1 - EPS)
    del m
    return out


def fit_cb_var(cfg, Xt, yt, Xv, seed, gpu, st=None, gt=None):
    """CB 다양성 멤버. 동결값(CB_P)을 cfg 로 덮어쓴 것만 다르다. **GPU 를 쓴다.**

    `rows` 는 학습행을 (season, game_type) 으로 거른다. `wF` 는 F 행 가중치다.
    둘 다 라벨도 다른 행도 보지 않으므로 행 독립성·시간 인과에 영향이 없다.
    """
    from catboost import CatBoostRegressor, Pool
    from run_arm import CB_P
    skip = {"lib", "rows", "wF"}
    p = {**CB_P, **{k: v for k, v in cfg.items() if k not in skip}}
    w = None
    if cfg.get("rows"):
        m_ = ROW_FILTERS[cfg["rows"]](st, gt)
        if m_.sum() < 1000:
            raise ValueError(f"행 필터 {cfg['rows']} 가 {int(m_.sum())}행만 남긴다")
        Xt, yt, gt = np.ascontiguousarray(Xt[m_]), yt[m_], gt[m_]
    if cfg.get("wF"):                       # 행 필터와 **같이** 걸 수 있다
        w = np.where(gt == "F", float(cfg["wF"]), 1.0)
    dev = dict(task_type="GPU", devices="0", border_count=128) if gpu else {}
    m = CatBoostRegressor(**p, random_seed=seed, **dev)
    m.fit(Pool(Xt, yt.astype(np.float64), weight=w))
    out = np.clip(m.predict(Pool(Xv)), EPS, 1 - EPS)
    del m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=10,
                    help="GPU 잡과 병행할 때는 코어를 다 쓰지 않는다")
    ap.add_argument("--folds", default="2024")
    ap.add_argument("--trees", type=int, default=0, help="0 이면 기본 3000")
    ap.add_argument("--cpu", action="store_true", help="cb 계열 arm 을 CPU 로")
    ap.add_argument("--atoms", default="",
                    help="[축1] 전처리 원자를 붙이고 학습 (예: id_freq). + 로 결합")
    a = ap.parse_args()
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    bad = [s for s in arms if s not in ARMS]
    if bad:
        sys.exit(f"[중단] 모르는 arm: {bad}\n  가능: {sorted(ARMS)}")
    folds = [int(s) for s in a.folds.split(",")]
    if a.trees:
        LGB_BASE["n_estimators"] = a.trees
        XGB_BASE["n_estimators"] = a.trees
    for d in (PREDS, RESULTS, PT_VAL):
        d.mkdir(parents=True, exist_ok=True)

    from common import load_labels, score, render

    t0 = time.time()
    X, y, season, row_id = load_base()
    log("=" * 88)
    log("[축7] 모델 계열 %s | arm %s | 짝시드 %d | 스레드 %d | 폴드 %s"
        % (a.name, ",".join(arms), a.seeds, a.threads, folds))
    log("=" * 88)

    out = {"name": a.name, "axis": "family", "seeds": a.seeds,
           "threads": a.threads, "arms": {k: ARMS[k] for k in arms}, "folds": {}}

    for fold in folds:
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        yv = y[va].astype(np.float64)
        log("\n── val%d  학습 %s행 · 검증 %s행" % (fold, f"{len(tri):,}", f"{len(va):,}"))
        Xall = np.asarray(X)
        if a.atoms:
            # ★ **피처 조건 × 학습행 조건**. 오늘 잰 `단독~ρ` 선(r=0.994)은 전부
            # 원자 없이 잰 것이라, 원자가 base 와 전문가를 다른 비율로 올리면
            # 선이 움직이고 최적 조합이 바뀐다. 그 칸을 여기서 채운다.
            import atoms as A
            _names = json.load(open(WORK / "meta.json", encoding="utf-8"))["names"]
            E, en = A.build(X, _names, season <= fold - 1, fold,
                            [t.strip() for t in a.atoms.split("+") if t.strip()])
            Xall = np.concatenate([Xall, E], axis=1)
            log("     원자 %s → %d열 추가 (총 %d피처)" % (a.atoms, len(en), Xall.shape[1]))
            del E
        Xt = np.ascontiguousarray(Xall[tri]); Xv = np.ascontiguousarray(Xall[va])
        # [축8] 행 필터·F 가중이 쓰는 game_type. build_features 가 남긴 배열이다.
        gt_all = np.load(WORK / "game_type.npy", allow_pickle=True)
        gt_tr = np.asarray(gt_all, dtype=str)[tri]
        del gt_all
        del Xall

        # ★ 동결 멤버가 없어도 죽지 않는다. **arm 자신의 예측이 산출물**이고 동결
        # 3멤버와의 블렌드 비교는 편의 기능이다. 편의 기능이 없다고 잡을 죽이면
        # fold 2023 처럼 기준선이 아직 없는 폴드에서 아무것도 못 만든다
        # (실제로 FAM_f3 이 이 이유로 fold 2023 에서 중단됐다).
        frz = {}
        for k in ("cb", "ft", "mlp"):
            fp = PREDS / f"{FROZEN}__{k}_{fold}.npy"
            if fp.exists():
                v = np.load(fp)
                if len(v) == len(yv):
                    frz[k] = v
        has_frz = len(frz) == 3
        if has_frz:
            w0, s0 = blend_w([frz["cb"], frz["ft"], frz["mlp"]], yv)
            log("     동결 3멤버 블렌드 %8.1f   w = %s"
                % (s0, " ".join("%.3f" % v for v in w0)))
        else:
            s0 = float("nan")
            log("     ⚠ 동결 멤버 없음 — 예측만 저장하고 P-6 비교는 건너뛴다")

        rec = {}
        for arm in arms:
            cfg = ARMS[arm]
            acc = np.zeros(len(va), np.float64); per = []
            for sd in range(a.seeds):
                ts = time.time()
                if cfg["lib"] == "lgb":
                    p = fit_lgb(cfg, Xt, y[tri], Xv, sd, a.threads)
                elif cfg["lib"] == "xgb":
                    p = fit_xgb(cfg, Xt, y[tri], Xv, sd, a.threads)
                else:                       # cb — GPU 를 쓴다 (다른 GPU 잡과 겹치지 말 것)
                    p = fit_cb_var(cfg, Xt, y[tri], Xv, sd, not a.cpu,
                                   st=season[tri], gt=gt_tr)
                acc += p
                per.append(float(calib(p, yv)[0]))
                log("     %-10s seed%d  단독 %8.1f  (%.0f초)"
                    % (arm, sd, per[-1], time.time() - ts))
            s, k, q = calib(acc / a.seeds, yv)
            np.save(PREDS / f"{a.name}__{arm}_{fold}.npy", q)

            if has_frz:
                rho = {m: float(np.corrcoef(q, frz[m])[0, 1]) for m in ("cb", "ft", "mlp")}
                w, s_bl = blend_w([frz["cb"], frz["ft"], frz["mlp"], q], yv)
                log("     %-10s ▸ 단독 %8.1f | ρ cb %.4f ft %.4f mlp %.4f | 4멤버 블렌드 %8.1f (Δ %+.1f)  w=%s"
                    % (arm, s, rho["cb"], rho["ft"], rho["mlp"], s_bl, s_bl - s0,
                       " ".join("%.3f" % v for v in w)))
            else:
                rho, w, s_bl = {}, [], float("nan")
                log("     %-10s ▸ 단독 %8.1f  (동결 멤버 없음 — 예측만 저장)" % (arm, s))
            rec[arm] = {"cfg": cfg, "solo": float(s), "scale": float(k), "per_seed": per,
                        "rho": rho, "blend4": float(s_bl), "blend3": float(s0),
                        "w": [float(v) for v in w]}
            # arm 하나가 끝날 때마다 남긴다 — 뒤 arm 이 멈춰도 여기까지는 보존된다
            out["folds"][str(fold)] = {"blend3": None if not has_frz else float(s0),
                                       "arms": rec}
            save_json(a.name, out)
            if not has_frz:
                continue                       # 채점이 블렌드 예측 기준이라 건너뛴다

            r = float(yv.mean())
            pb = np.clip(r + np.column_stack(
                [frz["cb"] - r, frz["ft"] - r, frz["mlp"] - r, q - r]) @ w, 1e-6, 1 - 1e-6)
            pd.DataFrame({"row_id": row_id[va], "pred": pb}).to_csv(
                PT_VAL / f"{a.name}_{arm}_{fold}.csv", index=False)
            lab = load_labels(fold)
            m = score(pd.DataFrame({"row_id": row_id[va], "pred": pb})
                      .set_index("row_id").loc[lab["row_id"], "pred"].to_numpy(np.float64),
                      fold, lab)
            rec[arm]["score"] = {k2: (float(v) if isinstance(v, (int, float)) else v)
                                 for k2, v in m.items()}

        if not has_frz:
            log("     val%d 완료 — 예측 저장됨. 동결 멤버가 생기면"
                " analyze_members.py 로 판정한다." % fold)
            out["folds"][str(fold)] = {"blend3": None, "arms": rec}
            del Xt, Xv
            continue
        log("\n── val%d 요약 — 판정은 4멤버 블렌드다 (P-6). 단독은 낮아도 된다." % fold)
        log("  %-10s %9s %9s %9s %9s" % ("arm", "단독", "ρ(cb)", "블렌드4", "Δ vs 3멤버"))
        for arm in arms:
            r = rec[arm]
            d = r["blend4"] - s0
            log("  %-10s %9.1f %9.4f %9.1f %+9.1f  %s"
                % (arm, r["solo"], r["rho"]["cb"], r["blend4"], d,
                   "★승격" if d >= 7 else ""))
        out["folds"][str(fold)] = {"blend3": float(s0), "arms": rec}
        del Xt, Xv

    out["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(out, open(RESULTS / f"{a.name}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("\n총 %.1f분" % out["minutes"])
    return 0


if __name__ == "__main__":
    import traceback
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); raise
