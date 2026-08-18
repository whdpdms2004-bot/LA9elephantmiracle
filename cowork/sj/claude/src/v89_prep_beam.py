"""V89: 전처리 원자 변환의 조합을 빔 서치로 훑는다.

배경
    v85_preprocess_screen 은 원자 변환 15개를 "하나씩" 또는 "전부" 만 돌린다.
    그 사이가 비어 있다. 피처군마다 다른 전처리를 골라 쓰는 조합이 그 공간에 있다.

방법
    빔 서치. 라운드마다 현재 빔의 각 조합에 남은 변환을 하나씩 더해 보고
    상위 BEAM 개만 남긴다. 개선이 멈추면 종료한다.
    v85 가 이미 만들어 둔 단일 arm 결과를 1라운드로 재사용한다 (재학습 없음).

점수
    offset/logit 보정은 적용하지 않는다 (2026-08-18 결정).
    다만 순위는 bss_centered 로 매긴다 - 예측 평균을 실제 평균에 맞춘 뒤의 BSS 로,
    "평균 정렬로 번 점수" 를 뺀 순수 신호다. 예측을 바꾸는 게 아니라 재기만 한다.

    근거: v88 에서 temporal_cyclic 이 raw -17.18(탈락) 인데 centered +4.75(2위) 였다.
    raw 로 고르면 신호 있는 전처리를 버린다.

효율
    조합마다 "피처 100개당 이득" 도 같이 낸다. 같은 이득이면 피처가 적은 쪽이 낫다.

배타 규칙
    drop_ids 와 id_frequency, no_trackman 과 trackman_*, no_component 와 component_*
    는 함께 켜지 않는다.

실행 (GPU 작업은 한 번에 하나. v85 가 끝난 뒤에)
    python v89_prep_beam.py --beam 3 --rounds 4
    python v89_prep_beam.py --dry          # 학습 없이 조합/피처수만 확인
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

HERE = Path(__file__).resolve().parent
SJ = HERE.parent
MODEL_OPT = SJ / "experiment" / "model_optimization"
sys.path.insert(0, str(MODEL_OPT))
sys.path.insert(0, str(HERE))

import v85_preprocess_screen as M
from run_optuna_enhanced import load_enhanced_frame
from run_optuna_family import (CATEGORICAL_COLUMNS, TARGET, probability_metrics,
                               recency_weights)
from v77_single_xgb_screen import build_component_unique, build_component_unique_forward
from v80_single_catboost import make_features, raw_bss

FOLD = 2024
SEED = 20262844
OUT = HERE / "outputs" / "prep_beam"
CACHE = HERE / "outputs" / "preprocess_screen"

ATOMS = ["ordinal_numeric", "id_frequency", "rate_multiscale", "rate_geometry",
         "count_multiscale", "recent_shape", "temporal_cyclic", "context_robust",
         "trackman_quality", "trackman_compact", "component_shape",
         "component_compact", "no_trackman", "no_component", "drop_ids"]
EXCLUSIVE = [{"drop_ids", "id_frequency"},
             {"no_trackman", "trackman_quality"}, {"no_trackman", "trackman_compact"},
             {"no_component", "component_shape"}, {"no_component", "component_compact"}]


def compatible(names) -> bool:
    s = set(names)
    return not any(pair <= s for pair in EXCLUSIVE)


def build_combo(base, base_features, names, train_mask, fold):
    """v85 의 원자 변환들을 순서대로 적용한다. v85 파일은 건드리지 않는다."""
    frame = base
    features = list(base_features)
    categorical = [c for c in CATEGORICAL_COLUMNS if c in features]
    extras, extra_cat = {}, []
    for name in names:
        if name == "ordinal_numeric":
            categorical = [c for c in categorical if c in M.NOMINAL_CATEGORICAL]
        elif name in ("drop_ids", "id_frequency"):
            features = [c for c in features if c not in M.ID_COLUMNS]
            categorical = [c for c in categorical if c not in M.ID_COLUMNS]
            if name == "id_frequency":
                extras.update(M.id_frequency(frame, train_mask))
        elif name == "rate_multiscale":
            extras.update(M.rate_multiscale(frame, train_mask))
        elif name == "rate_geometry":
            extras.update(M.rate_geometry(frame))
        elif name == "count_multiscale":
            values, cats = M.count_multiscale(frame)
            extras.update(values)
            extra_cat.extend(cats)
        elif name == "recent_shape":
            extras.update(M.recent_shape(frame))
        elif name == "temporal_cyclic":
            extras.update(M.temporal_cyclic(frame, fold))
        elif name == "context_robust":
            extras.update(M.context_robust(frame))
        elif name == "trackman_quality":
            extras.update(M.trackman_quality(frame, train_mask))
        elif name == "trackman_compact":
            features = M.compact_trackman(features)
        elif name == "component_shape":
            extras.update(M.component_shape(frame))
        elif name == "component_compact":
            features = M.compact_component(features)
        elif name == "no_trackman":
            features = [c for c in features if not c.startswith(("tm500_", "cw_"))]
        elif name == "no_component":
            features = [c for c in features if not c.startswith("sx_cf_")]
        elif name == "baseline":
            pass
        else:
            raise ValueError(name)
    frame = M.add_columns(frame, extras)
    features = list(dict.fromkeys(features + list(extras)))
    categorical = list(dict.fromkeys(
        [c for c in categorical + extra_cat if c in features]))
    return frame, features, categorical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--learning-rate", type=float, default=0.015)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--dry", action="store_true", help="학습 없이 조합만 확인")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    frame, enhanced_features = load_enhanced_frame()
    train_mask = frame["season"].lt(FOLD)
    valid_mask = frame["season"].eq(FOLD)
    y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    ybar = float(y.mean())

    static = build_component_unique(frame, enhanced_features, FOLD)
    forward = build_component_unique_forward(frame, enhanced_features, FOLD,
                                             cache={FOLD: static})
    base, f1_features = make_features(frame, enhanced_features, FOLD, "F1", forward)

    best = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(best.pop("half_life"))
    best.update({"iterations": args.iterations, "learning_rate": args.learning_rate,
                 "depth": args.depth, "random_seed": SEED,
                 "task_type": "GPU", "devices": "0", "verbose": 0})
    weights = recency_weights(frame.loc[train_mask, "season"], FOLD, half_life)

    scored: dict[tuple, dict] = {}

    def record(key, pred, n_features, src):
        s_raw = raw_bss(probability_metrics(y, pred))
        ctr = np.clip(pred - (float(pred.mean()) - ybar), 1e-6, 1 - 1e-6)
        scored[key] = {
            "combo": "+".join(key) if key else "baseline",
            "n_features": n_features,
            "bss_raw": s_raw,
            "bss_centered": raw_bss(probability_metrics(y, ctr)),
            "offset": float(pred.mean()) - ybar,
            "src": src,
        }
        return scored[key]

    def score(names):
        key = tuple(sorted(names))
        if key in scored:
            return scored[key]
        tag = "+".join(key) if key else "baseline"
        cached = CACHE / f"{tag}_{FOLD}.npy"
        if len(key) <= 1 and cached.exists():
            return record(key, np.load(cached).astype(np.float64), -1, "cache")
        own = OUT / f"{tag}_{FOLD}.npy"
        if own.exists():
            return record(key, np.load(own).astype(np.float64), -1, "cache")
        fr, feats, cats = build_combo(base, f1_features, key, train_mask, FOLD)
        if args.dry:
            scored[key] = {"combo": tag, "n_features": len(feats), "bss_raw": np.nan,
                           "bss_centered": np.nan, "offset": np.nan, "src": "dry"}
            del fr
            gc.collect()
            return scored[key]
        t0 = time.time()
        tr = Pool(fr.loc[train_mask, feats], frame.loc[train_mask, TARGET],
                  cat_features=cats, weight=weights)
        va = Pool(fr.loc[valid_mask, feats], y, cat_features=cats)
        model = CatBoostClassifier(**best)
        model.fit(tr, eval_set=va, use_best_model=True)
        pred = model.predict_proba(va)[:, 1].astype(np.float64)
        np.save(own, pred)
        out = record(key, pred, len(feats), f"fit {time.time() - t0:.0f}s")
        del fr, tr, va, model
        gc.collect()
        return out

    b0 = score(())
    print(f"baseline  centered {b0['bss_centered']:.2f}  raw {b0['bss_raw']:.2f}",
          flush=True)

    print(f"{chr(10)}라운드 1 - 단일 (v85 캐시 재사용)", flush=True)
    singles = []
    for atom in ATOMS:
        r = score((atom,))
        singles.append((atom, r["bss_centered"]))
        print(f"  {atom:<20}{r['bss_centered']:>10.2f}"
              f"{r['bss_centered'] - b0['bss_centered']:>+9.2f}  {r['src']}", flush=True)
    beam = [(a,) for a, _ in sorted(singles, key=lambda t: -t[1])[:args.beam]]

    for rd in range(2, args.rounds + 1):
        print(f"{chr(10)}라운드 {rd}", flush=True)
        cands = set()
        for combo in beam:
            for atom in ATOMS:
                if atom in combo:
                    continue
                nxt = tuple(sorted(combo + (atom,)))
                if compatible(nxt):
                    cands.add(nxt)
        results = []
        for c in sorted(cands):
            r = score(c)
            results.append((c, r))
            print(f"  {'+'.join(c):<58}{r['bss_centered']:>10.2f}"
                  f"{r['bss_centered'] - b0['bss_centered']:>+9.2f}  {r['src']}",
                  flush=True)
            pd.DataFrame(scored.values()).to_csv(OUT / "v89_prep_beam.csv", index=False)
        prev = max(scored[c]["bss_centered"] for c in beam)
        beam = [c for c, _ in
                sorted(results, key=lambda t: -t[1]["bss_centered"])[:args.beam]]
        if not beam or max(scored[c]["bss_centered"] for c in beam) <= prev:
            print("  개선 없음 - 종료", flush=True)
            break

    t = pd.DataFrame(scored.values()).sort_values("bss_centered", ascending=False)
    t["d_centered"] = t["bss_centered"] - b0["bss_centered"]
    t["d_per_100feat"] = np.where(t["n_features"] > 0,
                                  t["d_centered"] / (t["n_features"] / 100.0), np.nan)
    t.to_csv(OUT / "v89_prep_beam.csv", index=False)
    print(f"{chr(10)}{'=' * 96}{chr(10)}상위 12{chr(10)}{'=' * 96}")
    print(t.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
