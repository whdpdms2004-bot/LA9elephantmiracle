"""전처리 조합을 학습·평가한다. GPU 사용.

내장 변환 15개와 transforms/ 폴더의 기여 변환을 모두 쓸 수 있다.

사용
    # 등록된 변환 목록만 보기 (GPU 불필요)
    python run_combo.py --list

    # 특정 조합 하나
    python run_combo.py --combos id_frequency+temporal_cyclic

    # 여러 조합 (쉼표로 구분, + 로 결합)
    python run_combo.py --combos baseline,id_frequency,id_frequency+temporal_cyclic

    # 빔 서치로 조합 공간 탐색
    python run_combo.py --beam 3 --rounds 4

    # 학습 없이 피처 수/충돌만 확인
    python run_combo.py --combos my_idea --dry

주의
    GPU 작업은 한 번에 하나만 돌린다. 겹쳐 돌리면 둘 다 죽는다 (실제로 두 번 겪었다).
    시작 전에 nvidia-smi 로 확인할 것.
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

LAB = Path(__file__).resolve().parents[1]
SJ = LAB.parent
REPO = SJ.parents[1]
CAMPAIGN = SJ / "feature_campaign_1000"
MODEL_OPT = SJ / "experiment" / "model_optimization"
for p in (MODEL_OPT, CAMPAIGN, str(LAB)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = LAB / "outputs"
SEED = 20262844


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="등록된 변환 목록")
    ap.add_argument("--combos", default="", help="쉼표 구분. 각 조합은 + 로 결합")
    ap.add_argument("--beam", type=int, default=0, help=">0 이면 빔 서치")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--fold", type=int, default=2024)
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--learning-rate", type=float, default=0.015)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    import transforms as T
    reg = T.load_all()

    if args.list:
        print(f"등록된 변환 {len(reg)}개   (랩: {LAB})")
        print(f"  {'이름':<20}{'출처':<16}{'대상 피처':<34}설명")
        for name in sorted(reg):
            r = reg[name]
            print(f"  {name:<20}{r['source']:<16}"
                  f"{', '.join(r['targets'])[:32]:<34}{r['note']}")
            if r["conflicts"]:
                print(f"  {'':<20}배타: {', '.join(r['conflicts'])}")
        return

    print(f"랩       {LAB}")
    print(f"저장소   {REPO}")
    OUT.mkdir(parents=True, exist_ok=True)

    from catboost import CatBoostClassifier, Pool
    from run_optuna_enhanced import load_enhanced_frame
    from run_optuna_family import (CATEGORICAL_COLUMNS, TARGET,
                                   probability_metrics, recency_weights)
    from v77_single_xgb_screen import (build_component_unique,
                                       build_component_unique_forward)
    from v80_single_catboost import make_features, raw_bss
    import v85_preprocess_screen as M

    frame, enhanced = load_enhanced_frame()
    train_mask = frame["season"].lt(args.fold)
    valid_mask = frame["season"].eq(args.fold)
    y = frame.loc[valid_mask, TARGET].to_numpy("int8")
    ybar = float(y.mean())

    static = build_component_unique(frame, enhanced, args.fold)
    forward = build_component_unique_forward(frame, enhanced, args.fold,
                                             cache={args.fold: static})
    base, f1_features = make_features(frame, enhanced, args.fold, "F1", forward)
    # 변환이 라벨 기반 테이블(EB 평활 등)을 만들 수 있도록 라벨과 시즌을 넘긴다.
    # 학습 행에서만 쓰라는 규약은 transforms/example_template.py 에 명시돼 있다.
    for col in (TARGET, "season"):
        if col not in base.columns:
            base[col] = frame[col].to_numpy()

    best = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(best.pop("half_life"))
    best.update({"iterations": args.iterations, "learning_rate": args.learning_rate,
                 "depth": args.depth, "random_seed": SEED, "task_type": "GPU",
                 "devices": "0", "verbose": 0})
    weights = recency_weights(frame.loc[train_mask, "season"], args.fold, half_life)

    scored: dict[tuple, dict] = {}

    def record(key, pred, n_features, src):
        pred = np.clip(np.asarray(pred, np.float64), 1e-7, 1 - 1e-7)
        off = float(pred.mean()) - ybar
        ctr = np.clip(pred - off, 1e-7, 1 - 1e-7)
        scored[key] = {"combo": "+".join(key) if key else "baseline",
                       "n_atoms": len(key), "n_features": n_features,
                       "bss_raw": raw_bss(probability_metrics(y, pred)),
                       "bss_centered": raw_bss(probability_metrics(y, ctr)),
                       "offset": off, "src": src}
        return scored[key]

    def score(names):
        key = tuple(sorted(names))
        if key in scored:
            return scored[key]
        tag = "+".join(key) if key else "baseline"
        for cand in (OUT / f"{tag}_{args.fold}.npy",
                     CAMPAIGN / "outputs" / "preprocess_screen" / f"{tag}_{args.fold}.npy",
                     CAMPAIGN / "outputs" / "prep_beam" / f"{tag}_{args.fold}.npy"):
            if cand.exists():
                return record(key, np.load(cand), -1, "cache")
        cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
        fr, feats, cats = T.build(base, f1_features, cats0, key, train_mask, args.fold)
        if args.dry:
            scored[key] = {"combo": tag, "n_atoms": len(key), "n_features": len(feats),
                           "bss_raw": np.nan, "bss_centered": np.nan,
                           "offset": np.nan, "src": "dry"}
            del fr
            gc.collect()
            return scored[key]
        t0 = time.time()
        tr = Pool(fr.loc[train_mask, feats], frame.loc[train_mask, TARGET],
                  cat_features=cats, weight=weights)
        va = Pool(fr.loc[valid_mask, feats], y, cat_features=cats)
        model = CatBoostClassifier(**best)
        model.fit(tr, eval_set=va, use_best_model=True)
        pred = model.predict_proba(va)[:, 1]
        np.save(OUT / f"{tag}_{args.fold}.npy", pred)
        out = record(key, pred, len(feats), f"fit {time.time() - t0:.0f}s")
        del fr, tr, va, model
        gc.collect()
        return out

    def show(r, b0):
        print(f"  {r['combo']:<58}{r['bss_centered']:>10.2f}"
              f"{r['bss_centered'] - b0:>+9.2f}  {r['src']}", flush=True)

    b0 = score(())["bss_centered"]
    print(f"baseline centered {b0:.2f}")

    if args.combos:
        for spec in [c.strip() for c in args.combos.split(",") if c.strip()]:
            key = () if spec == "baseline" else tuple(spec.split("+"))
            if not T.compatible(key):
                print(f"  {spec:<58}  배타 규칙 위반 — 건너뜀")
                continue
            show(score(key), b0)

    if args.beam:
        atoms = sorted(reg)
        beam = [()]
        for rd in range(1, args.rounds + 1):
            print(f"{chr(10)}라운드 {rd}", flush=True)
            cands = {tuple(sorted(c + (a,))) for c in beam for a in atoms
                     if a not in c and T.compatible(c + (a,))}
            res = []
            for c in sorted(cands):
                res.append((c, score(c)))
                show(res[-1][1], b0)
                pd.DataFrame(scored.values()).to_csv(
                    OUT / f"run_combo_fold{args.fold}.csv", index=False)
            prev = max(scored[c]["bss_centered"] for c in beam)
            beam = [c for c, _ in
                    sorted(res, key=lambda t: -t[1]["bss_centered"])[:args.beam]]
            if not beam or max(scored[c]["bss_centered"] for c in beam) <= prev:
                print("  개선 없음 - 종료", flush=True)
                break

    t = pd.DataFrame(scored.values()).sort_values("bss_centered", ascending=False)
    t["d_centered"] = t["bss_centered"] - b0
    t.to_csv(OUT / f"run_combo_fold{args.fold}.csv", index=False)
    print(f"{chr(10)}{'=' * 96}{chr(10)}상위 15{chr(10)}{'=' * 96}")
    print(t.head(15).round(2).to_string(index=False))
    print(f"{chr(10)}saved -> {OUT / f'run_combo_fold{args.fold}.csv'}")


if __name__ == "__main__":
    main()
