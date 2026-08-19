"""3WAY 재결합을 정확하게 만든다 — 그리고 1WAY 직접 예측과 정면 비교한다.

왜 필요한가 (실측)
    way 별 BSS 는 각자의 null 기준으로 계산되므로 (배수 4.01~9.27) 최종 지표로
    번역되지 않는다. way 별로 1500~1770 을 받아도 세 예측을 합쳐 제구 성공
    확률로 만들면 fold 2024 473.5 / fold 2023 -218.2 였다.
    같은 조건의 1WAY 직접 예측은 fold 2024 717.2 다 — 3WAY 재결합이 243 낮다.

    다만 그 재결합은 조잡했다 (합 x 0.939 스케일링). 정확히 할 수 있다.

정확한 항등식
    세 way 합의 분포를 세면  0: 771,703  1: 651,561  2: 50,244  3: **0 행**
    3중 겹침이 없으므로 포함-배제가 한 항으로 끝난다.

        p(fail) = p(m) + p(r) + p(o) - p(겹침)
        겹침 = 1{m+r+o == 2}     기저율 3.41%

    겹침을 모델로 채우면 재결합에 근사가 남지 않는다.
    fail 과 success 는 여집합이므로 p(success) = 1 - p(fail).

무엇을 비교하나
    A. 3WAY 정확 재결합   (세 way + 겹침 모델)
    B. 3WAY 근사 재결합   (겹침 무시 / 상수 스케일)
    C. 1WAY 직접 예측     (control_success 를 바로 학습)

    셋 다 같은 전처리/가중치/시드 조건에서 잰다. 이게 3WAY 가 값이 있는지에 대한
    직접적인 답이다.

규정
    겹침 라벨은 학습 시즌의 라벨로만 만든다. 채점 fold 라벨은 쓰지 않는다.
    조기 종료도 쓰지 않는다.

사용
    python three_way_combine.py --folds 2024,2023
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guards import assert_features_clean, save_prediction, train_season_trend
from harness3 import LAB, OUT, SUCCESS, TARGETS, bss, load_labeled

SEED = 20262844
EPS = 1e-7
lgt = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
sig = lambda z: 1.0 / (1.0 + np.exp(-z))
HONEST = ("ww_", "ms_", "mn_", "wb_", "mh_", "mt_")

# way 별 정직한 최선 (시드 접미사는 제외한 기본 이름)
WAY_BEST = {
    "middle": "split_ball_drop_2019+drop_2020+drop_2022+fw020",
    "reverse": "single_drop_2022+fw020_no_trackman_kb_it90",
    "outside": "single_recency_kb",
}
OVL_COMBO = "drop_ids+no_trackman+rate_multiscale"


def load_way(tg: str, name: str, fold: int, n: int) -> np.ndarray | None:
    """같은 설정의 시드들을 모아 로짓 평균(시드 배깅)한다."""
    got = []
    for p in glob.glob(str(OUT / f"*{tg}__*__{fold}.npy")):
        b = Path(p).name
        if not b.startswith(HONEST):
            continue
        m = re.match(rf"\w+?_{tg}__(.+)__{fold}\.npy$", b)
        if not m or re.sub(r"_s\d+$", "", m.group(1)) != name:
            continue
        v = np.load(p).astype(np.float64)
        if len(v) == n:
            got.append(v)
    if not got:
        return None
    return sig(np.mean([lgt(v) for v in got], axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="2024,2023")
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--seeds", default="20262844,7,101")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    from catboost import CatBoostClassifier, Pool
    from run_optuna_enhanced import load_enhanced_frame
    from run_optuna_family import CATEGORICAL_COLUMNS, recency_weights
    from v77_single_xgb_screen import (build_component_unique,
                                       build_component_unique_forward)
    from v80_single_catboost import make_features
    import v85_preprocess_screen as M
    sys.path.insert(0, str(LAB))
    import transforms as T
    T.load_all()

    frame, enhanced = load_enhanced_frame()
    labeled = load_labeled()
    season = frame["season"].to_numpy()
    num = lambda c: pd.to_numeric(labeled[c], errors="coerce").to_numpy(np.float64)
    ym, yr, yo = num("y_middle"), num("y_reverse"), num("y_outside")
    suc = num(SUCCESS)
    ok = ((labeled["label_ok"].to_numpy() == 1) & ~np.isnan(suc)
          & ~np.isnan(ym) & ~np.isnan(yr) & ~np.isnan(yo))
    tot = ym + yr + yo
    assert int((tot[ok] > 2).sum()) == 0, "3중 겹침이 있으면 항등식이 달라진다"
    assert np.abs((tot[ok] > 0).astype(float) - (1 - suc[ok])).max() < 1e-9, \
        "실패 = m∪r∪o 가 성립하지 않는다"
    yov = (tot == 2).astype(np.float64)          # 겹침 라벨
    print(f"겹침 라벨 기저율 {yov[ok].mean():.4f}   3중 겹침 0행 확인")

    P0 = json.loads(M.PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(P0.pop("half_life"))
    P0.update({"iterations": args.iterations, "learning_rate": 0.015, "depth": 8,
               "task_type": "GPU", "devices": "0", "verbose": 0})

    rows = []
    for fold in [int(f) for f in args.folds.split(",") if f.strip()]:
        tr, va = (season < fold) & ok, (season == fold) & ok
        n = int(va.sum())
        Ys = suc[va]
        print(f"{chr(10)}{'=' * 96}")
        print(f"fold {fold}   검증 {n:,}행   제구성공 실제 평균 {Ys.mean():.4f}")
        print("=" * 96, flush=True)

        # 겹침 모델 (시드 배깅)
        pov = None
        need = [int(x) for x in args.seeds.split(",") if x.strip()]
        cached = [OUT / f"ov_overlap__s{s}__{fold}.npy" for s in need]
        if all(c.exists() for c in cached):
            pov = sig(np.mean([lgt(np.load(c).astype(float)) for c in cached], 0))
        else:
            t0 = time.time()
            static = build_component_unique(frame, enhanced, fold)
            forward = build_component_unique_forward(frame, enhanced, fold,
                                                     cache={fold: static})
            base_fr, f1 = make_features(frame, enhanced, fold, "F1", forward)
            for c in (SUCCESS, "season"):
                if c not in base_fr.columns:
                    base_fr[c] = frame[c].to_numpy()
            cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1]
            fr, feats, cat_cols = T.build(
                base_fr, f1, cats0, tuple(x for x in OVL_COMBO.split("+") if x),
                pd.Series(tr, index=frame.index), fold)
            assert_features_clean(feats, "overlap")
            w = np.asarray(recency_weights(frame.loc[tr, "season"], fold,
                                           half_life), np.float64)
            pr = train_season_trend(yov[tr], season[tr], fold)
            bl = float(np.log(pr / (1 - pr)))
            outs = []
            for sd, cp in zip(need, cached):
                if cp.exists():
                    outs.append(np.load(cp).astype(float)); continue
                p = dict(P0); p.update({"random_seed": sd, "bootstrap_type": "Bayesian"})
                p.pop("subsample", None)
                p.update({"depth": 6, "l2_leaf_reg": 10.0})     # 얇은 타깃
                p_tr = Pool(fr.loc[tr, feats], yov[tr].astype("int8"),
                            cat_features=cat_cols, weight=w,
                            baseline=np.full(int(tr.sum()), bl))
                p_va = Pool(fr.loc[va, feats], cat_features=cat_cols,
                            baseline=np.full(n, bl))
                mdl = CatBoostClassifier(**p)
                mdl.fit(p_tr)
                v = np.clip(mdl.predict_proba(p_va)[:, 1], EPS, 1 - EPS)
                save_prediction(cp, v, yov[va], where=f"overlap/{sd}")
                outs.append(v)
                del p_tr, p_va, mdl
                gc.collect()
            pov = sig(np.mean([lgt(v) for v in outs], 0))
            print(f"  겹침 모델 {len(outs)}시드 {time.time() - t0:.0f}s  "
                  f"평균 {pov.mean():.4f} (실제 {yov[va].mean():.4f})  "
                  f"BSS {bss(yov[va].astype(int), pov)['bss_raw']:.1f}", flush=True)
            del fr, base_fr
            gc.collect()

        ways = {k: load_way(k, v, fold, n) for k, v in WAY_BEST.items()}
        miss = [k for k, v in ways.items() if v is None]
        if miss:
            print(f"  way 예측 없음: {miss} — 건너뛴다"); continue
        pm, prv, po = ways["middle"], ways["reverse"], ways["outside"]
        for k, v in ways.items():
            print(f"  {k:<9} 평균 {v.mean():.4f}")

        cand = {
            "3WAY 정확 (겹침 모델)": np.clip(pm + prv + po - pov, EPS, 1 - EPS),
            "3WAY 겹침 무시": np.clip(pm + prv + po, EPS, 1 - EPS),
            "3WAY 독립가정 곱": np.clip(1 - (1 - pm) * (1 - prv) * (1 - po), EPS, 1 - EPS),
        }
        for nm, name in (("1WAY 직접", "single_recency_kb"),
                         ("1WAY 직접(drop22)", "single_drop_2022+fw020_kb")):
            v = load_way("success", name, fold, n)
            if v is not None:
                cand[nm] = np.clip(1 - v, EPS, 1 - EPS)   # v 는 성공확률

        print(f"{chr(10)}  {'방식':<26}{'제구성공 BSS':>14}{'예측평균':>10}"
              f"{'실제':>9}{'오프셋':>10}")
        for nm, pf in cand.items():
            ps = np.clip(1 - pf, EPS, 1 - EPS)
            b = bss(Ys, ps)
            print(f"  {nm:<26}{b['bss_raw']:>14.1f}{ps.mean():>10.4f}"
                  f"{Ys.mean():>9.4f}{b['offset']:>+10.4f}")
            rows.append({"fold": fold, "방식": nm, **b})

    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(OUT / "three_way_combine.csv", index=False)
        print(f"{chr(10)}saved -> {OUT / 'three_way_combine.csv'}")
        p = t.pivot_table(index="방식", columns="fold", values="bss_raw")
        print(f"{chr(10)}{p.to_string(float_format=lambda v: f'{v:10.1f}')}")


if __name__ == "__main__":
    main()
