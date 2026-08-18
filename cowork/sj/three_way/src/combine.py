"""S5: 세 하위 모델을 연결하고 최종 라벨로 미세조정한다. CPU 전용.

정직한 설계
    결합기를 fold 2023 하위예측으로 학습하고 fold 2024 에서 평가한다.
    같은 fold 로 학습·평가하면 결합기가 그 fold 에 과적합해 의미가 없다.

비교할 구성
    M/R/B     지시받은 구성
    M/R/O     항등식이 닫힌다 (실패 = m ∪ r ∪ o, 오차 0.000000)
    M/R/B/O   넷 다
    1WAY      프로덕션 base + 성분 라인 (기준선)

결합기 후보
    logit_lr    로지스틱 (하위 로짓의 선형 결합). 과적합이 가장 적다
    gbdt        얕은 GBDT. 상호작용 허용
    ie_resid    포함-배제 항등식으로 시작해 잔차만 학습 (M/R/O 에서만 가능)

★ 3WAY 가 이겼다고 말하려면
    정확도만으로 부족하다. **base 와의 상관이 1WAY(0.8219)보다 낮아야** 한다.
    V65·V66 이 양방향으로 확인한 원리 — 결합 가치는 정확도가 아니라 비상관성이 정한다.

사용
    python combine.py
    python combine.py --fit-fold 2023 --eval-fold 2024
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness3 import OUT, SUCCESS, TARGETS, bss, load_labeled

BEST = {
    "middle": "id_frequency+no_trackman+temporal_cyclic",
    "reverse": "count_multiscale+drop_ids+trackman_quality",
    "ball": "drop_ids+no_trackman+rate_multiscale",
}
EPS = 1e-7
lgt = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
sig = lambda z: 1.0 / (1.0 + np.exp(-z))


def load_pred(target: str, fold: int, combo: str | None = None):
    """저장된 하위 예측을 읽는다. 없으면 None."""
    tags = [combo] if combo else [BEST.get(target, "")]
    tags += ["drop_ids+no_trackman+rate_multiscale",
             "id_frequency+no_trackman+temporal_cyclic", "baseline"]
    for tag in tags:
        if not tag:
            continue
        for p in (OUT / f"{target}__{tag}__{fold}.npy",
                  OUT / f"s3_{target}__base__{fold}.npy"):
            if p.exists():
                return np.load(p).astype(np.float64), p.name
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-fold", type=int, default=2023)
    ap.add_argument("--eval-fold", type=int, default=2024)
    args = ap.parse_args()

    df = load_labeled()
    season = df["season"].to_numpy()
    y_all = pd.to_numeric(df[SUCCESS], errors="coerce").to_numpy(np.float64)
    ok_all = df["label_ok"].to_numpy() == 1

    subs = ["middle", "reverse", "ball", "outside", "mr"]
    data = {}
    for fold in (args.fit_fold, args.eval_fold):
        va = season == fold
        cols, srcs, n_ok = {}, {}, (va & ok_all).sum()
        for t in subs:
            p, src = load_pred(t, fold)
            if p is None:
                print(f"  ! {t} fold {fold} 예측 없음 — 건너뜀")
                continue
            if len(p) != n_ok:
                print(f"  ! {t} fold {fold} 길이 {len(p)} != {n_ok} — 건너뜀")
                continue
            cols[t], srcs[t] = p, src
        data[fold] = {"mask": va & ok_all, "p": cols, "src": srcs}
        print(f"fold {fold}: {len(cols)}개 하위예측  "
              f"({', '.join(f'{k}<-{v}' for k, v in srcs.items())})")

    have = sorted(set(data[args.fit_fold]["p"]) & set(data[args.eval_fold]["p"]))
    if len(have) < 2:
        print(f"{chr(10)}두 fold 모두에 있는 하위예측이 {len(have)}개뿐이다. "
              f"fold {args.fit_fold} 예측을 먼저 생성해야 한다:")
        for t, c in BEST.items():
            print(f"  python src/screen_target.py --fold {args.fit_fold} "
                  f"--target {t} --combos \"{c}\"")
        return
    print(f"{chr(10)}공통 하위예측: {have}")

    COMBOS = [c for c in (("middle", "reverse", "ball"),
                          ("middle", "reverse", "outside"),
                          ("middle", "reverse", "ball", "outside"),
                          ("middle", "reverse"))
              if set(c) <= set(have)]

    def XY(fold, keys):
        d = data[fold]
        m = d["mask"]
        X = np.column_stack([lgt(d["p"][k]) for k in keys])
        return X, y_all[m]

    def fit_lr(X, y):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=500, C=1.0).fit(X, y)

    def fit_gbdt(X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.05,
            random_state=20262844).fit(X, y)

    rows = []
    print(f"{chr(10)}{'=' * 104}")
    print(f"결합기 학습 fold {args.fit_fold}  ->  평가 fold {args.eval_fold}")
    print("=" * 104)
    print(f"  {'구성':<34}{'결합기':<10}{'BSS':>10}{'centered':>10}"
          f"{'오프셋':>9}{'base상관':>10}")

    ev = data[args.eval_fold]
    base_ref = None
    try:
        from harness3 import CAMPAIGN
        prod = (CAMPAIGN.parent / "experiment" / "model_optimization" /
                "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
        if prod.exists() and args.eval_fold == 2024:
            pr = pd.read_parquet(prod).set_index("row_id")
            rid = df.loc[ev["mask"], "row_id"].to_numpy()
            v = pr.reindex(rid)["submit021_reverse20_s040_tabm"].to_numpy(np.float64)
            if not np.isnan(v).all():
                base_ref = np.clip(v, EPS, 1 - EPS)
                m = bss(y_all[ev["mask"]], base_ref)
                print(f"  {'(프로덕션 base = 1WAY 기준선)':<34}{'':<10}"
                      f"{m['bss_raw']:>10.2f}{m['bss_centered']:>10.2f}"
                      f"{m['offset']:>+9.4f}{1.0:>10.4f}")
    except Exception as exc:                                    # noqa: BLE001
        print(f"  (프로덕션 base 로드 실패: {exc})")

    for keys in COMBOS:
        Xf, yf = XY(args.fit_fold, keys)
        Xe, ye = XY(args.eval_fold, keys)
        for name, fitter in (("logit_lr", fit_lr), ("gbdt", fit_gbdt)):
            try:
                model = fitter(Xf, yf)
                pe = np.clip(model.predict_proba(Xe)[:, 1], EPS, 1 - EPS)
            except Exception as exc:                            # noqa: BLE001
                print(f"  {'+'.join(keys):<34}{name:<10}  실패 {exc}")
                continue
            m = bss(ye, pe)
            corr = (float(np.corrcoef(lgt(base_ref), lgt(pe))[0, 1])
                    if base_ref is not None else np.nan)
            print(f"  {'+'.join(keys):<34}{name:<10}{m['bss_raw']:>10.2f}"
                  f"{m['bss_centered']:>10.2f}{m['offset']:>+9.4f}{corr:>10.4f}")
            rows.append({"combo": "+".join(keys), "combiner": name,
                         "base_corr": corr, **m})

        # 포함-배제 항등식 + 잔차 (M/R/O 에서만 성립)
        if set(keys) >= {"middle", "reverse", "outside"}:
            d_f, d_e = data[args.fit_fold]["p"], data[args.eval_fold]["p"]
            # mr 모델이 있으면 정확한 포함-배제, 없으면 독립 근사
            ident = lambda d: np.clip(
                1 - (d["middle"] + d["reverse"]
                     - (d["mr"] if "mr" in d else d["middle"] * d["reverse"])
                     + d["outside"]), EPS, 1 - EPS)
            zf = np.column_stack([lgt(ident(d_f))] + [lgt(d_f[k]) for k in keys])
            ze = np.column_stack([lgt(ident(d_e))] + [lgt(d_e[k]) for k in keys])
            model = fit_lr(zf, yf)
            pe = np.clip(model.predict_proba(ze)[:, 1], EPS, 1 - EPS)
            m = bss(ye, pe)
            corr = (float(np.corrcoef(lgt(base_ref), lgt(pe))[0, 1])
                    if base_ref is not None else np.nan)
            print(f"  {'+'.join(keys):<34}{'ie_resid':<10}{m['bss_raw']:>10.2f}"
                  f"{m['bss_centered']:>10.2f}{m['offset']:>+9.4f}{corr:>10.4f}")
            rows.append({"combo": "+".join(keys), "combiner": "ie_resid",
                         "base_corr": corr, **m})
            # 항등식 그대로 (학습 없음)
            m = bss(ye, ident(d_e))
            rows.append({"combo": "+".join(keys), "combiner": "identity",
                         "base_corr": (float(np.corrcoef(lgt(base_ref), lgt(ident(d_e)))[0, 1])
                                       if base_ref is not None else np.nan), **m})
            print(f"  {'+'.join(keys):<34}{'identity':<10}{m['bss_raw']:>10.2f}"
                  f"{m['bss_centered']:>10.2f}{m['offset']:>+9.4f}"
                  f"{float(np.corrcoef(lgt(base_ref), lgt(ident(d_e)))[0,1]) if base_ref is not None else np.nan:>10.4f}")

    if rows:
        t = pd.DataFrame(rows).sort_values("bss_centered", ascending=False)
        t.to_csv(OUT / f"combine_{args.eval_fold}.csv", index=False)
        print(f"{chr(10)}{'=' * 104}")
        print("★ 3WAY 가 이겼는지 판정")
        print("=" * 104)
        best = t.iloc[0]
        print(f"  최고: {best['combo']}  /  {best['combiner']}")
        print(f"    BSS {best['bss_raw']:.2f}   centered {best['bss_centered']:.2f}"
              f"   base 상관 {best['base_corr']:.4f}")
        print(f"{chr(10)}  1WAY 성분 라인의 base 상관은 0.8219 였다.")
        if not np.isnan(best["base_corr"]):
            if best["base_corr"] < 0.8219:
                print(f"  -> 3WAY 가 더 낮다 ({best['base_corr']:.4f}). "
                      f"결합 멤버로서 값이 있다.")
            else:
                print(f"  -> 3WAY 가 더 높다 ({best['base_corr']:.4f}). "
                      f"정확도가 올라도 결합에서 이기기 어렵다 (V65·V66).")
        print(f"{chr(10)}saved -> {OUT / f'combine_{args.eval_fold}.csv'}")


if __name__ == "__main__":
    main()
