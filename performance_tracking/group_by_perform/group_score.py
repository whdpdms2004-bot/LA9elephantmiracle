"""(모델 x 축 x 구간) 지표 전부 -> out/by_<axis>.csv · out/overall.csv.

지표 정의는 gbp_common.bin_metrics, 축은 gbp_common.AXES. PLAN.md §1~§3.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import gbp_common as G
from common import load_labels, load_pred          # performance_tracking/tools


def team_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """비음수·합=1 에서 Brier 최소. 자유해의 음수 가중은 전이되지 않는다
    (COLLAB_final_submissions.md §4-4)."""
    k = P.shape[1]
    # 예측 분산이 라벨 분산의 1% 미만이라 Brier 원척도에서는 기울기가 기본 tol
    # 아래로 깔린다 - SLSQP 가 초기점을 그대로 돌려준다. 1e6 배로 세워서 푼다.
    S = 1e6
    f = lambda w: S * float(np.mean((P @ w - y) ** 2))
    jac = lambda w: S * 2.0 * (P.T @ (P @ w - y)) / len(y)
    r = minimize(f, np.full(k, 1 / k), method="SLSQP", jac=jac,
                 bounds=[(0, 1)] * k,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                               "jac": lambda w: np.ones(k)}],
                 options={"maxiter": 500, "ftol": 1e-12})
    if not r.success:
        raise RuntimeError(f"TEAM 가중 적합 실패: {r.message}")
    return r.x


def build_preds(season: int, names: list[str], y: np.ndarray,
                labels: pd.DataFrame) -> tuple[dict, dict]:
    """이름 -> 예측. 행이 모자란 모델은 자기 부분집합 마스크를 같이 돌려준다."""
    out, masks = {}, {}
    for n in names:
        try:
            out[n] = load_pred(n, season, labels)
        except Exception as e:
            p = G.VAL / f"{n}_{season}.csv"
            df = pd.read_csv(p, dtype={"row_id": str}) if p.exists() else None
            if df is None or list(df.columns[:2]) != ["row_id", "pred"]:
                print(f"  [skip] {n}_{season}: {type(e).__name__} {e}")
                continue
            s = df.drop_duplicates("row_id").set_index("row_id")["pred"]
            hit = labels["row_id"].isin(s.index).to_numpy()
            if hit.mean() < 0.98:
                print(f"  [skip] {n}_{season}: 행 적중 {hit.mean():.1%} - 부분집합으로도 못 본다")
                continue
            out[n] = s.reindex(labels["row_id"][hit]).to_numpy(np.float64)
            masks[n] = hit
            print(f"  [부분집합] {n}_{season}: {hit.sum():,}/{len(hit):,} 행 "
                  f"({(~hit).sum()} 누락) - 이 모델만 자기 부분집합에서 잰다")
    return out, masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="val/ 의 54개 전부 (기본은 MAIN)")
    a = ap.parse_args()

    G.OUT.mkdir(exist_ok=True)
    rows, overall = [], []
    preds_by_season: dict[int, dict[str, np.ndarray]] = {}
    masks_by_season: dict[int, dict[str, np.ndarray]] = {}
    labels_by_season, axes_by_season = {}, {}

    for season in G.SEASONS:
        lab = load_labels(season)
        ax = G.load_axes(season)
        assert (ax.row_id.to_numpy() == lab.row_id.to_numpy()).all(), f"{season}: 행 정합 실패"
        y = lab["y"].to_numpy()
        names = G.both_season_models() if a.all else list(G.MAIN)
        if season == 2024:
            names = names + [m for m in G.MAIN_2024_ONLY if not a.all]
        print(f"[{season}] n={len(y):,}  모델 {len(names)}개")
        preds_by_season[season], masks_by_season[season] = build_preds(season, names, y, lab)
        labels_by_season[season], axes_by_season[season] = lab, ax

    # --- TEAM: 반대 시즌에서 가중을 적합해 동결한다 (PLAN.md §4) ------------- #
    mem = [m for m in G.TEAM_MEMBERS
           if all(m in preds_by_season[s] and m not in masks_by_season[s]
                  for s in G.SEASONS)]
    if len(mem) >= 2:
        for season in G.SEASONS:
            other = 2022 if season == 2024 else 2024
            Pf = np.column_stack([preds_by_season[other][m] for m in mem])
            w = team_weights(Pf, labels_by_season[other]["y"].to_numpy())
            Pe = np.column_stack([preds_by_season[season][m] for m in mem])
            preds_by_season[season]["TEAM"] = Pe @ w
            print(f"[TEAM] eval{season} <- fit{other}  " +
                  " ".join(f"{m}={wi:.3f}" for m, wi in zip(mem, w)))

    # scope: all = 시즌 전체 / R = 정규경기만.
    # game_type 은 2022->2023 에 구조적 단절이 있어(README §3) 2022 의 F 는 기저율이
    # 0.69~0.74 다. all 로만 보면 축 결론이 F 의 기저율 분리에 끌려간다.
    for season in G.SEASONS:
        lab, ax = labels_by_season[season], axes_by_season[season]
        y_full = lab["y"].to_numpy()
        is_R = (lab["game_type"].to_numpy() == "R")
        for scope in ("all", "R"):
            for name, p_full in preds_by_season[season].items():
                msub = masks_by_season[season].get(name)
                keep = np.ones(len(y_full), bool) if msub is None else msub
                sel = keep & is_R if scope == "R" else keep
                # p_full 은 keep 위에서만 정의돼 있다
                psel = p_full if scope == "all" else p_full[is_R[keep]]
                yy, axx = y_full[sel], ax[sel]
                nn = len(yy)
                nu = float(yy.mean() * (1 - yy.mean()))
                m = G.bin_metrics(yy, psel, nu, nn)
                overall.append({"model": name, "season": season, "scope": scope,
                                "subset": msub is not None, **m})
                for axis, (title, order) in G.AXES.items():
                    vals = axx[axis].to_numpy()
                    for b in (order or sorted(pd.unique(vals))):
                        msk = vals == b
                        if msk.sum() < 200:
                            continue
                        mm = G.bin_metrics(yy[msk], psel[msk], nu, nn)
                        rows.append({"model": name, "season": season, "scope": scope,
                                     "axis": axis, "axis_title": title, "bin": b, **mm})

    df = pd.DataFrame(rows)
    ov = pd.DataFrame(overall)
    df.to_csv(G.OUT / "by_bin.csv", index=False)
    ov.to_csv(G.OUT / "overall.csv", index=False)

    # 항등식 검증: Sigma deficit == 100000 - BSS_all (n<200 로 버린 구간만큼은 남는다)
    chk = df.groupby(["model", "season", "scope", "axis"]).deficit.sum().reset_index()
    chk = chk.merge(ov[["model", "season", "scope", "bss_local"]],
                    on=["model", "season", "scope"])
    chk["resid"] = chk.deficit - (100000 - chk.bss_local)
    print()
    print(f"항등식 잔차 max {chk.resid.abs().max():.2f} (n<200 구간 제외분 · "
          f"0.5 초과 {int((chk.resid.abs() > 0.5).sum())}건)")
    print(f"out/by_bin.csv {len(df):,}행 · out/overall.csv {len(ov)}행")
    for k, r in (("BSS", 1), ("AUC", 4)):
        col = "bss_local" if k == "BSS" else "auc"
        print()
        print(f"=== {k} ===")
        print(ov.pivot_table(index="model", columns=["season", "scope"], values=col)
                .round(r).to_string())


if __name__ == "__main__":
    main()
