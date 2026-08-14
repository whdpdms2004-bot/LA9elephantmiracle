"""P7 (E2 확정): 성분 결합을 올바른 분모로 재판정.

P6의 판정에는 방법론 오류가 있었다. 채택 문턱을 '절대 BSS의 시드 sigma'로 잡았는데,
비교 대상인 혼합 예측은 direct 예측 p_a 를 그대로 포함한다. 같은 시드/데이터/트리를
공유하므로 두 BSS의 노이즈가 거의 완전히 상관돼 있고, 차이의 표준오차는 절대 sigma보다
훨씬 작다 (w=0이면 차이가 정확히 0, 분산도 0). 절대 sigma로 차이를 재면 문턱이
필요 이상으로 빡빡해진다.

세 가지 분모를 모두 낸다.
  (1) sigma_abs   절대 BSS의 시드 산포          - 독립 구성 비교용 (참고)
  (2) sigma_pair  시드별 dBSS 의 산포            - 같은 base를 공유하는 변형 판정용
  (3) SE_row      행 단위 쌍대 Brier 차의 표준오차 - 다른 행으로의 전이 판정용
                  d_i = (p_new_i - y_i)^2 - (p_old_i - y_i)^2, SE = sd(d)/sqrt(n)

문턱을 유의수준보다 높게 두는 근거는 노이즈가 아니라 다중비교다. 이번 라운드에서
변형을 12개 이상 보고 그중 최고를 고르므로 선택 편향 마진이 필요하다.

동시에 P6에서 드러난 두 가지를 반영한다.
  - R BSS 는 세 fold 모두 개선 (+7.2 / +23.1 / +17.2), F 는 모두 악화
    -> 결합을 R 행에만 적용하는 변형을 함께 평가한다
  - 가중치 w 는 2022 에서만 고르고 2023 부호 확인, 2024 는 게이트 1회

출력: outputs/p7_component_confirm.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics)

SEEDS = list(range(101, 125))          # 24 seeds
DROP = ["row_id", TARGET]
N_ROUNDS = 300
VALID_SEASONS = [2022, 2023, 2024]
EPS = 1e-6
WS = [0.10, 0.20, 0.30, 0.40, 0.50]

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_succ = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1


def as_label(col):
    v = df[col].to_numpy(np.float64)
    return np.where(ok, v, np.nan)


y_m, y_r, y_o = as_label("y_middle"), as_label("y_reverse"), as_label("y_outside")
y_mr = np.where(ok, (y_m == 1) & (y_r == 1), np.nan)


def forecast_rate(labels, seasons, tr_mask, valid_season):
    m = tr_mask & ~np.isnan(labels)
    s = pd.Series(labels[m]).groupby(pd.Series(seasons[m])).mean().sort_index()
    last_s, last_r = float(s.index[-1]), float(s.iloc[-1])
    slope = (last_r - float(s.iloc[0])) / (last_s - float(s.index[0]))
    return float(np.clip(last_r + slope * (valid_season - last_s), 0.005, 0.995))


def bag_predict_perseed(X, tr_mask, pr_mask, labels, base_score):
    """시드별 예측을 모두 반환한다. 시드 산포 측정에 쓴다."""
    m = tr_mask & ~np.isnan(labels)
    d_tr = xgb.DMatrix(X[m], label=labels[m])
    d_pr = xgb.DMatrix(X[pr_mask])
    out = np.empty((len(SEEDS), int(pr_mask.sum())), dtype=np.float64)
    for i, seed in enumerate(SEEDS):
        bst = xgb.train({**BASE_PARAMS, "base_score": base_score, "seed": seed},
                        d_tr, num_boost_round=N_ROUNDS, verbose_eval=False)
        out[i] = bst.predict(d_pr)
    return out


rows = []
for vs in VALID_SEASONS:
    tr_mask, va_mask = season < vs, season == vs
    priors = make_priors(df.loc[tr_mask])
    feat = encode(add_stateless(df, priors))
    cols = [c for c in feat.columns if c not in DROP and not c.startswith("y_")
            and c != "label_ok"]
    X = feat[cols].to_numpy(np.float32)
    yv = y_succ[va_mask]
    gt = df.loc[va_mask, "game_type"].astype(str).to_numpy()
    is_r = gt == "R"
    print(f"\n{'='*94}\nvalid {vs}   seeds {len(SEEDS)}   val {va_mask.sum():,}  "
          f"(R {is_r.sum():,} / F {(~is_r).sum():,})\n{'='*94}", flush=True)

    per = {}
    for tag, arr in [("succ", y_succ), ("m", y_m), ("r", y_r), ("o", y_o), ("mr", y_mr)]:
        bs = forecast_rate(arr, season, tr_mask, vs)
        per[tag] = bag_predict_perseed(X, tr_mask, va_mask, arr, bs)
        print(f"  [{tag:>4}] done  pred_mean {per[tag].mean():.5f}", flush=True)

    # (1) 절대 sigma — 참고용
    single = np.array([metrics(yv, np.clip(per["succ"][i], EPS, 1 - EPS))["bss_raw"]
                       for i in range(len(SEEDS))])
    sd_abs = float(single.std(ddof=1))
    null = yv.mean() * (1 - yv.mean())
    print(f"  (1) sigma_abs 단일시드 {sd_abs:.3f}  -> {len(SEEDS)}시드 배깅 "
          f"{sd_abs/np.sqrt(len(SEEDS)):.3f}", flush=True)

    p = {k: np.clip(v.mean(axis=0), EPS, 1 - EPS) for k, v in per.items()}
    p_a = p["succ"]
    p_ie = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["o"]), EPS, 1 - EPS)
    base_m = metrics(yv, p_a, game_type=gt)

    def build(w, r_only):
        out = p_a.copy()
        sel = is_r if r_only else np.ones(len(p_a), bool)
        out[sel] = np.clip(w * p_ie[sel] + (1 - w) * p_a[sel], EPS, 1 - EPS)
        return out

    variants = {"A_direct": (p_a, None, None), "C_incl_excl": (p_ie, 1.0, False)}
    for w in WS:
        variants[f"E_ie_w{w:.2f}"] = (build(w, False), w, False)
        variants[f"F_ieR_w{w:.2f}"] = (build(w, True), w, True)

    print(f"\n  {'variant':<16}{'ΔBSS':>9}{'σ_pair':>8}{'t_pair':>8}"
          f"{'SE_row':>8}{'t_row':>8}{'ΔR':>9}{'ΔF':>10}", flush=True)
    for name, (pv, w, r_only) in variants.items():
        mm = metrics(yv, pv, game_type=gt)
        dbss = mm["bss_raw"] - base_m["bss_raw"]

        # (2) 쌍대 시드 sigma — 시드별로 같은 구성을 만들어 dBSS 산포를 잰다
        if w is None:
            sd_pair = t_pair = np.nan
        else:
            ds = []
            for i in range(len(SEEDS)):
                pa_i = np.clip(per["succ"][i], EPS, 1 - EPS)
                ie_i = np.clip(1 - (per["m"][i] + per["r"][i]
                                    - per["mr"][i] + per["o"][i]), EPS, 1 - EPS)
                new_i = pa_i.copy()
                sel = is_r if r_only else np.ones(len(pa_i), bool)
                new_i[sel] = np.clip(w * ie_i[sel] + (1 - w) * pa_i[sel], EPS, 1 - EPS)
                b_old = np.mean((pa_i - yv) ** 2)
                b_new = np.mean((new_i - yv) ** 2)
                ds.append(100000 * (b_old - b_new) / null)
            ds = np.array(ds)
            sd_pair = float(ds.std(ddof=1))
            t_pair = float(ds.mean() / (sd_pair / np.sqrt(len(ds)))) if sd_pair > 0 else np.nan

        # (3) 행 단위 쌍대 검정 — 다른 행으로 전이될지
        d_row = (p_a - yv) ** 2 - (pv - yv) ** 2
        se_row = 100000 * float(d_row.std(ddof=1) / np.sqrt(len(d_row))) / null
        t_row = dbss / se_row if se_row > 0 else np.nan

        row = {"valid_season": vs, "variant": name, "w": w, "r_only": r_only,
               "sigma_abs_single": sd_abs, "sigma_abs_bag": sd_abs / np.sqrt(len(SEEDS)),
               "bss": mm["bss_raw"], "dbss": dbss,
               "sigma_pair": sd_pair, "t_pair": t_pair,
               "se_row": se_row, "t_row": t_row,
               "r_bss": mm.get("r_bss"), "dr": mm.get("r_bss") - base_m.get("r_bss"),
               "f_bss": mm.get("f_bss"), "df": mm.get("f_bss") - base_m.get("f_bss"),
               "brier": mm["brier"], "pred_mean": mm["pred_mean"]}
        rows.append(row)
        print(f"  {name:<16}{dbss:>9.2f}{sd_pair:>8.2f}{t_pair:>8.2f}"
              f"{se_row:>8.2f}{t_row:>8.2f}{row['dr']:>9.2f}{row['df']:>10.2f}",
              flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "p7_component_confirm.csv", index=False)

print("\n" + "=" * 94)
print("★ 전체 BSS 기준 ΔBSS — 이것이 대회 지표다")
print("=" * 94)
print(res.pivot(index="variant", columns="valid_season", values="dbss").round(2).to_string())
print("\n분해 (참고): R ΔBSS")
print(res.pivot(index="variant", columns="valid_season", values="dr").round(2).to_string())
print("\n분해 (참고): F ΔBSS")
print(res.pivot(index="variant", columns="valid_season", values="df").round(2).to_string())

print("\n" + "=" * 94)
print("정직한 게이트 — w는 2022 전체 BSS로만 고르고 2024는 1회 적용")
print("=" * 94)
for fam in ["E_ie", "F_ieR"]:
    sub22 = res[(res.valid_season == 2022) & res.variant.str.startswith(fam)]
    best = sub22.sort_values("dbss", ascending=False).iloc[0]
    chosen = res[res.variant == best.variant].set_index("valid_season")
    print(f"\n[{fam}] 2022 전체 BSS 최고: {best.variant}  (w={best.w})")
    print(chosen[["dbss", "sigma_pair", "t_pair", "se_row", "t_row", "dr", "df"]]
          .round(3).to_string())
    g = chosen.loc[2024]
    marks = []
    marks.append(f"쌍대시드 t={g.t_pair:.2f}" if np.isfinite(g.t_pair) else "")
    marks.append(f"행쌍대 t={g.t_row:.2f}")
    print(f"  2024 전체 ΔBSS {g.dbss:+.2f}   {' / '.join(m for m in marks if m)}")
    print(f"  참고: 절대 sigma 기준 1.5σ = {1.5*g.sigma_abs_bag:.2f} "
          f"(차이 판정에는 과도한 자)")

print("\n" + "=" * 94)
print("전체 BSS가 세 fold 모두 양수인 변형")
print("=" * 94)
piv = res.pivot(index="variant", columns="valid_season", values="dbss")
allpos = piv[(piv > 0).all(axis=1)]
print(allpos.round(2).to_string() if len(allpos) else "  없음")
print("\n2022/2024 두 fold 양수 (2023은 F 레짐붕괴가 전체를 삼킴)")
two = piv[(piv[2022] > 0) & (piv[2024] > 0)]
print(two.round(2).to_string() if len(two) else "  없음")

print(f"\nsaved -> {OUT/'p7_component_confirm.csv'}")
