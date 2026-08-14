"""P10: 검증된 기법을 전부 넣어 성분 분해 라인을 '본체'로 세운다.

P9의 교훈: 성분 결합을 프로덕션(836) 위 증분으로 얹으면 p_ie 가 652밖에 안 돼서
가중치가 0.05까지 눌리고 이득이 +2.6 (노이즈)로 사라진다. 약한 재료를 강한 예측에
소량 섞는 구조 자체가 한계다.

방향 전환: 성분 라인을 강하게 만들어 '독립 라인'으로 세우고 강한 base 와 대등하게
결합한다. 836 을 혼자 이기는 게 목표가 아니라 상관이 낮은 파트너를 만드는 것이다.

강한 base
    enhanced_seed_oof_parts/ 의 25종 x 3 fold (209피처 seedbag3 OOF).
    2022/2023/2024 전부 있으므로 결합 가중치를 2022에서 정직하게 고를 수 있다.

성분 라인
    피처를 77 -> 약 110 으로 늘린다 (원본47 + stateless파생 + 프로파일 + 최근폼 파생).
    5개 타깃(succ, m, r, o, mr) 각각 시즌 drift 보정 base_score + 시드 배깅.
    p_ie = 1 - (p_m + p_r - p_mr + p_o)      # 포함배제, 독립 가정 없음

결합
    R 행에만 적용 (F 는 레짐 붕괴로 성분 모델도 같이 망가진다 — P7 실측)
    w 는 2022 전체 BSS 에서만 선택, 2023 부호 확인, Val2024 1회 게이트

출력: outputs/p10_full_stack.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, forecast_base_rate, metrics)

OOF_DIR = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
           / "enhanced_seed_oof_parts")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 400
FOLDS = [2022, 2023, 2024]
WS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]
EPS = 1e-7

RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
         "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
         "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
         "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
PREV_S = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
PREV_M = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]


def logit(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
asof_n = df["asof_pitcher_n"].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
gt_all = df["game_type"].astype(str).to_numpy()


def as_label(c):
    return np.where(ok, df[c].to_numpy(np.float64), np.nan)


y_m, y_r, y_o = as_label("y_middle"), as_label("y_reverse"), as_label("y_outside")
y_mr = np.where(ok, (y_m == 1) & (y_r == 1), np.nan)

# ------------------------------------------------------------- 강한 base
print("강한 base 로드 (enhanced 209피처 seedbag3 OOF, 25종)", flush=True)
models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", f).group(1)
                 for f in map(lambda p: p.name, OOF_DIR.glob("*.parquet"))})
strong = {}
for fold in FOLDS:
    acc, cnt, ids = None, 0, None
    for mname in models:
        p = OOF_DIR / f"{mname}_fold{fold}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p).set_index("row_id")
        if ids is None:
            ids = df.loc[season == fold, "row_id"].to_numpy()
        v = d.reindex(ids)["prediction"].to_numpy(np.float64)
        acc = v if acc is None else acc + v
        cnt += 1
    strong[fold] = np.clip(acc / cnt, EPS, 1 - EPS)
    m = metrics(y_all[season == fold], strong[fold], game_type=gt_all[season == fold])
    print(f"  fold {fold}  {cnt}종 평균  BSS {m['bss_raw']:9.3f}  "
          f"R {m['r_bss']:9.3f}  F {m['f_bss']:10.3f}  pred_mean {m['pred_mean']:.5f}",
          flush=True)


# -------------------------------------------------------- 확장 피처 (약 110)
def build_features(tr_mask):
    priors = make_priors(df.loc[tr_mask])
    base = encode(add_stateless(df, priors))
    cols = [c for c in base.columns if c not in DROP and not c.startswith("y_")
            and c != "label_ok"]
    out = base[cols].copy()
    n = asof_n
    for c in RATES:                                  # 수축 프로파일
        pr = float(df.loc[tr_mask, c].median())
        r = np.where(np.isnan(df[c].to_numpy(np.float64)), pr,
                     df[c].to_numpy(np.float64))
        out[f"prof200_{c}"] = (n * r + 200 * pr) / (n + 200)
    ps = {c: np.where(np.isnan(df[c].to_numpy(np.float64)),
                      float(df.loc[tr_mask, c].median()),
                      df[c].to_numpy(np.float64)) for c in PREV_S + PREV_M}
    out["prev_trend_s"] = ps[PREV_S[0]] - ps[PREV_S[2]]
    out["prev_trend_m"] = ps[PREV_M[0]] - ps[PREV_M[2]]
    out["prev_slope_s"] = (ps[PREV_S[0]] - ps[PREV_S[1]]) - (ps[PREV_S[1]] - ps[PREV_S[2]])
    out["prev_std_s"] = np.std(np.vstack([ps[c] for c in PREV_S]), axis=0)
    out["prev_std_m"] = np.std(np.vstack([ps[c] for c in PREV_M]), axis=0)
    out["prev_range_s"] = (np.max(np.vstack([ps[c] for c in PREV_S]), axis=0)
                           - np.min(np.vstack([ps[c] for c in PREV_S]), axis=0))
    out["prev_miss_cnt"] = sum(np.isnan(df[c].to_numpy(np.float64)).astype(np.float64)
                               for c in PREV_S + PREV_M)
    for k, (cs, cm) in enumerate(zip(PREV_S, PREV_M)):   # 최근 실패가 middle 때문인가
        out[f"faildir_{k}"] = ps[cm] - (1 - ps[cs])
    rel = n / (n + 200.0)
    for c in PREV_S:
        out[f"relw_{c}"] = ps[c] * rel
    out["rel200"] = rel
    return out.to_numpy(np.float32), list(out.columns)


def bag(X, tr_mask, pr_mask, labels, bs):
    m = tr_mask & ~np.isnan(labels)
    d_tr = xgb.DMatrix(X[m], label=labels[m])
    d_pr = xgb.DMatrix(X[pr_mask])
    acc = np.zeros(int(pr_mask.sum()))
    for s in SEEDS:
        acc += xgb.train({**BASE_PARAMS, "base_score": bs, "seed": s},
                         d_tr, num_boost_round=N_ROUNDS,
                         verbose_eval=False).predict(d_pr)
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


def forecast_rate(labels, tr_mask, vs):
    m = tr_mask & ~np.isnan(labels)
    s = pd.Series(labels[m]).groupby(pd.Series(season[m])).mean().sort_index()
    ls, lr = float(s.index[-1]), float(s.iloc[-1])
    slope = (lr - float(s.iloc[0])) / (ls - float(s.index[0]))
    return float(np.clip(lr + slope * (vs - ls), 0.005, 0.995))


rows = []
for vs in FOLDS:
    tr, va = season < vs, season == vs
    X, names = build_features(tr)
    y_va, gt = y_all[va], gt_all[va]
    is_r = gt == "R"
    p_s = strong[vs]
    base_m = metrics(y_va, p_s, game_type=gt)
    null = y_va.mean() * (1 - y_va.mean())
    print(f"\n{'='*100}\nfold {vs}   강한base BSS {base_m['bss_raw']:.3f}   "
          f"피처 {len(names)}   R {is_r.sum():,}/F {(~is_r).sum():,}\n{'='*100}",
          flush=True)

    comp = {}
    for tag, arr in [("succ", y_all), ("m", y_m), ("r", y_r), ("o", y_o), ("mr", y_mr)]:
        comp[tag] = bag(X, tr, va, arr, forecast_rate(arr, tr, vs))
        act = np.nanmean(arr[va])
        print(f"  [{tag:>4}] pred_mean {comp[tag].mean():.5f}  actual {act:.5f}",
              flush=True)
    p_dir = comp["succ"]
    p_ie = np.clip(1 - (comp["m"] + comp["r"] - comp["mr"] + comp["o"]), EPS, 1 - EPS)

    m_dir = metrics(y_va, p_dir, game_type=gt)
    m_ie = metrics(y_va, p_ie, game_type=gt)
    print(f"\n  내 direct  BSS {m_dir['bss_raw']:9.3f}", flush=True)
    print(f"  내 p_ie    BSS {m_ie['bss_raw']:9.3f}", flush=True)
    c_dir = float(np.corrcoef(logit(p_s), logit(p_dir))[0, 1])
    c_ie = float(np.corrcoef(logit(p_s), logit(p_ie))[0, 1])
    c_di = float(np.corrcoef(logit(p_dir), logit(p_ie))[0, 1])
    print(f"  로짓 상관  strong~direct {c_dir:.4f}   strong~ie {c_ie:.4f}   "
          f"direct~ie {c_di:.4f}", flush=True)

    def ev(name, p):
        p = np.clip(p, EPS, 1 - EPS)
        mm = metrics(y_va, p, game_type=gt)
        d = mm["bss_raw"] - base_m["bss_raw"]
        dr = (p_s - y_va) ** 2 - (p - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": vs, "variant": name, "bss": mm["bss_raw"], "dbss": d,
                     "se_row": se, "t_row": d / se if se > 0 else np.nan,
                     "r_bss": mm["r_bss"], "f_bss": mm["f_bss"],
                     "brier": mm["brier"], "pred_mean": mm["pred_mean"],
                     "corr_strong_ie": c_ie, "corr_strong_direct": c_dir,
                     "strong_bss": base_m["bss_raw"], "ie_bss": m_ie["bss_raw"],
                     "direct_bss": m_dir["bss_raw"]})

    ev("S_strong", p_s)
    ev("M_direct", p_dir)
    ev("M_ie", p_ie)
    for w in WS:
        q = p_s.copy(); q[is_r] = w * p_ie[is_r] + (1 - w) * p_s[is_r]
        ev(f"Bie_R_w{w:.2f}", q)
        ev(f"Bie_all_w{w:.2f}", w * p_ie + (1 - w) * p_s)
        ev(f"Bdir_all_w{w:.2f}", w * p_dir + (1 - w) * p_s)
        q2 = p_s.copy(); q2[is_r] = w * (0.5 * p_ie + 0.5 * p_dir)[is_r] + (1 - w) * p_s[is_r]
        ev(f"Bmix_R_w{w:.2f}", q2)

    cur = pd.DataFrame([r for r in rows if r["fold"] == vs])
    print(f"\n  {'variant':<18}{'BSS':>11}{'ΔBSS':>10}{'SE_row':>8}{'t_row':>8}", flush=True)
    for _, r in cur.sort_values("dbss", ascending=False).head(10).iterrows():
        print(f"  {r.variant:<18}{r.bss:>11.3f}{r.dbss:>10.3f}{r.se_row:>8.3f}"
              f"{r.t_row:>8.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "p10_full_stack.csv", index=False)

piv = res.pivot(index="variant", columns="fold", values="dbss")
print("\n" + "=" * 100)
print("★ 강한 base 대비 ΔBSS (전체 BSS)")
print("=" * 100)
print(piv.round(2).to_string())
print("\n세 fold 모두 양수:")
allpos = piv[(piv > 0).all(axis=1)]
print(allpos.sort_values(2024, ascending=False).round(2).to_string()
      if len(allpos) else "  없음")

print("\n" + "=" * 100)
print("정직한 게이트 — 2022에서만 선택")
print("=" * 100)
cand = res[(res.fold == 2022) & res.variant.str.startswith("B")]
best = cand.sort_values("dbss", ascending=False).iloc[0]
ch = res[res.variant == best.variant].set_index("fold")
print(f"2022 선택: {best.variant}")
print(ch[["strong_bss", "bss", "dbss", "se_row", "t_row", "r_bss", "f_bss"]].round(3).to_string())
g = ch.loc[2024]
print(f"\n★ Val2024 게이트   강한base {g.strong_bss:.3f} -> {g.bss:.3f}   "
      f"ΔBSS {g.dbss:+.3f}   t_row {g.t_row:+.2f}")
print(f"\nsaved -> {OUT/'p10_full_stack.csv'}")
