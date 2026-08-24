"""V9: 성분 모델에 프로덕션 예측을 피처로 넣는다 (스태킹).

문제 진단
    프로덕션 base 836.5 / 내 성분 라인 750.5 / 결합 853.8
    86점 뒤진 파트너로 +17 을 얻고 있다. 파트너가 강해지면 가중치가 올라가고
    이득도 커진다. 성분 라인이 약한 이유는 97피처만 쓰기 때문이다 (프로덕션은
    211피처 + TrackMan + insight prior + 클러스터 + residual 적층).

방법
    피처 파이프라인 전체 재구축 대신, 프로덕션 계열의 OOF 예측을 성분 모델의
    피처로 넣는다. 성분 모델이 그 지식에서 출발해 분해 구조만 추가로 학습한다.

    enhanced_seed_oof_parts/ 에 25종 x fold{2022,2023,2024} 순방향 OOF 가 있다.
    fold2022 = 2019~2021 학습 -> 2022 예측, 이런 식이라 누수가 없다.

대가와 격리
    OOF 가 2022/2023 에만 있으므로 스태킹 학습 행이 492,997 로 준다
    (현행은 2019~2023 전체 1,221,585). 행 감소 자체의 손해를 분리해야 한다.

        A_full_noStack    2019~2023, 스태킹 없음        <- 현행 (submit_025 구성)
        B_2223_noStack    2022~2023, 스태킹 없음        <- 행 감소만의 효과
        C_2223_stack      2022~2023, base 예측 피처 포함 <- 스태킹 순효과 = C - B
        D_2223_stackLogit 2022~2023, base 로짓 + 성분별 base_margin

    D 는 base 예측을 피처가 아니라 base_margin 으로 준다. 성분 k 의 base_margin 은
    프로덕션 로짓을 그대로 쓸 수 없으므로 (타깃이 다르다) 성분 기저율로 스케일한다.

판정: Val2024 전체 BSS, 프로덕션 836.503 대비, 고정 w=0.20.
출력: outputs/v9_stacked_components.csv
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics)

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 400
WS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
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


df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
asof_n = df["asof_pitcher_n"].to_numpy(np.float64)
pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
ok = df["label_ok"].to_numpy() == 1
tr, va = season < 2024, season == 2024

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}

# ------------------------------------------------- 프로덕션 계열 OOF (25종 평균)
print("enhanced OOF 로드 (25종 x fold2022/2023/2024)", flush=True)
models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
base_oof = np.full(len(df), np.nan)
for fold in (2022, 2023, 2024):
    ids = df.loc[season == fold, "row_id"].to_numpy()
    acc, cnt = None, 0
    for mn in models:
        f = OOF_DIR / f"{mn}_fold{fold}.parquet"
        if f.exists():
            v = pd.read_parquet(f).set_index("row_id").reindex(ids)["prediction"].to_numpy()
            acc = v if acc is None else acc + v
            cnt += 1
    base_oof[season == fold] = acc / cnt
    print(f"  fold {fold}  {cnt}종  BSS "
          f"{metrics(y_all[season==fold], base_oof[season==fold])['bss_raw']:8.2f}",
          flush=True)
has_oof = ~np.isnan(base_oof)
tr_stack = tr & has_oof
print(f"  스태킹 학습 가능 행 {int(tr_stack.sum()):,} / 전체 학습 {int(tr.sum()):,}",
      flush=True)


def eb_split(target, K=300):
    m = tr & ~np.isnan(target)
    d = pd.DataFrame({"p": pid[m], "h": bhand[m], "y": target[m]})
    lg = float(d["y"].mean())
    ga = d.groupby("p")["y"].agg(["sum", "size"])
    gh = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    eb_p = (ga["sum"] + K * lg) / (ga["size"] + K)
    eb_ph = (gh["sum"] + K * lg) / (gh["size"] + K)
    key = pd.MultiIndex.from_arrays([pid, bhand])
    lv = np.where(np.isnan(eb_ph.reindex(key).to_numpy()), lg,
                  eb_ph.reindex(key).to_numpy())
    pv = np.where(np.isnan(pd.Series(pid).map(eb_p).to_numpy()), lg,
                  pd.Series(pid).map(eb_p).to_numpy())
    sz = gh["size"].reindex(key).fillna(0.0).to_numpy()
    return lv - pv, sz / (sz + K)


def build(with_stack):
    priors = make_priors(df.loc[tr])
    base = encode(add_stateless(df, priors))
    cols = [c for c in base.columns if c not in DROP and not c.startswith("y_")
            and c != "label_ok"]
    out = base[cols].copy()
    n = asof_n
    for c in RATES:
        pr = float(df.loc[tr, c].median())
        r = np.where(np.isnan(df[c].to_numpy(np.float64)), pr, df[c].to_numpy(np.float64))
        out[f"prof200_{c}"] = (n * r + 200 * pr) / (n + 200)
    ps = {c: np.where(np.isnan(df[c].to_numpy(np.float64)),
                      float(df.loc[tr, c].median()),
                      df[c].to_numpy(np.float64)) for c in PREV_S + PREV_M}
    out["prev_trend_s"] = ps[PREV_S[0]] - ps[PREV_S[2]]
    out["prev_trend_m"] = ps[PREV_M[0]] - ps[PREV_M[2]]
    out["prev_std_s"] = np.std(np.vstack([ps[c] for c in PREV_S]), axis=0)
    out["prev_std_m"] = np.std(np.vstack([ps[c] for c in PREV_M]), axis=0)
    out["prev_miss_cnt"] = sum(np.isnan(df[c].to_numpy(np.float64)).astype(np.float64)
                               for c in PREV_S + PREV_M)
    for k, (cs, cm) in enumerate(zip(PREV_S, PREV_M)):
        out[f"faildir_{k}"] = ps[cm] - (1 - ps[cs])
    out["rel200"] = n / (n + 200.0)
    sp, rel = eb_split(y_all)
    out["platoon_split"], out["platoon_split_rel"] = sp, rel
    out["platoon_split_w"] = sp * rel
    if with_stack:
        out["base_pred"] = np.where(has_oof, base_oof, np.nan)
        out["base_logit"] = np.where(has_oof, logit(np.where(has_oof, base_oof, 0.5)),
                                     np.nan)
    return out.to_numpy(np.float32), list(out.columns)


def extrap(a, mask):
    m = mask & ~np.isnan(a)
    s = pd.Series(a[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    if len(s) < 2:
        return last
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def fit(X, arr, fit_mask, margin=None):
    m = fit_mask & ~np.isnan(arr)
    prm = {**BASE_PARAMS, "base_score": extrap(arr, fit_mask),
           **params_for(float(np.nanmean(arr[fit_mask])))}
    d_tr = xgb.DMatrix(X[m], label=arr[m])
    d_va = xgb.DMatrix(X[va])
    if margin is not None:
        d_tr.set_base_margin(margin[m]); d_va.set_base_margin(margin[va])
    acc = np.zeros(int(va.sum()))
    for s in SEEDS:
        b = xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                      verbose_eval=False)
        acc += (b.predict(d_va) if margin is None
                else 1.0 / (1.0 + np.exp(-b.predict(d_va, output_margin=True))))
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
bm = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())

X_plain, c_plain = build(False)
X_stack, c_stack = build(True)
print(f"피처 plain {len(c_plain)}  stack {len(c_stack)}", flush=True)

ARMS = [("A_full_noStack", X_plain, tr, False),
        ("B_2223_noStack", X_plain, tr_stack, False),
        ("C_2223_stack", X_stack, tr_stack, False),
        ("D_2223_stackMargin", X_stack, tr_stack, True)]

t0, rows = time.time(), []
print(f"\n{'arm':<20}{'학습행':>10}{'단독BSS':>10}{'corr':>8}   "
      + "".join(f"w{w:<6.2f}" for w in WS), flush=True)
for name, X, fmask, use_margin in ARMS:
    p = {}
    for tag, arr in LAB.items():
        mg = None
        if use_margin:
            rate = float(np.nanmean(arr[fmask]))
            # 프로덕션 로짓을 성분 기저율 수준으로 평행이동해 base_margin 으로 쓴다
            shift = logit(rate) - np.nanmean(logit(np.where(has_oof, base_oof, 0.5)))
            mg = np.where(has_oof, logit(np.where(has_oof, base_oof, 0.5)) + shift,
                          logit(rate))
            if tag in ("m", "r", "mr", "ob", "oz"):
                mg = -mg + 2 * logit(rate)      # 실패 성분은 성공 로짓과 방향이 반대
        p[tag] = fit(X, arr, fmask, mg)
    p_ie = np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(logit(p_prod), logit(p_ie))[0, 1])
    line = f"{name:<20}{int(fmask.sum()):>10,}{solo:>10.2f}{corr:>8.4f}   "
    for w in WS:
        q = p_prod.copy()
        q[is_r] = w * p_ie[is_r] + (1 - w) * p_prod[is_r]
        q = np.clip(q, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - bm["bss_raw"]
        dr = (p_prod - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"arm": name, "n_fit": int(fmask.sum()), "solo_bss": solo,
                     "corr": corr, "w": w, "bss": mm["bss_raw"], "dbss": d,
                     "se_row": se, "t_row": d / se, "pred_mean": mm["pred_mean"]})
        line += f"{d:+7.2f}"
    print(line + f"   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v9_stacked_components.csv", index=False)
g = res[res.w == 0.20].set_index("arm")["dbss"]
print(f"\n고정 w=0.20 분해")
print(f"  A 현행(2019~2023, 스태킹 없음)   {g['A_full_noStack']:+.3f}")
print(f"  B 행 감소만 (2022~2023)          {g['B_2223_noStack']:+.3f}  "
      f"(행 감소 손해 {g['B_2223_noStack']-g['A_full_noStack']:+.3f})")
print(f"  C 스태킹 피처                    {g['C_2223_stack']:+.3f}  "
      f"(스태킹 순효과 {g['C_2223_stack']-g['B_2223_noStack']:+.3f})")
print(f"  D 스태킹 base_margin             {g['D_2223_stackMargin']:+.3f}  "
      f"(순효과 {g['D_2223_stackMargin']-g['B_2223_noStack']:+.3f})")
best = res[res.w == 0.20].sort_values("dbss", ascending=False).iloc[0]
print(f"\n최고 {best.arm} {best.dbss:+.3f}  t_row {best.t_row:+.2f}  "
      f"(현행 대비 {best.dbss-g['A_full_noStack']:+.3f})")
print(f"\nsaved -> {OUT/'v9_stacked_components.csv'}")
