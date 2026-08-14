"""P11 (E1 + E3): 플래툰 스플릿 + 감쇠 외삽 격리 ablation.

P10 진단
    Val2024 게이트 실패 (2022 선택 w -> 2024 -25.9). 원인 두 가지가 특정됐다.
    (a) 내 라인이 2024 에서 강한 base 대비 95점 뒤진다. 그중 약 25점이 순수 레벨 편향
        (내 pred_mean +1.12%p vs base +0.80%p, 팀 벌점공식 401,000 x 오차^2)
    (b) [m] pred_mean 0.15566 vs 실제 0.17629 -> MIDDLE 을 2.06%p 과소예측.
        MIDDLE 은 2023 0.1533 -> 2024 0.1763 으로 +2.30%p 급등했는데 선형 외삽이 못 따라갔다.

두 가지 보완을 격리해서 잰다.

E1 플래툰 스플릿 (팀통합 1-1, 4/5명 독립 발견, 찬우 LB +32.5)
    split(p,h) = EB(투수 p 의 좌우 h 타자 상대 성공률) - EB(투수 p 전체 성공률)
    EB(x) = (성공수 + K x 리그평균) / (표본수 + K),  K in {100, 200, 300}
    주효과를 빼는 것이 핵심 — 안 빼면 asof_pitcher_success_rate 와 중복이다
    (찬우 실험: 투수 주효과 단독 기여 = 정확히 0.0).
    학습 시즌만으로 lookup 을 만들고 pitcher_id 로 조인한다. test 행 간 참조 없음.

E3 감쇠 외삽 (팀통합 2-1, 찬우 백테스트)
    linear_full : r_last + (전구간 연평균 변화량)          <- 현행
    damped_l    : r_last + lambda x (r_last - r_prev)      lambda in {0.33, 0.5, 1.0}
    last        : r_last (변화 없음)
    찬우 실측 2024 추정 — 3년 선형 .4877 / 감쇠 .4815~.4855 / 직전시즌 .5000 (실제 .4861)

arm 구성 (격리)
    A base          P10 그대로
    B +damped       외삽만 교체
    C +platoon      피처만 추가
    D +both

판정: Val2024 전체 BSS 단일. 선택 2022, 부호 확인 2023.
출력: outputs/p11_platoon_damped.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics)

OOF_DIR = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
           / "enhanced_seed_oof_parts")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 400
FOLDS = [2022, 2023, 2024]
WS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
KS = [100, 200, 300]
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
gt_all = df["game_type"].astype(str).to_numpy()
ok = df["label_ok"].to_numpy() == 1


def as_label(c):
    return np.where(ok, df[c].to_numpy(np.float64), np.nan)


LABELS = {"succ": y_all, "m": as_label("y_middle"), "r": as_label("y_reverse"),
          "o": as_label("y_outside")}
LABELS["mr"] = np.where(ok, (LABELS["m"] == 1) & (LABELS["r"] == 1), np.nan)

# ------------------------------------------------------------- 강한 base
models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
strong = {}
for fold in FOLDS:
    ids = df.loc[season == fold, "row_id"].to_numpy()
    acc, cnt = None, 0
    for mn in models:
        f = OOF_DIR / f"{mn}_fold{fold}.parquet"
        if f.exists():
            v = pd.read_parquet(f).set_index("row_id").reindex(ids)["prediction"].to_numpy()
            acc = v if acc is None else acc + v
            cnt += 1
    strong[fold] = np.clip(acc / cnt, EPS, 1 - EPS)
    print(f"강한base fold {fold}  BSS "
          f"{metrics(y_all[season == fold], strong[fold])['bss_raw']:9.3f}", flush=True)


# --------------------------------------------------------------- 외삽 규칙
def extrapolate(labels, tr_mask, vs, rule):
    m = tr_mask & ~np.isnan(labels)
    s = pd.Series(labels[m]).groupby(pd.Series(season[m])).mean().sort_index()
    r_last = float(s.iloc[-1])
    if len(s) < 2:
        return r_last
    if rule == "last":
        return r_last
    if rule == "linear_full":
        slope = (r_last - float(s.iloc[0])) / (float(s.index[-1]) - float(s.index[0]))
        return float(np.clip(r_last + slope, 0.005, 0.995))
    lam = float(rule.split("_")[1])
    return float(np.clip(r_last + lam * (r_last - float(s.iloc[-2])), 0.005, 0.995))


# ------------------------------------------------------- 플래툰 스플릿 lookup
def platoon_split(tr_mask, K):
    """split(p,h) = EB(투수 p, 타자손 h) - EB(투수 p 전체). 학습 시즌만 사용."""
    d = pd.DataFrame({"pid": pid[tr_mask], "bh": bhand[tr_mask], "y": y_all[tr_mask]})
    league = float(d["y"].mean())
    g_all = d.groupby("pid")["y"].agg(["sum", "size"])
    eb_all = (g_all["sum"] + K * league) / (g_all["size"] + K)
    g_ph = d.groupby(["pid", "bh"])["y"].agg(["sum", "size"])
    eb_ph = (g_ph["sum"] + K * league) / (g_ph["size"] + K)
    split = (eb_ph - eb_ph.index.get_level_values(0).map(eb_all)).rename("split")
    rel = (g_ph["size"] / (g_ph["size"] + K)).rename("rel")
    tab = pd.concat([split, rel], axis=1).reset_index()
    key = pd.MultiIndex.from_arrays([pid, bhand])
    t = tab.set_index(["pid", "bh"]).reindex(key)
    return (t["split"].fillna(0.0).to_numpy(np.float32),
            t["rel"].fillna(0.0).to_numpy(np.float32))


def build_features(tr_mask, platoon_K):
    priors = make_priors(df.loc[tr_mask])
    base = encode(add_stateless(df, priors))
    cols = [c for c in base.columns if c not in DROP and not c.startswith("y_")
            and c != "label_ok"]
    out = base[cols].copy()
    n = asof_n
    for c in RATES:
        pr = float(df.loc[tr_mask, c].median())
        r = np.where(np.isnan(df[c].to_numpy(np.float64)), pr, df[c].to_numpy(np.float64))
        out[f"prof200_{c}"] = (n * r + 200 * pr) / (n + 200)
    ps = {c: np.where(np.isnan(df[c].to_numpy(np.float64)),
                      float(df.loc[tr_mask, c].median()),
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
    if platoon_K is not None:
        sp, rel = platoon_split(tr_mask, platoon_K)
        out["platoon_split"] = sp
        out["platoon_rel"] = rel
        out["platoon_split_w"] = sp * rel
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


ARMS = [("A_base", "linear_full", None), ("B_damped33", "damped_0.33", None),
        ("B_damped50", "damped_0.5", None), ("B_last", "last", None),
        ("C_platoon200", "linear_full", 200),
        ("D_both33_K200", "damped_0.33", 200), ("D_both50_K200", "damped_0.5", 200),
        ("D_both50_K100", "damped_0.5", 100), ("D_both50_K300", "damped_0.5", 300)]

rows = []
for vs in FOLDS:
    tr, va = season < vs, season == vs
    y_va, gt = y_all[va], gt_all[va]
    is_r = gt == "R"
    p_s = strong[vs]
    base_m = metrics(y_va, p_s, game_type=gt)
    null = y_va.mean() * (1 - y_va.mean())
    print(f"\n{'='*102}\nfold {vs}   강한base {base_m['bss_raw']:.3f}\n{'='*102}",
          flush=True)

    Xcache = {}
    for arm, rule, K in ARMS:
        if K not in Xcache:
            Xcache[K] = build_features(tr, K)
        X, names = Xcache[K]
        comp = {}
        for tag, arr in LABELS.items():
            comp[tag] = bag(X, tr, va, arr, extrapolate(arr, tr, vs, rule))
        p_dir = comp["succ"]
        p_ie = np.clip(1 - (comp["m"] + comp["r"] - comp["mr"] + comp["o"]),
                       EPS, 1 - EPS)
        m_dir, m_ie = metrics(y_va, p_dir), metrics(y_va, p_ie)
        c_ie = float(np.corrcoef(logit(p_s), logit(p_ie))[0, 1])
        mid_bias = float(comp["m"].mean() - np.nanmean(LABELS["m"][va]))
        print(f"  [{arm:<14}] feats {len(names):>3}  direct {m_dir['bss_raw']:8.2f}  "
              f"p_ie {m_ie['bss_raw']:8.2f}  corr {c_ie:.4f}  "
              f"dir_bias {m_dir['pred_mean']-y_va.mean():+.5f}  "
              f"mid_bias {mid_bias:+.5f}", flush=True)

        for w in WS:
            q = p_s.copy()
            q[is_r] = w * p_ie[is_r] + (1 - w) * p_s[is_r]
            q = np.clip(q, EPS, 1 - EPS)
            mm = metrics(y_va, q, game_type=gt)
            d = mm["bss_raw"] - base_m["bss_raw"]
            dr = (p_s - y_va) ** 2 - (q - y_va) ** 2
            se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
            rows.append({"fold": vs, "arm": arm, "rule": rule, "K": K, "w": w,
                         "strong_bss": base_m["bss_raw"], "bss": mm["bss_raw"],
                         "dbss": d, "se_row": se,
                         "t_row": d / se if se > 0 else np.nan,
                         "direct_bss": m_dir["bss_raw"], "ie_bss": m_ie["bss_raw"],
                         "corr_strong_ie": c_ie,
                         "dir_bias": m_dir["pred_mean"] - y_va.mean(),
                         "mid_bias": mid_bias, "n_features": len(names)})

res = pd.DataFrame(rows)
res.to_csv(OUT / "p11_platoon_damped.csv", index=False)

print("\n" + "=" * 102)
print("★ arm x w 별 Val2024 ΔBSS (강한base 대비)")
print("=" * 102)
v24 = res[res.fold == 2024].pivot(index="arm", columns="w", values="dbss")
print(v24.round(2).to_string())

print("\n" + "=" * 102)
print("arm 별 재료 품질 (Val2024)")
print("=" * 102)
q = res[(res.fold == 2024) & (res.w == 0.0)][
    ["arm", "n_features", "direct_bss", "ie_bss", "corr_strong_ie",
     "dir_bias", "mid_bias"]]
print(q.round(5).to_string(index=False))

print("\n" + "=" * 102)
print("정직한 게이트 — arm/w 를 2022 에서만 고르고 Val2024 1회 적용")
print("=" * 102)
sel = res[(res.fold == 2022) & (res.w > 0)].sort_values("dbss", ascending=False).iloc[0]
ch = res[(res.arm == sel.arm) & (res.w == sel.w)].set_index("fold")
print(f"2022 선택: arm={sel.arm}  w={sel.w}")
print(ch[["strong_bss", "bss", "dbss", "se_row", "t_row"]].round(3).to_string())
g = ch.loc[2024]
print(f"\n★ Val2024   {g.strong_bss:.3f} -> {g.bss:.3f}   ΔBSS {g.dbss:+.3f}   "
      f"t_row {g.t_row:+.2f}")

print("\n세 fold 모두 양수인 (arm, w):")
p3 = res.pivot_table(index=["arm", "w"], columns="fold", values="dbss")
ap = p3[(p3 > 0).all(axis=1)]
print(ap.sort_values(2024, ascending=False).round(2).to_string() if len(ap) else "  없음")
print(f"\nsaved -> {OUT/'p11_platoon_damped.csv'}")
