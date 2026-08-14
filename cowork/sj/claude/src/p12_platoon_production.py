"""P12: 플래툰 강화 성분 라인을 실제 제출 base(836.503) 에 이식 — 최종 게이트.

P11 결과
    강한 base(25종 enhanced 앙상블, Val2024 777.675) 대비 플래툰 추가 성분 라인이
    w=0.30 에서 +19.47. 플래툰이 direct 를 +23, p_ie 를 +53 올렸고 상관은
    0.880 -> 0.847 로 오히려 더 독립적이 됐다.

이 테스트
    프로덕션 submit_021 (Val2024 836.503) 에 같은 라인을 혼합한다.
    D2 는 이 관문에서 죽었지만(강한base +16 -> 프로덕션 전 scale 음수) 성격이 다르다.
    D2 는 프로덕션이 이미 갖고 있던 정보의 재탕이었고, 플래툰은 asof_* 에 아예 없는
    좌우 스플릿이다. 프로덕션에도 투수별 플래툰 피처는 없다
    (pitcher_lookup_2025.csv 는 pitcher_id/hand/type/cohort 4컬럼뿐).

판정: Val2024 전체 BSS. 분모 SE_row. w 는 데이터로 고르지 않고 보수값 고정도 함께 본다.
출력: outputs/p12_platoon_production.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (TARGET, OUT, CACHE, BASE_PARAMS, load, make_priors,
                     add_stateless, encode, metrics)

PROD = (Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
        / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
DROP = ["row_id", TARGET]
N_ROUNDS = 400
WS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
K_PLATOON = 300
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


def as_label(c):
    return np.where(ok, df[c].to_numpy(np.float64), np.nan)


LABELS = {"succ": y_all, "m": as_label("y_middle"), "r": as_label("y_reverse"),
          "o": as_label("y_outside")}
LABELS["mr"] = np.where(ok, (LABELS["m"] == 1) & (LABELS["r"] == 1), np.nan)

tr, va = season < 2024, season == 2024
prod = pd.read_parquet(PROD).set_index("row_id").reindex(
    df.loc[va, "row_id"].to_numpy())
y_va = y_all[va]
assert prod["control_success"].to_numpy().astype(int).tolist() == y_va.astype(int).tolist()
gt = prod["game_type"].astype(str).to_numpy()
is_r = gt == "R"
p_prod = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
base_m = metrics(y_va, p_prod, game_type=gt)
null = y_va.mean() * (1 - y_va.mean())
print(f"프로덕션 submit_021 Val2024  BSS {base_m['bss_raw']:.3f}  "
      f"R {base_m['r_bss']:.3f}  F {base_m['f_bss']:.3f}  "
      f"pred_mean {base_m['pred_mean']:.5f}", flush=True)


def platoon_split(tr_mask, K):
    d = pd.DataFrame({"pid": pid[tr_mask], "bh": bhand[tr_mask], "y": y_all[tr_mask]})
    league = float(d["y"].mean())
    g_all = d.groupby("pid")["y"].agg(["sum", "size"])
    eb_all = (g_all["sum"] + K * league) / (g_all["size"] + K)
    g_ph = d.groupby(["pid", "bh"])["y"].agg(["sum", "size"])
    eb_ph = (g_ph["sum"] + K * league) / (g_ph["size"] + K)
    split = (eb_ph - eb_ph.index.get_level_values(0).map(eb_all)).rename("split")
    rel = (g_ph["size"] / (g_ph["size"] + K)).rename("rel")
    t = pd.concat([split, rel], axis=1).reindex(
        pd.MultiIndex.from_arrays([pid, bhand]))
    return (t["split"].fillna(0.0).to_numpy(np.float32),
            t["rel"].fillna(0.0).to_numpy(np.float32))


def build_features(tr_mask, K):
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
    if K is not None:
        sp, rel = platoon_split(tr_mask, K)
        out["platoon_split"] = sp
        out["platoon_rel"] = rel
        out["platoon_split_w"] = sp * rel
    return out.to_numpy(np.float32), list(out.columns)


def extrap(labels, tr_mask, vs):
    m = tr_mask & ~np.isnan(labels)
    s = pd.Series(labels[m]).groupby(pd.Series(season[m])).mean().sort_index()
    r_last = float(s.iloc[-1])
    slope = (r_last - float(s.iloc[0])) / (float(s.index[-1]) - float(s.index[0]))
    return float(np.clip(r_last + slope, 0.005, 0.995))


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


rows = []
for tag_arm, K in [("no_platoon", None), ("platoon_K300", K_PLATOON)]:
    X, names = build_features(tr, K)
    comp = {t: bag(X, tr, va, a, extrap(a, tr, 2024)) for t, a in LABELS.items()}
    p_dir = comp["succ"]
    p_ie = np.clip(1 - (comp["m"] + comp["r"] - comp["mr"] + comp["o"]), EPS, 1 - EPS)
    m_dir, m_ie = metrics(y_va, p_dir), metrics(y_va, p_ie)
    c_dir = float(np.corrcoef(logit(p_prod), logit(p_dir))[0, 1])
    c_ie = float(np.corrcoef(logit(p_prod), logit(p_ie))[0, 1])
    print(f"\n[{tag_arm}] feats {len(names)}  direct {m_dir['bss_raw']:8.2f}  "
          f"p_ie {m_ie['bss_raw']:8.2f}  corr(prod,dir) {c_dir:.4f}  "
          f"corr(prod,ie) {c_ie:.4f}", flush=True)

    def ev(name, p):
        p = np.clip(p, EPS, 1 - EPS)
        mm = metrics(y_va, p, game_type=gt)
        d = mm["bss_raw"] - base_m["bss_raw"]
        dr = (p_prod - y_va) ** 2 - (p - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"arm": tag_arm, "variant": name, "bss": mm["bss_raw"], "dbss": d,
                     "se_row": se, "t_row": d / se if se > 0 else np.nan,
                     "r_bss": mm["r_bss"], "dr": mm["r_bss"] - base_m["r_bss"],
                     "f_bss": mm["f_bss"], "df": mm["f_bss"] - base_m["f_bss"],
                     "pred_mean": mm["pred_mean"], "direct_bss": m_dir["bss_raw"],
                     "ie_bss": m_ie["bss_raw"], "corr_prod_ie": c_ie,
                     "corr_prod_dir": c_dir})

    for w in WS:
        q = p_prod.copy(); q[is_r] = w * p_ie[is_r] + (1 - w) * p_prod[is_r]
        ev(f"ieR_w{w:.2f}", q)
        ev(f"ieAll_w{w:.2f}", w * p_ie + (1 - w) * p_prod)
        ev(f"dirAll_w{w:.2f}", w * p_dir + (1 - w) * p_prod)
        q2 = p_prod.copy()
        q2[is_r] = w * (0.5 * p_ie + 0.5 * p_dir)[is_r] + (1 - w) * p_prod[is_r]
        ev(f"mixR_w{w:.2f}", q2)

res = pd.DataFrame(rows)
res.to_csv(OUT / "p12_platoon_production.csv", index=False)

print("\n" + "=" * 96)
print(f"★ 프로덕션 submit_021 (Val2024 {base_m['bss_raw']:.3f}) 대비 ΔBSS")
print("=" * 96)
for arm in ["no_platoon", "platoon_K300"]:
    sub = res[res.arm == arm]
    piv = sub.assign(fam=sub.variant.str.split("_w").str[0],
                     w=sub.variant.str.split("_w").str[1].astype(float)
                     ).pivot(index="fam", columns="w", values="dbss")
    print(f"\n[{arm}]")
    print(piv.round(2).to_string())

print("\n" + "=" * 96)
print("상위 12 (전체 BSS 기준)")
print("=" * 96)
top = res.sort_values("dbss", ascending=False).head(12)
print(f"{'arm':<14}{'variant':<16}{'BSS':>11}{'ΔBSS':>9}{'SE_row':>8}{'t_row':>8}"
      f"{'ΔR':>9}{'ΔF':>9}")
for _, r in top.iterrows():
    print(f"{r.arm:<14}{r.variant:<16}{r.bss:>11.3f}{r.dbss:>9.3f}{r.se_row:>8.3f}"
          f"{r.t_row:>8.2f}{r.dr:>9.2f}{r.df:>9.2f}")
print(f"\nsaved -> {OUT/'p12_platoon_production.csv'}")
