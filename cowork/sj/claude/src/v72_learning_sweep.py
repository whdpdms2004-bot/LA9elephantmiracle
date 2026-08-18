"""V72: 학습 방식 대규모 스윕 — 19_BAYESIAN_COMPONENT_PLAN.md 기반.

지시: "계획을 기반으로 학습방식을 개선할 방향 다방면으로 실험 후 기록,
       다양하게 시도할 수 있는 스크립트, 10시간 걸려도 돼"

설계 원칙
    - arm 하나 = (두 fold) x (5성분 x 8시드 x XGB+CatBoost)  ~ 560초
    - 결과를 arm 마다 CSV 에 즉시 append 한다. 중단돼도 결과가 남는다.
    - 재시작하면 CSV 에 있는 arm 을 건너뛴다. 10시간을 나눠 돌려도 된다.
    - 판정은 두 fold 모두에서 단독과 ΔBSS 가 함께 올라야 한다.
      단일 fold 선별은 네 번 뒤집혔다(V23, V38, V47, V64).

스윕 축 (계획서 §3, §5 순위 순)

  A  EB 수축 상수 K            <- 계획 1순위
     현행은 모든 테이블·모든 성분에 K=300 이다.
     이론상 K* = sigma2_within / sigma2_between 이고 성분마다 다르다.
         m 137   r 83   ob 244   oz 304   (계획서 §3-3 실측 도출)
     현행 300 은 oz 에는 맞고 r 에는 3.6배 과하다.

  B  시즌 외삽기               <- 계획 2순위
     현행은 첫 시즌과 마지막 시즌을 잇는 직선이다.
     r|fail 은 2021 에 정점을 찍고 내려온다 — 단조가 아니라 과대 추정이 난다.
     직전유지 / 선형 / 최근2차분 / 최근3선형 / 지수가중 을 비교한다.

  C  시즌 recency 가중          <- 계획에 없던 축, 드리프트가 근거
     성분 기저율이 19->24 로 m +41.0%, ob -23.6% 로 크게 움직인다.
     오래된 시즌을 같은 가중으로 학습하는 것이 최선일 이유가 없다.

  D  목적함수
     현행 logloss. 대회 지표는 Brier 다. 직접 최적화가 나을 수 있다.

  E  성분별 트리 용량
     현행 params_for(rate) 3단계. 성분별로 따로 준다.

  F  샘플 가중
     현행 F행 0.20. V64 에서 짧은 등판 0.5 가 두 fold 양수였다(미탑재).

  G  카운트 조건부 강화 (ob, oz)
     카운트 CV 가 ob 0.29, oz 0.62 로 크다(계획서 §2-3).

사용법
    python v72_learning_sweep.py              전체 실행 (재개 가능)
    python v72_learning_sweep.py --only A,B   특정 축만
    python v72_learning_sweep.py --list       arm 목록만 출력

출력: outputs/v72_learning_sweep.csv  (arm 마다 즉시 append)
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, OUT, CACHE, BASE_PARAMS, load, metrics

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
RESULT = OUT / "v72_learning_sweep.csv"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
FOLDS = [2023, 2024]
K_STAR = {"m": 137, "r": 83, "mr": 300, "ob": 244, "oz": 304}   # 계획서 §3-3
EPS = 1e-7

ONLY = None
for i, a in enumerate(sys.argv):
    if a == "--only" and i + 1 < len(sys.argv):
        ONLY = set(sys.argv[i + 1].split(","))
LIST_ONLY = "--list" in sys.argv

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
IS_F = df["game_type"].astype(str).to_numpy() == "F"
NVOL = df["asof_pitcher_n"].to_numpy()
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt3 = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])

o_ = np.argsort(pid.astype(np.int64) * 10_000_000 + NVOL, kind="stable")
pv = df["asof_pitcher_prev1_game_success_rate"].to_numpy()[o_]
gp = pid[o_]
chg = np.r_[True, (gp[1:] != gp[:-1]) | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
outing = np.empty(len(df), dtype=np.int64)
outing[o_] = np.cumsum(chg) - 1
ag = pd.DataFrame({"outing": outing, "pid": pid,
                   "inn": df["inning"].to_numpy()}).groupby("outing").agg(
    n=("outing", "size"), pid=("pid", "first"), first_inn=("inn", "min"))
ag["start"] = (ag["first_inn"] == 1).astype(int)
ag = ag.join(ag.groupby(["pid", "start"])["n"].median().rename("med"),
             on=["pid", "start"])
SHORT = np.nan_to_num((ag["n"] / ag["med"].clip(lower=1)).reindex(outing).to_numpy(),
                      nan=1.0) < 0.5

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)


def AND(*a):
    m = np.ones(len(df), bool)
    for x in a:
        m &= (x == 1)
    return np.where(ok, m.astype(float), np.nan)


LAB = {"m": ym, "r": yr, "mr": AND(ym, yr), "ob": AND(yo, yb), "oz": AND(yo, 1 - yb)}

models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
BASE_P = {}
for f in FOLDS:
    fid = df.loc[season == f, "row_id"].to_numpy()
    if f == 2024:
        pr = pd.read_parquet(PROD).set_index("row_id").reindex(fid)
        BASE_P[f] = np.clip(pr["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                            EPS, 1 - EPS)
    else:
        acc, c = None, 0
        for mn in models:
            p = OOF_DIR / f"{mn}_fold{f}.parquet"
            if p.exists():
                v = pd.read_parquet(p).set_index("row_id").reindex(fid)["prediction"].to_numpy()
                acc = v if acc is None else acc + v
                c += 1
        BASE_P[f] = np.clip(acc / c, EPS, 1 - EPS)


# ---------------------------------------------------------------- 시즌 외삽기
def extrapolate(vals, seasons, target, how):
    """vals/seasons: 학습 시즌별 평균. target: 예측할 시즌."""
    v = np.asarray(vals, float)
    s = np.asarray(seasons, float)
    if how == "last" or len(v) < 2:
        r = v[-1]
    elif how == "linear_ends":                       # 현행
        r = v[-1] + (v[-1] - v[0]) / (s[-1] - s[0]) * (target - s[-1])
    elif how == "diff2":
        r = v[-1] + (v[-1] - v[-2]) * (target - s[-1])
    elif how == "ols3":
        k = min(3, len(v))
        r = np.polyval(np.polyfit(s[-k:], v[-k:], 1), target)
    elif how == "ewm":                               # 지수가중 국소 선형
        w = 0.6 ** (s[-1] - s)
        b1 = np.sum(w * (s - np.average(s, weights=w))
                    * (v - np.average(v, weights=w))) / \
             max(np.sum(w * (s - np.average(s, weights=w)) ** 2), 1e-9)
        r = np.average(v, weights=w) + b1 * (target - np.average(s, weights=w))
    else:
        raise ValueError(how)
    return float(np.clip(r, 0.005, 0.995))


def base_score(tag, tr, how):
    a = LAB[tag]
    m_ = tr & ~np.isnan(a)
    g = pd.Series(a[m_]).groupby(pd.Series(season[m_])).mean().sort_index()
    return extrapolate(g.to_numpy(), g.index.to_numpy(), season[~tr].min()
                       if (~tr).any() else g.index[-1] + 1, how)


# ---------------------------------------------------------------- 피처 조립
def layered(tr, axis, kk):
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": axis[tr], "y": y_all[tr]})
    l0 = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
    e2 = (g2["sum"] + kk * l0) / (g2["size"] + kk)
    e3 = (g3["sum"] + kk * l0) / (g3["size"] + kk)
    pidx = pd.MultiIndex.from_arrays([pid, bhand])
    i3 = pd.MultiIndex.from_arrays([pid, bhand, axis])
    v2 = np.where(np.isnan(e2.reindex(pidx).to_numpy()), l0, e2.reindex(pidx).to_numpy())
    v3 = np.where(np.isnan(e3.reindex(i3).to_numpy()), l0, e3.reindex(i3).to_numpy())
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + kk)


def make_features(fold, cfg):
    tr = season < fold
    td = df.loc[tr]
    ks = cfg.get("k_success", 300)
    kc = cfg.get("k_comp", {t: 300 for t in COMPONENTS})
    F = CF.build(df[INPUT_COLS], CF.make_spec(td),
                 CF.make_platoon_table(td, K=ks),
                 CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()},
                                              K=int(np.mean(list(kc.values())))))
    for tag, ax in [("cnt", cnt3), ("inn", inn_b)]:
        sp, rel = layered(tr, ax, ks)
        F[f"{tag}_split"], F[f"{tag}_rel"] = sp, rel
        F[f"{tag}_w"] = sp * rel
    if cfg.get("count_boost"):
        for nm, ax in [("cb6", np.where(strikes > balls, np.where(strikes == 2, 0, 1),
                                        np.where(balls == 3, 2, 3)))]:
            sp, rel = layered(tr, ax, ks)
            F[f"{nm}_split"], F[f"{nm}_rel"] = sp, rel
            F[f"{nm}_w"] = sp * rel
    return F.to_numpy(np.float32)


def sample_weight(tr, cfg):
    w = np.where(IS_F, cfg.get("f_weight", 0.20), 1.0)
    if cfg.get("short_weight") is not None:
        w = w * np.where(SHORT, cfg["short_weight"], 1.0)
    dec = cfg.get("recency", 1.0)
    if dec != 1.0:
        last = int(season[tr].max())
        w = w * (dec ** (last - season))
    return w


def params_for(tag, rate, cfg):
    per = cfg.get("tree_per_comp")
    if per and tag in per:
        return dict(per[tag])
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def run_line(X, fold, cfg):
    tr, va = season < fold, season == fold
    W = sample_weight(tr, cfg)
    rounds = cfg.get("rounds", 400)
    obj = cfg.get("objective", "logloss")
    cat_loss = cfg.get("cat_loss", "Logloss")
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS,
               "base_score": base_score(tag, tr, cfg.get("extrap", "linear_ends")),
               **params_for(tag, float(np.nanmean(arr[mm])), cfg)}
        if obj == "brier":
            prm["objective"] = "reg:squarederror"
            prm.pop("eval_metric", None)
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=W[mm])
        d_va = xgb.DMatrix(X[va])
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=rounds,
                                   verbose_eval=False).predict(d_va)
        p_tr = Pool(X[mm], arr[mm], weight=W[mm])
        for s in SEEDS:
            c = CatBoostClassifier(iterations=rounds, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function=cat_loss,
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(X[va])[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


# ---------------------------------------------------------------- arm 등록
ARMS = []


def add(axis, name, **cfg):
    ARMS.append({"axis": axis, "name": name, "cfg": cfg})


add("BASE", "A00_current")

# A. EB 수축 상수
for k in [100, 150, 200, 500, 800]:
    add("A", f"A_ksucc{k}", k_success=k)
add("A", "A_kstar_comp", k_comp=dict(K_STAR))
add("A", "A_kstar_half", k_comp={t: max(20, v // 2) for t, v in K_STAR.items()})
add("A", "A_kstar_double", k_comp={t: v * 2 for t, v in K_STAR.items()})
add("A", "A_kstar_both", k_success=150, k_comp=dict(K_STAR))
add("A", "A_ksucc150_kstar_half", k_success=150,
    k_comp={t: max(20, v // 2) for t, v in K_STAR.items()})

# B. 시즌 외삽기
for how in ["last", "diff2", "ols3", "ewm"]:
    add("B", f"B_extrap_{how}", extrap=how)

# C. 시즌 recency 가중
for dec in [0.95, 0.9, 0.85, 0.8, 0.7]:
    add("C", f"C_recency{dec}", recency=dec)

# D. 목적함수
add("D", "D_brier_xgb", objective="brier")
add("D", "D_cat_crossentropy", cat_loss="CrossEntropy")
add("D", "D_brier_both", objective="brier", cat_loss="CrossEntropy")

# E. 성분별 트리 용량
add("E", "E_leaves_up", tree_per_comp={
    "m": {"max_leaves": 18, "min_child_weight": 96, "reg_lambda": 4.0},
    "r": {"max_leaves": 24, "min_child_weight": 64, "reg_lambda": 2.0},
    "mr": {"max_leaves": 10, "min_child_weight": 256, "reg_lambda": 8.0},
    "ob": {"max_leaves": 16, "min_child_weight": 128, "reg_lambda": 4.0},
    "oz": {"max_leaves": 10, "min_child_weight": 192, "reg_lambda": 8.0}})
add("E", "E_leaves_down", tree_per_comp={
    "m": {"max_leaves": 8, "min_child_weight": 192, "reg_lambda": 6.0},
    "r": {"max_leaves": 12, "min_child_weight": 96, "reg_lambda": 4.0},
    "mr": {"max_leaves": 6, "min_child_weight": 320, "reg_lambda": 10.0},
    "ob": {"max_leaves": 8, "min_child_weight": 192, "reg_lambda": 6.0},
    "oz": {"max_leaves": 6, "min_child_weight": 320, "reg_lambda": 10.0}})
for r in [250, 600, 800]:
    add("E", f"E_rounds{r}", rounds=r)

# F. 샘플 가중
for sw in [0.3, 0.5, 0.7]:
    add("F", f"F_short{sw}", short_weight=sw)
for fw in [0.10, 0.35, 0.50]:
    add("F", f"F_fweight{fw}", f_weight=fw)

# G. 카운트 조건부 강화
add("G", "G_count_boost", count_boost=True)

# H. 상위 축 조합 (앞 결과와 무관하게 미리 등록 — 조합 자체가 가설이다)
add("H", "H_kstar_recency90", k_comp=dict(K_STAR), recency=0.9)
add("H", "H_kstar_short05", k_comp=dict(K_STAR), short_weight=0.5)
add("H", "H_recency90_short05", recency=0.9, short_weight=0.5)
add("H", "H_kstar_ewm_short05", k_comp=dict(K_STAR), extrap="ewm", short_weight=0.5)
add("H", "H_all", k_success=150, k_comp=dict(K_STAR), extrap="ewm",
    recency=0.9, short_weight=0.5, count_boost=True)

if ONLY:
    ARMS = [a for a in ARMS if a["axis"] in ONLY or a["axis"] == "BASE"]
if LIST_ONLY:
    for a in ARMS:
        print(f"  [{a['axis']}] {a['name']}   {a['cfg']}")
    print(f"{chr(10)}총 {len(ARMS)} arm   예상 {len(ARMS)*560/3600:.1f}시간")
    sys.exit(0)

done = set()
if RESULT.exists():
    done = set(pd.read_csv(RESULT)["name"])
    print(f"기존 결과 {len(done)}개 arm — 건너뛴다", flush=True)
todo = [a for a in ARMS if a["name"] not in done]
print(f"전체 {len(ARMS)} arm, 남은 {len(todo)}개   "
      f"예상 {len(todo)*560/3600:.1f}시간", flush=True)

FEAT_CACHE = {}


def features_cached(fold, cfg):
    key = (fold, cfg.get("k_success", 300),
           tuple(sorted(cfg.get("k_comp", {}).items())), bool(cfg.get("count_boost")))
    if key not in FEAT_CACHE:
        FEAT_CACHE.clear()
        FEAT_CACHE[key] = make_features(fold, cfg)
    return FEAT_CACHE[key]


t0 = time.time()
lgf = lambda z: np.log(z / (1 - z))
for ai, a in enumerate(todo, 1):
    rec = {"name": a["name"], "axis": a["axis"], "cfg": str(a["cfg"])}
    okrun = True
    for fold in FOLDS:
        try:
            X = features_cached(fold, a["cfg"])
            p_ie = run_line(X, fold, a["cfg"])
        except Exception as e:                       # arm 하나가 죽어도 스윕은 계속
            print(f"  !! {a['name']} fold {fold} 실패: {type(e).__name__} {e}",
                  flush=True)
            okrun = False
            break
        va = season == fold
        y, b = y_all[va], BASE_P[fold]
        null = y.mean() * (1 - y.mean())
        ref = metrics(y, b)["bss_raw"]
        wv = BW[bucket_all[va]]
        q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
        d = metrics(y, q)["bss_raw"] - ref
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rec[f"solo_{fold}"] = metrics(y, p_ie)["bss_raw"]
        rec[f"dbss_{fold}"] = d
        rec[f"t_{fold}"] = d / se
        rec[f"corr_{fold}"] = float(np.corrcoef(lgf(b), lgf(p_ie))[0, 1])
        np.save(CACHE / f"v72_{a['name']}_{fold}.npy", p_ie)
    if not okrun:
        continue
    pd.DataFrame([rec]).to_csv(RESULT, mode="a", header=not RESULT.exists(),
                               index=False)
    el = time.time() - t0
    print(f"[{ai:>2}/{len(todo)}] {a['name']:<26} "
          f"단독 {rec['solo_2023']:>9.2f} / {rec['solo_2024']:>8.2f}   "
          f"ΔBSS {rec['dbss_2023']:>+7.2f} / {rec['dbss_2024']:>+7.2f}   "
          f"[{el/60:.0f}분, 남은 {(len(todo)-ai)*el/ai/60:.0f}분]", flush=True)

res = pd.read_csv(RESULT)
b0 = res[res.name == "A00_current"]
if len(b0):
    b0 = b0.iloc[0]
    for f in FOLDS:
        res[f"d_vs_base_{f}"] = res[f"dbss_{f}"] - b0[f"dbss_{f}"]
        res[f"s_vs_base_{f}"] = res[f"solo_{f}"] - b0[f"solo_{f}"]
    res["worst_d"] = res[[f"d_vs_base_{f}" for f in FOLDS]].min(axis=1)
    res["worst_s"] = res[[f"s_vs_base_{f}" for f in FOLDS]].min(axis=1)
    print(f"{chr(10)}{'='*92}{chr(10)}기준선 대비 — 두 fold 모두 양수인 것만 "
          f"채택 대상{chr(10)}{'='*92}")
    print(f"{'arm':<28}{'축':>4}{'Δ2023':>9}{'Δ2024':>9}{'단독2023':>10}"
          f"{'단독2024':>10}{'최악Δ':>8}")
    for _, r in res.sort_values("worst_d", ascending=False).head(20).iterrows():
        print(f"{r['name']:<28}{r['axis']:>4}{r['d_vs_base_2023']:>+9.2f}"
              f"{r['d_vs_base_2024']:>+9.2f}{r['s_vs_base_2023']:>+10.2f}"
              f"{r['s_vs_base_2024']:>+10.2f}{r['worst_d']:>+8.2f}")
    good = res[(res.worst_d > 0) & (res.worst_s > 0)]
    print(f"{chr(10)}두 지표 두 fold 전부 양수: {len(good)}개")
    for _, r in good.sort_values("worst_d", ascending=False).iterrows():
        print(f"  {r['name']:<28}최악Δ {r['worst_d']:+.2f}  최악단독 {r['worst_s']:+.2f}")
print(f"{chr(10)}saved -> {RESULT}")
