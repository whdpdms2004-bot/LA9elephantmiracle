"""V60: M2 (재시도, 물리 전용 커널) — 플래툰 EB 를 리그 평균이 아니라 '커널 이웃' 쪽으로 수축시킨다.

근거 (02_EMBEDDING_METHODS.md §3)
    팀 실측에서 주입 방식이 명확히 갈린다.
        군집 ID 를 GBDT 에 직접        780~784   (기준선 784.56 미달)
        matchup 피처를 GBDT 에 직접    748.96~761.01
        임베딩 48차원을 GBDT 에 직접   395.38
        군집을 correction 의 '평활 계층'으로   806.49~815.08   <- 유일한 성공 패턴
    M2 = hard 군집을 커널 이웃(soft)으로 교체. 문서의 1순위다.

    원문 M2 는 프로덕션 correction 계층을 고치는 것이라 내 학습 루프 밖이다.
    같은 메커니즘을 성분 라인의 플래툰 테이블에 적용한다.

        현행  EB = (합 + K*리그평균) / (n + K)
        M2    EB = (합 + K*이웃평균) / (n + K)
              이웃평균 = sum_q w(p,q) * 합_q / sum_q w(p,q) * n_q
              w(p,q) = exp(-||z_p - z_q||^2 / (2 sigma^2))

    차감도 이웃평균 기준으로 한다 (V43 의 단위 조건 유지 — 두 항 모두 성공률).
        split(p,h) = EB_kernel(p,h) - 이웃평균(p,h)

왜 이 자리인가
    V30 에서 잰 것: 격차(base - 성분단독)가 저물량 구간에서 2023 에 +5481 이다.
    표본이 얇은 투수의 개인화 추정이 레짐 변화에 뒤집히는 것이 원인이다.
    리그 평균 대신 '비슷한 투수들' 쪽으로 수축시키면 그 구간이 바로 개선될 자리다.

V59 가 실패한 이유 (이 파일의 수정 근거)
    V59 는 커널 공간을 '최근 실패 구성 + 실패율'로 만들었다. 그러면 이웃이
    "결과가 비슷한 투수"로 뽑히고 이웃평균이 그 투수 자신의 수준을 담는다.
    split = EB - 이웃평균 이 0 으로 붕괴한다. 실측 744.08 -> 533.42.
    대각선을 0 으로 해 자기 자신은 뺐지만 '이웃 선택 자체'가 자기 라벨로 이뤄졌다.

        V1  정적 레벨          705.7 -> 187.5
        V35 투수x타자 개별      745.9 -> 259.1
        V37 성분 테이블 1겹     743.8 -> 507.8
        V59 outcome 커널       744.1 -> 533.4     <- 같은 실패의 네 번째

    > 평활 커널은 평활 대상과 같은 라벨로 만들면 안 된다.

커널 공간 (수정)
    TrackMan 물리 7종만 쓴다. outcome 은 한 개도 넣지 않는다.
        recent_{rel_speed, spin_rate, induced_vert_break, horz_break,
                extension, rel_height, rel_side}_mean
    TrackMan 이 없는 투수(300구 미만)는 커널 대상에서 빼고 리그 평균 수축을
    그대로 쓴다. 커버리지를 억지로 채우려다 V59 의 오류가 났다.

arm
    H0  현행 (리그 평균 수축)
    H1  커널 수축, sigma = 중앙 쌍거리 x 0.5
    H2  커널 수축, sigma = 중앙 쌍거리 x 1.0
    H3  H1 을 카운트/이닝 플래툰에도 적용

판정: fold 2024 선별 -> 단독과 ΔBSS 가 함께 오르면 fold 2023 확인.
출력: outputs/v59_kernel_shrink.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, OUT, CACHE, BASE_PARAMS, load, metrics

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
TM = MO / "trackman500_asof_train.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, F_WEIGHT, K = 400, 0.20, 300
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
VS, DECAY = 2024, 0.6
TM_COLS = [f"tm500_recent_{c}_mean" for c in
           ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
            "extension", "rel_height", "rel_side"]]
EPS = 1e-7

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
tr, va = season < VS, season == VS
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)

pid = df["pitcher_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
IS_F = df["game_type"].astype(str).to_numpy() == "F"
row_w = np.where(IS_F, F_WEIGHT, 1.0)
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])

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
FAIL = 1.0 - y_all

# ---------------------------------------------- 투수 프로파일 -> 커널
tmdf = df[["row_id"]].merge(
    pd.read_parquet(TM, columns=["row_id"] + TM_COLS), on="row_id", how="left")
TMV = tmdf[TM_COLS].to_numpy(np.float64)
last_tr = int(season[tr].max())
sw = np.where(tr, DECAY ** (last_tr - season), 0.0)

tg = pd.DataFrame(TMV[tr], columns=TM_COLS).assign(k=pid[tr]).groupby("k").mean()
have = tg.dropna(how="all")
prof = have.to_numpy(np.float64)
lo, hi = np.nanpercentile(prof, [1, 99], axis=0)
prof = np.clip(prof, lo, hi)
mu_, sd_ = np.nanmean(prof, 0), np.nanstd(prof, 0) + 1e-9
Z = np.nan_to_num((prof - mu_) / sd_)
PIDS = have.index.to_numpy()                      # TrackMan 있는 투수만
D2 = cdist(Z, Z, "sqeuclidean")
med = float(np.median(np.sqrt(D2[np.triu_indices_from(D2, 1)])))
POS = {p: i for i, p in enumerate(PIDS)}
ROW = np.array([POS.get(p, -1) for p in pid])
HAS = ROW >= 0
print(f"TrackMan 물리 커널: 투수 {len(PIDS)}명 / {len(set(pid))}명   "
      f"{Z.shape[1]}차원   중앙 쌍거리 {med:.3f}   행 커버 {HAS.mean():.1%} "
      f"(2024 {HAS[va].mean():.1%})   outcome 피처 0개", flush=True)


def kernel_tables(sigma, keys, axis=None):
    """(투수[, 축]) 셀에서 커널 이웃 수축 EB 와 이웃평균 차감값.

    이웃평균  mu_p = sum_q w(p,q) * sum_q / sum_q w(p,q) * n_q
    EB        (sum_ph + K*mu) / (n_ph + K)
    split     EB - mu           <- 두 항 모두 성공률. V43 단위 조건 충족.
    """
    W = np.exp(-D2 / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)                       # 자기 자신 제외 (누수 방지)
    sub = tr & HAS
    cols = {"p": ROW[sub], "h": bhand[sub], "y": y_all[sub]}
    if axis is not None:
        cols["a"] = axis[sub]
    dd = pd.DataFrame(cols)
    grp = ["p", "h"] + (["a"] if axis is not None else [])
    agg = dd.groupby(grp)["y"].agg(["sum", "size"]).reset_index()
    lg = float(dd["y"].mean())
    npit = len(PIDS)
    ctx = agg.groupby(grp[1:]) if len(grp) > 1 else [((), agg)]
    S_all, N_all, keyidx = {}, {}, {}
    for kk, sub in ctx:
        s = np.zeros(npit); n = np.zeros(npit)
        s[sub["p"].to_numpy()] = sub["sum"].to_numpy()
        n[sub["p"].to_numpy()] = sub["size"].to_numpy()
        S_all[kk], N_all[kk] = W @ s, W @ n
        keyidx[kk] = (s, n)
    out = np.zeros(len(df))          # 커버 밖은 0 (리그 수축과 동일 의미)
    rowkey = (list(zip(bhand, axis)) if axis is not None else list(bhand))
    kser = pd.Series(rowkey, index=df.index)
    for kk in S_all:
        mask = (kser == kk).to_numpy() if not isinstance(kk, tuple) or len(kk) > 1 \
            else (kser == kk).to_numpy()
        if mask.sum() == 0:
            continue
        mu = np.where(N_all[kk] > 0, S_all[kk] / np.maximum(N_all[kk], 1e-9), lg)
        s, n = keyidx[kk]
        eb = (s + K * mu) / (n + K)
        m2 = mask & HAS
        out[m2] = (eb - mu)[ROW[m2]]
    return out, HAS.astype(float)


def layered_league(axis):
    dd = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": axis[tr], "y": y_all[tr]})
    l0 = float(dd["y"].mean())
    g2 = dd.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = dd.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
    e2 = (g2["sum"] + K * l0) / (g2["size"] + K)
    e3 = (g3["sum"] + K * l0) / (g3["size"] + K)
    pidx = pd.MultiIndex.from_arrays([pid, bhand])
    i3 = pd.MultiIndex.from_arrays([pid, bhand, axis])
    v2 = np.where(np.isnan(e2.reindex(pidx).to_numpy()), l0, e2.reindex(pidx).to_numpy())
    v3 = np.where(np.isnan(e3.reindex(i3).to_numpy()), l0, e3.reindex(i3).to_numpy())
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    return v3 - v2, sz / (sz + K)


td = df.loc[tr]
BASE_F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
    sp, rel = layered_league(ax)
    BASE_F[f"{tag}_split"], BASE_F[f"{tag}_rel"] = sp, rel
    BASE_F[f"{tag}_w"] = sp * rel
print(f"기준 피처 {BASE_F.shape[1]}개", flush=True)


def extrap(a):
    m_ = tr & ~np.isnan(a)
    s = pd.Series(a[m_]).groupby(pd.Series(season[m_])).mean().sort_index()
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def line(X):
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr),
               **params_for(float(np.nanmean(arr[mm])))}
        Xv = X[va]
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=row_w[mm], missing=np.nan)
        p_tr = Pool(np.nan_to_num(X[mm], nan=-999.0), arr[mm], weight=row_w[mm])
        Xc = np.nan_to_num(Xv, nan=-999.0)
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(
                xgb.DMatrix(Xv, missing=np.nan))
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(Xc)[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b)["bss_raw"]
wv = BW[bucket_all[va]]

ARMS = [("H0_league", None, False), ("H1_kernel_s05", 0.5, False),
        ("H2_kernel_s10", 1.0, False), ("H3_kernel_all", 0.5, True)]
t0, rows = time.time(), []
print(f"{chr(10)}{'arm':<18}{'피처':>6}{'단독':>10}{'ΔBSS':>9}{'t_row':>8}{'경과':>8}")
for name, smul, apply_all in ARMS:
    F = BASE_F.copy()
    if smul is not None:
        sig = med * smul
        sp, hasv = kernel_tables(sig, None)
        F["k_platoon_split"] = sp
        F["k_has"] = hasv
        if apply_all:
            for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
                s2, _h = kernel_tables(sig, None, axis=ax)
                F[f"k_{tag}_split"] = s2
    p_ie = line(F.to_numpy(np.float32))
    np.save(CACHE / f"v60_{name}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
    dd_ = metrics(y_va, q)["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "n_features": F.shape[1], "solo_bss": solo,
                 "dbss": dd_, "t_row": dd_ / se})
    print(f"{name:<18}{F.shape[1]:>6}{solo:>10.2f}{dd_:>+9.2f}{dd_/se:>8.2f}"
          f"{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v60_kernel_physical.csv", index=False)
r0 = res[res.arm == "H0_league"].iloc[0]
print(f"{chr(10)}{'='*54}{chr(10)}{'arm':<18}{'ΔBSS 대비':>12}{'단독 대비':>12}")
for _, r in res.iterrows():
    print(f"{r.arm:<18}{r.dbss-r0.dbss:>+12.2f}{r.solo_bss-r0.solo_bss:>+12.2f}")
print(f"{chr(10)}둘 다 오르면 fold 2023 으로 확인한다. 팀 실측에서 '평활 계층'은")
print(f"유일하게 성공한 주입 방식이다(806~815).")
print(f"{chr(10)}saved -> {OUT/'v59_kernel_shrink.csv'}")
