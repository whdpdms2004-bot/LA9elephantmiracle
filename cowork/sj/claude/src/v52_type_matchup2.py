"""V52: 투수 5유형 x 타자 4유형 상성. 최근 실패 구성 + TrackMan 으로 유형화.

V51 첫 시도가 틀린 점
    손(hand) 안에서 따로 군집해 투수 7종 x 타자 6종 = 42셀이 됐고 셀 중앙값이
    8~11k 행에 그쳤다. 유형화의 목적은 셀을 키우는 것인데 오히려 잘게 쪼갰다.
    그리고 유형을 '구종 배합'으로 잡았는데 이 대회의 표적은 제구 실패다.

이번 설계
    투수 5유형, 타자 4유형, 손 구분 없음 -> 20셀. 중앙 7만 행대를 노린다.
    (손은 이미 플래툰 피처로 따로 들어가 있어 유형 축에서 다시 쪼갤 이유가 없다.)

    투수 프로파일
        최근 실패 유형 구성   m/r/mr/ob/oz 를 최근 시즌 가중(0.6^거리)으로 집계하고
                              실패 대비 구성비로 정규화 + 전체 실패율
        TrackMan 물리         recent_{rel_speed, spin_rate, induced_vert_break,
                              horz_break, extension, rel_height, rel_side}_mean
                              strict as-of (시즌 S 행은 season < S 증거만)
        물량                  log n

    타자 프로파일
        최근 '당한' 실패 구성 + 성공률 + 물량

    학습 300구 미만은 군집에서 빼고 '저물량' 유형으로 모은다(V51 에서 저물량
    잡음이 군집을 퇴화시키는 걸 봤다). TrackMan 결측은 표준화 후 0 으로 채운다.

상호작용 (2겹 차감 — V19 원리, 단위 동일이라 계수 추정 없음. V43 조건 충족)
    inter(pt, bt) = EB(pt, bt) − EB(pt) − EB(bt) + lg

arm
    P0  현행
    P1  유형 x 유형 상호작용 (성공률)
    P2  P1 + 성분별 상호작용 5개
    P3  유형 ID 를 범주형으로 직접 투입   <- 프로덕션이 '악화'로 기록한 것, 재확인

출력: outputs/v52_type_matchup2.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, OUT, CACHE, BASE_PARAMS, load, metrics

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
TM = MO / "trackman500_asof_train.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W, F_WEIGHT, K = 400, 0.25, 0.20, 300
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
K_P, K_B, MIN_VOL, DECAY = 5, 4, 300, 0.6
VS = 2024
EPS = 1e-7
TM_COLS = [f"tm500_recent_{c}_mean" for c in
           ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
            "extension", "rel_height", "rel_side"]]

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
tr, va = season < VS, season == VS

pid = df["pitcher_id"].to_numpy()
bid = df["batter_id"].to_numpy()
bhand = df["batter_hand"].to_numpy()
row_w = np.where(df["game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
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

tmdf = pd.read_parquet(TM, columns=["row_id"] + TM_COLS)
tmdf = df[["row_id"]].merge(tmdf, on="row_id", how="left")
TMV = tmdf[TM_COLS].to_numpy(np.float64)
last_tr = int(season[tr].max())
sw = DECAY ** (last_tr - season)          # 최근 시즌 가중


def profile(key, extra_tm):
    """최근 가중 실패 구성 + (선택) TrackMan + 물량. 학습 시즌만 사용한다."""
    w = np.where(tr, sw, 0.0)
    d = pd.DataFrame({"k": key, "w": w, "fail": FAIL * w})
    for c in COMPONENTS:
        d[c] = np.nan_to_num(LAB[c]) * w * (~np.isnan(LAB[c]))
    g = d.groupby("k").sum()
    n = pd.Series(np.ones(len(key)) * (tr * 1.0)).groupby(key).sum()
    tot = g["w"].to_numpy()
    fr = g["fail"].to_numpy() / np.maximum(tot, 1e-9)
    comp = np.column_stack([g[c].to_numpy() / np.maximum(g["fail"].to_numpy(), 1e-9)
                            for c in COMPONENTS])
    feat = np.column_stack([fr, comp, np.log1p(n.reindex(g.index).to_numpy())])
    names = ["실패율"] + [f"구성_{c}" for c in COMPONENTS] + ["물량"]
    if extra_tm:
        t = pd.DataFrame(TMV[tr], columns=TM_COLS)
        t["k"] = key[tr]
        tg = t.groupby("k").mean().reindex(g.index)
        feat = np.column_stack([feat, tg.to_numpy()])
        names += [c.replace("tm500_recent_", "").replace("_mean", "") for c in TM_COLS]
    return g.index.to_numpy(), feat, n.reindex(g.index).to_numpy(), names


def cluster(key, extra_tm, kk, tag):
    ids, feat, n, names = profile(key, extra_tm)
    lo, hi = np.nanpercentile(feat, [1, 99], axis=0)
    feat = np.clip(feat, lo, hi)
    mu = np.nanmean(feat, axis=0)
    sd = np.nanstd(feat, axis=0) + 1e-9
    Z = np.nan_to_num((feat - mu) / sd)          # TrackMan 결측 -> 표준화 후 0
    sel = n >= MIN_VOL
    km = KMeans(n_clusters=kk, n_init=50, random_state=17).fit(Z[sel])
    lab_of = dict(zip(ids[sel], km.labels_))
    arr = np.array([lab_of.get(k, kk) for k in key], dtype=int)
    print(f"{chr(10)}{tag} 유형 (저물량 포함 {kk+1}종)")
    print(f"  {'유형':<6}{'행비중':>8}{'선수':>6}"
          + "".join(f"{nm[:9]:>10}" for nm in names))
    for c in range(kk + 1):
        rows_ = (arr[tr] == c).mean() * 100
        mm = sel & (np.array([lab_of.get(i, kk) for i in ids]) == c)
        if mm.sum() == 0:
            mm = np.array([lab_of.get(i, kk) == c for i in ids])
        v = np.nanmean(feat[mm], axis=0) if mm.sum() else np.full(len(names), np.nan)
        print(f"  {c:<6}{rows_:>7.1f}%{int(mm.sum()):>6}"
              + "".join(f"{x:>10.3f}" for x in v))
    return arr, kk + 1


PT, n_p = cluster(pid, True, K_P, "투수")
BT, n_b = cluster(bid, False, K_B, "타자")


def interaction(a, b, vals, tag):
    m = tr & ~np.isnan(vals)
    d = pd.DataFrame({"a": a[m], "b": b[m], "y": vals[m]})
    lg = float(d["y"].mean())
    g = d.groupby(["a", "b"])["y"].agg(["sum", "size"])
    e_ab = (g["sum"] + K * lg) / (g["size"] + K)
    ga = d.groupby("a")["y"].agg(["sum", "size"])
    gb = d.groupby("b")["y"].agg(["sum", "size"])
    e_a = (ga["sum"] + K * lg) / (ga["size"] + K)
    e_b = (gb["sum"] + K * lg) / (gb["size"] + K)
    iab = pd.MultiIndex.from_arrays([a, b])
    v_ab = np.where(np.isnan(e_ab.reindex(iab).to_numpy()), lg,
                    e_ab.reindex(iab).to_numpy())
    v_a = np.where(np.isnan(e_a.reindex(pd.Index(a)).to_numpy()), lg,
                   e_a.reindex(pd.Index(a)).to_numpy())
    v_b = np.where(np.isnan(e_b.reindex(pd.Index(b)).to_numpy()), lg,
                   e_b.reindex(pd.Index(b)).to_numpy())
    nab = g["size"].reindex(iab).fillna(0.0).to_numpy()
    inter = v_ab - v_a - v_b + lg
    print(f"  {tag:<20} 셀 {len(g):>3}개  최소 {int(g['size'].min()):>7,}행  "
          f"중앙 {int(g['size'].median()):>7,}행  sd(행가중) "
          f"{np.sqrt(np.average(inter[m]**2)):.5f}", flush=True)
    return inter, nab / (nab + K)


td = df.loc[tr]
BASE_F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
pidx = pd.MultiIndex.from_arrays([pid, bhand])
for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
    d = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": ax[tr], "y": y_all[tr]})
    l0 = float(d["y"].mean())
    g2 = d.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
    e2 = (g2["sum"] + K * l0) / (g2["size"] + K)
    e3 = (g3["sum"] + K * l0) / (g3["size"] + K)
    i3 = pd.MultiIndex.from_arrays([pid, bhand, ax])
    v2 = np.where(np.isnan(e2.reindex(pidx).to_numpy()), l0, e2.reindex(pidx).to_numpy())
    v3 = np.where(np.isnan(e3.reindex(i3).to_numpy()), l0, e3.reindex(i3).to_numpy())
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    BASE_F[f"{tag}_split"], BASE_F[f"{tag}_rel"] = v3 - v2, sz / (sz + K)
    BASE_F[f"{tag}_w"] = (v3 - v2) * sz / (sz + K)
print(f"{chr(10)}기준 피처 {BASE_F.shape[1]}개{chr(10)}상호작용 셀 규모")
INT = interaction(PT, BT, y_all, "투수유형 x 타자유형")
INT_C = {k: interaction(PT, BT, LAB[k], f"  성분 [{k}]") for k in COMPONENTS}


def extrap(a, mask):
    m = mask & ~np.isnan(a)
    s = pd.Series(a[m]).groupby(pd.Series(season[m])).mean().sort_index()
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
        m = tr & ~np.isnan(arr)
        prm = {**BASE_PARAMS, "base_score": extrap(arr, tr),
               **params_for(float(np.nanmean(arr[m])))}
        Xv = X[va]
        d_tr = xgb.DMatrix(X[m], label=arr[m], weight=row_w[m])
        d_va = xgb.DMatrix(Xv)
        p_tr = Pool(X[m], arr[m], weight=row_w[m])
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(Xv)[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b, game_type=gt)["bss_raw"]


def build(arm):
    F = BASE_F.copy()
    if arm in ("P1", "P2"):
        F["ty_int"], F["ty_rel"] = INT
        F["ty_w"] = INT[0] * INT[1]
    if arm == "P2":
        for k in COMPONENTS:
            F[f"ty_{k}"] = INT_C[k][0]
    if arm == "P3":
        F["ptype"], F["btype"] = PT.astype(np.float32), BT.astype(np.float32)
    return F


t0, rows = time.time(), []
NAME = {"P0": "현행", "P1": "유형 상호작용", "P2": "P1 + 성분별",
        "P3": "유형 ID 직접"}
print(f"{chr(10)}{'arm':<20}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}"
      f"{'t_row':>8}{'경과':>8}")
for arm in ["P0", "P1", "P2", "P3"]:
    F = build(arm)
    p_ie = line(F.to_numpy(np.float32))
    np.save(CACHE / f"v52_{arm}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(b / (1 - b)), np.log(p_ie / (1 - p_ie)))[0, 1])
    q = np.clip(W * p_ie + (1 - W) * b, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": arm, "n_features": F.shape[1], "solo_bss": solo, "corr": corr,
                 "dbss": d, "t_row": d / se})
    print(f"{arm} {NAME[arm]:<17}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}"
          f"{d:>+9.2f}{d/se:>8.2f}{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v52_type_matchup2.csv", index=False)
r0 = res[res.arm == "P0"].iloc[0]
print(f"{chr(10)}{'='*58}{chr(10)}{'arm':<20}{'ΔBSS 대비':>12}{'단독 대비':>12}")
for _, r in res.iterrows():
    print(f"{r.arm} {NAME[r.arm]:<17}{r.dbss-r0.dbss:>+12.2f}"
          f"{r.solo_bss-r0.solo_bss:>+12.2f}")
print(f"{chr(10)}V35 P4(투수x타자 개별)는 단독 -486.85 였다. 유형화로 셀을 키운 것이")
print(f"그 붕괴를 없애는지가 이 실험의 핵심이다.")
print(f"{chr(10)}saved -> {OUT/'v52_type_matchup2.csv'}")
