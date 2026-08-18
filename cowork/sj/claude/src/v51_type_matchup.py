"""V51: 투수 유형 x 타자 유형 상성을 성분 라인 안에서 쓴다.

프로덕션이 이미 한 것 (PITCHER_CLUSTER_MATCHUP_PLAN.md)
    TrackMan 물리 임베딩 -> 투수 군집(좌2/우4), 타자 군집(좌4/우6)
    투수유형 x 타자유형 상성 보정 -> submit_013, 2024 BSS 812.70
    기록된 결론: "군집 피처 직접 투입 | 완료 | hard/soft ID는 성능 악화"

프로덕션이 안 한 것
    그 유형 구조를 '성분 라인 안에서' 쓰는 것. 성공률 모델에만 적용했고
    5개 실패유형 모델에는 안 넣었다.

왜 지금 해야 하는가 — V35 P4 의 처방이다
    V35 에서 투수 x 타자 '개별' EB 를 넣었더니 성분단독이 745.9 -> 259.1 로
    붕괴했다. 셀이 작아 자기 라벨이 그대로 샜기 때문이다.
    유형 x 유형이면 셀이 수만 행이라 그 실패 원인이 사라진다.
    유형화는 P4 가 실패한 이유에 정확히 대응하는 수단이다.

    그리고 V43 의 단위 조건도 만족한다 — 상호작용과 주효과가 전부 성공률 EB 라
    환산 계수를 추정할 필요가 없다(계수 1).

유형 만들기 (TrackMan 없이 메인 피처만, 학습 시즌만 사용)
    style   투수 구종 배합 + 물량        fastball/breaking/offspeed rate, log n
    control 투수 제구 프로파일            success/middle/reverse/ball rate
    batter  타자 프로파일                 success/middle rate, log n
    손(hand) 안에서 따로 KMeans. 유형 수는 프로덕션 값을 따른다(좌 2·우 4, 타자 4).

상호작용 (2겹 차감 — V19 원리, 단위 동일)
    inter(pt, bt) = EB(pt, bt) − EB(pt) − EB(bt) + lg
    주효과 둘을 다 빼면 "이 투수 유형이 이 타자 유형에 특별히 강하거나 약한 정도"만
    남는다. 안 빼면 유형별 평균 실력과 중복이다.

arm
    O0  현행
    O1  style x batter 상호작용
    O2  control x batter 상호작용
    O3  O2 + 성분별 상호작용 (5개)
    O4  투수 유형별 전문가 모델          <- V50 이 예측하는 대로 질 것이나 질문에 답한다

출력: outputs/v51_type_matchup.csv
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
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W, F_WEIGHT, K = 400, 0.25, 0.20, 300
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
K_P = {1: 2, 2: 4}          # 투수 손별 유형 수 (프로덕션 좌2/우4)
K_B = 4                     # 타자 유형 수
VS = 2024
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

pid = df["pitcher_id"].to_numpy()
bid = df["batter_id"].to_numpy()
phand = df["pitcher_hand"].to_numpy()
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

STYLE = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
         "asof_pitcher_offspeed_rate"]
CTRL = ["asof_pitcher_success_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_ball_rate"]
BAT = ["asof_batter_success_rate", "asof_batter_middle_rate"]


MIN_VOL = 300


def build_types(key, hand, cols, nvol, kmap, tag):
    """선수별 프로파일 -> 손 안에서 군집.

    첫 시도에서 군집이 퇴화했다 — 타자 유형이 45.7%/50.3% 에 몰리고 나머지가
    0.0~2.2% 였다. 저물량 선수의 잡음 프로파일이 특이점으로 빠져 군집을
    끌고 간 것이다. 세 가지를 고친다.
        1) 학습 투구 300개 미만 선수는 군집에서 빼고 '저물량' 유형으로 모은다
        2) 프로파일을 마지막 asof 값이 아니라 학습 구간 실측 평균으로 만든다
        3) 각 축을 1/99 분위로 winsorize 한 뒤 표준화한다
    그리고 최소 군집 비중이 5% 미만이면 k 를 줄여 다시 잡는다.
    """
    sub = df.loc[tr, cols].copy()
    sub["k"], sub["h"] = key[tr], hand[tr]
    prof = sub.groupby("k").agg(["mean", "size"])
    n_obs = prof[(cols[0], "size")].to_numpy()
    feat = np.column_stack([prof[(c, "mean")].to_numpy() for c in cols]
                           + [np.log1p(n_obs)])
    hh = sub.groupby("k")["h"].first().to_numpy()
    ids = prof.index.to_numpy()
    lo, hi = np.nanpercentile(feat, [1, 99], axis=0)
    feat = np.clip(np.nan_to_num(feat, nan=np.nanmean(feat, axis=0)), lo, hi)

    lab_of = {}
    base = 0
    for h in sorted(set(hh)):
        sel = (hh == h) & (n_obs >= MIN_VOL)
        kk = kmap[h] if isinstance(kmap, dict) else kmap
        if sel.sum() < kk * 5:
            kk = max(1, sel.sum() // 5)
        Z = feat[sel]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        while kk > 1:
            km = KMeans(n_clusters=kk, n_init=30, random_state=17).fit(Z)
            share = np.bincount(km.labels_, weights=n_obs[sel], minlength=kk)
            if share.min() / share.sum() >= 0.05:
                break
            kk -= 1
        else:
            km = KMeans(n_clusters=1, n_init=1, random_state=17).fit(Z)
        for i, lb in zip(ids[sel], km.labels_):
            lab_of[i] = int(lb) + base
        base += kk
    LOW = base
    arr = np.array([lab_of.get(k, LOW) for k in key], dtype=int)
    total = base + 1
    sz = pd.Series(arr[tr]).value_counts().sort_index()
    print(f"  {tag:<8} 유형 {total}종(저물량 포함)   행 비중 "
          + " ".join(f"{v/tr.sum()*100:.1f}%" for v in sz.to_numpy()), flush=True)
    return arr, total


PT_STYLE, n_ps = build_types(pid, phand, STYLE, "asof_pitcher_pitchmix_n", K_P, "style")
PT_CTRL, n_pc = build_types(pid, phand, CTRL, "asof_pitcher_n", K_P, "control")
BT, n_b = build_types(bid, bhand, BAT, "asof_batter_n", K_B, "batter")


def interaction(a, b, vals, tag):
    """EB(a,b) − EB(a) − EB(b) + lg. 2겹 차감, 단위 동일(계수 1)."""
    m = tr & ~np.isnan(vals)
    d = pd.DataFrame({"a": a[m], "b": b[m], "y": vals[m]})
    lg = float(d["y"].mean())

    def eb(keys):
        g = d.groupby(keys)["y"].agg(["sum", "size"])
        return (g["sum"] + K * lg) / (g["size"] + K), g["size"]

    e_ab, sz = eb(["a", "b"])
    e_a, _ = eb(["a"])
    e_b, _ = eb(["b"])
    iab = pd.MultiIndex.from_arrays([a, b])
    v_ab = e_ab.reindex(iab).to_numpy()
    v_a = e_a.reindex(pd.Index(a)).to_numpy()
    v_b = e_b.reindex(pd.Index(b)).to_numpy()
    v_ab = np.where(np.isnan(v_ab), lg, v_ab)
    v_a = np.where(np.isnan(v_a), lg, v_a)
    v_b = np.where(np.isnan(v_b), lg, v_b)
    n_ab = sz.reindex(iab).fillna(0.0).to_numpy()
    print(f"  {tag:<16} 셀 {len(e_ab):>3}개   최소 {int(sz.min()):>7,}행   "
          f"중앙 {int(sz.median()):>7,}행   상호작용 sd {np.std(v_ab-v_a-v_b+lg):.5f}",
          flush=True)
    return v_ab - v_a - v_b + lg, n_ab / (n_ab + K)


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
INT_S = interaction(PT_STYLE, BT, y_all, "style x batter")
INT_C = interaction(PT_CTRL, BT, y_all, "control x batter")
INT_COMP = {k: interaction(PT_CTRL, BT, LAB[k], f"control x bat [{k}]")
            for k in COMPONENTS}


def extrap(a, mask):
    m = mask & ~np.isnan(a)
    s = pd.Series(a[m]).groupby(pd.Series(season[m])).mean().sort_index()
    if len(s) < 2:
        return float(np.clip(s.iloc[-1] if len(s) else 0.1, 0.005, 0.995))
    last = float(s.iloc[-1])
    return float(np.clip(last + (last - float(s.iloc[0]))
                         / (float(s.index[-1]) - float(s.index[0])), 0.005, 0.995))


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def fit_predict(X, trm, vam, tag):
    arr = LAB[tag]
    m = trm & ~np.isnan(arr)
    prm = {**BASE_PARAMS, "base_score": extrap(arr, trm),
           **params_for(float(np.nanmean(arr[m])))}
    Xv = X[vam]
    d_tr = xgb.DMatrix(X[m], label=arr[m], weight=row_w[m])
    d_va = xgb.DMatrix(Xv)
    p_tr = Pool(X[m], arr[m], weight=row_w[m])
    acc = np.zeros(int(vam.sum()))
    for s in SEEDS:
        acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                               verbose_eval=False).predict(d_va)
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        acc += 0.5 * c.predict_proba(Xv)[:, 1]
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


def line(X, seg=None):
    p = {}
    for tag in COMPONENTS:
        if seg is None:
            p[tag] = fit_predict(X, tr, va, tag)
        else:
            out = np.zeros(int(va.sum()))
            sv = seg[va]
            for c in np.unique(seg[tr]):
                vm = va & (seg == c)
                if vm.sum():
                    out[sv == c] = fit_predict(X, tr & (seg == c), vm, tag)
            p[tag] = out
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b, game_type=gt)["bss_raw"]


def build(arm):
    F = BASE_F.copy()
    if arm in ("O1",):
        F["ty_int"], F["ty_rel"] = INT_S
        F["ty_w"] = INT_S[0] * INT_S[1]
    if arm in ("O2", "O3"):
        F["ty_int"], F["ty_rel"] = INT_C
        F["ty_w"] = INT_C[0] * INT_C[1]
    if arm == "O3":
        for k in COMPONENTS:
            F[f"ty_{k}"] = INT_COMP[k][0]
    return F


t0, rows = time.time(), []
print(f"{chr(10)}{'arm':<24}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}"
      f"{'t_row':>8}{'경과':>8}")
for arm in ["O0", "O1", "O2", "O3", "O4"]:
    if arm == "O4":
        F, seg = BASE_F, PT_CTRL
    else:
        F, seg = build(arm), None
    p_ie = line(F.to_numpy(np.float32), seg)
    np.save(CACHE / f"v51_{arm}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(b / (1 - b)), np.log(p_ie / (1 - p_ie)))[0, 1])
    q = np.clip(W * p_ie + (1 - W) * b, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": arm, "n_features": F.shape[1], "solo_bss": solo, "corr": corr,
                 "dbss": d, "t_row": d / se})
    print(f"{arm:<24}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}{d/se:>8.2f}"
          f"{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v51_type_matchup.csv", index=False)
r0 = res[res.arm == "O0"].iloc[0]
NAME = {"O0": "현행", "O1": "style x batter", "O2": "control x batter",
        "O3": "O2 + 성분별", "O4": "투수유형별 전문가"}
print(f"{chr(10)}{'='*62}{chr(10)}{'arm':<24}{'ΔBSS 대비':>12}{'단독 대비':>12}")
for _, r in res.iterrows():
    print(f"{r.arm} {NAME[r.arm]:<21}{r.dbss-r0.dbss:>+12.2f}"
          f"{r.solo_bss-r0.solo_bss:>+12.2f}")
print(f"{chr(10)}V35 P4(투수x타자 개별)는 단독 -486.85 였다. 유형화로 셀을 키웠을 때")
print(f"그 붕괴가 사라지는지가 이 실험의 핵심이다.")
print(f"{chr(10)}saved -> {OUT/'v51_type_matchup.csv'}")
