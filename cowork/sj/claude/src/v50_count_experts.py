"""V50: 볼카운트 x 실패유형 전문가 모델. — 아직 한 번도 안 해본 것.

지금까지 카운트를 쓴 방식
    V19  계층 차감 EB 스플릿을 '피처'로              +8.44
    프로덕션  r_context = count x inning4 x hands 보정 룩업   +18
    둘 다 '피처/보정'이다. 모델을 카운트로 쪼갠 적은 없다.

    성분(실패유형)으로는 쪼갰다 — m/r/mr/ob/oz 5개 모델.
    프로덕션에는 game_type_experts 가 있다 — 경기유형으로 쪼갠 전문가 모델.
    카운트 축에만 그 패턴을 안 썼다.

왜 될 만한가
    제구 실패의 성격이 카운트에 따라 다르다. 3볼에서는 스트라이크를 넣어야 하니
    한가운데(m)로 몰리고, 2스트라이크에서는 유인구라 존 밖(oz)이 정상이다.
    한 모델이 두 체제를 같은 파라미터로 맞추면 서로를 방해한다.

    기저율부터 확인한다 — 카운트군별 성분 기저율이 크게 다르면 근거가 된다.

arm
    N0  전역 (현행)                       성분 5모델
    N1  카운트 3군 전문가                  3 x 5 = 15모델, 각 1/3 데이터
    N2  카운트 12셀 전문가                12 x 5 = 60모델, 각 1/12 데이터
    N3  0.5 x N0 + 0.5 x N1               전문가와 전역을 반씩

    학습 총량은 N0 과 N1/N2 가 거의 같다(행을 나눠 갖는다).
    전문가별로 base_score 를 그 셀의 기저율로 따로 외삽한다.

판정: 2024 로 선별하고 이긴 것만 세 fold. 지금까지 2024 단독 선별이 세 번
      뒤집혔으므로(V23, V38, V47) 선별은 방향 확인용으로만 본다.

출력: outputs/v50_count_experts.csv
"""
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

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W, F_WEIGHT, K = 400, 0.25, 0.20, 300
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
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
bhand = df["batter_hand"].to_numpy()
row_w = np.where(df["game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
balls = df["balls_before"].to_numpy()
strikes = df["strikes_before"].to_numpy()
cnt3 = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
cnt12 = balls * 3 + strikes
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

# ---------------------------------------------- 근거 확인: 셀별 기저율
print("카운트 12셀 x 성분 기저율 (학습 시즌)")
print(f"  {'B-S':<6}{'비중':>7}" + "".join(f"{c:>8}" for c in COMPONENTS))
for b in range(4):
    for s in range(3):
        c = b * 3 + s
        m = tr & (cnt12 == c)
        if m.sum() == 0:
            continue
        print(f"  {b}-{s:<4}{m.mean()*100:>6.2f}%"
              + "".join(f"{np.nanmean(LAB[k][m]):>8.3f}" for k in COMPONENTS))
sp = {k: np.nanstd([np.nanmean(LAB[k][tr & (cnt12 == c)]) for c in range(12)])
      for k in COMPONENTS}
lg = {k: np.nanmean(LAB[k][tr]) for k in COMPONENTS}
print(f"  {'셀간 표준편차/평균':<13}"
      + "".join(f"{sp[k]/lg[k]:>8.2f}" for k in COMPONENTS), flush=True)

td = df.loc[tr]
F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
             CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
pidx = pd.MultiIndex.from_arrays([pid, bhand])
for tag, ax in [("cnt", cnt3), ("inn", inn_b)]:
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
    F[f"{tag}_split"], F[f"{tag}_rel"] = v3 - v2, sz / (sz + K)
    F[f"{tag}_w"] = (v3 - v2) * sz / (sz + K)
X = F.to_numpy(np.float32)
print(f"{chr(10)}피처 {X.shape[1]}개", flush=True)


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


def fit_predict(tr_mask, va_mask, tag):
    """tr_mask 로 학습해 va_mask 행을 예측한다."""
    arr = LAB[tag]
    m = tr_mask & ~np.isnan(arr)
    rate = float(np.nanmean(arr[m]))
    prm = {**BASE_PARAMS, "base_score": extrap(arr, tr_mask), **params_for(rate)}
    d_tr = xgb.DMatrix(X[m], label=arr[m], weight=row_w[m])
    d_va = xgb.DMatrix(X[va_mask])
    p_tr = Pool(X[m], arr[m], weight=row_w[m])
    acc = np.zeros(int(va_mask.sum()))
    for s in SEEDS:
        acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                               verbose_eval=False).predict(d_va)
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        acc += 0.5 * c.predict_proba(X[va_mask])[:, 1]
    return np.clip(acc / len(SEEDS), EPS, 1 - EPS)


def line(seg):
    """seg=None 이면 전역, 아니면 seg 배열의 값마다 전문가를 따로 학습한다."""
    p = {}
    for tag in COMPONENTS:
        out = np.zeros(int(va.sum()))
        if seg is None:
            out = fit_predict(tr, va, tag)
        else:
            sv = seg[va]
            for c in np.unique(seg[tr]):
                vm = va & (seg == c)
                if vm.sum() == 0:
                    continue
                out[sv == c] = fit_predict(tr & (seg == c), vm, tag)
        p[tag] = out
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b, game_type=gt)["bss_raw"]

t0, rows, keep = time.time(), [], {}
print(f"{chr(10)}{'arm':<20}{'모델수':>7}{'단독':>10}{'corr':>8}{'ΔBSS':>9}"
      f"{'t_row':>8}{'경과':>8}")
for name, seg, nm in [("N0_global", None, 5), ("N1_count3", cnt3, 15),
                      ("N2_count12", cnt12, 60)]:
    p_ie = line(seg)
    keep[name] = p_ie
    np.save(CACHE / f"v50_{name}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    corr = float(np.corrcoef(np.log(b / (1 - b)), np.log(p_ie / (1 - p_ie)))[0, 1])
    q = np.clip(W * p_ie + (1 - W) * b, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "n_models": nm, "solo_bss": solo, "corr": corr,
                 "dbss": d, "t_row": d / se})
    print(f"{name:<20}{nm:>7}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}{d/se:>8.2f}"
          f"{time.time()-t0:>7.0f}s", flush=True)

for name, mix in [("N3_half_c3", 0.5 * keep["N0_global"] + 0.5 * keep["N1_count3"]),
                  ("N4_half_c12", 0.5 * keep["N0_global"] + 0.5 * keep["N2_count12"])]:
    p_ie = np.clip(mix, EPS, 1 - EPS)
    np.save(CACHE / f"v50_{name}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    q = np.clip(W * p_ie + (1 - W) * b, EPS, 1 - EPS)
    mm = metrics(y_va, q, game_type=gt)
    d = mm["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    rows.append({"arm": name, "n_models": 20, "solo_bss": solo,
                 "corr": float(np.corrcoef(np.log(b / (1 - b)),
                                           np.log(p_ie / (1 - p_ie)))[0, 1]),
                 "dbss": d, "t_row": d / se})
    print(f"{name:<20}{'-':>7}{solo:>10.2f}{'':>8}{d:>+9.2f}{d/se:>8.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v50_count_experts.csv", index=False)
r0 = res[res.arm == "N0_global"].iloc[0]
print(f"{chr(10)}{'='*58}{chr(10)}{'arm':<20}{'ΔBSS 대비':>12}{'단독 대비':>12}")
for _, r in res.iterrows():
    print(f"{r.arm:<20}{r.dbss-r0.dbss:>+12.2f}{r.solo_bss-r0.solo_bss:>+12.2f}")
print(f"{chr(10)}둘 다 양수면 세 fold 로 넘긴다. 2024 단독 선별은 지금까지 "
      f"세 번 뒤집혔다(V23, V38, V47).")
print(f"{chr(10)}saved -> {OUT/'v50_count_experts.csv'}")
