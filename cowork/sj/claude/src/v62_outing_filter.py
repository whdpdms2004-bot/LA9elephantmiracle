"""V62: 학습 표본 선택 — 짧은 등판과 저물량 행을 거른다.

사용자 지시
    "임베딩이나 모델링 할 때 몇 구 이하인 투수가 던졌거나 해당 선수가 평소에 비해
     금방 마운드에서 내려 했을 때 그 값을 학습에 사용하지 않도록"

규칙상 안전하다
    학습 행을 거르는 것은 '피처'가 아니라 '표본 선택'이다. 추론 경로가 바뀌지
    않으므로 test 행간 집계 문제가 생기지 않는다.

등판을 어떻게 잡았나
    asof_pitcher_prev1_game_success_rate 는 경기 단위로만 갱신된다.
    투수별로 asof_pitcher_n 순서로 정렬한 뒤 그 값이 일정한 구간이 한 번의 등판이다.

    검증 (야구 상식)
        등판 45,121개 / 투구 1,475,092행
        이닝1 시작(선발)  9,784개   등판당 중앙  90구
        그 외(구원)      35,337개   등판당 중앙  16구
    KBO 실제와 맞는다. 분할이 성립한다.

    '짧은 등판' = 그 투수의 (선발/구원 구분별) 중앙 등판 길이 대비 비율.
    중앙값은 학습 시즌만으로 계산한다.

필터 크기와 성격
    ratio<0.5   제외 1.97%   그 행 성공률 0.5181  (전체 0.5238)  <- 그날 안 좋았던 행
    n<100       제외 4.85%   그 행 성공률 0.5415               <- 시즌과 교락(2019 0.5647)

비용도 알고 있다
    V9 에서 학습 행을 절반으로 줄이는 비용을 -4.39 dBSS 로 쟀다.
    2~5% 제거의 볼륨 비용은 -0.2 ~ -0.4 수준이다. 거른 행이 진짜 잡음이면 순이득이 난다.

    다만 검증셋에서는 그 행들을 뺄 수 없다. 학습에서만 빼면 분포 불일치가 생긴다.
    그래서 V10 의 F행 패턴(버리지 말고 가중치 0.20 -> +4.07)도 함께 잰다.

arm
    W0  현행
    W1  짧은 등판 제외 (ratio < 0.5)
    W2  짧은 등판 가중치 0.5
    W3  저물량 제외 (asof_pitcher_n < 100)
    W4  W2 + W3

판정: fold 2024 선별. V61 이 확정한 대로 내부 +3 미만은 제출 근거로 쓰지 않는다.
출력: outputs/v62_outing_filter.csv
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

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, F_WEIGHT, K = 400, 0.20, 300
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
VS, SHORT_TH, LOWVOL_TH = 2024, 0.5, 100
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
balls, strikes = df["balls_before"].to_numpy(), df["strikes_before"].to_numpy()
cnt_b = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))
inn_b = np.digitize(df["inning"].to_numpy(), [4, 7, 10])

# ---------------------------------------------- 등판 분할 (학습 시즌 통계만)
o = np.argsort(pid * 10_000_000 + df["asof_pitcher_n"].to_numpy(), kind="stable")
pv = df["asof_pitcher_prev1_game_success_rate"].to_numpy()[o]
gp = pid[o]
chg = np.r_[True, (gp[1:] != gp[:-1]) | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
outing = np.empty(len(df), dtype=np.int64)
outing[o] = np.cumsum(chg) - 1
od = pd.DataFrame({"outing": outing, "pid": pid, "inn": df["inning"].to_numpy(),
                   "tr": tr})
agg = od.groupby("outing").agg(n=("outing", "size"), pid=("pid", "first"),
                               first_inn=("inn", "min"), ntr=("tr", "sum"))
agg["start"] = (agg["first_inn"] == 1).astype(int)
med = (agg[agg["ntr"] > 0].groupby(["pid", "start"])["n"].median().rename("med"))
agg = agg.join(med, on=["pid", "start"])
agg["ratio"] = agg["n"] / agg["med"].clip(lower=1)
RATIO = agg["ratio"].reindex(outing).to_numpy()
SHORT = np.nan_to_num(RATIO, nan=1.0) < SHORT_TH
LOWVOL = df["asof_pitcher_n"].to_numpy() < LOWVOL_TH
print(f"등판 {len(agg):,}개   선발 {int(agg['start'].sum()):,} (중앙 "
      f"{int(agg.loc[agg['start'] == 1, 'n'].median())}구)   구원 "
      f"{int((1-agg['start']).sum()):,} (중앙 "
      f"{int(agg.loc[agg['start'] == 0, 'n'].median())}구)")
print(f"학습행 중 짧은등판 {SHORT[tr].mean()*100:.2f}%   "
      f"저물량 {LOWVOL[tr].mean()*100:.2f}%   "
      f"둘 중 하나 {(SHORT | LOWVOL)[tr].mean()*100:.2f}%", flush=True)

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

td = df.loc[tr]
BASE_F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                  CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}))
pidx = pd.MultiIndex.from_arrays([pid, bhand])
for tag, ax in [("cnt", cnt_b), ("inn", inn_b)]:
    d2 = pd.DataFrame({"p": pid[tr], "h": bhand[tr], "a": ax[tr], "y": y_all[tr]})
    l0 = float(d2["y"].mean())
    g2 = d2.groupby(["p", "h"])["y"].agg(["sum", "size"])
    g3 = d2.groupby(["p", "h", "a"])["y"].agg(["sum", "size"])
    e2 = (g2["sum"] + K * l0) / (g2["size"] + K)
    e3 = (g3["sum"] + K * l0) / (g3["size"] + K)
    i3 = pd.MultiIndex.from_arrays([pid, bhand, ax])
    v2 = np.where(np.isnan(e2.reindex(pidx).to_numpy()), l0, e2.reindex(pidx).to_numpy())
    v3 = np.where(np.isnan(e3.reindex(i3).to_numpy()), l0, e3.reindex(i3).to_numpy())
    sz = g3["size"].reindex(i3).fillna(0.0).to_numpy()
    BASE_F[f"{tag}_split"], BASE_F[f"{tag}_rel"] = v3 - v2, sz / (sz + K)
    BASE_F[f"{tag}_w"] = (v3 - v2) * sz / (sz + K)
X = BASE_F.to_numpy(np.float32)
print(f"피처 {X.shape[1]}개", flush=True)


def extrap(a, keep):
    m_ = tr & keep & ~np.isnan(a)
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


def line(keep, wmul):
    p = {}
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & keep & ~np.isnan(arr)
        w = np.where(IS_F, F_WEIGHT, 1.0) * wmul
        prm = {**BASE_PARAMS, "base_score": extrap(arr, keep),
               **params_for(float(np.nanmean(arr[mm])))}
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=w[mm])
        d_va = xgb.DMatrix(X[va])
        p_tr = Pool(X[mm], arr[mm], weight=w[mm])
        acc = np.zeros(int(va.sum()))
        for s in SEEDS:
            acc += 0.5 * xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                                   verbose_eval=False).predict(d_va)
            c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                                   l2_leaf_reg=6.0, loss_function="Logloss",
                                   random_seed=s, task_type="GPU", verbose=0)
            c.fit(p_tr)
            acc += 0.5 * c.predict_proba(X[va])[:, 1]
        p[tag] = np.clip(acc / len(SEEDS), EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
b = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
y_va = y_all[va]
null = y_va.mean() * (1 - y_va.mean())
ref = metrics(y_va, b)["bss_raw"]
wv = BW[bucket_all[va]]
ALL = np.ones(len(df), bool)
ONE = np.ones(len(df))

ARMS = [
    ("W0_current", ALL, ONE),
    ("W1_short_drop", ~SHORT, ONE),
    ("W2_short_w05", ALL, np.where(SHORT, 0.5, 1.0)),
    ("W3_lowvol_drop", ~LOWVOL, ONE),
    ("W4_short_w05_lowvol_drop", ~LOWVOL, np.where(SHORT, 0.5, 1.0)),
]
t0, rows = time.time(), []
print(f"{chr(10)}{'arm':<26}{'학습행':>10}{'단독':>10}{'ΔBSS':>9}{'t_row':>8}{'경과':>8}")
for name, keep, wmul in ARMS:
    p_ie = line(keep, wmul)
    np.save(CACHE / f"v62_{name}.npy", p_ie)
    solo = metrics(y_va, p_ie)["bss_raw"]
    q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
    d = metrics(y_va, q)["bss_raw"] - ref
    dr = (b - y_va) ** 2 - (q - y_va) ** 2
    se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
    ntr = int((tr & keep).sum())
    rows.append({"arm": name, "n_train": ntr, "solo_bss": solo, "dbss": d,
                 "t_row": d / se})
    print(f"{name:<26}{ntr:>10,}{solo:>10.2f}{d:>+9.2f}{d/se:>8.2f}"
          f"{time.time()-t0:>7.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v62_outing_filter.csv", index=False)
r0 = res[res.arm == "W0_current"].iloc[0]
print(f"{chr(10)}{'='*62}{chr(10)}{'arm':<26}{'ΔBSS 대비':>12}{'단독 대비':>12}")
for _, r in res.iterrows():
    print(f"{r.arm:<26}{r.dbss-r0.dbss:>+12.2f}{r.solo_bss-r0.solo_bss:>+12.2f}")
print(f"{chr(10)}V61 기준: 내부 +3 미만은 제출 근거로 쓰지 않는다.")
print(f"{chr(10)}saved -> {OUT/'v62_outing_filter.csv'}")
