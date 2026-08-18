"""V73: submit_035.zip — 2차 상호작용 피처 (V71 C4).

V71 결과 (fold 2023 / 2024, C0 현행 대비)
    arm            단독            base상관            ΔBSS
    C1 제곱항      -4.88 / +0.09   0.8886 / 0.8212   +0.01 / -0.34
    C2 곱          +0.21 / +4.09   0.8883 / 0.8202   +2.75 / +3.14
    C3 차/비       -4.20 / -1.83   0.8890 / 0.8220   +0.95 / -0.60
    C4 곱+차/비    +9.03 / +5.26   0.8877 / 0.8194   +4.73 / +3.50   <- 채택

    C1(제곱)이 0 인 것은 사전 예측대로다 — 트리는 단일 피처의 단조 변환에 불변이라
    x^2 는 정보를 안 늘린다. 대조군이 예측대로 나왔으므로 나머지 수치를 믿을 수 있다.

    C4 는 단독 상승 / 상관 하락 / ΔBSS 상승 세 신호가 전부 같은 방향인 유일한 사례이고,
    V61 의 '+3 미만은 제출 근거로 쓰지 않는다' 기준을 두 fold 모두에서 넘긴 첫 항목이다.

기준 패키지
    submit_032 (Public 979). Tier E 는 submit_033 에서 973.14 로 빠졌으므로 제외.
    짧은 등판 가중치(submit_034)는 넣지 않는다 — V64 는 +2.21/+0.63 이었으나
    V72 의 F_short0.5 가 +0.19/-1.74 로 재현되지 않았다. 잡음 범위다.

넣는 것: 곱 9개 + 차/비 6개 = 12열 (111 -> 123)
    곱   platoon_split x cnt_split,  platoon_split x inn_split,  cnt_split x inn_split
         pitcher_success x batter_success,  pitcher_middle x batter_middle
         pitcher_success x li,  pitcher_reverse x fastball_rate
         pitcher_success x log1p(n),  platoon_split x log1p(n)
    차   prev1/prev3/prev5 game success - career success
         prev1 game middle - career middle
         pitcher_success - batter_success
    비   pitcher_success / batter_success

    전부 이미 있는 열에서 파생되므로 추론 시 추가 룩업이 없다. 행 독립성 무관.

출력: submit/2026-08-15/submit_035.zip
"""
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, CACHE, BASE_PARAMS, load

SJ = Path(__file__).resolve().parents[2]
BASE_ZIP = SJ / "submit" / "2026-08-15" / "submit_032.zip"
OUT_ZIP = SJ / "submit" / "2026-08-15" / "submit_035.zip"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, F_WEIGHT = 400, 0.20
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
NL = chr(10)
EPS = 1e-7

# V71 은 계층 차감을 직접 만들며 cnt_split / inn_split 로 이름 붙였는데,
# CF.build 는 같은 값을 count_platoon_split / inning_platoon_split 로 낸다.
# 값은 같고 이름만 다르다 — 여기서는 CF.build 의 이름을 쓴다.
PRODUCTS = [
    ("platoon_split", "count_platoon_split"),
    ("platoon_split", "inning_platoon_split"),
    ("count_platoon_split", "inning_platoon_split"),
    ("asof_pitcher_success_rate", "asof_batter_success_rate"),
    ("asof_pitcher_middle_rate", "asof_batter_middle_rate"),
    ("asof_pitcher_success_rate", "li"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_fastball_rate"),
]
DIFFS = [
    ("asof_pitcher_prev1_game_success_rate", "asof_pitcher_success_rate"),
    ("asof_pitcher_prev3_game_success_rate", "asof_pitcher_success_rate"),
    ("asof_pitcher_prev5_game_success_rate", "asof_pitcher_success_rate"),
    ("asof_pitcher_prev1_game_middle_rate", "asof_pitcher_middle_rate"),
    ("asof_pitcher_success_rate", "asof_batter_success_rate"),
]

if OUT_ZIP.exists():
    raise FileExistsError(OUT_ZIP)
assert len(OUT_ZIP.name) < 30

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
season = df["season"].to_numpy()
IS_F = df["game_type"].astype(str).to_numpy() == "F"
ROW_W = np.where(IS_F, F_WEIGHT, 1.0)

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}

print("2019~2024 전체로 spec / platoon 생성", flush=True)
spec = CF.make_spec(df)
platoon = CF.make_platoon_table(df)
bat_platoon = CF.make_batter_platoon_table(df, LAB)
count_platoon = CF.make_count_platoon_table(df)
inning_platoon = CF.make_inning_platoon_table(df)
feat = CF.build(df[INPUT_COLS], spec, platoon, bat_platoon, count_platoon,
                inning_platoon)
n_before = feat.shape[1]


def add_second_order(F):
    """V71 C4. 이미 있는 열에서만 파생한다."""
    lg_n = np.log1p(pd.to_numeric(F["asof_pitcher_n"]).to_numpy())
    for a, b in PRODUCTS:
        F[f"x_{a[:14]}_{b[:14]}"] = F[a].to_numpy() * F[b].to_numpy()
    F["x_psucc_logn"] = F["asof_pitcher_success_rate"].to_numpy() * lg_n
    F["x_split_logn"] = F["platoon_split"].to_numpy() * lg_n
    for a, b in DIFFS:
        F[f"d_{a[:16]}_{b[:12]}"] = F[a].to_numpy() - F[b].to_numpy()
    F["r_pitch_bat"] = (F["asof_pitcher_success_rate"].to_numpy()
                        / np.clip(F["asof_batter_success_rate"].to_numpy(),
                                  1e-3, None))
    return F


feat = add_second_order(feat)
spec["columns"] = list(feat.columns)
spec["second_order"] = {"products": [list(p) for p in PRODUCTS],
                        "diffs": [list(d) for d in DIFFS],
                        "extra": ["x_psucc_logn", "x_split_logn", "r_pitch_bat"]}
X = feat.to_numpy(np.float32)
print(f"  피처 {n_before} -> {X.shape[1]}개 (2차 {X.shape[1]-n_before}열)", flush=True)


def extrap_2025(a):
    m = ~np.isnan(a)
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


work = CACHE / "submit035_build"
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)

meta_comp, files = {}, []
for tag in COMPONENTS:
    arr = LAB[tag].astype(np.float64)
    m = ~np.isnan(arr)
    rate = float(np.nanmean(arr))
    bs = extrap_2025(arr)
    extra = params_for(rate)
    prm = {**BASE_PARAMS, "base_score": bs, **extra}
    d_tr = xgb.DMatrix(X[m], label=arr[m], weight=ROW_W[m])
    for s in SEEDS:
        b = xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                      verbose_eval=False)
        fn = f"c35_{tag}_x{s}.ubj"
        b.save_model(str(work / fn))
        files.append(fn)
    p_tr = Pool(X[m], arr[m], weight=ROW_W[m])
    for s in SEEDS:
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        fn = f"c35_{tag}_c{s}.cbm"
        c.save_model(str(work / fn))
        files.append(fn)
    meta_comp[tag] = {"base_rate": rate, "base_score": bs, "xgb_params": extra,
                      "n_train": int(m.sum())}
    print(f"  [{tag:>2}] rate {rate:.6f}  base_score {bs:.8f}  "
          f"leaves {extra['max_leaves']}  모델 {2*len(SEEDS)}개", flush=True)

spec["base_scores"] = {k: v["base_score"] for k, v in meta_comp.items()}
spec["xgb_params"] = {k: v["xgb_params"] for k, v in meta_comp.items()}
spec["seeds"] = SEEDS
spec["components"] = COMPONENTS
json.dump(spec, open(work / "c35_spec.json", "w"), indent=1, sort_keys=True)
platoon.to_csv(work / "platoon_2025.csv", index=False)
bat_platoon.to_csv(work / "bat_platoon_2025.csv", index=False)
count_platoon.to_csv(work / "count_platoon_2025.csv", index=False)
inning_platoon.to_csv(work / "inning_platoon_2025.csv", index=False)

with ZipFile(BASE_ZIP) as z:
    script = z.read("script.py").decode("utf-8")
    metadata = json.loads(z.read("model/metadata.json"))
    payload = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}

for tag in ["def component_features", "def component_blend",
            "prediction = component_blend"]:
    assert script.count(tag) == 1, f"'{tag}' {script.count(tag)}회"
script = script.replace("c29_%s_x%d.ubj", "c35_%s_x%d.ubj", 1)
script = script.replace("c29_%s_c%d.cbm", "c35_%s_c%d.cbm", 1)

RET_OLD = '    return pd.DataFrame(out)[spec["columns"]].to_numpy(np.float32)'
RET_NEW = NL.join([
    "    # V71 C4: 2차 상호작용. 이미 만든 열에서만 파생하므로 추가 룩업이 없다.",
    "    # 곱은 트리가 축 정렬 분할로 근사하기 어려운 항이고, 제곱은 트리가",
    "    # 단조 변환에 불변이라 넣지 않는다 (V71 C1 이 실측으로 0 을 확인했다).",
    '    so = spec["second_order"]',
    '    logn = np.log1p(pd.to_numeric(frame["asof_pitcher_n"],'
    ' errors="coerce").to_numpy())',
    '    for a, b in so["products"]:',
    '        out["x_%s_%s" % (a[:14], b[:14])] = out[a] * out[b]',
    '    out["x_psucc_logn"] = out["asof_pitcher_success_rate"] * logn',
    '    out["x_split_logn"] = out["platoon_split"] * logn',
    '    for a, b in so["diffs"]:',
    '        out["d_%s_%s" % (a[:16], b[:12])] = out[a] - out[b]',
    '    out["r_pitch_bat"] = (out["asof_pitcher_success_rate"]',
    '                          / np.clip(out["asof_batter_success_rate"], 1e-3, None))',
    '    return pd.DataFrame(out)[spec["columns"]].to_numpy(np.float32)'])
assert script.count(RET_OLD) == 1
script = script.replace(RET_OLD, RET_NEW, 1)
metadata["component_blend"]["spec_file"] = "c35_spec.json"

for tag in ["def component_features", "def component_blend",
            "prediction = component_blend"]:
    assert script.count(tag) == 1
assert "c29_" not in script
print(f"{NL}script.py 정리 완료  {len(script.splitlines())}줄", flush=True)

cb = metadata["component_blend"]
cb["model_prefix"] = "c35"
cb["second_order_features"] = {
    "source": "V71 C4",
    "products": len(PRODUCTS) + 2,
    "diffs_ratios": len(DIFFS) + 1,
    "rationale": (
        "trees are invariant to monotone transforms of a single feature, so squares "
        "add nothing - V71 C1 measured exactly that (+0.01 / -0.34). products and "
        "differences are what axis-aligned splits approximate poorly"),
    "validation": {
        "fold_2023": {"d_bss": 4.73, "d_solo": 9.03, "corr": 0.8877},
        "fold_2024": {"d_bss": 3.50, "d_solo": 5.26, "corr": 0.8194},
        "baseline_corr": {"fold_2023": 0.8893, "fold_2024": 0.8216},
        "note": ("solo up, correlation with base down, blended delta up - all three "
                 "signals aligned. the only candidate this session to satisfy the "
                 "V65 criterion fully and the first to clear the +3 internal bar on "
                 "both folds")},
    "rejected_variants": {
        "C1_squares": "+0.01 / -0.34  (predicted zero, confirms measurement)",
        "C3_diffs_only": "+0.95 / -0.60  (only pays together with products)"},
}
cb["excluded"] = {
    "tier_e": "submit_033 scored 973.14 public versus submit_032 979",
    "short_outing_weight": (
        "V64 measured +2.21 / +0.63 but V72 arm F_short0.5 measured +0.19 / -1.74 "
        "under an identical harness. does not replicate, treated as noise"),
    "trackman": (
        "V66: solo rose on 2023 (+21.04) but correlation with base rose too "
        "(0.8889 -> 0.8905) because the base already carries 98 tm500 columns. "
        "blended delta -0.30 / -0.11"),
    "learning_method_sweep": (
        "V72 swept 40 arms over EB shrinkage, season extrapolators, recency "
        "weighting, objectives, tree capacity and sample weights. zero arms were "
        "positive on both metrics on both folds"),
}
metadata["track"] = "reverse20_s0475_comp5_platoon4_cal_volw_second_order"
metadata["version"] = 22

for stale in [n for n in list(payload) if "/c29_" in n]:
    del payload[stale]
payload["script.py"] = script.encode("utf-8")
payload["model/metadata.json"] = json.dumps(metadata, indent=1,
                                            sort_keys=True).encode("utf-8")
for fn in files + ["c35_spec.json", "platoon_2025.csv", "bat_platoon_2025.csv",
                   "count_platoon_2025.csv", "inning_platoon_2025.csv"]:
    payload[f"model/{fn}"] = (work / fn).read_bytes()


def info(name):
    zi = ZipInfo(name)
    zi.date_time = (2026, 8, 15, 0, 0, 0)
    zi.compress_type = ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    return zi


with ZipFile(OUT_ZIP, "w", ZIP_DEFLATED) as z:
    for name in sorted(payload):
        assert chr(92) not in name
        z.writestr(info(name), payload[name])
print(f"{NL}생성 완료 {OUT_ZIP}  {OUT_ZIP.stat().st_size/2**20:.1f}MB  "
      f"파일 {len(payload)}개")
print(f"  성분 모델 {len(files)}개 (c35_)   2차 피처 {X.shape[1]-n_before}열")
