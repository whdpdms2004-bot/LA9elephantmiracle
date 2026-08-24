"""V67: submit_034.zip — 지금까지 검증된 것을 전부 통합한 최종본.

지시: "실험하는게 아니라 지금까지 아이디어들 종합해서 가장 높은 성능의 모델"

기준 패키지 선택
    submit_032  Public 979   <- 채택 기준
    submit_033  Public 973.14 (032 + Tier E)
    Tier E 는 내부 두 fold 양수였으나(+3.93 / +2.47) Public 이 -5.86 이다.
    V61 이 확정한 대로 내부 +-2.5 는 Public 분해능 아래라 둘은 통계적으로 동률이지만,
    측정된 최고점은 032 이고 Tier E 를 넣을 적극적 근거가 없다. 뺀다.

이미 032 에 실려 있는 것 (Val2024 기여)
    투수 플래툰 계층차감      +13.75
    카운트 플래툰 계층차감     +8.44
    F행 학습 가중치 0.20      +4.07
    이닝 플래툰 계층차감      +2.98
    OUTSIDE 성분 분할         +2.64
    타자 성분별 플래툰        +2.44
    합성 절편 0.0077          +1.37
    CatBoost 계열 추가        +1.23
    성분별 파라미터           +0.91
    구간별 결합 가중치 [.25 .25 .25 .30 .40]  (V48, Public +15 로 확인)

새로 넣는 것
    (1) 짧은 등판 학습 가중치 0.5   V64 두 fold 두 지표 양수
            dBSS +2.21 / +0.63,  단독 +12.48 / +2.33
        등판 분할은 asof_pitcher_prev1_game_success_rate 가 경기 단위로만 갱신되는
        성질을 이용한다. 투수별 asof_pitcher_n 순 정렬 -> 값이 일정한 구간이 한 등판.
        '짧다' = 그 투수의 (선발/구원별) 중앙 등판 길이 대비 0.5 미만.
        학습 가중치만 건드리므로 추론 경로가 바뀌지 않는다 — 행 독립성 무관.
        V63 이 확정: 거를 행은 버리지 말고 가중치를 낮춘다. 제거는 볼륨 비용을 문다.

    (2) TrackMan 물리 요약   V66 이 통과하면 (실행 결과에 따라 결정)

성분 분해를 유지하는 근거 (V65)
    같은 111피처로 control_success 를 직접 예측한 모델은 단독이 더 좋다(751.20 vs
    745.30). 그런데 결합은 +23.43 vs +41.04 로 크게 밀린다. 차이는 base 와의
    logit 상관뿐이다(0.8529 vs 0.8219). 분해는 정확도가 아니라 비상관성을 만든다.

사용법
    python v67_build034.py [--trackman]
        --trackman 를 주면 V66 의 tm500 피처를 성분 모델에 포함한다.

출력: submit/2026-08-15/submit_034.zip
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
MO = SJ / "experiment" / "model_optimization"
BASE_ZIP = SJ / "submit" / "2026-08-15" / "submit_032.zip"
OUT_ZIP = SJ / "submit" / "2026-08-15" / "submit_034.zip"
TM = MO / "trackman500_asof_train.parquet"
TMAN = MO / "trackman500_asof_manifest.json"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, F_WEIGHT, SHORT_W, SHORT_TH = 400, 0.20, 0.5, 0.5
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
USE_TM = "--trackman" in sys.argv
NL = chr(10)
EPS = 1e-7

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
pid = df["pitcher_id"].to_numpy()
NVOL = df["asof_pitcher_n"].to_numpy()
IS_F = df["game_type"].astype(str).to_numpy() == "F"

# ---------------------------------------------- (1) 짧은 등판 가중치
o = np.argsort(pid.astype(np.int64) * 10_000_000 + NVOL, kind="stable")
pv = df["asof_pitcher_prev1_game_success_rate"].to_numpy()[o]
gp = pid[o]
chg = np.r_[True, (gp[1:] != gp[:-1]) | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
outing = np.empty(len(df), dtype=np.int64)
outing[o] = np.cumsum(chg) - 1
od = pd.DataFrame({"outing": outing, "pid": pid, "inn": df["inning"].to_numpy()})
agg = od.groupby("outing").agg(n=("outing", "size"), pid=("pid", "first"),
                               first_inn=("inn", "min"))
agg["start"] = (agg["first_inn"] == 1).astype(int)
agg = agg.join(agg.groupby(["pid", "start"])["n"].median().rename("med"),
               on=["pid", "start"])
agg["ratio"] = agg["n"] / agg["med"].clip(lower=1)
RATIO = np.nan_to_num(agg["ratio"].reindex(outing).to_numpy(), nan=1.0)
SHORT = RATIO < SHORT_TH
ROW_W = np.where(IS_F, F_WEIGHT, 1.0) * np.where(SHORT, SHORT_W, 1.0)
print(f"등판 {len(agg):,}개  선발 중앙 {int(agg.loc[agg.start==1,'n'].median())}구  "
      f"구원 중앙 {int(agg.loc[agg.start==0,'n'].median())}구")
print(f"짧은 등판 {SHORT.mean()*100:.2f}% 에 학습 가중치 {SHORT_W}", flush=True)

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}

print(f"{NL}2019~2024 전체로 spec / platoon 생성", flush=True)
spec = CF.make_spec(df)
platoon = CF.make_platoon_table(df)
bat_platoon = CF.make_batter_platoon_table(df, LAB)
count_platoon = CF.make_count_platoon_table(df)
inning_platoon = CF.make_inning_platoon_table(df)
feat = CF.build(df[INPUT_COLS], spec, platoon, bat_platoon, count_platoon,
                inning_platoon)

TM_COLS = []
if USE_TM:
    fc = json.load(open(TMAN, encoding="utf-8"))["feature_columns"]
    TM_COLS = [c for c in fc if c.startswith("tm500_") and "between" not in c]
    tmv = df[["row_id"]].merge(
        pd.read_parquet(TM, columns=["row_id"] + TM_COLS), on="row_id",
        how="left")[TM_COLS].to_numpy(np.float64)
    tmv[IS_F] = np.nan                       # 규칙 N3
    for i, c in enumerate(TM_COLS):
        feat[c] = tmv[:, i]
    print(f"TrackMan 요약 {len(TM_COLS)}개 추가 (cw_* 는 규칙 N2 로 제외)", flush=True)

spec["columns"] = list(feat.columns)
spec["trackman_columns"] = TM_COLS
X = feat.to_numpy(np.float32)
print(f"  피처 {X.shape[1]}개", flush=True)


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


work = CACHE / "submit034_build"
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
    d_tr = xgb.DMatrix(X[m], label=arr[m], weight=ROW_W[m], missing=np.nan)
    for s in SEEDS:
        b = xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                      verbose_eval=False)
        fn = f"c34_{tag}_x{s}.ubj"
        b.save_model(str(work / fn))
        files.append(fn)
    p_tr = Pool(np.nan_to_num(X[m], nan=-999.0), arr[m], weight=ROW_W[m])
    for s in SEEDS:
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        fn = f"c34_{tag}_c{s}.cbm"
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
json.dump(spec, open(work / "c34_spec.json", "w"), indent=1, sort_keys=True)
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
assert script.count("c29_%s_x%d.ubj") == 1
script = script.replace("c29_%s_x%d.ubj", "c34_%s_x%d.ubj", 1)
script = script.replace("c29_%s_c%d.cbm", "c34_%s_c%d.cbm", 1)

if USE_TM:
    RET_OLD = '    return pd.DataFrame(out)[spec["columns"]].to_numpy(np.float32)'
    RET_NEW = NL.join([
        "    # TrackMan 요약. 투수-시즌 as-of 룩업의 단건 조인이라 행 독립성이 유지된다.",
        "    # 규칙 N3: game_type=F 행에는 적용하지 않는다.",
        '    tmc = list(spec["trackman_columns"])',
        "    if tmc:",
        '        tt = trackman.set_index("pitcher_id")[tmc]',
        '        j = tt.reindex(pd.to_numeric(frame["pitcher_id"],'
        ' errors="coerce").to_numpy())',
        '        isf = frame["game_type"].astype(str).to_numpy() == "F"',
        "        for c in tmc:",
        "            out[c] = np.where(isf, np.nan, j[c].to_numpy(dtype=float))",
        '    return pd.DataFrame(out)[spec["columns"]].to_numpy(np.float32)'])
    assert script.count(RET_OLD) == 1
    script = script.replace(RET_OLD, RET_NEW, 1)
    SIG_OLD = ("def component_features(frame, spec, platoon, bat_platoon, "
               "count_platoon,\n                       inning_platoon):")
    script = script.replace(SIG_OLD, SIG_OLD[:-2] + ", trackman):", 1)
    L_OLD = ("    matrix = component_features(test, spec, platoon, bat_platoon, "
             "count_platoon,\n                                inning_platoon)")
    L_NEW = (
        '    trackman = pd.read_csv(MODEL_DIR / cfg["trackman_file"])' + NL
        + "    matrix = component_features(test, spec, platoon, bat_platoon, "
          "count_platoon," + NL
        + "                                inning_platoon, trackman)" + NL
        + "    matrix_cat = np.nan_to_num(matrix, nan=-999.0)")
    assert script.count(L_OLD) == 1
    script = script.replace(L_OLD, L_NEW, 1)
    script = script.replace("            v = model.predict_proba(matrix)[:, 1]",
                            "            v = model.predict_proba(matrix_cat)[:, 1]", 1)

for tag in ["def component_features", "def component_blend",
            "prediction = component_blend"]:
    assert script.count(tag) == 1
assert "c29_" not in script
print(f"{NL}script.py 정리 완료  {len(script.splitlines())}줄", flush=True)

cb = metadata["component_blend"]
cb["spec_file"] = "c34_spec.json"
cb["model_prefix"] = "c34"
cb["training_sample_weights"] = {
    "game_type_F": F_WEIGHT,
    "short_outing": SHORT_W,
    "short_outing_definition": (
        "outing segmented by runs of constant asof_pitcher_prev1_game_success_rate "
        "along each pitcher's asof_pitcher_n order; short = pitches in outing below "
        "0.5 x that pitcher's median outing length, computed separately for starts "
        "(first inning 1) and relief appearances"),
    "short_outing_share": float(SHORT.mean()),
    "validation": {"fold_2023": {"d_bss": 2.21, "d_solo": 12.48},
                   "fold_2024": {"d_bss": 0.63, "d_solo": 2.33},
                   "note": "V64. training weights only - inference path unchanged, "
                           "so row independence is unaffected. V63 swept 45 filter "
                           "combinations and found dropping rows pays a volume cost "
                           "(V500_drop -7.33 at 20.78 percent removed) while "
                           "down-weighting does not"},
}
cb["excluded_tier_e"] = (
    "submit_033 added Tier E and scored 973.14 public versus submit_032 979. "
    "internally Tier E was positive on both folds (+3.93 / +2.47) but V61 "
    "established that internal deltas below about 3 carry no public signal, so "
    "there is no active reason to include it")
cb["decomposition_rationale"] = (
    "V65: a direct control_success model on the same 111 features scores higher "
    "standalone (751.20 versus 745.30 on 2024) but blends far worse (+23.43 versus "
    "+41.04). the only difference is logit correlation with the base (0.8529 versus "
    "0.8219). the failure decomposition buys decorrelation, not accuracy")
if USE_TM:
    cb["trackman_file"] = "trackman_2025.csv"
    cb["trackman_columns"] = TM_COLS
metadata["track"] = "reverse20_s0475_comp5_platoon4_cal_volw_shortw" + \
                    ("_tm" if USE_TM else "")
metadata["version"] = 21

for stale in [n for n in list(payload) if "/c29_" in n]:
    del payload[stale]
payload["script.py"] = script.encode("utf-8")
payload["model/metadata.json"] = json.dumps(metadata, indent=1,
                                            sort_keys=True).encode("utf-8")
extra_files = ["c34_spec.json", "platoon_2025.csv", "bat_platoon_2025.csv",
               "count_platoon_2025.csv", "inning_platoon_2025.csv"]
for fn in files + extra_files:
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
print(f"  성분 모델 {len(files)}개 (c34_)   TrackMan {'포함' if USE_TM else '미포함'}")
