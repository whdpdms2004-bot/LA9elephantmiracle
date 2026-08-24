"""V55: submit_033.zip — Tier E 조건부 반응 프로파일을 성분 라인에 싣는다.

V53 (2024 선별)
    E0 현행           단독 745.17  ΔBSS +38.92
    E2 +TierE F행제외  단독 755.17  ΔBSS +41.56   (+10.00 / +2.63)
    E2 > E1 이라 규칙 N3(F행 제외)이 실측 확인됐다.

V54 (두 fold 확인, submit_032 구간 벡터 기준)
    fold   ΔBSS 대비   성분단독 대비   커버리지
    2023    +3.93      +14.30        65.4%
    2024    +2.47       +5.85        61.5%

    두 fold 모두 둘 다 양수다. 그리고 이번 세션에서 실패한 것들과 패턴이 반대다 —
    V34 수축·V38 세분화는 정상 연도에서 벌고 2023 에서 잃었는데, Tier E 는
    2023 에서 더 크게 번다. 레짐 변화에 강한 정보다.

    fold 2022 는 구조적으로 불가능하다. 규칙 N1 이 TrackMan 증거 시즌을 2022
    이후로 제한하는데 fold 2022 는 증거가 season < 2022 여야 해서 공집합이다.
    09_TIER_E_RESULTS.md 도 같은 이유로 월 블록 검증으로 대체했다.

왜 이제야 실리는가
    Tier E 는 2026-08-12 에 게이트를 통과했는데 어떤 제출본에도 안 들어갔다.
    submit_032 의 feature_sets 211개를 스캔하면 te/u_/p_ 계열이 0개다.
    그 문서의 '다음 단계 1순위'가 '211피처 파이프라인에서 재검증'이었고 미실행이었다.
    성분 라인에는 TrackMan 피처가 0개라 문서가 걱정한 중복이 아예 없다.

이 패키지
    submit_032 (구간 벡터 [.25 .25 .25 .30 .40], Public 979) 위에
    성분 모델 5종을 te_svd 12차원 포함해 전체 데이터로 재학습한다.
    추론용 룩업은 pitcher_id -> 12차원 하나뿐이고 조인이 단건이라 행 독립성이 유지된다.

    규칙 N1  증거 시즌 2022 이후만
    규칙 N3  game_type=F 행에는 NaN (미적용)
    규칙 N2  crosswalk 신뢰도 열은 투입하지 않는다

    XGBoost 는 NaN 을 그대로 받고 CatBoost 는 -999 로 채운다. V53/V54 에서
    검증한 구성 그대로다.

출력: submit/2026-08-15/submit_033.zip
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
OUT_ZIP = SJ / "submit" / "2026-08-15" / "submit_033.zip"
TE_DIR = SJ / "claude" / "outputs" / "tier_e"
CW_DIR = SJ / "experiment" / "pitcher_embedding" / "outputs" / "trackman500"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS = 400
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
MIN_EV_SEASON, HALF_LIFE, TARGET_SEASON = 2022, 2.0, 2025
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
IS_F = df["game_type"].astype(str).to_numpy() == "F"

# ---------------------------------------------- Tier E 프로파일
cw = pd.read_parquet(CW_DIR / f"cutoff_{TARGET_SEASON}" / "crosswalk.parquet")[
    ["pitcher_id", "pitcher_trackman_id"]]
emb = pd.read_parquet(TE_DIR / f"tier_e_cutoff{TARGET_SEASON}.parquet")
DIMS = [c for c in emb.columns if c.startswith("te_svd_")]
emb = emb[emb["season"] >= MIN_EV_SEASON]
link = cw.merge(emb, on="pitcher_trackman_id", how="inner").rename(
    columns={"season": "ev"})
print(f"crosswalk 투수 {cw.pitcher_id.nunique()}   차원 {len(DIMS)}   "
      f"증거 시즌 {sorted(link.ev.unique())}")


def asof_profile(target):
    past = link[link["ev"] < target]
    w = np.power(0.5, (target - past["ev"].to_numpy()) / HALF_LIFE)
    g = past.assign(_w=w, **{c: past[c].to_numpy() * w for c in DIMS}) \
            .groupby("pitcher_id", as_index=False).agg(
                _w=("_w", "sum"), **{c: (c, "sum") for c in DIMS})
    for c in DIMS:
        g[c] = g[c] / g["_w"].clip(lower=1e-9)
    return g.drop(columns="_w")


hist = pd.concat([asof_profile(S).assign(season=S)
                  for S in sorted(df["season"].unique())
                  if (link["ev"] < S).any()], ignore_index=True)
TE = df[["pitcher_id", "season"]].merge(hist, on=["pitcher_id", "season"], how="left")
TEV = TE[DIMS].to_numpy(np.float64)
TEV[IS_F] = np.nan                                   # 규칙 N3
lookup = asof_profile(TARGET_SEASON)
print(f"학습 커버리지 {np.isfinite(TEV[:, 0]).mean():.1%}   "
      f"2025 룩업 투수 {len(lookup)}명", flush=True)

print(f"{NL}2019~2024 전체로 spec / platoon 생성", flush=True)
spec = CF.make_spec(df)
platoon = CF.make_platoon_table(df)
_ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
_yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
_yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
_yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": _ym, "r": _yr,
       "mr": np.where(ok, (_ym == 1) & (_yr == 1), np.nan),
       "ob": np.where(ok, (_yo == 1) & (_yb == 1), np.nan),
       "oz": np.where(ok, (_yo == 1) & (_yb == 0), np.nan)}
bat_platoon = CF.make_batter_platoon_table(df, LAB)
count_platoon = CF.make_count_platoon_table(df)
inning_platoon = CF.make_inning_platoon_table(df)
feat = CF.build(df[INPUT_COLS], spec, platoon, bat_platoon, count_platoon,
                inning_platoon)
for i, c in enumerate(DIMS):
    feat[c] = TEV[:, i]
spec["columns"] = list(feat.columns)
spec["tier_e_dims"] = DIMS
X = feat.to_numpy(np.float32)
print(f"  피처 {X.shape[1]}개 (기존 111 + TierE {len(DIMS)})", flush=True)


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


work = CACHE / "submit033_build"
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
    d_tr = xgb.DMatrix(X[m], label=arr[m], missing=np.nan)
    for s in SEEDS:
        b = xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                      verbose_eval=False)
        fn = f"c33_{tag}_x{s}.ubj"
        b.save_model(str(work / fn))
        files.append(fn)
    p_tr = Pool(np.nan_to_num(X[m], nan=-999.0), arr[m])
    for s in SEEDS:
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        fn = f"c33_{tag}_c{s}.cbm"
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
json.dump(spec, open(work / "c33_spec.json", "w"), indent=1, sort_keys=True)
platoon.to_csv(work / "platoon_2025.csv", index=False)
bat_platoon.to_csv(work / "bat_platoon_2025.csv", index=False)
count_platoon.to_csv(work / "count_platoon_2025.csv", index=False)
inning_platoon.to_csv(work / "inning_platoon_2025.csv", index=False)
lookup.to_csv(work / "tier_e_2025.csv", index=False)

# ---------------------------------------------- script.py 수술
with ZipFile(BASE_ZIP) as z:
    script = z.read("script.py").decode("utf-8")
    metadata = json.loads(z.read("model/metadata.json"))
    payload = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}

for tag in ["def component_features", "def component_blend",
            "prediction = component_blend"]:
    assert script.count(tag) == 1, f"'{tag}' {script.count(tag)}회"

SIG_OLD = NL.join([
    "def component_features(frame, spec, platoon, bat_platoon, count_platoon,",
    "                       inning_platoon):"])
SIG_NEW = NL.join([
    "def component_features(frame, spec, platoon, bat_platoon, count_platoon,",
    "                       inning_platoon, tier_e):"])
assert script.count(SIG_OLD) == 1
script = script.replace(SIG_OLD, SIG_NEW, 1)

RET_OLD = '    return pd.DataFrame(out)[spec["columns"]].to_numpy(np.float32)'
RET_NEW = NL.join([
    "    # Tier E 조건부 반응 프로파일. pitcher_id 단건 조인이라 행 독립성이 유지된다.",
    "    # 규칙 N3: game_type=F 행에는 적용하지 않는다 (NaN).",
    '    dims = list(spec["tier_e_dims"])',
    '    te = tier_e.set_index("pitcher_id")[dims]',
    '    joined = te.reindex(pd.to_numeric(frame["pitcher_id"],'
    ' errors="coerce").to_numpy())',
    '    is_f = frame["game_type"].astype(str).to_numpy() == "F"',
    "    for c in dims:",
    "        v = joined[c].to_numpy(dtype=float)",
    "        out[c] = np.where(is_f, np.nan, v)",
    '    return pd.DataFrame(out)[spec["columns"]].to_numpy(np.float32)'])
assert script.count(RET_OLD) == 1
script = script.replace(RET_OLD, RET_NEW, 1)

LOAD_OLD = NL.join([
    '    inning_platoon = pd.read_csv(MODEL_DIR / cfg["inning_platoon_file"])',
    "    matrix = component_features(test, spec, platoon, bat_platoon, count_platoon,",
    "                                inning_platoon)"])
LOAD_NEW = NL.join([
    '    inning_platoon = pd.read_csv(MODEL_DIR / cfg["inning_platoon_file"])',
    '    tier_e = pd.read_csv(MODEL_DIR / cfg["tier_e_file"])',
    "    matrix = component_features(test, spec, platoon, bat_platoon, count_platoon,",
    "                                inning_platoon, tier_e)",
    "    matrix_cat = np.nan_to_num(matrix, nan=-999.0)"])
assert script.count(LOAD_OLD) == 1
script = script.replace(LOAD_OLD, LOAD_NEW, 1)

script = script.replace("c29_%s_x%d.ubj", "c33_%s_x%d.ubj", 1)
script = script.replace("c29_%s_c%d.cbm", "c33_%s_c%d.cbm", 1)
CAT_OLD = "            v = model.predict_proba(matrix)[:, 1]"
assert script.count(CAT_OLD) == 1
script = script.replace(CAT_OLD, "            v = model.predict_proba(matrix_cat)[:, 1]", 1)

for tag in ["def component_features", "def component_blend",
            "prediction = component_blend", "tier_e_file", "matrix_cat ="]:
    assert script.count(tag) == 1, f"'{tag}' {script.count(tag)}회"
assert "c29_" not in script
print(f"{NL}script.py 수술 완료  {len(script.splitlines())}줄", flush=True)

cb = metadata["component_blend"]
cb["spec_file"] = "c33_spec.json"
cb["tier_e_file"] = "tier_e_2025.csv"
cb["model_prefix"] = "c33"
cb["tier_e"] = {
    "source": "claude/outputs/tier_e/tier_e_cutoff2025.parquet + "
              "pitcher_embedding/outputs/trackman500/cutoff_2025/crosswalk.parquet",
    "dims": DIMS,
    "evidence_seasons": sorted(int(v) for v in link["ev"].unique()),
    "min_evidence_season": MIN_EV_SEASON,
    "recency_half_life_seasons": HALF_LIFE,
    "pitchers_in_lookup": int(len(lookup)),
    "train_row_coverage": float(np.isfinite(TEV[:, 0]).mean()),
    "rules": ["N1 evidence seasons restricted to 2022 and later",
              "N2 crosswalk confidence columns are not fed to the models",
              "N3 not applied to game_type F rows"],
    "validation": {"fold_2023": {"d_bss": 3.93, "d_solo": 14.30, "coverage": 0.654},
                   "fold_2024": {"d_bss": 2.47, "d_solo": 5.85, "coverage": 0.615},
                   "fold_2022": "structurally impossible under rule N1",
                   "note": "both folds positive on both metrics. unlike every other "
                           "candidate this session, the gain is larger on 2023 (the "
                           "regime-break fold) than on 2024"},
    "history": "Tier E passed its gate on 2026-08-12 (F24 +19.58 R-only, the first "
               "meaningful AUC movement in the project) but was never integrated "
               "into any submission. the component line carries zero TrackMan "
               "features, so the overlap that 09_TIER_E_RESULTS.md worried about "
               "does not exist here",
}
metadata["track"] = "reverse20_s0475_comp5_platoon4_cal_volw_tiere"
metadata["version"] = 20

for stale in [n for n in list(payload) if "/c29_" in n]:
    del payload[stale]
payload["script.py"] = script.encode("utf-8")
payload["model/metadata.json"] = json.dumps(metadata, indent=1,
                                            sort_keys=True).encode("utf-8")
for fn in files + ["c33_spec.json", "platoon_2025.csv", "bat_platoon_2025.csv",
                   "count_platoon_2025.csv", "inning_platoon_2025.csv",
                   "tier_e_2025.csv"]:
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
print(f"  ZIP 루트: {sorted({n.split('/')[0] for n in payload})}")
print(f"  성분 모델 {len(files)}개 (c33_), Tier E 룩업 {len(lookup)}행 x {len(DIMS)}차원")
