"""V13: submit_027.zip — 타자 플래툰 스플릿 추가 (V12 G4).

V12 실측 (Val2024, 균일 w=0.20, 8시드, 프로덕션 836.503 대비)
    G0_pitcher_only      +21.55  단독 748.40  corr 0.8655   <- submit_026
    G1_bat_platoon       +22.77  단독 751.16  corr 0.8651
    G4_bat_comp_platoon  +23.99  단독 747.47  corr 0.8576   <- 채택, t_row 5.93
    G2_bat_profile       +13.26  단독 626.50                 <- 붕괴
    G3_bat_both          +13.19  단독 617.49                 <- 붕괴

타자 플래툰 산포는 투수의 73%다 (sd 0.01300 vs 0.01775, 신뢰행 92.1%).
팀 문서 전체에 타자 플래툰이 없었다 - 투수 쪽만 5명 중 4명이 독립 발견했다.

비대칭이 흥미롭다
    V8  투수 성분별 플래툰  +0.16  기각  (전역 플래툰이 이미 있어 새 정보가 없다)
    V12 타자 성분별 플래툰  +1.22  채택  (전역조차 없던 상태라 그대로 새 정보)

G2/G3 붕괴는 이미 아는 함정이다. 타자별 성분 발생률에서 리그평균만 빼면
그 값은 사실상 타자 주효과 그 자체라 정적 테이블에서 자기 라벨이 샌다.
V1 의 V4_static_level(705.7 -> 187.5)과 같은 실패다. 스플릿(차이)만 쓴다.

026 대비 변경: 피처 97 -> 105, 성분 모델 80개 재학습, 타자 플래툰 룩업 추가.
출력: submit/2026-08-15/submit_027.zip
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

ROOT = Path(__file__).resolve().parents[2]
BASE_ZIP = ROOT / "submit" / "2026-08-15" / "submit_026.zip"
OUT_ZIP = ROOT / "submit" / "2026-08-15" / "submit_027.zip"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W_BLEND = 400, 0.20
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
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

print("2019~2024 전체로 spec / platoon 생성", flush=True)
spec = CF.make_spec(df)
platoon = CF.make_platoon_table(df)
_ok = df["label_ok"].to_numpy() == 1
_ym = np.where(_ok, df["y_middle"].to_numpy(np.float64), np.nan)
_yr = np.where(_ok, df["y_reverse"].to_numpy(np.float64), np.nan)
_yo = np.where(_ok, df["y_outside"].to_numpy(np.float64), np.nan)
_yb = np.where(_ok, df["y_ball"].to_numpy(np.float64), np.nan)
BAT_LAB = {"m": _ym, "r": _yr,
           "mr": np.where(_ok, (_ym == 1) & (_yr == 1), np.nan),
           "ob": np.where(_ok, (_yo == 1) & (_yb == 1), np.nan),
           "oz": np.where(_ok, (_yo == 1) & (_yb == 0), np.nan)}
bat_platoon = CF.make_batter_platoon_table(df, BAT_LAB)
feat = CF.build(df[INPUT_COLS], spec, platoon, bat_platoon)
spec["columns"] = list(feat.columns)
X = feat.to_numpy(np.float32)
print(f"  피처 {X.shape[1]}개  투수platoon {len(platoon)}행  타자platoon {len(bat_platoon)}행", flush=True)

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)
LAB = {"m": ym, "r": yr,
       "mr": np.where(ok, (ym == 1) & (yr == 1), np.nan),
       "ob": np.where(ok, (yo == 1) & (yb == 1), np.nan),
       "oz": np.where(ok, (yo == 1) & (yb == 0), np.nan)}


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


work = CACHE / "submit027_build"
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)

meta_comp, files = {}, []
for tag in COMPONENTS:
    arr = LAB[tag]
    m = ~np.isnan(arr)
    rate = float(np.nanmean(arr))
    bs = extrap_2025(arr)
    extra = params_for(rate)
    prm = {**BASE_PARAMS, "base_score": bs, **extra}
    d_tr = xgb.DMatrix(X[m], label=arr[m])
    for s in SEEDS:
        b = xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                      verbose_eval=False)
        fn = f"c27_{tag}_x{s}.ubj"
        b.save_model(str(work / fn)); files.append(fn)
    p_tr = Pool(X[m], arr[m])
    for s in SEEDS:
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        fn = f"c27_{tag}_c{s}.cbm"
        c.save_model(str(work / fn)); files.append(fn)
    meta_comp[tag] = {"base_rate": rate, "base_score": bs, "xgb_params": extra,
                      "n_train": int(m.sum())}
    print(f"  [{tag:>2}] rate {rate:.6f}  base_score {bs:.8f}  "
          f"leaves {extra['max_leaves']}  모델 {2*len(SEEDS)}개", flush=True)

spec["base_scores"] = {k: v["base_score"] for k, v in meta_comp.items()}
spec["xgb_params"] = {k: v["xgb_params"] for k, v in meta_comp.items()}
spec["w_blend"] = W_BLEND
spec["seeds"] = SEEDS
spec["components"] = COMPONENTS
json.dump(spec, open(work / "c27_spec.json", "w"), indent=1, sort_keys=True)
platoon.to_csv(work / "platoon_2025.csv", index=False)
bat_platoon.to_csv(work / "bat_platoon_2025.csv", index=False)

INJECT = '''

C27_RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
             "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
             "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
             "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
C27_PREV_S = ["asof_pitcher_prev1_game_success_rate",
              "asof_pitcher_prev3_game_success_rate",
              "asof_pitcher_prev5_game_success_rate"]
C27_PREV_M = ["asof_pitcher_prev1_game_middle_rate",
              "asof_pitcher_prev3_game_middle_rate",
              "asof_pitcher_prev5_game_middle_rate"]
C27_CAT = ["top_bottom", "game_type", "base_state"]
C27_RATE_SPECS = [
    ("pitcher_success", "asof_pitcher_success_rate", "asof_pitcher_n"),
    ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("batter_success", "asof_batter_success_rate", "asof_batter_n"),
    ("batter_middle", "asof_batter_middle_rate", "asof_batter_n"),
]


def component_features(frame, spec, platoon, bat_platoon):
    pri = spec["priors"]
    st = float(spec["strength"])
    lam = float(spec["lam_prof"])
    out = {}
    for c in frame.columns:
        if c == "row_id":
            continue
        if c in C27_CAT:
            out[c] = (frame[c].astype(str).map(spec["cat_map"][c])
                      .fillna(-1).to_numpy(float))
        else:
            out[c] = pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
    n = out["asof_pitcher_n"]
    out["count_state"] = out["balls_before"] * 3 + out["strikes_before"]
    out["handedness_matchup"] = out["pitcher_hand"] * 2 + out["batter_hand"]
    out["runner_out_state"] = out["num_runners_on"] * 3 + out["outs_before"]
    out["score_abs"] = np.abs(out["score_diff_pitcher_team"])
    out["late_inning"] = (out["inning"] >= 7).astype(float)
    out["high_leverage"] = (out["li"] >= 2).astype(float)
    out["log1p_asof_pitcher_n"] = np.log1p(n)
    out["log1p_asof_batter_n"] = np.log1p(out["asof_batter_n"])
    for k in (1, 3, 5):
        out["pitcher_success_delta_prev%d" % k] = (
            out["asof_pitcher_prev%d_game_success_rate" % k]
            - out["asof_pitcher_success_rate"])
        out["pitcher_middle_delta_prev%d" % k] = (
            out["asof_pitcher_prev%d_game_middle_rate" % k]
            - out["asof_pitcher_middle_rate"])
    out["ball_strike_gap"] = (out["asof_pitcher_ball_rate"]
                              - out["asof_pitcher_strike_rate"])
    for name, rate_col, n_col in C27_RATE_SPECS:
        nn = out[n_col]
        rate = np.where(np.isnan(out[rate_col]), pri[name], out[rate_col])
        out[name + "_is_missing"] = np.isnan(out[rate_col]).astype(float)
        out[name + "_smoothed"] = (nn * rate + st * pri[name]) / (nn + st)
        out[name + "_reliability"] = nn / (nn + st)
    for c in C27_RATES:
        med = spec["rate_median"][c]
        r = np.where(np.isnan(out[c]), med, out[c])
        out["prof200_" + c] = (n * r + lam * med) / (n + lam)
    ps = {c: np.where(np.isnan(out[c]), spec["prev_median"][c], out[c])
          for c in C27_PREV_S + C27_PREV_M}
    out["prev_trend_s"] = ps[C27_PREV_S[0]] - ps[C27_PREV_S[2]]
    out["prev_trend_m"] = ps[C27_PREV_M[0]] - ps[C27_PREV_M[2]]
    out["prev_std_s"] = np.std(np.vstack([ps[c] for c in C27_PREV_S]), axis=0)
    out["prev_std_m"] = np.std(np.vstack([ps[c] for c in C27_PREV_M]), axis=0)
    out["prev_miss_cnt"] = sum(np.isnan(out[c]).astype(float)
                               for c in C27_PREV_S + C27_PREV_M)
    for k, (cs, cm) in enumerate(zip(C27_PREV_S, C27_PREV_M)):
        out["faildir_%d" % k] = ps[cm] - (1 - ps[cs])
    out["rel200"] = n / (n + lam)
    key = pd.MultiIndex.from_arrays([pd.to_numeric(frame["pitcher_id"]),
                                     pd.to_numeric(frame["batter_hand"])])
    pt = platoon.set_index(["pitcher_id", "batter_hand"]).reindex(key)
    sp = pt["platoon_split"].fillna(0.0).to_numpy(float)
    rel = pt["platoon_rel"].fillna(0.0).to_numpy(float)
    out["platoon_split"] = sp
    out["platoon_rel"] = rel
    out["platoon_split_w"] = sp * rel
    bkey = pd.MultiIndex.from_arrays([pd.to_numeric(frame["batter_id"]),
                                      pd.to_numeric(frame["pitcher_hand"])])
    bt = bat_platoon.set_index(["batter_id", "pitcher_hand"]).reindex(bkey)
    bsp = bt["bat_platoon_split"].fillna(0.0).to_numpy(float)
    brel = bt["bat_platoon_rel"].fillna(0.0).to_numpy(float)
    out["bat_platoon_split"] = bsp
    out["bat_platoon_rel"] = brel
    out["bat_platoon_split_w"] = bsp * brel
    for tag in ["m", "r", "mr", "ob", "oz"]:
        out["bat_pl_" + tag] = bt["bat_pl_" + tag].fillna(0.0).to_numpy(float)
    return pd.DataFrame(out)[spec["columns"]].to_numpy(np.float32)


def component_blend(test, metadata, prediction):
    import xgboost as xgb
    from catboost import CatBoostClassifier

    cfg = metadata["component_blend"]
    spec = json.loads((MODEL_DIR / cfg["spec_file"]).read_text())
    platoon = pd.read_csv(MODEL_DIR / cfg["platoon_file"])
    bat_platoon = pd.read_csv(MODEL_DIR / cfg["bat_platoon_file"])
    matrix = component_features(test, spec, platoon, bat_platoon)
    dmat = xgb.DMatrix(matrix)
    parts = {}
    for tag in spec["components"]:
        total, count = None, 0
        for seed in spec["seeds"]:
            booster = xgb.Booster(params={"device": "cpu", "nthread": 6})
            booster.load_model(str(MODEL_DIR / ("c27_%s_x%d.ubj" % (tag, seed))))
            v = booster.predict(dmat)
            total = v if total is None else total + v
            count += 1
        for seed in spec["seeds"]:
            model = CatBoostClassifier()
            model.load_model(str(MODEL_DIR / ("c27_%s_c%d.cbm" % (tag, seed))))
            v = model.predict_proba(matrix)[:, 1]
            total = total + v
            count += 1
        parts[tag] = np.clip(total / count, 1e-7, 1.0 - 1e-7)
    union = np.clip(1.0 - (parts["m"] + parts["r"] - parts["mr"]
                           + parts["ob"] + parts["oz"]), 1e-7, 1.0 - 1e-7)
    weights = cfg["weight_by_game_type"]
    default_weight = float(cfg["weight_default"])
    game_type = test["game_type"].astype(str).to_numpy()
    weight = np.full(len(game_type), default_weight, dtype=float)
    for key, value in weights.items():
        weight[game_type == key] = float(value)
    blended = weight * union + (1.0 - weight) * prediction
    return np.clip(blended, 1e-6, 1.0 - 1e-6)
'''

HOOK_OLD = "    if len(prediction) != len(test) or not np.isfinite(prediction).all():"
HOOK_NEW = ("    if metadata.get(\"component_blend\"):\n"
            "        prediction = component_blend(test, metadata, prediction)\n"
            + HOOK_OLD)

with ZipFile(BASE_ZIP) as z:
    script = z.read("script.py").decode("utf-8")
    metadata = json.loads(z.read("model/metadata.json"))
    payload = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}

assert script.count(HOOK_OLD) == 1 and script.count("\ndef main():") == 1
script = script.replace("\ndef main():", INJECT + "\ndef main():", 1)
script = script.replace(HOOK_OLD, HOOK_NEW, 1)

metadata["track"] = "reverse20_s0475_comp5_uniform_batplatoon"
metadata["version"] = 14
metadata["component_blend"] = {
    "spec_file": "c27_spec.json", "platoon_file": "platoon_2025.csv",
    "bat_platoon_file": "bat_platoon_2025.csv",
    "weight_by_game_type": {"R": W_BLEND, "F": W_BLEND},
    "weight_default": W_BLEND,
    "formula": "P(success) = 1 - (p_m + p_r - p_mr + p_ob + p_oz)",
    "components": COMPONENTS, "component_meta": meta_comp,
    "families": ["xgboost", "catboost"], "seeds": SEEDS, "n_rounds": N_ROUNDS,
    "platoon_k": CF.PLATOON_K,
    "label_source": "train.csv asof cumulative-count difference between consecutive "
                    "pitches of the same pitcher; training only, never at inference",
    "validation_val2024": {
        "base": "submit_021", "base_bss": 836.503, "delta_bss": 23.986,
        "t_row": 5.93, "prev_submit_026_delta": 21.547,
        "weight_policy": ("w uniform 0.20 pre-registered; F treated same as R "
                          "since the 5-component split reversed F damage"),
        "ablation": {"platoon": 13.75, "outside_split": 2.64,
                     "f_row_weight": 4.07, "batter_platoon": 2.44,
                     "catboost_family": 1.23, "per_component_params": 0.91},
        "seed_saturation": "8 vs 20 seeds differ by +0.05 BSS",
    },
}

payload["script.py"] = script.encode("utf-8")
payload["model/metadata.json"] = json.dumps(metadata, indent=1,
                                            sort_keys=True).encode("utf-8")
for fn in files + ["c27_spec.json", "platoon_2025.csv", "bat_platoon_2025.csv"]:
    payload[f"model/{fn}"] = (work / fn).read_bytes()
for stale in [n for n in payload
              if "/c25_" in n or "/comp_" in n]:
    del payload[stale]                       # 이전 세대 성분 모델 제거


def info(name):
    zi = ZipInfo(name)
    zi.date_time = (2026, 8, 14, 0, 0, 0)
    zi.compress_type = ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    return zi


OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(OUT_ZIP, "w", ZIP_DEFLATED) as z:
    for name in sorted(payload):
        assert "\\" not in name
        z.writestr(info(name), payload[name])
print(f"\n생성 완료 {OUT_ZIP}  {OUT_ZIP.stat().st_size/2**20:.1f}MB  "
      f"파일 {len(payload)}개  이름 {len(OUT_ZIP.name)}자")
print(f"  ZIP 루트: {sorted({n.split('/')[0] for n in payload})}")
