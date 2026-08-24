"""V24: submit_029.zip — 이닝별 플래툰 추가.

V22 (Val2024, w=0.25) 5개 후보 중 J_P_hi 가 최고 (+3.42)
V23 세 fold 검증 (w=0.25, submit_028 대비)
    2022  +50.92 -> +52.23   (+1.31)
    2023   +3.68 ->  +3.38   (-0.30)   2023 시그마 안, 사실상 평평
    2024  +36.27 -> +39.25   (+2.98)

    2 fold 양수. w=0.25 가 세 fold 모두 양수인 최대값이라는 조건도 유지된다.
    w=0.30 은 여전히 안 된다 - 기대값 0.83*(+3.24) + 0.17*(-22.15) = -1.08.

    split(p, h, inning) = EB(투수 x 타자손 x 이닝군) - EB(투수 x 타자손)
    카운트별(V19)과 같은 2단계 차감이다.

같이 시도했다가 기각 (V22)
    B_hc  타자 x 투수손 x 카운트  -1.97   V19 의 완전 대칭인데 실패.
                                        타자 쪽은 V12 에서 성분별까지 넣어놔 중복.
    J_all 5개 전부                +3.13   최고 단일(+3.42)보다 나쁘다. 과적합.

출력: submit/2026-08-15/submit_029.zip
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
BASE_ZIP = ROOT / "submit" / "2026-08-15" / "submit_028.zip"
OUT_ZIP = ROOT / "submit" / "2026-08-15" / "submit_029.zip"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS, W_BLEND = 400, 0.25
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
count_platoon = CF.make_count_platoon_table(df)
inning_platoon = CF.make_inning_platoon_table(df)
feat = CF.build(df[INPUT_COLS], spec, platoon, bat_platoon, count_platoon,
                inning_platoon)
spec["columns"] = list(feat.columns)
X = feat.to_numpy(np.float32)
print(f"  피처 {X.shape[1]}개  투수platoon {len(platoon)}행  타자platoon {len(bat_platoon)}행  카운트 {len(count_platoon)}  이닝 {len(inning_platoon)}", flush=True)

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


work = CACHE / "submit029_build"
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
        fn = f"c29_{tag}_x{s}.ubj"
        b.save_model(str(work / fn)); files.append(fn)
    p_tr = Pool(X[m], arr[m])
    for s in SEEDS:
        c = CatBoostClassifier(iterations=N_ROUNDS, depth=6, learning_rate=0.05,
                               l2_leaf_reg=6.0, loss_function="Logloss",
                               random_seed=s, task_type="GPU", verbose=0)
        c.fit(p_tr)
        fn = f"c29_{tag}_c{s}.cbm"
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
json.dump(spec, open(work / "c29_spec.json", "w"), indent=1, sort_keys=True)
platoon.to_csv(work / "platoon_2025.csv", index=False)
bat_platoon.to_csv(work / "bat_platoon_2025.csv", index=False)
count_platoon.to_csv(work / "count_platoon_2025.csv", index=False)
inning_platoon.to_csv(work / "inning_platoon_2025.csv", index=False)

INJECT = '''

C29_RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
             "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
             "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
             "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
C29_PREV_S = ["asof_pitcher_prev1_game_success_rate",
              "asof_pitcher_prev3_game_success_rate",
              "asof_pitcher_prev5_game_success_rate"]
C29_PREV_M = ["asof_pitcher_prev1_game_middle_rate",
              "asof_pitcher_prev3_game_middle_rate",
              "asof_pitcher_prev5_game_middle_rate"]
C29_CAT = ["top_bottom", "game_type", "base_state"]
C29_RATE_SPECS = [
    ("pitcher_success", "asof_pitcher_success_rate", "asof_pitcher_n"),
    ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("batter_success", "asof_batter_success_rate", "asof_batter_n"),
    ("batter_middle", "asof_batter_middle_rate", "asof_batter_n"),
]


def component_features(frame, spec, platoon, bat_platoon, count_platoon,
                       inning_platoon):
    pri = spec["priors"]
    st = float(spec["strength"])
    lam = float(spec["lam_prof"])
    out = {}
    for c in frame.columns:
        if c == "row_id":
            continue
        if c in C29_CAT:
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
    for name, rate_col, n_col in C29_RATE_SPECS:
        nn = out[n_col]
        rate = np.where(np.isnan(out[rate_col]), pri[name], out[rate_col])
        out[name + "_is_missing"] = np.isnan(out[rate_col]).astype(float)
        out[name + "_smoothed"] = (nn * rate + st * pri[name]) / (nn + st)
        out[name + "_reliability"] = nn / (nn + st)
    for c in C29_RATES:
        med = spec["rate_median"][c]
        r = np.where(np.isnan(out[c]), med, out[c])
        out["prof200_" + c] = (n * r + lam * med) / (n + lam)
    ps = {c: np.where(np.isnan(out[c]), spec["prev_median"][c], out[c])
          for c in C29_PREV_S + C29_PREV_M}
    out["prev_trend_s"] = ps[C29_PREV_S[0]] - ps[C29_PREV_S[2]]
    out["prev_trend_m"] = ps[C29_PREV_M[0]] - ps[C29_PREV_M[2]]
    out["prev_std_s"] = np.std(np.vstack([ps[c] for c in C29_PREV_S]), axis=0)
    out["prev_std_m"] = np.std(np.vstack([ps[c] for c in C29_PREV_M]), axis=0)
    out["prev_miss_cnt"] = sum(np.isnan(out[c]).astype(float)
                               for c in C29_PREV_S + C29_PREV_M)
    for k, (cs, cm) in enumerate(zip(C29_PREV_S, C29_PREV_M)):
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
    bb = pd.to_numeric(frame["balls_before"]).to_numpy()
    ss = pd.to_numeric(frame["strikes_before"]).to_numpy()
    cbk = np.where(ss > bb, 0, np.where(bb > ss, 2, 1))
    ckey = pd.MultiIndex.from_arrays([pd.to_numeric(frame["pitcher_id"]),
                                      pd.to_numeric(frame["batter_hand"]), cbk])
    ct = count_platoon.set_index(
        ["pitcher_id", "batter_hand", "count_bucket"]).reindex(ckey)
    csp = ct["count_platoon_split"].fillna(0.0).to_numpy(float)
    crel = ct["count_platoon_rel"].fillna(0.0).to_numpy(float)
    out["count_platoon_split"] = csp
    out["count_platoon_rel"] = crel
    out["count_platoon_w"] = csp * crel
    ibk = np.digitize(pd.to_numeric(frame["inning"]).to_numpy(), [4, 7, 10])
    ikey = pd.MultiIndex.from_arrays([pd.to_numeric(frame["pitcher_id"]),
                                      pd.to_numeric(frame["batter_hand"]), ibk])
    it = inning_platoon.set_index(
        ["pitcher_id", "batter_hand", "inning_bucket"]).reindex(ikey)
    isp = it["inning_platoon_split"].fillna(0.0).to_numpy(float)
    irel = it["inning_platoon_rel"].fillna(0.0).to_numpy(float)
    out["inning_platoon_split"] = isp
    out["inning_platoon_rel"] = irel
    out["inning_platoon_w"] = isp * irel
    return pd.DataFrame(out)[spec["columns"]].to_numpy(np.float32)


def component_blend(test, metadata, prediction):
    import xgboost as xgb
    from catboost import CatBoostClassifier

    cfg = metadata["component_blend"]
    spec = json.loads((MODEL_DIR / cfg["spec_file"]).read_text())
    platoon = pd.read_csv(MODEL_DIR / cfg["platoon_file"])
    bat_platoon = pd.read_csv(MODEL_DIR / cfg["bat_platoon_file"])
    count_platoon = pd.read_csv(MODEL_DIR / cfg["count_platoon_file"])
    inning_platoon = pd.read_csv(MODEL_DIR / cfg["inning_platoon_file"])
    matrix = component_features(test, spec, platoon, bat_platoon, count_platoon,
                                inning_platoon)
    dmat = xgb.DMatrix(matrix)
    parts = {}
    for tag in spec["components"]:
        total, count = None, 0
        for seed in spec["seeds"]:
            booster = xgb.Booster(params={"device": "cpu", "nthread": 6})
            booster.load_model(str(MODEL_DIR / ("c29_%s_x%d.ubj" % (tag, seed))))
            v = booster.predict(dmat)
            total = v if total is None else total + v
            count += 1
        for seed in spec["seeds"]:
            model = CatBoostClassifier()
            model.load_model(str(MODEL_DIR / ("c29_%s_c%d.cbm" % (tag, seed))))
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

metadata["track"] = "reverse20_s0475_comp5_uniform_platoon4"
metadata["version"] = 16
metadata["component_blend"] = {
    "spec_file": "c29_spec.json", "platoon_file": "platoon_2025.csv",
    "bat_platoon_file": "bat_platoon_2025.csv",
    "count_platoon_file": "count_platoon_2025.csv",
    "inning_platoon_file": "inning_platoon_2025.csv",
    "weight_by_game_type": {"R": W_BLEND, "F": W_BLEND},
    "weight_default": W_BLEND,
    "formula": "P(success) = 1 - (p_m + p_r - p_mr + p_ob + p_oz)",
    "components": COMPONENTS, "component_meta": meta_comp,
    "families": ["xgboost", "catboost"], "seeds": SEEDS, "n_rounds": N_ROUNDS,
    "platoon_k": CF.PLATOON_K,
    "label_source": "train.csv asof cumulative-count difference between consecutive "
                    "pitches of the same pitcher; training only, never at inference",
    "validation_val2024": {
        "base": "submit_021", "base_bss": 836.503, "delta_bss": 39.25,
        "t_row": 8.0, "prev_submit_027_delta": 23.986,
        "weight_policy": ("w uniform 0.25 = largest weight positive on all three "
                          "folds (2022 +50.92 / 2023 +3.68 / 2024 +36.27); "
                          "0.30 turns 2023 negative"),
        "ablation": {"platoon": 13.75, "outside_split": 2.64,
                     "f_row_weight": 4.07, "batter_platoon": 2.44, "count_platoon": 8.44, "inning_platoon": 2.98,
                     "catboost_family": 1.23, "per_component_params": 0.91},
        "seed_saturation": "8 vs 20 seeds differ by +0.05 BSS",
    },
}

payload["script.py"] = script.encode("utf-8")
payload["model/metadata.json"] = json.dumps(metadata, indent=1,
                                            sort_keys=True).encode("utf-8")
for fn in files + ["c29_spec.json", "platoon_2025.csv", "bat_platoon_2025.csv", "count_platoon_2025.csv", "inning_platoon_2025.csv"]:
    payload[f"model/{fn}"] = (work / fn).read_bytes()
for stale in [n for n in payload
              if "/c28_" in n or "/c27_" in n or "/comp_" in n]:
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
