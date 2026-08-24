"""P13-B: submit_024.zip 생성.

submit_022 (reverse scale 0.475, offset 없음) 를 base 로 삼고 성분 결합 계층을 추가한다.
  024 - 022 = 플래툰 성분 결합 효과
  023 - 022 = logit offset 효과
같은 기준점에서 두 축이 각각 분리된다.

산출물
  model/comp_{tag}_s{seed}.ubj   4성분 x 8시드 = 32개 (XGBoost 네이티브, pickle 없음)
  model/comp_spec.json           피처 상수 + 컬럼 순서 + base_score
  model/platoon_2025.csv         (pitcher_id, batter_hand) -> split, rel
  model/metadata.json            component_blend 블록 추가
  script.py                      피처 빌더 + 성분 혼합 함수 삽입

학습 데이터는 2019~2024 전체 (최종 추론 시즌 2025 직전까지).
"""
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, CACHE, BASE_PARAMS, load

ROOT = Path(__file__).resolve().parents[2]
BASE_ZIP = ROOT / "submit" / "2026-08-13" / "submit_022.zip"
OUT_DIR = ROOT / "submit" / "2026-08-14"
OUT_ZIP = OUT_DIR / "submit_024.zip"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
N_ROUNDS = 400
W_BLEND = 0.20
COMPONENTS = ["m", "r", "o", "mr"]
EPS = 1e-7

OUT_DIR.mkdir(parents=True, exist_ok=True)
if OUT_ZIP.exists():
    raise FileExistsError(f"이미 존재: {OUT_ZIP}")
assert len(OUT_ZIP.name) < 30, OUT_ZIP.name

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]

print("2019~2024 전체로 spec / platoon 생성", flush=True)
spec = CF.make_spec(df)
platoon = CF.make_platoon_table(df)
feat = CF.build(df[INPUT_COLS], spec, platoon)
spec["columns"] = list(feat.columns)
X = feat.to_numpy(np.float32)
print(f"  피처 {X.shape[1]}개  platoon {len(platoon)}행  "
      f"league_mean {platoon.attrs['league_mean']:.8f}", flush=True)

labels = {
    "m": np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan),
    "r": np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan),
    "o": np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan),
}
labels["mr"] = np.where(ok, (labels["m"] == 1) & (labels["r"] == 1), np.nan)


def extrap_2025(arr):
    m = ~np.isnan(arr)
    s = pd.Series(arr[m]).groupby(pd.Series(season[m])).mean().sort_index()
    last = float(s.iloc[-1])
    slope = (last - float(s.iloc[0])) / (float(s.index[-1]) - float(s.index[0]))
    return float(np.clip(last + slope, 0.005, 0.995))


work = CACHE / "submit024_build"
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)

base_scores, model_files = {}, []
for tag in COMPONENTS:
    arr = labels[tag]
    bs = extrap_2025(arr)
    base_scores[tag] = bs
    m = ~np.isnan(arr)
    d_tr = xgb.DMatrix(X[m], label=arr[m])
    for s in SEEDS:
        b = xgb.train({**BASE_PARAMS, "base_score": bs, "seed": s}, d_tr,
                      num_boost_round=N_ROUNDS, verbose_eval=False)
        fn = f"comp_{tag}_s{s}.ubj"
        b.save_model(str(work / fn))
        model_files.append(fn)
    print(f"  [{tag:>2}] base_score {bs:.8f}  학습행 {int(m.sum()):,}  "
          f"모델 {len(SEEDS)}개", flush=True)

spec["base_scores"] = base_scores
spec["w_blend"] = W_BLEND
spec["seeds"] = SEEDS
spec["components"] = COMPONENTS
json.dump(spec, open(work / "comp_spec.json", "w"), indent=1, sort_keys=True)
platoon.to_csv(work / "platoon_2025.csv", index=False)

# ------------------------------------------------------- script.py 패치
INJECT = '''

COMP_RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
              "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
              "asof_pitcher_strike_rate", "asof_pitcher_fastball_rate",
              "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
COMP_PREV_S = ["asof_pitcher_prev1_game_success_rate",
               "asof_pitcher_prev3_game_success_rate",
               "asof_pitcher_prev5_game_success_rate"]
COMP_PREV_M = ["asof_pitcher_prev1_game_middle_rate",
               "asof_pitcher_prev3_game_middle_rate",
               "asof_pitcher_prev5_game_middle_rate"]
COMP_CAT = ["top_bottom", "game_type", "base_state"]
COMP_RATE_SPECS = [
    ("pitcher_success", "asof_pitcher_success_rate", "asof_pitcher_n"),
    ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("batter_success", "asof_batter_success_rate", "asof_batter_n"),
    ("batter_middle", "asof_batter_middle_rate", "asof_batter_n"),
]


def component_features(frame, spec, platoon):
    pri = spec["priors"]
    st = float(spec["strength"])
    lam = float(spec["lam_prof"])
    out = {}
    for c in frame.columns:
        if c == "row_id":
            continue
        if c in COMP_CAT:
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
    for name, rate_col, n_col in COMP_RATE_SPECS:
        nn = out[n_col]
        rate = np.where(np.isnan(out[rate_col]), pri[name], out[rate_col])
        out[name + "_is_missing"] = np.isnan(out[rate_col]).astype(float)
        out[name + "_smoothed"] = (nn * rate + st * pri[name]) / (nn + st)
        out[name + "_reliability"] = nn / (nn + st)
    for c in COMP_RATES:
        med = spec["rate_median"][c]
        r = np.where(np.isnan(out[c]), med, out[c])
        out["prof200_" + c] = (n * r + lam * med) / (n + lam)
    ps = {c: np.where(np.isnan(out[c]), spec["prev_median"][c], out[c])
          for c in COMP_PREV_S + COMP_PREV_M}
    out["prev_trend_s"] = ps[COMP_PREV_S[0]] - ps[COMP_PREV_S[2]]
    out["prev_trend_m"] = ps[COMP_PREV_M[0]] - ps[COMP_PREV_M[2]]
    out["prev_std_s"] = np.std(np.vstack([ps[c] for c in COMP_PREV_S]), axis=0)
    out["prev_std_m"] = np.std(np.vstack([ps[c] for c in COMP_PREV_M]), axis=0)
    out["prev_miss_cnt"] = sum(np.isnan(out[c]).astype(float)
                               for c in COMP_PREV_S + COMP_PREV_M)
    for k, (cs, cm) in enumerate(zip(COMP_PREV_S, COMP_PREV_M)):
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
    return pd.DataFrame(out)[spec["columns"]].to_numpy(np.float32)


def component_blend(test, metadata, prediction):
    import xgboost as xgb

    cfg = metadata["component_blend"]
    spec = json.loads((MODEL_DIR / cfg["spec_file"]).read_text())
    platoon = pd.read_csv(MODEL_DIR / cfg["platoon_file"])
    matrix = xgb.DMatrix(component_features(test, spec, platoon))
    parts = {}
    for tag in spec["components"]:
        total = None
        for seed in spec["seeds"]:
            booster = xgb.Booster(params={"device": "cpu", "nthread": 6})
            booster.load_model(str(MODEL_DIR / ("comp_%s_s%d.ubj" % (tag, seed))))
            value = booster.predict(matrix)
            total = value if total is None else total + value
        parts[tag] = np.clip(total / len(spec["seeds"]), 1e-7, 1.0 - 1e-7)
    union = np.clip(1.0 - (parts["m"] + parts["r"] - parts["mr"] + parts["o"]),
                    1e-7, 1.0 - 1e-7)
    weight = float(cfg["weight"])
    mask = (test["game_type"].astype(str).to_numpy() == cfg["apply_game_type"])
    blended = prediction.copy()
    blended[mask] = weight * union[mask] + (1.0 - weight) * prediction[mask]
    return np.clip(blended, 1e-6, 1.0 - 1e-6)
'''

HOOK_OLD = '''    if len(prediction) != len(test) or not np.isfinite(prediction).all():'''
HOOK_NEW = '''    if metadata.get("component_blend"):
        prediction = component_blend(test, metadata, prediction)
    if len(prediction) != len(test) or not np.isfinite(prediction).all():'''

with ZipFile(BASE_ZIP) as z:
    script = z.read("script.py").decode("utf-8")
    metadata = json.loads(z.read("model/metadata.json"))
    members = [n for n in z.namelist() if not n.endswith("/")]
    payload = {n: z.read(n) for n in members}

assert HOOK_OLD in script, "훅 지점을 찾지 못했다"
assert script.count(HOOK_OLD) == 1, "훅 지점이 유일하지 않다"
marker = "\ndef main():"
assert script.count(marker) == 1
script = script.replace(marker, INJECT + marker, 1)
script = script.replace(HOOK_OLD, HOOK_NEW, 1)
if "import json" not in script.split("def ")[0]:
    raise RuntimeError("script.py 상단에 json import 없음 — 확인 필요")

metadata["track"] = "reverse20_s0475_component_platoon"
metadata["version"] = 11
metadata["component_blend"] = {
    "spec_file": "comp_spec.json",
    "platoon_file": "platoon_2025.csv",
    "weight": W_BLEND,
    "apply_game_type": "R",
    "formula": "P(success) = 1 - (p_middle + p_reverse - p_middle_and_reverse + p_outside)",
    "components": COMPONENTS,
    "seeds": SEEDS,
    "n_rounds": N_ROUNDS,
    "platoon_k": CF.PLATOON_K,
    "label_source": "train.csv asof cumulative-count difference between consecutive "
                    "pitches of the same pitcher; training only, never at inference",
    "validation_val2024": {
        "base": "submit_021", "base_bss": 836.503, "blended_bss": 850.255,
        "delta_bss": 13.752, "t_row": 3.648,
        "weight_policy": "fixed 0.20, not selected on validation",
    },
}

payload["script.py"] = script.encode("utf-8")
payload["model/metadata.json"] = json.dumps(metadata, indent=1,
                                            sort_keys=True).encode("utf-8")
for fn in model_files + ["comp_spec.json", "platoon_2025.csv"]:
    payload[f"model/{fn}"] = (work / fn).read_bytes()


def info(name):
    zi = ZipInfo(name)
    zi.date_time = (2026, 8, 14, 0, 0, 0)
    zi.compress_type = ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    return zi


with ZipFile(OUT_ZIP, "w", ZIP_DEFLATED) as z:
    for name in sorted(payload):
        assert "\\\\" not in name, name
        z.writestr(info(name), payload[name])

size = OUT_ZIP.stat().st_size / 2 ** 20
print(f"\n생성 완료 {OUT_ZIP}  {size:.1f}MB  파일 {len(payload)}개", flush=True)
roots = sorted({n.split("/")[0] for n in payload})
print(f"  ZIP 루트: {roots}")
print(f"  파일명 길이: {len(OUT_ZIP.name)}자")
