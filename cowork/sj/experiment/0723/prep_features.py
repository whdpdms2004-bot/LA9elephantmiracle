"""피처 1회 생성 후 캐시 저장 (노트북/실험이 즉시 로드). 실행: python prep_features.py"""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
import csw_pipeline as P
import pp_features as PP

TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 40
t0 = time.time()
df = P.load_subset(top_pitchers=TOP)
df, base = P.add_base_features(df)
df, ppg = PP.add_all(df)                       # Pitch Predict 차용 피처
df, feats = P.add_history_features(df)
pte = P.add_target_encoding(df, "pitcher")
bte = P.add_target_encoding(df, "batter")

ph = feats["pitcher_hist"]
groups = {
    "situation": base["groups"]["situation"],
    "count_only": base["groups"]["count_only"],
    "basic_history": base["groups"]["basic_history"],
    "ids": [pte, bte],
    # 창별로 세분화(기여도 해석용)
    "hist_recent": [c for c in ph if c.endswith(("_day", "_2w")) or "_l100" in c],
    "hist_season": [c for c in ph if c.endswith(("_szn", "_pszn"))],
    "hist_career": [c for c in ph if c.endswith("_car") or "_l500" in c],
    "pitcher_hist": ph,
    "arsenal": feats["arsenal"],
    "release_rep": feats["release_rep"],
    **ppg,                                      # batter_scout / gameflow / prior_ab / lineup
}
# 피처셋 정의
basic_feats = base["base_num"] + base["base_cat"] + groups["ids"]
derived_feats = basic_feats + groups["pitcher_hist"] + groups["arsenal"] + groups["release_rep"]
pp_feats = sum([ppg[k] for k in ["batter_scout", "gameflow", "prior_ab", "lineup"]], [])
full_feats = derived_feats + pp_feats

keep = sorted(set(["game_year","pitcher","batter","game_date","game_pk","is_csw","pitch_type",
                   "is_swing","is_whiff","is_called","is_take","in_zone","out_zone","p_throws"] +
                  base["base_num"] + base["base_cat"] + groups["ids"] +
                  groups["pitcher_hist"] + groups["arsenal"] + groups["release_rep"] + pp_feats))
cache = HERE / "cache"; cache.mkdir(exist_ok=True)
df[keep].to_parquet(cache / "features.parquet", index=False)
meta = {"top_pitchers": TOP, "groups": groups,
        "basic_feats": basic_feats, "derived_feats": derived_feats,
        "pp_feats": pp_feats, "full_feats": full_feats, "batter_te": bte,
        "n_rows": int(len(df)),
        "n_train": int(df["game_year"].isin(P.TRAIN_YEARS).sum()),
        "n_test": int(df["game_year"].isin(P.TEST_YEARS).sum())}
(cache / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print(f"cached {len(df):,} rows x {len(keep)} cols | basic {len(basic_feats)} / derived {len(derived_feats)} feats | {time.time()-t0:.1f}s")
print("train", meta["n_train"], "test", meta["n_test"], "csw", round(float(df.is_csw.mean()),4))
