"""G단계 2차 전처리: 기존 캐시에 '타자 이력 확장' 피처를 추가해 features_g 저장.
(투수 창 계산과 동시 수행 시 3GB 초과 → 단계 분리)
실행: python prep_g.py
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
import pp_features as PP

t0 = time.time()
C = HERE / "cache"
meta = json.loads((C / "meta.json").read_text())
df = pd.read_parquet(C / "features.parquet")
print("loaded", df.shape)

df, bh = PP.add_batter_history(df)
df, wl = PP.add_workload(df)                      # H) 피로/워크로드
bte = meta.get("batter_te", "batter_te")
batter_all = meta["groups"]["batter_scout"] + bh + [bte]
g_feats = meta["full_feats"] + bh
h_feats = g_feats + wl                            # 워크로드까지 포함
no_batter = [c for c in g_feats if c not in set(batter_all)]

meta_g = dict(meta)
meta_g.update({"batter_hist": bh, "g_feats": g_feats, "workload": wl, "h_feats": h_feats,
               "batter_all": batter_all, "no_batter_feats": no_batter})
meta_g["groups"] = dict(meta["groups"])
meta_g["groups"]["batter_hist"] = bh; meta_g["groups"]["workload"] = wl

keep = sorted(set(["game_year","pitcher","batter","game_date","game_pk","is_csw","pitch_type",
                   "is_swing","is_whiff","is_called","is_take","in_zone","out_zone","p_throws"]
                  + h_feats))
keep = [c for c in keep if c in df.columns]
df[keep].to_parquet(C / "features_g.parquet", index=False)
(C / "meta_g.json").write_text(json.dumps(meta_g, ensure_ascii=False, indent=2))
print(f"saved features_g {len(df):,}행 × {len(keep)}열 | 타자이력 +{len(bh)} | 워크로드 +{len(wl)} "
      f"| g_feats {len(g_feats)} → h_feats {len(h_feats)} | {time.time()-t0:.1f}s")
