"""V56: Tier E 조건 축 확장 — 아웃카운트·타석깊이·이닝 블록 추가.

근거
    09_TIER_E_RESULTS.md 의 다음 단계 5순위:
    "Tier E 조건 변수 확장 (아웃·주자·li 구간). 현재는 카운트·타자손 2축뿐.
     O2 상한 917.7 까지 여지"

    V53/V54 에서 기존 Tier E 가 성분 라인에서 두 fold 통과했다(2023 +3.93 /
    2024 +2.47). 성분단독은 755 이고 O2 oracle 상한은 917.7 이라 여지가 남아 있다.

설계에서 지킨 것 — V50 의 교훈
    축을 '곱하면' 셀이 얇아진다. 아웃을 기존 셀에 곱하면 36 -> 108 셀이 되고
    셀당 10~25구로 떨어진다. V50 에서 12셀 전문가가 무너진 것과 같은 경로다.

    그래서 곱하지 않고 '블록으로 병렬 추가'한다. 각 블록의 셀은 두껍게 유지된다.

        A (기존)  구종군 x 카운트버킷(6) x 타자손  =  36셀
        B (신규)  구종군 x 아웃(3)       x 타자손  =  18셀
        C (신규)  구종군 x 타석깊이(3)    x 타자손  =  18셀
        D (신규)  구종군 x 이닝군(3)      x 타자손  =  18셀

    주자 상태는 TrackMan 에 없어서(30개 컬럼 확인) 제외했다.

    타석깊이(pitch_of_pa)가 특히 값이 있다 — 메인 데이터에 없는 열이고,
    파울 때문에 볼카운트로 유도되지도 않는다. 6구 걸린 2-2 와 4구 만에 온 2-2 는
    다르다. "이 투수가 타석이 길어질 때 어떻게 반응하는가"는 새 정보다.

전처리는 tier_e_build.py 를 그대로 따른다 (04_PREPROCESSING_SPEC 준수).
    이상치 -> NaN / 구조 이상 행 제거 / 구종군 3종 / 좌우 미러 정규화 /
    season x 구종군 robust z (cutoff 미만 시즌 통계만) / 개인-리그 잔차 / EB 축소

출력: outputs/tier_e2/tier_e2_cutoff{2023,2024,2025}.parquet
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

SJ = Path(__file__).resolve().parents[2]
DATA = SJ / "data"
# 사용법: python v56_tiere_build2.py [블록조합] [SVD차원]
#   블록조합 예) abcd (기본), abc (이닝 제외), a (기존 Tier E 와 동일 축)
BLOCKS = (sys.argv[1] if len(sys.argv) > 1 else "abcd").lower()
DIM_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TAGNAME = f"tier_e2_{BLOCKS}{DIM_ARG}"
OUT = SJ / "claude" / "outputs" / TAGNAME
OUT.mkdir(parents=True, exist_ok=True)

TM_METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
              "extension", "rel_height", "rel_side", "zone_speed"]
GROUPS = ["fastball", "breaking", "offspeed"]
PHYS = ["rel_speed_rz", "induced_vert_break_rz", "rel_side_rz"]
CLIP = {"extension": (0.5, None), "rel_height": (0.8, None), "rel_speed": (90.0, None),
        "spin_rate": (800.0, None), "induced_vert_break": (-90.0, 90.0),
        "horz_break": (-90.0, 90.0)}
CUTOFFS = [2023, 2024, 2025]
MIN_PITCHES, EB_M = 300, 200.0


def load_trackman():
    t0 = time.time()
    tm = pd.read_csv(DATA / "trackman_history.csv")
    n0 = len(tm)
    bad = ((tm.balls_before > 3) | (tm.strikes_before > 2)
           | (tm.outs_before > 2) | (tm.inning < 1))
    tm = tm[~bad]
    tm = tm.drop_duplicates(subset=["trackman_game_id", "pitch_no"], keep="first")
    for c, (lo, hi) in CLIP.items():
        if lo is not None:
            tm.loc[tm[c] < lo, c] = np.nan
        if hi is not None:
            tm.loc[tm[c] > hi, c] = np.nan
    tm.loc[tm.zone_speed > tm.rel_speed, ["zone_speed", "rel_speed"]] = np.nan
    sign = np.where(tm.pitcher_hand.eq("Left"), -1.0, 1.0)
    tm["horz_break"] = tm.horz_break * sign
    tm["rel_side"] = tm.rel_side * sign
    tm["bhand"] = np.where(tm.batter_hand.eq("Left"), 1, 2)
    tm = tm[tm.pitch_type_group.isin(GROUPS)]
    print(f"trackman {n0:,} -> {len(tm):,}행  {time.time()-t0:.1f}s", flush=True)
    return tm


def robust_z(tm, cutoff):
    fit = tm[tm.season < cutoff]
    stats = (fit.groupby(["season", "pitch_type_group"])[TM_METRICS]
             .agg(["median", lambda s: s.quantile(.75) - s.quantile(.25)]))
    stats.columns = [f"{m}__{'med' if k == 'median' else 'iqr'}"
                     for m, k in stats.columns]
    tm = tm.merge(stats.reset_index(), on=["season", "pitch_type_group"], how="left")
    for c in TM_METRICS:
        iqr = tm[f"{c}__iqr"].clip(lower=1e-6) / 1.349
        tm[c + "_rz"] = (tm[c] - tm[f"{c}__med"]) / iqr
    return tm.drop(columns=[c for c in tm.columns if c.endswith(("__med", "__iqr"))])


def count_bucket(b, s):
    """볼카운트 6버킷 (tier_e_build.py 와 동일)."""
    ahead, even = s > b, s == b
    three_b, two_s = b == 3, s == 2
    out = np.zeros(len(b), dtype=np.int8)
    out[ahead & ~two_s] = 1
    out[two_s & ~three_b] = 2
    out[three_b & ~two_s] = 3
    out[three_b & two_s] = 4
    out[(~ahead) & (~even) & (~three_b)] = 5
    return out


KEY = ["pitcher_trackman_id", "season"]


def block(d, axis_col, tag, eb_m):
    """구종군 x axis x 타자손 셀에서 사용률 잔차와 물리 반응 잔차를 뽑는다."""
    cell = ["pitch_type_group", axis_col, "bhand"]
    ctx = [axis_col, "bhand"]

    cnt = d.groupby(KEY + cell).size().rename("n").reset_index()
    tot = d.groupby(KEY + ctx).size().rename("tot").reset_index()
    cnt = cnt.merge(tot, on=KEY + ctx)
    lg = (d.groupby(["season"] + cell).size().rename("ln").reset_index()
          .merge(d.groupby(["season"] + ctx).size().rename("ltot").reset_index(),
                 on=["season"] + ctx))
    lg["lg_rate"] = lg.ln / lg.ltot
    cnt = cnt.merge(lg[["season"] + cell + ["lg_rate"]], on=["season"] + cell,
                    how="left")
    cnt["rate"] = (cnt.n + eb_m * cnt.lg_rate) / (cnt.tot + eb_m)
    cnt["resid"] = cnt.rate - cnt.lg_rate
    cnt["feat"] = (f"u{tag}_" + cnt.pitch_type_group.astype(str) + "_a"
                   + cnt[axis_col].astype(str) + "_h" + cnt.bhand.astype(str))

    pm = d.groupby(KEY + cell)[PHYS].agg(["mean", "count"])
    pm.columns = [f"{a}|{b}" for a, b in pm.columns]
    pm = pm.reset_index()
    lgp = d.groupby(["season"] + cell)[PHYS].mean().add_suffix("|lg").reset_index()
    pm = pm.merge(lgp, on=["season"] + cell, how="left")
    prows = []
    for c in PHYS:
        nn = pm[f"{c}|count"].to_numpy()
        shr = nn / (nn + eb_m)
        prows.append(pd.DataFrame({
            **{k: pm[k] for k in KEY},
            "feat": (f"p{tag}_" + c.replace("_rz", "") + "_a"
                     + pm[axis_col].astype(str) + "_h" + pm.bhand.astype(str)
                     + "_" + pm.pitch_type_group.astype(str)),
            "resid": shr * (pm[f"{c}|mean"].to_numpy() - pm[f"{c}|lg"].to_numpy())}))
    return pd.concat([cnt[KEY + ["feat", "resid"]]] + prows, ignore_index=True)


def build_profile(tm, cutoff, min_pitches, eb_m):
    d = tm[tm.season < cutoff].copy()
    n_by = d.groupby(KEY).size().rename("n")
    keep = n_by[n_by >= min_pitches].reset_index()[KEY]
    d = d.merge(keep, on=KEY, how="inner")
    if d.empty:
        return None
    d["cb"] = count_bucket(d.balls_before.to_numpy(), d.strikes_before.to_numpy())
    d["ob"] = d.outs_before.astype(np.int8)
    d["pa"] = np.digitize(d.pitch_of_pa.to_numpy(), [2, 4]).astype(np.int8)
    d["ib"] = np.digitize(d.inning.to_numpy(), [4, 7]).astype(np.int8)

    parts, sizes = [], {}
    for axis, tag, name, key in [("cb", "", "A 카운트6", "a"), ("ob", "o", "B 아웃3", "b"),
                                 ("pa", "p", "C 타석깊이3", "c"), ("ib", "i", "D 이닝3", "d")]:
        if key not in BLOCKS:
            continue
        b = block(d, axis, tag, eb_m)
        parts.append(b)
        n_cell = d.groupby(KEY + ["pitch_type_group", axis, "bhand"]).size()
        sizes[name] = (int(b.feat.nunique()), int(n_cell.median()))
    long = pd.concat(parts, ignore_index=True).dropna(subset=["resid"])
    long = long.groupby(KEY + ["feat"], as_index=False)["resid"].mean()
    for k, (nf, med) in sizes.items():
        print(f"    {k:<12} 피처 {nf:>4}개   셀당 중앙 {med:>5}구", flush=True)
    return long


def to_matrix(long):
    rows = long.groupby(KEY).ngroup()
    feats = pd.Categorical(long.feat)
    M = sparse.csr_matrix((long.resid.to_numpy(), (rows.to_numpy(), feats.codes)),
                          shape=(rows.max() + 1, len(feats.categories)))
    idx = long.groupby(KEY, as_index=False).size()[KEY].reset_index(drop=True)
    return M, idx


tm_raw = load_trackman()
manifest = []
for cutoff in CUTOFFS:
    t0 = time.time()
    print(f"{chr(10)}cutoff {cutoff}  블록={BLOCKS} dim={DIM_ARG}", flush=True)
    long = build_profile(robust_z(tm_raw, cutoff), cutoff, MIN_PITCHES, EB_M)
    M, idx = to_matrix(long)
    k = min(DIM_ARG, M.shape[1] - 1, M.shape[0] - 1)
    svd = TruncatedSVD(n_components=k, random_state=0)
    Z = svd.fit_transform(M)
    for j in range(k):
        v = svd.components_[j]
        if v[np.argmax(np.abs(v))] < 0:
            Z[:, j] *= -1
            svd.components_[j] *= -1
    emb = pd.concat([idx, pd.DataFrame(Z, columns=[f"te_svd_{j:02d}"
                                                   for j in range(k)])], axis=1)
    emb["te_n_cells"] = long.groupby(KEY).size().to_numpy()
    emb.to_parquet(OUT / f"{TAGNAME}_cutoff{cutoff}.parquet", index=False)
    rec = dict(cutoff=cutoff, pitcher_seasons=int(M.shape[0]),
               features=int(M.shape[1]), svd_dim=k,
               explained=float(svd.explained_variance_ratio_.sum()),
               min_pitches=MIN_PITCHES, eb=EB_M, sec=round(time.time() - t0, 1))
    manifest.append(rec)
    print(f"  투수-시즌 {M.shape[0]}  피처 {M.shape[1]}  SVD {k}차원  "
          f"설명 {rec['explained']:.3f}  {rec['sec']}s", flush=True)
json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
print(f"{chr(10)}saved -> {OUT}")
