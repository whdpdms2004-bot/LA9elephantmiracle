"""Tier E: TrackMan 조건부 반응 프로파일.

근거 (P0-4):
  oracle (투수,시즌)              -> 2024 BSS 720.9  <- 이미 팀의 단일 모델(784.6)이 넘어섬
  oracle (투수,시즌,타자손)        -> 2024 BSS 917.7  <- 현재 최고 815.08 대비 +100 여지
  BASE + oracle(투수,시즌,카운트,타자손) -> 864.6 (BASE 567.0 대비 +297.6)
따라서 가치는 '투수를 더 잘 표현'이 아니라 '투수 x 상황 조건부'에 있다.

만드는 것: 투수 x (구종군 x 볼카운트버킷 x 타자손) 사용률 + 물리량 반응 -> SVD 압축
전처리 (04_PREPROCESSING_SPEC 준수):
  1) 이상치 -> NaN, 구조 이상 행 제거
  2) pitch_type_group 4->3 (other 제외)
  3) 좌우 미러 정규화 (horz_break, rel_side 부호)
  4) season x pitch_type_group robust z   <- fold 학습 시즌 통계만
  5) 절대 수준 제거: 각 셀을 '해당 fold 학습셋의 시즌x손x카운트 평균 대비 잔차'로
  6) EB 축소
  7) soft crosswalk 로 main pitcher_id 결합
출력: fold별 (pitcher_id, season) -> Tier E 임베딩 parquet
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

DATA = "/mnt/user-data/uploads/LGAIMERS/data"
OUT = "/home/claude/work/outputs/tier_e"
os.makedirs(OUT, exist_ok=True)

TM_METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
              "extension", "rel_height", "rel_side", "zone_speed"]
GROUPS = ["fastball", "breaking", "offspeed"]          # other 제외 (0.82%, 감소 추세)

# ---- 이상치 규칙 (04_PREPROCESSING_SPEC §2-1)
CLIP = {"extension": (0.5, None), "rel_height": (0.8, None), "rel_speed": (90.0, None),
        "spin_rate": (800.0, None), "induced_vert_break": (-90.0, 90.0),
        "horz_break": (-90.0, 90.0)}


def load_trackman():
    t0 = time.time()
    tm = pd.read_csv(f"{DATA}/trackman_history.csv")
    n0 = len(tm)
    # 구조 이상 행
    bad = (tm.balls_before > 3) | (tm.strikes_before > 2) | (tm.outs_before > 2) | (tm.inning < 1)
    tm = tm[~bad]
    tm = tm.drop_duplicates(subset=["trackman_game_id", "pitch_no"], keep="first")
    # 물리 이상치 -> NaN
    for c, (lo, hi) in CLIP.items():
        if lo is not None:
            tm.loc[tm[c] < lo, c] = np.nan
        if hi is not None:
            tm.loc[tm[c] > hi, c] = np.nan
    tm.loc[tm.zone_speed > tm.rel_speed, ["zone_speed", "rel_speed"]] = np.nan
    # 좌우 미러 정규화
    sign = np.where(tm.pitcher_hand.eq("Left"), -1.0, 1.0)
    tm["horz_break"] = tm.horz_break * sign
    tm["rel_side"] = tm.rel_side * sign
    tm["b_same"] = (tm.batter_hand.eq("Left") == tm.pitcher_hand.eq("Left")).astype(int)
    tm["bhand"] = np.where(tm.batter_hand.eq("Left"), 1, 2)
    tm = tm[tm.pitch_type_group.isin(GROUPS)]
    print(f"trackman loaded {n0} -> {len(tm)} in {time.time()-t0:.1f}s", flush=True)
    return tm


def robust_z(tm, cutoff):
    """season x pitch_type_group 내 robust z. 통계는 cutoff 미만 시즌만으로."""
    fit = tm[tm.season < cutoff]
    stats = (fit.groupby(["season", "pitch_type_group"])[TM_METRICS]
             .agg(["median", lambda s: s.quantile(.75) - s.quantile(.25)]))
    stats.columns = [f"{m}__{'med' if k=='median' else 'iqr'}" for m, k in stats.columns]
    tm = tm.merge(stats.reset_index(), on=["season", "pitch_type_group"], how="left")
    for c in TM_METRICS:
        iqr = tm[f"{c}__iqr"].clip(lower=1e-6) / 1.349
        tm[c + "_rz"] = (tm[c] - tm[f"{c}__med"]) / iqr
    return tm.drop(columns=[c for c in tm.columns if c.endswith(("__med", "__iqr"))])


def count_bucket(b, s):
    """볼카운트를 6개 버킷으로. 셀 희소성 대비 (12셀 -> 6셀)."""
    ahead = s > b                      # 투수 유리
    even = s == b
    three_b = b == 3
    two_s = s == 2
    out = np.full(len(b), 0, dtype=np.int8)
    out[even & ~two_s] = 0             # 0-0, 1-1
    out[ahead & ~two_s] = 1            # 0-1, 1-2 아님... (아래 two_s 로 덮음)
    out[two_s & ~three_b] = 2          # x-2 (3-2 제외)
    out[three_b & ~two_s] = 3          # 3-0, 3-1
    out[three_b & two_s] = 4           # 3-2
    out[(~ahead) & (~even) & (~three_b)] = 5   # 1-0, 2-0, 2-1
    return out


def build_profile(tm, cutoff, min_pitches, eb_m):
    """투수-시즌 x (구종군 x 카운트버킷 x 타자손) 사용률 + 물리 반응. 잔차 형태."""
    d = tm[tm.season < cutoff].copy()
    n_by = d.groupby(["pitcher_trackman_id", "season"]).size().rename("n")
    ok = n_by[n_by >= min_pitches].reset_index()[["pitcher_trackman_id", "season"]]
    d = d.merge(ok, on=["pitcher_trackman_id", "season"], how="inner")
    if d.empty:
        return None
    d["cb"] = count_bucket(d.balls_before.values, d.strikes_before.values)

    key = ["pitcher_trackman_id", "season"]
    cell = ["pitch_type_group", "cb", "bhand"]

    # --- (a) 사용률: 셀 조건부 구종 선택 확률
    cnt = d.groupby(key + cell).size().rename("n").reset_index()
    tot = d.groupby(key + ["cb", "bhand"]).size().rename("tot").reset_index()
    cnt = cnt.merge(tot, on=key + ["cb", "bhand"])
    # 리그 기준 (cutoff 미만) 로 EB 축소  <- 절대 수준 제거의 1단계
    lg = (d.groupby(["season"] + cell).size().rename("ln").reset_index()
          .merge(d.groupby(["season", "cb", "bhand"]).size().rename("ltot").reset_index(),
                 on=["season", "cb", "bhand"]))
    lg["lg_rate"] = lg.ln / lg.ltot
    cnt = cnt.merge(lg[["season"] + cell + ["lg_rate"]], on=["season"] + cell, how="left")
    cnt["rate"] = (cnt.n + eb_m * cnt.lg_rate) / (cnt.tot + eb_m)
    # 잔차 = 개인 - 리그 (절대 수준 제거)
    cnt["resid"] = cnt.rate - cnt.lg_rate
    cnt["feat"] = ("u_" + cnt.pitch_type_group.astype(str) + "_c" + cnt.cb.astype(str)
                   + "_h" + cnt.bhand.astype(str))

    # --- (b) 물리 반응: 셀별 rz 평균의 개인-리그 잔차 (핵심 물리량 3개만)
    phys = ["rel_speed_rz", "induced_vert_break_rz", "rel_side_rz"]
    pm = d.groupby(key + cell)[phys].agg(["mean", "count"])
    pm.columns = [f"{a}|{b}" for a, b in pm.columns]
    pm = pm.reset_index()
    lgp = d.groupby(["season"] + cell)[phys].mean().add_suffix("|lg").reset_index()
    pm = pm.merge(lgp, on=["season"] + cell, how="left")
    prows = []
    for c in phys:
        nn = pm[f"{c}|count"].values
        shr = nn / (nn + eb_m)
        r = shr * (pm[f"{c}|mean"].values - pm[f"{c}|lg"].values)
        prows.append(pd.DataFrame({
            **{k: pm[k] for k in key},
            "feat": ("p_" + c.replace("_rz", "") + "_c" + pm.cb.astype(str)
                     + "_h" + pm.bhand.astype(str) + "_" + pm.pitch_type_group.astype(str)),
            "resid": r}))

    long = pd.concat([cnt[key + ["feat", "resid"]]] + prows, ignore_index=True)
    long = long.dropna(subset=["resid"])
    long = long.groupby(key + ["feat"], as_index=False)["resid"].mean()
    return long, ok


def to_matrix(long, key=("pitcher_trackman_id", "season")):
    key = list(key)
    rows = long.groupby(key).ngroup()
    feats = pd.Categorical(long.feat)
    M = sparse.csr_matrix((long.resid.values, (rows.values, feats.codes)),
                          shape=(rows.max() + 1, len(feats.categories)))
    idx = long[key].drop_duplicates().reset_index(drop=True)
    idx = long.groupby(key, as_index=False).size()[key]
    return M, idx.reset_index(drop=True), list(feats.categories)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", default="2023,2024,2025")
    ap.add_argument("--min-pitches", type=int, default=300)
    ap.add_argument("--eb", type=float, default=200.0)
    ap.add_argument("--dim", type=int, default=12)
    args = ap.parse_args()

    tm_raw = load_trackman()
    manifest = []
    for cutoff in [int(x) for x in args.cutoffs.split(",")]:
        t0 = time.time()
        tm = robust_z(tm_raw, cutoff)
        out = build_profile(tm, cutoff, args.min_pitches, args.eb)
        if out is None:
            continue
        long, elig = out
        M, idx, feat_names = to_matrix(long)
        k = min(args.dim, M.shape[1] - 1, M.shape[0] - 1)
        svd = TruncatedSVD(n_components=k, random_state=0)
        Z = svd.fit_transform(M)
        # 부호 고정: 각 성분의 최대 절대 로딩이 양수가 되도록 (fold 간 정렬)
        for j in range(k):
            v = svd.components_[j]
            if v[np.argmax(np.abs(v))] < 0:
                Z[:, j] *= -1
                svd.components_[j] *= -1
        emb = pd.DataFrame(Z, columns=[f"te_svd_{j:02d}" for j in range(k)])
        emb = pd.concat([idx.reset_index(drop=True), emb], axis=1)
        emb["te_n_cells"] = long.groupby(["pitcher_trackman_id", "season"]).size().values
        emb.to_parquet(f"{OUT}/tier_e_cutoff{cutoff}.parquet", index=False)
        rec = dict(cutoff=cutoff, pitcher_seasons=int(M.shape[0]), features=int(M.shape[1]),
                   svd_dim=k, explained=float(svd.explained_variance_ratio_.sum()),
                   min_pitches=args.min_pitches, eb=args.eb, sec=round(time.time() - t0, 1))
        manifest.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
