"""V42: 상류 r_context 보정이 성분 층과 중복되는가. (CPU 전용)

발견
    프로덕션 파이프라인의 VALIDATION_LOG 를 읽다가 중복을 찾았다.

        프로덕션  r_context_correction = 다년 count x inning4 x hands, scale 1.15
                  submit_015/016 에서 Val2024 +18 을 낸 보정
        성분 라인 count_platoon  (+8.44)  =  EB(투수, 타자손, 카운트군) 2겹 차감
                  inning_platoon (+2.98)  =  EB(투수, 타자손, 이닝군) 2겹 차감

    같은 축이다. 두 층이 같은 보정을 두 번 하고 있다.

    submit_021 메타데이터에도 선례가 있다 — "downstream R/F 보정과 중복을 줄이려
    reverse scale 0.40". 하류에 층이 붙으면 상류 보정이 과해진다는 논리를
    프로덕션 자신이 이미 적용했다. 성분 층은 그보다 훨씬 강하다.

측정
    npz 의 r_correction 이 적용된 델타라고 보고
        p(a) = p021 + (a − 1) * r_correction
    로 스케일을 되돌린다. a=1 이 현행, a=0 이 보정 제거.
    먼저 a=1 이 기록값 836.503 을 재현하는지 확인한다(V25 의 교훈).

    성분 층은 submit_031 구성(구간별 가중치 + 절편)으로 얹는다.

판정
    2022/2023 에는 프로덕션 예측이 없어 2024 에서만 잴 수 있다. 게이트 fold 다.
        곡선이 평평하면(폭 < 5)  현행 유지. 중복이 없다는 뜻이다.
        a<1 쪽으로 뚜렷하면      중복이 실재한다. 다만 최적점을 그대로 쓰지 않고
                                 사전 예측(하류가 강하면 상류를 줄인다)이 맞았다는
                                 확인으로만 쓰고, 채택은 보수적 값으로 한다.
    reverse scale 과 동시에 보는 2차원 격자도 낸다.

출력: outputs/v42_upstream_rcontext.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load, metrics

MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
NPZ = PROD.parent / "reverse20_submission_components.npz"
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
SCALES = [0.30, 0.40, 0.475, 0.55, 0.70]
OUTER_W, C0 = 0.6085, 0.0077
BW = np.array([0.25, 0.25, 0.25, 0.25, 0.45])
CUTS = [100, 500, 2000, 4000]
EPS = 1e-7

df = load()
season = df["season"].to_numpy()
va = season == 2024
ids = df.loc[va, "row_id"].to_numpy()
y = df[TARGET].to_numpy(np.float64)[va]
bk = np.digitize(df.loc[va, "asof_pitcher_n"].to_numpy(), CUTS)
w = BW[bk]
null = y.mean() * (1 - y.mean())

prod = pd.read_parquet(PROD).set_index("row_id").reindex(ids)
z = np.load(NPZ, allow_pickle=True)
order = pd.Index(z["row_id"]).get_indexer(ids)
assert (order >= 0).all(), "npz 에 없는 row_id 가 있다"
rc = z["r_correction"].astype(np.float64)[order]
rev = z["reverse20"].astype(np.float64)[order]
p021 = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
pie = np.clip(np.load(CACHE / "v25_p_ie_029.npy") - C0, EPS, 1 - EPS)

ref = metrics(y, p021, game_type=gt)["bss_raw"]
print(f"기준 submit_021  {ref:.6f}  (기록 836.502924, 차이 {ref-836.502924:+.6f})")
print(f"r_correction  mean {rc.mean():+.6f}  sd {rc.std():.6f}  "
      f"비영 {np.mean(np.abs(rc) > 1e-12)*100:.1f}%")
print(f"성분 라인 단독 {metrics(y, pie)['bss_raw']:.2f}")


def base_at(a, s):
    return np.clip(p021 + (a - 1.0) * rc + (s - 0.40) * OUTER_W * rev, EPS, 1 - EPS)


chk = base_at(1.0, 0.40)
print(f"\n재구성 검증  a=1,s=0.40 -> {metrics(y, chk)['bss_raw']:.6f}  "
      f"최대오차 {np.max(np.abs(chk-p021)):.3e}")
assert np.max(np.abs(chk - p021)) < 1e-9

rows = []
print(f"\n{'a\\s':>7}" + "".join(f"{s:>11.3f}" for s in SCALES) + f"{'base단독':>11}")
for a in ALPHAS:
    line = f"{a:>7.2f}"
    solo = metrics(y, base_at(a, 0.475))["bss_raw"]
    for s in SCALES:
        b = base_at(a, s)
        q = np.clip(w * pie + (1 - w) * b, EPS, 1 - EPS)
        mm = metrics(y, q, game_type=gt)
        d = mm["bss_raw"] - ref
        dr = (p021 - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"alpha": a, "scale": s, "bss": mm["bss_raw"], "dbss": d,
                     "t_row": d / se, "base_solo": metrics(y, b)["bss_raw"]})
        line += f"{d:>+11.2f}"
    print(line + f"{solo:>11.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v42_upstream_rcontext.csv", index=False)
cur = res[(res.alpha == 1.0) & (res.scale == 0.475)].iloc[0]
col = res[res.scale == 0.475].sort_values("alpha")
span = float(col["dbss"].max() - col["dbss"].min())
best = col.sort_values("dbss", ascending=False).iloc[0]
print(f"\n현행 (a=1.00, s=0.475)  ΔBSS {cur.dbss:+.3f}")
print(f"s=0.475 고정, a 최고 {best.alpha:.2f}  ΔBSS {best.dbss:+.3f}  "
      f"차이 {best.dbss-cur.dbss:+.3f}")
print(f"a 0.00~1.50 구간 폭 {span:.2f}  -> "
      f"{'평평. 중복 없음, 현행 유지' if span < 5 else '중복 실재'}")
print(f"\nsaved -> {OUT/'v42_upstream_rcontext.csv'}")
