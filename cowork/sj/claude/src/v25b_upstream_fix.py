"""V25b: 상류 reverse scale 재최적화 — 앵커 수정판.

V25 의 오류
    p(s) = p019 + (s - 0.40) * 0.6085 * reverse20   <- 틀렸다
    submit_019 는 reverse3 시스템이고 reverse20 보정이 아예 없다.
    검증에서 base_at(0.40) 이 836.503 대신 835.861(=019 값)을 내며 잡혔다.

올바른 식 (앞서 021/020/022 로 검증 완료, 재구성 최대오차 6e-08)
    p(s) = p021 + (s - 0.40) * 0.6085 * reverse20
    p021 이 s=0.40 앵커다.

성분 라인 예측은 v25_p_ie_029.npy 에 캐시돼 있어 재학습 없이 격자만 다시 낸다.

판정 (V25 에서 사전 등록)
    2022/2023 에는 프로덕션 예측이 없어 s 재최적화가 2024 에서만 가능하다.
    게이트 fold 에서 최적점을 고르면 지금까지 지킨 원칙을 깬다.
        곡선이 평평하면(폭 < 5)  채택 가능 - 어느 값이든 비슷해 과적합 위험이 낮다
        뾰족하면                 진단만 하고 현행 0.475 유지

출력: outputs/v25b_upstream_scale.csv
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
SCALES = [0.20, 0.30, 0.40, 0.475, 0.55, 0.65, 0.75, 0.90]
WS = [0.15, 0.20, 0.25, 0.30, 0.35]
OUTER_W = 0.6085
EPS = 1e-7

df = load()
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
va = season == 2024
y_va = y_all[va]

prod = pd.read_parquet(PROD).set_index("row_id").reindex(df.loc[va, "row_id"].to_numpy())
z = np.load(NPZ, allow_pickle=True)
rev = z["reverse20"].astype(np.float64)
p021 = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
p020 = np.clip(prod["submit020_reverse20_s055_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
null = y_va.mean() * (1 - y_va.mean())
p_ie = np.load(CACHE / "v25_p_ie_029.npy")
print(f"성분 라인 단독 {metrics(y_va, p_ie)['bss_raw']:.2f}  (submit_029 구성)",
      flush=True)


def base_at(s):
    return np.clip(p021 + (s - 0.40) * OUTER_W * rev, EPS, 1 - EPS)


print("\n재구성 검증 (기록값과 대조)")
for s, name, rec in [(0.40, "021", 836.502924), (0.475, "022", 836.242000),
                     (0.55, "020", 835.794539)]:
    b = base_at(s)
    print(f"  s={s:<5} {name}  {metrics(y_va, b)['bss_raw']:.6f}  기록 {rec:.6f}"
          f"  차이 {metrics(y_va, b)['bss_raw']-rec:+.6f}", flush=True)
print(f"  020 최대절대오차 {np.max(np.abs(base_at(0.55)-p020)):.3e}", flush=True)

ref = metrics(y_va, p021, game_type=gt)["bss_raw"]
rows = []
print(f"\nreverse scale x w  ->  ΔBSS  (기준 submit_021 {ref:.3f})")
print(f"{'s':>8}" + "".join(f"{w:>10.2f}" for w in WS) + f"{'base단독':>11}")
for s in SCALES:
    b = base_at(s)
    solo = metrics(y_va, b)["bss_raw"]
    line = f"{s:>8.3f}"
    for w in WS:
        q = np.clip(w * p_ie + (1 - w) * b, EPS, 1 - EPS)
        mm = metrics(y_va, q, game_type=gt)
        d = mm["bss_raw"] - ref
        dr = (p021 - y_va) ** 2 - (q - y_va) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"scale": s, "w": w, "base_solo": solo, "bss": mm["bss_raw"],
                     "dbss": d, "se_row": se, "t_row": d / se})
        line += f"{d:>+10.2f}"
    print(line + f"{solo:>11.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "v25b_upstream_scale.csv", index=False)
cur = res[(res.scale == 0.475) & (res.w == 0.25)].iloc[0]
sub = res[res.w == 0.25].sort_values("scale")
flat = float(sub["dbss"].max() - sub["dbss"].min())
best_s = sub.sort_values("dbss", ascending=False).iloc[0]
print(f"\n현행 submit_029 (s=0.475, w=0.25)  ΔBSS {cur.dbss:+.3f}  t_row {cur.t_row:+.2f}")
print(f"w=0.25 고정, scale 최고 s={best_s.scale}  ΔBSS {best_s.dbss:+.3f}  "
      f"차이 {best_s.dbss-cur.dbss:+.3f}")
print(f"scale 0.20~0.90 구간 ΔBSS 폭 {flat:.2f}  "
      f"-> {'평평. 상류 scale 재조정 이득 없음, 현행 유지' if flat < 5 else '뾰족. 추가 검토 필요'}")
print(f"\nsaved -> {OUT/'v25b_upstream_scale.csv'}")
