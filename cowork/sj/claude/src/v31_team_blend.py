"""V31: sj × hw 2자 결합 재측정 (CPU 전용).

V5 에서 한 번 기각했다. 그때 hw 예측은 구버전이었고, 이후 v7(Public 847.76)과
v8 이 나왔다. "버렸던 것도 다시 본다"는 원칙에 따라 다시 잰다.

공통 규약
    cowork/*/val2024_pred.csv   (row_id, control_success)
    2019~2023 학습 -> 2024 예측. hw 와 sj 모두 이 규약으로 올렸다.

측정
    A  hw 단독
    B  sj 파일 단독 (제출 라인)
    C  sj 성분결합 라인 (submit_030 구성, 여기서 재구성)
    D  C x hw 결합 곡선

주의
    2024 는 게이트 fold 다. 여기서 결합 비율을 고르면 지금까지 지킨 원칙을 깬다.
    이 실험은 '팀 결합에 여지가 있는가'를 진단한다. 비율 확정은 hw 가 2022/2023
    예측도 올려준 뒤에 한다.

출력: outputs/v31_team_blend.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load, metrics

CW = Path(__file__).resolve().parents[3]
MO = Path(__file__).resolve().parents[2] / "experiment" / "model_optimization"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
WS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EPS = 1e-7

df = load()
season = df["season"].to_numpy()
va = season == 2024
ids = df.loc[va, "row_id"].to_numpy()
y = df[TARGET].to_numpy(np.float64)[va]
null = y.mean() * (1 - y.mean())


def read(name):
    d = pd.read_csv(CW / name).set_index("row_id").reindex(ids)["control_success"]
    v = d.to_numpy(np.float64)
    miss = int(np.isnan(v).sum())
    return np.clip(np.nan_to_num(v, nan=float(np.nanmean(v))), EPS, 1 - EPS), miss


hw, hw_miss = read("hw/val2024_pred.csv")
sj, sj_miss = read("sj/val2024_pred.csv")
prod = pd.read_parquet(PROD).set_index("row_id").reindex(ids)
base = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
gt = prod["game_type"].astype(str).to_numpy()
p_ie = np.load(CACHE / "v25_p_ie_029.npy")
sj030 = np.clip(0.25 * np.clip(p_ie - 0.0077, EPS, 1 - EPS) + 0.75 * base, EPS, 1 - EPS)

print(f"2024 행 {len(ids):,}   결측 hw {hw_miss}  sj {sj_miss}")
solo = {"hw": hw, "sj_파일": sj, "sj_base(021)": base, "sj_030라인": sj030}
print(f"\n{'':<14}{'BSS':>11}{'pred_mean':>11}")
for k, v in solo.items():
    m = metrics(y, v, game_type=gt)
    print(f"{k:<14}{m['bss_raw']:>11.2f}{m['pred_mean']:>11.5f}")
print(f"{'실제 평균':<14}{'':>11}{y.mean():>11.5f}")

print(f"\n상관 (logit)")
lg = {k: np.log(v / (1 - v)) for k, v in solo.items()}
ks = list(solo)
print(f"  {'':<14}" + "".join(f"{k:>14}" for k in ks))
for a in ks:
    print(f"  {a:<14}" + "".join(f"{np.corrcoef(lg[a], lg[b])[0,1]:>14.4f}" for b in ks))

rows = []
print(f"\nw 는 hw 비중.  q = w*hw + (1-w)*파트너")
print(f"{'w':>6}{'hw x sj030':>14}{'t_row':>9}{'hw x sj파일':>14}{'t_row':>9}")
for w in WS:
    line = f"{w:>6.1f}"
    for tag, partner in [("sj030", sj030), ("sjfile", sj)]:
        q = np.clip(w * hw + (1 - w) * partner, EPS, 1 - EPS)
        m = metrics(y, q, game_type=gt)
        d = m["bss_raw"] - metrics(y, partner, game_type=gt)["bss_raw"]
        dr = (partner - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"w_hw": w, "partner": tag, "bss": m["bss_raw"], "dbss": d,
                     "t_row": d / se, "pred_mean": m["pred_mean"]})
        line += f"{m['bss_raw']:>14.2f}{d/se:>9.2f}"
    print(line, flush=True)

# ---------------------------------------------- 구간별 — 전역이 죽어도 국소는?
CUTS = [100, 500, 2000, 4000]
BNAME = ["0-99", "100-499", "500-1999", "2000-3999", "4000+"]
bk = np.digitize(df.loc[va, "asof_pitcher_n"].to_numpy(), CUTS)
print(f"\n구간별 — hw 가 이기는 구간이 있는가")
print(f"  {'구간':<11}{'비중':>7}{'hw':>10}{'sj030':>10}{'차이':>9}{'최적w_hw':>10}{'국소이득':>9}")
for k in range(5):
    m = bk == k
    yk, hk, sk = y[m], hw[m], sj030[m]
    nk = yk.mean() * (1 - yk.mean())
    bh = 100000 * (nk - ((hk - yk) ** 2).mean()) / nk
    bs = 100000 * (nk - ((sk - yk) ** 2).mean()) / nk
    cur = [(w, 100000 * (nk - ((np.clip(w * hk + (1 - w) * sk, EPS, 1 - EPS) - yk) ** 2
                              ).mean()) / nk) for w in WS]
    bw, bb = max(cur, key=lambda t: t[1])
    rows.append({"w_hw": bw, "partner": f"bucket_{BNAME[k]}", "bss": bb,
                 "dbss": bb - bs, "t_row": np.nan, "pred_mean": float(m.mean())})
    print(f"  {BNAME[k]:<11}{m.mean()*100:>6.2f}%{bh:>10.1f}{bs:>10.1f}"
          f"{bh-bs:>+9.1f}{bw:>10.1f}{bb-bs:>+9.2f}")

print(f"\n경기유형별")
for g in ["R", "F"]:
    m = gt == g
    if m.sum() == 0:
        continue
    yk, hk, sk = y[m], hw[m], sj030[m]
    nk = yk.mean() * (1 - yk.mean())
    bh = 100000 * (nk - ((hk - yk) ** 2).mean()) / nk
    bs = 100000 * (nk - ((sk - yk) ** 2).mean()) / nk
    print(f"  {g}  {m.mean()*100:5.2f}%   hw {bh:9.1f}   sj030 {bs:9.1f}   차이 {bh-bs:+9.1f}")

res = pd.DataFrame(rows)
res.to_csv(OUT / "v31_team_blend.csv", index=False)
for tag in ["sj030", "sjfile"]:
    s = res[res.partner == tag]
    b = s.sort_values("bss", ascending=False).iloc[0]
    at0 = s[s.w_hw == 0.0].iloc[0]
    print(f"\n{tag}: 최적 w_hw={b.w_hw:.1f}  BSS {b.bss:.2f}  "
          f"단독 {at0.bss:.2f} 대비 {b.bss-at0.bss:+.2f}")
print("\n2024 는 게이트 fold 다. 위 최적 w 는 진단용이며 그대로 채택하지 않는다.")
print(f"\nsaved -> {OUT/'v31_team_blend.csv'}")
