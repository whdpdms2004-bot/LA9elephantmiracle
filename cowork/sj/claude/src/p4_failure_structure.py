"""P4 (아이디어 3 전제 확인): 실패 유형 세 개가 배타인가 중첩인가.

"세 유형 확률의 합사건으로 최종 확률" 설계는 세 사건의 결합 구조에 따라
식이 완전히 달라진다.

  배타(mutually exclusive)   -> P(fail) = a + b + c          (합)
  독립(independent)          -> P(fail) = 1 - (1-a)(1-b)(1-c) (포함배제 근사)
  일반                        -> 포함배제 항 전부 필요

배타인데 독립 공식을 쓰면 1-(1-a)(1-b)(1-c) >= a+b+c 라 실패 확률을 항상
과대평가한다. 따라서 식을 고르기 전에 실제 구조를 측정한다.

방법: 같은 투수의 연속 행에서 asof 누적 카운트 차분으로 각 투구의 유형을 복원한다.
      README 3.1이 '누수 진단'으로 명시한 용도이며 여기서는 라벨 구조 파악에만 쓴다.
      추론에는 절대 쓰지 않는다.

출력: outputs/p4_failure_structure.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, load

RATE_COLS = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}

df = load()
d = df[["row_id", "season", "pitcher_id", "asof_pitcher_n", TARGET]
       + list(RATE_COLS.values())].copy()
d = d.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index(drop=True)

n = d["asof_pitcher_n"].to_numpy(np.float64)
pid = d["pitcher_id"].to_numpy()

# 누적 카운트 복원
cnt = {}
for k, col in RATE_COLS.items():
    cnt[k] = np.round(d[col].to_numpy(np.float64) * n)

# 연속 행 판정: 같은 투수이고 n이 정확히 1 증가
nxt_same = np.r_[pid[1:] == pid[:-1], False]
nxt_step = np.r_[(n[1:] - n[:-1]) == 1, False]
usable = nxt_same & nxt_step
print(f"전체 {len(d):,}행 중 연속쌍으로 유형 복원 가능 {usable.sum():,}행 "
      f"({usable.mean()*100:.2f}%)", flush=True)

flags = {}
for k in RATE_COLS:
    delta = np.r_[cnt[k][1:] - cnt[k][:-1], np.nan]
    flags[k] = delta

u = pd.DataFrame({k: flags[k][usable] for k in RATE_COLS})
u[TARGET] = d[TARGET].to_numpy()[usable]

print("\n각 유형 delta 값 분포 (0/1이 아니면 복원 실패)")
for k in RATE_COLS:
    vc = u[k].value_counts(dropna=False).head(4)
    clean = u[k].isin([0, 1]).mean() * 100
    print(f"  {k:<8} 0/1 비율 {clean:6.2f}%   상위값 {dict(vc)}", flush=True)

ok = np.logical_and.reduce([u[k].isin([0, 1]).to_numpy() for k in RATE_COLS])
u = u[ok].astype(int)
print(f"\n다섯 지표 모두 0/1로 깨끗이 복원된 행: {len(u):,}", flush=True)

# 복원 검증 — success delta가 실제 라벨과 일치하는가
agree = float((u["success"] == u[TARGET]).mean())
print(f"복원 success 와 실제 control_success 일치율: {agree*100:.4f}%", flush=True)

rows = []
print("\n" + "=" * 78)
print("[1] control_success 와 각 유형의 교차")
print("=" * 78)
for k in ["reverse", "middle", "ball", "strike"]:
    ct = pd.crosstab(u[TARGET], u[k])
    print(f"\n{k} (행=control_success, 열={k})")
    print(ct.to_string())
    rows.append({"check": f"crosstab_{k}", "detail": ct.to_dict()})

print("\n" + "=" * 78)
print("[2] 실패(control_success=0) 행에서 세 유형의 동시 발생")
print("=" * 78)
fail = u[u[TARGET] == 0]
combo = fail.groupby(["middle", "ball", "reverse"]).size().rename("n").reset_index()
combo["pct"] = (combo["n"] / len(fail) * 100).round(3)
combo["k"] = combo[["middle", "ball", "reverse"]].sum(axis=1)
print(combo.sort_values("n", ascending=False).to_string(index=False))
print(f"\n실패 행 {len(fail):,}개 중 정확히 하나만 발생: "
      f"{(combo.loc[combo['k']==1,'n'].sum()/len(fail)*100):.3f}%")
print(f"                        두 개 이상 동시: "
      f"{(combo.loc[combo['k']>=2,'n'].sum()/len(fail)*100):.3f}%")
print(f"                        하나도 없음    : "
      f"{(combo.loc[combo['k']==0,'n'].sum()/len(fail)*100):.3f}%")

print("\n" + "=" * 78)
print("[3] 성공(control_success=1) 행에서 세 유형의 발생")
print("=" * 78)
suc = u[u[TARGET] == 1]
combo_s = suc.groupby(["middle", "ball", "reverse"]).size().rename("n").reset_index()
combo_s["pct"] = (combo_s["n"] / len(suc) * 100).round(3)
print(combo_s.sort_values("n", ascending=False).to_string(index=False))

print("\n" + "=" * 78)
print("[4] middle + ball + strike 가 전체를 분할하는가")
print("=" * 78)
mbs = u.groupby(["middle", "ball", "strike"]).size().rename("n").reset_index()
mbs["pct"] = (mbs["n"] / len(u) * 100).round(3)
print(mbs.sort_values("n", ascending=False).to_string(index=False))

print("\n" + "=" * 78)
print("[5] 유형별 주변 확률과 상관")
print("=" * 78)
marg = u[["success", "reverse", "middle", "ball", "strike"]].mean()
print("주변 확률:\n" + marg.round(6).to_string())
print("\n상관행렬:")
print(u[["success", "reverse", "middle", "ball", "strike"]].corr().round(4).to_string())

# 합사건 공식 비교 — 주변 확률만으로 두 식이 얼마나 어긋나는가
a, b, c = float(marg["middle"]), float(marg["ball"]), float(marg["reverse"])
sum_rule = a + b + c
indep_rule = 1 - (1 - a) * (1 - b) * (1 - c)
actual_fail = 1 - float(marg["success"])
print(f"\n실제 실패율            {actual_fail:.6f}")
print(f"배타 가정 (a+b+c)      {sum_rule:.6f}   오차 {sum_rule-actual_fail:+.6f}")
print(f"독립 가정 1-(1-a)(1-b)(1-c)  {indep_rule:.6f}   오차 {indep_rule-actual_fail:+.6f}")

out = pd.DataFrame({
    "metric": ["n_usable", "n_clean", "success_agreement", "actual_fail_rate",
               "sum_rule", "indep_rule", "p_middle", "p_ball", "p_reverse",
               "p_strike", "exactly_one_pct", "two_or_more_pct", "none_pct"],
    "value": [int(usable.sum()), len(u), agree, actual_fail, sum_rule, indep_rule,
              a, b, c, float(marg["strike"]),
              float(combo.loc[combo["k"] == 1, "n"].sum() / len(fail) * 100),
              float(combo.loc[combo["k"] >= 2, "n"].sum() / len(fail) * 100),
              float(combo.loc[combo["k"] == 0, "n"].sum() / len(fail) * 100)],
})
out.to_csv(OUT / "p4_failure_structure.csv", index=False)
print(f"\nsaved -> {OUT/'p4_failure_structure.csv'}")
