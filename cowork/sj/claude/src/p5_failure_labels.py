"""P5: 실패 유형 라벨의 정식 산출 + 검증.

산출 경로 (train.csv 내부만 사용, 추론에는 일절 사용하지 않음)
--------------------------------------------------------------
asof_* rate 컬럼은 "해당 투구 직전까지의 누적 비율"이다. 따라서
    누적카운트(i) = rate(i) x asof_pitcher_n(i)
이고, 같은 투수의 바로 다음 투구에서
    누적카운트(i+1) - 누적카운트(i) in {0, 1}
이 값이 투구 i의 해당 유형 발생 여부다. 즉 라벨 i는 행 i+1의 공식 제공 컬럼에서
정확히 유도된다.

이 유도는 학습 라벨 생성에만 쓴다. 추론 시 모델 입력은 투구 이전 피처뿐이며
test.csv의 다른 행을 참조하지 않는다.

검증 항목
    V1 asof_pitcher_n이 투수 내에서 유일한가 (정렬 근거)
    V2 연속쌍 판정 가능 행 비율
    V3 delta가 {0,1} 밖으로 나가는 행
    V4 복원 success == 실제 control_success  (100%여야 함)
    V5 ball/strike 배타성
    V6 middle/reverse가 success=1에서 0건인가
    V7 성분 분해의 합이 실패 총계와 일치하는가
    V8 시즌별 성분 비율 (regime 변화 확인)

출력: cache/failure_labels.parquet  (row_id, y_middle, y_reverse, y_outside,
      y_ball, y_strike, label_ok)
      outputs/p5_failure_labels_report.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, CACHE, load

RATE = {"success": "asof_pitcher_success_rate",
        "reverse": "asof_pitcher_reverse_rate",
        "middle": "asof_pitcher_middle_rate",
        "ball": "asof_pitcher_ball_rate",
        "strike": "asof_pitcher_strike_rate"}
KINDS = list(RATE)
checks = []


def rec(name, value, note=""):
    checks.append({"check": name, "value": value, "note": note})
    print(f"  {name:<34} {value}   {note}", flush=True)


df = load()
d = df[["row_id", "season", "pitcher_id", "asof_pitcher_n", TARGET]
       + list(RATE.values())].copy()

print("\n[V1] 정렬 근거 — asof_pitcher_n이 투수 내에서 유일한가", flush=True)
dup = d.duplicated(["pitcher_id", "asof_pitcher_n"]).sum()
rec("duplicate (pitcher, n) rows", int(dup),
    "0이어야 정렬이 유일하게 결정된다")

d = d.sort_values(["pitcher_id", "asof_pitcher_n"], kind="mergesort").reset_index(drop=True)
n = d["asof_pitcher_n"].to_numpy(np.float64)
pid = d["pitcher_id"].to_numpy()

print("\n[V2] 연속쌍 판정", flush=True)
same = np.r_[pid[1:] == pid[:-1], False]
step1 = np.r_[(n[1:] - n[:-1]) == 1, False]
usable = same & step1
rec("total rows", int(len(d)))
rec("same pitcher as next row", int(same.sum()))
rec("n increments by exactly 1", int((same & step1).sum()))
rec("last pitch per pitcher (no next)", int((~same).sum()),
    f"투수 {d['pitcher_id'].nunique()}명")
rec("gap rows (same pitcher, n jump != 1)", int((same & ~step1).sum()))
rec("usable pct", round(100 * usable.mean(), 4))

# 누적 카운트 차분
cnt = {k: np.round(d[RATE[k]].to_numpy(np.float64) * n) for k in KINDS}
delta = {k: np.r_[cnt[k][1:] - cnt[k][:-1], np.nan] for k in KINDS}

print("\n[V3] delta가 {0,1} 안에 있는가", flush=True)
in01 = np.ones(len(d), dtype=bool)
for k in KINDS:
    ok_k = np.isin(delta[k], [0.0, 1.0]) & usable
    in01 &= (ok_k | ~usable)
    bad = int((usable & ~np.isin(delta[k], [0.0, 1.0])).sum())
    rec(f"{k}: out-of-range among usable", bad)
label_ok = usable & in01
rec("fully recovered rows", int(label_ok.sum()),
    f"{100*label_ok.mean():.4f}%")

y = {k: np.where(label_ok, delta[k], np.nan) for k in KINDS}
tgt = d[TARGET].to_numpy()

print("\n[V4] 복원 success == 실제 control_success", flush=True)
agree = float((y["success"][label_ok] == tgt[label_ok]).mean())
rec("agreement", round(agree * 100, 6), "100이어야 유도식이 정확하다")
mism = int((y["success"][label_ok] != tgt[label_ok]).sum())
rec("mismatched rows", mism)

sub = pd.DataFrame({k: y[k][label_ok].astype(int) for k in KINDS})
sub["y"] = tgt[label_ok]
N = len(sub)

print("\n[V5] ball / strike 배타성", flush=True)
rec("ball=1 & strike=1", int(((sub.ball == 1) & (sub.strike == 1)).sum()), "0이어야 함")
inplay = int(((sub.ball == 0) & (sub.strike == 0)).sum())
rec("neither (인플레이 추정)", inplay, f"{100*inplay/N:.2f}%")

print("\n[V6] middle / reverse 는 순수 실패 유형인가", flush=True)
rec("middle=1 & success=1", int(((sub.middle == 1) & (sub.y == 1)).sum()), "0이어야 함")
rec("reverse=1 & success=1", int(((sub.reverse == 1) & (sub.y == 1)).sum()), "0이어야 함")
rec("ball=1 & success=1", int(((sub.ball == 1) & (sub.y == 1)).sum()),
    "0이 아니다 -> ball은 실패 유형이 아니다")

print("\n[V7] 3성분 분해", flush=True)
M = (sub.middle == 1).to_numpy()
R = (sub.reverse == 1).to_numpy()
F = (sub.y == 0).to_numpy()
O = F & ~M & ~R                       # OUTSIDE = 실패 and not middle and not reverse
rec("failures", int(F.sum()))
rec("MIDDLE", int(M.sum()), f"{100*M.sum()/F.sum():.2f}% of failures")
rec("REVERSE", int(R.sum()), f"{100*R.sum()/F.sum():.2f}% of failures")
rec("MIDDLE & REVERSE", int((M & R).sum()))
rec("OUTSIDE (배타적 잔여)", int(O.sum()), f"{100*O.sum()/F.sum():.2f}% of failures")
rec("M or R or O == failures", bool(np.array_equal(M | R | O, F)), "True여야 함")
rec("OUTSIDE & (M or R)", int((O & (M | R)).sum()), "0 — 정의상 배타")
rec("  of which ball=1", int((O & (sub.ball == 1).to_numpy()).sum()))
rec("  of which ball=0", int((O & (sub.ball == 0).to_numpy()).sum()),
    "존 안인데 실패 — 목록에 없는 네번째 모드")

pm, pr, po = M.mean(), R.mean(), O.mean()
ps = float((sub.y == 1).mean())
formula = (1 - pm) * (1 - pr) - po
rec("P(middle)", round(float(pm), 6))
rec("P(reverse)", round(float(pr), 6))
rec("P(outside)", round(float(po), 6))
rec("P(m)P(r) vs P(m&r)",
    f"{pm*pr:.6f} vs {(M&R).mean():.6f}", "차이가 작으면 주변독립")
rec("corr(middle, reverse)", round(float(np.corrcoef(M, R)[0, 1]), 6))
rec("(1-pm)(1-pr)-po", round(float(formula), 6))
rec("actual P(success)", round(ps, 6))
rec("formula error", round(float(formula - ps), 8))

print("\n[V8] 시즌별 성분 비율 — regime 변화", flush=True)
seas = d["season"].to_numpy()[label_ok]
gt = df.loc[d.index[label_ok] if False else slice(None)]  # placeholder
tab = pd.DataFrame({"season": seas, "M": M, "R": R, "O": O, "y": sub.y.to_numpy()})
g = tab.groupby("season").agg(n=("y", "size"), success=("y", "mean"),
                              middle=("M", "mean"), reverse=("R", "mean"),
                              outside=("O", "mean"))
g["sum_check"] = 1 - g["success"] - (g["middle"] + g["reverse"] + g["outside"])
print(g.round(6).to_string(), flush=True)
print("  sum_check = 실패율 - (M+R+O). M,R 중복분만큼 양수여야 한다", flush=True)

# 산출물 저장 — 원래 row_id 순서로 되돌린다
lab = pd.DataFrame({
    "row_id": d["row_id"].to_numpy(),
    "label_ok": label_ok.astype(np.int8),
})
for name, arr in [("y_middle", M), ("y_reverse", R), ("y_outside", O)]:
    full = np.full(len(d), -1, dtype=np.int8)
    full[label_ok] = arr.astype(np.int8)
    lab[name] = full
for k in ["ball", "strike"]:
    full = np.full(len(d), -1, dtype=np.int8)
    full[label_ok] = sub[k].to_numpy().astype(np.int8)
    lab[f"y_{k}"] = full
lab = lab.sort_values("row_id", kind="mergesort").reset_index(drop=True)
lab.to_parquet(CACHE / "failure_labels.parquet", index=False)

pd.DataFrame(checks).to_csv(OUT / "p5_failure_labels_report.csv", index=False)
g.to_csv(OUT / "p5_failure_labels_by_season.csv")
print(f"\nsaved -> {CACHE/'failure_labels.parquet'}  ({len(lab):,} rows)")
print(f"saved -> {OUT/'p5_failure_labels_report.csv'}")
