"""T1: TrackMan 매핑 실태와 투구수 임계값 결정.

배경 (팀통합 1-4)
  TrackMan 물리 피처는 팀에서 가장 많은 시간을 쓰고 가장 확실하게 실패한 방향이다.
  예나 5회 이상 독립 시도 전부 기각, 정희원 tm_pred_8 은 검증 3/3 개선인데 실LB -36.53.
  다만 '매칭 자체'는 여러 명이 성공했다 (찬우 578명 1:1, 좌우손 576/578 일치).
  팀 결론은 "연결은 되는데 쓸모가 없다"이다.

그럼에도 재는 이유
  1. sj 현행 시스템이 이미 TrackMan 을 쓰고 있다 (500구 게이트, 72컬럼).
     쓰고 있는 것의 실제 커버리지를 모르면 유지/축소/제거 판단을 못 한다.
  2. 예나 제안 "정적 요약통계가 아니라 유사투수 prior 로 재시도"는 아직 아무도 안 했다.
     그걸 하려면 먼저 매핑이 몇 명 / 몇 %를 덮는지 알아야 한다.
  3. P1 에서 투수 투구수 임계값을 정할 때 썼던 것과 같은 커버리지 분석이 필요하다.

측정
  A 메인/TrackMan 규모와 ID 교집합
  B 현행 크로스워크(336명) 신뢰도 분포
  C 크로스워크가 덮는 메인 행 비율 (시즌별)
  D TrackMan 투수-시즌 투구수 임계값별 자격 투수 수와 행 커버리지
  E 커버리지 상한 - 크로스워크를 완벽하게 해도 얼마까지 가능한가

출력: outputs/t1_trackman_mapping.csv
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TARGET, OUT, load

ROOT = Path(__file__).resolve().parents[2]
TM_CSV = ROOT / "data" / "trackman_history.csv"
SUB = ROOT / "submit" / "2026-08-13" / "submit_022.zip"
THRESHOLDS = [1, 100, 200, 300, 500, 750, 1000, 1500]
rows = []


def rec(section, metric, value, note=""):
    rows.append({"section": section, "metric": metric, "value": value, "note": note})
    print(f"  {metric:<44} {value:>14}   {note}", flush=True)


df = load()
season = df["season"].to_numpy()
pid = df["pitcher_id"].to_numpy()

print("=" * 92)
print("A. 규모와 ID 교집합")
print("=" * 92)
tm = pd.read_csv(TM_CSV, usecols=["season", "pitcher_trackman_id", "pitcher_hand"])
rec("A", "메인 행", f"{len(df):,}")
rec("A", "메인 투수", f"{df['pitcher_id'].nunique():,}")
rec("A", "TrackMan 행", f"{len(tm):,}")
rec("A", "TrackMan 투수", f"{tm['pitcher_trackman_id'].nunique():,}")
inter = len(set(df["pitcher_id"].unique()) & set(tm["pitcher_trackman_id"].unique()))
rec("A", "raw ID 직접 교집합", f"{inter:,}", "0이면 크로스워크 필수")

print("\n" + "=" * 92)
print("B. 현행 크로스워크 신뢰도 (submit_022 의 trackman500_lookup_2025.csv)")
print("=" * 92)
with zipfile.ZipFile(SUB) as z:
    lk = pd.read_csv(io.BytesIO(z.read("model/trackman500_lookup_2025.csv")))
rec("B", "매핑된 투수", f"{len(lk):,}", f"메인 792명 중 {100*len(lk)/792:.1f}%")
for c in ["cw_match_seasons", "cw_mean_sim", "cw_min_margin",
          "cw_total_main_n", "cw_total_trackman_n", "tm500_total_pitches",
          "tm500_eligible_seasons", "tm500_season_gap"]:
    if c in lk:
        s = lk[c].dropna()
        rec("B", c, f"{s.median():.3f}",
            f"p10 {s.quantile(.1):.3f} / p90 {s.quantile(.9):.3f}")
hi = lk[(lk["cw_mean_sim"] >= lk["cw_mean_sim"].median())]
rec("B", "cw_mean_sim 중앙값 이상", f"{len(hi):,}")

print("\n" + "=" * 92)
print("C. 크로스워크가 덮는 메인 행 비율")
print("=" * 92)
mapped = set(lk["pitcher_id"])
cov_all = np.isin(pid, list(mapped))
rec("C", "전체 행 커버리지", f"{100*cov_all.mean():.2f}%",
    f"{int(cov_all.sum()):,} / {len(df):,}")
for s in sorted(df["season"].unique()):
    m = season == s
    rec("C", f"  {s} 행 커버리지", f"{100*cov_all[m].mean():.2f}%",
        f"투수 {df.loc[m & cov_all, 'pitcher_id'].nunique()}"
        f" / {df.loc[m, 'pitcher_id'].nunique()}")

va = season == 2024
sub24 = df[va]
sub24_cov = cov_all[va]
rec("C", "2024 커버 구간 성공률", f"{sub24.loc[sub24_cov, TARGET].mean():.6f}")
rec("C", "2024 미커버 구간 성공률", f"{sub24.loc[~sub24_cov, TARGET].mean():.6f}")

print("\n" + "=" * 92)
print("D. TrackMan 투수-시즌 투구수 임계값별 (2019~2023 -> 2024 적용 기준)")
print("=" * 92)
tm_pre = tm[tm["season"] < 2024]
cnt = tm_pre.groupby(["pitcher_trackman_id", "season"]).size().rename("n").reset_index()
# 크로스워크 방향을 모르므로 TrackMan 쪽 자격만 먼저 세고, 메인 커버리지는
# 현행 매핑(336명)이 상한이라는 점을 함께 본다.
print(f"  {'임계':>6}{'자격 투수-시즌':>16}{'자격 투수':>12}{'TrackMan 투구':>16}"
      f"{'전체 대비':>12}")
for t in THRESHOLDS:
    e = cnt[cnt["n"] >= t]
    rec("D", f"TM season>={t}구",
        f"{e['pitcher_trackman_id'].nunique():,}명",
        f"투수-시즌 {len(e):,} / 투구 {e['n'].sum():,} "
        f"({100*e['n'].sum()/cnt['n'].sum():.1f}%)")

print("\n" + "=" * 92)
print("E. 커버리지 상한 — 크로스워크가 완벽해도 얼마까지인가")
print("=" * 92)
main_cnt = pd.Series(pid[season < 2024]).value_counts()
tm_pitchers_pre = tm_pre["pitcher_trackman_id"].nunique()
rec("E", "2019~2023 메인 투수", f"{main_cnt.size:,}")
rec("E", "2019~2023 TrackMan 투수", f"{tm_pitchers_pre:,}")
rec("E", "현행 매핑", f"{len(lk):,}",
    f"메인 대비 {100*len(lk)/main_cnt.size:.1f}%")
# 2024 행 중 '2019~2023 메인 이력이 있는' 행 = 매핑 가능 상한
has_hist = np.isin(pid[va], main_cnt.index.to_numpy())
rec("E", "2024 행 중 과거 메인 이력 보유", f"{100*has_hist.mean():.2f}%",
    "매핑이 완벽해도 이 이상은 못 덮는다")
rec("E", "현행 매핑의 2024 커버리지", f"{100*sub24_cov.mean():.2f}%",
    f"상한 대비 {100*sub24_cov.mean()/(100*has_hist.mean())*100:.1f}%")

out = pd.DataFrame(rows)
out.to_csv(OUT / "t1_trackman_mapping.csv", index=False)
print(f"\nsaved -> {OUT/'t1_trackman_mapping.csv'}")
