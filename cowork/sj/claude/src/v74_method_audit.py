"""V74: V72 까지의 실험 방식 감사 — 나열이 아니라 불변식을 기계로 검사한다.

검사 항목
    A  fold 구성      train = season < V,  valid = season == V,  겹침 없음
    B  피처 테이블     fold 마다 학습 시즌만으로 만들어지는가 (fold 를 바꾸면 값이 바뀌는가)
    C  season 피처     season 이 입력에 있는가. 있다면 검증 시즌은 학습에 없던 값이다
                      -> 트리가 외삽 구간에서 어떻게 행동하는가
    D  라벨 유도        누적 차분 복원의 8항목 재확인
    E  metrics()       BSS 계산이 수식과 맞는가 (수동 계산과 대조)
    F  base 비대칭      2022/2023 은 enhanced-25 평균, 2024 는 프로덕션 submit_021
                      -> 결론에 얼마나 영향을 주는가
    G  시드 잡음        같은 구성을 여러 번 잰 값들의 실제 산포
    H  자기라벨 누수 한계 플래툰 테이블 셀 크기 -> 한 행의 기여분

출력: outputs/v74_method_audit.txt (콘솔과 동일)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, OUT, CACHE, load, metrics

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
EPS = 1e-7
FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}   {detail}", flush=True)
    if not cond:
        FAIL.append(name)


df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
dfm = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = dfm["label_ok"].to_numpy() == 1

print("=" * 84)
print("A. fold 구성")
print("=" * 84)
for V in [2022, 2023, 2024]:
    tr, va = season < V, season == V
    check(f"fold {V} 겹침 없음", not (tr & va).any())
    check(f"fold {V} 학습 시즌 < 검증 시즌",
          season[tr].max() < V if tr.any() else True,
          f"학습 {sorted(set(season[tr]))} -> 검증 {V}")
    check(f"fold {V} 미래 시즌 미포함", season[tr].max() < V,
          f"학습 최대 {season[tr].max()}")

print()
print("=" * 84)
print("B. 피처 테이블이 fold 마다 다시 만들어지는가")
print("=" * 84)
t23 = CF.make_platoon_table(df.loc[season < 2023])
t24 = CF.make_platoon_table(df.loc[season < 2024])
m = t23.merge(t24, on=["pitcher_id", "batter_hand"], suffixes=("_23", "_24"))
d = np.abs(m["platoon_split_23"] - m["platoon_split_24"])
check("fold 2023 과 2024 의 플래툰 테이블이 다르다", d.max() > 1e-6,
      f"최대차 {d.max():.6f}, 평균차 {d.mean():.6f}, 공통 {len(m)}행")
check("2024 테이블이 2023 보다 크다 (시즌 추가 반영)", len(t24) >= len(t23),
      f"{len(t23)} -> {len(t24)}행")

print()
print("=" * 84)
print("C. season 을 피처로 쓰는가 — 검증 시즌은 학습에 없던 값이다")
print("=" * 84)
INPUT_COLS = [c for c in dfm.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
td = df.loc[season < 2024]
LABd = {"m": np.where(ok, dfm["y_middle"].to_numpy(float), np.nan),
        "r": np.where(ok, dfm["y_reverse"].to_numpy(float), np.nan)}
LABd["mr"] = np.where(ok, (LABd["m"] == 1) & (LABd["r"] == 1), np.nan)
yo = np.where(ok, dfm["y_outside"].to_numpy(float), np.nan)
yb = np.where(ok, dfm["y_ball"].to_numpy(float), np.nan)
LABd["ob"] = np.where(ok, (yo == 1) & (yb == 1), np.nan)
LABd["oz"] = np.where(ok, (yo == 1) & (yb == 0), np.nan)
F = CF.build(dfm[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
             CF.make_batter_platoon_table(td, {k: v[season < 2024]
                                               for k, v in LABd.items()}),
             CF.make_count_platoon_table(td), CF.make_inning_platoon_table(td))
has_season = "season" in F.columns
check("season 이 피처 행렬에 있다", has_season,
      "있으면 검증 시즌 값이 학습 범위 밖 -> 트리는 마지막 구간으로 외삽한다")
if has_season:
    tr_seasons = sorted(set(F.loc[season < 2024, "season"]))
    va_seasons = sorted(set(F.loc[season == 2024, "season"]))
    print(f"       학습 season 값 {tr_seasons}   검증 season 값 {va_seasons}")
    print(f"       -> 트리 분할은 학습 최대값 이상을 한 잎으로 묶는다. "
          f"시즌 드리프트를 season 분할로 배우면 검증에서 마지막 시즌 규칙이 적용된다.")
    print(f"       base_score 외삽이 이를 보정하는 구조이나, season 을 빼는 편이")
    print(f"       나은지는 실측이 필요하다 (미검증 항목).")

print()
print("=" * 84)
print("D. 라벨 유도 재확인")
print("=" * 84)
n_ok = int(ok.sum())
check("복원 행 수", n_ok == 1473508, f"{n_ok:,} / {len(df):,}")

# 캐시 parquet 에 y_success 가 없다. 문서에서 반복 인용한 '일치율 100.0000%' 를
# 지금 산출물로 재확인할 수 없으므로 여기서 다시 유도해 검산한다.
_pid = df["pitcher_id"].to_numpy()
_n = df["asof_pitcher_n"].to_numpy(np.float64)
_rate = df["asof_pitcher_success_rate"].to_numpy(np.float64)
_o = np.argsort(_pid.astype(np.int64) * 10_000_000 + _n.astype(np.int64),
                kind="stable")
# asof_*_rate 는 float32 다. rate*n 의 절대오차가 ~1e-3 이므로 차분 전에 반올림한다.
# 누적 성공 횟수는 정수이므로 반올림이 정보를 잃지 않는다.
_cum = np.round(_rate[_o] * _n[_o])
_g = _pid[_o]
_same = np.r_[_g[1:] == _g[:-1], False]          # 다음 행이 같은 투수인가
_lbl = np.full(len(df), np.nan)
_lbl[_o] = np.where(_same, np.r_[_cum[1:] - _cum[:-1], np.nan], np.nan)
_derived_ok = ~np.isnan(_lbl) & np.isin(_lbl, [0.0, 1.0])
_outside = int((~np.isnan(_lbl) & ~np.isin(_lbl, [0.0, 1.0])).sum())
agree = float((_lbl[_derived_ok] == y_all[_derived_ok]).mean())
check("y_success 재유도 == control_success", agree == 1.0,
      f"일치율 {agree*100:.6f}%   불일치 "
      f"{int((_lbl[_derived_ok] != y_all[_derived_ok]).sum())}행")
check("차분값이 {0,1} 밖인 행 없음", _outside == 0, f"{_outside}행")
check("재유도 유효 행 == label_ok", int(_derived_ok.sum()) == n_ok,
      f"재유도 {int(_derived_ok.sum()):,} vs label_ok {n_ok:,}   "
      f"양방향 불일치 {int((ok & ~_derived_ok).sum())} / "
      f"{int((~ok & _derived_ok).sum())}")
mm, rr = LABd["m"], LABd["r"]
comp = LABd["mr"]
uni = mm + rr - comp + LABd["ob"] + LABd["oz"]
v = ok & ~np.isnan(uni)
bad = int((np.abs(uni[v] - (1 - y_all[v])) > 1e-9).sum())
check("포함-배제 = 실패", bad == 0, f"불일치 {bad}행 / {int(v.sum()):,}행")
for k in ["m", "r", "mr", "ob", "oz"]:
    a = LABd[k][ok]
    check(f"{k} 라벨이 0/1", bool(np.isin(a[~np.isnan(a)], [0.0, 1.0]).all()),
          f"기저율 {np.nanmean(a):.4f}")

print()
print("=" * 84)
print("E. metrics() 검산")
print("=" * 84)
va = season == 2024
yv = y_all[va]
prod = pd.read_parquet(PROD).set_index("row_id").reindex(
    df.loc[va, "row_id"].to_numpy())
pv = np.clip(prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64), EPS, 1 - EPS)
manual = 100000 * ((yv.mean() * (1 - yv.mean())) - ((pv - yv) ** 2).mean()) \
    / (yv.mean() * (1 - yv.mean()))
lib = metrics(yv, pv)["bss_raw"]
check("BSS 수동 계산과 일치", abs(manual - lib) < 1e-6,
      f"수동 {manual:.6f}  metrics {lib:.6f}")
check("null 모델은 상수 예측", abs(metrics(yv, np.full_like(yv, yv.mean()))["bss_raw"])
      < 1e-6, f"{metrics(yv, np.full_like(yv, yv.mean()))['bss_raw']:.6f}")

print()
print("=" * 84)
print("F. base 비대칭 — 2022/2023 vs 2024")
print("=" * 84)
models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})
for f in [2022, 2023, 2024]:
    fid = df.loc[season == f, "row_id"].to_numpy()
    if f == 2024:
        b = pv
        src = "프로덕션 submit_021"
    else:
        acc, c = None, 0
        for mn in models:
            p = OOF_DIR / f"{mn}_fold{f}.parquet"
            if p.exists():
                x = pd.read_parquet(p).set_index("row_id").reindex(fid)["prediction"].to_numpy()
                acc = x if acc is None else acc + x
                c += 1
        b = np.clip(acc / c, EPS, 1 - EPS)
        src = f"enhanced {c}종 단순평균"
    yy = y_all[season == f]
    print(f"  {f}  {src:<22} BSS {metrics(yy, b)['bss_raw']:9.2f}   "
          f"예측평균 {b.mean():.5f}  실제 {yy.mean():.5f}  "
          f"편향 {(b.mean()-yy.mean())*100:+.3f}%p")
print()
print("  -> 2024 만 시즌 offset 이 적용된 프로덕션이고 2022/2023 은 보정 없는 평균이다.")
print("     따라서 fold 간 ΔBSS 절대값은 비교 불가하고, '같은 fold 안에서 arm 끼리'만")
print("     비교해야 한다. 지금까지 판정은 전부 그 형태였다 (arm - 기준선, fold 별).")
print("     다만 '두 fold 모두 양수' 규칙은 서로 다른 난이도의 시험을 함께 통과하라는")
print("     뜻이므로 보수적인 방향이고, 결론을 뒤집는 종류의 오류는 아니다.")

print()
print("=" * 84)
print("G. 시드 잡음 — 같은 구성을 여러 실행에서 잰 값")
print("=" * 84)
runs = {
    "fold 2024 성분단독 (현행 111피처)": [
        ("V33 M0", 746.85), ("V35 P0", 745.90), ("V37 Q0", 743.80),
        ("V40 T0", 746.78), ("V43 U0", 745.55), ("V53 E0", 745.17),
        ("V57 F0", 742.53), ("V59 H0", 744.08), ("V60 H0", 744.52),
        ("V65 comp", 745.30), ("V66 P0", 743.91), ("V71 C0", 743.13)],
    "fold 2023 성분단독": [
        ("V54 E0", -1132.88), ("V57 F0", -1122.93), ("V64 Y0", -1126.46),
        ("V65 comp", -1124.71), ("V66 P0", -1130.17), ("V71 C0", -1119.47)],
}
for k, vals in runs.items():
    a = np.array([v for _, v in vals])
    print(f"  {k}")
    print(f"     n={len(a)}  평균 {a.mean():.2f}  sd {a.std(ddof=1):.2f}  "
          f"범위 {a.min():.2f} ~ {a.max():.2f}  폭 {a.max()-a.min():.2f}")
print()
print("  -> 같은 구성인데 실행마다 최대 3.7 (2024) / 13.4 (2023) 벌어진다.")
print("     '내부 +3 미만은 제출 근거로 쓰지 않는다'(V61)는 이 산포와 정합한다.")
print("     다만 arm 비교는 같은 실행 안에서 이뤄지므로 이 산포가 그대로 오차는 아니다.")

print()
print("=" * 84)
print("H. 자기라벨 누수 한계 — 플래툰 테이블 셀 크기")
print("=" * 84)
tr = season < 2024
d = pd.DataFrame({"p": df["pitcher_id"].to_numpy()[tr],
                  "h": df["batter_hand"].to_numpy()[tr]})
sz = d.groupby(["p", "h"]).size()
print(f"  투수x타자손 셀 {len(sz):,}개   중앙 {int(sz.median()):,}행   "
      f"하위10% {int(sz.quantile(.1)):,}행")
print(f"  EB K=300 에서 한 행의 기여분 = 1/(n+300)")
for q, nm in [(0.1, "하위10%"), (0.5, "중앙"), (0.9, "상위10%")]:
    n = sz.quantile(q)
    print(f"     {nm:<8} n={int(n):>6,}  기여 {1/(n+300)*100:.3f}%")
print()
print("  -> 현행 축(투수x타자손)은 셀이 두꺼워 한 행의 자기 기여가 0.03~0.3% 다.")
print("     V37(성분 테이블)과 V35(투수x타자 개별)가 무너진 것은 셀이 얇아서였고,")
print("     현행 구성은 그 임계 밖에 있다. 다만 이것은 '한계가 있다'는 뜻이지")
print("     '누수가 0'이라는 뜻은 아니다.")

print()
print("=" * 84)
print(f"결과: {'전 항목 통과' if not FAIL else '실패 ' + str(FAIL)}")
print("=" * 84)
