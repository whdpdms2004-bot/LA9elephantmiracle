"""항목 N (8~10월 블록) 진단 -- 빠진 피처인가, 신호 자체가 없는가.

## 배경

`group_by_perform/RESULTS.md` §4 가 8~10월 정규경기(R·late)를 **가장 큰 안정
열세**로 찍었다 (gap_w -158/-291 · head_w -56/-88 · 전체 행의 30%). 그런데
§6 이 "월도 이미 입력에 있다(game_month). '8월에 약하다'를 피처로 옮기면 이미
있는 열을 한 번 더 넣는 것" 이라며 피처가 아니라 진단 항목(N)으로 뺐고,
`PLAN_next` 에서 계속 '대기' 상태였다.

0-2 카운트는 빈칸이 이름으로 찍혔다 -- "asof_pitcher_success_rate 가 카운트를
섞은 평균이라 '이 투수가 0-2 에서 어떤가' 가 입력에 아예 없다". late 에도
그런 이름이 있는지 찾는 것이 이 스크립트다.

## 다섯 가지를 순서대로 본다

1. 치우침인가        블록별 평균 잔차 (y - p)
2. 잔차가 무엇과 도나 예측값 통제 후 부분상관 (6개 후보)
3. 신호가 있긴 한가   챔피언 AUC 와 **원시 asof 단독** AUC 를 나란히
4. 모델이 아는가     블록별 예측 표준편차 (모르면 평균으로 수축한다)
5. 구성 탓인가       asof_pitcher_n 5분위를 고정하고 early vs late

재학습 없이 등록 val 예측과 train.csv 만 쓴다 (CPU 몇 분).

실행:
    python cowork/hw/diag_late_season.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
PT = REPO / "performance_tracking"
W_CHAMP = np.array([0.45461, 0.64333])
CENTER_SHIFT = 0.003223
FOLDS = [(2024, "sj3way"), (2022, "sj3way_nv")]
USE = ["row_id", "season", "game_month", "game_type", "control_success",
       "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate",
       "asof_pitcher_prev5_game_success_rate", "asof_batter_n", "inning"]


def pred(name, season):
    p = PT / "val" / f"{name}_{season}.csv"
    return pd.read_csv(p)[["row_id", "pred"]].rename(columns={"pred": name})


def build(season, sj_member, raw):
    d = raw[raw.season == season].copy()
    for m in ("sj_stdmlp", sj_member):
        d = d.merge(pred(m, season), on="row_id", how="inner")
    r = d.control_success.mean()
    P = d[["sj_stdmlp", sj_member]].to_numpy(float)
    d["p"] = np.clip(r + (P - r) @ W_CHAMP - CENTER_SHIFT, 1e-6, 1 - 1e-6)
    d = d[d.game_type == "R"].copy()
    d["blk"] = np.where(d.game_month <= 5, "early",
                        np.where(d.game_month <= 7, "mid", "late"))
    return d


def resid_on(x, pv):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    b = np.polyfit(pv[m], x[m], 1)
    out = np.full(len(x), np.nan)
    out[m] = x[m] - np.polyval(b, pv[m])
    return out


def main():
    raw = pd.read_csv(REPO / "data" / "train.csv", usecols=USE)
    d24 = build(2024, "sj3way", raw)

    print("=" * 72)
    print("1·4. 블록별 치우침과 예측 표준편차 (val2024 · R)")
    print("=" * 72)
    d24["resid"] = d24.control_success - d24.p
    g = d24.groupby("blk").agg(n=("resid", "size"), bias=("resid", "mean"),
                               pred=("p", "mean"), true=("control_success", "mean"),
                               pred_std=("p", "std"))
    print(g.round(5).to_string())
    print("  -> late 의 bias 는 0 에 가깝다. 치우침 문제가 아니다.")

    print()
    print("=" * 72)
    print("2. late 잔차가 무엇과 도나 (예측값 통제 후 부분상관)")
    print("=" * 72)
    s = d24[d24.blk == "late"].copy()
    s["log_pn"] = np.log1p(s.asof_pitcher_n.fillna(0))
    s["form_gap"] = (s.asof_pitcher_prev5_game_success_rate
                     - s.asof_pitcher_success_rate)
    pv = s.p.to_numpy()
    ry = resid_on(s.resid, pv)
    for c in ["asof_pitcher_n", "log_pn", "asof_pitcher_success_rate",
              "form_gap", "asof_batter_n", "inning"]:
        rx = resid_on(s[c], pv)
        m = np.isfinite(rx) & np.isfinite(ry)
        print(f"  {c:36} {np.corrcoef(rx[m], ry[m])[0, 1]:+.4f}")
    print("  -> 전부 0.01 미만. 잡을 수 있는 잔여 구조가 안 보인다.")

    print()
    print("=" * 72)
    print("3. 신호가 있긴 한가 -- 챔피언 AUC 와 원시 asof 단독 AUC")
    print("=" * 72)
    print(f"  {'시즌':6}{'블록':7}{'n':>9}{'챔피언':>10}{'asof단독':>10}{'격차':>9}")
    for season, sjm in FOLDS:
        d = d24 if season == 2024 else build(season, sjm, raw)
        for b in ("early", "mid", "late"):
            x = d[d.blk == b]
            y = x.control_success.to_numpy(float)
            a = x.asof_pitcher_success_rate.to_numpy(float)
            m = np.isfinite(a)
            ac, aa = roc_auc_score(y, x.p), roc_auc_score(y[m], a[m])
            print(f"  {season:<6}{b:<7}{len(x):>9,}{ac:>10.4f}{aa:>10.4f}{ac-aa:>+9.4f}")
        print()
    print("  -> 원시 피처도 late 에서 같이 떨어진다. 모델이 못 쓰는 게 아니라")
    print("     쓸 정보가 적다. 챔피언의 '원시 대비 우위'도 late 에서 줄어든다.")

    print("=" * 72)
    print("5. 구성 탓인가 -- asof_pitcher_n 5분위 고정 후 early vs late")
    print("=" * 72)
    for season, sjm in FOLDS:
        d = d24 if season == 2024 else build(season, sjm, raw)
        d = d[d.blk.isin(["early", "late"])].copy()
        e = d.asof_pitcher_n.quantile([0, .2, .4, .6, .8, 1.0]).to_numpy()
        e[0] -= 1
        d["q"] = pd.cut(d.asof_pitcher_n, e, labels=list("12345"))
        print(f"  == {season} ==")
        num_e = num_l = wsum = 0.0
        for q in "12345":
            a = d[(d.q == q) & (d.blk == "early")]
            b = d[(d.q == q) & (d.blk == "late")]
            if len(a) < 500 or len(b) < 500:
                continue
            ae = roc_auc_score(a.control_success, a.p)
            al = roc_auc_score(b.control_success, b.p)
            print(f"    Q{q}  early {len(a):>7,} {ae:.4f} | late {len(b):>7,} "
                  f"{al:.4f}  {al-ae:+.4f}")
            num_e += ae * len(b); num_l += al * len(b); wsum += len(b)
        ae = roc_auc_score(d[d.blk == "early"].control_success, d[d.blk == "early"].p)
        al = roc_auc_score(d[d.blk == "late"].control_success, d[d.blk == "late"].p)
        print(f"    층가중 격차 {(num_l-num_e)/wsum:+.4f}  ·  층미보정 격차 {al-ae:+.4f}")
    print("  -> 층을 고정해도 격차가 그대로다. 투수 구성 차이가 아니다.")


if __name__ == "__main__":
    main()
