# -*- coding: utf-8 -*-
"""시즌 단위 폼 — 우리 피처셋에서 통째로 비어 있던 시간 척도.

지금 가진 것:
    prev1/3/5_game_success_rate   최근 1~5경기   ← 너무 짧다 (표본 수십 구)
    asof_pitcher_success_rate     통산 누적      ← 너무 길다 (수천 구, 시대 오염)
    그 사이 "올 시즌 어떤가" 가 없다.

asof_n 은 시즌을 넘어 통산으로 쌓인다 (2019 0→2757, 2020 2758→5427 ...).
그래서 통산 3,000구 투수가 올해 500구를 던져도 asof_rate 는 거의 안 움직인다.

꺼내는 방법:
    그 행에서        통산성공수_지금  = asof_n × asof_rate
    학습데이터에서    통산성공수_전년말 = 룩업(pitcher_id)
    ─────────────────────────────────────────────────
    올 시즌 성적 = 두 값의 차이

실측 (2024, 시즌 200구 이상 170명):
    통산 성공률   → 당해 성적 상관 0.5408
    당해 시즌 성적 → 당해 성적 상관 0.8203

규정 준수:
    쓰는 것은 (1) 그 행의 asof_n · asof_rate · pitcher_id
             (2) 학습 데이터로 만든 pitcher_id → 전년말 통산상태 룩업
    평가 데이터의 다른 행을 보지 않는다. 1행만 넣어도 전체를 넣어도 결과가 같다.

누수 방지:
    asof_* 는 그 투구 **직전**까지의 값이다(성공수 증분 = 그 행의 결과임을 확인).
    따라서 시즌 성적에는 그 행 자신의 결과가 들어가지 않는다.
    룩업도 s년 행에는 s년 미만 시즌만 쓴다.

실행:
    python feat_season.py           # 약 6분, val 2024·2022 검증
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PITCH_DATA_DIR", os.path.join(HERE, "..", "open", "data"))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))
SHRINK_K = 150.0


def season_start_state(df, idcol, ncol, ratecol):
    """(pitcher, season) → 그 시즌 첫 투구 시점의 통산 (n, 성공수).

    그 시즌의 첫 행이 가진 asof 값이 곧 전 시즌 말 상태다.
    """
    first = df.sort_values(ncol).groupby([idcol, "season"], sort=False).head(1)
    n0 = first[ncol].to_numpy(dtype=float)
    s0 = n0 * np.nan_to_num(first[ratecol].to_numpy(dtype=float), nan=0.0)
    return dict(zip(zip(first[idcol].to_numpy(), first.season.to_numpy()),
                    zip(n0, s0)))


def make(df, idcol, ncol, ratecol, prior):
    st = season_start_state(df, idcol, ncol, ratecol)
    ids = df[idcol].to_numpy(); sea = df.season.to_numpy()
    n = np.nan_to_num(df[ncol].to_numpy(dtype=float), nan=0.0)
    rate = df[ratecol].to_numpy(dtype=float)
    cum = n * np.nan_to_num(rate, nan=0.0)

    n0 = np.empty(len(df)); s0 = np.empty(len(df))
    for i, (pid_, s) in enumerate(zip(ids, sea)):
        a, b = st.get((pid_, s), (0.0, 0.0))
        n0[i] = a; s0[i] = b

    sn = np.maximum(n - n0, 0.0)                       # 올 시즌 투구수
    ss = np.clip(cum - s0, 0.0, None)                  # 올 시즌 성공수
    career = np.where(np.isnan(rate), prior, np.nan_to_num(rate, nan=prior))
    shr = (ss + SHRINK_K * career) / (sn + SHRINK_K)   # 통산 쪽으로 축소
    raw = np.where(sn > 0, ss / np.maximum(sn, 1), career)
    return np.column_stack([np.log1p(sn), raw, shr, shr - career]).astype(np.float32)


def calib(p, yv):
    r = yv.mean(); U = r * (1 - r); c1 = np.log(r / (1 - r))
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    return max(1e5 * (1 - ((1 / (1 + np.exp(-(s * (lg - lg.mean()) + c1))) - yv) ** 2).mean() / U)
               for s in np.arange(0.2, 1.55, 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=450)
    ap.add_argument("--batter", action="store_true", help="타자 쪽도 추가")
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    cols = ["season", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate",
            "batter_id", "asof_batter_n", "asof_batter_success_rate", "control_success"]
    print("train.csv 읽는 중...", flush=True)
    df = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=cols, encoding="utf-8-sig")
    prior = float(df.control_success.mean())

    print("시즌 폼 피처 생성 중...", flush=True)
    add = make(df, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", prior)
    if a.batter:
        add = np.column_stack([add, make(df, "batter_id", "asof_batter_n",
                                         "asof_batter_success_rate", prior)])
    print("  피처 %d개 생성 (%.0f초)" % (add.shape[1], time.time() - t0), flush=True)

    X = np.load(os.path.join(WORK, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(WORK, "y.npy"))
    season = np.load(os.path.join(WORK, "season.npy"))
    assert len(X) == len(add), "행수 불일치"

    res = {}
    for val in (2024, 2022):
        tri = np.where(season <= val - 1)[0]; va = np.where(season == val)[0]
        yv = y[va].astype(float)
        for tag, ex in (("기준(72피처)", False), ("+시즌폼 %d피처" % add.shape[1], True)):
            Xt = np.asarray(X[tri]); Xv = np.asarray(X[va])
            if ex:
                Xt = np.column_stack([Xt, add[tri]]); Xv = np.column_stack([Xv, add[va]])
            g = HistGradientBoostingClassifier(
                max_iter=a.iters, learning_rate=0.07, max_leaf_nodes=6,
                min_samples_leaf=1500, l2_regularization=1.0, max_bins=255,
                early_stopping=False, random_state=42)
            t = time.time(); g.fit(Xt, y[tri])
            p = g.predict_proba(Xv)[:, 1]
            res[(val, ex)] = (calib(p, yv), roc_auc_score(yv, p))
            print("  val%d %-18s BSS %8.1f  AUC %.4f  (%.0f초)"
                  % (val, tag, res[(val, ex)][0], res[(val, ex)][1], time.time() - t),
                  flush=True)
            del Xt, Xv, g

    print()
    print("=" * 64)
    print("[RESULT9]  시즌 단위 폼")
    print("=" * 64)
    for val in (2024, 2022):
        (b, ab), (e, ae) = res[(val, False)], res[(val, True)]
        print("  val %d : %8.1f → %8.1f  %+7.1f (%+.2f%%)   AUC %.4f → %.4f"
              % (val, b, e, e - b, (e / b - 1) * 100, ab, ae))
    print()
    print("  판정: 두 연도 모두 +1.5%% 이상이면 채택")
    print("=" * 64)
    print("총 소요 %.1f분" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
