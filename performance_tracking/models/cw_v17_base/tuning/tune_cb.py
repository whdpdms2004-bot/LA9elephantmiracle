# -*- coding: utf-8 -*-
"""CatBoost 구조 튜닝 — 피처가 아니라 모델 쪽을 판다.

오늘까지의 측정 결과
    배율 축 닫힘(+0.0) · 시드 축 닫힘(최대 +2.4) · 룩업 피처 5블록 전부 0 이하.
    잔차 스캔은 투수x월 336, 투수x좌우 289, 투수x이닝 239 가 남았다고 말하지만,
    feat_lut 이 증명했듯 "실재하는 신호"와 "작년 데이터로 가져올 수 있는 신호"는 다르다.
    그렇다면 남은 건 모델 자체다. 그리고 여기는 손을 안 댄 곳이 많다.

실험 (전부 현재 구성에서 한 번에 하나씩만 바꾼다)

  [손실함수]  Logloss 로 학습하는데 채점은 Brier 다.
              0/1 라벨에 RMSE 를 쓰면 그게 정확히 Brier 다.
              두 손실은 다른 걸 최적화한다 — Logloss 는 확신에 찬 오답을 더 세게 벌한다.

  [시즌가중]  성공률이 매년 떨어진다 (.565 -> .486 -> 2025 는 .474).
              투수간 산포도 2022 는 0.071, 2024 는 0.048 로 구조가 다르다.
              2025 는 2024 형이다. 그런데 2019~2022 를 2024 와 같은 무게로 배우고 있다.
              트리는 외삽을 못 하므로 season=2025 는 2024 와 같은 칸이다.
              최근 시즌에 무게를 주면 완화된다.

  [깊이/속도] depth 6 / 900트리 / lr 0.06 은 피처가 훨씬 적던 시절 값이다.
              168피처로 바꾼 뒤 한 번도 재조정하지 않았다.

  [오프셋]    잔차 스캔이 보여준 것: 전체 843점 중 대부분이 투수 주효과다.
              트리가 용량을 거기 다 쓴다. 투수 기준선을 baseline 으로 미리 주면
              트리는 상호작용만 학습한다. 잔차에 남은 200~340 이 그 상호작용이다.

측정 규율 (오늘 얻은 교훈)
    3시드는 기준선을 2.2% 흔들어 결과를 통째로 뒤집었다. 8시드 고정.
    기준선도 같은 시드로 매번 다시 잰다.
    val 2024 와 2022 두 해, min 규칙 (낮은 쪽이 LB 를 예측, 4-0 전적).

실행
    python tune_cb.py --gpu                    # 전체, 약 35분
    python tune_cb.py --gpu --only loss 시즌가중   # 일부만
"""

import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))

BASE = dict(iterations=900, depth=6, learning_rate=0.06, l2_leaf_reg=6.0,
            loss_function="Logloss")
SHRINK = 300.0          # 투수 기준선의 수축 강도


def calib(p, yv):
    """로짓 배율만 맞춰 BSS 를 잰다. 절편은 검증 평균에 고정."""
    r = yv.mean(); U = r * (1 - r); c1 = np.log(r / (1 - r))
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return max(1e5 * (1 - ((1 / (1 + np.exp(-(k * (z - z.mean()) + c1))) - yv) ** 2).mean() / U)
               for k in np.arange(0.2, 1.55, 0.02))


def season_weights(sea, mode):
    if mode == "none":
        return None
    age = sea.max() - sea                       # 2024 기준 0,1,2,...
    if mode == "exp85":
        return 0.85 ** age
    if mode == "exp70":
        return 0.70 ** age
    if mode == "exp50":
        return 0.50 ** age
    if mode == "drop1920":
        return (sea >= 2021).astype(float) + 1e-6
    if mode == "drop2022":
        return (sea != 2022).astype(float) + 1e-6
    if mode == "last3":
        return (sea >= sea.max() - 2).astype(float) + 1e-6
    raise ValueError(mode)


def pitcher_baseline(X, names, r_season):
    """투수의 현재 시즌 상태를 로짓으로 만들어 baseline 으로 준다.

    asof_pitcher_success_rate 는 그 투구 시점까지의 시즌 성공률이다.
    표본이 적으면 그 시즌 평균 쪽으로 수축한다.
    학습·추론 규칙이 같아야 한다 — 추론 때 시즌 평균은 target_rate 를 쓴다.
    """
    n = X[:, names.index("asof_pitcher_n")].astype(np.float64)
    rate = X[:, names.index("asof_pitcher_success_rate")].astype(np.float64)
    n = np.nan_to_num(n, nan=0.0)
    rate = np.where(np.isfinite(rate), rate, r_season)
    shr = (n * rate + SHRINK * r_season) / (n + SHRINK)
    shr = np.clip(shr, 1e-4, 1 - 1e-4)
    return np.log(shr / (1 - shr)).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    dev = dict(task_type="GPU", devices="0", border_count=128) if a.gpu else {}

    t0 = time.time()
    X = np.load(os.path.join(WORK, "X168.npy"), mmap_mode="r")
    y = np.load(os.path.join(WORK, "y.npy"))
    season = np.load(os.path.join(WORK, "season.npy"))
    names = json.load(open(os.path.join(WORK, "meta.json")))["names80"]

    # ── 실험 목록: (그룹, 이름, 파라미터변경, 가중모드, 오프셋여부) ──
    EXP = [
        ("loss",     "RMSE(=Brier)",   dict(loss_function="RMSE"),        "none",     False),
        ("시즌가중", "exp0.85",        {},                                "exp85",    False),
        ("시즌가중", "exp0.70",        {},                                "exp70",    False),
        ("시즌가중", "exp0.50",        {},                                "exp50",    False),
        ("시즌가중", "2019-20 제외",   {},                                "drop1920", False),
        ("시즌가중", "2022 제외",      {},                                "drop2022", False),
        ("깊이",     "depth 7",        dict(depth=7),                     "none",     False),
        ("깊이",     "depth 8",        dict(depth=8),                     "none",     False),
        ("학습률",   "lr.03 x1800",    dict(learning_rate=0.03, iterations=1800), "none", False),
        ("학습률",   "lr.02 x3000",    dict(learning_rate=0.02, iterations=3000), "none", False),
        ("정칙화",   "l2 = 20",        dict(l2_leaf_reg=20.0),            "none",     False),
        ("오프셋",   "투수기준선",     {},                                "none",     True),
        ("오프셋",   "투수기준선+d8",  dict(depth=8),                     "none",     True),

        # ── 2차: 1차 승자 셋을 합치고 그 방향으로 더 민다 ──────────────
        # 1차 결론: 모델이 과적합하고 있었다. 깊게 하면 무너지고(d8 -11.5%),
        # 천천히 배우면 좋아지고(lr.02 +3.9%), 정칙화를 키우면 좋아진다(l2 20 +1.1%).
        # 셋 다 같은 방향이므로 합쳤을 때와 더 갔을 때를 본다.
        ("R2", "lr02+RMSE",
         dict(learning_rate=0.02, iterations=3000, loss_function="RMSE"), "none", False),
        ("R2", "lr02+RMSE+l20",
         dict(learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=20.0), "none", False),
        ("R2", "lr03+RMSE+l20",
         dict(learning_rate=0.03, iterations=1800, loss_function="RMSE",
              l2_leaf_reg=20.0), "none", False),
        ("R2", "lr01x6000+R+l20",
         dict(learning_rate=0.01, iterations=6000, loss_function="RMSE",
              l2_leaf_reg=20.0), "none", False),
        ("R2", "lr02x5000+R+l20",
         dict(learning_rate=0.02, iterations=5000, loss_function="RMSE",
              l2_leaf_reg=20.0), "none", False),
        ("R2", "d5+lr02+R+l20",
         dict(depth=5, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=20.0), "none", False),
        ("R2", "lr02+R+l50",
         dict(learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=50.0), "none", False),
        ("R2", "lr02+R+l20+서브",
         dict(learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=20.0, bootstrap_type="Bernoulli", subsample=0.66),
         "none", False),

        # ── 3차 ────────────────────────────────────────────────────
        # 2차 결과: depth 6->5 로 내려도 더 좋아지고, l2 는 20 보다 50 이 낫다.
        # 방향이 아직 안 꺾였다. 더 단순하게 · 더 규제하게 쪽으로 계속 민다.
        # 또 하나: lr02x5000 만 혼자 -1.33% 였다. 같은 lr 에서 트리만 늘리면
        # 과적합한다. lr x 트리수 = 60 근처가 최적선이고 그 위는 손해다.
        # 3차는 그 곡선을 유지한 채 depth 와 l2 만 민다.
        ("R3", "d4+lr02+l20",
         dict(depth=4, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=20.0), "none", False),
        ("R3", "d5+lr02+l50",
         dict(depth=5, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=50.0), "none", False),
        ("R3", "d4+lr02+l50",
         dict(depth=4, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=50.0), "none", False),
        ("R3", "d5+lr02+l150",
         dict(depth=5, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=150.0), "none", False),
        ("R3", "d3+lr02+l50",
         dict(depth=3, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=50.0), "none", False),
        ("R3", "d5+lr01x6000+l50",
         dict(depth=5, learning_rate=0.01, iterations=6000, loss_function="RMSE",
              l2_leaf_reg=50.0), "none", False),
        ("R3", "d4+lr01x6000+l50",
         dict(depth=4, learning_rate=0.01, iterations=6000, loss_function="RMSE",
              l2_leaf_reg=50.0), "none", False),

        # ── 4차 ────────────────────────────────────────────────────
        # 3차 결론: depth 는 3 에서 꺾였다 -> 4~5 로 확정.
        # 그런데 l2 는 20 -> 50 -> 150 으로 계속 좋아진다. 아직 안 꺾였다.
        # 여기만 끝까지 민다. 꺾이는 지점을 찾으면 CatBoost 축은 닫힌다.
        ("R4", "d5+lr02+l400",
         dict(depth=5, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=400.0), "none", False),
        ("R4", "d5+lr02+l1000",
         dict(depth=5, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=1000.0), "none", False),
        ("R4", "d6+lr02+l150",
         dict(depth=6, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=150.0), "none", False),
        ("R4", "d4+lr01x6000+l150",
         dict(depth=4, learning_rate=0.01, iterations=6000, loss_function="RMSE",
              l2_leaf_reg=150.0), "none", False),
        ("R4", "d4+lr01x6000+l400",
         dict(depth=4, learning_rate=0.01, iterations=6000, loss_function="RMSE",
              l2_leaf_reg=400.0), "none", False),
        ("R4", "d5+lr01x6000+l150",
         dict(depth=5, learning_rate=0.01, iterations=6000, loss_function="RMSE",
              l2_leaf_reg=150.0), "none", False),

        # ── 5차 ────────────────────────────────────────────────────
        # 4차에서도 l2 가 안 꺾였다. 20 -> 50 -> 150 -> 400 -> 1000 계속 상승.
        # depth 는 6 이 5 보다 나빠 5 로 확정. 남은 축은 l2 하나뿐이다.
        # l2 를 극단으로 올리면 잎 값이 0 으로 강하게 수축돼 "아주 약한 트리 3000개"
        # 가 된다. 그 형태가 이 데이터에 맞는 듯하다. 끝까지 민다.
        # 잎이 너무 죽으면 트리를 늘려 보상해야 하므로 큰 l2 는 6000트리도 같이 본다.
        ("R5", "d5+lr02+l3000",
         dict(depth=5, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=3000.0), "none", False),
        ("R5", "d5+lr02+l10000",
         dict(depth=5, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=10000.0), "none", False),
        ("R5", "d5+lr02+l30000",
         dict(depth=5, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=30000.0), "none", False),
        ("R5", "d5+lr02x6000+l3000",
         dict(depth=5, learning_rate=0.02, iterations=6000, loss_function="RMSE",
              l2_leaf_reg=3000.0), "none", False),
        ("R5", "d5+lr04x3000+l3000",
         dict(depth=5, learning_rate=0.04, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=3000.0), "none", False),
        ("R5", "d6+lr02+l3000",
         dict(depth=6, learning_rate=0.02, iterations=3000, loss_function="RMSE",
              l2_leaf_reg=3000.0), "none", False),
    ]
    if a.only:
        EXP = [e for e in EXP if e[0] in a.only or e[1] in a.only]

    def run(val, chg, wmode, off, sd):
        tri = np.where(season <= val - 1)[0]
        va = np.where(season == val)[0]
        Xt = np.asarray(X[tri]); Xv = np.asarray(X[va])
        yt = y[tri]; yv = y[va].astype(float)
        w = season_weights(season[tri], wmode)
        prm = dict(BASE); prm.update(chg)
        prm.update(random_seed=sd, verbose=0, allow_writing_files=False,
                   thread_count=-1, **dev)
        bt = bv = None
        if off:
            r_tr = float(yt.mean()); r_va = float(yv.mean())
            bt = pitcher_baseline(Xt, names, r_tr)
            bv = pitcher_baseline(Xv, names, r_va)
        if prm["loss_function"] == "RMSE":
            m = CatBoostRegressor(**prm)
            m.fit(Pool(Xt, yt.astype(float), weight=w, baseline=bt))
            p = np.clip(m.predict(Pool(Xv, baseline=bv)), 1e-4, 1 - 1e-4)
        else:
            m = CatBoostClassifier(**prm)
            m.fit(Pool(Xt, yt, weight=w, baseline=bt))
            p = m.predict_proba(Pool(Xv, baseline=bv))[:, 1]
        del m, Xt, Xv
        return calib(p, yv)

    print("기준선부터 (%d시드)" % a.seeds, flush=True)
    ref = {}
    for val in (2024, 2022):
        ss = [run(val, {}, "none", False, sd) for sd in range(a.seeds)]
        ref[val] = (float(np.mean(ss)), float(np.std(ss, ddof=1)))
        print("  기준 val%d  %8.1f +- %4.1f  (%.1f분)"
              % (val, ref[val][0], ref[val][1], (time.time() - t0) / 60), flush=True)

    res = {}
    for grp, nm, chg, wmode, off in EXP:
        for val in (2024, 2022):
            ss = [run(val, chg, wmode, off, sd) for sd in range(a.seeds)]
            res[(nm, val)] = (float(np.mean(ss)), float(np.std(ss, ddof=1)))
            print("  %-14s val%d  %8.1f +- %4.1f  (%.1f분)"
                  % (nm, val, res[(nm, val)][0], res[(nm, val)][1],
                     (time.time() - t0) / 60), flush=True)

    print()
    print("=" * 76)
    print("[CatBoost 구조 튜닝]  기준 val2024 %.1f / val2022 %.1f"
          % (ref[2024][0], ref[2022][0]))
    print("=" * 76)
    print("%-9s %-14s %9s %8s %9s %8s %8s %8s"
          % ("그룹", "변경", "val2024", "t", "val2022", "t", "min%", "LB투영"))
    rows = []
    for grp, nm, chg, wmode, off in EXP:
        line = [grp, nm]
        gs, ts = [], []
        for val in (2024, 2022):
            b, sb = ref[val]; e, se = res[(nm, val)]
            sed = np.sqrt(sb ** 2 + se ** 2) / np.sqrt(a.seeds)
            g = e / b - 1
            t = (e - b) / sed if sed > 0 else 0.0
            gs.append(g); ts.append(t)
            line += ["%+.2f%%" % (100 * g), "%.1f" % t]
        mn = min(gs)
        line += ["%+.2f%%" % (100 * mn), "%.0f" % (990 * (1 + 0.62 * mn))]
        rows.append((mn, min(ts), nm, grp))
        print("%-9s %-14s %9s %8s %9s %8s %8s %8s" % tuple(line))

    rows.sort(reverse=True)
    print()
    print("상위 (min 이득 기준)")
    for mn, t, nm, grp in rows[:4]:
        mark = "채택" if (mn > 0 and t > 2) else ("보류" if mn > 0 else "기각")
        print("  %-14s %-9s min %+.2f%%  t %.1f   %s" % (nm, grp, 100 * mn, t, mark))
    print()
    print("t > 2 이고 두 해 모두 양수인 것만 채택한다.")
    print("여러 개가 살아남으면 조합해서 다시 잰다 (상호작용이 있을 수 있다).")
    print("=" * 76)
    print("총 %.1f분" % ((time.time() - t0) / 60))

    json.dump({"ref": {str(k): v for k, v in ref.items()},
               "res": {"%s|%d" % (k[0], k[1]): v for k, v in res.items()}},
              open(os.path.join(WORK, "tune_cb.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("저장: _work/tune_cb.json")


if __name__ == "__main__":
    main()
