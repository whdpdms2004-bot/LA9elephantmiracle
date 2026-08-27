# -*- coding: utf-8 -*-
"""[네 번째 표현 축] **관계적 표현** — 이 투수 자신의 분포 대비 얼마나 특이한가.

## 왜 이것인가

지금 cw 3멤버가 담은 표현은 전부 **행 단위 단조 변환**이다.

    cb   176열 원시값        트리 — 순서만 씀
    ft   168열 분위수 순위    순서만 씀 (정규화)
    mlp  176열 z 점수        크기 보존, 전역 선형 스케일

셋 다 `x -> f(x)` 꼴이고 `f` 가 단조다. [축2]·[항목F] 가 보인 것은
**표현 레버는 표현당 한 번**이라는 것이고, 위 셋으로 그 축은 다 덮였다.
`ft` 를 z 점수로 바꿔도 단독이 두 폴드 최고인데(+22.0/+41.6) 결합 이득이 0 이었다.

**근본적으로 다른 축은 단조 변환이 아닌 것이다.** 투수별 정규화는

    z_pitcher(x) = (x - mean_p) / std_p

로, **같은 원시값이라도 투수마다 다른 값**이 된다. 트리는 이걸 못 만든다
(투수 ID 로 분기해도 176개 피처마다 다른 중심을 잡아야 한다). z 점수도 못 만든다
(전역 상수 하나뿐). 순위도 못 만든다.

즉 이건 **다른 멤버가 구조적으로 접근할 수 없는 정보**다 — [축7 재검증]이
결합 유용성의 조건으로 지목한 바로 그것이다.

## 행 독립성 · 시간 인과

투수별 평균·표준편차는 **학습행에서만** 만든 상수표다 (`id_freq` 룩업과 같은 구조).
추론은 그 행 자신의 `pitcher_id` 로 조회만 하므로
`predict(단독 행) == predict(전체)[i]` 가 유지된다. 학습에 없던 투수는 0 으로 두고
미출현 플래그를 세운다 — val2024 행의 19.86% 가 여기 해당한다.

## 함께 보는 것

    plog:176   sign 보존 log1p 크기 압축 — 여전히 단조지만 기하가 다르다 (대조군)
    pz:176     투수별 z 점수 — 관계적 (본 후보)

`plog` 는 **단조 변환이면 안 된다는 가설의 대조군**이다. 단조인데도 이득이 나면
가설이 틀린 것이고, 안 나면 가설이 한 번 더 지지된다.

## 판정

현행 3멤버(cb2 + ft 원시 + std_mlp)에 **4번째로 얹어** 배포 순서로 잰다.
가중을 받고 2024 양방향 + 2022 이 비하락이어야 채택.

    python rep4.py --arms pz:176,plog:176
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
PT = ROOT / "performance_tracking"
sys.path.insert(0, str(ROOT / "cowork" / "cw" / "v17" / "src"))
sys.path.insert(0, str(FINAL / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

RIDGE = 0.02
MIN_N = 30          # 투수별 통계를 믿을 최소 학습행 수


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def fitw(M, y):
    r = y.mean()
    D = M - r
    A = D.T @ (y - r) / len(y)
    Q = D.T @ D / len(y)
    Q = Q + RIDGE * np.trace(Q) / len(Q) * np.eye(len(Q))
    return np.linalg.solve(Q, A)


def cal(p, q):
    eps = 1e-6
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    lg = np.log(p / (1 - p))
    o = 1.0 / (1.0 + np.exp(-(q["logit_scale"] * (lg - q["logit_center_C0"])
                              + q["logit_target_C1"])))
    return np.clip(o, max(eps, q["target_rate"] - q["cap"]),
                   min(1 - eps, q["target_rate"] + q["cap"]))


def group_stats(Xtr, pid_tr, gmed, giqr):
    """투수별 평균·표준편차. **학습행에서만** 만든다.

    정렬 후 구간합으로 낸다 — 176열 x 1.2M행에서 pandas groupby 보다 가볍다.

    ★ 표본이 적은 투수(MIN_N 미만)는 **전역 상수로 채운다.** 0/1 로 두면 그 행만
    z 점수가 아니라 원시값을 받아 같은 열 안에 두 가지 스케일이 섞인다.
    """
    order = np.argsort(pid_tr, kind="stable")
    ps = pid_tr[order]
    uniq, start = np.unique(ps, return_index=True)
    bounds = np.append(start, len(ps))
    d = Xtr.shape[1]
    mean = np.tile(gmed.astype(np.float64), (len(uniq), 1))
    std = np.tile(giqr.astype(np.float64), (len(uniq), 1))
    cnt = np.diff(bounds)
    for i in range(len(uniq)):
        sl = order[bounds[i]:bounds[i + 1]]
        if len(sl) < MIN_N:
            continue
        blk = np.asarray(Xtr[sl], np.float64)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(blk, axis=0)
            s = np.nanstd(blk, axis=0)
        mean[i] = np.nan_to_num(m)
        std[i] = np.where(np.isfinite(s) & (s > 1e-9), s, 1.0)
    return uniq, mean, std, cnt


def apply_pz(X, pid, uniq, mean, std, gmed, giqr):
    """투수별 z 점수 + 미출현 플래그. 조회만 한다 — 행 독립적이다."""
    n, d = X.shape
    pos = np.clip(np.searchsorted(uniq, pid), 0, max(len(uniq) - 1, 0))
    hit = (len(uniq) > 0) & (uniq[pos] == pid)
    Z = np.empty((n, d + 1), np.float32)
    Xd = np.asarray(X, np.float64)
    ok = np.isfinite(Xd)
    # 미출현 투수는 전역 통계로 대체한다 (버리면 그 행이 통째로 0 이 된다)
    mu = np.where(hit[:, None], mean[pos], gmed[None, :])
    sd = np.where(hit[:, None], std[pos], giqr[None, :])
    z = (np.where(ok, Xd, mu) - mu) / sd
    Z[:, :d] = np.clip(z, -4.0, 4.0).astype(np.float32)
    Z[:, d] = (~hit).astype(np.float32)
    return Z


def apply_plog(X, gmed, giqr):
    """sign 보존 log1p 크기 압축. 단조지만 기하가 다르다 (대조군)."""
    Xd = np.asarray(X, np.float64)
    ok = np.isfinite(Xd)
    u = (np.where(ok, Xd, gmed[None, :]) - gmed[None, :]) / giqr[None, :]
    Z = np.empty((len(Xd), X.shape[1] * 2), np.float32)
    Z[:, :X.shape[1]] = (np.sign(u) * np.log1p(np.abs(u))).astype(np.float32)
    Z[:, X.shape[1]:] = ok.astype(np.float32)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="pz:176,plog:176")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--zip", default=str(FINAL / "submit" / "submit_sj_stdmlp.zip"))
    a = ap.parse_args()
    arms = [t.split(":")[0].strip() for t in a.arms.split(",")]

    import importlib.util as iu
    sp = iu.spec_from_file_location("pt_common", PT / "tools" / "common.py")
    mod = iu.module_from_spec(sp)
    sp.loader.exec_module(mod)
    load_labels = mod.load_labels

    from run_arm import fit_torch, load_base
    import atoms as A

    X, y_all, season, row_id = load_base()
    names = json.load(open(FINAL / "work" / "meta.json", encoding="utf-8"))["names"]
    pi = names.index("pitcher_id")

    for fold in (2024, 2022):
        tri = np.where(season <= fold - 1)[0]
        va = np.where(season == fold)[0]
        y = load_labels(fold)["y"].to_numpy(np.float64)
        E, en = A.build(X, names, season <= fold - 1, fold, ["id_freq"])
        X176 = np.concatenate([np.asarray(X), E], axis=1)
        del E
        Xt = np.ascontiguousarray(X176[tri])
        Xv = np.ascontiguousarray(X176[va])
        pid_t = np.asarray(X176[tri][:, pi], np.float64)
        pid_v = np.asarray(X176[va][:, pi], np.float64)
        del X176

        # 전역 로버스트 상수 — 미출현 투수 대체용
        gmed = np.nanmedian(np.asarray(Xt, np.float64), axis=0)
        q1, q3 = np.nanpercentile(np.asarray(Xt, np.float64), [25, 75], axis=0)
        giqr = np.maximum(q3 - q1, 1e-6)
        gmed = np.nan_to_num(gmed)

        for kind in arms:
            fp = FINAL / "preds" / ("REP4_%s_%d.npy" % (kind, fold))
            if fp.exists():
                print("  fold%d %-5s (이미 있음, 단독 %.1f)"
                      % (fold, kind, bss(np.load(fp), y)), flush=True)
                continue
            t0 = time.time()
            if kind == "pzc":
                # ★ 수준과 특이도를 **둘 다** 준다.
                # `pz` 단독이 -461.6 으로 무너진 이유는 투수별 z 가 그 투수 자신의
                # 수준을 빼버리기 때문이다 — "좋은 투수는 좋다" 는 가장 강한 신호를
                # 통째로 버린다. 전역 z(수준) 옆에 투수별 z(특이도)를 붙이면
                # 모델이 둘을 다 본다. 트리도 전역 z 도 만들 수 없는 조합이다.
                from prep_mlp import apply_prep, make_prep
                uniq, mean, std, cnt = group_stats(Xt, pid_t, gmed, giqr)
                gp = make_prep(Xt, "std")
                Zt = np.hstack([apply_prep(Xt, gp, True),
                                apply_pz(Xt, pid_t, uniq, mean, std, gmed, giqr)])
                Zv = np.hstack([apply_prep(Xv, gp, True),
                                apply_pz(Xv, pid_v, uniq, mean, std, gmed, giqr)])
                print("  fold%d pzc  전역z %d열 + 투수별z %d열 = %d  (%.0f초)"
                      % (fold, 2 * Xt.shape[1], Xt.shape[1] + 1, Zt.shape[1],
                         time.time() - t0), flush=True)
            elif kind == "pz":
                uniq, mean, std, cnt = group_stats(Xt, pid_t, gmed, giqr)
                nok = int((cnt >= MIN_N).sum())
                Zt = apply_pz(Xt, pid_t, uniq, mean, std, gmed, giqr)
                Zv = apply_pz(Xv, pid_v, uniq, mean, std, gmed, giqr)
                unseen = float(Zv[:, -1].mean())
                print("  fold%d pz   투수 %d명 중 %d명이 %d행 이상 · 검증행 미출현 %.2f%%  (%.0f초)"
                      % (fold, len(uniq), nok, MIN_N, 100 * unseen, time.time() - t0),
                      flush=True)
            else:
                Zt = apply_plog(Xt, gmed, giqr)
                Zv = apply_plog(Xv, gmed, giqr)
            acc = np.zeros(len(va))
            for sd in range(a.seeds):
                acc += np.clip(fit_torch("mlp", Zt, y_all[tri], Zv, sd), 1e-6, 1 - 1e-6)
            p = acc / a.seeds
            np.save(fp, p)
            print("  fold%d %-5s 입력 %d열  %.0f초  단독 %.1f"
                  % (fold, kind, Zt.shape[1], time.time() - t0, bss(p, y)), flush=True)
            del Zt, Zv
        del Xt, Xv

    # ── 판정 ────────────────────────────────────────────────────────────────
    P = FINAL / "preds"
    par = json.loads(zipfile.ZipFile(a.zip).read("model/cw/model/params.json"))
    Y = {f: load_labels(f)["y"].to_numpy(np.float64) for f in (2024, 2022)}

    def mats(extra=None):
        out = []
        for f in (2024, 2022):
            cols = [cal(np.load(P / ("E2_var_cb2_a0.15_%d.npy" % f)), par["model_cb"]),
                    cal(np.load(P / ("FTX_q64_168_%d.npy" % f)), par["model_ft"]),
                    cal(np.load(P / ("PREP_std_176_%d.npy" % f)), par["model_mlp"])]
            if extra:
                cols.append(cal(np.load(P / ("%s_%d.npy" % (extra, f))), par["model_mlp"]))
            out.append((np.column_stack(cols), Y[f]))
        return out

    def sc(ms):
        w = np.mean([fitw(M, y) for M, y in ms], axis=0)
        return [bss(np.clip(y.mean() + (M - y.mean()) @ w, 1e-6, 1 - 1e-6), y)
                for M, y in ms], w

    b, wb = sc(mats())
    y24 = Y[2024]
    cb = np.load(P / "E2_var_cb2_a0.15_2024.npy")
    print("\n" + "=" * 92)
    print("[네 번째 표현 축] 현행 3멤버(cb2 + ft + std_mlp)에 4번째로 얹어 판정")
    print("=" * 92)
    print("%-26s %9s %9s   %9s %9s   %s"
          % ("4번째 멤버", "단독24", "rho(cb)", "Δ24", "Δ22", "w(4번째)"))
    print("-" * 92)
    print("%-26s %9s %9s   %9s %9s   w %.3f/%.3f/%.3f"
          % ("없음 (3멤버)", "-", "-", "기준", "기준", wb[0], wb[1], wb[2]))
    for kind in arms:
        f = "REP4_%s" % kind
        if not (P / ("%s_2024.npy" % f)).exists():
            continue
        s, w = sc(mats(f))
        p = np.load(P / ("%s_2024.npy" % f))
        tag = {"pz": "관계적(수준 제거)", "pzc": "관계적+수준",
               "plog": "단조(대조군)"}.get(kind, "")
        print("%-26s %9.1f %9.4f   %+9.1f %+9.1f   %.3f   %s"
              % (kind, bss(p, y24), np.corrcoef(p, cb)[0, 1],
                 s[0] - b[0], s[1] - b[1], w[3], tag))
    print("\n(가중을 받고 2024 · 2022 이 모두 비하락이어야 채택)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
