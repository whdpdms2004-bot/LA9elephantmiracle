# -*- coding: utf-8 -*-
"""제출용 — cw 의 CatBoost 를 **`id_freq` 8열을 붙여 전체 데이터로** 재학습하고
numpy(`cb.npz`)로 내보낸다. `ft`·`mlp` 는 건드리지 않는다.

## 왜 cb 만 바꾸나

챔피언 zip(`submit_cw_sj_final`)은 모듈 구조다:

```text
script.py                       팀 결합. p = r + w_cw(p_cw−r) + w_sj(p_sj−r) − shift
model/cw/{script.py, model/}    <- 여기만 손댄다
model/sj/{script.py, model/}    팀원 것. 그대로 둔다
```

cw 안에서 `ft`·`mlp` 는 **cb 와 같은 X 를 `prep_apply` 로 변환해 쓴다.** 열을 늘리면
셋 다 깨지므로, **cb 에만 176열을 주고 ft/mlp 에는 168 을 그대로 준다.**
그래서 FT 재학습(시드당 ~700초 × 8)이 필요 없다.

## `id_freq` 가 무엇이고 왜 큰가

ID 4열(투수·타자·양팀)의 **학습 빈도 log1p + 미출현 플래그** 8열. 원본 ID 는 그대로 둔다.

```text
cb 단독 val2024   881.8 -> 902.8   (+21.0)
```

실체는 표본 크기와 미출현 표시다 — **val2024 행의 19.86% 가 학습에 없던 투수**이고
(새 ID 81명), 그 행에서 원시 `pitcher_id` 는 트리가 본 적 없는 값이라 아무 분기나 탄다.
168 피처에는 `log_pitcher_n` 하나뿐이고 타자·팀에는 표본 크기 정보가 아예 없다.

행 독립성·시간 인과는 `check_atoms.py` 가 실증했다 (검증행 253,207개를 망가뜨려도
표본 300행 최대차 0.000e+00). 빈도표는 **학습행에서만** 만든다.

## 제출본은 학습 시즌이 다르다

val 실행은 시즌을 빼고 학습하지만(2019~2023 → 2024), **제출본은 2019~2024 전부**로
학습해 2025 를 예측한다. 그래서 이 스크립트가 따로 필요하다.

    python build_final_cb.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ROOT = FINAL.parents[2]
WORK = FINAL / "work"

sys.path.insert(0, str(ROOT / "cowork" / "cw" / "v17" / "src"))
sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import atoms as A                                                   # noqa: E402
from cb_export import cb_predict, export_catboost                   # noqa: E402
from run_arm import CB_P, load_base                                 # noqa: E402

ID_COLS = A.ID_COLS


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="cb.npz · idfreq_lut.npz 를 쓸 폴더")
    ap.add_argument("--seeds", type=int, default=3, help="cw params 의 seeds 와 같게")
    ap.add_argument("--atoms", default="id_freq")
    ap.add_argument("--cpu", action="store_true")
    # §29.2 조건부 재탐색의 결과를 넣는 자리. 동결값(depth 5 · l2 10000)은 **X168 기준**이고
    # `id_freq` 로 176 이 되면서 최적점이 이동했다. GRID_idfreq 가 잰 값을 여기로 넘긴다.
    ap.add_argument("--depth", type=int, default=0, help="0 이면 동결값 유지")
    ap.add_argument("--l2", type=float, default=0.0, help="0 이면 동결값 유지")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X, y, season, row_id = load_base()
    names = json.load(open(WORK / "meta.json", encoding="utf-8"))["names"]
    n = len(y)
    log("=" * 78)
    log("제출용 CatBoost — 전체 %s행 학습 (시즌 %d~%d) · 원자 %s · %d시드"
        % (f"{n:,}", int(season.min()), int(season.max()), a.atoms, a.seeds))
    log("=" * 78)

    # ── 원자 ────────────────────────────────────────────────────────────────
    # **전체 행이 학습행**이다 (제출본은 시즌을 빼지 않는다). 그래서 train_mask 는
    # 전부 True 이고, 빈도표도 전체에서 만든다. 이것이 배포 시점의 정직한 상태다 —
    # 2025 행은 이 빈도표를 조회만 하고 만드는 데 기여하지 않는다.
    tr = np.ones(n, bool)
    fold = int(season.max()) + 1                    # 2025
    E, en = A.build(X, names, tr, fold, [t.strip() for t in a.atoms.split("+") if t.strip()])
    Xall = np.ascontiguousarray(np.concatenate([np.asarray(X), E], axis=1))
    log("  원자 %d열 → 총 %d피처" % (len(en), Xall.shape[1]))
    del E

    # ── 추론용 룩업 저장 ────────────────────────────────────────────────────
    # 평가 서버는 numpy·pandas 뿐이라 `atoms.py` 를 못 돌린다.
    # ID -> log1p(빈도) 표를 내보내고, 추론에서는 조회만 한다.
    lut = {}
    for c in ID_COLS:
        col = np.asarray(X[:, names.index(c)], np.float64)
        u, cnt = np.unique(col, return_counts=True)
        lut["%s__key" % c] = u
        lut["%s__val" % c] = np.log1p(cnt.astype(np.float64))
        log("  룩업 %-18s 고유 %4d개  log1p 빈도 [%.2f, %.2f]"
            % (c, len(u), lut["%s__val" % c].min(), lut["%s__val" % c].max()))
    # ★ object 배열을 쓰지 않는다 — cw 의 `_load` 가 `allow_pickle=False` 로 열기 때문에
    # 문자열 배열이 들어가면 추론에서 터진다. 열 순서는 ID_COLS 고정 순서로 약속한다.
    np.savez_compressed(out / "idfreq_lut.npz", **lut)
    log("  → %s" % (out / "idfreq_lut.npz").name)

    # 룩업으로 만든 8열이 atoms.build 와 **비트단위로 같은지** 확인한다.
    # 다르면 학습과 추론이 어긋나므로 여기서 잡아야 한다.
    chk = []
    for c in ID_COLS:
        col = np.asarray(X[:, names.index(c)], np.float64)
        pos = np.searchsorted(lut["%s__key" % c], col)
        pos = np.clip(pos, 0, len(lut["%s__key" % c]) - 1)
        hit = lut["%s__key" % c][pos] == col
        chk.append(np.where(hit, lut["%s__val" % c][pos], 0.0))
        chk.append((~hit).astype(np.float64))
    chk = np.column_stack(chk).astype(np.float32)
    d = float(np.abs(chk - Xall[:, len(names):]).max())
    log("  룩업 재현 최대절대차 %.3e  %s" % (d, "통과" if d == 0 else "★실패"))
    if d != 0:
        sys.exit("[중단] 룩업이 학습 피처를 재현하지 못한다")

    # ── 학습 ────────────────────────────────────────────────────────────────
    from catboost import CatBoostRegressor, Pool
    P_CB = dict(CB_P)
    if a.depth:
        P_CB["depth"] = a.depth
    if a.l2:
        P_CB["l2_leaf_reg"] = a.l2
    if P_CB != CB_P:
        log("  ★ CB 하이퍼 변경  depth %s -> %s · l2 %s -> %s"
            % (CB_P["depth"], P_CB["depth"], CB_P["l2_leaf_reg"], P_CB["l2_leaf_reg"]))
    dev = dict(task_type="GPU", devices="0", border_count=128) if not a.cpu else {}
    blob = {}
    for sd in range(a.seeds):
        ts = time.time()
        m = CatBoostRegressor(**P_CB, random_seed=sd, **dev)
        m.fit(Pool(Xall, y.astype(np.float64)))
        b = export_catboost(m)
        # numpy 내보내기가 CatBoost 와 같은 값을 내는지 표본으로 확인
        idx = np.random.default_rng(0).choice(len(Xall), 20000, replace=False)
        ref = m.predict(Pool(Xall[idx]))
        got = cb_predict(Xall[idx], b)
        dd = float(np.abs(ref - got).max())
        log("  seed%d  %.0f초   numpy 내보내기 최대절대차 %.3e  %s"
            % (sd, time.time() - ts, dd, "통과" if dd < 1e-6 else "★실패"))
        if dd >= 1e-6:
            sys.exit("[중단] numpy 내보내기가 CatBoost 를 재현하지 못한다")
        for k, v in b.items():
            blob["s%d_%s" % (sd, k)] = v
        del m

    # cw 추론이 `int(cbz["n_seeds"][0])` 로 시드 수를 읽는다. 원본 형식을 맞춘다.
    blob["n_seeds"] = np.array([a.seeds], dtype=np.int64)
    np.savez_compressed(out / "cb.npz", **blob)
    sz = (out / "cb.npz").stat().st_size / 1e6
    log("  → cb.npz  %.1fMB" % sz)
    json.dump({"atoms": a.atoms, "extra_cols": en, "n_features": int(Xall.shape[1]),
               "depth": P_CB["depth"], "l2_leaf_reg": P_CB["l2_leaf_reg"],
               "seeds": a.seeds, "train_rows": int(n),
               "train_seasons": [int(season.min()), int(season.max())]},
              open(out / "idfreq_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("\n총 %.1f분" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
