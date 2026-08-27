# -*- coding: utf-8 -*-
"""챔피언 zip 의 `model/cw/script.py` 에 `id_freq` 8열을 넣는다. **cb 에만** 준다.

## 무엇을 바꾸나

원본 cw 추론은 168열 `X` 하나를 만들어 cb·ft·mlp 셋 다에 쓴다
(`ft`/`mlp` 는 `prep_apply(X, bnds, ...)` 로 변환). **열을 늘리면 셋 다 깨진다.**

그래서 `X` 는 168 그대로 두고 **cb 에만** 176열 `X_cb` 를 따로 만들어 넘긴다.
`ft`·`mlp` 의 가중치 파일과 `prep.npz` 는 손대지 않는다 → **FT 재학습이 필요 없다.**

```text
원본    X(168) -> cb_predict / prep_apply -> ft / mlp
패치    X(168) -> prep_apply -> ft / mlp          (그대로)
        X_cb = [X | idfreq(8)] -> cb_predict      (새로)
```

## 열 순서

학습(`atoms.build`)이 만든 순서와 **정확히 같아야** 한다. `build_final_cb.py` 가
`idfreq_lut.npz` 의 `__order` 에 그 순서를 저장하므로 추론이 그것을 따른다.

```text
pa_pitcher_id_logfreq · pa_pitcher_id_unseen ·
pa_batter_id_logfreq  · pa_batter_id_unseen  ·
pa_pitcher_team_id_logfreq · pa_pitcher_team_id_unseen ·
pa_batter_team_id_logfreq  · pa_batter_team_id_unseen
```

## 행 독립성

룩업은 **학습 데이터로만** 만든 상수표다. 추론은 그 행 자신의 ID 로 조회만 하므로
`predict(단독 행) == predict(전체)[i]` 가 유지된다. test 행끼리 집계하지 않는다.

    python patch_cw_script.py --script <cw/script.py> --out <patched script.py>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HELPER = '''

# ── [sj_final] id_freq — cb 전용 8열 ────────────────────────────────────────
# ID 4열(투수·타자·양팀)의 **학습 빈도 log1p + 미출현 플래그**.
# val2024 행의 19.86% 가 학습에 없던 투수라(새 ID 81명), 원시 `pitcher_id` 만으로는
# 트리가 본 적 없는 값에 아무 분기나 탄다. 이 8열이 표본 크기와 미출현을 명시한다.
#   cb 단독 val2024  881.8 -> 902.8  (+21.0)
# 룩업은 학습 데이터로만 만든 상수표이고 추론은 그 행 자신의 ID 로 조회만 한다.
IDFREQ_COLS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]


def idfreq_apply(df, lut):
    # 열 순서는 학습(`atoms.id_freq`)과 **고정 약속**이다. 룩업에 이름을 넣지 않는 것은
    # cw 의 `_load` 가 `allow_pickle=False` 라 object 배열을 못 읽기 때문이다.
    cols = IDFREQ_COLS
    out = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
        key = lut["%s__key" % c]
        val = lut["%s__val" % c]
        pos = np.clip(np.searchsorted(key, v), 0, len(key) - 1)
        hit = key[pos] == v
        out.append(np.where(hit, val[pos], 0.0))      # 미출현이면 0
        out.append((~hit).astype(np.float64))         # 미출현 플래그
    return np.column_stack(out).astype(np.float32)
'''

OLD_CB = '''    if len(X):
        print("Inference CatBoost...")
        acc = np.zeros(len(X))
        for i in range(n_cb):
            blob = {k[3:]: v for k, v in cbz.items() if k.startswith("s%d_" % i)}
            acc += cb_predict(X, blob)'''

NEW_CB = '''    if len(X):
        print("Inference CatBoost...")
        # [sj_final] cb 만 176열을 받는다. ft/mlp 는 아래에서 X(168) 을 그대로 쓴다.
        Xcb = np.ascontiguousarray(
            np.concatenate([X, idfreq_apply(test, iflut)], axis=1), dtype=np.float32)
        assert Xcb.shape[1] == X.shape[1] + 8, "id_freq 열 수 불일치"
        print(" cb 입력 %d열 (기본 %d + id_freq 8)" % (Xcb.shape[1], X.shape[1]))
        acc = np.zeros(len(X))
        for i in range(n_cb):
            blob = {k[3:]: v for k, v in cbz.items() if k.startswith("s%d_" % i)}
            acc += cb_predict(Xcb, blob)
        del Xcb'''

OLD_LOAD = '''    cbz = _load(MODEL_DIR, "cb.npz")'''
NEW_LOAD = '''    cbz = _load(MODEL_DIR, "cb.npz")
    iflut = _load(MODEL_DIR, "idfreq_lut.npz")     # [sj_final] cb 전용 id_freq 룩업'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.script).read_text(encoding="utf-8")

    for name, old in (("cb 로드", OLD_LOAD), ("cb 추론", OLD_CB)):
        if old not in src:
            sys.exit(f"[중단] 패치 지점을 못 찾았다: {name}")
    if "idfreq_apply" in src:
        sys.exit("[중단] 이미 패치된 스크립트다")

    src = src.replace(OLD_LOAD, NEW_LOAD)
    src = src.replace(OLD_CB, NEW_CB)

    # 헬퍼는 main() 앞에 넣는다
    marker = "\ndef main():"
    if marker not in src:
        sys.exit("[중단] def main() 을 못 찾았다")
    src = src.replace(marker, HELPER + marker, 1)

    Path(a.out).write_text(src, encoding="utf-8")
    import ast
    ast.parse(src)
    print("패치 완료 → %s" % a.out)
    print("  cb 로드부에 idfreq_lut 추가")
    print("  cb 추론부를 Xcb(176열) 로 교체 — ft/mlp 는 X(168) 그대로")
    print("  문법 검사 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
