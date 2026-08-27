# -*- coding: utf-8 -*-
"""`model/cw/script.py` 의 **MLP 입력**을 로버스트 z 점수 + 176열로 바꾼다.

이미 `patch_cw_script.py` 로 cb 에 `id_freq` 를 붙인 스크립트 위에 얹는다.

## 무엇을 바꾸나

```text
원본    Zm = prep_apply(X(168), bnds, True)          -> 336열, 분위수 순위
패치    Zm = std_apply([X | idfreq(8)], med, iqr)    -> 352열, 로버스트 z 점수
```

`cb` 는 이미 176열을 쓰고 `ft` 는 168 분위수 순위를 그대로 쓴다 — **셋이 서로 다른
표현을 본다.** 그것이 이 변경의 요점이다: 분위수 순위는 순서만 남기고 크기를 버리는데
트리도 순서만 쓰므로 순위 MLP 는 cb 와 정보가 겹친다. z 점수는 크기를 보존한다.

## 행 독립성

`med`·`iqr` 은 **학습 데이터에서만** 뽑은 상수표다. 추론은 그 행 자신의 값에
상수를 적용할 뿐이라 `predict(단독 행) == predict(전체)[i]` 가 유지된다.

    python patch_cw_mlp.py --script <patched script.py> --out <script.py>
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HELPER = '''

# ── [sj_final] 로버스트 z 점수 전처리 — mlp 전용 ────────────────────────────
# 분위수 순위 변환(`prep_apply`)은 **순서만 남기고 크기를 버린다.** 트리도 순서만
# 쓰므로 순위 MLP 는 cb 와 정보가 겹쳐 블렌드에서 가중 0.00 이 된다 (원시 예측·
# 월전방분할, cb depth 5~8 전부에서 실측). 크기를 보존하면 가중 0.15 가 붙고
# val2024 정방향 +12.3 · 역방향 +6.1 이다.
# med·iqr 은 학습 데이터에서만 뽑은 상수라 행 독립성에 영향이 없다.
def std_apply(X, med, iqr, with_mask=True):
    n, d = X.shape
    Z = np.empty((n, d * 2 if with_mask else d), dtype=np.float32)
    for j in range(d):
        col = X[:, j]
        ok = np.isfinite(col)
        z = (np.where(ok, col, med[j]) - med[j]) / iqr[j]
        Z[:, j] = np.clip(z, -4.0, 4.0).astype(np.float32)
        if with_mask:
            Z[:, d + j] = ok.astype(np.float32)
    return Z
'''

OLD_MLP = '''        print("Inference MLP...")
        Zm = prep_apply(X, bnds, True)'''

NEW_MLP = '''        print("Inference MLP...")
        # [sj_final] mlp 도 176열을 받고, 전처리는 로버스트 z 점수를 쓴다.
        # ft 는 아래·위에서 X(168) 분위수 순위를 그대로 쓴다 — 셋이 서로 다른 표현을 본다.
        Xml = np.ascontiguousarray(
            np.concatenate([X, idfreq_apply(test, iflut)], axis=1), dtype=np.float32)
        Zm = std_apply(Xml, stdp["med"], stdp["iqr"], True)
        assert Zm.shape[1] == (X.shape[1] + 8) * 2, "std_apply 열 수 불일치"
        print(" mlp 입력 %d열 (기본 %d + id_freq 8, 마스크 포함)" % (Zm.shape[1], X.shape[1]))
        del Xml'''

OLD_LOAD = '''    iflut = _load(MODEL_DIR, "idfreq_lut.npz")     # [sj_final] cb 전용 id_freq 룩업'''
NEW_LOAD = '''    iflut = _load(MODEL_DIR, "idfreq_lut.npz")     # [sj_final] cb·mlp 공용 id_freq 룩업
    stdp = _load(MODEL_DIR, "stdprep.npz")        # [sj_final] mlp 전용 로버스트 z 상수'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.script).read_text(encoding="utf-8")

    if "idfreq_apply" not in src:
        sys.exit("[중단] cb 패치가 안 된 스크립트다 — patch_cw_script.py 를 먼저 돌려라")
    if "std_apply" in src:
        sys.exit("[중단] 이미 mlp 패치된 스크립트다")
    for name, old in (("mlp 로드", OLD_LOAD), ("mlp 추론", OLD_MLP)):
        if old not in src:
            sys.exit("[중단] 패치 지점을 못 찾았다: %s" % name)

    src = src.replace(OLD_LOAD, NEW_LOAD)
    src = src.replace(OLD_MLP, NEW_MLP)
    marker = "\ndef main():"
    if marker not in src:
        sys.exit("[중단] def main() 을 못 찾았다")
    src = src.replace(marker, HELPER + marker, 1)

    Path(a.out).write_text(src, encoding="utf-8")
    ast.parse(src)
    print("패치 완료 → %s" % a.out)
    print("  로드부에 stdprep.npz 추가")
    print("  mlp 추론을 Xml(176열) + std_apply(352열) 로 교체")
    print("  ft 는 X(168) 분위수 순위 그대로 — 셋이 서로 다른 표현을 본다")
    print("  문법 검사 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
