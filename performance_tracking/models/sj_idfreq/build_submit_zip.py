# -*- coding: utf-8 -*-
"""챔피언 zip 을 열어 `model/cw` 만 교체하고 다시 싼다. 나머지는 **바이트 그대로** 옮긴다.

## 왜 이 방식인가

챔피언(`submit_cw_sj_final`, Public 1072.865)은 모듈 구조다.

```text
script.py                      팀 결합.  p = r + w_cw(p_cw−r) + w_sj(p_sj−r) − shift
model/blend_weights.json       w_cw 0.4546 · w_sj 0.6433 · center_shift 0.003223
model/cw/{script.py, model/}   <- 여기만 바꾼다
model/sj/{script.py, model/}   팀원 것. **한 바이트도 안 건드린다**
```

팀원 코드를 안 건드리므로 그쪽이 검증한 동작이 그대로 남는다.
바꾸는 것은 cw 의 `cb.npz`(전체 데이터 + id_freq 로 재학습) · `script.py`(패치) ·
`idfreq_lut.npz`(추가) 셋뿐이다.

**결합 가중치는 그대로 둔다.** cw 가 좋아졌으니 w_cw 를 올리고 싶지만, 그러려면
val2024 로 재적합해야 하고 그건 이번 세션에서 정한 판정 규약(fit 2022·R → eval 2024)을
어긴다. 가중치를 건드리지 않는 것이 보수적이고, cw 개선분은 그대로 통과한다.

    python build_submit_zip.py --base <champion.zip> --cw <stage dir> --out <new.zip>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# stage 폴더의 이 파일들이 model/cw/model/ 안으로 들어간다
REPLACE = {"cb.npz": "model/cw/model/cb.npz",
           "idfreq_lut.npz": "model/cw/model/idfreq_lut.npz"}
SCRIPT = ("script_patched.py", "model/cw/script.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="원본 챔피언 zip")
    ap.add_argument("--cw", required=True, help="cb.npz · idfreq_lut.npz · script_patched.py 가 있는 폴더")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stage = Path(a.cw)

    need = list(REPLACE) + [SCRIPT[0]]
    miss = [f for f in need if not (stage / f).exists()]
    if miss:
        sys.exit("[중단] stage 에 없는 파일: %s" % miss)

    src = zipfile.ZipFile(a.base)
    names = src.namelist()
    swap = {v: (stage / k) for k, v in REPLACE.items()}
    swap[SCRIPT[1]] = stage / SCRIPT[0]
    for tgt in swap:
        if tgt not in names and tgt != "model/cw/model/idfreq_lut.npz":
            sys.exit("[중단] 원본에 없는 교체 대상: %s" % tgt)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_copy = n_swap = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n in names:
            if n in swap:
                z.write(swap[n], n)
                n_swap += 1
                print("  교체  %-34s <- %s" % (n, swap[n].name))
            else:
                z.writestr(src.getinfo(n), src.read(n))   # 원본 메타 유지
                n_copy += 1
        # 원본에 없던 새 파일
        for tgt, p in swap.items():
            if tgt not in names:
                z.write(p, tgt)
                n_swap += 1
                print("  추가  %-34s <- %s" % (tgt, p.name))

    print("\n원본 그대로 %d개 · 교체/추가 %d개" % (n_copy, n_swap))
    print("→ %s  (%.1fMB)" % (out, out.stat().st_size / 1e6))

    # sj 멤버가 바이트 그대로인지 확인한다 — 팀원 것을 건드리면 안 된다.
    # **원본을 새로 연다** — 쓰기 루프에서 쓰던 핸들을 재사용하면 헤더 위치가 어긋난다.
    src.close()
    z1 = zipfile.ZipFile(a.base)
    z2 = zipfile.ZipFile(out)
    bad = []
    for n in names:
        if n.startswith("model/sj/") and not n.endswith("/"):
            if hashlib.md5(z1.read(n)).hexdigest() != hashlib.md5(z2.read(n)).hexdigest():
                bad.append(n)
    print("sj 멤버 %d개 파일 무결성: %s"
          % (sum(1 for n in names if n.startswith("model/sj/") and not n.endswith("/")),
             "통과" if not bad else "★변경됨 %s" % bad[:3]))
    if bad:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
