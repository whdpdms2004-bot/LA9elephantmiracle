# -*- coding: utf-8 -*-
"""완성된 zip 안의 `blend_weights.json` · `model/cw/model/params.json` 을 고쳐 쓴다.

## 왜 별도 단계인가

`build_submit_zip.py` 는 **모델 파일만** 교체한다(cb.npz · idfreq_lut.npz · cw/script.py).
결합 가중은 모델이 아니라 정책이고, 모델을 다시 만들지 않고도 바꿀 수 있어야
가중만 다른 탐색본을 싸게 찍을 수 있다.

## 합 = 1 을 강제하는 이유

제출3 이 `w_cw 0.5 / w_sj 0.6433`(합 1.143) 로 나갔다가 1076.72 -> 1070 으로
**-6.7 떨어졌다.** 정직 하네스는 이 하락을 -6.1 로 미리 맞혔다. 합이 1 에서
멀어지면 결합 예측의 분산이 부풀고 `center_shift` 로는 평균만 맞출 뿐
분산은 못 되돌린다. 합을 1 로 묶으면 shift 자체가 필요 없다.

    python set_blend.py --zip <in.zip> --out <out.zip> --w cw=0.60,sj=0.40 --shift 0
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BW = "model/blend_weights.json"
PJ = "model/cw/model/params.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", required=True, help="cw=0.60,sj=0.40")
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    W = {}
    for tok in a.w.split(","):
        k, v = tok.split("=")
        W[k.strip()] = float(v)
    s = sum(W.values())
    print("가중 %s  합 %.4f" % (W, s))
    if abs(s - 1.0) > 1e-9 and a.shift == 0.0:
        print("  ★주의 합이 1 이 아닌데 shift 가 0 이다 — 평균이 어긋난다")

    z = zipfile.ZipFile(a.zip)
    bw = json.loads(z.read(BW))
    for lab in bw["bucket_labels"]:
        old = dict(bw["buckets"][lab]["w"])
        for k in old:
            if k not in W:
                sys.exit("[중단] 멤버 %s 의 새 가중이 없다" % k)
        bw["buckets"][lab]["w"] = {k: W[k] for k in old}
        print("  %-6s  %s  ->  %s" % (lab, old, bw["buckets"][lab]["w"]))
    bw["center_shift"] = a.shift
    if a.note:
        bw["blend_source"] = a.note

    pj = json.loads(z.read(PJ)) if PJ in z.namelist() else None
    if pj is not None and a.note:
        pj["team_blend_note"] = a.note

    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as o:
        for n in z.namelist():
            if n.endswith("/"):
                continue
            if n == BW:
                o.writestr(n, json.dumps(bw, ensure_ascii=False, indent=1))
            elif n == PJ and pj is not None:
                o.writestr(n, json.dumps(pj, ensure_ascii=False, indent=1))
            else:
                o.writestr(n, z.read(n))
    print("-> %s  (%.1fMB)  center_shift %.6f" % (out, out.stat().st_size / 1e6, a.shift))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
