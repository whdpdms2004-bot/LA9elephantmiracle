# -*- coding: utf-8 -*-
"""완성된 zip 안의 `model/cw/model/params.json` 을 고쳐 쓴다.

## 왜 별도 단계인가

`build_submit_zip.py` 는 **모델 파일만** 교체한다. cw 내부 결합 가중과 모델별
보정 상수는 모델이 아니라 정책이고, 모델을 다시 만들지 않고도 바꿀 수 있어야
가중만 다른 탐색본을 싸게 찍을 수 있다.

## cw 내부는 합 제약을 걸지 않는다

팀 결합층은 합=1 로 묶어야 한다 (제출3 이 합 1.143 으로 나가 Public −6.7).
**cw 내부는 반대다** — 팀 층에 `center_shift` 가 따로 있어 스케일을 흡수하기 때문에
자유 적합이 낫다. 실측:

    재적합 자유 (합 1.0676)   val2024 +22.3 · val2022 +11.0
    합=1 정규화               val2024 +19.7 · val2022 **−8.1**

    python set_cw_params.py --zip <in.zip> --out <out.zip> \
        --w cb=0.7432,ft=0.1598,mlp=0.1646 --mlp-cal <mlp_meta.json>
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

PJ = "model/cw/model/params.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", default="", help="cb=..,ft=..,mlp=..")
    ap.add_argument("--mlp-cal", default="", help="mlp_meta.json — model_mlp 상수를 갈아끼운다")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    z = zipfile.ZipFile(a.zip)
    pj = json.loads(z.read(PJ))

    if a.w:
        W = {}
        for tok in a.w.split(","):
            k, v = tok.split("=")
            W[k.strip()] = float(v)
        for k in ("cb", "ft", "mlp"):
            if k in W:
                old = pj["blend_w_%s" % k]
                pj["blend_w_%s" % k] = W[k]
                print("  blend_w_%-4s %.4f -> %.4f" % (k, old, W[k]))
        print("  합 %.4f" % sum(pj["blend_w_%s" % k] for k in ("cb", "ft", "mlp")))

    if a.mlp_cal:
        new = json.load(open(a.mlp_cal, encoding="utf-8"))["model_mlp"]
        old = pj.get("model_mlp", {})
        pj["model_mlp"] = new
        print("  model_mlp  logit_scale %.4f -> %.4f · C0 %.6f -> %.6f"
              % (old.get("logit_scale", 0), new["logit_scale"],
                 old.get("logit_center_C0", 0), new["logit_center_C0"]))

    if a.note:
        pj["blend_source"] = (pj.get("blend_source", "") + " | " + a.note)[:2000]

    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as o:
        for n in z.namelist():
            if n.endswith("/"):
                continue
            o.writestr(n, json.dumps(pj, ensure_ascii=False, indent=1)
                       if n == PJ else z.read(n))
    print("-> %s  (%.1fMB)" % (out, out.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
