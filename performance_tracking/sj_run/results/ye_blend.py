# -*- coding: utf-8 -*-
"""ye_hand x sj_stdmlp 결합 — 정직 프로토콜: 2022 적합 -> 2024 동결 판정."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
WT = Path(sys.argv[1])
sys.path.insert(0, str(WT / "performance_tracking" / "tools"))
from common import load_labels
VAL = WT / "performance_tracking/val"
def bss(p,y):
    r=y.mean(); return 100000.0*(1.0-((p-y)**2).mean()/(r*(1.0-r)))
def get(n,s,L):
    d=pd.read_csv(VAL/f"{n}_{s}.csv"); m=L[["row_id"]].merge(d,on="row_id",how="left")
    assert m["pred"].notna().all(), f"{n}_{s}"; return m["pred"].to_numpy(np.float64)

D={}
for s in (2022,2024):
    L=load_labels(s); y=L["y"].to_numpy(np.float64); r=y.mean()
    a=get("sj_stdmlp",s,L); b=get("ye_hand",s,L)
    D[s]=(L,y,r,a,b)
    ea,eb=a-y,b-y
    print("val%d  단독 sj_stdmlp %8.1f · ye_hand %8.1f   확률ρ %.4f  오차ρ %.4f"
          % (s,bss(a,y),bss(b,y),np.corrcoef(a,b)[0,1],np.corrcoef(ea,eb)[0,1]))
print()
# 2022 에서 최적 lam 적합
L,y,r,a,b = D[2022]
lams=np.linspace(0,0.5,51)
sc=[bss(np.clip((1-l)*a+l*b,1e-6,1-1e-6),y) for l in lams]
best=int(np.argmax(sc)); lam=lams[best]
print("2022 적합: 최적 lam = %.3f  (2022 %8.1f -> %8.1f, Δ%+.1f)"
      % (lam,bss(a,y),sc[best],sc[best]-bss(a,y)))
# 2024 에서 동결 판정
L,y,r,a,b = D[2024]
base=bss(a,y); new=bss(np.clip((1-lam)*a+lam*b,1e-6,1-1e-6),y)
print("2024 동결 판정: %8.1f -> %8.1f   Δ%+.2f  -> %s"
      % (base,new,new-base,"통과" if new>base else "기각"))
print()
print("참고 — 2024 에서 직접 고른 lam (착시용, 판정 아님):")
sc24=[bss(np.clip((1-l)*a+l*b,1e-6,1-1e-6),y) for l in lams]
i=int(np.argmax(sc24)); print("   lam %.3f -> %8.1f (Δ%+.2f)" % (lams[i],sc24[i],sc24[i]-base))
