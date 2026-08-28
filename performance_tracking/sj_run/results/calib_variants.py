# -*- coding: utf-8 -*-
"""배포순서재현 893.7 / 2342.1 을 어떤 보정 조합이 재현하는지 스윕."""
import io, json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"C:\Users\isj67\Desktop\LA9elephantmiracle")
sys.path.insert(0, str(ROOT / "performance_tracking" / "tools"))
from common import load_labels
PREDS = ROOT / "cowork/sj/sj_final/preds"
PAR = json.load(io.open(ROOT / "cowork/cw/v17/model/params.json", encoding="utf-8"))

def bss(p, y):
    r = y.mean(); return 100000.0*(1.0-((p-y)**2).mean()/(r*(1.0-r)))
def cal(p, q):
    eps=1e-6; p=np.clip(np.asarray(p,np.float64),eps,1-eps); lg=np.log(p/(1-p))
    z=q["logit_scale"]*(lg-q["logit_center_C0"])+q["logit_target_C1"]
    v=1.0/(1.0+np.exp(-z)); lo=q["target_rate"]-q["cap"]; hi=q["target_rate"]+q["cap"]
    return np.clip(v,max(eps,lo),min(1-eps,hi))
RIDGE=0.02
def fit(P,y):
    r=y.mean(); D=P-r; M=D.T@D/len(y); A=D.T@(y-r)/len(y)
    M=M+RIDGE*np.trace(M)/len(M)*np.eye(len(M)); return np.linalg.solve(M,A)
W_SUB2=np.array([0.7103,0.2356,0.0693]); FOLDS=(2024,2022); CB="GRID_idfreq__g_d6_l3k"
MLP_NEW=dict(PAR["model_mlp"]); MLP_NEW["logit_scale"]=0.80; MLP_NEW["logit_center_C0"]=-0.061225

lab={}; base={}
for f in FOLDS:
    L=load_labels(f); lab[f]=L
    base[f]=[np.load(PREDS/f"{CB}_{f}.npy"), np.load(PREDS/f"S1_base__ft_{f}.npy"), np.load(PREDS/f"S1_base__mlp_{f}.npy")]

TARGET={2024:893.7, 2022:2342.1}
variants=[]
for c_cb in (0,1):
  for c_ft in (0,1):
    for c_mlp in (0,1,2):
      variants.append((c_cb,c_ft,c_mlp))
rows=[]
for v in variants:
    cols={}
    for f in FOLDS:
        cb,ft,ml=base[f]
        cb2 = cal(cb,PAR["model_cb"]) if v[0] else cb
        ft2 = cal(ft,PAR["model_ft"]) if v[1] else ft
        ml2 = ml if v[2]==0 else (cal(ml,PAR["model_mlp"]) if v[2]==1 else cal(ml,MLP_NEW))
        cols[f]=np.column_stack([cb2,ft2,ml2])
    w=np.mean([fit(cols[f],lab[f]["y"].to_numpy(np.float64)) for f in FOLDS],axis=0)
    out={}
    for f in FOLDS:
        y=lab[f]["y"].to_numpy(np.float64); r=y.mean()
        out[(f,"refit")]=bss(np.clip(r+(cols[f]-r)@w,1e-6,1-1e-6),y)
        out[(f,"frozen")]=bss(np.clip(r+(cols[f]-r)@W_SUB2,1e-6,1-1e-6),y)
    for wn in ("refit","frozen"):
        d24=out[(2024,wn)]-TARGET[2024]; d22=out[(2022,wn)]-TARGET[2022]
        rows.append((v,wn,out[(2024,wn)],out[(2022,wn)],d24,d22,abs(d24)+abs(d22)/10.0))
rows.sort(key=lambda t:t[6])
lbl={0:"raw",1:"cal",2:"cal_new"}
print("%-28s %-7s %9s %10s %8s %9s" % ("cb/ft/mlp","w","val2024","val2022","d2024","d2022"))
print("-"*80)
for v,wn,a,b,d1,d2,_ in rows[:10]:
    print("%-28s %-7s %9.1f %10.1f %+8.1f %+9.1f" % ("/".join(lbl[x] for x in v),wn,a,b,d1,d2))
print()
print("목표: val2024 893.7 · val2022 2342.1")
