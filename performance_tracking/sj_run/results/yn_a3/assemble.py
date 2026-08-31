# -*- coding: utf-8 -*-
"""yn A3 1단계 — 4구성을 고정 보정·고정 결합가중으로 조립해 판정."""
import io, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(r"C:\Users\isj67\Desktop\LA9elephantmiracle")
sys.path.insert(0, str(ROOT/"performance_tracking"/"tools"))
from common import load_labels
VAL   = ROOT/"performance_tracking/val"
PREDS = ROOT/"cowork/sj/sj_final/preds"
CWP   = json.load(io.open(ROOT/"cowork/cw/v17/model/params.json", encoding="utf-8"))
MLPM  = json.load(io.open(ROOT/"performance_tracking/models/sj_stdmlp/mlp_meta.json", encoding="utf-8"))
W_CW  = dict(cb=0.743, ft=0.160, mlp=0.165)          # sj_stdmlp 배포 내부가중
W_TEAM= dict(cw=0.4546, sj=0.6433, shift=0.003223)   # 배포 팀가중

def bss(p,y):
    r=y.mean(); return 100000.0*(1.0-((p-y)**2).mean()/(r*(1.0-r)))
def brier(p,y): return ((p-y)**2).mean()
def cal(p,q):
    eps=1e-6; p=np.clip(np.asarray(p,np.float64),eps,1-eps); lg=np.log(p/(1-p))
    z=q["logit_scale"]*(lg-q["logit_center_C0"])+q["logit_target_C1"]
    v=1.0/(1.0+np.exp(-z)); lo=q["target_rate"]-q["cap"]; hi=q["target_rate"]+q["cap"]
    return np.clip(v,max(eps,lo),min(1-eps,hi))
def getcsv(n,s,L):
    d=pd.read_csv(VAL/f"{n}_{s}.csv"); m=L[["row_id"]].merge(d,on="row_id",how="left")
    assert m["pred"].notna().all(), f"{n}_{s} 불일치"; return m["pred"].to_numpy(np.float64)

ARMS = {"BASE176":("YNA3_cb_base","YNA3_mlp_base"), "A_CB":("YNA3_cb_a3","YNA3_mlp_base"),
        "A_MLP":("YNA3_cb_base","YNA3_mlp_a3"),     "A_BOTH":("YNA3_cb_a3","YNA3_mlp_a3")}
out={}
for SEASON, SJ3 in ((2024,"sj3way"), (2022,"sj3way_nv")):
    L=load_labels(SEASON)
    if SEASON==2022:
        d3=pd.read_csv(VAL/f"{SJ3}_2022.csv"); L=L[L["row_id"].isin(d3["row_id"])].reset_index(drop=True)
    y=L["y"].to_numpy(np.float64); r=y.mean()
    gt=L["game_type"].to_numpy(); segs=[("all",np.ones(len(y),bool)),("R",gt=="R"),("F",gt=="F")]
    ft_raw=np.load(PREDS/f"S1_base__ft_{SEASON}.npy")
    if SEASON==2022 and len(ft_raw)!=len(L):
        Lfull=load_labels(2022); keep=Lfull["row_id"].isin(L["row_id"]).to_numpy(); ft_raw=ft_raw[keep]
    sj3=getcsv(SJ3,SEASON,L)
    for arm,(cbn,mln) in ARMS.items():
        cb=getcsv(cbn,SEASON,L); ml=getcsv(mln,SEASON,L)
        cw_raw = np.clip(r + W_CW["cb"]*(cb-r) + W_CW["ft"]*(ft_raw-r) + W_CW["mlp"]*(ml-r), 1e-6,1-1e-6)
        cw_cal = np.clip(r + W_CW["cb"]*(cal(cb,CWP["model_cb"])-r) + W_CW["ft"]*(cal(ft_raw,CWP["model_ft"])-r)
                           + W_CW["mlp"]*(cal(ml,MLPM["model_mlp"])-r), 1e-6,1-1e-6)
        for inst,cw in (("무보정",cw_raw),("배포보정",cw_cal)):
            team=np.clip(r + W_TEAM["cw"]*(cw-r) + W_TEAM["sj"]*(sj3-r) - W_TEAM["shift"], 1e-6,1-1e-6)
            for sn,m in segs:
                out[(SEASON,inst,arm,sn)]=(int(m.sum()),brier(team[m],y[m]),bss(team[m],y[m]),
                                            team[m].mean(),y[m].mean(),team[m].mean()-y[m].mean())
print("팀 결합 후 (cw 내부 %s · 팀 %s)" % (W_CW, W_TEAM))
for SEASON in (2024,2022):
    for inst in ("무보정","배포보정"):
        print("\n== val%d · %s" % (SEASON,inst))
        print("  %-8s %10s %10s %10s   %9s %9s %9s" % ("arm","all BSS","R BSS","F BSS","all brier","pred_m","bias"))
        for arm in ARMS:
            a=out[(SEASON,inst,arm,"all")]; R=out[(SEASON,inst,arm,"R")]; F=out[(SEASON,inst,arm,"F")]
            print("  %-8s %10.1f %10.1f %10.1f   %9.6f %9.4f %+9.4f" % (arm,a[2],R[2],F[2],a[1],a[3],a[5]))
        b=out[(SEASON,inst,"BASE176","all")][2]
        print("  Δ vs BASE176: " + " · ".join("%s %+.1f"%(k,out[(SEASON,inst,k,"all")][2]-b) for k in ARMS if k!="BASE176"))
print("\nn: 2024 all %d R %d F %d / 2022 all %d R %d F %d" % tuple(
    out[(s,"무보정","BASE176",sn)][0] for s in (2024,2022) for sn in ("all","R","F")))
