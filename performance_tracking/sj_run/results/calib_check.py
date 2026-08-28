# -*- coding: utf-8 -*-
"""148점 계기 불일치 검증 — cw 내부 3멤버 결합에서 apply_calibration 누락 여부."""
import io, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"C:\Users\isj67\Desktop\LA9elephantmiracle")
sys.path.insert(0, str(ROOT / "performance_tracking" / "tools"))
from common import load_labels

PREDS = ROOT / "cowork/sj/sj_final/preds"
VAL   = ROOT / "performance_tracking/val"
PAR   = json.load(io.open(ROOT / "cowork/cw/v17/model/params.json", encoding="utf-8"))

def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))

def apply_calibration(p_raw, params):
    eps = 1e-6
    p = np.clip(np.asarray(p_raw, dtype=np.float64), eps, 1 - eps)
    lg = np.log(p / (1 - p))
    z = params["logit_scale"] * (lg - params["logit_center_C0"]) + params["logit_target_C1"]
    q = 1.0 / (1.0 + np.exp(-z))
    lo = params["target_rate"] - params["cap"]
    hi = params["target_rate"] + params["cap"]
    return np.clip(q, max(eps, lo), min(1 - eps, hi))

RIDGE = 0.02
def fit(P, y):
    r = y.mean(); D = P - r
    M = D.T @ D / len(y); A = D.T @ (y - r) / len(y)
    M = M + RIDGE * np.trace(M) / len(M) * np.eye(len(M))
    return np.linalg.solve(M, A)

W_SUB2 = np.array([0.7103, 0.2356, 0.0693])
FOLDS = (2024, 2022)
CB = "GRID_idfreq__g_d6_l3k"

lab, raw, cal = {}, {}, {}
for f in FOLDS:
    L = load_labels(f); lab[f] = L
    y = L["y"].to_numpy(np.float64)
    cb = np.load(PREDS / f"{CB}_{f}.npy")
    ft = np.load(PREDS / f"S1_base__ft_{f}.npy")
    ml = np.load(PREDS / f"S1_base__mlp_{f}.npy")
    raw[f] = np.column_stack([cb, ft, ml])
    cal[f] = np.column_stack([
        apply_calibration(cb, PAR["model_cb"]),
        apply_calibration(ft, PAR["model_ft"]),
        apply_calibration(ml, PAR["model_mlp"]),
    ])

print("=" * 78)
print("0. 등록된 val 파일 실측 (기준점)")
print("=" * 78)
reg = {}
for f in FOLDS:
    p = pd.read_csv(VAL / f"sj_grid_w060_{f}.csv")
    L = lab[f]
    m = L[["row_id"]].merge(p, on="row_id", how="left")
    assert m["pred"].notna().all(), "row_id 불일치"
    reg[f] = m["pred"].to_numpy(np.float64)
    y = L["y"].to_numpy(np.float64)
    print("  val%d  등록 sj_grid_w060 = %9.1f   (범위 %.4f~%.4f)"
          % (f, bss(reg[f], y), reg[f].min(), reg[f].max()))

print()
print("=" * 78)
print("1. 멤버 예측이 raw 인가 calibrated 인가")
print("=" * 78)
lo = PAR["model_cb"]["target_rate"] - PAR["model_cb"]["cap"]
hi = PAR["model_cb"]["target_rate"] + PAR["model_cb"]["cap"]
print("  calibration 밴드 = [%.4f, %.4f]" % (lo, hi))
for f in FOLDS:
    for i, nm in enumerate(("cb", "ft", "mlp")):
        a = raw[f][:, i]
        out = ((a < lo) | (a > hi)).sum()
        print("    val%d %-4s min=%.4f max=%.4f  밴드밖 %6d행 (%.2f%%)  -> %s"
              % (f, nm, a.min(), a.max(), out, 100.0 * out / len(a),
                 "RAW" if out else "판정불가"))

print()
print("=" * 78)
print("2. 결합 BSS — 무보정(현재 코드) vs 보정후")
print("=" * 78)
w_raw = np.mean([fit(raw[f], lab[f]["y"].to_numpy(np.float64)) for f in FOLDS], axis=0)
w_cal = np.mean([fit(cal[f], lab[f]["y"].to_numpy(np.float64)) for f in FOLDS], axis=0)
print("  재적합 w (무보정) = cb %.4f ft %.4f mlp %.4f" % tuple(w_raw))
print("  재적합 w (보정후) = cb %.4f ft %.4f mlp %.4f" % tuple(w_cal))
print("  제출2 동결    w = cb %.4f ft %.4f mlp %.4f" % tuple(W_SUB2))
print()
hdr = "  %-6s %14s %14s %14s %14s" % ("폴드", "무보정·재적합", "무보정·동결w", "보정·재적합", "보정·동결w")
print(hdr); print("  " + "-" * (len(hdr) - 2))
res = {}
for f in FOLDS:
    y = lab[f]["y"].to_numpy(np.float64); r = y.mean()
    cells = []
    for P, w in ((raw[f], w_raw), (raw[f], W_SUB2), (cal[f], w_cal), (cal[f], W_SUB2)):
        cells.append(bss(np.clip(r + (P - r) @ w, 1e-6, 1 - 1e-6), y))
    res[f] = cells
    print("  val%-5d %14.1f %14.1f %14.1f %14.1f" % (f, *cells))

print()
print("=" * 78)
print("3. 목표 숫자와의 거리")
print("=" * 78)
TARGETS = {2024: {"등록val": 903.0, "배포순서재현": 893.7},
           2022: {"등록val": 2490.2, "배포순서재현": 2342.1}}
names = ["무보정·재적합", "무보정·동결w", "보정·재적합", "보정·동결w"]
for f in FOLDS:
    print("  val%d" % f)
    for tn, tv in TARGETS[f].items():
        best = min(range(4), key=lambda i: abs(res[f][i] - tv))
        print("    %-14s %8.1f  <- 가장 가까운 경로: %-14s %8.1f (Δ%+.1f)"
              % (tn, tv, names[best], res[f][best], res[f][best] - tv))
    print("    등록 val 파일 실측 %8.1f" % bss(reg[f], lab[f]["y"].to_numpy(np.float64)))
