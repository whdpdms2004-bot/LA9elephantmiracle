"""TabPFN(Prior-Fitted Network)로 CSW 예측 — 우리 파이프라인 캐시/피처/지표 재사용.

⚠️ 이 샌드박스(무 GPU, RAM 3GB, 라이선스 토큰 필요)에서는 사실상 실행 불가.
   → Colab 무료 GPU / 로컬 GPU 에서 실행하세요.
   설치:  pip install tabpfn            # 로컬/GPU (또는)  pip install tabpfn-client  # 클라우드 API
   최초 실행 시 라이선스 로그인 또는  export TABPFN_TOKEN=...  필요.

핵심 규칙(README 반영):
   · 전처리 금지: 스케일링·원핫 하지 말 것. 범주형은 category dtype 그대로, 결측 그대로.
   · predict는 매번 train을 다시 계산 → test는 1000행 청크로 예측.
   · 크기 한도(TabPFN-3): 1e6×200 / 1e5×2000 / 1e3×2e4. CPU는 ~1000행만 현실적.
   · 누수·분할·prequential 규칙은 기존과 동일(피처는 cache/features.parquet 그대로 사용).

실행:
   python tabpfn_demo.py global   --feat derived --sub-train 10000
   python tabpfn_demo.py pitcher  --feat basic   --min-train 500
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd
import csw_pipeline as P


def build_tabpfn_matrix(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """원핫 금지: 수치=float32(결측 유지), 범주형=category dtype 그대로.
    TabPFN이 범주형/결측을 네이티브 처리하므로 인코딩하지 않는다."""
    leaked = P.BANNED & set(feat_cols)
    assert not leaked, f"누수 열 포함: {sorted(leaked)}"
    X = df[feat_cols].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


def make_clf(device: str, seed: int = 0):
    import os
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # Windows 브라우저 로그인 크래시 방지 → 토큰 사용
    from tabpfn import TabPFNClassifier
    # ignore_pretraining_limits: 한도 초과 시 가드 해제(서브샘플과 함께 사용 권장)
    try:
        return TabPFNClassifier(device=device, ignore_pretraining_limits=True, random_state=seed)
    except TypeError:
        return TabPFNClassifier(device=device)  # 구버전 호환


def predict_chunked(clf, X, chunk=1000):
    out = np.empty(len(X), dtype=float)
    for i in range(0, len(X), chunk):
        out[i:i+chunk] = clf.predict_proba(X.iloc[i:i+chunk])[:, 1]
    return out


def run_global(feat="derived", sub_train=10000, device="auto", seed=0):
    meta = json.loads((HERE / "cache/meta.json").read_text())
    df = pd.read_parquet(HERE / "cache/features.parquet")
    cols = meta["basic_feats"] if feat == "basic" else meta["derived_feats"]
    # TabPFN은 target-encoding 컬럼(누수-안전 expanding)을 그대로 사용해도 되지만,
    # 순수 TabPFN 비교를 원하면 pitcher_te/batter_te 제외도 가능.
    X = build_tabpfn_matrix(df, cols)
    y = df["is_csw"].to_numpy()
    tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = df["game_year"].eq(2019).to_numpy()
    rng = np.random.default_rng(seed)
    tr_idx = np.where(tr)[0]
    if len(tr_idx) > sub_train:                      # 한도/속도 위해 train 서브샘플(층화 권장)
        tr_idx = rng.choice(tr_idx, sub_train, replace=False)
    dev = ("cuda" if _has_cuda() else "cpu") if device == "auto" else device
    print(f"[global/{feat}] device={dev} train={len(tr_idx)} test={int(te.sum())} feats={X.shape[1]}")
    clf = make_clf(dev, seed); clf.fit(X.iloc[tr_idx], y[tr_idx])
    p = predict_chunked(clf, X.iloc[np.where(te)[0]])
    print("TabPFN TEST:", P.metrics(y[te], p))
    print("baselines  :", {k: v["logloss"] for k, v in P.baselines(df, tr, te).items()})


def run_pitcher(feat="basic", min_train=500, device="auto", seed=0):
    meta = json.loads((HERE / "cache/meta.json").read_text())
    df = pd.read_parquet(HERE / "cache/features.parquet")
    cols = meta["basic_feats"] if feat == "basic" else meta["derived_feats"]
    X = build_tabpfn_matrix(df, cols); y = df["is_csw"].to_numpy()
    tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = df["game_year"].eq(2019).to_numpy()
    tr_idx, te_idx = df.index[tr], df.index[te]
    dev = ("cuda" if _has_cuda() else "cpu") if device == "auto" else device
    counts = df.loc[tr_idx].groupby("pitcher").size()
    elig = counts[counts >= min_train].index
    preds, ys = [], []
    for pid in elig:                                 # 소표본 → TabPFN의 강점 영역
        ptr = tr_idx[df.loc[tr_idx, "pitcher"].eq(pid).to_numpy()]
        pte = te_idx[df.loc[te_idx, "pitcher"].eq(pid).to_numpy()]
        if len(pte) == 0 or df.loc[ptr, "is_csw"].nunique() < 2:
            continue
        clf = make_clf(dev, seed); clf.fit(X.loc[ptr], df.loc[ptr, "is_csw"].to_numpy())
        preds.append(predict_chunked(clf, X.loc[pte])); ys.append(df.loc[pte, "is_csw"].to_numpy())
    p, yv = np.concatenate(preds), np.concatenate(ys)
    print(f"[pitcher/{feat}] device={dev} eligible={len(elig)} min_train={min_train}")
    print("TabPFN per-pitcher (weighted):", P.metrics(yv, p))


def _has_cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["global", "pitcher"])
    ap.add_argument("--feat", default="derived", choices=["basic", "derived"])
    ap.add_argument("--sub-train", type=int, default=10000)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    (run_global if a.mode == "global" else run_pitcher)(
        feat=a.feat, device=a.device,
        **({"sub_train": a.sub_train} if a.mode == "global" else {"min_train": a.min_train}))
