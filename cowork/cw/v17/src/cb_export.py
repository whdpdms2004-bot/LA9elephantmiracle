# -*- coding: utf-8 -*-
"""CatBoost → 순수 numpy 내보내기.

평가 서버에는 numpy/pandas 밖에 없다. 그런데 CatBoost 는 oblivious tree(대칭 트리)라
내보내기가 sklearn 보다 오히려 쉽다.

    깊이 d 트리 = (피처, 임계값) d쌍  +  리프값 2^d개
    예측 = 비트를 세워 인덱스를 만들고 리프값을 읽는다

    idx = Σ_i 2^i · [ x[f_i] > border_i ]

NaN 은 비교가 항상 False 라 0비트로 간다 — CatBoost 기본 nan_mode="Min" 과 일치한다.
범주형(cat_features)을 쓴 모델은 CTR 계산이 필요해 이 방식으로 못 내보낸다.

사용:
    from cb_export import export_catboost, cb_predict
    blob = export_catboost(model)          # dict of numpy arrays
    np.savez_compressed("model/cb.npz", **blob)
    p = cb_predict(X, blob)                # 확률
"""

import json
import os
import tempfile

import numpy as np


def export_catboost(model):
    """CatBoostClassifier → numpy 배열 묶음. 수치 피처 전용."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        model.save_model(path, format="json")
        m = json.load(open(path, encoding="utf-8"))
    finally:
        os.remove(path)

    if "oblivious_trees" not in m:
        raise ValueError("oblivious_trees 가 없습니다. 범주형 피처를 쓴 모델은 지원하지 않습니다.")
    ff = m["features_info"].get("float_features", [])
    if m["features_info"].get("categorical_features"):
        raise ValueError("범주형 피처가 있습니다. CTR 때문에 numpy 내보내기가 불가능합니다.")

    # float_feature_index → 원본 컬럼 인덱스
    flat = {i: f["feature_index"] for i, f in enumerate(ff)}

    feats, borders, leaves, depths = [], [], [], []
    for t in m["oblivious_trees"]:
        sp = t["splits"]
        for s in sp:
            if s.get("split_type") not in (None, "FloatFeature"):
                raise ValueError("FloatFeature 가 아닌 분기: %s" % s.get("split_type"))
            feats.append(flat[s["float_feature_index"]])
            borders.append(float(s["border"]))
        depths.append(len(sp))
        lv = np.asarray(t["leaf_values"], dtype=np.float64).ravel()
        assert len(lv) == 2 ** len(sp), "리프 수 불일치"
        leaves.append(lv)

    sb = m.get("scale_and_bias", [1.0, [0.0]])
    scale = float(sb[0])
    bias = float(sb[1][0] if isinstance(sb[1], (list, tuple)) else sb[1])

    # 연결함수. Logloss 로 학습하면 raw 가 로짓이라 시그모이드를 씌워야 하고,
    # RMSE(=Brier) 로 학습하면 raw 자체가 확률이라 씌우면 안 된다.
    # 튜닝 결과 RMSE 가 이겨서(그게 평가지표 그 자체다) 두 경우를 다 지원한다.
    link = 1 if _is_identity(model) else 0

    return dict(
        cb_feature=np.asarray(feats, dtype=np.int32),
        cb_border=np.asarray(borders, dtype=np.float64),
        cb_depth=np.asarray(depths, dtype=np.int32),
        cb_leaf=np.concatenate(leaves),
        cb_leaf_off=np.asarray(np.r_[0, np.cumsum([len(l) for l in leaves])], dtype=np.int64),
        cb_split_off=np.asarray(np.r_[0, np.cumsum(depths)], dtype=np.int64),
        cb_scale=np.asarray([scale]),
        cb_bias=np.asarray([bias]),
        cb_link=np.asarray([link], dtype=np.int64),
    )


def _is_identity(model):
    """RMSE/MAE 등 회귀 손실이면 raw 가 곧 예측값이다."""
    try:
        loss = str(model.get_params().get("loss_function", "Logloss"))
    except Exception:
        loss = "Logloss"
    return loss.split(":")[0] in ("RMSE", "MAE", "Quantile", "Huber", "Poisson", "Tweedie")


def cb_raw(X, b, chunk=200000):
    """로짓(raw score)."""
    feat = b["cb_feature"]; bord = b["cb_border"]
    leaf = b["cb_leaf"]; loff = b["cb_leaf_off"]; soff = b["cb_split_off"]
    ntree = len(soff) - 1
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    for s0 in range(0, n, chunk):
        e0 = min(n, s0 + chunk)
        Xb = np.asarray(X[s0:e0], dtype=np.float64)
        acc = np.zeros(e0 - s0, dtype=np.float64)
        for t in range(ntree):
            a, z = int(soff[t]), int(soff[t + 1])
            idx = np.zeros(e0 - s0, dtype=np.int64)
            for k, j in enumerate(range(a, z)):
                # NaN > border 는 False → 0비트 (CatBoost nan_mode="Min" 과 동일)
                idx |= (Xb[:, feat[j]] > bord[j]).astype(np.int64) << k
            acc += leaf[int(loff[t]):int(loff[t + 1])][idx]
        out[s0:e0] = acc
    return out * float(b["cb_scale"][0]) + float(b["cb_bias"][0])


def cb_predict(X, b, chunk=200000):
    z = cb_raw(X, b, chunk)
    # cb_link 가 없는 예전 모델(Logloss)은 0(로짓)으로 본다.
    if int(np.asarray(b.get("cb_link", [0])).ravel()[0]) == 1:
        return np.clip(z, 1e-6, 1.0 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ─────────────────────────── 자체 검증 ───────────────────────────

def verify(model, X, b=None, n=20000, tol=1e-6):
    """CatBoost 원본과 numpy 재현이 일치하는지 확인. 제출 전 필수."""
    b = b or export_catboost(model)
    Xs = np.asarray(X[:n], dtype=np.float64)
    if _is_identity(model):
        ref = np.clip(model.predict(Xs), 1e-6, 1.0 - 1e-6)
    else:
        ref = model.predict_proba(Xs)[:, 1]
    got = cb_predict(Xs, b)
    d = float(np.abs(ref - got).max())
    print("트리 %d개  분기 %d개  최대오차 %.3e  %s"
          % (len(b["cb_split_off"]) - 1, len(b["cb_feature"]), d,
             "통과" if d < tol else "★ 실패 — 제출 금지"))
    return d < tol, b


if __name__ == "__main__":
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    WORK = os.environ.get("PITCH_WORK_DIR", os.path.join(HERE, "_work"))
    sys.path.insert(0, HERE)
    from catboost import CatBoostClassifier

    X = np.load(os.path.join(WORK, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(WORK, "y.npy"))
    season = np.load(os.path.join(WORK, "season.npy"))
    tri = np.where(season <= 2023)[0][:300000]
    print("소규모 모델로 내보내기 경로 검증 중...", flush=True)
    m = CatBoostClassifier(iterations=60, depth=6, learning_rate=0.1, verbose=0,
                           allow_writing_files=False, random_seed=0)
    m.fit(np.asarray(X[tri]), y[tri])
    ok, _ = verify(m, np.asarray(X[np.where(season == 2024)[0][:20000]]))
    raise SystemExit(0 if ok else 1)
