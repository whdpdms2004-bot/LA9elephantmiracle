"""E) Optuna 튜닝 — SQLite 스터디로 이어달리기(45s 제한 대응).
실행: python run_optuna.py 8    # 8 trial씩 추가. 목표치 도달 시 최종 학습·저장.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
import csw_pipeline as P

optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = HERE / "out" / "v3"; OUT.mkdir(parents=True, exist_ok=True)
DB = "sqlite:////tmp/optuna_csw.db"
TARGET_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ADD = int(sys.argv[1]) if len(sys.argv) > 1 else 8

meta = json.loads((HERE / "cache/meta.json").read_text())
df = pd.read_parquet(HERE / "cache/features.parquet")
y = df["is_csw"].to_numpy()
tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = df["game_year"].eq(2019).to_numpy()
trp, tep = np.where(tr)[0], np.where(te)[0]
rng = np.random.default_rng(0)
X = P.build_matrix(df, meta["full_feats"])
ai, bi = list(GroupKFold(n_splits=4).split(trp, y[trp], df["game_pk"].to_numpy()[trp]))[0]
FIT, VAL = rng.choice(trp[ai], 18_000, replace=False), trp[bi]


def obj(t):
    pr = dict(n_estimators=t.suggest_int("n_estimators", 150, 400, step=50),
              learning_rate=t.suggest_float("learning_rate", 0.02, 0.12, log=True),
              num_leaves=t.suggest_int("num_leaves", 15, 127, log=True),
              min_child_samples=t.suggest_int("min_child_samples", 50, 800, log=True),
              subsample=t.suggest_float("subsample", 0.6, 1.0),
              colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
              reg_lambda=t.suggest_float("reg_lambda", 1e-2, 30, log=True),
              reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True))
    m = LGBMClassifier(**pr, random_state=0, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[FIT], y[FIT])
    return log_loss(y[VAL], m.predict_proba(X.iloc[VAL])[:, 1], labels=[0, 1])


st = optuna.create_study(direction="minimize", study_name="csw", storage=DB, load_if_exists=True,
                         sampler=optuna.samplers.TPESampler(seed=0))
done = len([t for t in st.trials if t.value is not None])
if done < TARGET_TRIALS:
    st.optimize(obj, n_trials=min(ADD, TARGET_TRIALS - done), show_progress_bar=False)
    done = len([t for t in st.trials if t.value is not None])
    print(f"trials={done}/{TARGET_TRIALS} best={st.best_value:.5f}")
if done >= TARGET_TRIALS:
    m = LGBMClassifier(**st.best_params, random_state=0, n_jobs=-1, verbose=-1)
    f = rng.choice(trp, 45_000, replace=False)
    m.fit(X.iloc[f], y[f])
    mt = P.metrics(y[tep], m.predict_proba(X.iloc[tep])[:, 1])
    (OUT / "E_optuna.json").write_text(json.dumps(
        {"best_params": st.best_params, "best_cv_logloss": round(st.best_value, 5), "n_trials": done,
         "test_2019": mt, "history": [round(t.value, 5) for t in st.trials if t.value is not None],
         "baselines": P.baselines(df, tr, te)}, ensure_ascii=False, indent=2, default=float))
    print(f"[FINAL] CV {st.best_value:.5f} → TEST {mt['logloss']:.4f} auc {mt['roc_auc']:.4f}")
