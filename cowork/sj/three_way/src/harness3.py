"""3WAY 공용 하네스 — 타깃별 라벨, 타깃별 BSS, fold 규칙.

1WAY vs 3WAY
    1WAY  성분 5개(m, r, mr, ob, oz)가 같은 111피처를 공유하고
          트리 파라미터만 기저율에 따라 다르다. 합성은 포함-배제 항등식.
    3WAY  하위 확률마다 전처리·피처·모델을 따로 잡는다.
          결합은 항등식이 아니라 학습된 결합기이고, 최종 라벨로 미세조정한다.

타깃
    middle   0.1496   한가운데
    reverse  0.2290   반대 코스
    ball     0.3695   볼          <- 실패 유형이 아니라 교차 속성이다
    outside  0.1317   실패 & !m & !r   <- 항등식을 닫는 쪽
    success  0.5237   최종 라벨

    실패 = middle OR reverse OR outside 가 오차 0.000000 으로 성립한다.
    ball 은 outside=1 에서 82.4%, outside=0 에서 30.1% 로 걸쳐 있어 항등식을 닫지 못한다.
    그래서 outside 도 등록해 두고 결합 단계에서 M/R/B 와 M/R/O 를 실측 비교한다.

BSS 기준을 타깃마다 다시 잡는 이유
    BSS = 100000 x (1 - Brier / null),  null = p(1-p)
    타깃          null      배율 1/null
    success     0.2494        4.01
    middle      0.1272        7.86
    reverse     0.1766        5.66
    ball        0.2330        4.29
    같은 Brier 개선이라도 배율이 두 배 차이 난다. **타깃 간 BSS 직접 비교는 무효다.**
    타깃끼리 견주려면 bss_norm (= Brier 개선률) 을 쓴다.

fold 규칙 (1WAY 에서 확립한 것을 그대로 승계)
    2024  결정 fold
    2023  보조. bss_centered 로만 본다 (오프셋 교란)
    2022  쓰지 않는다 (2022->2024 순위상관이 두 계열 모두 음수)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TW = Path(__file__).resolve().parents[1]
SJ = TW.parent
REPO = SJ.parents[1]
CAMPAIGN = SJ / "feature_campaign_1000"
MODEL_OPT = SJ / "experiment" / "model_optimization"
LAB = SJ / "preprocess_lab"
CLAUDE_SRC = SJ / "claude" / "src"
for p in (MODEL_OPT, CAMPAIGN, LAB, CLAUDE_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = TW / "outputs"
SUCCESS = "control_success"
TARGETS = {"middle": "y_middle", "reverse": "y_reverse", "ball": "y_ball",
           "outside": "y_outside", "success": SUCCESS}
DECISION_FOLD = 2024
AUX_FOLD = 2023
BANNED_FOLD = 2022           # 역신호. 쓰지 않는다


def load_labeled() -> pd.DataFrame:
    """대회 프레임 + 복원 라벨. label_ok 플래그 포함."""
    from harness import CACHE, load
    df = load()
    lab = pd.read_parquet(CACHE / "failure_labels.parquet")
    return df.merge(lab, on="row_id", how="left", validate="one_to_one")


def target_vector(df: pd.DataFrame, target: str):
    """(값, 사용가능 마스크). success 는 전 행, 나머지는 label_ok 행만."""
    col = TARGETS[target]
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
    ok = (df["label_ok"].to_numpy() == 1) if target != "success" else np.ones(len(df), bool)
    return v, ok & ~np.isnan(v)


def bss(y, p, eps: float = 1e-7) -> dict:
    """타깃별 BSS. null 은 그 타깃 그 fold 의 기저율로 계산한다.

    bss_raw       그대로 채점
    bss_centered  예측 평균을 실제 평균에 맞춘 뒤 (평균 정렬 이득 제거)
    bss_norm      Brier 개선률 x 1000. null 배율을 나눠 타깃 간 비교가 가능하다
    """
    y = np.asarray(y, np.float64)
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    ybar = float(y.mean())
    null = ybar * (1 - ybar)
    brier = float(np.mean((p - y) ** 2))
    pc = np.clip(p - (float(p.mean()) - ybar), eps, 1 - eps)
    return {
        "n": int(len(y)), "target_mean": ybar, "pred_mean": float(p.mean()),
        "offset": float(p.mean()) - ybar, "null": null, "brier": brier,
        "bss_raw": 100000.0 * (1 - brier / null),
        "bss_centered": 100000.0 * (1 - float(np.mean((pc - y) ** 2)) / null),
        "bss_norm": 1000.0 * (1 - brier / null),
    }


def seed_noise(target: str) -> float:
    """시드 잡음 sd 추정. 1WAY fold2024 실측 1.37 을 null 배율로 환산한다.

    잡음은 Brier 스케일에서 비슷하므로 BSS 스케일에서는 1/null 에 비례한다.
    이보다 작은 차이는 같은 값으로 본다.
    """
    base_null = 0.2494          # success 의 null
    base_sd = 1.37
    nulls = {"middle": 0.1272, "reverse": 0.1766, "ball": 0.2330,
             "outside": 0.1144, "success": 0.2494}
    return base_sd * base_null / nulls.get(target, base_null)


def fold_masks(df: pd.DataFrame, fold: int):
    s = df["season"].to_numpy()
    if fold == BANNED_FOLD:
        raise ValueError(f"fold {BANNED_FOLD} 는 역신호라 쓰지 않는다 (METHOD 참조)")
    return s < fold, s == fold


def verdict(deltas: dict, target: str) -> str:
    """채택 판정. 2024 결정 / 2023 보조 / 잡음 폭 고려."""
    sd = seed_noise(target)
    d24 = deltas.get(DECISION_FOLD)
    if d24 is None:
        return "미확인"
    if d24 <= sd:
        return f"잡음 이내 (sd {sd:.2f})"
    d23 = deltas.get(AUX_FOLD)
    if d23 is not None and d23 < -sd:
        return "보조 fold 반대 방향 — 재검토"
    return "채택 후보"
