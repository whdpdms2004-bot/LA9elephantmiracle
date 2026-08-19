"""규정 위반을 구조적으로 막는 관문. 문서 경고가 아니라 코드가 거부한다.

막는 것 둘 (RULES.md)

  ㄱ 조항 2 — 평가 데이터의 분포·평균·순위로 예측값을 보정
      centered 예측 배열이 밖으로 나가지 못하게 한다.
      centering 은 오직 centered_bss() 안에서만 일어나고 float 만 반환한다.
      예측 저장은 save_prediction() 을 통해야 하고, 검증 라벨 평균과
      일치하도록 이동된 배열은 거부한다.

  ㄴ 조항 1 — test 행을 가로지르는 값을 피처로 사용
      행 간 연산(.shift/.rolling/.cumsum/groupby)으로 만든 값은
      TRAIN_ONLY 에 등록하고, assert_features_clean() 이 피처 목록에서 거부한다.

허용되는 것 — 헷갈리기 쉬워 명시한다

  ○ 학습 시즌에서 얻은 추세 (train_season_trend)
      season < fold 의 라벨 평균으로 만든 외삽값. 검증/테스트 라벨을 보지 않는다.
      전 행에 같은 상수로 적용되므로 행 독립성도 지킨다.
      1WAY 의 base_score 외삽, 3WAY 의 Pool baseline 이 이것이다.

  ○ 학습 행만으로 만든 룩업 테이블 (EB 평활, 빈도 인코딩)
      추론은 그 행의 키로 조인할 뿐이라 행 독립이다.

  ○ 학습 가중치 계산에 쓰는 행 간 연산
      가중치는 학습에만 영향을 주고 추론 경로에 없다.
      단, 그 값이 피처가 되면 즉시 위반이므로 TRAIN_ONLY 에 등록한다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


class RuleViolation(AssertionError):
    """규정 위반. 실행을 멈춘다."""


# ── ㄴ. 절대 피처가 될 수 없는 것들 ────────────────────────────────────────
# 행 간 연산으로 만들어져 단일 test 행에서 재현 불가능한 값.
TRAIN_ONLY = {
    "short_outing_mask": "등판 구간을 .shift()/groupby 로 잡는다. 학습 가중치 전용",
}
TRAIN_ONLY_PATTERNS = [
    re.compile(r"^short(_|$)"), re.compile(r"_short_outing"),
    re.compile(r"^outing_"), re.compile(r"_run_len"),
]
LABEL_PATTERNS = [
    re.compile(r"^y_"), re.compile(r"^label_ok$"),
    re.compile(r"^control_success$"),
]


def assert_features_clean(features, where: str = "") -> None:
    """피처 목록에 학습 전용 값이나 라벨이 섞였는지 검사한다."""
    bad_train = [f for f in features
                 if f in TRAIN_ONLY or any(p.search(f) for p in TRAIN_ONLY_PATTERNS)]
    bad_label = [f for f in features if any(p.search(f) for p in LABEL_PATTERNS)]
    if bad_train:
        raise RuleViolation(
            f"[조항1] 학습 전용 값이 피처에 들어갔다{' — ' + where if where else ''}: "
            f"{bad_train[:5]}\n"
            f"  행 간 연산으로 만든 값은 단일 test 행에서 재현할 수 없다.")
    if bad_label:
        raise RuleViolation(
            f"[조항4] 라벨이 피처에 들어갔다{' — ' + where if where else ''}: "
            f"{bad_label[:5]}")


def mark_train_only(name: str, reason: str) -> None:
    """행 간 연산으로 만든 새 값을 등록한다. 등록하면 피처가 될 수 없다."""
    TRAIN_ONLY[name] = reason


# ── ㄱ. 평가 라벨로 예측을 이동시킨 배열은 밖으로 못 나간다 ─────────────────
_EPS = 1e-7


def centered_bss(y, p, null: float) -> float:
    """예측 평균을 실제 평균에 맞춘 뒤의 BSS. **float 만 반환한다.**

    이동된 배열은 이 함수 밖으로 나가지 않는다. 지표 계산이 유일한 용도다.
    """
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    shifted = np.clip(p - (float(p.mean()) - float(y.mean())), _EPS, 1 - _EPS)
    return 100000.0 * (1.0 - float(np.mean((shifted - y) ** 2)) / null)


def save_prediction(path, pred, y_valid=None, *, where: str = "") -> None:
    """예측 저장의 유일한 경로. 검증 라벨 평균에 맞춰진 배열은 거부한다.

    y_valid 를 주면 pred 의 평균이 검증 라벨 평균과 지나치게 같은지 본다.
    모델이 우연히 잘 맞춘 경우와 인위적 이동을 구분하려고 임계를 매우 좁게 둔다.
    """
    pred = np.asarray(pred, np.float64)
    if pred.ndim != 1:
        raise RuleViolation(f"예측은 1차원이어야 한다: shape={pred.shape}")
    if not np.isfinite(pred).all():
        raise RuleViolation("예측에 비유한값이 있다")
    if (pred < 0).any() or (pred > 1).any():
        raise RuleViolation("예측이 [0,1] 밖이다")
    if y_valid is not None:
        gap = abs(float(pred.mean()) - float(np.asarray(y_valid, np.float64).mean()))
        if gap < 1e-12:
            raise RuleViolation(
                f"[조항2] 예측 평균이 검증 라벨 평균과 정확히 같다 (차이 {gap:.2e})"
                f"{' — ' + where if where else ''}.\n"
                f"  평가 데이터의 평균으로 예측을 보정한 배열로 보인다. 저장을 막는다.\n"
                f"  centering 은 지표 계산(centered_bss)에서만 허용된다.")
    np.save(path, pred)


# ── 허용: 학습 시즌에서 얻는 추세 ─────────────────────────────────────────
def train_season_trend(y, seasons, fold: int, *, clip=(0.005, 0.995),
                       rule: str = "linear_all") -> float:
    """season < fold 의 시즌별 평균을 선형 외삽한 사전확률.

    **검증/테스트 라벨을 보지 않는다.** 전 행에 같은 상수로 적용되므로
    행 독립성도 지킨다. 1WAY 의 base_score, 3WAY 의 Pool baseline 이 이것이다.
    """
    y = np.asarray(y, np.float64)
    seasons = np.asarray(seasons)
    if (seasons >= fold).any():
        raise RuleViolation(
            f"[조항4] 추세 계산에 fold({fold}) 이상 시즌이 섞였다: "
            f"{sorted(set(seasons[seasons >= fold].tolist()))[:5]}")
    s = pd.Series(y).groupby(pd.Series(seasons)).mean().sort_index()
    if len(s) < 2:
        return float(np.clip(s.iloc[-1], *clip))
    v = s.to_numpy(float)
    x = s.index.to_numpy(float)
    if rule == "last":                      # 직전 시즌 그대로
        out = v[-1]
    elif rule == "linear_3":                # 최근 3시즌 선형
        k = min(3, len(v))
        out = float(np.polyval(np.polyfit(x[-k:], v[-k:], 1), fold))
    elif rule == "median_diff":             # 차분 중앙값만큼 이동
        out = v[-1] + float(np.median(np.diff(v)))
    elif rule == "ewm":                     # 지수가중 추세 (감쇠 0.7)
        d = np.diff(v)
        w = 0.7 ** np.arange(len(d) - 1, -1, -1)
        out = v[-1] + float((d * w).sum() / w.sum())
    else:                                   # linear_all — 첫-마지막 기울기 (현행)
        span = x[-1] - x[0]
        out = v[-1] + ((v[-1] - v[0]) / span if span > 0 else 0.0)
    return float(np.clip(out, *clip))


# ── 정적 검사 — 금지 패턴이 코드에 들어왔는지 ─────────────────────────────
FORBIDDEN_SOURCE = [
    (re.compile(r"np\.save\([^)]*centered"), "centered 배열을 저장하려 한다 [조항2]"),
    (re.compile(r"pred\s*=\s*.*-\s*\(.*mean\(\)\s*-\s*y.*mean\(\)"),
     "예측을 검증 평균으로 이동시킨다 [조항2]"),
    (re.compile(r"test\w*\.groupby|test\w*\.rolling|test\w*\.cumsum"),
     "test 행을 가로질러 집계한다 [조항1]"),
]


def scan_source(paths, *, quiet: bool = False) -> list[str]:
    """소스에서 금지 패턴을 찾는다. 발견하면 목록을 반환한다."""
    hits = []
    for p in paths:
        p = Path(p)
        if not p.exists() or p.suffix != ".py" or p.name == "guards.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pat, why in FORBIDDEN_SOURCE:
                if pat.search(line):
                    hits.append(f"{p.name}:{i}  {why}\n      {line.strip()[:88]}")
    if hits and not quiet:
        for h in hits:
            print(f"  [FAIL] {h}")
    return hits


def assert_source_clean(paths) -> None:
    hits = scan_source(paths, quiet=True)
    if hits:
        raise RuleViolation("[정적 검사] 금지 패턴 발견:\n  " + "\n  ".join(hits))
