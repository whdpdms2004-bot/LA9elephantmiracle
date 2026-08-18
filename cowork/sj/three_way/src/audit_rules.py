"""3WAY 규정 준수 감사 — test 데이터 취급 관점. CPU 전용.

RULES.md 금지 조항
    1 test.csv 의 다른 행을 이용한 예측 (누적/rolling/lag, 선수·팀·월·경기 단위 집계)
    2 평가 데이터의 분포·평균·순위를 이용한 예측값 보정
    3 외부 데이터
    4 투구 이후 시점 정보
    5 외부 API

검사 항목
    A 행 독립성 (조항 1) — 부분 행으로 만든 피처가 전체로 만든 것과 같은가.
      제출 검증의 "부분 실행 == 전체 실행" 과 같은 검사를 피처 층에서 한다.
    B 학습 전용 요소가 추론 경로에 새지 않는가 — 라벨 유래 값, 학습 가중치
    C 평가 라벨을 쓰는 지점이 어디이고 예측에 반영되는가 (조항 2)
    D 시간 인과 (조항 4) — fold 이전 시즌만 쓰는가

사용
    python audit_rules.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness3 import LAB, OUT, SUCCESS, TARGETS, load_labeled

FOLD = 2024
FAIL, WARN = [], []


def check(name, ok, detail="", warn=False):
    tag = "OK  " if ok else ("WARN" if warn else "FAIL")
    print(f"  [{tag}] {name}   {detail}", flush=True)
    if not ok:
        (WARN if warn else FAIL).append(name)


def main() -> None:
    from run_optuna_enhanced import load_enhanced_frame
    from run_optuna_family import CATEGORICAL_COLUMNS
    from v77_single_xgb_screen import (build_component_unique,
                                       build_component_unique_forward)
    from v80_single_catboost import make_features
    sys.path.insert(0, str(LAB))
    import transforms as T
    import train_arms as TA
    T.load_all()

    frame, enhanced = load_enhanced_frame()
    season = frame["season"].to_numpy()
    tr_mask = season < FOLD
    va_mask = season == FOLD
    print(f"학습 {tr_mask.sum():,}  검증 {va_mask.sum():,}")

    static = build_component_unique(frame, enhanced, FOLD)
    forward = build_component_unique_forward(frame, enhanced, FOLD,
                                             cache={FOLD: static})
    base, f1_features = make_features(frame, enhanced, FOLD, "F1", forward)
    for c in (SUCCESS, "season"):
        if c not in base.columns:
            base[c] = frame[c].to_numpy()
    cats0 = [c for c in CATEGORICAL_COLUMNS if c in f1_features]
    tr_series = pd.Series(tr_mask, index=frame.index)

    print(f"{chr(10)}{'=' * 92}")
    print("A. 행 독립성 (조항 1) — 검증 행을 잘라내도 피처가 같은가")
    print("=" * 92)
    combos = {
        "middle":  ("id_frequency", "no_trackman", "temporal_cyclic"),
        "reverse": ("count_multiscale", "drop_ids", "trackman_quality"),
        "ball":    ("drop_ids", "no_trackman", "rate_multiscale"),
    }
    rng = np.random.default_rng(7)
    va_idx = np.flatnonzero(va_mask)
    subset = np.sort(rng.choice(va_idx, size=min(400, len(va_idx)), replace=False))

    for tag, combo in combos.items():
        import gc
        full_fr, feats, _ = T.build(base, f1_features, cats0, combo, tr_series, FOLD)
        full_vals = full_fr.loc[subset, feats].apply(
            pd.to_numeric, errors="coerce").to_numpy(np.float64)

        # 검증 행 중 subset 만 남기고 나머지 검증 행을 지운 프레임으로 다시 만든다.
        keep = tr_mask.copy()
        keep[subset] = True
        part_base = base.loc[keep].reset_index(drop=True)
        part_tr = pd.Series(part_base["season"].to_numpy() < FOLD,
                            index=part_base.index)
        part_fr, part_feats, _ = T.build(part_base, f1_features, cats0, combo,
                                         part_tr, FOLD)
        pos = np.flatnonzero(part_base["season"].to_numpy() == FOLD)
        part_vals = part_fr.loc[pos, part_feats].apply(
            pd.to_numeric, errors="coerce").to_numpy(np.float64)

        same_cols = part_feats == feats
        if not same_cols:
            check(f"{tag} 열 구성 동일", False, f"{len(feats)} vs {len(part_feats)}")
            continue
        d = np.abs(np.nan_to_num(full_vals) - np.nan_to_num(part_vals))
        nan_same = (np.isnan(full_vals) == np.isnan(part_vals)).all()
        worst = float(d.max()) if d.size else 0.0
        bad = [feats[j] for j in range(d.shape[1]) if d[:, j].max() > 1e-6]
        check(f"{tag} 부분 == 전체 ({len(feats)}열, {len(subset)}행)",
              worst < 1e-6 and nan_same,
              f"최대 차이 {worst:.3e}" + (f"  문제열 {bad[:3]}" if bad else ""))
        del full_fr, part_fr, part_base; gc.collect()

    print(f"{chr(10)}{'=' * 92}")
    print("B. 학습 전용 요소가 추론 경로에 새는가")
    print("=" * 92)
    short = TA.short_outing_mask(frame)
    src = Path(TA.__file__).read_text(encoding="utf-8")
    used_as_weight = "w[short[tr_mask]]" in src.replace(" ", "")
    used_as_feature = ("short" in src.split("feats =")[-1][:300]
                       if "feats =" in src else False)
    check("짧은 등판 마스크가 학습 가중치로만 쓰인다",
          used_as_weight and not used_as_feature,
          "행 간 .shift()/groupby 로 만들어 추론 피처로는 쓸 수 없다")
    check("짧은 등판 마스크가 피처 목록에 없다",
          not any("short" in f.lower() for f in f1_features))

    lab = load_labeled()
    label_cols = [c for c in lab.columns if c.startswith("y_")] + ["label_ok"]
    leaked = [c for c in label_cols if c in f1_features]
    check("복원 라벨이 피처에 없다", not leaked, f"{leaked}" if leaked else "")

    print(f"{chr(10)}{'=' * 92}")
    print("C. 평가 라벨을 쓰는 지점 (조항 2)")
    print("=" * 92)
    h = Path(Path(TA.__file__).parent / "harness3.py").read_text(encoding="utf-8")
    check("bss_centered 는 metrics 안에서만 계산된다",
          "pc = np.clip" in h and h.count("pc = np.clip") == 1)
    saved_raw = all("np.save(npy, pred)" in Path(p).read_text(encoding="utf-8")
                    for p in Path(TA.__file__).parent.glob("*.py")
                    if "np.save(npy" in Path(p).read_text(encoding="utf-8"))
    check("저장되는 예측은 raw (centered 가 아님)", saved_raw,
          "centered 는 순위 판정용 지표일 뿐 예측을 바꾸지 않는다")
    print("       주의: bss_centered 는 검증 시즌 실제 평균으로 예측을 이동시켜 계산한다.")
    print("             지표로만 쓴다. 이 방식을 제출에 적용하면 조항 2 위반이다.")

    print(f"{chr(10)}{'=' * 92}")
    print("D. 시간 인과 (조항 4)")
    print("=" * 92)
    # 전처리 테이블이 fold 를 바꾸면 값이 바뀌는가 = 학습 시즌만 쓴다는 증거
    # 전체 프레임을 두 번 만들면 메모리가 부족하므로 변환 함수만 직접 호출한다.
    import gc
    import v85_preprocess_screen as MM
    e23 = MM.id_frequency(frame, pd.Series(season < 2023, index=frame.index))
    e24 = MM.id_frequency(frame, tr_series)
    k = sorted(e24)[0]
    check("id_frequency 가 fold 마다 다르다 (학습 시즌만 사용)",
          not np.allclose(np.nan_to_num(np.asarray(e23[k], float)),
                          np.nan_to_num(np.asarray(e24[k], float))), k)
    del e23, e24; gc.collect()
    prior_src = "yv[tr_mask]" in Path(TA.__file__).read_text(encoding="utf-8")
    check("시즌 외삽 사전확률이 학습 행에서만 계산된다", prior_src,
          "Pool baseline 은 전 행 동일한 상수 -> 행 독립")

    print(f"{chr(10)}{'=' * 92}")
    print("E. 관문 (guards.py) 이 실제로 막는가")
    print("=" * 92)
    import guards as G
    def blocked(name, fn):
        try:
            fn(); check(name, False, "관문을 통과해버렸다")
        except G.RuleViolation:
            check(name, True, "관문이 거부")
    yy = np.array([1, 0, 1, 0], float)
    pp = np.array([0.7, 0.3, 0.6, 0.4])
    blocked("ㄱ centered 배열 저장 시도",
            lambda: G.save_prediction(OUT / "_t.npy", pp - (pp.mean() - yy.mean()), yy))
    blocked("ㄴ 학습전용 값을 피처로",
            lambda: G.assert_features_clean(["short_outing_mask"]))
    blocked("라벨을 피처로", lambda: G.assert_features_clean(["y_middle"]))
    blocked("검증 시즌으로 추세 계산",
            lambda: G.train_season_trend(yy, [2023, 2023, 2024, 2024], 2024))
    check("학습 시즌 추세는 허용",
          isinstance(G.train_season_trend(yy, [2022, 2022, 2023, 2023], 2024), float))
    hits = G.scan_source(sorted(Path(__file__).parent.glob("*.py")), quiet=True)
    check("소스 정적 검사 (금지 패턴)", not hits, f"{len(hits)}건" if hits else "")

    print(f"{chr(10)}{'=' * 92}")
    print(f"결과: {'전 항목 통과' if not FAIL else 'FAIL ' + str(FAIL)}"
          f"{'   경고 ' + str(WARN) if WARN else ''}")
    print("=" * 92)


if __name__ == "__main__":
    main()
