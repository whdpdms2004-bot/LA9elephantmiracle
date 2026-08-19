"""저장된 예측 중 어떤 것이 '정직한' 숫자인지 표시한다.

무엇이 문제인가
    실험 스크립트 대부분이 이렇게 학습한다.

        mdl.fit(p_tr, eval_set=p_va, use_best_model=True)

    p_va 는 **그 예측을 채점할 바로 그 fold** 다. 즉 검증 라벨로 반복 횟수를
    고르고 같은 라벨로 채점한다. 2025 테스트에는 라벨이 없어 쓸 수 없는 방식이므로
    이렇게 나온 BSS 는 테스트 성능 추정치로서 낙관 편향이다.

    규정 위반은 아니다 — 평가(테스트) 데이터가 아니라 검증 fold 를 쓴 것이다.
    그러나 **제출 성적을 예상하는 근거로는 쓸 수 없다.**

실측한 부풀림 (같은 설정을 조기 종료 없이 재적합해 비교)
    outside  f23 1794.3 -> 1774.1   f24 1556.4 -> 1530.1
    reverse  f24 1462.7 -> 1409.8   f23  410.8 -> -310.1
    fold 2024 는 20~50 점, fold 2023 은 최대 721 점이다.
    fold 2023 이 훨씬 심하게 과적합하므로 조기 종료의 구제 효과가 크다.

정직한 대안
    way_weights.py --inner-es
    학습 시즌 중 가장 최근(가중치>0) 해를 내부 검증으로 떼어 반복 횟수를 정하고
    그 횟수로 학습 전체에 재적합한다. 채점 fold 라벨을 보지 않는다.

사용
    python audit_earlystop.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from harness3 import OUT

# 예측 파일 접두사 -> 그것을 만든 스크립트
PRODUCER = {
    "tr_": "train_arms.py",
    "mf_": "middle_focus.py",
    "s3_": "s3_layered.py",
    "ms_": "middle_sweep.py",
    "mn_": "middle_next.py",
    "mh_": "middle_hier.py",
    "mt_": "middle_tune.py",
    "wb_": "way_base.py",
    "ww_": "way_weights.py",
    "": "screen_target.py",          # 접두사 없는 것 (screen 계열)
}
EXTRA_SOURCES = [SRC.parent.parent / "claude" / "src"]


def uses_eval_set(path: Path) -> bool:
    if not path.exists():
        return False
    txt = path.read_text(encoding="utf-8", errors="replace")
    return bool(re.search(r"use_best_model\s*=\s*True", txt))


def main() -> None:
    # way_weights 는 --inner-es 일 때만 조기 종료를 쓰고, 그것도 학습 내부다.
    honest_by_flag = {"ww_"}

    status = {}
    for pre, script in PRODUCER.items():
        p = SRC / script
        status[pre] = (script, uses_eval_set(p))

    print("=" * 92)
    print("예측 생성 스크립트별 — 채점 fold 를 조기 종료에 썼는가")
    print("=" * 92)
    for pre, (script, bad) in sorted(status.items()):
        tag = "부풀림" if bad and pre not in honest_by_flag else "정직"
        note = ""
        if pre in honest_by_flag:
            note = "  (내부 조기종료만 사용 — 채점 fold 라벨 미사용)"
        print(f"  {pre or '(없음)':<6}{script:<22}{tag:<8}{note}")

    counts = {}
    for f in sorted(OUT.glob("*.npy")):
        pre = next((p for p in PRODUCER if p and f.name.startswith(p)), "")
        counts[pre] = counts.get(pre, 0) + 1

    print(f"{chr(10)}{'=' * 92}")
    print("저장된 예측 파일 분포")
    print("=" * 92)
    tot_bad = tot_ok = 0
    for pre, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        script, bad = status.get(pre, ("?", False))
        bad = bad and pre not in honest_by_flag
        tot_bad, tot_ok = (tot_bad + n, tot_ok) if bad else (tot_bad, tot_ok + n)
        print(f"  {pre or '(없음)':<6}{script:<22}{n:>5}개   "
              f"{'부풀림' if bad else '정직'}")
    print(f"{chr(10)}  합계  정직 {tot_ok}개 / 부풀림 {tot_bad}개")

    print(f"{chr(10)}{'=' * 92}")
    print("1WAY 쪽 (같은 문제)")
    print("=" * 92)
    for d in EXTRA_SOURCES:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.py")):
            if uses_eval_set(p):
                print(f"  {p.name}")

    print(f"{chr(10)}{'=' * 92}")
    print("해석")
    print("=" * 92)
    print("  · 규정 위반이 아니다. 검증 fold 를 쓴 것이고 테스트 데이터는 건드리지 않았다.")
    print("  · 다만 제출 성적 예상 근거로는 못 쓴다. 2025 에는 라벨이 없다.")
    print("  · fold 2024 부풀림 20~50, fold 2023 부풀림 최대 721 — f23 숫자를 특히 조심.")
    print("  · 정직한 비교가 필요하면 way_weights.py --inner-es 로 재적합해 쓴다.")


if __name__ == "__main__":
    main()
