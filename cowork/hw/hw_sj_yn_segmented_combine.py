"""hw x sj x yn 구간별 3자 결합 -- hw_sj_segmented_blend_analysis.md 제안의 실제 구현.

기존 문서(hw_sj_segmented_blend_analysis.md)는 "구간별로 결합 이득이 몰려있다"는
관찰(analysis)까지였다. 이 스크립트는 그걸 실제로 동작하는 결합(weights + 검증)으로
만든 것 -- 투수표본수 구간(hw 라인 오프셋에서 이미 실LB로 검증된 축, +8.84 확인됨,
submission_v9_bucketoffset)을 그대로 재사용해 3자 블렌드 가중치를 구간별로 따로
계산한다. game_type 축은 별도로 연도 간 재현성을 확인했더니 방향이 뒤집혀서
(validate_gametype_offset_multiyear.py 참고) 이번 결합에는 쓰지 않았다.

방법:
    각 구간에서 w* = M^-1 A (D = [p_hw-r, p_sj-r, p_yn-r], Val2024 실제 라벨)
    구간별 가중치를 그 구간 행에 적용한 예측을 이어붙여, 전체에 대해 하나의
    풀드(pooled) BSS를 계산 -- 구간별 BSS의 단순 가중평균이 아니라, 실제 제출
    시나리오와 동일하게 "전체 행에 대한 하나의 브라이어 스코어"로 비교한다.

데이터: cowork/hw/val2024_pred.csv, cowork/sj/val2024_pred.csv,
        cowork/yn/val2024_pred.csv, data/train.csv 만 사용.
        리더보드 미참조 (RULES.md §2 준수).

저장소 루트 기준 상대경로만 사용 (AGENTS.md A6 -- 절대경로 하드코딩 금지).

실행 (저장소 루트에서):
    py cowork/hw/hw_sj_yn_segmented_combine.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "control_success"
REPO = Path(__file__).resolve().parents[2]
HW_PRED_PATH = REPO / "cowork" / "hw" / "val2024_pred.csv"
SJ_PRED_PATH = REPO / "cowork" / "sj" / "val2024_pred.csv"
YN_PRED_PATH = REPO / "cowork" / "yn" / "val2024_pred.csv"
TRAIN_PATH = REPO / "data" / "train.csv"
OUT_WEIGHTS_PATH = Path(__file__).with_name("hw_sj_yn_segmented_weights.json")

COLS = ["p_hw", "p_sj", "p_yn"]
# hw 라인 오프셋에서 실LB로 검증된 동일한 구간 경계 (submission_v9_bucketoffset)
BUCKET_EDGES = [-1, 200, 2000, float("inf")]
BUCKET_LABELS = ["low(<200)", "mid(200-2000)", "high(2000+)"]


def bss_of(y, p):
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 0.0, 1.0)
    brier = np.mean((y - p) ** 2)
    r = y.mean()
    u = r * (1 - r) if 0 < r < 1 else 1e-9
    return max(0.0, 100000 * (1 - brier / u))


def solve_weights(sub):
    yy = sub[TARGET].to_numpy()
    r = yy.mean()
    P = sub[COLS].to_numpy()
    D = P - r
    M = D.T @ D / len(sub)
    A = D.T @ (yy - r) / len(sub)
    try:
        w = np.linalg.solve(M, A)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(M, A, rcond=None)[0]
    return w, r


def main():
    if not (SJ_PRED_PATH.exists() and YN_PRED_PATH.exists()):
        print(f"필요 파일 없음 -- sj: {SJ_PRED_PATH.exists()}, yn: {YN_PRED_PATH.exists()}")
        print("각자 브랜치가 병합되면 자동으로 동작합니다.")
        return

    raw = pd.read_csv(TRAIN_PATH)
    val = raw[raw.season == 2024][["row_id", "asof_pitcher_n", TARGET]].copy()

    hw = pd.read_csv(HW_PRED_PATH).rename(columns={TARGET: "p_hw"})
    sj = pd.read_csv(SJ_PRED_PATH).rename(columns={TARGET: "p_sj"})
    yn = pd.read_csv(YN_PRED_PATH).rename(columns={TARGET: "p_yn"})[["row_id", "p_yn"]]
    df = val.merge(hw, on="row_id").merge(sj, on="row_id").merge(yn, on="row_id")
    print(f"병합된 행수: {len(df)}\n")

    df["bucket"] = pd.cut(df["asof_pitcher_n"].fillna(0), BUCKET_EDGES, labels=BUCKET_LABELS)

    # 1) 단일 3자 블렌드 (구간 없음, 기준선)
    w_all, r_all = solve_weights(df)
    pred_single = r_all + (df[COLS].to_numpy() - r_all) @ w_all
    bss_single = bss_of(df[TARGET], pred_single)
    print(f"[기준선] 단일 3자 블렌드 (구간 없음)  pooled BSS = {bss_single:.2f}  "
          f"w=[hw{w_all[0]:+.3f} sj{w_all[1]:+.3f} yn{w_all[2]:+.3f}]")

    # 2) 구간별 3자 블렌드
    pred_seg = np.zeros(len(df))
    weights_out = {}
    print("\n[구간별 3자 블렌드]")
    for b in BUCKET_LABELS:
        mask = (df["bucket"] == b).to_numpy()
        sub = df[mask]
        w, r = solve_weights(sub)
        pred_seg[mask] = r + (sub[COLS].to_numpy() - r) @ w
        weights_out[b] = {"n": int(mask.sum()), "r": float(r),
                           "w": {c: float(x) for c, x in zip(COLS, w)}}
        seg_bss = bss_of(sub[TARGET], pred_seg[mask])
        print(f"  {b:15s} n={mask.sum():7d}  w=[hw{w[0]:+.3f} sj{w[1]:+.3f} yn{w[2]:+.3f}]  "
              f"구간자체BSS={seg_bss:8.2f}")
    bss_segmented = bss_of(df[TARGET], pred_seg)

    print(f"\n[전체 비교 -- 하나의 풀드 BSS로 공정 비교]")
    print(f"  solo:  hw={bss_of(df[TARGET], df['p_hw']):.2f}  "
          f"sj={bss_of(df[TARGET], df['p_sj']):.2f}  yn={bss_of(df[TARGET], df['p_yn']):.2f}")
    print(f"  단일 3자 블렌드    = {bss_single:.2f}")
    print(f"  구간별 3자 블렌드  = {bss_segmented:.2f}")
    print(f"  구간화 이득        = {bss_segmented - bss_single:+.2f}")

    json_safe_edges = [None if e == float("inf") else e for e in BUCKET_EDGES]
    with open(OUT_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"bucket_edges": json_safe_edges, "labels": BUCKET_LABELS,
                    "weights": weights_out,
                    "pooled_bss": {"solo_hw": bss_of(df[TARGET], df["p_hw"]),
                                    "solo_sj": bss_of(df[TARGET], df["p_sj"]),
                                    "solo_yn": bss_of(df[TARGET], df["p_yn"]),
                                    "single_blend": bss_single,
                                    "segmented_blend": bss_segmented}}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n가중치 저장: {OUT_WEIGHTS_PATH.name}")
    print("\n※ 주의: Val2024 한 해만으로 계산한 가중치라 과최적화 위험이 있음.")
    print("   투수표본 구간 경계 자체는 hw 라인 오프셋에서 실LB로 검증된 축을 그대로")
    print("   재사용해 리스크를 낮췄지만(3구간, 넓은 경계), 실제 제출용으로 쓰려면")
    print("   sj/yn의 test.csv 예측값이 있어야 함 (지금은 Val2024만 있어서 결합")
    print("   자체는 아직 제출 불가 -- 방법/가중치 검증까지가 이 스크립트의 목적).")


if __name__ == "__main__":
    main()
