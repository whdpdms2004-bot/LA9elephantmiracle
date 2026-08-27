"""new_val 3WAY 러너 — 설정 가능한 way 학습 + 포함-배제 결합 + policy 채점.

meta_fusion/src/build_honest_oof.py 와 같은 strict-forward 고정 900회 프로토콜이다.
채점 fold 라벨을 eval_set / early stopping 에 쓰지 않는다.

새로 붙인 손잡이
    fpre0        단절 이전(season<2023) F 행을 학습에서 제외. way 별로 켠다
                 VALIDATION_POLICY §2.3 — 학습 시즌 라벨만 쓰므로 규정 위반 아님
    late_boost   후반(8~10월) 학습 행 가중치 배수. L7 용
    month_decay  시즌 내 월에 대한 recency. 후반 판별력 실험용
    combo        way 별 전처리 조합 교체

사용
    python run_way.py --exp b2_fpre0_ro --fpre0 reverse,outside
    python run_way.py --exp b2_late15 --late-boost 1.5
    python run_way.py --list
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEW_VAL = HERE.parents[1]
SJ = NEW_VAL.parent
TW = SJ / "three_way"
CAMPAIGN = SJ / "feature_campaign_1000"
MODEL_OPT = SJ / "experiment" / "model_optimization"
CLAUDE_SRC = SJ / "claude" / "src"
LAB = SJ / "preprocess_lab"
for p in (NEW_VAL / "common", TW / "src", CAMPAIGN, MODEL_OPT, CLAUDE_SRC, LAB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import policy as P  # noqa: E402

OUT = NEW_VAL / "outputs" / "three_way"
PARAMS_PATH = MODEL_OPT / "catboost_v2r200_tm500_robust_best.json"
SEED = 20262844
BREAK_SEASON = 2023          # game_type 구조적 단절. 이전 F 행이 문제다

WAYS = {
    "middle":  ["id_frequency", "no_trackman", "temporal_cyclic"],
    "reverse": ["count_multiscale", "drop_ids", "trackman_quality"],
    "outside": ["drop_ids", "no_trackman", "rate_multiscale"],
    "mr":      ["id_frequency", "no_trackman", "temporal_cyclic"],
}
LABELS = {"middle": "y_middle", "reverse": "y_reverse",
          "outside": "y_outside", "mr": "y_mr"}
SUCCESS_LABEL = "control_success"


def short_outing_mask(frame: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """짧은 등판 행. claude/src/v67_build034.py 의 정의를 그대로 옮겼다.

    1WAY 에서 이 행들에 학습 가중치 0.5 를 준 것이 **Public +10.44** 였다
    (submit_033 973.14 -> submit_034 983.58). 3WAY 에는 미이식이었다.

    등판 = 같은 투수의 연속 행 중 asof_pitcher_prev1_game_success_rate 가 같은 구간.
    비율 = 그 등판 길이 / 그 투수의 (선발|구원별) 중앙 등판 길이.

    **학습 행 가중치에만 쓴다. 피처가 아니므로 추론 행 독립성과 무관하다.**
    """
    pid = frame["pitcher_id"].to_numpy()
    nvol = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy()
    o = np.argsort(pid.astype(np.int64) * 10_000_000 + nvol, kind="stable")
    pv = pd.to_numeric(frame["asof_pitcher_prev1_game_success_rate"],
                       errors="coerce").to_numpy()[o]
    gp = pid[o]
    chg = np.r_[True, (gp[1:] != gp[:-1]) | ~np.isclose(pv[1:], pv[:-1],
                                                        equal_nan=True)]
    outing = np.empty(len(frame), dtype=np.int64)
    outing[o] = np.cumsum(chg) - 1
    od = pd.DataFrame({"outing": outing, "pid": pid,
                       "inn": frame["inning"].to_numpy()})
    agg = od.groupby("outing").agg(n=("outing", "size"), pid=("pid", "first"),
                                   first_inn=("inn", "min"))
    agg["start"] = (agg["first_inn"] == 1).astype(int)
    agg = agg.join(agg.groupby(["pid", "start"])["n"].median().rename("med"),
                   on=["pid", "start"])
    ratio = np.nan_to_num(
        (agg["n"] / agg["med"].clip(lower=1)).reindex(outing).to_numpy(), nan=1.0)
    return ratio < threshold


def outing_len_mask(frame: pd.DataFrame, lo: float, hi: float) -> np.ndarray:
    """등판 **절대 길이**가 [lo, hi] 인 행. `short_outing_mask` 와 다른 축이다.

    `short_outing_mask` 는 그 투수의 **중앙 등판 길이 대비 비율**로 판정한다
    (Public +10.44 실적). 여기는 **절대 투구수**다.

    근거 (last_week `cond_bss.py` 실측, fold2024 챔피언 891.0):
        등판 길이 <=15 828.5 / 16-40 719.0 / **41-70 463.8** / 71-100 900.6 / 101+ 1165.0
        격차 **701.2** — 볼카운트(1,329) 다음으로 큰 축이다. 41-70 은 27,605행(10.9%).
        `fus`(−70.0) `stu`(−271.9) 가 그 구간에서 **음수**다.

    41-70 이 최악인 것은 야구적으로 설명된다 — **애매한 중간**이다.
    <=15 는 명확한 불펜, 101+ 는 명확한 퀄리티스타트인데 41-70 은
    "조기 강판된 선발" 과 "롱릴리프" 가 섞여 역할이 모호하다.

    등판 경계는 `short_outing_mask` 와 **같은 정의**를 쓴다 (prev1 런 구간).
    **학습 행 가중치에만 쓴다 — 피처가 아니므로 추론 행 독립성과 무관하다.**
    """
    pid = frame["pitcher_id"].to_numpy()
    nvol = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(np.float64)
    o = np.argsort(pid.astype(np.int64) * 10_000_000 + np.nan_to_num(nvol),
                   kind="stable")
    pv = pd.to_numeric(frame["asof_pitcher_prev1_game_success_rate"],
                       errors="coerce").to_numpy(np.float64)[o]
    gp = pid[o]
    chg = np.r_[True, (gp[1:] != gp[:-1])
                | ~np.isclose(pv[1:], pv[:-1], equal_nan=True)]
    oid = np.empty(len(frame), np.int64)
    oid[o] = np.cumsum(chg) - 1
    ln = pd.Series(oid).groupby(oid).transform("size").to_numpy()
    return (ln >= lo) & (ln <= hi)


def build_weights(base_w, season, month, gtype, fold, cfg, way, short=None,
                  olmask=None):
    """학습 행 가중치. 전부 학습 시즌 정보만 쓴다."""
    w = np.asarray(base_w, np.float64).copy()
    sw = float(cfg.get("short_weight", 1.0))
    if sw != 1.0 and short is not None:
        w = np.where(short, w * sw, w)
    olw = float(cfg.get("outing_len_weight", 1.0))
    if olw != 1.0 and olmask is not None:
        # ★ olmask 는 **이미 학습행으로 잘린** 배열이어야 한다.
        #   전체 프레임 마스크를 그대로 쓰면 broadcast 오류가 난다 (실제로 냈다).
        w = np.where(olmask, w * olw, w)
    fw_all = float(cfg.get("f_weight", 1.0))
    if fw_all != 1.0:
        # fw020 계열 — 전 시즌 F 행 축소 (fpre0 는 단절 이전만)
        w = np.where(gtype == "F", w * fw_all, w)
    if way in cfg.get("fpre0", []):
        # 단절 이전 F 행 축소. 0.0 이면 완전 제외. VALIDATION_POLICY §2.3
        # fold 2023 은 학습이 전부 단절 이전이라 0.0 이면 F 를 100% 잃는다.
        # 그래서 강도를 손잡이로 뺐다 (B3).
        fw = float(cfg.get("fpre_weight", 0.0))
        w = np.where((season < BREAK_SEASON) & (gtype == "F"), w * fw, w)
    lb = float(cfg.get("late_boost", 1.0))
    if lb != 1.0:
        w = np.where(month >= 8, w * lb, w)
    md = float(cfg.get("month_decay", 0.0))
    if md > 0.0:
        # 시즌 내 월을 연속 시간으로 보고 최근 월에 가중. 학습 행만 건드린다.
        w = w * np.exp(md * (month - 3.0) / 7.0)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="실험 id. 파일 접두사가 된다")
    ap.add_argument("--fpre0", default="", help="쉼표구분 way. 단절이전 F 제외")
    ap.add_argument("--fpre-weight", type=float, default=0.0,
                    help="단절 이전 F 행 가중치 배수. 0=완전제외")
    ap.add_argument("--outing-len-weight", type=float, default=1.0,
                    help="등판 절대길이 [lo,hi] 구간 학습가중 배율. "
                         "cond_bss 실측: 41-70 구간 BSS 463.8 (최악, 27,605행)")
    ap.add_argument("--outing-len-range", default="41,70",
                    help="위 구간 경계 (포함)")
    ap.add_argument("--short-weight", type=float, default=1.0,
                    help="짧은 등판 학습 가중치. 1WAY 에서 0.5 가 Public +10.44")
    ap.add_argument("--f-weight", type=float, default=1.0,
                    help="전 시즌 F 행 학습 가중치 (fw020 계열)")
    ap.add_argument("--late-boost", type=float, default=1.0)
    ap.add_argument("--month-decay", type=float, default=0.0)
    ap.add_argument("--combo", default="", help="way=a+b+c 형식, 쉼표로 여러 개")
    ap.add_argument("--folds", default="2023,2024")
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--lr", type=float, default=0.015,
                    help="learning_rate. 0.015 는 옛 Optuna 값이고 최종 지표로 조정된 적이 없다")
    ap.add_argument("--l2", type=float, default=0.0,
                    help="전 way l2_leaf_reg. 기본 124.89 는 iterations=4487/lr=0.0078/depth=9 "
                         "영역에서 나온 값이다. 내 영역(900/0.015/8)에는 안 맞을 수 있다")
    ap.add_argument("--bagging-temp", type=float, default=-1.0,
                    help="bagging_temperature. 기본 2.736 (같은 이유로 미조정)")
    ap.add_argument("--random-strength", type=float, default=-1.0,
                    help="random_strength. 기본 0.0004")
    ap.add_argument("--half-life", type=float, default=0.0,
                    help=">0 이면 recency 반감기를 덮어쓴다. 기본 1.675 는 옛 Optuna 값이고 "
                         "최종 지표로 조정된 적이 없다. 신경망에선 0.8 이 +4.7 이었다")
    ap.add_argument("--mr-conditional", action="store_true",
                    help="mr 을 p(m) x p(r|m) 로 분해한다. r|m 은 **m=1 행만**으로 학습. "
                         "독립가정(m x r)은 -123 이었지만 조건부는 미시도")
    ap.add_argument("--arm-combo", default="",
                    help="arm 별 전처리. 예 outside:b0=id_frequency+no_trackman+temporal_cyclic. "
                         "ob(볼실패 82.4%%)와 oz(존안실패 17.6%%)는 성격이 다른 사건이다")
    ap.add_argument("--grow-policy", default="",
                    choices=("", "SymmetricTree", "Lossguide", "Depthwise"),
                    help="CatBoost 트리 성장 방식. Lossguide 는 leaf-wise. "
                         "LGB 에서 leaf-wise + 적은 리프가 좋았다(31리프 699.6)")
    ap.add_argument("--max-leaves", type=int, default=0,
                    help="Lossguide 일 때 최대 리프 수")
    ap.add_argument("--lgb-leaves", type=int, default=63,
                    help="LGB num_leaves. **CatBoost depth 와 등가가 아니다** — "
                         "대칭트리 depth 8(256리프) 대비 LGB 는 leaf-wise 라 훨씬 자유롭다. "
                         "255 로 주면 과적합한다 (실측 -612.9)")
    ap.add_argument("--xgb-depth", type=int, default=8,
                    help="XGBoost max_depth. CatBoost depth 8 과 맞춘 기본값")
    ap.add_argument("--model", default="cat", choices=("cat", "lgb", "xgb"),
                    help="way 모델 계열. lgb 는 배깅 다양성용 (팀 실측 +151.35)")
    ap.add_argument("--disjoint", action="store_true",
                    help="겹치는 m/r 대신 서로소 사건 4개를 직접 모델링한다. "
                         "p_fail = m_only + r_only + mr + outside (뺄셈 없음)")
    ap.add_argument("--direct", action="store_true",
                    help="1WAY — control_success 를 직접 예측한다 (포함-배제 없음)")
    ap.add_argument("--mask", default="",
                    help="이상치 마스킹 clipQ/nanQ. 예 clip995. "
                         "VALIDATION_POLICY 6.5 — 검증행에도 같은 상수를 적용한다")
    ap.add_argument("--mask-augment", action="store_true",
                    help="원본을 덮지 않고 마스킹 사본을 학습에 덧붙인다")
    ap.add_argument("--arm-split", default="",
                    help="way 를 ball 로 분할. 예 middle=ball,mr=ball")
    ap.add_argument("--keep-top", type=int, default=0,
                    help=">0 이면 CatBoost 중요도 상위 N 피처만 남긴다. "
                         "중요도는 **학습 행만으로** 짧은 예비 모델(200회)을 돌려 구한다 — "
                         "평가 fold 를 보지 않는다. 트리에서 단조변환은 무효지만 "
                         "**피처 선별은 유효**하다")
    ap.add_argument("--min-pitcher-n", type=float, default=0.0,
                    help="학습 행을 asof_pitcher_n >= 이 값으로 제한한다. "
                         "표본수 축 학습행 선택 — fpre0(game_type)·short_outing(등판)과 "
                         "다른 축이고 **미시험**이다. B30 에서 <100 구간이 최약(466.2)")
    ap.add_argument("--exclude-seasons", default="",
                    help="학습에서 제외할 시즌(쉼표). 단절 이후 비율을 인위로 낮춰 "
                         "fold 2024 를 fold 2023 조건(0%%)으로 맞추는 데 쓴다")
    ap.add_argument("--min-season", type=int, default=0,
                    help="학습 시즌 하한(포함). 지수감쇠(half_life)와 다르다 — "
                         "하드 컷은 id_frequency 통계와 범주 인코딩까지 바꾼다. 미시험 축")
    ap.add_argument("--drop-foldrel", action="store_true",
                    help="temporal_cyclic 의 fold 상대 피처 2종을 뺀다. "
                         "검증행의 0.0%% 만 학습 범위 안이라 외삽 불가.")
    ap.add_argument("--drift-drop", type=int, default=0,
                    help="학습 시즌 간 분포이동 큰 수치 피처 N개 제거")
    ap.add_argument("--way-depth", default="",
                    help="way 별 depth. 예 middle=6,reverse=11")
    ap.add_argument("--way-l2", default="",
                    help="way 별 l2_leaf_reg. 예 mr=10")
    ap.add_argument("--way-iter", default="",
                    help="way 별 iterations. 예 middle=1500,reverse=400")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cfg = {"exp": args.exp,
           "fpre0": [x.strip() for x in args.fpre0.split(",") if x.strip()],
           "model": args.model, "lgb_leaves": args.lgb_leaves,
           "grow_policy": args.grow_policy, "max_leaves": args.max_leaves,
           "fpre_weight": args.fpre_weight,
           "short_weight": args.short_weight, "f_weight": args.f_weight,
           "outing_len_weight": args.outing_len_weight,
           "late_boost": args.late_boost, "month_decay": args.month_decay,
           "iterations": args.iterations, "lr": args.lr,
           "seed": args.seed, "note": args.note}
    combos = dict(WAYS)
    if args.disjoint:
        # 서로소 분해. m_only/r_only 는 새 라벨이라 아래에서 만든다.
        combos = {"m_only": WAYS["middle"], "r_only": WAYS["reverse"],
                  "mr": WAYS["mr"], "outside": WAYS["outside"]}
        LABELS["m_only"] = "__M_ONLY__"
        LABELS["r_only"] = "__R_ONLY__"
    if args.direct:
        # 1WAY 트랙. way 를 하나로 두고 결합을 건너뛴다.
        combos = {"success": ["id_frequency", "temporal_cyclic",
                              "trackman_quality"]}
        LABELS["success"] = SUCCESS_LABEL
    for item in [x for x in args.combo.split(",") if x.strip()]:
        k, v = item.split("=", 1)
        combos[k.strip()] = sorted(x for x in v.split("+") if x)
    kv = lambda t: {a.split("=")[0].strip(): int(a.split("=")[1])
                    for a in t.split(",") if a.strip()}
    cfg["arm_split"] = {a.split("=")[0].strip(): a.split("=")[1].strip()
                        for a in args.arm_split.split(",") if a.strip()}
    cfg["mr_conditional"] = args.mr_conditional
    cfg["arm_combo"] = {}
    for _it in [x for x in args.arm_combo.split(",") if x.strip()]:
        _k, _v = _it.split("=", 1)
        cfg["arm_combo"][_k.strip()] = sorted(x for x in _v.split("+") if x)
    cfg["keep_top"] = int(args.keep_top)
    cfg["drop_foldrel"] = bool(args.drop_foldrel)
    cfg["drift_drop"] = args.drift_drop
    cfg["mask"] = args.mask
    cfg["mask_augment"] = bool(args.mask_augment)
    cfg["xgb_depth"] = int(args.xgb_depth)
    cfg["way_depth"] = kv(args.way_depth)
    cfg["way_iter"] = kv(args.way_iter)
    cfg["way_l2"] = {a.split("=")[0].strip(): float(a.split("=")[1])
                     for a in args.way_l2.split(",") if a.strip()}
    cfg["combos"] = combos
    for w in cfg["fpre0"]:
        if w not in WAYS and not (args.direct or args.disjoint):
            raise SystemExit(f"모르는 way: {w}")
    if "middle" in cfg["fpre0"] and not (args.direct or args.disjoint):
        raise SystemExit("middle 에는 fpre0 를 쓰지 않는다 (VALIDATION_POLICY §5.1 — "
                         "F 가 -275 -> -873 으로 악화된다)")

    OUT.mkdir(parents=True, exist_ok=True)
    from catboost import CatBoostClassifier, Pool
    from guards import assert_features_clean, train_season_trend
    from harness3 import SUCCESS, load_labeled
    from run_optuna_enhanced import load_enhanced_frame
    from run_optuna_family import CATEGORICAL_COLUMNS, recency_weights
    from v77_single_xgb_screen import (build_component_unique,
                                       build_component_unique_forward)
    from v80_single_catboost import make_features
    import transforms as T
    T.load_all()

    frame, enhanced = load_enhanced_frame()
    labeled = load_labeled()
    if not np.array_equal(frame["row_id"].to_numpy(), labeled["row_id"].to_numpy()):
        raise RuntimeError("row order mismatch")
    season = frame["season"].to_numpy()
    month = pd.to_numeric(frame["game_month"], errors="coerce").to_numpy(float)
    gtype = frame["game_type"].astype(str).to_numpy()
    comp_ok = labeled["label_ok"].to_numpy() == 1
    if args.disjoint:
        _m = pd.to_numeric(labeled["y_middle"], errors="coerce").to_numpy(float)
        _r = pd.to_numeric(labeled["y_reverse"], errors="coerce").to_numpy(float)
        labeled["__M_ONLY__"] = _m * (1.0 - _r)
        labeled["__R_ONLY__"] = _r * (1.0 - _m)
    yball = pd.to_numeric(labeled["y_ball"], errors="coerce").to_numpy(np.float64)
    late_v = pd.to_numeric(frame["late_inning"], errors="coerce").to_numpy(np.float64)
    outs_v = pd.to_numeric(frame["outs_before"], errors="coerce").to_numpy(np.float64)

    # ── 구종 성향 밴드 (--arm-split way=mix_ent / mix_dom 용) ──────────
    # 근거: `package/src/cell_diag.py` 실측. 챔피언의 부분군 격차가
    #   mix_ent 축에서 헤드룸 **+118.1** (pitcher_n 은 +48.4). 최대 미탐색 축이다.
    #   단조 624.2 / 보통 965.1 / 다양 572.4  (f23R 626 / 862 / 379 — 부호 일치)
    #
    # ⚠️ **그 투구의 구종은 데이터에 없다.** train/test 에 구종 열이 없고
    #   있는 것은 투수의 직전까지 레퍼토리 비율(asof)뿐이다. 그래서 분할축은
    #   행의 구종이 아니라 **투수 구종 성향**이다. as-of 라 행 독립이 유지된다.
    _mixc = ("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
             "asof_pitcher_offspeed_rate")
    _MX = np.column_stack([pd.to_numeric(frame[c], errors="coerce")
                           .to_numpy(np.float64) for c in _mixc])
    _mok = np.isfinite(_MX).all(1)
    _S = np.where(_mok[:, None], _MX, np.nan)
    _S = _S / np.clip(np.nansum(_S, 1, keepdims=True), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        _H = -np.nansum(np.where(_S > 0, _S * np.log(_S), 0.0), axis=1)
    # ln(3)=1.0986 이 최대. 0.80 / 1.00 경계는 cell_diag 와 **같은 값**이다.
    # 결측 투수는 '보통' 으로 넣는다 (분할이 전 행을 덮어야 한다 — assert 가 검사)
    ent_lo = (_mok & (_H < 0.80)).astype(np.float64)
    ent_hi = (_mok & (_H >= 1.00)).astype(np.float64)
    ent_mid = 1.0 - ent_lo - ent_hi
    _arg = np.where(_mok, np.nanargmax(np.nan_to_num(_MX, nan=-1.0), axis=1), 1)
    dom_fb = (_mok & (_arg == 0)).astype(np.float64)
    dom_os = (_mok & (_arg == 2)).astype(np.float64)
    dom_br = 1.0 - dom_fb - dom_os          # 변화구우세 + 결측
    print(f"구종 성향 밴드  엔트로피 단조 {ent_lo.mean():.1%} / "
          f"보통 {ent_mid.mean():.1%} / 다양 {ent_hi.mean():.1%}   "
          f"주구종 직구 {dom_fb.mean():.1%} / 변화구 {dom_br.mean():.1%} / "
          f"체인지업 {dom_os.mean():.1%}", flush=True)
    # ★ cfg 는 report 로 직렬화된다 — numpy 배열을 넣으면 json.dumps 에서 죽는다.
    #   그래서 마스크는 cfg 밖 지역변수로 둔다 (한 번 겪었다).
    olmask_full = None
    if args.outing_len_weight != 1.0:
        _lo, _hi = (float(x) for x in args.outing_len_range.split(","))
        olmask_full = outing_len_mask(frame, _lo, _hi)
        cfg["outing_len_range"] = [_lo, _hi]
        print(f"등판길이 {_lo:.0f}~{_hi:.0f} 구간 {olmask_full.mean()*100:.2f}% "
              f"-> 학습 가중치 {args.outing_len_weight}", flush=True)
    short = None
    if args.short_weight != 1.0:
        short = short_outing_mask(frame)
        print(f"짧은 등판 {short.mean()*100:.2f}% -> 학습 가중치 {args.short_weight}",
              flush=True)

    base_params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))["best_params"]
    half_life = float(base_params.pop("half_life"))
    if args.half_life > 0:
        print(f"반감기 {half_life:.3f} -> {args.half_life:.3f}", flush=True)
        half_life = args.half_life
    if args.l2 > 0:
        base_params["l2_leaf_reg"] = args.l2
    if args.bagging_temp >= 0:
        base_params["bagging_temperature"] = args.bagging_temp
    if args.random_strength >= 0:
        base_params["random_strength"] = args.random_strength
    base_params.update({"iterations": args.iterations, "learning_rate": args.lr,
                        "depth": 8, "loss_function": "Logloss",
                        "eval_metric": "Logloss", "random_seed": args.seed,
                        "task_type": "GPU", "devices": "0", "verbose": False,
                        "allow_writing_files": False})

    ctx = P.load_context()
    report = {"exp": args.exp, "cfg": cfg, "folds": {}}
    print(f"{'=' * 100}\n{args.exp}   fpre0={cfg['fpre0']}  "
          f"late_boost={args.late_boost}  month_decay={args.month_decay}\n{'=' * 100}",
          flush=True)

    success_by_fold = {}
    for fold in [int(f) for f in args.folds.split(",") if f.strip()]:
        t0 = time.time()
        tr = (season < fold) & comp_ok
        if args.min_pitcher_n > 0:
            _pn = pd.to_numeric(frame["asof_pitcher_n"],
                                errors="coerce").to_numpy(float)
            tr = tr & (np.nan_to_num(_pn, nan=0.0) >= args.min_pitcher_n)
            print(f"  학습행 asof_pitcher_n >= {args.min_pitcher_n:g} "
                  f"-> {int(tr.sum()):,}행", flush=True)
        _ex = [int(x) for x in args.exclude_seasons.split(",") if x.strip()]
        if _ex:
            tr = tr & ~np.isin(season, _ex)
            print(f"  시즌 제외 {_ex} -> {sorted(set(season[tr].tolist()))} "
                  f"({int(tr.sum()):,}행)", flush=True)
        if args.min_season > 0:
            tr = tr & (season >= args.min_season)
            print(f"  학습 시즌 하한 {args.min_season} -> "
                  f"{sorted(set(season[tr].tolist()))} ({int(tr.sum()):,}행)", flush=True)
        va = (season == fold) & comp_ok
        static = build_component_unique(frame, enhanced, fold)
        forward = build_component_unique_forward(frame, enhanced, fold,
                                                 cache={fold: static})
        base_fr, base_feats = make_features(frame, enhanced, fold, "F1", forward)
        for c in (SUCCESS, "season", "row_id"):
            if c not in base_fr:
                base_fr[c] = frame[c].to_numpy()
        base_cats = [c for c in CATEGORICAL_COLUMNS if c in base_feats]
        tr_series = pd.Series(tr, index=frame.index)
        fr_rep = {"train_rows": int(tr.sum()), "valid_rows": int(va.sum()),
                  "targets": {}}
        preds = {}

        for way, combo in combos.items():
            path = OUT / f"{args.exp}__{way}__{fold}.npy"
            vals = pd.to_numeric(labeled[LABELS[way]],
                                 errors="coerce").to_numpy(np.float64)
            mr_cond = cfg.get("mr_conditional") and way == "mr"
            if mr_cond:
                # p(mr) = p(m) x p(r|m).  여기서는 r|m 을 만든다 —
                # 타깃은 y_reverse, 학습행은 y_middle==1 인 행만.
                # 검증행은 전 행 (예측 후 p_middle 을 곱한다).
                vals = pd.to_numeric(labeled["y_reverse"],
                                     errors="coerce").to_numpy(np.float64)
                _m1 = pd.to_numeric(labeled["y_middle"],
                                    errors="coerce").to_numpy(np.float64) == 1
            t_tr, t_va = tr & np.isfinite(vals), va & np.isfinite(vals)
            if mr_cond:
                t_tr = t_tr & _m1
            if not mr_cond and (not np.array_equal(t_tr, tr)
                                or not np.array_equal(t_va, va)):
                raise RuntimeError(f"{way} 라벨 결측")
            if path.exists():
                pred = np.load(path).astype(np.float64)
                print(f"  cached {path.name}", flush=True)
            else:
                X, feats, cats = T.build(base_fr, base_feats, base_cats,
                                         sorted(combo), tr_series, fold)
                assert_features_clean(feats, f"new_val/{args.exp}/{way}/{fold}")
                if cfg["keep_top"] > 0 and len(feats) > cfg["keep_top"]:
                    # ★ 중요도는 **학습 행만**으로 구한다 (t_tr). 평가 fold 미사용.
                    from catboost import CatBoostClassifier as _CB, Pool as _P
                    _X = X.loc[t_tr, feats].copy()
                    for _c in cats:
                        _X[_c] = _X[_c].fillna("__MISSING__").astype(str)
                    _m = _CB(iterations=200, depth=6, learning_rate=0.1,
                             task_type="GPU", devices="0", verbose=False,
                             allow_writing_files=False,
                             random_seed=int(args.seed))
                    _m.fit(_P(_X, label=(vals[t_tr] == 1).astype(int),
                              cat_features=cats))
                    _imp = _m.get_feature_importance()
                    _rank = np.argsort(-np.asarray(_imp))[:cfg["keep_top"]]
                    _keep = [feats[i] for i in sorted(_rank)]
                    print(f"    중요도 선별 {len(feats)} -> {len(_keep)}피처",
                          flush=True)
                    feats = _keep
                    cats = [c for c in cats if c in set(feats)]
                    del _X, _m
                    gc.collect()
                if cfg["drop_foldrel"]:
                    # temporal_cyclic 의 fold 상대 피처 2 종.
                    #   years_to_prediction   = season - fold
                    #   season_month_progress = (season - fold) * 12 + month
                    # 학습 행은 season < fold 라 값이 음수이고, 검증/시험 행은
                    # season == fold 라 양수다. **검증 행의 0.0% 만 학습 범위 안**
                    # 이어서 트리가 외삽을 못 하고 전부 "학습셋 최근 슬라이스"로
                    # 뭉갠다. 시즌이 갈수록 그 앵커가 낡아 9 월에 붕괴한다.
                    _fr = [c for c in feats if c in
                           ("prep_years_to_prediction",
                            "prep_season_month_progress")]
                    if _fr:
                        feats = [c for c in feats if c not in _fr]
                        print(f"    fold상대 피처 제거 {_fr} -> {len(feats)}피처",
                              flush=True)
                if cfg["drift_drop"] > 0:
                    # 학습 시즌 중 최근 두 해를 비교해 이동이 큰 피처를 뺀다.
                    # 평가 데이터는 보지 않는다 (학습 시즌 두 개만 비교).
                    live = np.unique(season[t_tr])
                    s_a, s_b = int(live[-2]), int(live[-1])
                    idx = np.flatnonzero(t_tr)
                    ma, mb = season[t_tr] == s_a, season[t_tr] == s_b
                    drift = {}
                    for c in feats:
                        if c in cats:
                            continue
                        vv = pd.to_numeric(X.iloc[idx][c],
                                           errors="coerce").to_numpy(float)
                        a_, b_ = vv[ma], vv[mb]
                        sd_ = np.nanstd(np.concatenate([a_, b_]))
                        if not np.isfinite(sd_) or sd_ < 1e-12:
                            continue
                        qs = np.arange(0.1, 1.0, 0.1)
                        drift[c] = float(np.nanmedian(np.abs(
                            np.nanquantile(a_, qs) - np.nanquantile(b_, qs)))) / sd_
                    bad = [c for c, _ in sorted(drift.items(),
                                                key=lambda kv_: -kv_[1])
                           ][:cfg["drift_drop"]]
                    feats = [c for c in feats if c not in bad]
                    print(f"    분포이동 {len(bad)}개 제거 ({s_a} vs {s_b}) "
                          f"-> {len(feats)}피처", flush=True)
                Xtr, Xva = X.loc[t_tr, feats].copy(), X.loc[t_va, feats].copy()
                Xva_raw = None
                Xtr_mask = None
                if cfg["mask"]:
                    import re as _re
                    mt_ = _re.fullmatch(r"(clip|nan)(\d{2,3})", cfg["mask"])
                    if not mt_:
                        raise SystemExit(f"모르는 마스킹 {cfg['mask']}")
                    mode, qq = mt_.group(1), int(mt_.group(2))
                    q = 1.0 - qq / (10 ** len(str(qq)))
                    lo_q, hi_q = q, 1.0 - q
                    nums = [c for c in feats if c not in cats]
                    # 임계는 **학습 행에서만** 구한다. 검증/테스트에는 상수를 적용한다.
                    bounds = {}
                    for c in nums:
                        vv = pd.to_numeric(Xtr[c], errors="coerce").to_numpy(float)
                        if not np.isfinite(vv).any():
                            continue
                        bounds[c] = (float(np.nanquantile(vv, lo_q)),
                                     float(np.nanquantile(vv, hi_q)))

                    def _apply(fr_):
                        out_ = fr_.copy()
                        for c_, (lo_, hi_) in bounds.items():
                            vv_ = pd.to_numeric(out_[c_],
                                                errors="coerce").to_numpy(float)
                            out_[c_] = (np.clip(vv_, lo_, hi_) if mode == "clip"
                                        else np.where((vv_ < lo_) | (vv_ > hi_),
                                                      np.nan, vv_))
                        return out_

                    Xva_raw = Xva.copy()            # 6.7 — val 원본도 함께 낸다
                    Xva = _apply(Xva)               # 6.5 — 기본은 val 마스킹
                    if cfg["mask_augment"]:
                        Xtr_mask = _apply(Xtr)      # 원본 + 사본
                    else:
                        Xtr = _apply(Xtr)
                    print(f"    {mode}{qq} {len(bounds)}열 "
                          f"[{lo_q:.4f},{hi_q:.4f}]  augment={cfg['mask_augment']}",
                          flush=True)
                for c in cats:
                    Xtr[c] = Xtr[c].fillna("__MISSING__").astype(str)
                    Xva[c] = Xva[c].fillna("__MISSING__").astype(str)
                    if Xva_raw is not None:
                        Xva_raw[c] = Xva_raw[c].fillna("__MISSING__").astype(str)
                    if Xtr_mask is not None:
                        Xtr_mask[c] = Xtr_mask[c].fillna("__MISSING__").astype(str)
                if way in cfg["arm_split"]:
                    ax = cfg["arm_split"][way]
                    if ax == "ball":
                        parts = [("b1", vals * yball),
                                 ("b0", vals * (1.0 - yball))]
                    elif ax == "ball_late":
                        parts = [(f"b{b}l{l}",
                                  vals * (yball if b else 1 - yball)
                                  * (late_v if l else 1 - late_v))
                                 for b in (1, 0) for l in (1, 0)]
                    elif ax == "ball_outs":
                        parts = [(f"b{b}o{o}",
                                  vals * (yball if b else 1 - yball)
                                  * (outs_v == o))
                                 for b in (1, 0) for o in (0.0, 1.0, 2.0)]
                    elif ax == "mix_ent":
                        # 구종 레퍼토리 엔트로피 3분할 (사용자 지시 — 구종별 분리 모델링)
                        parts = [("me_lo", vals * ent_lo),
                                 ("me_mid", vals * ent_mid),
                                 ("me_hi", vals * ent_hi)]
                    elif ax == "mix_dom":
                        # 주 구종 3분할
                        parts = [("md_fb", vals * dom_fb),
                                 ("md_br", vals * dom_br),
                                 ("md_os", vals * dom_os)]
                    elif ax == "ball_mixent":
                        # 기존 ball 분할 × 엔트로피 (outside 는 ball 축이 최적이었다)
                        parts = [(f"b{b}me{k}", vals * (yball if b else 1 - yball) * mk)
                                 for b in (1, 0)
                                 for k, mk in (("lo", ent_lo), ("mid", ent_mid),
                                               ("hi", ent_hi))]
                    else:
                        raise SystemExit(f"모르는 분할축 {ax}")
                    tot = sum(pv_ for _t, pv_ in parts)
                    err_ = np.nanmax(np.abs(tot[t_tr] - vals[t_tr]))
                    assert err_ < 1e-9, (
                        f"분할이 타깃을 안 덮는다 (축 {ax}, {len(parts)}분할, "
                        f"최대오차 {err_:.2e})")
                else:
                    parts = [("all", vals)]
                yv = vals[t_tr].astype("int8")
                prior = train_season_trend(yv, season[t_tr], fold)
                bl = float(np.log(prior / (1.0 - prior)))
                w0 = np.asarray(recency_weights(
                    frame.loc[t_tr, "season"], fold, half_life), np.float64)
                w = build_weights(w0, season[t_tr], month[t_tr], gtype[t_tr],
                                  fold, cfg, way,
                                  None if short is None else short[t_tr],
                                  None if olmask_full is None
                                  else olmask_full[t_tr])
                kept = float((w > 0).mean())
                p_tr = Pool(Xtr, label=yv, cat_features=cats, weight=w,
                            baseline=np.full(int(t_tr.sum()), bl))
                p_va = Pool(Xva, cat_features=cats,
                            baseline=np.full(int(t_va.sum()), bl))
                pp = dict(base_params)
                if way in cfg["way_depth"]:
                    pp["depth"] = cfg["way_depth"][way]
                if way in cfg["way_iter"]:
                    pp["iterations"] = cfg["way_iter"][way]
                if way in cfg["way_l2"]:
                    pp["l2_leaf_reg"] = cfg["way_l2"][way]
                if cfg["grow_policy"]:
                    pp["grow_policy"] = cfg["grow_policy"]
                    if cfg["grow_policy"] == "Lossguide":
                        # leaf-wise 에서는 depth 가 상한일 뿐이라 max_leaves 가 실질 용량이다
                        pp["max_leaves"] = int(cfg["max_leaves"] or 31)
                        pp["depth"] = 16
                        pp.pop("boosting_type", None)
                t1 = time.time()
                print(f"  {way:<8} fold {fold}  {int(t_tr.sum()):,}행 "
                      f"{len(feats)}피처  가중치>0 {kept:.1%}  "
                      f"d{pp['depth']} it{pp['iterations']} "
                      f"arm={len(parts)}", flush=True)
                pred = np.zeros(int(t_va.sum()))
                pred_raw = np.zeros(int(t_va.sum()))
                for _tag, pv_ in parts:
                    ac = cfg["arm_combo"].get(f"{way}:{_tag}")
                    if ac:
                        # 이 arm 만 다른 전처리로 다시 만든다.
                        # outside 분할이 +10.5 였던 건 두 사건이 이질적이기 때문이다 —
                        # 그렇다면 피처도 달라야 한다는 게 자연스러운 확장이다.
                        Xa, fa, ca = T.build(base_fr, base_feats, base_cats,
                                             ac, tr_series, fold)
                        assert_features_clean(fa, f"new_val/{args.exp}/{way}/{_tag}")
                        Xtr_a, Xva_a = Xa.loc[t_tr, fa].copy(), Xa.loc[t_va, fa].copy()
                        for _c in ca:
                            Xtr_a[_c] = Xtr_a[_c].fillna("__MISSING__").astype(str)
                            Xva_a[_c] = Xva_a[_c].fillna("__MISSING__").astype(str)
                        print(f"    arm {_tag}: 조합 {ac} ({len(fa)}피처)", flush=True)
                    else:
                        Xtr_a, Xva_a, ca = Xtr, Xva, cats
                    yp = pv_[t_tr].astype("int8")
                    pr_ = train_season_trend(yp, season[t_tr], fold)
                    bl_ = float(np.log(pr_ / (1.0 - pr_)))
                    if Xtr_mask is not None:
                        XX = pd.concat([Xtr, Xtr_mask], ignore_index=True)
                        yy_ = np.concatenate([yp, yp])
                        ww_ = np.concatenate([w * 0.5, w * 0.5])
                    else:
                        XX, yy_, ww_ = Xtr_a, yp, w
                    if cfg["model"] == "xgb":
                        # XGBoost 경로 — LGB 와 **같은 계약**이다.
                        #   baseline=logit(prior) -> base_margin (예측 시 다시 더한다)
                        #   범주형은 pandas category dtype + enable_categorical
                        # 팀 실측상 다양성은 **라이브러리 경계**에서 나온다
                        # (LGB 편입이 블렌드에 +9.1). XGB 는 세 번째 경계다.
                        from xgboost import XGBClassifier
                        A, B_ = XX.copy(), Xva.copy()
                        obj = [c for c in A.columns
                               if c in cats or A[c].dtype == object
                               or str(A[c].dtype) == "string"]
                        for c in obj:
                            u = pd.Index(sorted(set(A[c].astype(str)) |
                                                set(B_[c].astype(str))))
                            A[c] = pd.Categorical(A[c].astype(str), categories=u)
                            B_[c] = pd.Categorical(B_[c].astype(str), categories=u)
                        for c in A.columns:
                            if str(A[c].dtype) != "category":
                                A[c] = pd.to_numeric(A[c], errors="coerce")
                                B_[c] = pd.to_numeric(B_[c], errors="coerce")
                        # LGB 와 같은 이유로 가중치를 평균 1 로 정규화한다
                        # (min_child_weight 가 헤시안 합 기준이다).
                        ww_n = np.asarray(ww_, np.float64)
                        ww_n = ww_n / max(ww_n.mean(), 1e-12)
                        m = XGBClassifier(
                            n_estimators=int(pp["iterations"]),
                            learning_rate=float(pp["learning_rate"]),
                            max_depth=int(cfg["xgb_depth"]),
                            tree_method="hist", device="cuda",
                            enable_categorical=True, max_cat_to_onehot=1,
                            min_child_weight=1e-3, reg_lambda=1.0,
                            subsample=1.0, colsample_bytree=1.0,
                            random_state=int(pp["random_seed"]),
                            n_jobs=-1, verbosity=0,
                            # ★ base_margin 대신 base_score 로 사전확률을 준다.
                            # base_margin 은 fit 에만 반영되고 predict 에는 안 들어가서
                            # 첫 시도에서 p_success 평균이 0.8518 로 튀었다.
                            # base_score 는 모델에 저장돼 predict_proba 에 포함된다.
                            base_score=float(np.clip(pr_, 1e-4, 1 - 1e-4)))
                        m.fit(A, yy_, sample_weight=ww_n)
                        pred = pred + m.predict_proba(B_)[:, 1]
                        if Xva_raw is not None:
                            Braw = Xva_raw.copy()
                            for c in obj:
                                Braw[c] = pd.Categorical(
                                    Braw[c].astype(str),
                                    categories=B_[c].cat.categories)
                            for c in Braw.columns:
                                if str(Braw[c].dtype) != "category":
                                    Braw[c] = pd.to_numeric(Braw[c], errors="coerce")
                            pred_raw = pred_raw + m.predict_proba(Braw)[:, 1]
                        del A, B_, m
                        gc.collect()
                        continue
                    if cfg["model"] == "lgb":
                        # LightGBM 경로 — CatBoost 와 같은 계약을 맞춘다.
                        #   baseline=logit(prior)  ->  init_score
                        #   depth d (대칭 트리 2^d 리프) -> num_leaves 2^d - 1
                        #   범주형은 pandas category dtype 으로 넘긴다
                        # yn 실측: LGB 단독 748.41 -> +CatBoost 50:50 = 899.76 (+151.35)
                        # 단독 성능이 아니라 **섞는 것**이 이득이라는 게 요점이다.
                        from lightgbm import LGBMClassifier
                        A, B_ = XX.copy(), Xva.copy()
                        # LightGBM 은 object dtype 을 못 받는다.
                        # cats 목록 밖에도 object 열이 남아 있어 실패했었다 —
                        # **실제 dtype 으로 판정**해서 전부 category 로 바꾼다.
                        obj = [c for c in A.columns
                               if c in cats or A[c].dtype == object
                               or str(A[c].dtype) == "string"]
                        for c in obj:
                            u = pd.Index(sorted(set(A[c].astype(str)) |
                                                set(B_[c].astype(str))))
                            A[c] = pd.Categorical(A[c].astype(str), categories=u)
                            B_[c] = pd.Categorical(B_[c].astype(str), categories=u)
                        for c in A.columns:
                            if str(A[c].dtype) not in ("category",):
                                A[c] = pd.to_numeric(A[c], errors="coerce")
                                B_[c] = pd.to_numeric(B_[c], errors="coerce")
                        cats = obj
                        # ★ 가중치를 평균 1 로 정규화한다.
                        # recency 가중은 2019 행이 약 0.13 이라 평균이 1 을 한참 밑돈다.
                        # LightGBM 의 min_child_weight 는 **헤시안 합** 기준이라
                        # 작은 가중치에서 분할이 막힌다. CatBoost 는 영향이 없다.
                        # 첫 시도(255.9, 오차 +0.0154)의 과소적합 원인이 여기다.
                        ww_n = np.asarray(ww_, np.float64)
                        ww_n = ww_n / max(ww_n.mean(), 1e-12)
                        m = LGBMClassifier(
                            n_estimators=int(pp["iterations"]),
                            learning_rate=float(pp["learning_rate"]),
                            num_leaves=int(cfg["lgb_leaves"]),
                            max_depth=-1,
                            min_child_samples=20, min_child_weight=1e-3,
                            subsample=1.0, colsample_bytree=1.0,
                            reg_lambda=1.0, random_state=int(pp["random_seed"]),
                            n_jobs=-1, verbose=-1)
                        m.fit(A, yy_, sample_weight=ww_n,
                              init_score=np.full(len(yy_), bl_),
                              categorical_feature=cats)
                        raw = m.predict(B_, raw_score=True) + bl_
                        pred = pred + 1.0 / (1.0 + np.exp(-raw))
                        if Xva_raw is not None:
                            Braw = Xva_raw.copy()
                            for c in cats:
                                Braw[c] = pd.Categorical(
                                    Braw[c].astype(str),
                                    categories=B_[c].cat.categories)
                            r2 = m.predict(Braw, raw_score=True) + bl_
                            pred_raw = pred_raw + 1.0 / (1.0 + np.exp(-r2))
                        del A, B_, m
                        gc.collect()
                        continue
                    q_tr = Pool(XX, label=yy_, cat_features=ca, weight=ww_,
                                baseline=np.full(len(yy_), bl_))
                    q_va = Pool(Xva_a, cat_features=ca,
                                baseline=np.full(int(t_va.sum()), bl_))
                    m = CatBoostClassifier(**pp)
                    m.fit(q_tr)
                    pred = pred + m.predict_proba(q_va, thread_count=6)[:, 1]
                    if Xva_raw is not None:
                        q_raw = Pool(Xva_raw, cat_features=cats,
                                     baseline=np.full(int(t_va.sum()), bl_))
                        pred_raw = pred_raw + m.predict_proba(
                            q_raw, thread_count=6)[:, 1]
                        del q_raw
                    del q_tr, q_va, m
                    gc.collect()
                pred = np.clip(pred, 1e-7, 1 - 1e-7)
                if Xva_raw is not None:
                    np.save(OUT / f"{args.exp}__{way}__{fold}__valraw.npy",
                            np.clip(pred_raw, 1e-7, 1 - 1e-7))
                np.save(path, pred)
                print(f"    -> {time.time() - t1:.0f}s", flush=True)
                del X, Xtr, Xva, p_tr, p_va
                gc.collect()
            preds[way] = pred
            from harness3 import bss as way_bss
            fr_rep["targets"][way] = way_bss(vals[t_va].astype("int8"), pred)

        if args.direct:
            succ = np.clip(preds["success"], 1e-7, 1 - 1e-7)
        elif args.disjoint:
            # 서로소이므로 단순 합. 포함-배제 보정항이 없다.
            succ = np.clip(1.0 - (preds["m_only"] + preds["r_only"]
                                  + preds["mr"] + preds["outside"]),
                           1e-7, 1 - 1e-7)
        else:
            _pmr = (preds["middle"] * preds["mr"] if cfg.get("mr_conditional")
                    else preds["mr"])
            succ = np.clip(1.0 - (preds["middle"] + preds["reverse"]
                                  - _pmr + preds["outside"]),
                           1e-7, 1 - 1e-7)
        np.save(OUT / f"{args.exp}__success__{fold}.npy", succ)
        success_by_fold[fold] = succ
        fr_rep["elapsed_sec"] = time.time() - t0
        report["folds"][str(fold)] = fr_rep
        del base_fr, forward, static
        gc.collect()

    # ---- policy 채점 (최종 성공확률만) ----
    res = P.evaluate(success_by_fold.get(2024),
                     success_by_fold.get(2023), ctx) if 2024 in success_by_fold else None
    if res:
        print("\n" + P.HDR)
        print(P.render(args.exp, res))
        report["policy"] = res
    (OUT / f"{args.exp}__report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    print(f"\nsaved -> {OUT / (args.exp + '__report.json')}")


if __name__ == "__main__":
    main()
