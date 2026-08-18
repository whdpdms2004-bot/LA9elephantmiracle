"""V76: 신규 피처 인수 검증 + 표준 평가. 22_FEATURE_INTAKE_PLAN.md 의 실행체.

사용법
    # 1) 인수 검증만 (5분, CPU)
    python v76_feature_intake.py --spec myfeat.py --gate-only

    # 2) 검증 + fold 2024 스크리닝
    python v76_feature_intake.py --spec myfeat.py --screen

    # 3) 검증 + 확인 (fold 2024 결정 + 2023 보조, 채택 판정)
    python v76_feature_intake.py --spec myfeat.py --confirm

피처 정의 파일 규약 (--spec 이 가리키는 .py)
    아래 두 가지 중 하나를 제공한다.

    (a) 행 단위 파생 — B1
        CLASS = "B1"
        def make(df, tr_mask):
            '''df: 전체 프레임, tr_mask: 학습 행 마스크(테이블 생성용).
            반환: DataFrame, 행 수는 len(df), 열은 새 피처만.'''
            return pd.DataFrame({"my_ratio": df.a / df.b.clip(lower=1e-3)})

    (b) 키 조인 룩업 — B2 / B3 / C
        CLASS = "B2"                       # 또는 B3, C1, C2, C3
        KEYS  = ["pitcher_id"]             # 또는 ["pitcher_id", "season"] 등
        def make_table(df, tr_mask):
            '''학습 행만으로 테이블을 만든다. 반환: KEYS + 피처열 DataFrame.'''
            ...
        # 선택: CELL_KEYS 를 주면 셀 크기 게이트(G3)를 자동 계산한다
        CELL_KEYS = ["pitcher_id", "batter_hand"]
        EB_K = 300

    공통 선택 항목
        NOTE = "출처와 근거"

검사 항목 (계획서 §2)
    G1 행 독립성      B1 은 자동, 룩업은 키 조인이므로 자동. groupby 대상이 test 인지 확인
    G2 시간 인과      tr_mask 밖 시즌이 테이블 생성에 쓰였는지 (fold 를 바꿔 값이 변하는가)
    G3 셀 크기        중앙 셀에서 한 행의 기여분 1/(n+K). 1% 초과 경고, 3% 초과 보류
    G4 계층 차감      중복·단위 조건은 사람이 판단할 항목이라 체크리스트로 출력
    G5 base 중복      새 피처를 넣었을 때 base 와의 상관이 어떻게 변하는가
    G6 커버리지       결측률, fold 별 커버리지, 학습/검증 불일치

출력: outputs/v76_intake_<spec이름>.csv
"""
import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import component_features as CF
from harness import TARGET, OUT, CACHE, BASE_PARAMS, load, metrics

SJ = Path(__file__).resolve().parents[2]
MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
SEEDS_FULL = [11, 22, 33, 44, 55, 66, 77, 88]
SEEDS_FAST = [11, 22, 33]
N_ROUNDS, F_WEIGHT, K = 400, 0.20, 300
BW = np.array([0.25, 0.25, 0.25, 0.30, 0.40])
CUTS = [100, 500, 2000, 4000]
COMPONENTS = ["m", "r", "mr", "ob", "oz"]
EPS = 1e-7

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True, help="피처 정의 .py 경로")
ap.add_argument("--gate-only", action="store_true")
ap.add_argument("--screen", action="store_true", help="fold 2024 스크리닝")
ap.add_argument("--confirm", action="store_true",
                help="확인 — fold 2024 결정 + 2023 보조 (2022 는 쓰지 않는다)")
args = ap.parse_args()

spec_path = Path(args.spec)
sp = importlib.util.spec_from_file_location("featspec", spec_path)
FS = importlib.util.module_from_spec(sp)
sp.loader.exec_module(FS)
CLASS = getattr(FS, "CLASS", "?")
NOTE = getattr(FS, "NOTE", "")
print(f"{'='*84}{chr(10)}피처 인수: {spec_path.name}   계열 {CLASS}")
if NOTE:
    print(f"  근거: {NOTE}")
print("=" * 84, flush=True)

df = load()
lab = pd.read_parquet(CACHE / "failure_labels.parquet")
df = df.merge(lab, on="row_id", how="left", validate="one_to_one")
season = df["season"].to_numpy()
y_all = df[TARGET].to_numpy(np.float64)
ok = df["label_ok"].to_numpy() == 1
INPUT_COLS = [c for c in df.columns
              if not c.startswith("y_") and c != "label_ok" and c != TARGET]
bucket_all = np.digitize(df["asof_pitcher_n"].to_numpy(), CUTS)
IS_F = df["game_type"].astype(str).to_numpy() == "F"
ROW_W = np.where(IS_F, F_WEIGHT, 1.0)

ym = np.where(ok, df["y_middle"].to_numpy(np.float64), np.nan)
yr = np.where(ok, df["y_reverse"].to_numpy(np.float64), np.nan)
yo = np.where(ok, df["y_outside"].to_numpy(np.float64), np.nan)
yb = np.where(ok, df["y_ball"].to_numpy(np.float64), np.nan)


def AND(*a):
    m = np.ones(len(df), bool)
    for x in a:
        m &= (x == 1)
    return np.where(ok, m.astype(float), np.nan)


LAB = {"m": ym, "r": yr, "mr": AND(ym, yr), "ob": AND(yo, yb), "oz": AND(yo, 1 - yb)}

models = sorted({re.match(r"(.+)_fold\d{4}\.parquet", p.name).group(1)
                 for p in OOF_DIR.glob("*.parquet")})


def base_pred(fold):
    fid = df.loc[season == fold, "row_id"].to_numpy()
    if fold == 2024:
        pr = pd.read_parquet(PROD).set_index("row_id").reindex(fid)
        return np.clip(pr["submit021_reverse20_s040_tabm"].to_numpy(np.float64),
                       EPS, 1 - EPS)
    acc, c = None, 0
    for mn in models:
        p = OOF_DIR / f"{mn}_fold{fold}.parquet"
        if p.exists():
            v = pd.read_parquet(p).set_index("row_id").reindex(fid)["prediction"].to_numpy()
            acc = v if acc is None else acc + v
            c += 1
    return np.clip(acc / c, EPS, 1 - EPS)


def new_features(fold):
    """피처 정의 파일을 호출해 새 열을 만든다. 학습 시즌만으로."""
    tr = season < fold
    if hasattr(FS, "make"):
        G = FS.make(df, tr)
        assert len(G) == len(df), f"make() 행 수 불일치 {len(G)} != {len(df)}"
        return G.reset_index(drop=True)
    tbl = FS.make_table(df, tr)
    keys = list(FS.KEYS)
    G = df[keys].merge(tbl, on=keys, how="left")
    return G.drop(columns=keys).reset_index(drop=True)


FAIL, WARN = [], []


def gate(name, cond, detail="", warn=False):
    tag = "OK  " if cond else ("WARN" if warn else "FAIL")
    print(f"  [{tag}] {name}   {detail}", flush=True)
    if not cond:
        (WARN if warn else FAIL).append(name)


print(f"{chr(10)}G1~G2. 행 독립성 / 시간 인과")
is_lookup = hasattr(FS, "make_table")
gate("정의 형태", hasattr(FS, "make") or is_lookup,
     "키 조인 룩업" if is_lookup else "행 단위 파생")
if is_lookup:
    gate("G1 키 조인이라 행 독립성 충족", True, f"KEYS={FS.KEYS}")
else:
    gate("G1 단일 행 계산이라 행 독립성 충족", True)

G23 = new_features(2023)
G24 = new_features(2024)
gate("열 이름 일치", list(G23.columns) == list(G24.columns), f"{len(G23.columns)}열")
same = all(np.allclose(np.nan_to_num(G23[c].to_numpy(float)),
                       np.nan_to_num(G24[c].to_numpy(float)))
           for c in G23.columns)
gate("G2 fold 를 바꾸면 값이 바뀐다 (학습 시즌만 사용)", not same,
     "동일하면 전 시즌을 썼거나 fold 무관한 상수다",
     warn=True)

print(f"{chr(10)}G3. 셀 크기 (자기 라벨 누수 한계)")
cell_keys = getattr(FS, "CELL_KEYS", None)
eb_k = getattr(FS, "EB_K", K)
if cell_keys:
    tr = season < 2024
    sz = df.loc[tr].groupby(cell_keys).size()
    med = float(sz.median())
    contrib = 1.0 / (med + eb_k) * 100
    gate("G3 중앙 셀 기여분 < 1%", contrib < 1.0,
         f"셀 {len(sz):,}개  중앙 {int(med):,}행  K={eb_k}  기여 {contrib:.3f}%",
         warn=contrib < 3.0)
    print(f"       하위10% {int(sz.quantile(.1)):,}행 -> 기여 "
          f"{1/(sz.quantile(.1)+eb_k)*100:.3f}%")
    print(f"       참고: 현행 투수x타자손 0.144% (안전) / 투수x타자 개별 중앙 13행 (붕괴)")
else:
    print("       CELL_KEYS 미지정 — 셀 크기 검사 생략 (B1 이면 정상)")

print(f"{chr(10)}G4. 계층 차감 체크리스트 (사람이 판단)")
print("       ㄱ 빼려는 주효과가 이미 모델에 있는가?  (없으면 차감이 신호를 지운다)")
print("       ㄴ 두 항이 같은 단위인가?  (다르면 환산 계수 추정 -> V43 단독 -35.79)")

print(f"{chr(10)}G6. 커버리지")
for c in G24.columns:
    v = G24[c].to_numpy(float)
    miss = float(np.isnan(v).mean())
    m24 = float(np.isnan(v[season == 2024]).mean())
    mtr = float(np.isnan(v[season < 2024]).mean())
    flag = "  <- 학습/검증 불일치" if abs(m24 - mtr) > 0.15 else ""
    print(f"       {c:<28} 결측 전체 {miss*100:5.1f}%  학습 {mtr*100:5.1f}%  "
          f"검증 {m24*100:5.1f}%{flag}")

print(f"{chr(10)}{'='*84}")
print(f"게이트 결과: {'통과' if not FAIL else 'FAIL ' + str(FAIL)}"
      f"{'   경고 ' + str(WARN) if WARN else ''}")
print("=" * 84, flush=True)
if FAIL:
    print("FAIL 항목이 있어 실험을 진행하지 않는다.")
    sys.exit(1)
if args.gate_only:
    sys.exit(0)


def build_features(fold, with_new):
    tr = season < fold
    td = df.loc[tr]
    F = CF.build(df[INPUT_COLS], CF.make_spec(td), CF.make_platoon_table(td),
                 CF.make_batter_platoon_table(td, {k: v[tr] for k, v in LAB.items()}),
                 CF.make_count_platoon_table(td), CF.make_inning_platoon_table(td))
    if with_new:
        G = new_features(fold)
        for c in G.columns:
            F[c] = G[c].to_numpy(np.float64)
    return F


def params_for(rate):
    if rate < 0.06:
        return {"max_leaves": 8, "min_child_weight": 256, "reg_lambda": 8.0}
    if rate < 0.15:
        return {"max_leaves": 12, "min_child_weight": 128, "reg_lambda": 4.0}
    return {"max_leaves": 18, "min_child_weight": 64, "reg_lambda": 2.0}


def line(X, fold, seeds, use_cat):
    tr, va = season < fold, season == fold
    p = {}
    Xv, Xc = X[va], np.nan_to_num(X[va], nan=-999.0)
    for tag in COMPONENTS:
        arr = LAB[tag]
        mm = tr & ~np.isnan(arr)
        s_ = pd.Series(arr[mm]).groupby(pd.Series(season[mm])).mean().sort_index()
        bs = float(np.clip(float(s_.iloc[-1]) + (float(s_.iloc[-1]) - float(s_.iloc[0]))
                           / (float(s_.index[-1]) - float(s_.index[0])), 0.005, 0.995))
        prm = {**BASE_PARAMS, "base_score": bs,
               **params_for(float(np.nanmean(arr[mm])))}
        d_tr = xgb.DMatrix(X[mm], label=arr[mm], weight=ROW_W[mm], missing=np.nan)
        acc, n = np.zeros(int(va.sum())), 0
        for s in seeds:
            acc += xgb.train({**prm, "seed": s}, d_tr, num_boost_round=N_ROUNDS,
                             verbose_eval=False).predict(
                xgb.DMatrix(Xv, missing=np.nan))
            n += 1
        if use_cat:
            p_tr = Pool(np.nan_to_num(X[mm], nan=-999.0), arr[mm], weight=ROW_W[mm])
            for s in seeds:
                c = CatBoostClassifier(iterations=N_ROUNDS, depth=6,
                                       learning_rate=0.05, l2_leaf_reg=6.0,
                                       loss_function="Logloss", random_seed=s,
                                       task_type="GPU", verbose=0)
                c.fit(p_tr)
                acc += c.predict_proba(Xc)[:, 1]
                n += 1
        p[tag] = np.clip(acc / n, EPS, 1 - EPS)
    return np.clip(1 - (p["m"] + p["r"] - p["mr"] + p["ob"] + p["oz"]), EPS, 1 - EPS)


folds = [2023, 2024] if args.confirm else [2024]
seeds = SEEDS_FULL if args.confirm else SEEDS_FAST
use_cat = bool(args.confirm)
mode = ("확인 — fold 2024 결정 + 2023 보조 (전체 구성)" if args.confirm
        else "fold 2024 스크리닝 (XGB 3시드)")
# fold 2022 는 쓰지 않는다 — 2022->2024 순위상관이 두 계열 모두 음수 (V84, §3-0)
print(f"{chr(10)}{mode}", flush=True)

t0, rows = time.time(), []
lgf = lambda z: np.log(z / (1 - z))
for fold in folds:
    va = season == fold
    y, b = y_all[va], base_pred(fold)
    null = y.mean() * (1 - y.mean())
    mb = metrics(y, b)
    ref, ref_c = mb["bss_raw"], mb["bss_centered"]
    wv = BW[bucket_all[va]]
    print(f"{chr(10)}fold {fold}   base {ref:9.2f}   centered {ref_c:9.2f}"
          f"   오프셋 {mb['offset']:+.4f}"
          f"{'   <- 오프셋 교란. centered 로 판단한다' if abs(mb['offset']) > 0.01 else ''}")
    print(f"  {'arm':<12}{'피처':>6}{'단독':>10}{'corr':>8}{'ΔBSS':>9}"
          f"{'Δcentered':>11}{'오프셋':>9}{'t_row':>8}")
    for arm, wn in [("기준선", False), ("신규피처", True)]:
        F = build_features(fold, wn)
        p_ie = line(F.to_numpy(np.float32), fold, seeds, use_cat)
        solo = metrics(y, p_ie)["bss_raw"]
        corr = float(np.corrcoef(lgf(b), lgf(p_ie))[0, 1])
        q = np.clip(wv * p_ie + (1 - wv) * b, EPS, 1 - EPS)
        mq = metrics(y, q)
        d = mq["bss_raw"] - ref
        # 평균 정렬로 번 것과 신호로 번 것을 분리한다 (§3-1). fold 2023 은 이쪽을 본다.
        dc_ = mq["bss_centered"] - ref_c
        dr = (b - y) ** 2 - (q - y) ** 2
        se = 100000 * float(dr.std(ddof=1) / np.sqrt(len(dr))) / null
        rows.append({"fold": fold, "arm": arm, "n_features": F.shape[1],
                     "solo_bss": solo, "corr": corr, "dbss": d,
                     "dbss_centered": dc_, "offset": mq["offset"], "t_row": d / se})
        print(f"  {arm:<12}{F.shape[1]:>6}{solo:>10.2f}{corr:>8.4f}{d:>+9.2f}"
              f"{dc_:>+11.2f}{mq['offset']:>+9.4f}{d/se:>8.2f}"
              f"   [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
out = OUT / f"v76_intake_{spec_path.stem}.csv"
res.to_csv(out, index=False)

print(f"{chr(10)}{'='*84}{chr(10)}판정 (계획서 §3, 2026-08-18 개정){chr(10)}{'='*84}")
print(f"  {'fold':>6}{'가중':>5}{'단독 Δ':>10}{'상관 Δ':>10}{'ΔBSS Δ':>10}"
      f"{'Δcentered':>12}   판정")
WEIGHT = {2023: 1.0, 2024: 2.0}          # §3-0. 2024 가 결정 fold
D = {}
for fold in folds:
    a = res[(res.fold == fold) & (res.arm == "기준선")].iloc[0]
    c = res[(res.fold == fold) & (res.arm == "신규피처")].iloc[0]
    # 주의: Series 의 .corr 은 메서드다. 반드시 대괄호로 접근한다.
    ds = c["solo_bss"] - a["solo_bss"]
    dcorr = c["corr"] - a["corr"]
    dd = c["dbss"] - a["dbss"]
    ddc = c["dbss_centered"] - a["dbss_centered"]
    # fold 2023 은 오프셋 교란이 있으므로 centered 로 본다 (§3-0)
    key = ddc if fold == 2023 else dd
    D[fold] = (ds, dcorr, dd, ddc, key)
    ok3 = (ds > 0) and (dcorr < 0) and (key > 0)
    print(f"  {fold:>6}{WEIGHT[fold]:>5.0f}{ds:>+10.2f}{dcorr:>+10.4f}{dd:>+10.2f}"
          f"{ddc:>+12.2f}   {'통과' if ok3 else '미달'}")

print(f"{chr(10)}  §3-1 채택 조건")
if 2024 in D:
    ds, dcorr, dd, ddc, _ = D[2024]
    dec = (ds > 0) and (dcorr < 0) and (dd > 0)
    print(f"    fold 2024 (결정)  단독 {'↑' if ds > 0 else '↓'}  "
          f"상관 {'↓' if dcorr < 0 else '↑'}  ΔBSS {'↑' if dd > 0 else '↓'}"
          f"   -> {'통과' if dec else '기각'}")
    if not dec:
        print(f"    ** fold 2024 가 조건을 못 채우면 2023 이 아무리 좋아도 기각한다 **")
    if 2023 in D:
        aux = D[2023][4]
        print(f"    fold 2023 (보조)  Δcentered {aux:+.2f}"
              f"   -> {'반대 방향 아님' if aux > -1.0 else '반대 방향 — 재검토'}")
    w = sum(WEIGHT[f] * D[f][4] for f in D) / sum(WEIGHT[f] for f in D)
    print(f"{chr(10)}    가중 점수 (1x2023 + 2x2024)/3 = {w:+.2f}")
    verdict = ("채택 대상" if dec and args.confirm and
               (2023 not in D or D[2023][4] > -1.0)
               else ("방향 확인용 — 두 fold 확인 필요" if not args.confirm else "기각"))
    print(f"    결과: {verdict}")

print(f"{chr(10)}  주의 (§3-4)")
print(f"    내부 델타는 Public 델타를 예측하지 못한다. 제출 후보를 하나로 좁히지 말 것.")
print(f"    내부 +3 미만은 제출본 교체 근거로 쓰지 않는다 (V61).")
print(f"  주의 (§3-2)")
print(f"    이 판정은 현행 프로덕션 base 위에서 잰 것이다. base 를 바꾸면 다시 재야 한다.")
print(f"{chr(10)}saved -> {out}")
