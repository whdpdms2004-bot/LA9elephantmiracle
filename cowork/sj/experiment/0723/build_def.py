"""D(EDA) / E(피처확장·튜닝) / F(CS·W 분해) 노트북 생성."""
import nbformat as nbf
from pathlib import Path
HERE = Path(__file__).resolve().parent

SETUP = """
import json, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "src"); import csw_pipeline as P
import plotstyle; plotstyle.apply()   # 한글 폰트
V3 = Path("out/v3")
meta = json.load(open("cache/meta.json")); df = pd.read_parquet("cache/features.parquet")
tr = df["game_year"].isin(P.TRAIN_YEARS).to_numpy(); te = df["game_year"].eq(2019).to_numpy()
BASE = {"league_mean": 0.5957, "count_only": 0.5790}
print(f"train {int(tr.sum()):,} / test {int(te.sum()):,} | 피처 basic {len(meta['basic_feats'])} "
      f"/ derived {len(meta['derived_feats'])} / +PitchPredict {len(meta['pp_feats'])} = full {len(meta['full_feats'])}")
"""

def nb_write(name, cells):
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                      "language_info": {"name":"python","version":"3.10"}}
    nbf.write(nb, str(HERE/name)); print("wrote", name)

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

# ══════════════════ D: EDA ══════════════════
nb_write("D_eda.ipynb", [
md("""
# D · EDA — 신호가 어디에 있는가

**목적**: 모델을 더 돌리기 전에 *신호의 위치와 크기*를 먼저 확인한다.
앞선 A/B/C에서 성능이 baseline을 거의 못 넘었기 때문에, **과제의 난이도 자체**를 진단한다.

**핵심 질문**
1. CSW는 무엇에 따라 변하는가? (카운트·이닝·좌우·투수)
2. 투구 전 정보로 도달 가능한 **천장**은 어디인가?
3. 원래 목표인 **CSW%(비율) 예측**에서는 쓸 만한가?
"""),
code(SETUP),
md("## 1. 타깃 분해 — CSW는 두 개의 다른 사건이다\n`CSW = 콜드스트라이크(지켜봄) + 헛스윙(휘두름)`. 두 경로의 결정요인이 다르다 → F에서 분리 모델링의 근거."),
code("""
E = json.load(open(V3/"eda.json")); comp = E["components"]
print(f"CSW {comp['csw']:.3f} = called {comp['called']:.3f} + whiff {comp['whiff']:.3f}")
print(f"스윙률 {comp['swing']:.3f} | 스윙 중 헛스윙 {comp['whiff_given_swing']:.3f} | 지켜본 중 콜스트라이크 {comp['called_given_take']:.3f}")
fig, ax = plt.subplots(1,2, figsize=(11,3.6))
ax[0].bar(["called","whiff"], [comp["called"], comp["whiff"]], color=["#4C72B0","#DD8452"])
ax[0].set_title("CSW 구성 (전체 투구 대비)"); ax[0].set_ylabel("비율")
ax[1].bar(["P(swing)","P(whiff|swing)","P(called|take)"],
          [comp["swing"], comp["whiff_given_swing"], comp["called_given_take"]], color="#55A868")
ax[1].set_title("분해 확률"); plt.tight_layout(); plt.show()
"""),
md("## 2. 카운트가 가장 강한 신호\n2스트라이크에서 CSW가 급락(타자가 커트/보호 스윙) — 카운트만으로도 상당한 설명력."),
code("""
bc = pd.DataFrame(E["by_count"]).T.sort_values("mean")
fig, ax = plt.subplots(1,2, figsize=(12,3.6))
bc["mean"].plot.bar(ax=ax[0], color="#4C72B0"); ax[0].axhline(comp["csw"], ls="--", c="r", label="전체 평균")
ax[0].set_title("카운트별 CSW율"); ax[0].legend()
pd.Series(E["by_strikes"]).plot.bar(ax=ax[1], color="#DD8452"); ax[1].set_title("스트라이크 수별 CSW율")
plt.tight_layout(); plt.show()
print("최저", bc.index[0], round(bc['mean'].iloc[0],3), "| 최고", bc.index[-1], round(bc['mean'].iloc[-1],3))
"""),
md("## 3. 투수 간 차이 vs 상황 차이"),
code("""
ps = E["pitcher_spread"]
print(f"투수별 CSW율: 최저 {ps['min']:.3f} ~ 최고 {ps['max']:.3f} (SD {ps['sd']:.4f}, n={ps['n']})")
fig, ax = plt.subplots(1,2, figsize=(11,3.6))
pd.Series(E["by_inning"]).plot(marker="o", ax=ax[0]); ax[0].set_title("이닝별 CSW율"); ax[0].set_xlabel("inning")
pd.Series(E["by_matchup"]).plot.bar(ax=ax[1], color="#937860"); ax[1].set_title("좌우 매치업별 CSW율")
plt.tight_layout(); plt.show()
"""),
md("""## 4. ★ 천장 진단 — 신호의 90%는 '그 공이 어디로 갔는가'에 있다
투구 전 정보만으로는 알 수 없는 **현재 투구의 위치·구위**를 넣어보면 성능이 어떻게 변하는지 비교한다.
(`diagnose_ceiling.py` 결과)"""),
code("""
C = json.load(open(V3.parent/"v2"/"ceiling_diagnosis.json"))
rows = {"(1) 카운트+좌우": C["1_context_only"], "(2) +현재 투구 위치": C["2_plus_location"],
        "(3) +구속·무브·구종": C["3_plus_stuff"]}
t = pd.DataFrame(rows).T[["logloss","roc_auc","pr_auc"]]
fig, ax = plt.subplots(figsize=(7,3.5))
t["roc_auc"].plot.bar(ax=ax, color=["#C44E52","#4C72B0","#55A868"])
ax.axhline(0.6332, ls="--", c="k", label="우리 최고 pre-pitch 모델(0.633)")
ax.set_ylabel("ROC-AUC"); ax.set_title("정보를 추가할수록: 위치가 압도적"); ax.legend(); plt.xticks(rotation=15)
plt.tight_layout(); plt.show()
print("위치 추가 효과: AUC +%.3f | 우리 205~342개 투구전 피처 전부: AUC +%.3f"
      % (C["2_plus_location"]["roc_auc"]-C["1_context_only"]["roc_auc"], 0.6332-C["1_context_only"]["roc_auc"]))
t
"""),
md("""## 5. ★ 비율(CSW%) 예측에서는? — 단순 이동평균에 진다
원래 목표는 투수의 **CSW%** 예측이었다. 투구 단위 확률을 경기 단위로 평균내 비교."""),
code("""
R = json.load(open(V3.parent/"v2"/"rate_diagnosis.json"))
r = pd.DataFrame(R["results"]).set_index("방법")
print(f"등판 {R['n_outings']}개 | 실제 CSW% SD {R['actual_sd']:.4f}")
fig, ax = plt.subplots(1,2, figsize=(11,3.4))
r["R2"].plot.bar(ax=ax[0], color="#4C72B0"); ax[0].set_title("경기단위 CSW% 예측 R²"); plt.setp(ax[0].get_xticklabels(), rotation=12)
r["MAE"].plot.bar(ax=ax[1], color="#DD8452"); ax[1].set_title("MAE (낮을수록 좋음)"); plt.setp(ax[1].get_xticklabels(), rotation=12)
plt.tight_layout(); plt.show()
r
"""),
md("""## 6. EDA 결론 → 방향 조정
1. **CSW는 두 사건의 합** → 분리 모델링 시도 (**F**)
2. **카운트가 지배적**, 나머지 피처의 한계 효용이 작다 → 피처를 늘리기보다 **구조**를 바꿔야 함
3. **천장이 낮다**: 위치를 모르는 한 AUC 0.63 부근이 현실적 상한
4. **비율 예측은 이동평균이 우세** → 투구 단위 평균은 비율 예측에 부적합
→ 다음: **E** 피처 확장(Pitch Predict 차용)+고강도 튜닝, **F** CS/W 분해
"""),
])

# ══════════════════ E: 피처확장 + 튜닝 ══════════════════
nb_write("E_features_tuning.ipynb", [
md("""
# E · 피처 확장 (Pitch Predict 차용) + 고강도 튜닝

**차용 출처**: Josh Mancuso, *Pitch Predict* Part 1–3 (Analytics Vidhya)

**추가한 피처** (전부 `shift(1)` expanding — 투구 전 시점 안전)
| 그룹 | 내용 |
|---|---|
| `batter_scout` | **타자 스카우팅 리포트** — 구종 카테고리(fb/br/off)별 상대빈도·헛스윙률·콜스트라이크허용률·chase률·스윙률 (+표본수) |
| `gameflow` | 직전 3구의 존/스윙/CSW 여부, 최근 5·15구의 카테고리 비율·스트라이크율 |
| `prior_ab` | 직전 타석 결과 플래그 (삼진·볼넷/사구·안타·홈런 직후) |
| `lineup` | 타순 슬롯 근사 |

원문과의 차이: 원문은 **월별 순차 집계**로 누수를 막았지만, 여기서는 **투구 단위 expanding + shift(1)** 로 더 촘촘하게 처리했다.
구종은 원문처럼 **3개 카테고리**(fastball/breaking/offspeed)로 묶어 희소성을 줄였다.
"""),
code(SETUP),
md("## 1. 단계적 Ablation — 어느 그룹이 실제로 기여하나"),
code("""
A = json.load(open(V3/"E_ablation.json"))
a = pd.DataFrame(A).T[["logloss","roc_auc","pr_auc","n_feats"]]
fig, ax = plt.subplots(1,2, figsize=(13,3.8))
a["logloss"].plot(marker="o", ax=ax[0], color="#C44E52"); ax[0].axhline(BASE["count_only"], ls="--", c="gray", label="count-only")
ax[0].set_title("누적 LogLoss ↓"); ax[0].legend(); plt.setp(ax[0].get_xticklabels(), rotation=30, ha="right")
a["roc_auc"].plot(marker="o", ax=ax[1], color="#4C72B0"); ax[1].set_title("누적 ROC-AUC ↑")
plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right"); plt.tight_layout(); plt.show()
a
"""),
md("""**읽는 법**: 피처를 더할수록 **AUC(순위 능력)는 오르지만 LogLoss(확률 품질)는 개선되지 않는다.**
즉 새 피처가 순서는 조금 더 잘 매기지만 확률 보정은 나빠진다 → 과적합 경향. 튜닝으로 이를 잡아본다."""),
md("## 2. Optuna 40 trials (LogLoss 최소화, 경기그룹 fold)"),
code("""
O = json.load(open(V3/"E_optuna.json"))
h = pd.Series(O["history"]); best = h.cummin()
fig, ax = plt.subplots(figsize=(7,3.4))
ax.plot(h.values, "o", alpha=.5, label="trial"); ax.plot(best.values, "-", c="r", label="best so far")
ax.set_xlabel("trial"); ax.set_ylabel("val LogLoss"); ax.set_title(f"Optuna {O['n_trials']} trials"); ax.legend()
plt.tight_layout(); plt.show()
print("best CV:", O["best_cv_logloss"], "→ TEST:", O["test_2019"]["logloss"], "| AUC", O["test_2019"]["roc_auc"])
pd.Series(O["best_params"]).to_frame("best_params")
"""),
md("## 3. 최종 비교"),
code("""
rows = {"baseline: 리그평균": {"logloss": BASE["league_mean"]}, "baseline: count-only": {"logloss": BASE["count_only"]},
        "A/B라운드 최고(HistGB)": {"logloss": 0.5760},
        "E: full피처+Optuna40": {"logloss": O["test_2019"]["logloss"]}}
t = pd.DataFrame(rows).T
ax = t["logloss"].plot.bar(figsize=(7,3.4), color=["gray","gray","#DD8452","#55A868"])
ax.set_ylim(0.565, 0.60); ax.set_ylabel("TEST LogLoss ↓"); plt.xticks(rotation=15, ha="right")
plt.title("E 라운드 결과"); plt.tight_layout(); plt.show()
t
"""),
])

# ══════════════════ F: CS/W 분해 ══════════════════
nb_write("F_cs_w_decomposition.ipynb", [
md("""
# F · CS / W 분해 모델

**동기**: CSW는 성격이 다른 두 사건의 합이다.
```
P(CSW) = P(swing)·P(whiff | swing)  +  P(take)·P(called strike | take)
```
- **콜드스트라이크**: 타자의 '지켜봄' 판단 + 심판 + 존 공략 성향이 좌우
- **헛스윙**: 구위·구종 배합·타자 컨택 능력이 좌우

하나의 타깃으로 합치면 두 메커니즘이 섞여 학습이 흐려진다. **세 모델을 따로 학습해 결합**한다.
(모든 모델은 동일한 **엄격 투구 전** 피처만 사용 — 현재 투구 정보 없음)

**추가 효과**: 성능 변화가 *콜스트라이크 쪽*인지 *헛스윙 쪽*인지 설명할 수 있다.
"""),
code(SETUP),
md("## 1. 세 모델의 개별 성능\n각 모델은 자기 대상 투구에서만 평가한다 (B=스윙한 공, C=지켜본 공)."),
code("""
F = json.load(open(V3/"F_decomp.json"))
sub = pd.DataFrame({k: F[k] for k in ["A_swing","B_whiff_given_swing","C_called_given_take"]}).T
sub.index = ["A: P(swing)", "B: P(whiff|swing)", "C: P(called|take)"]
ax = sub["roc_auc"].plot.bar(figsize=(7,3.2), color="#4C72B0"); ax.set_ylabel("ROC-AUC")
ax.set_title("분해된 세 모델 (각자 대상 투구에서 평가)"); plt.xticks(rotation=10); plt.tight_layout(); plt.show()
sub[["n","base","logloss","roc_auc","pr_auc"]]
"""),
md("**관찰**: 세 사건의 예측 난이도가 서로 다르다. 특히 `P(swing)`은 카운트 의존이 커서 상대적으로 잘 맞고, 나머지는 어렵다."),
md("## 2. 결합 결과 — 분해가 단일 모델을 이긴다"),
code("""
FF = json.load(open(V3/"F_final.json"))
rows = {"단일 is_csw (기본 파라미터)": F["single_is_csw"],
        "분해 결합 (기본 파라미터)": F["combined_decomposed"],
        "단일 (Optuna 튜닝)": FF["tuned_single"],
        "분해 결합 (Optuna 튜닝)": FF["tuned_decomposed"],
        "앙상블 단일+분해 (튜닝)": FF["tuned_ensemble"]}
t = pd.DataFrame(rows).T[["logloss","roc_auc","pr_auc","ece"]]
fig, ax = plt.subplots(1,2, figsize=(13,3.8))
t["logloss"].plot.bar(ax=ax[0], color=["#DD8452","#55A868","#DD8452","#55A868","#4C72B0"])
ax[0].axhline(BASE["count_only"], ls="--", c="gray", label="count-only"); ax[0].set_ylim(0.565,0.59)
ax[0].set_ylabel("LogLoss ↓"); ax[0].legend(); plt.setp(ax[0].get_xticklabels(), rotation=25, ha="right")
t["roc_auc"].plot.bar(ax=ax[1], color=["#DD8452","#55A868","#DD8452","#55A868","#4C72B0"])
ax[1].set_ylabel("ROC-AUC ↑"); plt.setp(ax[1].get_xticklabels(), rotation=25, ha="right")
plt.tight_layout(); plt.show()
t
"""),
md("""## 3. 결론
- **분해가 단일보다 일관되게 낫다** (기본 파라미터에서도, 튜닝 후에도).
- 최종 최고: **분해 결합 + Optuna** → 아래 셀 수치.
- 여전히 count-only 대비 개선폭은 제한적 — **D의 천장 진단**(위치를 모르면 AUC 0.63 부근)과 일치한다.
"""),
code("""
best = FF["tuned_decomposed"]
print(f"최종 최고 모델: 분해 결합 + Optuna")
print(f"  TEST LogLoss {best['logloss']:.4f}  (count-only {BASE['count_only']:.4f} 대비 {best['logloss']-BASE['count_only']:+.4f})")
print(f"  ROC-AUC {best['roc_auc']:.4f} | PR-AUC {best['pr_auc']:.4f} | ECE {best['ece']:.4f}")
print(f"  리그평균({BASE['league_mean']:.4f}) 대비 {best['logloss']-BASE['league_mean']:+.4f}")
"""),
])
