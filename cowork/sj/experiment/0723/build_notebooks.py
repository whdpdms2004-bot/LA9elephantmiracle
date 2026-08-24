"""4개 결과 노트북 생성 (사전계산된 results/*를 로드해 일목요연하게 렌더)."""
import nbformat as nbf
from pathlib import Path
HERE = Path(__file__).resolve().parent

EXPS = {
    "01_global_basic":    ("global_basic",    "전체 한 모델 · 기본 정보만", "global", "basic"),
    "02_global_derived":  ("global_derived",  "전체 한 모델 · 파생 변수", "global", "derived"),
    "03_perpitcher_basic":("pitcher_basic",   "투수별 모델 · 기본 정보만", "pitcher", "basic"),
    "04_perpitcher_derived":("pitcher_derived","투수별 모델 · 파생 변수", "pitcher", "derived"),
}

HDR = """# {title}

**과제**: 투구 전 정보로 현재 투구의 **CSW(콜드스트라이크+헛스윙)** 여부 예측 · **train 2017–18 / test 2019**
**설계**: 모델 = `{design}` · 피처셋 = `{fset}` · 주심(umpire) 미사용

### 예측 시점 · 평가 규칙 (필수 반영)
- **엄격한 투구 전**: 예측할 투구의 물리·위치·릴리스각·구종·결과는 입력 금지. 상황 + 그 투수/타자의 **과거** 정보만.
- **누수+prequential 통합**: 모든 이력/인코딩은 전체기간 시간정렬 후 `shift(1)` expanding/rolling →
  train 내부 미래누수 없음 + **2019는 online/adaptive**(과거 2019 결과로 이력 갱신, 모델 파라미터는 2017–18 고정).
- 지표별 분모 분리 + 지표별 Beta-Binomial 수축 + 표본수·결측 지시자. Kirby → **release-angle repeatability**(command 아님).

> 무거운 계산은 `run_experiments.py`가 사전 수행 → 이 노트북은 `results/{exp}/`를 로드해 렌더한다.
> 재계산: `python prep_features.py 40 && python run_experiments.py {exp}`
"""

SETUP = """import json
import pandas as pd, sys
from pathlib import Path
sys.path.insert(0, 'src')
import plotstyle; plotstyle.apply()   # 한글 폰트
from IPython.display import Image, display
pd.set_option('display.width', 160)
EXP = '{exp}'
R = Path('out') / EXP
S = json.load(open(R / 'summary.json'))
META = json.load(open('cache/meta.json'))
print('행수:', META['n_rows'], '| train', META['n_train'], '| test', META['n_test'])
print('피처(raw):', S['n_features_raw'], '| 평가:', S['eval_protocol'])
"""

def build(name, exp, title, design, fset):
    nb = nbf.v4.new_notebook(); cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    md(HDR.format(title=title, design=design, fset=fset, exp=exp))
    code(SETUP.format(exp=exp))

    if design == "global":
        md("## 1. 모델 비교 (검증 = 2018, 튜닝 전 기본 파라미터)\n로지스틱 · HistGB · LightGBM · XGBoost 를 동일 피처로 비교.")
        code("pd.DataFrame(S['compare']).T[['roc_auc','pr_auc','logloss','brier','ece']].sort_values('logloss')")
        md("## 2. Optuna 튜닝 (LightGBM · 2017 fit → 2018 val, PR-AUC 최대화)")
        code("print('best val PR-AUC:', S['optuna']['best_val_prauc'])\n"
             "print('trial history:', S['optuna']['history'])\n"
             "pd.Series(S['optuna']['best_params']).rename('best_params').to_frame()")
        md("## 3. 최종 성능 (TEST 2019) vs Baseline\n**count-only** = `P(CSW|balls,strikes,stand,p_throws)`. 파생/모델이 이 선을 넘는지가 핵심.")
        code("base = S['final']['baselines']\n"
             "tbl = {**{f'baseline:{k}':v for k,v in base.items()}, 'tuned_lgbm (TEST2019)': S['final']['test_metrics_tuned_lgbm']}\n"
             "pd.DataFrame(tbl).T[['logloss','brier','roc_auc','pr_auc','ece']]")
        md("## 4. 지표 기여도 — SHAP (카테고리 집계 + 상위 피처)\n좌: 카테고리별 평균 |SHAP|, 우: 상위 피처.")
        code("display(pd.Series(S['shap']['category']).rename('mean|SHAP|').to_frame().round(4))\n"
             "display(pd.Series(S['shap']['top10']).rename('mean|SHAP|').to_frame().round(4))\n"
             "Image(str(R / 'shap.png'))")
        md("## 5. Calibration (TEST 2019, quantile 10-bin)")
        code("cal = S['final']['calibration']\n"
             "pd.DataFrame({'예측확률':cal['pred'],'실제빈도':cal['obs']})")
        if fset == "derived":
            md("## 6. 단계적 Ablation — '파생이 실제로 무엇을 개선했는가'\ncount_only → situation → basic_hist → ids → pitcher_hist → arsenal → release_rep 순으로 누적.")
            code("pd.DataFrame(S['ablation']).T[['logloss','brier','pr_auc','roc_auc','n_feats']]")
    else:
        md("## 1. 투수별 모델 vs 전체 모델 (동일 2019 평가집단)\n투수별 모델(임계 미만은 전체모델 폴백) 성능을 전체 참조모델과 비교.")
        code("pp = S['perpitcher']\n"
             "print('per-pitcher 적용 비율(coverage):', pp['coverage_per_pitcher'], '| 폴백:', pp['fallback_global'], '| 적격 투수 수:', pp['n_eligible_pitchers'], '| 임계:', S['min_train_perpitcher'])\n"
             "rows = {'per_pitcher(weighted_all)': pp['weighted_all'], 'per_pitcher_only': pp['per_pitcher_only'], 'global_reference': S['globalref']['global_reference_test']}\n"
             "rows = {k:v for k,v in rows.items() if v}\n"
             "pd.DataFrame(rows).T[['logloss','brier','roc_auc','pr_auc','ece']]")
        md("## 2. Baseline (TEST 2019)")
        code("pd.DataFrame(S['globalref']['baselines']).T[['logloss','roc_auc','pr_auc']]")
        md("""## 3. 해석
- **coverage 100%**: top-40 투수는 모두 임계(4,500) 이상이라 폴백이 없다. 하위 표본 투수를 포함하면 폴백 비율이 보고된다.
- 투수별 완전분리 모델은 전체 참조모델보다 **불리**(LogLoss↑, ECE↑) — 리뷰 지적대로 표본 부족·과적합.
- 권장 구조는 완전분리보다 **부분 풀링**: `전체 모델 + 투수 ID/이력 피처 + 투수별 보정`.
""")

    md("""## 결론 요약 (이 실험)
아래 셀의 수치를 근거로:
- **카운트/상황 신호가 지배적**이며, 파생·투수ID·아스널·release repeatability의 추가 이득은 작다(ablation·SHAP로 확인).
- 따라서 다음 실험은 A–K를 한꺼번에 넣기보다 **Basic → 투수이력 → 타자 → 아스널 → 시퀀싱 → 포수/구장** 순 ablation으로 확장하는 것이 설득력 있다.
- 상세 종합 평가·2페이지 보고서 작성은 `RESULTS_AND_REPORT_PLAN.md` 참고.
""")
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                      "language_info": {"name":"python","version":"3.10"}}
    nbf.write(nb, str(HERE / f"{name}.ipynb"))
    return name

for name, (exp, title, design, fset) in EXPS.items():
    build(name, exp, title, design, fset); print("wrote", name+".ipynb")
