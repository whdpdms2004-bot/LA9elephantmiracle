# 0723 — CSW 투구 전 예측 실험 (4개 노트북)

투구 전 정보만으로 현재 투구의 **CSW(콜드스트라이크+헛스윙)** 를 예측. train 2017–18 / test 2019, 주심 미사용.

| 노트북 | 모델 | 피처 |
|---|---|---|
| `01_global_basic.ipynb` | 전체 한 모델(투수=ID 인코딩) | 기본 |
| `02_global_derived.ipynb` | 전체 한 모델 | 기본+파생(4-창·아스널·release repeatability) |
| `03_perpitcher_basic.ipynb` | 투수별 모델(+폴백) | 기본 |
| `04_perpitcher_derived.ipynb` | 투수별 모델(+폴백) | 파생 |

각 노트북: 모델 비교 → Optuna 튜닝 → TEST vs Baseline → **SHAP 기여도(카테고리+상위피처)** → (파생) 단계적 ablation. 무거운 계산은 `run_experiments.py`가 사전 수행하고 노트북은 `results/`를 로드해 렌더한다.

- 파이프라인: `src/csw_pipeline.py` · 피처캐시: `prep_features.py` · 실험: `run_experiments.py` · 노트북생성: `build_notebooks.py`
- 종합 평가 + 2페이지 보고서 계획: **`RESULTS_AND_REPORT_PLAN.md`**
- 재현: `python prep_features.py 40 && python run_experiments.py <exp> && python build_notebooks.py`

> 현 수치는 상위 40투수 데모(속도용). 전체 재실행 방법은 계획 문서 참고.
