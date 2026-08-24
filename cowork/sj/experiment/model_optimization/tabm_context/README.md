# TabM 시간 검증 실험

## 검증 규약

- 모델 선택: `Val2022`, `Val2023`
- 최종 게이트: 설정을 완전히 고정한 뒤 `Val2024`를 한 번만 평가
- 각 fold의 전처리 통계와 범주 사전은 `season < fold`에서만 적합
- Val2024 게이트에는 TrackMan을 사용하지 않음
- 최종 2025 모델에서만 2024까지 학습하고, TrackMan은 투수-시즌 500구 이상 및 strict-as-of 조건으로 결합

## 피처 단계

| 단계 | 입력 | 목적 |
|---|---|---|
| T0 | 기존 BASE43 | XGB와 다른 오차를 만드는 순수 TabM 기준선 |
| T1 | T0 + count/손잡이/최근 추세/신뢰도 상호작용 | 상황 반응을 부드럽게 학습 |
| T2 | T1 + 투수·타자·팀 ID | 첫 선형층을 ID 임베딩처럼 사용 |
| T3 | T0 + 과거 시즌만으로 만든 residual/component 피처 | pitcher×hand×count 및 middle/reverse/outside 보정 |
| T3R | T3에서 절대 rate를 빼고 residual·표본수·변동성만 유지 | 시즌 평균 이동에 강한 보조 모델 |

T3는 `pitcher_profile_cutoff_S`만 사용하며 cutoff `S`의 행에는 `season < S`만
집계되어 있다. TrackMan 열은 전부 제외한다. 실패 유형은 `middle`, `reverse`,
`outside`를 독립 이진값으로 보존하며, middle/reverse 교집합을 제거하지 않는다.

## 실행 예시

```powershell
python experiment/model_optimization/tabm_context/run_tabm_temporal.py `
  --name t0_default `
  --feature-set t0 `
  --folds 2022 2023
```

R 전용 모델:

```powershell
python experiment/model_optimization/tabm_context/run_tabm_temporal.py `
  --name r_t1 `
  --feature-set t1 `
  --game-type R `
  --folds 2022 2023
```

F는 2023년에 분포 단절이 있으므로 최종 게이트용 전문가는 `--game-type F
--min-train-season 2023`처럼 post-break 데이터만 사용한다. 이 설정은 R/T0~T2 구조를
먼저 고정한 후 Val2024에서 한 번만 평가한다.

스모크 테스트:

```powershell
python experiment/model_optimization/tabm_context/run_tabm_temporal.py `
  --name smoke_t0 `
  --feature-set t0 `
  --folds 2022 `
  --max-train-rows 50000 `
  --max-valid-rows 20000 `
  --epochs 2 `
  --max-steps-per-epoch 20
```

각 실행 폴더에는 fold별 `model.pt`, `oof.parquet`, `summary.json`과 전체
`oof_all.parquet`, `run_summary.json`이 저장된다.

연속형 수치 임베딩 실험은 `--num-embedding linear_relu
--num-embedding-dim 8`로 실행한다. BASE43 원본과 같은 fold 전처리를 사용하므로
비교 시 모델 구조 외 조건은 동일하다.
