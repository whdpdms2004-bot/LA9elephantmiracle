# 제출 로그 — submit_040

작성일: 2026-08-20  
상태: **Public 채점 완료 — 976.9464685456점**

> 이 제출은 현재 **3WAY 최고**다. `sj` 전체 최고는
> `submit_034` **983.5821977654**, 다음은 `submit_035` **981.8287622303**이다.

## 1. 제출 후보

| 항목 | 값 |
|---|---|
| ZIP | `cowork/sj/submit/2026-08-20/submit_040.zip` |
| SHA-256 | `627133e23385514b10d5975ffd4d59cc415fa1141ea82464bc6cf104a3db476e` |
| 압축 크기 / 파일 수 | 7,167,466 bytes / 22개 |
| teacher | submit_037의 MIDDLE, REVERSE, OUTSIDE, MIDDLE∩REVERSE CatBoost 4개 |
| 추가 계층 | 사건별 약한 logit calibration 후 확률 합집합 계산 |
| requirements | `catboost==1.2.8` |
| Public Score | **976.9464685456** |
| submit_037 대비 | **+18.6901238313** |

`M`, `R`, `O` 중 하나라도 발생하면 제구 실패다. `O`는 `M`, `R`과 배타적이고
`M`, `R`은 동시에 발생할 수 있으므로 최종 확률은 다음 포함-배제식으로만 계산한다.

```text
p_failure = p_middle + p_reverse + p_outside - p_middle_and_reverse
p_success = clip(1 - p_failure, 0, 1)
```

네 사건확률은 각각 학습 OOF에서 구한 고정 logit 계수로 25%만 보정한다. 보정된
`p_mr`은 행별 Fréchet 범위 `max(0,p_m+p_r-1) <= p_mr <= min(p_m,p_r)`로 제한한다.
평가 행끼리 결합하거나 평가 배치의 평균·분포를 사용하지 않는다.

## 2. 학습 검증 결과와 성능 한계

보정기는 2023 forward-season OOF에서 적합하고 2024 OOF raw Brier로 강도와 L2를
선택했다. 동일 recipe를 2023+2024 OOF 498,378행에 재적합해 2025용 고정 자산으로
저장했다. 리더보드와 test 데이터는 적합·선택에 사용하지 않았다.

| Val2024 | Brier | raw BSS | prediction mean | target mean |
|---|---:|---:|---:|---:|
| submit_037 항등식 | 0.247736119 | 828.590 | 0.491463 | 0.486071 |
| 사건 보정 + 합집합 | **0.247701567** | **842.421** | **0.487497** | 0.486071 |
| 차이 | **-0.000034552** | **+13.831** |  |  |

| 구분 | 기준 BSS | 후보 BSS |
|---|---:|---:|
| R | 848.395 | **860.737** |
| F | 350.160 | **375.256** |

월별 Brier(후보 / 기준): 3월 0.24715909 / 0.24717862, 4월 0.24825082 /
0.24822181, 5월 0.24713830 / 0.24720268, 6월 0.24726026 / 0.24730105,
7월 0.24708975 / 0.24718263, 8월 0.24813058 / 0.24814903,
9월 0.24848054 / 0.24852988, 10월 0.25030403 / 0.25006696다. 8개월 중 6개월이
개선됐고 4월·10월은 악화됐다.

### Public 결과

- submit_037: **958.2563447143**
- submit_040: **976.9464685456**
- 절대 향상: **+18.6901238313**
- 상대 점수 향상: 약 **1.9504%**
- 서버 추론시간은 전달받지 않아 기록하지 않는다.
- 이 결과는 사후 기록으로만 사용하며 calibration 계수나 향후 offset을 역산하는 데
  사용하지 않는다.

### 반드시 남기는 한계

기존 screen OOF teacher는 검증 fold를 `eval_set`으로 사용하고
`use_best_model=True`로 최적 반복 수를 골랐다. 최종 teacher는 고정 900회라서
분포가 정확히 같지 않다. 이는 2025 평가 데이터나 외부 데이터를 사용한 규정 위반은
아니지만, 위 개선폭을 낙관적으로 만들 수 있는 **성능 검증 위험**이다. 따라서
submit_040은 초기 실험 후보이고, 고정 900회 honest forward OOF 재검증 전에는
submit_037보다 우월하다고 확정하지 않는다. honest OOF 생성은 GPU를 다른 작업이
사용 중이라 중단했으며 `middle_2023.npy`까지만 안전하게 캐시했다.

## 3. 고정 보정값 출처

| 사건 | intercept | logit slope |
|---|---:|---:|
| middle | 0.4400578254 | 1.2127473881 |
| reverse | 0.0070976800 | 1.0159168717 |
| outside | 0.0952006098 | 1.0559010576 |
| middle∩reverse | -0.0245499910 | 1.0098662900 |

- 적용 강도: 0.25, L2: 0.0001.
- 출처: 2019~2024 공식 학습 데이터에서 만든 2023·2024 forward-season OOF와
  복원 사건 라벨.
- `test.csv`의 값·분포·평균·순위와 리더보드 점수를 사용하지 않았다.
- 보정값은 모든 행에 같은 함수로 적용되며 각 행의 네 teacher 확률만 입력한다.

## 4. season_logit_offset 출처 명시

본 제출의 `season_logit_offset = 0.0`은 **학습 데이터(2019~2024 시즌)만을 이용해
사전 결정된 상수**이며, 모든 평가 행에 동일하게 적용된다.

- 산출 근거: 최종 성공확률에 별도 season logit offset을 적용하지 않고 사건별
  확률 보정 뒤 합집합의 여집합을 계산한다. 계산 코드는
  `cowork/sj/three_way/meta_fusion/src/run_initial_fusion.py`.
- **리더보드 점수를 참조하거나 역산하여 조정한 값이 아니다.**
- 평가 데이터(test.csv)의 값, 분포, 평균, 순위를 일절 사용하지 않았다.
- 따라서 `predict(단독 행) == predict(전체 test)[i]`를 만족한다.

teacher의 2025 성분 prior는 submit_037과 동일하며 2019~2024 성분별 시즌율의 선형
외삽값이다: middle 0.18654480, reverse 0.26131076, outside 0.11614189,
middle∩reverse 0.03437079.

## 5. B1 6단계 점검

### 1단계 — RULES.md 전문 재독

`cowork/RULES.md`의 절대 금지, 추론 독립성, 데이터·모델 제한, ZIP 규격과 프로젝트
리스크를 다시 대조했다. 외부 데이터·외부 API·2025 TrackMan·test 분포 보정은 없다.

### 2단계 — 행 독립성 기계 검증

- 실제 ZIP을 임시 서버형 경로에 풀어 전체 5행과 각 행 단독 추론 비교.
- 최대 절대차 **0.0**, 통과.
- 출력 열 순서, 행 수, `row_id` 순서, finite, `[0,1]` 범위 통과.

### 3단계 — 금지 패턴 전수 스캔

| 검사 | 결과와 판정 |
|---|---|
| 행 간 `groupby/rolling/cumsum/expanding/shift/rank/transform` | 0건 |
| 평가 배치 통계·분포 보정 | 0건. `nanmean(..., axis=1)`은 한 행의 고정 TrackMan 열만 요약 |
| 네트워크·다운로드 | 0건 |
| fit/train/backward/optimizer | 0건, 추론 전용 |
| 절대 경로 | 실제 경로 0건. `base_runtime.py:70`의 `/ (count + strength)`는 나눗셈 오탐 |

### 4단계 — 패키지·환경 검증

- ZIP 최상위 `model/`, `script.py`, `requirements.txt`만 존재.
- CRC 통과, 총 22개 파일. Python 캐시(`__pycache__`, `*.pyc`) 제외 확인.
- 오프라인 smoke **1.937초**, 통과.
- 245,789행 benchmark **10.938초 / 600초**, 통과.
- benchmark prediction mean 0.44954007은 5행 반복 시간 측정값이며 보정에 사용하지 않음.

### 5단계 — 자산과 TrackMan 시점 게이트

- middle model SHA-256 `FE1039B55E84F3CA9D701C22AA3F019A70F58731F6FAAA8F461F716C1A26E703`
- reverse model SHA-256 `676AE248805F4AFFD7288D6149A5D8FF0BA9A510CFB4D53AEA27DDA2C9D11436`
- outside model SHA-256 `48DFA637EF41531BD807324F8861F807F0305695CCAC7162CCD18232A129B403`
- MR model SHA-256 `E174ECD2D28D04A519252A003B95FBE5982F67B2A76039052B7FDAF38C85B649`
- TrackMan lookup SHA-256 `8EC28A492E58C567CDCE7580A66CDBAE5CB958D0466568644E4124A59464E36E`
- event calibration SHA-256 `39F2DAE34BAFDB1FA43D210EE54DBEBBCC90A8A935BDF1087ED92D0384E8B5EE`

| cutoff | 최대 evidence | 최대 TrackMan | 투수 수 | 단독/배치 차 | 상태 |
|---:|---:|---:|---:|---:|---|
| 2023 | 2022 | 2022 | 269 | 0.0 | pass |
| 2024 | 2023 | 2023 | 295 | 0.0 | pass |
| 2025 | 2024 | 2024 | 336 | 0.0 | pass |

2025 TrackMan은 사용하지 않는다.

### 6단계 — 최종 체크리스트

- [x] 행 독립성, 금지 패턴, 오프라인·시간 제한 검사 통과
- [x] test 분포·평균·순위 및 리더보드 기반 조정 없음
- [x] 외부 데이터·API·네트워크 및 추론 중 학습 없음
- [x] TrackMan 시점 게이트 통과
- [x] ZIP 구조·CRC·경로·모델 자산·출력 형식 통과
- [x] Brier, BSS, R/F, 월별 Brier, prediction mean 기록
- [x] 2026-08-20 실제 리더보드 제출 횟수 1회
- [ ] `cowork/task.jsonl` append는 공용 파일이므로 팀 PR에서 반영

## 6. 재현 경로

1. 초기 결합 비교: `cowork/sj/three_way/meta_fusion/src/run_initial_fusion.py`
2. 검증 리포트: `cowork/sj/three_way/meta_fusion/src/report_initial_validation.py`
3. 고정 900회 honest OOF: `cowork/sj/three_way/meta_fusion/src/build_honest_oof.py`
4. ZIP 생성: `cowork/sj/three_way/meta_fusion/src/build_initial_package.py`
5. 패키지 검사: `cowork/sj/three_way/src/verify_package.py`
