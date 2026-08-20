# 제출 로그 — submit_037

작성일: 2026-08-19  
상태: **Public 채점 완료 — 958.2563447143점**

## 1. 제출 후보

| 항목 | 값 |
|---|---|
| ZIP | `cowork/sj/submit/2026-08-19/submit_037.zip` |
| SHA-256 | `baab5faf99daa12a0deb1f3d0cfbf8a675937ac39aef1bbc2123093f2a2a8984` |
| 압축 / 해제 크기 | 7,165,074 / 16,687,714 bytes |
| 모델 | 3WAY MIDDLE + REVERSE + OUTSIDE + MIDDLE∩REVERSE CatBoost 4개 |
| 결합 | `clip(1 - (middle + reverse - mr + outside))` 포함–배제 항등식 |
| 학습 범위 | 2019~2024 전체, 성분 라벨 복원 가능 1,473,508행 |
| 반복 / seed | 모델별 900 / 20262844 |
| 학습 장치 | NVIDIA RTX 4080 SUPER GPU, 네 모델 순차 학습 |
| requirements | `catboost==1.2.8` |
| Public Score | **958.2563447143** |
| 평가 서버 추론시간 | **8초** |

새 모델 탐색 없이 `cowork/sj/three_way`에서 이미 확정한 fold 2024 구성만 전체
학습했다. 타깃별 피처는 middle 186, reverse 297, outside 230, mr 186개다.

## 2. 검증 결과와 한계

검증은 성분 라벨을 복원할 수 있는 행에 대해 계산했다. `bss_centered`는 실험 순위
진단에만 썼으며 제출 예측에는 적용하지 않는다.

| fold | Brier | 전체 BSS raw | R BSS | F BSS | prediction mean | target mean |
|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 0.250766332 | -306.533 | 62.044 | -3809.562 | 0.513006 | 0.499982 |
| 2024 | 0.247736119 | **828.590** | **848.395** | **350.160** | 0.491463 | 0.486071 |

2023 raw BSS가 음수이므로 연도 강건성이 약한 후보임을 명시한다. 본 ZIP은
`three_way`의 현재 확정안을 그대로 제출 가능한 형태로 만든 것이며, 강건성이 새로
확인됐다는 의미는 아니다.

### Public 결과

- Public Score: **958.2563447143**
- 평가 서버 추론시간: **8초**
- 로컬 245,789행 benchmark 9.281초와 비슷한 수준으로 재현됐다.
- 3WAY 단독 포함–배제 모델이 서버에서 정상 실행됐으며, 10분 제한 대비 충분한
  시간 여유를 확인했다.

### Public 채점 후 규정 재점검

Public 결과 확인 후 `submit_037.zip` 원본을 다시 풀어 `cowork/RULES.md` 전문과
실행 코드를 재대조했다.

- ZIP CRC 및 최상위 3항목 재확인: 통과
- 전체 5행과 각 행 단독 추론 최대 절대차: **0.0**
- 평가 행 간 집계, 평가 배치 통계, 네트워크, 추론 중 학습 패턴: **모두 0건**
- 245,789행 재측정: **11.672초 / 600초**, 통과
- `base_runtime.py:70`의 `/` 1건은 나눗셈 수식이며 절대 경로가 아님
- 실행 중 출력된 DataFrame fragmentation 및 빈 TrackMan 행의 `nanmean` 경고는
  성능/결측 처리 경고로, 다른 평가 행 사용이나 분포 보정과 무관함

재점검 결론: **현재 확인 가능한 규정 위반 없음.** 2025 TrackMan, 외부 데이터,
리더보드 기반 offset 조정은 사용하지 않았다.

### 월별 Brier

- 2023 — 4: 0.25102923, 5: 0.25098220, 6: 0.25062845, 7: 0.25078486,
  8: 0.25082784, 9: 0.25078889, 10: 0.24978499
- 2024 — 3: 0.24717862, 4: 0.24822181, 5: 0.24720268, 6: 0.24730105,
  7: 0.24718263, 8: 0.24814903, 9: 0.24852988, 10: 0.25006696

상세 산출물: `cowork/sj/three_way/outputs/final_validation_report.json`

## 3. season_logit_offset 출처 명시

본 제출의 `season_logit_offset = 0.0`은 **학습 데이터(2019~2024 시즌)만을 이용해
확정한 설계 상수**이며, 모든 평가 행에 동일하게 적용된다.

- 산출 근거: 최종 성공확률에 별도 logit offset을 적용하지 않는 3WAY 항등식 설계.
  계산·학습 코드는 `cowork/sj/three_way/src/train_final.py`.
- **리더보드 점수를 참조하거나 역산하여 조정한 값이 아니다.**
- 평가 데이터(test.csv)의 값, 분포, 평균, 순위를 일절 사용하지 않았다.
- 따라서 `predict(단독 행) == predict(전체 test)[i]`를 만족한다.

최종 offset 대신 각 성분 CatBoost의 Pool baseline에 아래 2019~2024 학습 Target
시즌 추세 외삽값을 사용한다. 모든 값은 평가 데이터와 무관한 고정 상수다.

| 성분 | 2025 prior | baseline logit의 출처 |
|---|---:|---|
| middle | 0.18654480 | 2019~2024 `y_middle` 시즌율 선형 외삽 |
| reverse | 0.26131076 | 2019~2024 `y_reverse` 시즌율 선형 외삽 |
| outside | 0.11614189 | 2019~2024 `y_outside` 시즌율 선형 외삽 |
| mr | 0.03437079 | 2019~2024 `y_middle ∩ y_reverse` 시즌율 선형 외삽 |

성분 라벨은 `train.csv`의 공식 `asof_*` 누적률을 같은 투수의 다음 **학습 행**과
차분해 복원하며 학습 Target으로만 사용한다. 이 값과 차분 연산은 추론 피처나
`script.py`에 존재하지 않는다. 실제 추론은 현재 행의 공식 사전 피처와 2019~2024
고정 lookup만 사용한다.

## 4. B1 6단계 점검

### 1단계 — RULES.md 전문 재독

- `cowork/RULES.md` 347행 전문 재독 완료.
- 절대 금지 10개, 행 독립성, 데이터·모델 제한, ZIP 규격, 프로젝트 리스크 §10을
  제출 생성 직전에 다시 확인했다.

### 2단계 — 행 독립성 기계 검증

ZIP을 새 임시 폴더에 풀고 제공된 5행을 전체 및 한 행씩 각각 예측했다.

- 최대 절대차: **0.0**
- 출력 열 `row_id`, `control_success` 순서 통과
- 행 수와 `row_id` 순서 통과
- finite 및 `[0, 1]` 범위 통과
- 5행 샘플 예측: 0.42717666, 0.37966055, 0.45212331, 0.51536965, 0.49890294

### 3단계 — 금지 패턴 전수 스캔

검사 대상은 ZIP의 `script.py`, `model/base_runtime.py`,
`model/component_runtime.py`, `model/three_way_runtime.py` 전부다.

| 검사 | 결과와 판정 |
|---|---|
| `groupby/rolling/cumsum/expanding/shift/rank/transform` | 히트 0. 평가 행 간 집계 없음 |
| `mean/std/median/quantile/distribution_match` | B1 지정 정규식 히트 0. `np.nanmean(..., axis=1)`은 같은 행의 여러 고정 TrackMan 열을 요약하는 행 단위 피처이며 평가 행 간 통계가 아님 |
| `requests/urllib/httpx/socket/from_pretrained/hf_hub/download/api_key` | 히트 0 |
| `fit/train/partial_fit/backward/optimizer` | 히트 0. 추론 전용 |
| 절대 경로 | `base_runtime.py:70`의 줄 시작 `/ (count + strength)` 1건. 수식 줄바꿈의 나눗셈 연산자로 경로가 아니며, `/home`, `/workspace`, `/app`, Windows drive 히트는 0 |

ID 빈도, rate 중앙값, TrackMan robust scale은 2019~2024 학습 데이터로 미리
저장했다. 평가 시에는 각 행의 ID로 고정 테이블을 재색인할 뿐 평가 배치에서 빈도나
분포를 계산하지 않는다.

### 4단계 — 패키지·환경 검증

- ZIP 최상위: `model/`, `script.py`, `requirements.txt`만 존재.
- ZIP CRC: 통과, 20개 파일.
- 파일명에 공백·한글 없음, 기존 제출물 미덮어쓰기.
- 압축본 서버형 경로 5행 smoke: 통과.
- 소켓 연결을 강제 차단한 오프라인 smoke: **1.665초, 통과**.
- 245,789행 반복 입력 benchmark: **9.281초 / 600초, 통과**.
- benchmark prediction mean `0.45464644`는 5행 샘플 반복 시간 측정값일 뿐 실제
  평가 분포 보정에 사용하지 않는다.

### 5단계 — 근거·자산 기록

모델과 주요 고정 자산:

- `model/three_way_middle.cbm` — SHA-256 `FE1039B55E84F3CA9D701C22AA3F019A70F58731F6FAAA8F461F716C1A26E703`
- `model/three_way_reverse.cbm` — SHA-256 `676AE248805F4AFFD7288D6149A5D8FF0BA9A510CFB4D53AEA27DDA2C9D11436`
- `model/three_way_outside.cbm` — SHA-256 `48DFA637EF41531BD807324F8861F807F0305695CCAC7162CCD18232A129B403`
- `model/three_way_mr.cbm` — SHA-256 `E174ECD2D28D04A519252A003B95FBE5982F67B2A76039052B7FDAF38C85B649`
- `model/trackman500_lookup_2025.csv` — SHA-256 `8EC28A492E58C567CDCE7580A66CDBAE5CB958D0466568644E4124A59464E36E`
- 네 ID 빈도 lookup, 네 성분 hierarchy lookup, `three_way_metadata.json`, 추론 런타임 3개

TrackMan 시점 게이트를 이번 제출 생성 후 다시 실행했다.

| cutoff | 최대 evidence season | 최대 TrackMan season | 투수 수 | 단독/배치 최대차 | 상태 |
|---:|---:|---:|---:|---:|---|
| 2023 | 2022 | 2022 | 269 | 0.0 | pass |
| 2024 | 2023 | 2023 | 295 | 0.0 | pass |
| 2025 | 2024 | 2024 | 336 | 0.0 | pass |

2025년 TrackMan은 사용하지 않는다.

### 6단계 — 최종 체크리스트

규정:

- [x] test 행 간 집계·rolling·lag 없음
- [x] test 예측 분포·평균·순위를 이용한 보정 없음
- [x] 상수와 성분 prior의 출처가 학습 데이터임을 기록
- [x] 리더보드 기반 상수 재조정 없음
- [x] 외부 데이터·외부 API·네트워크 호출 없음
- [x] `script.py`와 런타임 모듈에 학습 코드 없음
- [x] TrackMan 시점 게이트 통과

패키지와 출력:

- [x] ZIP 루트 3항목만 존재
- [x] 새 날짜·새 번호, 기존 제출물 미덮어쓰기
- [x] `Path(__file__).resolve().parent` 기준 경로
- [x] 모델 폴더 비어 있지 않음
- [x] 오프라인 smoke와 10분 제한 통과
- [x] `output/submission.csv` 열·행·범위·finite 검증 통과

운영:

- [x] Brier / 전체 BSS / R·F BSS / 월별 Brier / prediction mean 기록
- [x] 2026-08-19 실제 리더보드 제출 횟수: **1회**
- [ ] `cowork/task.jsonl` 실험 항목 append — 공용 파일이므로 팀 PR로만 반영
- [x] Public 제출 후 점수와 서버 추론시간 기록

## 5. 재현 경로

1. 최종 네 모델 학습과 고정 자산 생성: `cowork/sj/three_way/src/train_final.py`
2. 검증 리포트: `cowork/sj/three_way/src/report_validation.py`
3. ZIP 생성: `cowork/sj/three_way/src/build_package.py`
4. 행 독립·오프라인·대규모 검증: `cowork/sj/three_way/src/verify_package.py`

학습 코드는 GPU 모델 네 개를 한 번에 하나씩 순차 실행하며, 2019~2024 외 시즌이
프레임에 있으면 즉시 중단한다.
