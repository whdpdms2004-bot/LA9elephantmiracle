# 제출 로그 — submit_036

작성일: 2026-08-18  
상태: **로컬 B1 검증 완료, Public 미제출**

## 1. 제출 후보

| 항목 | 값 |
|---|---|
| ZIP | `cowork/sj/submit/2026-08-18/submit_036.zip` |
| SHA-256 | `3f1bc1b66eb321549a5e6099eef61a6f01151c505ceddc65d5f3bcc68b09537b` |
| 압축 / 해제 크기 | 60,669,338 / 194,199,913 bytes |
| 모델 | 단독 CatBoostClassifier, strict forward-OOF F1 피처 |
| 학습 범위 | train 2019~2024 전체, final 2025 lookup은 2019~2024만 사용 |
| 피처 | 272개 = enhanced 209 + 행 파생 D0 18 + 성분 hierarchy 45 |
| 반복 / seed / half-life | 2595 / 20262843 / 1.6746021883599578 |
| 학습 장치 | NVIDIA RTX 4080 SUPER GPU |
| requirements | `catboost==1.2.8` |

`C1` 정적 학습 lookup 후보는 학습행 자신의 시즌 Target이 섞이는 구조가 확인되어
최종 후보에서 제외했다. `F1`은 2019년 학습행을 0 fallback으로 두고, 2020~2024년
각 학습행에는 해당 행 시즌보다 이전 시즌의 Target lookup만 연결한다. 2025 추론에는
2019~2024 전체 학습 데이터로 미리 만든 고정 lookup을 연결한다.

## 2. 순방향 검증 결과

보정은 각 fold 이전 시즌의 Target 시즌 평균만으로 계산한 `all_linear_d025`이다.
검증 예측 평균·검증 Target 평균·리더보드 값은 보정값 산출에 쓰지 않았다.

| fold | Brier | 전체 BSS(raw) | R BSS | F BSS | prediction mean | target mean | logit offset |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 0.243429700 | 2301.262 | 610.574 | -557.767 | 0.523084 | 0.528920 | -0.021500491 |
| 2023 | 0.249839938 | 64.024 | 174.657 | -1213.328 | 0.499921 | 0.499957 | -0.016112837 |
| 2024 | 0.247816630 | 796.734 | 814.547 | 337.209 | 0.493761 | 0.486105 | -0.008242908 |
| 2022~2024 연결 OOF | 0.247027794 | 1179.567 | 551.182 | 5064.825 | 0.505508 | 0.504855 | fold별 적용 |

- fold BSS 단순 평균: **1054.006700**
- 연결 OOF BSS: **1179.566787**
- 위 수치는 오프라인 순방향 검증이며 Public 1000 달성을 뜻하지 않는다.

### 월별 Brier

| fold | 월별 Brier |
|---:|---|
| 2022 | 4: 0.24149759, 5: 0.24194627, 6: 0.24382348, 7: 0.24407457, 8: 0.24581316, 9: 0.24373770, 10: 0.24433775 |
| 2023 | 4: 0.24950054, 5: 0.24996927, 6: 0.24982234, 7: 0.24990576, 8: 0.24999488, 9: 0.24996141, 10: 0.24968297 |
| 2024 | 3: 0.24734876, 4: 0.24828387, 5: 0.24717847, 6: 0.24734340, 7: 0.24745047, 8: 0.24821514, 9: 0.24859444, 10: 0.25055610 |
| 연결 OOF | 3: 0.24734876, 4: 0.24644239, 5: 0.24621134, 6: 0.24706444, 7: 0.24700132, 8: 0.24804882, 9: 0.24724428, 10: 0.24799945 |

상세 산출물: `cowork/sj/feature_campaign_1000/outputs/combined/f1_cat_validation_report.json`  
재현 코드: `cowork/sj/feature_campaign_1000/report_f1_validation.py`

## 3. season_logit_offset 출처 명시

본 제출의 `season_logit_offset = -0.011524988210223915`은 **학습 데이터(2019~2024 시즌)만을 이용해 사전 결정된 상수**이며,
모든 평가 행에 동일하게 적용된다.

- 산출 근거: 2019~2024 시즌별 제구 성공률 추세의 외삽. 계산 코드는 `cowork/sj/feature_campaign_1000/evaluate_train_only_season_offsets.py` 및 `cowork/sj/feature_campaign_1000/train_final_f1_cat.py`.
- **리더보드 점수를 참조하거나 역산하여 조정한 값이 아니다.**
- 평가 데이터(test.csv)의 값, 분포, 평균, 순위를 일절 사용하지 않았다.
- 따라서 `predict(단독 행) == predict(전체 test)[i]`를 만족한다.

최종 값은 2019~2024 전체 시즌 평균의 선형 logit 추세가 2025년으로 외삽한 변화량의
25%만 적용한 값이다. Public 결과를 본 뒤 이 값을 재조정하지 않는다.

## 4. B1 6단계 점검

### 1단계 — 규정 전문 재독

- `cowork/RULES.md` 347행 전문 재독 완료.
- 절대 금지 10개, 행 독립성, 데이터·모델 제한, ZIP 규격, 프로젝트 리스크 §10 재확인.

### 2단계 — 행 독립성 기계 검증

압축본을 `outputs/package_smoke_036/`에 풀고 제공된 5행을 전체 및 한 행씩 예측했다.

- 최대 절대차: **0.0**
- 출력 열: `row_id`, `control_success` 순서 통과
- 행 수·row_id 순서 통과
- finite 및 `[0, 1]` 범위 통과

### 3단계 — 금지 패턴 전수 스캔

검사 대상은 ZIP의 `script.py`, `model/base_runtime.py`,
`model/component_runtime.py` 전부다.

| 검사 | 결과와 판정 |
|---|---|
| `groupby/rolling/cumsum/expanding/shift/rank/transform` | 히트 0. test 행 간 집계 없음 |
| `mean/std/median/quantile/distribution_match` | 6개 히트. 모두 같은 행의 prev1/3/5 열을 `axis=1` 또는 `axis=0`으로 요약하는 행 단위 피처. test 행 간 통계가 아님 |
| `requests/urllib/httpx/socket/from_pretrained/hf_hub/download/api_key` | 히트 0 |
| `fit/train/partial_fit/backward/optimizer` | 히트 0. 추론 전용 |
| 절대 경로 `/home`, `/workspace`, `/app`, Windows drive | 히트 0 |

모든 lookup은 ZIP의 `model/`에 미리 저장된 2019~2024 학습 기반 자산이며
`pitcher_id` 등으로 단건 재색인하거나 many-to-one 조인한다. 평가 배치에서 lookup을
새로 만들지 않는다.

### 4단계 — 패키지·환경 검증

- ZIP 최상위: `model/`, `script.py`, `requirements.txt`만 존재.
- ZIP CRC: `testzip() == None`.
- 파일명에 공백·한글 없음.
- 모델 자산 비어 있지 않음.
- 압축본 서버형 경로 smoke: 통과.
- 245,789행 반복 입력 benchmark: **5.869초 / 600초**, 출력 계약 통과.
- 소켓 `connect/create_connection/connect_ex`를 강제 차단한 오프라인 smoke: 통과.
- 로컬 환경: Windows 11, Python 3.12.7, NumPy 1.26.4, pandas 2.2.2,
  CatBoost 1.2.8. 서버용 CatBoost 버전은 requirements에서 고정.

245,789행 benchmark의 prediction mean `0.458762`는 5행 샘플을 반복한 실행시간
검사용 값일 뿐이며 실제 평가 분포 보정에 사용하지 않는다.

### 5단계 — 근거·자산 기록

모델 자산:

- `model/base_runtime.py`
- `model/component_runtime.py`
- `model/f1_bat_platoon_2025.csv`
- `model/f1_catboost.cbm`
- `model/f1_count_platoon_2025.csv`
- `model/f1_inning_platoon_2025.csv`
- `model/f1_metadata.json`
- `model/f1_platoon_2025.csv`
- `model/trackman500_lookup_2025.csv`

TrackMan 시점 게이트 재검증:

| cutoff | 최대 evidence season | 최대 TrackMan season | 투수 수 | 상태 |
|---:|---:|---:|---:|---|
| 2023 | 2022 | 2022 | 269 | pass |
| 2024 | 2023 | 2023 | 295 | pass |
| 2025 | 2024 | 2024 | 336 | pass |

2025 TrackMan 데이터는 사용하지 않는다. 검증 코드는
`cowork/sj/feature_campaign_1000/verify_feature_artifacts.py`다.

### 6단계 — 최종 체크리스트

규정:

- [x] test 행 간 집계·rolling·lag 없음
- [x] test 예측 분포·평균·순위를 이용한 보정 없음
- [x] 상수 보정값 출처가 학습 데이터임을 기록
- [x] 리더보드 기반 상수 재조정 없음
- [x] 외부 데이터·외부 API·네트워크 호출 없음
- [x] `script.py` 및 런타임 모듈에 학습 코드 없음
- [x] TrackMan 시점 게이트 통과

패키지와 출력:

- [x] ZIP 루트 3항목만 존재
- [x] 새 날짜·새 번호, 기존 제출물 미덮어쓰기
- [x] `Path(__file__).resolve().parent` 기준 경로
- [x] 오프라인 smoke와 10분 제한 통과
- [x] `output/submission.csv` 열·행·범위·finite 검증 통과

운영:

- [x] Brier / 전체 BSS / R·F BSS / 월별 Brier / prediction mean 기록
- [x] 2026-08-18 실제 리더보드 제출 횟수: **0회** (본 로그 작성 시점)
- [ ] `cowork/task.jsonl` 실험 항목 append — 공용 파일이므로 팀 PR로만 반영
- [ ] Public 제출 후 점수 기록
- [ ] Public **1000 이상** 실제 확인

## 5. 재현 경로

1. raw data 전체 재현 wrapper: `cowork/sj/feature_campaign_1000/reproduce_final_f1.py`
2. 최종 모델 학습: `cowork/sj/feature_campaign_1000/train_final_f1_cat.py`
3. 검증 리포트: `cowork/sj/feature_campaign_1000/report_f1_validation.py`
4. 행 독립·대규모 검증: `cowork/sj/feature_campaign_1000/verify_final_f1.py`
5. ZIP 생성: `cowork/sj/feature_campaign_1000/build_f1_package.py`

wrapper는 raw train/TrackMan에서 strict-as-of 자산과 최종 모델·lookup을 순서대로
재생성한다. 학습 환경은 `requirements_training.txt`에 고정했다. 최종 제출 여부는
Public 점수와 운영팀 서버 실행 결과로 확정한다.
