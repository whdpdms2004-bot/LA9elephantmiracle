# 고연산 모델링 계획

최종 갱신: 2026-08-12

## 1. 결론

추가 컴퓨팅은 한 XGBoost의 트리·leaves를 더 키우는 데 쓰지 않는다. 이미 32~128 leaves는 과적합했고 20~24 leaves가 유효 구간이었다. 앞으로는 다음 순서로 연산을 배분한다.

1. 시간축 nested Optuna 탐색
2. 성공·reverse·middle·outside 전용 전문가 모델
3. full-context 신경망과 투수·타자 연속 임베딩
4. 게임 단위 bagging과 서로 다른 모델군 확장
5. OOF 기반 강건 stacking·calibration
6. 2024년까지 전체 재학습 및 제출용 경량화

학습 단계에서는 최대한 많은 모델을 만들고, 제출 단계에서는 검증된 다양성만 남긴다.

## 2. 현재 자원과 여유

| 항목 | 현재 값 | 운영 원칙 |
|---|---:|---|
| GPU | RTX 4080 SUPER, VRAM 16GB | GPU 학습 작업은 한 번에 1개 |
| CPU | 논리 24코어 | LightGBM·전처리 작업을 GPU 학습과 병행 |
| RAM | 63GB | fold matrix 캐시 후 반복 인코딩 제거 |
| 디스크 여유 | 약 606GB | OOF는 보존, 전체 trial 모델은 보존하지 않음 |
| 현재 제출 크기 | 약 106.5MB | 10GB 제한 대비 충분한 여유 |
| 현재 245,789행 추론 | 약 18.5초 | 10분 제한 대비 큰 여유 |
| 현재 제출 모델 | CatBoost 3 + XGBoost 5 | 검증 후 20~40개 expert까지 확대 가능 |

최종 모델 수는 미리 고정하지 않고 실제 L4형 smoke benchmark를 기준으로 정한다. 목표 안전선은 추론 180초 이하, RAM 20GB 이하, 압축 해제 4GB 이하이다.

## 3. 검증 과적합 방지

연산량이 커질수록 하이퍼파라미터가 2024 검증 정답에 과적합할 가능성이 커진다. 따라서 탐색과 최종 확인을 분리한다.

### 3.1 탐색 단계

- Inner fold: Val2022, Val2023
- Optuna 목적함수: 최근 연도 가중 normalized Brier + fold 변동 penalty
- 권장식: `0.40 * NB2022 + 0.60 * NB2023 + 0.25 * std(NB2022, NB2023)`
- R 전용 모델은 R normalized Brier 악화 penalty를 별도 추가
- TrackMan은 각 검증 시즌보다 이전 시즌만 사용하고 투수-시즌 500구 이상 규칙 유지

### 3.2 최종 gate

- 각 모델군 상위 10~20개만 Val2024에서 한 번 평가
- Val2024 결과를 보고 같은 탐색 공간을 다시 미세조정하지 않음
- 채택 조건:
  - Val2024 normalized Brier 개선
  - Val2023·Val2024 R이 모두 악화되지 않음
  - 기존 앙상블과 잔차 상관이 낮거나 단독 성능이 명확히 높음

### 3.3 최종 학습

- 선택 완료 후 2019~2024 전체 학습
- 2025 추론 artifact에는 2024까지의 과거 정보만 포함
- test 전체 분포를 사용한 재보정·재군집 금지

## 4. 100시간 계산 예산

| 단계 | 시간 | 핵심 작업 | 산출물 |
|---|---:|---|---|
| A. 캐시·실행 기반 | 4시간 | fold별 float32 matrix, category mapping, 누수 audit, resume runner | 재사용 fold cache |
| B. Robust tree Optuna | 24시간 | XGBoost·CatBoost·LightGBM nested search | 모델군별 상위 OOF |
| C. 실패 유형 전문가 | 14시간 | success/reverse/middle/outside 개별 탐색 | 4-head expert bank |
| D. 신경망·임베딩 | 24시간 | DCNv2/MLP/two-tower multi-task | neural OOF·embedding |
| E. bagging·factorization | 14시간 | game bootstrap, time bootstrap, matchup factorization | diversity expert OOF |
| F. stacking·calibration | 10시간 | 강건 가중치, residual stack, beta calibration | frozen ensemble spec |
| G. 전체 재학습·제출 검증 | 10시간 | 2024까지 재학습, 245,789행 smoke, ZIP | 제출 후보 2개 |

시간은 성능 신호에 따라 이동한다. 어떤 단계든 중간 gate를 통과하지 못하면 남은 계산을 다음 단계로 넘긴다.

## 5. 단계 B: Robust tree Optuna

### 5.1 현재 부족한 부분

- 기존 robust 탐색: XGBoost 160 trials, CatBoost 80, LightGBM 60
- 성공 사전값을 넣은 최신 XGBoost 탐색은 Val2024 단일 fold 25 trials뿐이다.
- 따라서 최신 R 문맥·성공 사전값·failure component가 포함된 feature set의 다중 연도 탐색이 우선이다.

### 5.2 목표 trial

| 모델군 | 1차 목표 | 상위 재검증 | 역할 |
|---|---:|---:|---|
| XGBoost | 300 trials | 상위 20 × Val2024 | 주력, GPU |
| CatBoost | 180 trials | 상위 15 × Val2024 | 범주형·상호작용 다양성, GPU |
| LightGBM | 160 trials | 상위 15 × Val2024 | CPU 기반 다양성 |

단순히 범위를 넓히지 않고 기존 좋은 구간 주변과 새로운 feature subset을 함께 탐색한다.

- max leaves: 12~28 중심, 32 이상은 소수 규제 강한 probe만 허용
- half-life: 최근형 0.3~1.0과 안정형 1.5~4.0을 별도 study로 분리
- feature bag: base, success prior, failure component, R context, compact embedding
- objective: Brier 중심, AUC는 동률 후보의 diversity 판단에만 사용
- pruning: Val2022/초기 iteration이 명확히 나쁜 trial 조기 종료

## 6. 단계 C: 실패 유형 전문가

최종 성공은 세 실패 유형이 섞인 결과이므로 하나의 모델만 반복 탐색하지 않는다.

| 전문가 | 목표 | 우선 피처 | 최종 결합 |
|---|---|---|---|
| success | 전체 `control_success` | 기존 전체 상황·과거 이력 | 중심 확률 |
| reverse | 포수 요구 반대 방향 | 투·타 유형, hand, count, matchup | residual correction |
| middle | 가운데 몰림 | 투수 command, 구종군, count, 이닝 | R 최근 drift 보정 |
| outside | 존에서 크게 벗어남 | 제구 분산, 피로, 주자·카운트 | residual correction |

각 head마다 80~150개의 tree trial을 허용한다. 멀티태스크 neural head와 별개로 tree expert를 만들어 잔차 다양성을 확보한다.

## 7. 단계 D: 신경망과 임베딩

기존 신경망은 투수 이력 중심이라 단독 BSS가 낮았다. 다음 모델은 현재 투구 상황과 타자 정보까지 모두 넣는다.

### 7.1 우선 모델

1. DCNv2: 범주형 embedding + 연속형 tower + cross layer
2. residual MLP: 3~5개 block, width 256/512/768
3. pitcher-batter two-tower: 투수·타자 표현과 상황 tower를 분리 후 결합
4. FT-Transformer 소수 probe: 성능이 확인될 때만 확대

### 7.2 학습 설정

- main loss: Brier 또는 `0.7 Brier + 0.3 BCE`
- auxiliary: reverse/middle/outside BCE
- pure F1 loss는 제외, macro soft-F1은 최대 0.05 보조 규제로만 probe
- mixed precision, gradient accumulation, early stopping
- width/depth/dropout/embedding dimension을 24~36개 구성으로 탐색
- inner fold 상위 6개만 3-seed 전체 재학습

### 7.3 임베딩 사용

- TrackMan500 투수 임베딩은 continuous vector와 reliability를 함께 전달
- 500구 미만 투수는 개별 TrackMan embedding을 만들지 않고 rookie/cohort embedding 사용
- 타자 임베딩은 pitcher type에 대한 상성 residual로 사전학습
- hard cluster ID보다 continuous embedding, cluster distance, posterior probability를 우선

## 8. 단계 E: bagging과 상성 factorization

동일 설정의 seed만 늘리는 방식은 이미 이득이 작았다. 다음에는 데이터 자체를 다르게 보게 한다.

- game 단위 block bootstrap 5개
- 최근 시즌 가중치를 달리한 time bootstrap 4개
- feature column bag 4개
- R/F와 손 조합을 약하게 분리한 mixture-of-experts
- 투수×타자 reverse/middle residual 행렬의 shrinkage SVD/NMF/two-tower 비교
- factor dimension 4/8/16/32, shrinkage 50/100/300/500/1000
- 각 후보는 기존 prediction과 잔차 상관을 함께 기록

행 단위 bootstrap은 같은 경기의 인접 투구가 train과 bag에 과도하게 중복되므로 사용하지 않는다.

## 9. 단계 F: 고연산 앙상블

모든 후보를 단순 평균하지 않는다.

1. OOF prediction matrix 생성
2. 모델별 normalized Brier, R/F Brier, 연도별 mean gap, residual correlation 계산
3. greedy ensemble selection으로 중복 가중치 탐색
4. non-negative ridge와 simplex constrained blend 비교
5. fold별 최적 가중치 편차 penalty 적용
6. beta calibration, temperature scaling, R-only context shrinkage 비교

최종 선택 목적은 다음을 함께 본다.

`robust_score = weighted_mean(NB2022, NB2023, NB2024) + 0.35 * worst_fold_gap + 0.25 * R_instability`

Val2024 단독 최적 가중치는 참고만 하고 최종 고정 가중치로 바로 쓰지 않는다.

## 10. 병렬 운영

### 동시에 실행 가능한 조합

- GPU: XGBoost 또는 CatBoost 또는 PyTorch 중 1개
- CPU: LightGBM 1개, 8~12 threads
- 경량: OOF 분석·문서화 1개

### 금지 조합

- GPU 학습 2개 동시 실행
- 대형 pandas frame을 여러 process에서 각각 복사
- 서로 다른 job이 같은 SQLite Optuna DB에 과도하게 동시 write

fold matrix는 한 번 생성해 memory-map 또는 binary cache로 재사용한다. trial마다 CSV/Parquet를 다시 읽고 category encoding을 반복하지 않는다.

## 11. 저장·중단 복구 규칙

- 모든 study는 `load_if_exists=True`로 재개 가능하게 구성
- trial마다 params, fold metric, best iteration, elapsed time 기록
- OOF는 모든 최종 후보에 보존
- 모델 파일은 상위 20개와 diversity 후보만 보존
- 30분마다 campaign summary 갱신
- 중단 시 RUNNING trial을 감사하고 실제 process가 없을 때만 FAIL 처리

현재 `catboost_v1_full_2023_2024` DB에 오래된 RUNNING trial 1개가 있으므로 재사용 전에 상태를 감사한다.

## 12. 우선 실행 순서

1. 최신 insight/R context feature의 nested Optuna runner 작성
2. fold matrix cache와 leakage audit 생성
3. XGBoost 30-trial smoke로 시간·VRAM 측정
4. XGBoost 300, CatBoost 180, LightGBM 160 campaign 실행
5. 동시에 middle 전용 expert의 데이터·평가 구조 작성
6. tree 상위 OOF가 모이면 full-context DCNv2 시작
7. 전체 OOF가 모인 뒤에만 stacking과 최종 2025 재학습 진행

## 13. 기대값과 중단 기준

추가 연산의 목표는 단일 모델 BSS를 크게 올리는 것보다, 기존 `submit_016`과 잔차가 다른 expert를 찾아 앙상블 BSS를 누적 개선하는 것이다.

- 단일 후보: Val2024 BSS +2 이상 또는 잔차 상관 0.98 이하
- correction 후보: Val2023·Val2024 ΔBrier가 모두 음수
- 앙상블 후보: 강건 목적에서 BSS +1 이상
- seedbag: 3-seed 평균이 두 fold에서 모두 개선될 때만 유지
- 최종 제출: 내부 개선 + 추론 180초 이하 + 확률/CRC/경로 검증 통과

이 기준을 못 넘는 모델군은 trial 수가 남아도 중단하고 계산을 다른 축으로 이동한다.
