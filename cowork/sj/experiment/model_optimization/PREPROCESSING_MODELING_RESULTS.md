# 전처리 계획 및 인사이트 기반 모델링 결과

업데이트: 2026-08-12

## 1. 전처리 원칙

| 영역 | 적용 원칙 | 이유 |
|---|---|---|
| 메인 학습 행 | 이상치라는 이유로 행을 삭제하지 않음 | 논리 범위 위반이 0건이며 희귀 경기 상황 자체가 유효 신호임 |
| 비율 피처 | 원 비율 + 표본 수 + 평활 비율 + 신뢰도를 함께 사용 | 소표본 0/1 극단값의 과신 방지 |
| 횟수 피처 | 원 횟수와 `log1p`를 함께 유지 | 대형 표본의 순서 정보와 긴 꼬리를 동시에 표현 |
| 결측치 | 중앙값 대치 후 결측 indicator 추가 | 결측 자체가 선수 경력·TrackMan 가용성 신호일 수 있음 |
| TrackMan 극단값 | 과거 시즌에서만 계산한 metric별 winsorization 후보 사용 | 현재 검증/테스트 분포를 이용한 누수 방지 |
| 임베딩·군집 | RobustScaler → clipping → PCA → 손 방향별 군집 | 217차원 대비 투수 수가 적어 극단값·차원 불안정성을 축소 |
| 투수 자격 | 투수-시즌 500구 이상만 TrackMan 개별 표현 | 사용자가 정한 최소 표본 기준 준수 |
| 신인·저표본 | 별도 rookie/control-only 코드와 신뢰도 사용 | 억지로 기존 군집에 할당하지 않음 |
| 검증 | Val2023=`≤2022→2023`, Val2024=`≤2023→2024` | 투구 직전 및 과거 정보만 사용 |
| Val2024 TrackMan | 2024 TrackMan을 사용하지 않음 | 검증 시점 이후 정보 유입 금지 |

상세 감사 결과는 `PREPROCESSING_REVIEW.md`에 기록했다.

## 2. 전처리 감사에서 발견한 핵심 문제

기존 프로필 캐시의 failure component가 오래된 상태였다. 고표본 투수 중 최신 reverse/middle/outside가 빠진 비율은 cutoff 2023에서 86.19%, cutoff 2024에서 12.44%였다. `build_profiles.py`에 스키마 버전과 coverage 검사를 추가하고 `profiles_clean_v2`를 별도 생성한 결과 cutoff 2022~2025의 고표본 누락이 모두 0건이 됐다.

또한 217개 투수 특성에 비해 시즌별 대상 투수가 49~169명뿐이어서 full feature 군집은 1명짜리 군집이 자주 발생했다. 따라서 downstream 실험에서는 compact/physical 표현을 우선했다.

## 3. 모델링 결과

기준 모델은 Public 895.404를 기록한 `submit_013`, Val2024 BSS 812.704다.

| 단계 | 방법 | Val2023 안정성 | Val2024 BSS | 판단 |
|---:|---|---|---:|---|
| 0 | `submit_013` 기준 | 기준 | 812.704 | Public 검증 완료 |
| 1 | clean 투수 군집 success/reverse | 전체 개선, Val2023 R 악화 | 808.130 부근 | 단독 채택 안 함 |
| 2 | R middle 잔차: compact 투수 2/4 + 타자 2/3 | 전체/R 모두 개선 | 813.035 | 효과는 있으나 작음 |
| 3 | 단년 `카운트×4이닝구간` 잔차 | 전체/R 모두 개선 | 822.941 | 문맥 신호 확인 |
| 4 | 단년 `카운트×정확 이닝` 잔차 | 전체/R 모두 개선 | 825.353 | 월별 안정성 우수 |
| 5 | 다년 `카운트×4이닝구간×투·타 손` 안정형 | 전체/R 모두 개선 | 825.889 | smoothing 500, scale 0.60 |
| 6 | 다년 `카운트×4이닝구간×투·타 손` 최근성능형 | 전체/R 모두 개선 | **830.523** | smoothing 5000, scale 1.15 |
| 7 | 위 문맥 보정 + 대형 XGB `submit_014` | 전체/R 모두 개선 | **831.321** | 현재 내부 최고 |

모든 R 문맥 후보는 `game_type=R`에만 적용했다. 따라서 F 예측과 F Brier는 완전히 동일하다. 선택된 최고 문맥 모델은 TrackMan과 투수 군집을 사용하지 않으므로 Val2024 TrackMan 제한과도 무관하다.

## 4. 가장 중요한 인사이트

1. R 최근 성능 저하의 핵심은 reverse보다 middle 증가였지만, 최종 잔차를 가장 잘 설명한 것은 선수 군집보다 `볼카운트×이닝×손 방향` 문맥이었다.
2. middle 군집 보정은 단독으로 Val2023·Val2024 R을 모두 개선했지만 개선폭은 약 0.33 BSS였다.
3. 문맥 잔차는 Val2023·Val2024 전체/R을 동시에 개선했다. F를 건드리지 않고 Val2024에서 약 17.9 BSS를 추가했다.
4. 직전 한 시즌보다 과거 두 시즌 OOF 잔차의 동일 가중 결합이 더 좋았다. 셀별 변동성을 줄이면서 2024 적합도가 상승했다.
5. 선택된 최근성능형 lookup은 192개 셀, 평균 절대 보정폭 0.00407, 최대 0.02259로 작다. smoothing 5000이 저표본 셀을 강하게 0으로 수축한다.
6. `카운트×정확 이닝` 후보는 Val2023 7/7개월, Val2024 7/8개월에서 개선됐다. 연도별 최적 배율도 모두 양수였다.

## 5. 최종 후보

| 후보 | 기반 | Val2024 전체 BSS | Val2024 R BSS | Public | 성격 |
|---|---|---:|---:|---:|---|
| `submit_015.zip` | `submit_013` | 830.523 | 832.860 | 미제출 | 검증된 Public 기반 |
| `submit_016.zip` | `submit_014` | **831.321** | **833.466** | 미제출 | 내부 최고, 대형 모델 포함 |

두 파일 모두 245,789행 모의 추론, ZIP CRC, 루트 구조, 확률 범위를 통과했다. 추론 시간은 각각 17.5초와 18.5초다.

## 6. 산출물

- 전처리 감사: `PREPROCESSING_REVIEW.md`
- clean 프로필: `pitcher_cluster_matchup/profiles_clean_v2/`
- clean 군집: `pitcher_cluster_matchup/clusters_preprocess_v2/`
- R middle 결과: `pitcher_cluster_matchup/reports/r_middle_current_blend_summary.csv`
- R 문맥 탐색: `pitcher_cluster_matchup/reports/r_context_history_summary.csv`
- 후보 안정성: `pitcher_cluster_matchup/reports/r_context_candidate_audit.json`
- 대형 모델 결합: `pitcher_cluster_matchup/reports/r_context_large014_summary.json`
- 2025 lookup: `pitcher_cluster_matchup/artifacts/r_context_2025/`
- 제출 manifest: `pitcher_cluster_matchup/final/r_context_submission_manifest.json`

## 7. 다음 실험 순서

1. `submit_015`와 `submit_016`의 Public 점수로 문맥 보정 및 대형 모델의 실제 전이성을 분리한다.
2. Public에서 문맥 보정이 재현되면 smoothing 3000/5000과 scale 0.9/1.15 사이의 국소 후보를 한 번만 추가한다.
3. 이후에는 R 문맥 잔차를 트리 모델의 입력 피처로 포함한 OOF 재학습을 진행한다. 사후 보정과 입력 피처 효과를 분리해서 비교한다.
4. middle 임베딩은 문맥 보정 이후 남은 잔차에만 다시 적합한다. 현재처럼 원 기준 잔차에 적합하면 문맥 신호와 중복된다.
