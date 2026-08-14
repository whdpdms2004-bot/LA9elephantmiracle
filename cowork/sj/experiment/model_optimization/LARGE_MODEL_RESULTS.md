# 대형 모델 용량 확장 결과

최종 갱신: 2026-08-12

## 결론

- 한 XGBoost를 32~128 leaves까지 크게 만드는 방식은 과적합으로 실패했다.
- 유효한 구간은 18→20~24 leaves의 완만한 확장이었다.
- 24-leaf 모델은 단일 Brier보다 AUC 개선이 뚜렷해, 기존 모델을 교체하기보다 별도 expert로 추가할 때 가치가 있었다.
- 최종 비교 후보 `submit_014.zip`은 기존 `submit_013` 구조를 보존하면서 24-leaf expert를 추가했다.
- 2024 내부 BSS는 `812.7040 → 813.4317`로 `+0.7277` 상승했다.
- 단, F23/F24의 최적 외부 가중치 차이가 커서 `submit_014`는 안정적 대체본이 아니라 고위험 비교 후보이다.

## 단일 모델 용량 실험

검증 조건은 2023 이하 학습·2024 검증이며, TrackMan은 2023 이하와 투수-시즌 500구 이상만 사용했다.

| 구조 | 실제 트리 | 최대 leaves | 모델 크기 | 2024 BSS | AUC | 판단 |
|---|---:|---:|---:|---:|---:|---|
| anchor | 2,642 | 18 | 5.73MB | 784.5568 | 0.549785 | 기준 |
| moderate20 | 2,371 | 20 | 5.64MB | **785.4658** | 0.549853 | 단일 BSS 최고 |
| moderate24 diverse | 2,398 | 24 | 6.47MB | 785.0671 | **0.550120** | expert 후보 |
| moderate28 | 2,429 | 28 | 7.32MB | 782.1512 | 0.549987 | 하락 시작 |
| wide32 flexible | 2,426 | 32 | 8.10MB | 780.8043 | 0.549966 | 제외 |
| wide48 flexible | 2,273 | 48 | 10.67MB | 774.0135 | 0.549990 | 제외 |
| wide64 regularized | 2,486 | 64 | 14.65MB | 761.8727 | 0.549757 | 제외 |
| wide96 regularized | 2,572 | 96 | 21.61MB | 744.4284 | 0.549484 | 제외 |
| wide128 regularized | 2,921 | 128 | 31.22MB | 726.3679 | 0.549108 | 제외 |

추가 트리만 늘린 slow/ultraslow 18-leaf도 각각 BSS 784.2326, 780.2591로 개선되지 않았다. 1024-bin 64-leaf 역시 BSS 758.7612로 제외했다.

## Seed 및 보정 실험

- anchor, 20-leaf, 24-leaf diverse를 각각 5-seed로 검증했다.
- 20-leaf는 seed 0 단독이 가장 좋았고 seed 평균 이득이 포화됐다.
- 24-leaf diverse 5-seed는 F23 개선은 커졌지만 F24가 anchor보다 낮아졌다.
- 따라서 모델 개수를 무조건 늘리는 seedbag은 채택하지 않았다.
- 24-leaf의 test-batch 평균·표준편차를 이용한 변환은 낙관적이므로 제출에서 제외하고 `large_xgb/rejected_transductive/`에 격리했다.
- 모든 fold에 같은 anchor 0.10 + large 0.90을 적용하는 causal 고정 구조를 최종 선택했다.

## 클러스터 correction 재학습

대형 expert의 OOF 잔차를 기준으로 success/reverse Ridge를 다시 학습했다.

| 항목 | 값 |
|---|---:|
| large 내부 비중 | anchor 0.10 + 24-leaf 0.90 |
| success correction scale | 0.200 |
| reverse correction scale | 0.575 |
| reverse seed | 17, 2026, 4099 평균 |
| F23 correction ΔBrier | -0.00001453 |
| F24 correction ΔBrier | -0.00004443 |
| large corrected 단일 2024 BSS | 803.3733 |

최종 3-way 가중치는 performance base 0.35955, 기존 corrected insight 0.25167, large corrected expert 0.38878이다. 이 조합의 2024 BSS는 813.4317이다.

## 제출 후보

| 파일 | 성격 | 2024 내부 BSS | Public LB | 추론 | ZIP |
|---|---|---:|---:|---:|---:|
| `submit_013.zip` | 현재 안전 기준 | 812.7040 | **895.4040** | 16.5초 | 104.6MB |
| `submit_014.zip` | 대형 expert 비교 | **813.4317** | 미제출 | 18.4초 | 106.5MB |

`submit_014.zip`의 파일명은 14자이고, ZIP 루트 구조·CRC·모델 파일 존재·245,789행 추론·확률 유효 범위를 모두 통과했다. SHA256은 `FAA513DF8F60E8DD88FFEF4B0A62B61DB965341AA8A471FE07BA4BAD9B02C926`이다.

## 산출물

- 용량 실험: `large_xgb/large_xgb_results.csv`
- seedbag 분석: `large_xgb/large_xgb_seedbags.csv`
- causal 구조 비교: `pitcher_cluster_matchup/reports/causal_large_summary.json`
- correction grid: `pitcher_cluster_matchup/reports/causal_large_corrections.csv`
- 3-way 혼합: `pitcher_cluster_matchup/reports/causal_large_threeway.csv`
- 제출 manifest: `pitcher_cluster_matchup/final/large_xgb_submission_manifest.json`
- 재현 코드: `benchmark_large_xgb.py`, `analyze_large_xgb.py`, `pitcher_cluster_matchup/src/search_causal_large_xgb.py`, `pitcher_cluster_matchup/src/build_large_xgb_submission.py`
