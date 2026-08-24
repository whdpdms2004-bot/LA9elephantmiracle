# LG Aimers 진행 상태

최종 업데이트: 2026-08-13 12:10 KST

## 현재 성능

| 모델/제출 | 핵심 구조 | Val2024 BSS | Public BSS | 상태 |
|---|---|---:|---:|---|
| Cat V1 | CatBoost 기준선 | 750.571 | 838.492 | 완료 |
| XGB V1 | XGBoost 기준선 | 745.131 | 873.075 | 완료 |
| `submit_013` | reverse 타자 클러스터 3-seed | 812.704 | 895.404 | 직전 Public 최고 |
| `submit_015` | 013 + R context residual | 830.523 | - | 내부 검증 |
| `submit_017` | 015 + F XGB/Cat expert | 833.342 | - | 내부 검증 |
| `submit_019` | 017 + F TabM | 835.861 | - | 생성 완료 |
| `submit_020` | 019 + reverse 균등 20-seed, scale 0.55 | 835.795 | - | 제출 후보 |
| `submit_021` | 019 + reverse 균등 20-seed, scale 0.40 | 836.503 | - | 대조군 |
| `submit_022` | 021 + reverse scale 0.475 중립화 | 836.242 | - | 대조군 |
| `submit_023` | 022 + 2025 고정 logit offset -0.0276337 | 817.283* | **916.700** | **Public 최고** |

`submit_019`의 과거 기록 837.214는 실제 추론 산식과 다른 계산이었다. 정확한 값은 835.861이다.

## 20-seed 실험 결과

### F TabM 20-seed

- epoch 1~6, 2023 월별 expanding validation, Val2024 gate까지 완료.
- 단독 seed 분산은 크게 감소했지만 2023 월별 robust objective에서 모든 epoch의 최적 혼합 가중치가 0이었다.
- 제출에는 사용하지 않고 기각했다.

### Reverse 타자 클러스터 20-seed

- 기존 3 seed를 포함한 총 20 seed 균등 평균.
- OOF와 2025 최종 lookup/pair/Ridge 60개 산출 완료.
- 기존 세 seed의 신규 산출물이 기존 파일과 완전히 동일함을 확인했다.
- `submit_020`: 기존 scale 0.55를 유지한 대조군.
- `submit_021`: downstream R/F 보정과 중복을 줄인 scale 0.40 공격형.
- 021의 019 대비 Val2024 개선은 +0.642 BSS지만 투수 단위 신뢰구간은 0을 포함한다.

## 제출 패키지

- 위치: `submit/2026-08-13/`
- 파일: `submit_020.zip`, `submit_021.zip`
- 파일명 길이: 각각 14자.
- ZIP 루트, CRC, 모델 파일, 절대 위치 기반 경로 확인 완료.
- 245,789행 R/F 혼합 추론 통과.
- 추론 시간: 020 약 22.7초, 021 약 22.9초.
- 상세 해시 및 설계: `submit/2026-08-13/SUBMISSION_LOG.md`.

## 실패유형 모델

- Middle와 Reverse gate 완료.
- Outside Optuna: 101회 시도, 85 완료, 15 pruning.
- Outside best trial 97:
  - Val2022 component BSS 1,586.007
  - Val2023 component BSS 1,790.799
- 아직 최종 `control_success` residual에는 연결하지 않았다.

## 다음 순서

1. `submit_021.zip` Public 제출 및 점수 기록.
2. `submit_020.zip` 제출해 scale 효과 분리 확인.
3. 두 Public 결과를 기준으로 reverse20 채택 여부 결정.
4. Outside top 모델을 낮은 가중치의 control-success residual로 연결.
5. Val2024 TrackMan 완전 미사용 기준선과 nested validation 재구축.

---

## submit_022 — 최종 종합 앙상블 (2026-08-13 추가)

`submit_021`의 컴포넌트 합집합을 유지하고 reverse scale만 0.475로 중립화했다.
`model/metadata.json` 한 파일만 다르며 나머지 79개 멤버와 `script.py`는 CRC까지 동일하다.

- Val2024 836.242 / R 833.518 / F 530.744
- SHA256 `A7D4DF24B219AD6B919BA10A64D42613ADC24D05D1215503FDD258146A4C74D1`
- 245,789행 스모크 통과 (실제 2024 행 + cold-start 10,000, 67.8초 / 4.31GB)
- 상세: `submit/2026-08-13/SUBMISSION_LOG.md`

### 중립화 근거

020(0.55)/021(0.40)은 사전 등록된 두 후보이고 이 축은 노이즈다.
021 vs 019의 투수 블록 95% CI가 [-1.36, +2.62]로 0을 포함하고,
scale 0.25~0.50 sweep의 BSS 폭이 0.2뿐이다. R은 scale↓에서, F는 scale↑에서
단조 개선해 서로 상쇄한다. 따라서 점을 고르지 않고 두 후보의 중점을 쓴다.

체인 자체는 견고하다: 021 vs 013 = +23.80, CI [+7.43, +40.41]로 0을 포함하지 않는다.

## 보정 실험 결과 (신규)

### smoothing / sharpening 전역 적용 — 기각

`a* = Cov(p,y)/Var(p)` 기준 전체 1.047, R 1.030, F 1.285. 전체·R은 CI가 1.0을 포함하고
leave-one-month-out으로 정직하게 적합하면 각각 -2.52 / -3.52 BSS로 **악화**한다.
10분위 reliability에서 9개 구간 gap이 ±0.007 이내다. 이미 잘 보정돼 있다.

시스템별 a*: 013 = 1.073 → 019 = 1.044 → 022 = 1.047.
R context와 F expert residual 추가가 이미 sharpening 역할을 수행했다.

### confidence 조건부 shrinkage — 기각

`p' = m + a(x)(p-m)`, `a(x) = w0 + Σ w_k z_k` 형태를 OLS 닫힌 해로 적합하고 LOMO 검증했다.

| 구성 | LOMO ΔBSS | 개선 월 |
|---|---:|:---:|
| 전역 a | -2.52 | 4/8 |
| a(x) 전체 (표본량·불일치·결측·F) | **-5.90** | 3/8 |
| a(x) 표본량만 | -6.02 | 4/8 |
| a(x) 모델 불일치만 | -0.78 | 4/8 |
| a(x) seed 불일치만 | -1.69 | 4/8 |

전부 음수다. in-sample로는 a(x) 전체가 +7.29를 내지만 out-of-sample -5.90으로,
파라미터 6개에 13점 과적합 갭이 생긴다.

**이유: 이미 구현돼 있다.** 예측 표준편차가 confidence에 따라 이미 단조 증가한다.

| `asof_pitcher_n` | n | 예측 sd |
|---|---:|---:|
| 0 | 81 | 0.01758 |
| 1~100 | 7,908 | 0.03607 |
| 101~1,000 | 50,745 | 0.03779 |
| 1,001~4,000 | 100,849 | 0.04226 |
| > 4,000 | 93,924 | 0.04326 |

cold start 투수는 베테랑 대비 예측 폭이 40% 수준이다. `prior_strength=200` Bayesian
smoothing, 클러스터/pair smoothing 1000/5000, unknown pair의 0-residual fallback이
confidence 조건부 축소를 **피처 단계에서** 이미 수행하고 있다. 사후 계층을 얹으면 이중 축소가 된다.

표본량 방향도 직관과 반대다: `pitcher_n` 하위 3분위 a* = 1.082, 상위 3분위 1.021로
저표본 행이 오히려 sharpening을 원한다. 상류 축소가 이미 (약간 과하게) 걸려 있다는 뜻이다.

## 다음 순서 (갱신)

1. `submit_022.zip` Public 제출 및 점수 기록.
2. TabM 추론 경로는 서버에서 실행된 적이 없다. 022가 0점이면 `script.py:357`의
   pandas 3.x read-only 문제를 의심하고 `submit_019.zip`으로 좁힌다.
3. **F expert 과소분산 보정** — a*_F = 1.285, LOMO에서 F BSS +16.92 (전체 +1.99)로
   정직 검증을 통과한 유일한 축이다. 단 CI [0.991, 1.542]가 1.0을 포함하고 현행 F
   가중치는 2023 월별 OOF 선택값이므로, 2022/2023 F 행에서 적합하고 2024를 1회
   게이트로 쓰는 nested 검증이 필요하다. 구현은 사후 배수보다 F expert 가중치 인상이 자연스럽다.
4. Val2024 TrackMan 완전 미사용 기준선과 nested validation 재구축.
5. Tier E 구성5를 211피처 파이프라인에서 재검증 (transformer 계열 착수의 선행 게이트).

---

## submit_023 — 최종 제출본 (2026-08-13)

`submit_022`의 컴포넌트·가중치를 그대로 두고 추론 마지막에 전 투구 공통 고정 logit bias를 적용한다.

```
z_final = logit(clip(p, 1e-7, 1-1e-7)) - 0.0276337141
p_final = sigmoid(z_final)
```

- 근거: 2024 성공률 0.4861에서 최근 연간 하락폭(-1.386%p)의 50%를 추가 반영 -> 2025 목표 0.4792
- `logit(0.4861) = -0.0556143299`, `logit(0.4792) = -0.0832480441`, offset = `-0.0276337141`
- 2025 평가 데이터 기반 재계산·재보정은 하지 않는다. 사전 결정 상수를 그대로 적용한다.
- `submit_021` 대비 변경 멤버는 `script.py`와 `model/metadata.json` 둘뿐 (모델 산출물 78개 CRC 동일)
- `script.py`의 주석·docstring 전부 제거
- SHA256 `0A9B1020E61E1616912EC86B945649EA01043B1B09CC43E32836B4D4A9C9CD73`
- 245,789행 스모크 통과, 162.4초 / 4.29GB
- 산식 검증: `max |p023 - sigmoid(logit(p022) + offset)| = 2.22e-16`, 실측 logit 이동 `-0.0276337141`

`*` **023의 Val2024 817.283은 설계대로 낮은 값이다.** Val2024 타깃 평균은 0.486105인데 offset은
2025의 0.4792를 겨냥하므로 2024 기준으로 -0.691%p의 의도된 편차가 생긴다 (이론 손실 19.13,
실측 18.96 BSS). **022와 Val2024로 비교하면 안 된다.** 순이득은 2025 실제 성공률이 0.4792에
얼마나 가까운지에만 달려 있다.

상세: `submit/2026-08-13/SUBMISSION_LOG.md`

---

## Public 결과 (2026-08-13)

`submit_023.zip` = **916.6997308612**  (직전 최고 `submit_013` 895.404000081 대비 **+21.296**)

| | Val2024 | Public |
|---|---:|---:|
| `submit_013` | 812.704 | 895.404 |
| `submit_022` (offset 없음) | 836.242 | 미제출 |
| `submit_023` | 817.283 | **916.700** |

- 내부 체인 이득(013 -> 022) +23.538 대비 Public 증가 +21.296 -> **전이율 90.5%**
- 잔차 적층식 개선은 Public으로 안정적으로 전이된다. Cat V1/XGB V1 때의 내부-Public 역전은
  모델 계열 교체 특유의 현상이었던 것으로 보인다.
- 평가 서버의 pandas가 3.x가 아님이 확인됐다 (TabM 경로 정상 실행). 해당 이식성 위험 해소.

### 아직 분해되지 않은 것

체인 채택과 logit offset을 동시에 넣었으므로 +21.296의 내역을 알 수 없다.
offset 손익분기는 2025 실제 성공률 **0.48265**다.

**다음 제출 1순위: `submit_022.zip`** (offset 없는 동일 구성). 두 Public 점수의 차이가
2025 레벨 가정의 직접 검증이 되고 2025 실제 성공률을 역산할 수 있다.

의사결정 전문: `submit_41.md`
