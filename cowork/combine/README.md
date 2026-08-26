# cowork/combine — 팀 모델 결합 공용 폴더

`plan.md` 우선순위 1 "팀 모델 결합"을 위한 공용 작업 공간이다. 각자 폴더(`cowork/<initial>/`)는
그대로 두고, **여러 명의 예측을 모아서 하는 분석**(상관계수, 블렌드 가중치, 결합 스크립트)만
여기 둔다.

AGENTS.md A1 규칙대로 이 폴더는 **공용 파일**이라 쓰기는 PR로만 한다.

## 필요한 입력 — 각자 폴더에 이 파일을 둔다

`cowork/<initial>/val2024_pred.csv` (필수), `cowork/<initial>/val2022_pred.csv` (선택, 안정성
체크용)

스펙 (전원 동일해야 상관계수·가중치 계산이 성립함):

- **fit**: `season < val_season`만 사용 (val_season 자신은 학습에 전혀 안 씀)
- **val**: `season == val_season` 전체 행에 예측 (원본 순서 그대로, 재정렬 없음)
- 컬럼: `row_id, control_success` 딱 두 개
- season/bucket 오프셋 등 **2025 외삽용 보정은 적용 안 함** — raw 앙상블 확률 그대로
- 리더보드 미참조 (RULES.md §2)

기준 스크립트: [`cowork/hw/build_val2024_pred_v12.py`](../hw/build_val2024_pred_v12.py)

## 현재 파일 현황

| 멤버 | val2024_pred.csv | val2022_pred.csv |
|---|---|---|
| hw | ✅ | ⬜ (준비 중) |
| sj | ✅ | ⬜ (요청함) |
| yn | ✅ | ⬜ (요청함) |
| cw | ✅ | ✅ |
| ye | ⬜ | ⬜ |

## 스크립트

- **`compute_correlation.py`** — `cowork/*/val2024_pred.csv`(있으면 `val2022_pred.csv`도)를
  자동으로 찾아서 상관행렬 + 멤버별 단독 BSS를 출력한다. 새 멤버가 파일을 추가하면 자동으로
  잡힌다.
- **`compute_blend_weights.py`** — Val2024 실제 라벨로 `w* = M^-1 A`를 투수표본수
  구간별(`asof_pitcher_n` < 200 / 200~2000 / 2000+)로 계산한다. 리더보드 미참조
  (RULES.md §2 준수 — Val2024 실제 라벨만 사용, 평가 데이터 값·분포·순위 일절 안 씀).
  출력: `blend_weights.json`, `val2024_oof_<members>.csv`

실행 (저장소 어디서나, 저장소 루트 기준 상대경로라 clone 위치 무관):

```bash
python cowork/combine/compute_correlation.py
python cowork/combine/compute_blend_weights.py
```

## 지금까지 나온 것 (참고, 리더보드 검증 결과는 각자 폴더의 SUBMISSION_LOG 참조)

- 3-way(hw+sj+yn) 구간별 블렌드: Val2024 868.11 → 실LB **1046.40**
- 4-way(hw+sj+yn+cw) 구간별 블렌드: Val2024 909.18 → 실LB **1068.42**
  (참고: cw+sj 2-way 전역 블렌드가 이미 실LB **1072** — 파라미터 늘린 구간별 4-way가
  오히려 못 미쳤음. hw/sj/yn/cw 상관이 0.88~0.96으로 높아서 구간별 가중치가
  Val2024에 과적합된 것으로 추정. 자세한 진단은 팀 채팅 공유 예정)

## 주의

- **여러 조합을 다 제출하지 않는다** — plan.md에 이미 적혀있듯 1일 5회 제한, 총 제출
  가능 횟수도 한정적이다(현재 잔여 30회). 정직 Val 이득 × 변화유형별 전이율로 예상
  실LB 이득을 추정해서, 기준(잠정 +15) 이상일 때만 제출한다.
- 가중치는 **Val2024/Val2022에서만** 산출한다. 리더보드 점수로 가중치를 역산하지 않는다
  (8/12 공식답변은 "여러 후보 중 리더보드로 선택"까지만 허용 — 연속값 역산은 별개 사안이니
  애매하면 먼저 공유하고 합의할 것).
