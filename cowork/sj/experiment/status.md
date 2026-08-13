# 프로젝트 진행 현황 (status)

> **Legacy CSW 문서입니다. 현재 Phase 2 진행 현황은 프로젝트 루트의 `status.md`를 사용하세요.**

> 마지막 업데이트: 2026-07-16 (저녁 2차)
> 목표: 2017–2019 MLB Statcast(TrackMan 측정) 투구 데이터로 **CSW(콜드스트라이크+헛스윙) 예측 모델** 구축 + Kirby Index 기반 커맨드 피처 적용
> 라벨 결정(7/16): `is_csw` = description ∈ {called_strike, swinging_strike, swinging_strike_blocked} — 표준 CSW 정의
> 예측 시점 결정(7/16): **엄격한 투구 전** — 현재 투구의 구종·물리값·릴리스 자세·당일 존 측정치 전부 입력 금지

## 한눈에 보기

| # | 단계 | 상태 | 비고 |
|---|---|---|---|
| 1 | 설계서 + 수집/피처 파이프라인 작성 | ✅ 완료 | `pybaseball_statcast_raw_sequence_strike_target.md`, `build_statcast_strike_dataset.py` |
| 2 | 파이프라인 동작 검증 (합성 데이터 스모크 테스트) | ✅ 완료 | build_features → make_model_dataset → integrity_report **FAIL 0** |
| 3 | Kirby Index 모듈 구현 + 물리 검증 테스트 | ✅ 완료 | `kirby_index.py`, `test_kirby_index.py` — **4/4 PASS** |
| 4 | 원본 데이터 수집 (2017–2019, 월별 parquet) | ✅ 완료 (7/16) | 22개 청크, 121열, 2,201,095행 (2017: 724,627 / 2018: 724,452 / 2019: 752,016), 279MB |
| 5 | 실데이터 무결성 점검 | ✅ 완료 (7/16) | **16/16 PASS** — 중복 0, 누수 열 0, 시간순 분할 정상, 스트라이크율 46.0% |
| 6 | Kirby Index 실데이터 검증·산출 (B~C단계) | ✅ 완료 (7/16) | `reports/kirby_index/` — 아래 결과 요약 참고 |
| 7 | **원본 통합 파일 + CSW 라벨** | ✅ 완료 (7/16) | `data/statcast_2017_2019_raw_csw.parquet` — 2,201,095행 × 122열 (원본 121열 무변경 + is_csw), 메타 JSON 동봉 |
| 8 | CSW 모델 설계 + 데모 검증 (2018) | ✅ 완료 (7/16) | `train_csw_model.py` — 아래 2-2 참고. base AUC 0.632 → +cmd/csw 이력 0.639 |
| 9 | CSW 모델 전체 학습 (train 2017–18 / test 2019) | ⏳ **로컬 실행** | `python train_csw_model.py` (전체 데이터, 수 분 소요) |

## 1. 데이터 수집 (각자 로컬 PC에서)

샌드박스 환경에서는 baseballsavant.mlb.com 접근이 차단되어 수집을 로컬에서 실행해야 함.
2026-07-16에 확인: 수집 코드 자체는 정상 (프록시 403이 유일한 원인).

```bash
cd <프로젝트 폴더>   # 예: cd C:\Users\isj67\Desktop\LGAIMERS  ← 반드시 프로젝트 폴더에서 실행
pip install pybaseball pandas numpy pyarrow scikit-learn

# 1) 소규모 동작 확인 (노트북 statcast_strike_dataset_builder.ipynb에서 QUICK_TEST=True 실행) 후
# 2) 전체 수집 (약 3시즌 × ~72만 투구 = 약 210만 행, 수십 분 소요, 수백 MB)
python -c "from build_statcast_strike_dataset import *; collect_raw(BuildConfig())"

# 3) 데이터셋 빌드 + 무결성 점검 + 저장 (data/processed/)
python build_statcast_strike_dataset.py
```

중단돼도 이미 받은 월별 파일은 자동 스킵되므로 그냥 다시 실행하면 됨.

## 2. Kirby Index 적용 계획

참고: [Introducing the Kirby Index (FanGraphs, Rosen 2024)](https://blogs.fangraphs.com/introducing-the-kirby-index-a-new-way-to-quantify-command/)

핵심: Statcast 운동학(vx0~az)을 릴리스 지점으로 역전파해 **수직/수평 릴리스 각도(VRA/HRA)** 를 계산하면,
포심의 플레이트 위치가 {VRA, HRA, 릴리스 좌표}만으로 거의 설명됨(원문 R² 0.92/0.85).
투수별 이 4개 변수의 표준편차(작을수록 반복성 좋음)를 백분위→가중평균한 것이 Kirby Index.

| 단계 | 내용 | 산출물 | 상태 |
|---|---|---|---|
| A | 릴리스 각도 계산 (`add_release_angles`) | vra_deg, hra_deg | ✅ 구현+검증 (복원 오차 <1e-3°) |
| B | 위치 모델 재현: FF 대상 {각도+릴리스 좌표} → plate_x/z RF 회귀 | R² 리포트 (원문 0.92/0.85와 비교) | 수집 후 실행 |
| C | 시즌×투수 Kirby Index (min 125 FF) + 2017→18→19 stickiness | `reports/kirby_index/*.csv` (원문 R²≈0.5 기대) | 수집 후 실행 |
| D | 스트라이크 모델 피처화: `add_command_features()` — 과거 투구만 쓰는 rolling SD (`cmd_vra_sd_last40` 등 3종) | 모델 피처 + ablation (있/없 성능 비교) | 수집 후 실행 |
| E | (선택) 확장: FF 외 구종, 좌우 분리, K-means 다중 타깃 | — | 아이디어 |

누수 방지 원칙: 현재 투구의 각도는 릴리스 후에만 알 수 있으므로 **원본 각도 열은 모델 입력 금지**,
`cmd_*` rolling 피처(shift(1) 적용)만 사용. `COMMAND_FEATURE_LEAKAGE` 집합으로 관리.

B~D는 수집만 끝나면 `python kirby_index.py` 한 번으로 실행됨.

## 2-1. Kirby Index 실데이터 결과 (2026-07-16)

- 대상: 정규시즌 포심(FF) 765,398구 (시즌당 ~25만)
- **B단계 위치 모델 재현 성공**: {VRA, HRA, 릴리스 좌표} → plate 위치 R² **수직 0.890 / 수평 0.837** (원문 0.92/0.85와 부합, 10만 구 샘플·RF 100트리)
- 가중치(RF 중요도): VRA 0.385, HRA 0.306, rel_x 0.190, rel_z 0.119 → 각도가 지배적 (원문과 일치)
- **C단계 Kirby Index**: 시즌×투수 1,397건 (min 125 FF). Face validity 양호 — 2019 상위: Paddack, Odorizzi, Bieber 등 커맨드 명성 투수
- **Stickiness**: 2017→18 r=0.51, 2018→19 r=0.58 (R² 0.26~0.34; min 500 기준 0.30~0.34)
  - 원문(2022→23, R²=0.5)보다 낮음 — Trackman 시대 측정 노이즈 또는 시대 차이로 추정. 회의 논의 사항
- 산출물: `reports/kirby_index/kirby_index_by_season.csv`, `stickiness.csv`, `stickiness_min500.csv`, `location_model_report.json`

## 2-2. CSW 모델 설계 (2026-07-16 확정)

**데이터**: `data/statcast_2017_2019_raw_csw.parquet` (통합 원본 + is_csw). 시즌별 CSW율 27.1/27.5/27.7% (called 16.6% + whiff 10.8% — 리그 평균과 일치, 라벨 결측 0)

**예측 시점 규칙** (누수 3중 안전망: `CURRENT_PITCH_LEAKAGE` + `COMMAND_FEATURE_LEAKAGE` + `STRICT_PREPITCH_DROP`, 위반 시 assert 실패):

| 구분 | 예시 | 사용 |
|---|---|---|
| 경기 상황 | 카운트, 주자, 점수차, 이닝, 매치업, 수비 배치 | ✅ |
| 과거 이력 | 이전 구종 1–3, 직전 투구 물리값(lag), rolling 사용률/구속, 투구 수, 휴식일, LI(train만) | ✅ |
| CSW 이력 | 투수/타자 과거 CSW율 (expanding + last100, shift 1) | ✅ |
| 커맨드 상태 | `cmd_ff_*` — 투수의 최근 포심 릴리스 각도 SD (과거 FF만, 전 투구에 브로드캐스트) | ✅ |
| 현재 투구 | 구종, 구속/회전/각도/위치, sz_top/sz_bot, 결과·WPA | ❌ 금지 |

주의: 기존 `add_command_features`(투수×구종 그룹)는 현재 구종 정보가 간접 유입되므로 이 설계에서는 사용 금지 → `add_prepitch_ff_command_features`로 대체 (kirby_index.py)

**모델**: HistGradientBoostingClassifier (NaN 네이티브 처리, 범주형 지원). 지표: ROC-AUC, PR-AUC, log loss, Brier

**데모 결과** (2018년, train 3–7월 25만 샘플 / test 8월~ 24.3만, max_iter 120):

| 피처셋 | ROC-AUC | PR-AUC | log loss |
|---|---|---|---|
| base 71개 (상황+이력) | 0.6315 | 0.3851 | 0.5659 |
| full 78개 (+cmd_ff 3, csw 이력 4) | **0.6392** | **0.3915** | **0.5634** |

투구 전 정보만으로 AUC 0.64 수준 — 물리값 없는 예측으로는 타당한 범위이며, cmd/csw 이력 7개 피처가 +0.008 AUC 기여 확인.

## 3. 진행 로그

### 2026-07-16 (저녁 2차)
- 라벨 변경 결정: is_strike(type=='S') → **is_csw (표준 CSW)**. 예측 시점은 엄격한 투구 전으로 확정
- 원본 통합 파일 생성: 22개 월별 청크 → `statcast_2017_2019_raw_csw.parquet` (행 수 2,201,095 정확 일치 검증)
- `kirby_index.py`에 `add_prepitch_ff_command_features` 추가 (구종 간접 누수 제거 버전)
- `train_csw_model.py` 작성 + 2018 데모 검증: 누수 안전망 assert가 실제로 vra/hra 잔존을 잡아냄 → 수정 후 통과
- Ablation: cmd/csw 이력 7개 피처 → AUC +0.008

### 2026-07-16 (저녁 추가)
- 로컬 수집 완료 (Joo): 22개 월별 청크, 2,201,095행 — audit·integrity **16/16 PASS**
- 교차 검증: manifest 합계 = audit 행 수 정확히 일치, 시즌별 행 수 정상 범위
- Kirby Index B~C단계 실데이터 실행 완료 (위 2-1 요약)

### 2026-07-16
- 프로젝트 리뷰: 설계·파이프라인은 완성 상태이나 **데이터 미수집 상태**임을 확인 (data/ 폴더 없음, 노트북 실행 이력 없음)
- 샌드박스에서 수집 시도 → baseballsavant 프록시 403 차단 확인 → 수집은 로컬 실행으로 결정
- 합성 Statcast 데이터로 기존 빌더 파이프라인 end-to-end 스모크 테스트: integrity FAIL 0
- FanGraphs Kirby Index 아티클 분석 완료
- `kirby_index.py` 구현: 각도 역전파, 위치모델 검증, 시즌별 인덱스, stickiness, 누수 방지 rolling 피처
- `test_kirby_index.py` 물리 시뮬레이션 테스트 4종 모두 PASS (각도 복원 1.2e-7°, 위치 R² 0.94/0.98, 랭킹 정합, 누수 없음)
- 참고: 코드-문서 소소한 불일치 발견 (times_faced 0-시작 vs 문서 1-시작, rest_days -1일 계산, rolling 피처의 시즌 경계 통과) → 논의 필요

## 4. 다음 액션 (7/21 회의 전)

- [x] 전체 수집 + integrity 16/16 PASS (7/16, Joo)
- [x] Kirby Index B~C단계 실행, reports/kirby_index/ 생성 (7/16)
- [x] 원본 통합 파일 + CSW 라벨 (7/16)
- [x] CSW 모델 설계 + 2018 데모 + ablation (7/16)
- [ ] **로컬에서 전체 학습**: `cd 프로젝트폴더` → `python train_csw_model.py` → `reports/csw_model/metrics_full.json` 확인 (train 2017–18 / test 2019)
- [ ] 각자: 파이프라인 직접 실행해보기 (status.md 1번 명령 참고)
- [ ] 각자: 진행 내용 개인 노션 정리 + 5분 발표 준비
- [ ] 회의 안건: ① 문서-코드 불일치 3건 처리 ② cmd 피처 window(40구)·csw 이력 window(100구) 튜닝 ③ stickiness가 원문보다 낮은 원인 ④ 2스트라이크 파울 등 라벨 경계 사례 확인 ⑤ 대회 데이터 공개 시 피처 매핑 계획
