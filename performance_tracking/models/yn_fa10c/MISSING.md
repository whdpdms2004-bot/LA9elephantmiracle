# yn_fa10c — 학습 코드 없음 (규칙 4 미충족)

이 폴더에는 **추론 코드만** 있다. `models/yn_fa10c.zip` 의 `model/` (CatBoost 40개 +
LightGBM 20개 + team-ID CatBoost, 총 63파일)을 `data/train.csv` 에서 처음부터
재생성하는 학습 스크립트가 저장소에 없다.

| 있는 것 | 없는 것 |
|---|---|
| `script_fa10c_inference.py` (zip 의 `script.py` 그대로) | 학습 파이프라인 전체 |
| `requirements.txt` (zip 그대로) | 71피처 생성 코드 (68 + 오염보정 3) |
| `SUBMISSION_LOG.md` (구성·상수 근거) | isotonic 적합 코드, 시드 고정 절차 |

`SUBMISSION_LOG.md` 에 구성(피처 71개, CB_mix 0.00/1.00, raw = 0.10·LGB + 0.90·CB_mix,
final = isotonic → clip(0, 0.80))과 상수 출처는 다 적혀 있다. 재현 코드만 없다.

**필요 조치** — yn 에게 요청할 것 (`cowork/sj/last_week/R1_REPRO_MAP.md` §2 와 같은 건):

1. `model/` 63파일을 raw data 에서 재생성하는 학습 스크립트 전체
2. 개발환경(OS)·라이브러리 버전, 시드 고정 여부
3. `<=2021` 학습으로 만든 **val2022 예측** (`val/yn_fa10c_2022.csv` 규격)

RULES §7 의 09.11 검증은 **학습 코드**를 본다. 이 제출본이 팀 최종 패키지에 들어가면
1·2 는 필수다. 3 은 규칙 1 의 비하락 관문을 이 모델에 적용하기 위한 것이다.
