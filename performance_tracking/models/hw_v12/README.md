# hw_v12 — 학습 코드 원본 위치

전부 `cowork/hw/` 에서 가져왔다 (origin/main). 원본이 단일 출처이고 이 폴더는 사본이다.

| 파일 | 원본 | 역할 |
|---|---|---|
| `train_best_model_v12.py` | `cowork/hw/train_best_model_v12.py` | **학습 전문** — `model/` 21파일 생성 |
| `script_v12_inference.py` | `cowork/hw/submission_v12/script.py` | 추론 (zip 의 `script.py` 와 동일) |
| `build_val2024_pred_v12.py` | `cowork/hw/build_val2024_pred_v12.py` | `val/hw_v12_2024.csv` 생성기 (fit season<2024) |
| `SUBMISSION_LOG.md` | `cowork/hw/submission_v12/SUBMISSION_LOG.md` | 구성·전이배수 기록 |
| `validate/*.py` | `cowork/hw/validate_*.py` | v12 채택 근거 (hand 인코딩 / id 범주형 기각 / drift-drop 기각) |

**없는 것: val2022 예측.** `build_val2024_pred_v12.py` 를 `<=2021` 학습으로 돌리면
같은 규격으로 만들 수 있다 (`--season` 인자 없음 — 상수 2024 를 고쳐야 한다).
