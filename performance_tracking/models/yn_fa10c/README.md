# yn_fa10c 재현 학습 파이프라인

이 폴더만으로 yn_fa10c.zip의 model/을 공식 train.csv에서 다시 만든다.
최종 모델은 TrackMan 피처를 사용하지 않으므로 trackman_history.csv는 읽지 않는다.

## 구성

- features.py — 기존 68피처 생성. 학습과 추론의 로직이 동일하다.
- pipeline.py — A 오염보정 3피처, 모델 파라미터, 학습·체크포인트 공용 코드.
- train_fa10c.py — 2024 calibration, 전체 재학습, 63엔트리 ZIP 생성.
- build_val_predictions.py — <=2021→2022, <=2022→2023 raw 예측 생성.
- verify_pipeline.py — 기존 제출 ZIP과 stats, lookup, 71피처 값·순서를 대조.
- script_fa10c_inference.py — 기존 제출 ZIP의 추론 코드.

## 개발환경

- macOS / Python 3.9.6
- numpy 2.0.2
- pandas 2.3.3
- scikit-learn 1.6.1
- LightGBM 4.6.0
- CatBoost 1.2.10

모든 모델 시드는 0..19로 고정한다. LightGBM과 CatBoost는 각각 네이티브
텍스트/CBM 형식으로 저장하며 sklearn 객체를 pickle하지 않는다.

## 데이터

기본적으로 현재 디렉터리에서 위로 올라가며 다음 파일이 함께 있는 data/를 찾는다.

    data/train.csv
    data/test.csv

다른 곳에 있으면 모든 명령에 --data-dir /path/to/data를 붙인다.
test.csv는 정답이나 분포를 사용하지 않고 47개 원본 피처의 이름과 순서만 읽는다.

## 1. 빠른 계약 검증

    python performance_tracking/models/yn_fa10c/train_fa10c.py --check-only
    python performance_tracking/models/yn_fa10c/verify_pipeline.py

두 번째 명령은 기존 models/yn_fa10c.zip의 meta.json과 다음을 대조한다.

- raw 47열과 최종 71열의 이름·순서
- 학습 통계량
- cutoff=2024 선수 lookup
- 5행 test에서 학습 코드와 제출 추론 코드가 만든 모든 피처 값

## 2. 최종 제출 재학습

    python performance_tracking/models/yn_fa10c/train_fa10c.py

단계:

1. 2019~2023 학습, 2024 예측으로 isotonic 좌표 적합
2. isotonic 출력 상한을 학습 데이터 근거의 0.80으로 고정
3. 2019~2024 전체로 LGB20 + numeric-CB20 + team-CB20 재학습
4. cutoff=2024 A피처 lookup을 meta.json에 저장
5. build/yn_fa10c_reproduced.zip 생성

중간 모델은 build/checkpoints/에 시드별로 저장되어 중단 후 같은 명령으로 재개할 수
있다. 최종 ZIP은 model/ 61개 + script.py + requirements.txt, 총 63엔트리다.

## 3. 2022·2023 walk-forward 예측

    python performance_tracking/models/yn_fa10c/build_val_predictions.py
    python performance_tracking/tools/score_val.py yn_fa10c --baseline cw_v17_base

생성 파일:

    performance_tracking/val/yn_fa10c_2022.csv  # <=2021 학습
    performance_tracking/val/yn_fa10c_2023.csv  # <=2022 학습

등록된 2024 파일과 동일하게 raw = 0.10*LGB20 + 0.90*teamCB20을 저장한다.
isotonic은 적용하지 않는다. 이는 팀 결합의 상관·가중치 계산에서 세 시즌의 스케일을
맞추기 위한 계약이며, 평가 시즌 라벨은 학습·조기종료·보정 어디에도 사용하지 않는다.

## 4. 재현 대조 결과 (2026-08-28)

전체 재학습 120모델을 4시간 50분에 완주했다. 재생성 ZIP 과 기존 제출 ZIP 을
대조한 결과는 MISSING.md 의 검증표에 있다. 요지는 63엔트리 이름 일치이고,
두 패키지의 script.py 를 같은 입력으로 실행한 예측이 3,000행 전부
최대절대차 0.000e+00 이다.

## 재현 판정

ZIP 자체의 SHA-256은 압축 순서와 타임스탬프에 따라 달라질 수 있다. 재현 기준은
63개 엔트리 각각의 내용 해시다. 기존 canonical 경로에서 63/63 내용 일치가 검증됐다:

    experiments/feature_engineering_20260819/build_submission.py
    experiments/repro_964_cap_20260822/apply_cap.py
    experiments/e2e_verify_20260822/
