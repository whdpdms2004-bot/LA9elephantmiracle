# performance_tracking — 제출 모델 성능 관리

이 폴더가 앞으로 **제출본 단위 모델 관리**의 단일 출처다. 여기 규약대로 등록되지
않은 모델은 앙상블 후보로 쓰지 않는다.

---

## 0. 규칙 4개 (확정: 2026-08-26)

| # | 규칙 |
|---|---|
| 1 | **Val 은 2022 와 2024 두 시즌.** 판정 주축은 **2024**, 단 **2022 가 떨어지면 채택하지 않는다.** |
| 2 | 모델 구성 설명은 **제출본과 같은 이름의 `.md`** 파일로 남긴다. |
| 3 | 2022·2024 val **예측을 파일로 남겨** 앙상블 상관계수를 바로 잴 수 있게 한다. |
| 4 | `zip` 은 **제출본과 동일한 양식**, 학습 스크립트 전문은 **제출본과 같은 이름의 폴더**에 전부 넣는다. |

---

## 1. 폴더 구조

```text
performance_tracking/
├── README.md                  # 이 파일 — 규약
├── results.csv                # 등록부 (모델 1개 = 1행)
├── templates/
│   └── MODEL_TEMPLATE.md      # 규칙 2 의 MD 서식
├── tools/
│   ├── score_val.py           # 규칙 1 채점 + results.csv 등록
│   └── corr.py                # 규칙 3 상관계수 / 블렌드 스캔
├── models/
│   ├── <name>.md              # 규칙 2 — 모델 구성
│   ├── <name>.zip             # 규칙 4 — 제출 zip (script.py + requirements.txt + model/)
│   └── <name>/                # 규칙 4 — 학습 스크립트 전문 (재현 가능한 전부)
└── val/
    ├── <name>_2022.csv        # 규칙 3 — val 예측
    └── <name>_2024.csv
```

`<name>` 은 **제출 zip 의 파일명(확장자 제외)** 과 글자 하나까지 같게 쓴다.
`.md` / `.zip` / 폴더 / val 예측 4곳의 이름이 어긋나면 도구가 등록을 거부한다.

### zip 양식 (규칙 4 — 제출본과 동일)

```text
<name>.zip
├── script.py          # 추론 전용. 학습 코드 금지 (cowork/RULES.md §0-7)
├── requirements.txt
└── model/             # 가중치·LUT·params.json
```

학습 코드는 zip 에 넣지 않고 `models/<name>/` 에 둔다. 그 폴더만 있으면
zip 의 `model/` 을 처음부터 다시 만들 수 있어야 한다 (경로·시드·데이터 as-of 포함).

---

## 2. val 예측 파일 규격 (규칙 3)

`val/<name>_2022.csv`, `val/<name>_2024.csv` — 컬럼 2개, 헤더 필수.

```csv
row_id,pred
TRAIN_1221586,0.5123
```

- `row_id` 는 `data/train.csv` 의 `row_id` 문자열 그대로 (`TRAIN_...`). 정렬 순서는 무관하다 (도구가 join 한다).
- `pred` 는 **최종 `control_success` 확률** 하나. way 확률(middle/reverse/outside)이나
  로짓을 넣지 않는다 — 평균이 0.35~0.65 밖이면 도구가 거부한다.
- 해당 시즌의 `control_success` 비결측 행을 **전부** 담는다. 결측·중복·누락은 오류다.
- 여기 저장하는 예측은 **그 시즌을 학습에서 제외하고** 낸 값이어야 한다.
  2022 예측을 2022 가 학습에 들어간 모델로 내면 규칙 1 의 비교가 무의미해진다.

---

## 3. 채점 (규칙 1)

```bash
python performance_tracking/tools/score_val.py <name>
```

- 대회 공식 지표: `Score = 100000 x (1 - Brier / (r(1-r)))`, `r` 은 그 부분군 실제 기저율.
- 시즌마다 `all / R / F` 와 월 블록(early 3-5, mid 6-7, late 8-10)을 함께 낸다.
  `game_type` 은 2022→2023 에 구조적 단절이 있어(`cowork/sj/three_way/VALIDATION_POLICY.md` §1.1)
  전체 BSS 하나만 보면 F 부분군이 만든 착시를 그대로 채택하게 된다.
- **채택 판정**: 기준 모델 대비 `2024 all` 이 올랐고 **`2022 all` 이 떨어지지 않았을 때만** 채택.
  기준 모델은 `--baseline <name>` 으로 지정한다.
- 통과하면 `--register` 로 `results.csv` 에 한 행 append 한다 (기존 행은 수정하지 않는다).

## 4. 앙상블 상관 (규칙 3)

```bash
python performance_tracking/tools/corr.py                      # 등록된 전 모델 상관 행렬
python performance_tracking/tools/corr.py -m a -m b --blend    # 두 모델 가중 스캔
```

상관은 **예측 확률**과 **오차(pred - y)** 두 축으로 낸다. 확률 상관이 0.99 여도
오차 상관이 낮으면 결합 이득이 남아 있다.

> 결합층 상수(가중치·온도)는 **행 CV 로 고르지 않는다.** fold(N-1) 에서 적합하고
> fold(N) 에서 동결해 판정한다 — 행 CV 로 고른 상수는 시즌 전이에서 뒤집힌 실측이 있다.

---

## 5. 등록 절차 체크리스트

1. `models/<name>/` 에 학습 스크립트 전문을 넣는다.
2. `models/<name>.zip` 을 제출 양식대로 만든다 (`script.py` 는 추론 전용).
3. 2022·2024 를 각각 학습에서 빼고 예측을 내 `val/<name>_2022.csv`, `val/<name>_2024.csv` 로 저장한다.
4. `templates/MODEL_TEMPLATE.md` 를 복사해 `models/<name>.md` 를 채운다.
5. `score_val.py <name> --baseline <기준> --register` 로 채점·등록한다.
6. Public 점수가 나오면 `results.csv` 의 해당 행 `public` 칸과 `.md` 의 실측 표를 갱신한다.

## 6. 커밋 주의

루트 `.gitignore` 가 `*.zip`, `*.pt`, `*.npz` 를 전부 막고 있다. 이 폴더는
`performance_tracking/.gitignore` 로 `*.zip` 만 예외 처리했다 (규칙 4). zip 은 수십 MB 이므로
**등록 대상 제출본만** 넣고 실험 중간본은 넣지 않는다. `model/` 안의 개별 가중치
파일은 계속 무시된다 — zip 하나로만 보관한다.

---

## 5. 팀 전체를 한눈에 — `TEAM_ASSETS.xlsx`

누구 것인지 / 어디서 왔는지 / val 은 어떻게 쟀는지 / 결합에서 쓸모가 있는지를
엑셀 한 권으로 본다. `results.csv` 는 기계용 등록부고, 이건 **사람이 보는 것**이다.

| 시트 | 내용 |
|---|---|
| `1_팀원자산` | 등록명 · 팀원 · 원 출처 · val 예측 출처 · Public · val 점수 |
| `2_val계산방법` | 폴드 정의 · 지표 · 행 정합 · **오염 주의 2건** |
| `3_상관행렬(val2024)` | 멤버 간 예측 상관 |
| `4_결합판정` | 어떤 조합이 채택/기각인지와 근거 |
| `5_피처목록` | 각 모델이 쓴 피처 수 · 완전성 · 출처 |

    python tools/build_team_xlsx.py      # 다시 만든다
    python tools/extract_features.py     # 피처 목록을 다시 뽑는다

### val 은 전부 sj 가 다시 계산했다

팀원 자체보고를 그대로 옮기지 않았다. **같은 폴드 · 같은 식 · 같은 행 정합**으로
다시 쟀다 — 그래야 서로 비교가 되고 결합 가중을 적합할 수 있다.
(예: yn 자체보고 880.79 vs 여기 815.6. **두 숫자를 섞지 않는다.**)

### ★ 오염 2건을 찾았다

1. **hw 공유분** — `build_val2024_pred_v12.py` 가 `eval_set=(x_val,y_val)` +
   `use_best_model` 로 **검증 라벨을 조기 종료에 썼다.** sj 가 정직본을 다시 만들어
   `hw_v12_honest` 로 등록했다 (val2024 808.9 → **720.6**, val2022 **2204.9** 신규).
2. **sj 자신의 계기** — `run_arm.calib` 이 평가 폴드 라벨로 로짓 스케일을 골랐다.
   같은 모델이 저장 904.1 / 원시 851.3 으로 갈렸다.

### 피처 목록은 `features/`

여섯 모델 중 **다섯은 완전**, `yn_fa10c` 만 불완전하다 — 파생 68개의 이름이
저장소에 없고 학습 코드도 없어 복원할 수 없다. 자세한 것은 `features/README.md`.
