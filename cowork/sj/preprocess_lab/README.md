# 전처리 랩 (preprocess_lab)

> **더 나은 전처리를 찾으면 여기에 파일 하나만 추가하면 됩니다.** 기존 코드는 건드리지 않습니다.
>
> LG Aimers 9기 Phase 2 · KBO 제구 성공 확률 · 지표 Brier Skill Score

---

## 무엇을 하는 곳인가

같은 모델·같은 피처 기반(`F1`) 위에서 **전처리만 바꿔가며** 무엇이 실제로 설명력을 올리는지 잽니다.
피처군마다 다른 전처리를 골라 쓰는 조합까지 탐색합니다.

지금까지 확인된 것 한 줄 요약:

> **다 켜는 것이 최선 하나보다 나쁘다.** `id_frequency` 단독 **+7.94**, 전부 켜기 **+2.78**.
> 최적은 부분집합이고, 그걸 찾는 게 이 랩의 일입니다.

전체 순위와 해석은 **[RESULTS.md](RESULTS.md)**, 측정 방식과 함정은 **[METHOD.md](METHOD.md)**.

---

## 폴더

```
cowork/sj/preprocess_lab/
├─ README.md            이 파일
├─ METHOD.md            ★ 어떻게 재는가 / 밟았던 함정 4가지
├─ RESULTS.md           현재 순위와 해석
├─ transforms/          원자 변환. 여기에 파일을 추가하면 자동 등록된다
│   ├─ __init__.py         레지스트리 (내장 15개 + 기여분 자동 발견)
│   └─ example_template.py 새 전처리 템플릿 — 이걸 복사해서 쓰세요
├─ scripts/
│   ├─ run_combo.py        조합 학습·평가 (GPU)
│   └─ score_arms.py       저장된 예측 채점 (CPU 전용)
└─ outputs/             예측 .npy 와 점수 CSV
```

스크립트는 모두 `__file__` 기준으로 절대 경로를 계산합니다. **어느 디렉토리에서 실행해도 됩니다.**
실행하면 첫 줄에 해석된 절대 경로를 찍습니다.

---

## 빠른 시작

```bash
# 등록된 변환 목록 (GPU 불필요)
python cowork/sj/preprocess_lab/scripts/run_combo.py --list

# 이미 나와 있는 결과 채점 (GPU 불필요, 5초)
python cowork/sj/preprocess_lab/scripts/score_arms.py

# 조합 하나 학습 (GPU, 약 1분)
python cowork/sj/preprocess_lab/scripts/run_combo.py --combos id_frequency+temporal_cyclic

# 조합 공간 빔 서치 (GPU, 라운드당 약 40분)
python cowork/sj/preprocess_lab/scripts/run_combo.py --beam 3 --rounds 4
```

> **GPU 작업은 반드시 한 번에 하나.** 겹쳐 돌리면 둘 다 죽습니다 (실제로 두 번 겪었습니다).
> 시작 전에 `nvidia-smi` 로 확인하세요.

---

## 새 전처리를 추가하는 법

### 1. 템플릿 복사

```bash
cd cowork/sj/preprocess_lab/transforms
cp example_template.py my_idea.py
```

### 2. 네 가지를 채운다

```python
NAME = "my_idea"                                   # 고유 이름
TARGETS = ["asof_pitcher_success_rate"]            # 어느 피처를 건드리는가
NOTE = "왜 이게 더 나을 것 같은지 한 줄"
CONFLICTS = []                                     # 같이 켜면 안 되는 변환

def apply(frame, features, categorical, train_mask, fold):
    ...
    return extras, features, categorical
```

- `extras` : `{새 열 이름: 배열}`. 길이는 `len(frame)`
- `features` / `categorical` : 열을 빼거나 범주형 지정을 바꿀 때만 수정, 아니면 그대로 반환
- **통계는 `train_mask` 안에서만 계산**하세요

`DISABLED = True` 줄은 지웁니다 (템플릿이 등록되지 않게 하는 안전장치입니다).

### 3. 확인하고 돌린다

```bash
python .../run_combo.py --list                      # 등록됐는지
python .../run_combo.py --combos my_idea --dry      # 피처 수/충돌 확인 (GPU 불필요)
python .../run_combo.py --combos my_idea            # 실제 학습
python .../run_combo.py --combos my_idea+id_frequency   # 최고 조합과 함께
```

### 4. 결과를 RESULTS.md 에 한 줄 추가하고 PR

---

## 반드시 지켜야 할 것 넷

어기면 제출 자체가 실격입니다. 자세한 내용은 [METHOD.md §6](METHOD.md).

| | |
|---|---|
| **행 독립성** | test 행끼리 집계 금지. `groupby` / `rolling` / `cumsum` 전부 |
| **시간 인과** | 모든 통계는 `season < fold` 에서만 fit. as-of 컬럼은 그대로 써도 됨 |
| **라벨 직접 사용 금지** | `control_success` 를 피처 계산에 쓰지 않기 |
| **결정적** | 난수 쓰면 시드 고정 |

라벨 집계 테이블(EB 평활 등)을 만들 거라면 **셀 크기를 먼저 확인**하세요.
자기 라벨 누수로 네 번 무너졌습니다 ([METHOD.md §7](METHOD.md)).

---

## 판정 기준 (요약)

| | |
|---|---|
| 순위 지표 | **`bss_centered`** — 평균 정렬로 번 점수를 뺀 순수 신호 |
| 스크리닝 | fold 2024 단독 |
| 확인 | fold 2023 + 2024. **2022 는 쓰지 않음** (역신호) |
| 잡음 | fold 2024 sd 1.37. **±1.4 미만 차이는 같은 값** |
| 채택선 | 내부 +3 미만은 제출 근거로 쓰지 않음 |

**`bss_raw` 로 순위를 매기면 안 됩니다.** 실제로 `temporal_cyclic` 이 raw 로는 −17.18(탈락)인데
신호만 보면 +4.75(2위)였습니다. 이유는 [METHOD.md §2](METHOD.md).

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| [METHOD.md](METHOD.md) | 측정 방식, 함정 4가지 |
| [RESULTS.md](RESULTS.md) | 현재 순위와 해석 |
| [../claude/21_TEAM_SUMMARY.md](../claude/21_TEAM_SUMMARY.md) | 전체 모델링 정리 |
| [../claude/24_APPROACH_TYPOLOGY.md](../claude/24_APPROACH_TYPOLOGY.md) | 접근법 유형화 (무엇이 통했나) |
| [../claude/22_FEATURE_INTAKE_PLAN.md](../claude/22_FEATURE_INTAKE_PLAN.md) | 신규 피처 인수 절차 |
