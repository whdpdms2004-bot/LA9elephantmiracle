# AGENTS.md

LG Aimers 9기 Phase 2 · 투구 제구 성공 확률 예측

세션을 시작하는 팀원과 에이전트는 이 파일을 먼저 읽는다.
**Part A는 협업 규칙**, **Part B는 제출 규칙**이다. 전체 개요는 [`README.md`](README.md), 데이터 컬럼 설명은 [`data_description.md`](data_description.md), 현재 상태와 다음 할 일은 [`cowork/plan.md`](cowork/plan.md)를 본다.

**제출본을 만든다면 [`cowork/RULES.md`](cowork/RULES.md)와 아래 [B1 점검 절차](#b1--제출본을-만들-때--규정-점검은-생략할-수-없다)를 반드시 거친다.**

---

# Part A. 협업 규칙

## A1. 폴더 구조

```text
cowork/
├── RULES.md     # 공용 — 대회 규정 원본 (PR 필수)
├── plan.md      # 공용 — 현재 상태 요약 + 다음 할 일 (PR 필수)
├── task.jsonl   # 공용 — append-only 작업 로그 (PR 필수)
├── sj/
├── ye/
├── cw/
├── yn/
└── hw/
```

- 각자 **자기 이니셜 폴더 안에서만** 작업하고 커밋한다.
- 공용 파일(`plan.md`, `task.jsonl`)은 `cowork/` 루트에 두고 모두가 읽는다. 쓰기는 PR로만.
- 다른 사람 폴더의 파일은 **참조·링크는 자유**, **직접 수정은 PR로만**.

## A2. 푸시 규칙

| 대상 | 규칙 |
| --- | --- |
| `cowork/<initial>/` (자기 폴더) | 자유 푸시, 리뷰 불필요 |
| `cowork/RULES.md`, `cowork/plan.md`, `cowork/task.jsonl` | **PR 필수** |
| 다른 사람 폴더 | **PR 필수** |
| 저장소 루트 파일 (`AGENTS.md`, `README.md`, `data_description.md` 등) | **PR 필수** |

- 자기 폴더(`cowork/<initial>/`) **이외의 경로는 직접 푸시하지 않는다.**
- `data/train.csv`와 `data/trackman_history.csv`는 대회 원본(합계 약 690MB)이라 커밋하지 않는다 (`.gitignore` 처리). 형식 확인용 5행 샘플 `data/test.csv`, `data/sample_submission.csv`만 커밋한다.
- 제출 ZIP, 모델 가중치(`*.pt`, `*.pkl` 등), `output/`, seed cache도 커밋하지 않는다.

## A3. task.jsonl — append-only 로그

"누가 언제 무엇을 했는지"의 원본 기록. **기존 줄은 절대 수정·삭제하지 않고 새 줄만 추가한다.**

**형식은 JSON이 아니라 JSONL(JSON Lines)이다.** 한 줄에 항목 하나, 줄바꿈으로 구분한다. JSON 배열을 쓰면 append할 때마다 닫는 `]`와 직전 항목의 쉼표를 건드려야 해서 동시 append가 반드시 같은 라인 충돌을 낸다. JSONL은 각자 파일 끝에 새 줄만 붙이므로 git이 line-based로 자동 머지한다.

- **항목 하나는 반드시 한 줄에** 쓴다. 보기 좋게 여러 줄로 pretty-print 하지 않는다.
- 파일 끝은 항상 개행으로 끝낸다. 그래야 다음 사람의 append가 마지막 줄에 붙지 않는다.
- `id`는 순차번호(`t001`) 대신 **타임스탬프 또는 UUID**를 쓴다.
- `author`에 자기 이니셜을 **반드시** 기입한다.
- 머지 충돌이 나면 **한쪽을 버리지 말고 둘 다 살린다.** 줄 순서는 중요하지 않다 (`ts`로 정렬 가능).

### 항목 스키마

한 줄 예시 (읽기 편하게 줄바꿈했지만 실제로는 **한 줄**):

```json
{"id":"2026-08-13T14:32:10Z-sj","author":"sj","ts":"2026-08-13T14:32:10+09:00","type":"exp","title":"투수 과거 성공률 prior 피처 추가","detail":"2023 이전 데이터만으로 shrinkage prior 계산","paths":["cowork/sj/feat_pitcher_prior.py"],"result":"Val2024 Brier 0.2431 / BSS 836.5","next":"타자손 교호작용 붙여보기"}
```

| 필드 | 필수 | 설명 |
| --- | --- | --- |
| `id` | ✔ | 타임스탬프 또는 UUID 기반 고유값 |
| `author` | ✔ | 자기 이니셜 |
| `ts` | ✔ | ISO 8601, KST 오프셋 포함 |
| `type` | ✔ | `feat` / `fix` / `exp` / `doc` / `chore` |
| `title` | ✔ | 한 줄 요약 |
| `detail` | ✔ | 무엇을 왜 했는지 |
| `paths` | ✔ | 관련 파일 경로 배열 |
| `result` | | 실험(`exp`)이면 **Brier와 BSS를 함께** 적는다. Part B1 참고 |
| `next` | | 다음에 이어서 할 일 |

### 안전한 append 방법

```bash
# 파일 끝에 한 줄만 추가. >> 를 쓰고 > 는 절대 쓰지 않는다.
echo '{"id":"...","author":"sj",...}' >> cowork/task.jsonl
```

```python
import json, datetime
entry = {"id": ..., "author": "sj", ...}
with open("cowork/task.jsonl", "a", encoding="utf-8") as f:   # "a" 모드
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

### 전체 읽기

```python
import json
with open("cowork/task.jsonl", encoding="utf-8") as f:
    tasks = [json.loads(line) for line in f if line.strip()]
tasks.sort(key=lambda t: t["ts"])
```

## A4. plan.md — 리뷰 시점 재작성

- **수시로 고치지 않는다.** 리뷰하는 시점에만 `task.jsonl`을 훑어서 다시 쓴다.
- 재작성할 때 문서 상단에 커서를 남긴다:

  ```markdown
  reviewed against task.jsonl up to 2026-08-13T14:32:10Z-sj
  ```

- 담는 내용: 지금 상태 요약 + 다음에 뭘 할지. 과거 이력의 나열은 `task.jsonl`의 몫이다.

## A5. 브랜치 · 커밋 규칙

- 브랜치: `<initial>/<short-desc>` — 예) `sj/cache-invalidation`
- 커밋: `[<initial>] <type>: <description>` — 예) `[sj] feat: add cache TTL config`
- `type`은 `task.jsonl`과 동일한 어휘를 쓴다: `feat`, `fix`, `exp`, `doc`, `chore`

---

# Part B. 제출 규칙

## B0. 규정 원본은 RULES.md다

**대회 규정의 원본은 [`cowork/RULES.md`](cowork/RULES.md)다.** 공식 페이지 원문, 8/13 재공지, 우리 코드의 실제 리스크 점검(§10)까지 들어 있다.

- 이 문서(Part B)는 **작업 절차**다. 규정 자체가 궁금하면 `RULES.md`를 본다.
- **두 문서가 충돌하면 `RULES.md`가 이긴다.** 충돌을 발견하면 이 문서를 고쳐서 맞춘다.
- `RULES.md`는 대회 공지가 갱신되면 함께 갱신한다. 갱신했으면 `task.jsonl`에 `type: "doc"`으로 남긴다.

## B1. ★ 제출본을 만들 때 — 규정 점검은 생략할 수 없다

제출 패키지를 새로 만들 때마다 **아래 6단계를 처음부터 끝까지** 밟는다.

> **"지난번 제출이 통과했으니 이번에도 괜찮다"는 근거가 되지 않는다.**
> 코드 한 줄, `metadata.json` 키 하나만 바뀌어도 전부 다시 본다. 8/13 재공지가 명시했듯 **제출 코드의 규정 준수 책임은 참가자에게 있다.**

점검 결과는 **반드시 `submit/<날짜>/SUBMISSION_LOG.md`에 기록**한다. 기록 없는 제출은 09.11 코드 검증에서 방어할 수 없다.

### 1단계 — RULES.md 전문 재독

- `RULES.md` §0(절대 금지 10개), §2(추론 독립성), §3(데이터·모델 제한), §5(ZIP 규격)를 **매번 다시 읽는다.**
- §10(우리 프로젝트 리스크 점검)에서 아직 해소되지 않은 항목이 이번 패키지에 남아 있는지 확인한다.

### 2단계 — 행 독립성 기계 검증

`RULES.md` §2의 판정 기준을 코드로 확인한다.

```text
predict(row_i 단독)  ==  predict(전체 test)[i]
```

```bash
# 전체(5행 샘플) 추론
python script.py && cp output/submission.csv /tmp/pred_full.csv

# 각 행을 단독 입력으로 추론해 비교
python - <<'PY'
import pandas as pd, subprocess, shutil, pathlib
full = pd.read_csv("/tmp/pred_full.csv")
test = pd.read_csv("data/test.csv")
shutil.copy("data/test.csv", "/tmp/test_backup.csv")
for i in range(len(test)):
    test.iloc[[i]].to_csv("data/test.csv", index=False)
    subprocess.run(["python", "script.py"], check=True)
    one = pd.read_csv("output/submission.csv")
    a, b = one["control_success"].iloc[0], full["control_success"].iloc[i]
    assert abs(a - b) < 1e-9, f"row {i} 위반: 단독 {a} != 전체 {b}"
shutil.copy("/tmp/test_backup.csv", "data/test.csv")
print("행 독립성 통과")
PY
```

불일치가 하나라도 나오면 **제출하지 않는다.** 원인을 찾아 제거한 뒤 다시 돌린다.

### 3단계 — script.py 금지 패턴 전수 스캔

```bash
grep -nE "groupby|rolling|cumsum|expanding|\.shift\(|\.rank\(|transform\(" script.py
grep -nE "\.mean\(\)|\.std\(\)|\.median\(\)|quantile|normalize|distribution_match" script.py
grep -nE "requests|urllib|httpx|socket|from_pretrained|hf_hub|download|api_key" script.py
grep -nE "\.fit\(|train\(|\.partial_fit\(|backward\(|optimizer" script.py
grep -nE "^\s*(/|[A-Za-z]:\\\\)|/home/|/workspace/|/app/" script.py
```

각 히트를 **한 줄씩 판정**하고 결과를 `SUBMISSION_LOG.md`에 남긴다.

| 검사 | 허용되는 경우 | 위반 |
| --- | --- | --- |
| 집계 함수 | `model/` 안 학습 기반 lookup을 `merge`, 같은 행의 seed 평균 `np.mean(..., axis=0)` | test 행을 가로지르는 `groupby` / `rolling` / `cumsum` / `shift` / `rank` |
| 통계량 | 학습 데이터에서 사전 결정된 **상수**를 전 행에 동일 적용 | `pred`의 mean/std/quantile로 재척도·재중심화 |
| 네트워크 | 없음 | 지연 import 포함 모든 원격 호출 |
| 학습 코드 | 없음 (`script.py`는 **추론 전용**) | fit / 가중치 업데이트 / 전처리 재학습 |
| 경로 | `Path(__file__).resolve().parent` 기준 | 절대 경로 하드코딩 |

**죽은 코드도 지운다.** 실행되지 않는 분기라도 09.11 검증에서 사람이 읽는다. `RULES.md` §10-B의 `distribution_match` 블록이 그 예다 — 활성 경로가 아니어도 설명 부담이 크므로 제거한다.

### 4단계 — 패키지 구조 · 실행 환경 검증

```bash
unzip -l submit_NNN.zip | head -20        # 최상위가 model/, script.py, requirements.txt 3개뿐인지
unzip -t submit_NNN.zip                    # CRC
```

- 서버와 **같은 경로 구조**로 smoke test를 돌린다. `model/`이 비어 있지 않은지, 필요한 산출물이 전부 들어갔는지 확인.
- 추론 시간을 실측한다. 로컬 5행이 아니라 **245,789행 기준으로 환산**해서 10분 안에 드는지 본다.
- 네트워크를 끊고 한 번 더 돌린다 (`unshare -n python script.py` 또는 오프라인 컨테이너).
- `requirements.txt`: 버전 고정, 서버 기본 패키지는 가급적 빼기 (`RULES.md` §5).

### 5단계 — 근거 문서화

09.11 코드 검증은 **"이 값이 어디서 나왔는가"**를 묻는다. 다음을 `SUBMISSION_LOG.md`에 남긴다.

**(필수) 상수 오프셋 출처 명시** — 아래 문구를 매 제출마다 그대로 기재한다. [`RULES.md`](cowork/RULES.md) §2 「제출 시 명시 문구」와 동일하다.

```markdown
### season_logit_offset 출처 명시

본 제출의 `season_logit_offset = <값>`은 **학습 데이터(2019~2024 시즌)만을 이용해 사전 결정된 상수**이며,
모든 평가 행에 동일하게 적용된다.

- 산출 근거: 2019~2024 시즌별 제구 성공률 추세의 외삽. 계산 코드는 `<경로>`.
- **리더보드 점수를 참조하거나 역산하여 조정한 값이 아니다.**
- 평가 데이터(test.csv)의 값, 분포, 평균, 순위를 일절 사용하지 않았다.
- 따라서 `predict(단독 행) == predict(전체 test)[i]`를 만족한다.
```

- 그 밖의 모든 상수 보정값(calibration 계수 등)도 같은 형식으로 **출처가 학습 데이터임을** 설명한다
- 리더보드 점수를 보고 조정한 값이 **하나도 없음**을 명시
- 2·3단계 검증 결과와 스캔 히트별 판정
- 검증 수치: Brier, 전체 BSS, R/F별 BSS, 월별 Brier, prediction mean
- ZIP 해시, 포함된 `model/` 산출물 목록

> **리더보드 역산 금지.** ablation을 제출해 점수를 재는 것은 정상 실험이지만, 거기서 역산한 값으로 offset을 재조정하면 "전체 평가 데이터의 평균을 이용한 보정"이 되어 8/13 재공지 위반이다. `RULES.md` §10-A 참조. **"2019~2024 추세 외삽"은 방어되지만 "리더보드에서 역산"은 방어되지 않는다.**

### 6단계 — 체크리스트 두 개를 모두 통과

- [`RULES.md` 부록](cowork/RULES.md) — 규정 항목
- 이 문서 [B9](#b9-제출-전-체크리스트) — 운영 항목

둘 중 하나라도 미체크 항목이 있으면 제출하지 않는다.

## B2. 평가 지표

공식 지표는 **Brier Skill Score**다.

```text
Brier      = mean((prediction - target)^2)
null_brier = r * (1 - r)          # r = 전체 평가 데이터 평균 제구 성공률
BSS        = max(0, 100000 * (1 - Brier / null_brier))
```

- 예측값은 확률이다. 0/1 하드 라벨을 내지 않는다.
- 모델 비교는 0으로 잘리는 BSS보다 **Brier / normalized Brier**를 우선한다.
- AUC나 0.5 임계값 F1이 높다고 좋은 확률 모델이 아니다. **calibration**이 핵심이다.
- 실험 기록 시 항상 함께 남긴다: Brier, 전체 BSS, R/F별 BSS, 월별 Brier, prediction mean.
- **Public = 전체 test 100%, Private = 대회 종료 시점의 Public과 동일.** shake-up도, 최종 제출 2개 선택도 없다.
- 수료 기준: Phase 2에서 **549.51 이상** (평가 탭 기재, 최종 확인 권장)

## B3. 제출 파일 형식

| 항목 | 규칙 |
| --- | --- |
| 출력 경로 | `output/submission.csv` (고정) |
| 컬럼 | `row_id`, `control_success` — **이 순서** |
| `row_id` | 평가 서버 `test.csv`와 정확히 일치. 행 수·순서 유지 |
| `control_success` | 0 이상 1 이하의 실수 (확률) |

- 배포본 `test.csv` / `sample_submission.csv`는 **형식 확인용 5행 샘플**이다. 실제 평가 데이터(245,789행)는 비공개이며 서버에서 동일 경로·동일 컬럼 구조로 교체된다.
- 따라서 행 수를 5로 가정하는 코드, 샘플에 맞춘 하드코딩은 전부 금지.
- 제출 전 확인: 확률의 finite 여부, `[0, 1]` 범위, 행 수, `row_id` 순서.

## B4. ZIP 패키징 규칙

```text
submit_NNN.zip
├── model/            ← 학습된 가중치·lookup 전부
├── script.py         ← 추론 전용
└── requirements.txt
```

- ZIP **루트에는 위 3개만** 둔다. "구조를 엄격히 준수" 조항이며, 추가 최상위 폴더/파일이 있으면 설치 오류가 난다.
- 저장 규칙: `submit/<날짜>/submit_NNN.zip` (예: `submit/2026-08-13/submit_021.zip`)
- **기존 제출물을 덮어쓰지 않는다.** 새 날짜 폴더 + 다음 제출 번호로 `OUTPUT_DIR`, `SPECS`를 바꾼 뒤 빌드한다.
- 파일명에 **공백·한글 금지**. (길이 30자 미만은 규정 명시 사항이 아니라 업로드 UI 제약으로 보인다 — `RULES.md` §9)
- ZIP CRC와 `model/` 산출물 누락 여부를 확인한다. 빈 `model/` 금지.

### script.py 요건

- 파일명 정확히 **`script.py`** (`Script.py`, `SCRIPT.PY` 불가)
- **추론 전용.** 학습 루프, 전처리 fit, 가중치 업데이트 코드 포함 금지
- 경로는 반드시 이 형태로 잡는다:

```python
from pathlib import Path
BASE = Path(__file__).resolve().parent
MODEL_DIR  = BASE / "model"
DATA_DIR   = BASE / "data"      # 읽기 전용
OUTPUT_DIR = BASE / "output"
```

- CWD 상대경로(`model/model.pt`)나 절대경로(`/app/model/model.pt`) 하드코딩 금지.

### requirements.txt

- 버전 고정 권장 (`xgboost==3.1.1` 형태)
- **서버 기본 설치 패키지는 가급적 넣지 않는다** — 이미 버전 호환이 맞춰져 있다.
- 오프라인 설치 불가능한 패키지 금지.
- 서버 기본값 일부: `torch==2.7.1+cu128`, `pandas==2.0.3`, `numpy==1.26.4`, `scipy==1.15.3`, `scikit-learn==1.8.0`, `joblib==1.5.3`, `transformers==4.46.3`

## B5. 평가 서버 환경 · 제출 횟수

```text
OS       : Ubuntu 22.04.5 LTS
Python   : 3.11.15
CPU/RAM  : 6 vCPU / 28GB
GPU      : NVIDIA L4 22.4GiB
설치 제한 : 10분
추론 제한 : 10분 / 245,789행
인터넷    : 패키지 설치 이후 불가
ZIP      : 압축 10GB 이하, 해제 32GB 이하
입력     : data/ 읽기 전용
출력     : output/submission.csv
```

- 인터넷이 없으므로 추론 중 다운로드(모델 허브, 폰트, 토크나이저 등)를 시도하는 코드는 실패한다. 필요한 자산은 전부 `model/`에 동봉한다.
- 로컬 개발 환경(RTX 4090)과 서버(L4 22.4GB)는 다르다. 메모리 여유를 가정하지 않는다.

**1일 제출 5회 제한.** 차감 규칙이 다르니 구분한다.

| 유형 | 예 | 차감 |
| --- | --- | --- |
| 설치 오류 | ZIP 구조 불일치, `requirements.txt` 설치 실패/시간 초과 | **없음** |
| 제출 오류 | `script.py` 런타임 에러, `submission.csv` 미생성, 추론 시간 초과 | **차감됨** |

로컬 smoke test를 건너뛰고 올리면 **횟수를 그냥 버린다.** 하루 5회는 금방 없어진다.

## B6. 누수 금지 — 시간 규칙

- **Val2023**: 2019~2022 학습 → 2023 검증
- **Val2024**: 2019~2023 학습 → 2024 검증
- **최종 2025 추론**: 설정 확정 후 2019~2024 전체로 재학습
- Target 집계, 임베딩, 클러스터, calibration, blend weight는 **검증 시점 이전 데이터 / 순방향 OOF만** 사용한다.
- TrackMan은 **투구 이후 측정값**이라 해당 투구의 피처가 될 수 없다. 예측 시즌 `S`에 대해 `season < S`만 허용 — Val2024 → 2019~2023, 최종 2025 → 2019~2024.
- **2025년 TrackMan 데이터는 사용 금지.**
- TrackMan 시점 게이트는 **매 제출마다** 확인한다 (`RULES.md` §10-D).
- Stateful 피처는 fold 안에서만 `fit`, validation/test에는 `transform`만.
- 같은 선수의 미래 시즌이나 validation Target으로 만든 profile/cluster를 validation에 연결하지 않는다.
- 실패 유형 라벨처럼 **투구 결과에서 파생되는 라벨**은 생성 경로가 투구 이전 정보만 쓰는지 확정하기 전에는 연결하지 않는다 (`RULES.md` §10-E).

## B7. 누수 금지 — test 행 독립성

평가 데이터의 각 행은 **독립적으로** 예측해야 한다. 판정 기준은 하나다.

> **단일 행에 대한 예측은, 그 행이 단독으로 존재하든 전체 test 데이터셋 안에 있든 동일해야 한다.**

8/13 재공지가 위반 사례로 명시한 것:

- test.csv의 다른 행을 이용한 **누적 통계**
- test 행들로부터 **rolling / lag 피처**
- test 데이터의 **분포·평균·순위**를 이용한 예측값 조정
- test 행을 가로지르는 **선수·팀·월·경기 단위 집계**

허용: 운영 측이 제공한 `asof_*` 컬럼(투구 직전까지의 과거 기록으로 계산된 공식 입력 피처), `model/` 안 학습 기반 lookup의 `merge`, 같은 행에 대한 seed/모델 평균, 학습 데이터에서 사전 결정된 상수 오프셋.

### ★ 상수 오프셋 — "전 행 동일 적용"은 충분조건이 아니다

`season_logit_offset` 같은 상수 보정값은 **값의 출처가 학습 데이터일 때만** 허용된다. 모든 행에 똑같이 더한다는 사실은 필요조건일 뿐이다.

| 상수의 출처 | 전 행 동일 적용 | 판정 |
| --- | --- | --- |
| 2019~2024 추세 외삽 | ✔ | ✅ 정상 |
| 리더보드 점수를 보고 조정 | ✔ | ❌ 위반 |
| offset 후보를 여러 개 제출해 최고점 선택 | ✔ | ❌ 위반 |
| test 예측값 평균에 맞춰 결정 | ✔ | ❌ 위반 |

아래 세 경우 모두 전 행 동일 적용을 만족하지만 위반이다. 리더보드를 경유했을 뿐 결과적으로 전체 평가 데이터의 평균을 이용한 보정이기 때문이다. 상세는 [`RULES.md`](cowork/RULES.md) §2·§10-A.

## B8. 사용 금지 정보

- 현재 투구 **이후에 확정되는** 모든 정보 — 실제 위치·코스, 판정·결과·제구 성공 여부, 실제 구종, TrackMan 측정값
- 2025년 TrackMan 데이터
- **외부 데이터 전면 금지** — KBO 공식 기록, Statcast, 크롤링, 선수 신상 DB 전부 불가
- **외부 API 금지** — OpenAI, Gemini 등 원격 서버 호출. 모든 작업은 로컬에서 재현 가능해야 한다
- 비공개·제한 라이선스 사전학습 가중치 (MIT/Apache 2.0 등 공개 라이선스만 허용, 가중치에 걸리는 조항)
- 평가 데이터 내부 다른 행으로 만든 누적·빈도·분포·rolling·target encoding 피처

사용 가능한 것은 `train.csv`, 평가 환경의 `test.csv`(자기 행), 2019~2024 `trackman_history.csv`뿐이다.

## B9. 마감 일정 — 4개다

| 날짜 | 마감 | 놓치면 |
| --- | --- | --- |
| **08.26** | 팀 병합 | 이후 팀 구성 변경 불가 |
| **09.01** | 리더보드 제출 | 점수 확정 |
| **09.07** | 코드 및 PPT 제출 | Phase 3 진출 불가 |
| **09.11** | 코드 검증 | 검증 실패 시 진출 불가 |

09.01과 09.07은 **별개**다. 리더보드 점수가 높아도 09.07 제출과 09.11 검증을 통과하지 못하면 Phase 3에 못 간다.

09.07 제출물 4종: 학습 코드 개발 환경·라이브러리 버전 / **Private Score 재현용 학습 코드** / 솔루션 PPT / 팀원별 Phase 3 참가 여부. 확장자는 `.py`·`.ipynb`, 인코딩 **UTF-8**.

> 추론 `script.py`만으로는 부족하다. **`model/` 산출물을 raw data에서 처음부터 만들어내는 학습 파이프라인 전체**가 필요하다. 09.11 검증은 이걸 본다. 리더보드 마감(09.01) 전에 착수한다 (`RULES.md` §10-F).

## B10. 실격 사유

- 코드 제출 기능을 악용한 **의도적 평가 데이터셋 유출** (에러 메시지·출력에 test 값을 실어 나르는 행위 포함) → 즉시 실격
- 동일인의 개인/복수 팀 중복 등록
- 그 외 B6~B8 위반 전반

## B11. 제출 전 체크리스트

B1의 6단계를 마친 뒤 마지막으로 훑는다. `RULES.md` 부록의 규정 체크리스트와 **둘 다** 통과해야 한다.

**규정**

- [ ] B1 6단계를 전부 밟았고 결과를 `SUBMISSION_LOG.md`에 기록했는가
- [ ] 행 독립성 기계 검증(2단계)을 통과했는가
- [ ] 금지 패턴 스캔(3단계) 히트를 한 줄씩 판정했는가
- [ ] `SUBMISSION_LOG.md`에 **「season_logit_offset 출처 명시」 문구를 기재**했는가 (B1 5단계)
- [ ] 상수 보정값의 출처가 학습 데이터임을 문서로 설명할 수 있는가
- [ ] 리더보드 점수를 보고 조정한 값이 하나도 없는가
- [ ] 외부 데이터·외부 API·네트워크 호출이 없는가
- [ ] `script.py`에 학습 코드가 없는가
- [ ] TrackMan 시점 게이트를 통과했는가

**패키지**

- [ ] ZIP 루트가 `model/`, `script.py`, `requirements.txt` 3개뿐인가
- [ ] 파일명에 공백·한글이 없고, 경로가 `submit/<날짜>/submit_NNN.zip`인가
- [ ] 기존 제출물을 덮어쓰지 않았는가 (새 번호·새 폴더)
- [ ] `script.py`가 `Path(__file__).resolve().parent` 기준 경로를 쓰는가
- [ ] `model/`이 비어 있지 않고 필요한 자산이 전부 들어갔는가
- [ ] 오프라인 상태에서 smoke test를 통과했는가
- [ ] 245,789행 환산 추론 시간이 10분 이내인가

**출력**

- [ ] `output/submission.csv`에 `row_id`, `control_success` 순서로 저장되는가
- [ ] 예측값이 모두 finite이고 `[0, 1]` 범위인가
- [ ] 행 수와 `row_id` 순서가 입력 `test.csv`와 일치하는가

**기록**

- [ ] Brier / 전체 BSS / R·F별 BSS / 월별 Brier / prediction mean을 기록했는가
- [ ] 오늘 제출 횟수가 5회 이내인가
- [ ] `task.jsonl`에 `type: "exp"` 항목으로 기록했는가 (A3 참고)
