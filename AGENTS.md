# AGENTS.md

LG Aimers 9기 Phase 2 · 투구 제구 성공 확률 예측

세션을 시작하는 팀원과 에이전트는 이 파일을 먼저 읽는다.
**Part A는 협업 규칙**, **Part B는 제출 규칙**이다. 전체 개요는 [`README.md`](README.md), 데이터 컬럼 설명은 [`data_description.md`](data_description.md), 현재 상태와 다음 할 일은 [`cowork/plan.md`](cowork/plan.md)를 본다.

---

# Part A. 협업 규칙

## A1. 폴더 구조

```text
cowork/
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
| `cowork/plan.md`, `cowork/task.jsonl` | **PR 필수** |
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

## B1. 평가 지표

공식 지표는 **Brier Skill Score**다.

```text
Brier      = mean((prediction - target)^2)
null_brier = target_mean * (1 - target_mean)
BSS        = max(0, 100000 * (1 - Brier / null_brier))
```

- 예측값은 확률이다. 0/1 하드 라벨을 내지 않는다.
- 모델 비교는 0으로 잘리는 BSS보다 **Brier / normalized Brier**를 우선한다.
- AUC나 0.5 임계값 F1이 높다고 좋은 확률 모델이 아니다. **calibration**이 핵심이다.
- 실험 기록 시 항상 함께 남긴다: Brier, 전체 BSS, R/F별 BSS, 월별 Brier, prediction mean.

## B2. 제출 파일 형식

| 항목 | 규칙 |
| --- | --- |
| 출력 경로 | `output/submission.csv` (고정) |
| 컬럼 | `row_id`, `control_success` — **이 순서** |
| `row_id` | 평가 서버 `test.csv`와 정확히 일치. 행 수·순서 유지 |
| `control_success` | 0 이상 1 이하의 실수 (확률) |

- 배포본 `test.csv` / `sample_submission.csv`는 **형식 확인용 5행 샘플**이다. 실제 평가 데이터는 비공개이며 서버에서 동일 경로·동일 컬럼 구조로 교체된다.
- 따라서 행 수를 5로 가정하는 코드, 샘플에 맞춘 하드코딩은 전부 금지.
- 제출 전 확인: 확률의 finite 여부, `[0, 1]` 범위, 행 수, `row_id` 순서.

## B3. ZIP 패키징 규칙

```text
submit_NNN.zip
├── model/            ← 학습된 가중치/아티팩트
├── script.py         ← 추론 진입점
└── requirements.txt  ← 설치 패키지
```

- ZIP **루트에는 위 3개만** 둔다. 추가 최상위 폴더를 만들지 않는다.
- 파일명은 **30자 미만**.
- 저장 규칙: `submit/<날짜>/submit_NNN.zip` (예: `submit/2026-08-13/submit_021.zip`)
- **기존 제출물을 덮어쓰지 않는다.** 새 날짜 폴더 + 다음 제출 번호로 `OUTPUT_DIR`, `SPECS`를 바꾼 뒤 빌드한다.
- 제출마다 `submit/<날짜>/SUBMISSION_LOG.md`에 설계·검증 수치·파일 해시를 남긴다.
- ZIP CRC와 모델 파일 누락 여부를 확인한다.

### script.py 경로 규칙

```python
from pathlib import Path
BASE = Path(__file__).resolve().parent
MODEL_DIR  = BASE / "model"
DATA_DIR   = BASE / "data"      # 읽기 전용
OUTPUT_DIR = BASE / "output"
```

- 반드시 `Path(__file__).resolve().parent` 기준으로 경로를 잡는다.
- CWD 상대경로(`model/model.pt`)나 절대경로(`/app/model/model.pt`) 하드코딩 금지.
- 제출 전 **서버와 동일한 경로 구조로 smoke test**를 돌린다.

## B4. 평가 서버 환경 제약

```text
OS       : Ubuntu 22.04.5 LTS
Python   : 3.11.15
CPU/RAM  : 6 vCPU / 28GB
GPU      : NVIDIA L4 22.4GiB, CUDA 12.8
설치 제한 : 10분
추론 제한 : 10분 / 245,789행
인터넷    : 패키지 설치 이후 불가
ZIP      : 압축 10GB 이하, 해제 32GB 이하
입력     : data/ 읽기 전용
출력     : output/submission.csv
```

- 인터넷이 없으므로 추론 중 다운로드(모델 허브, 폰트, 토크나이저 등)를 시도하는 코드는 실패한다. 필요한 자산은 전부 `model/`에 동봉한다.
- 245,789행을 10분 안에 처리해야 한다. 행 단위 파이썬 루프, 무거운 재학습, 대용량 재집계는 추론 단계에 두지 않는다.
- 로컬 개발 환경(RTX 4090)과 서버(L4 22.4GB)는 다르다. 메모리 여유를 가정하지 않는다.

## B5. 누수 금지 — 시간 규칙

- **Val2023**: 2019~2022 학습 → 2023 검증
- **Val2024**: 2019~2023 학습 → 2024 검증
- **최종 2025 추론**: 설정 확정 후 2019~2024 전체로 재학습
- Target 집계, 임베딩, 클러스터, calibration, blend weight는 **검증 시점 이전 데이터 / 순방향 OOF만** 사용한다.
- TrackMan은 예측 시즌 `S`에 대해 `season < S`만 허용. Val2024 → 2019~2023, 최종 2025 → 2019~2024.
- **2025년 TrackMan 데이터는 사용 금지.**
- Stateful 피처는 fold 안에서만 `fit`, validation/test에는 `transform`만.
- 같은 선수의 미래 시즌이나 validation Target으로 만든 profile/cluster를 validation에 연결하지 않는다.

## B6. 누수 금지 — test 행 독립성

평가 데이터의 각 행은 **독립적으로** 예측해야 한다. 서버에서 `test.csv` 전체가 주어져도 다른 행을 참조해 피처를 만들 수 없다.

금지:

- `test.csv` 내부 행 기반 선수별·팀별·월별 누적 통계
- `test.csv` 내부 빈도값 또는 분포 통계
- `test.csv` 내부 target encoding
- `test.csv` 행 순서 기반 rolling / expanding feature
- 평가 데이터 전체를 보고 만든 사후 보정값

허용: 운영 측이 제공한 `asof_*` 컬럼. 각 행의 투구 직전 시점까지의 과거 기록으로만 계산된 공식 입력 피처다.

## B7. 사용 금지 정보

- 현재 투구 **이후에 확정되는** 모든 정보
- 현재 투구의 실제 위치·코스
- 현재 투구의 실제 판정·결과·제구 성공 여부
- 현재 투구의 실제 구종
- 현재 투구의 TrackMan 측정값
- 2025년 TrackMan 데이터
- 평가 데이터 내부 다른 행으로 만든 누적·빈도·분포·rolling·target encoding 피처

사용 가능한 데이터는 `train.csv`, 평가 환경의 `test.csv`(자기 행), 2019~2024 `trackman_history.csv`, 그리고 대회 규칙상 허용되는 외부 데이터뿐이다.

## B8. 제출 전 체크리스트

- [ ] `output/submission.csv`에 `row_id`, `control_success` 순서로 저장되는가
- [ ] 예측값이 모두 finite이고 `[0, 1]` 범위인가
- [ ] 행 수와 `row_id` 순서가 입력 `test.csv`와 일치하는가
- [ ] ZIP 루트가 `model/`, `script.py`, `requirements.txt` 3개뿐인가
- [ ] 파일명 30자 미만, 경로가 `submit/<날짜>/submit_NNN.zip`인가
- [ ] 기존 제출물을 덮어쓰지 않았는가 (새 번호·새 폴더)
- [ ] `script.py`가 `Path(__file__).resolve().parent` 기준 경로를 쓰는가
- [ ] 추론 중 인터넷 접근이 없는가, 필요한 자산이 `model/`에 전부 있는가
- [ ] 서버와 동일한 경로 구조로 smoke test를 통과했는가
- [ ] 시간 규칙·TrackMan 시즌 제한을 위반하지 않았는가
- [ ] test 행 간 집계가 없는가
- [ ] Brier / 전체 BSS / R·F별 BSS / 월별 Brier / prediction mean을 기록했는가
- [ ] `submit/<날짜>/SUBMISSION_LOG.md`를 작성했는가
- [ ] `task.jsonl`에 `type: "exp"` 항목으로 기록했는가 (A3 참고)
