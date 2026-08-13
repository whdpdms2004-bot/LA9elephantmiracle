# LA9 elephant miracle

LG Aimers 9기 Phase 2 — **투구 제구 성공 확률 예측**

각 투구가 제구에 성공할 확률(`control_success`)을 0~1 실수로 예측한다. 평가 지표는 Brier Skill Score.

처음 온 사람은 이 문서만 읽으면 오늘 작업을 시작할 수 있다.
상세 규칙은 [`AGENTS.md`](AGENTS.md), 데이터 컬럼 설명은 [`data_description.md`](data_description.md)에 있다.

---

## 1. 문서가 셋인 이유

| 파일 | 무엇인가 | 언제 쓰나 |
| --- | --- | --- |
| [`cowork/task.jsonl`](cowork/task.jsonl) | **원본 로그.** 누가 언제 뭘 했는지 계속 쌓인다 | 작업할 때마다 |
| [`cowork/plan.md`](cowork/plan.md) | **현재 요약.** 지금 상태 + 다음 할 일 | 리뷰할 때만 |
| [`AGENTS.md`](AGENTS.md) | **규칙.** 협업 방식 + 제출 작업 절차 | 세션 시작할 때 |
| [`cowork/RULES.md`](cowork/RULES.md) | **대회 규정 원본.** 공식 페이지 + 재공지 정리 | 제출본 만들 때 |

`task.jsonl`은 지우지 않고 계속 쌓는 일기장, `plan.md`는 그 일기장을 읽고 다시 쓰는 요약본이라고 보면 된다.

## 2. 폴더 구조

```text
LA9elephantmiracle/
├── README.md            ← 지금 이 문서
├── AGENTS.md            ← 규칙 (Part A 협업 / Part B 제출)
├── data_description.md  ← 대회 데이터 설명서
├── data/                ← 원본 데이터 (대용량 파일은 .gitignore)
├── cowork/
│   ├── RULES.md         ← 대회 규정 원본 (PR로만 수정)
│   ├── plan.md          ← 공용 (PR로만 수정)
│   ├── task.jsonl       ← 공용 (PR로만 수정)
│   └── sj/ ye/ cw/ yn/ hw/   ← 각자 폴더 (자유 푸시)
├── final_submissions/
├── meeting_docs/
└── performance_tracking/
```

**내 폴더는 자유, 나머지는 PR.** 이 한 줄이 협업 규칙의 전부다.

## 3. 작업 흐름

```bash
git checkout -b sj/pitcher-prior      # 1. 내 이니셜로 브랜치
# 2. cowork/sj/ 안에서 작업
git commit -m "[sj] feat: add pitcher prior feature"
git push origin sj/pitcher-prior      # 3. 내 폴더는 바로 푸시 OK
```

작업이 끝나면 `cowork/task.jsonl`에 기록을 추가한다. 이건 공용 파일이라 **PR로** 올린다.

**`task.jsonl`은 JSON 배열이 아니라 JSONL이다.** 한 줄에 항목 하나, 파일 끝에 새 줄만 붙인다. 이렇게 해야 여러 명이 동시에 기록해도 git이 알아서 합쳐준다.

```bash
echo '{"id":"2026-08-13T14:32:10Z-sj","author":"sj","ts":"2026-08-13T14:32:10+09:00","type":"exp","title":"투수 과거 성공률 prior 피처 추가","detail":"2023 이전 데이터만으로 shrinkage prior 계산","paths":["cowork/sj/feat_pitcher_prior.py"],"result":"Val2024 Brier 0.2431 / BSS 836.5","next":"타자손 교호작용 붙여보기"}' >> cowork/task.jsonl
```

필드는 `id` · `author` · `ts` · `type` · `title` · `detail` · `paths`가 필수, `result` · `next`는 선택. 전체 스키마는 [`AGENTS.md`](AGENTS.md) A3에 있다.

- **한 항목은 반드시 한 줄.** 보기 좋게 여러 줄로 펼치지 않는다.
- `id`는 **타임스탬프 기반**으로 만든다. `t001`, `t002` 같은 순차번호는 동시에 추가할 때 겹친다.
- 기존 줄은 **절대 고치거나 지우지 않는다.** `>>`로 추가만 하고 `>`는 쓰지 않는다.
- 충돌이 나면 한쪽을 버리지 말고 **둘 다 살린다.** 줄 순서는 상관없다.

## 4. 커밋 · 브랜치 규칙

```text
브랜치   sj/cache-invalidation            <initial>/<short-desc>
커밋     [sj] feat: add cache TTL config   [<initial>] <type>: <설명>
```

`type`은 다섯 가지: `feat`(기능) · `fix`(수정) · `exp`(실험) · `doc`(문서) · `chore`(잡일)

## 5. plan.md는 함부로 고치지 않는다

리뷰하는 시점에만 `task.jsonl`을 처음부터 훑어서 다시 쓴다. 다시 쓸 때 맨 위에 어디까지 반영했는지 남긴다.

```markdown
reviewed against task.jsonl up to 2026-08-13T14:32:10Z-sj
```

## 6. 데이터

`data/` 아래 네 개 파일을 쓴다. 컬럼별 의미는 [`data_description.md`](data_description.md) 참고.

| 파일 | 내용 | git |
| --- | --- | --- |
| `train.csv` | 학습 입력 + 정답 (1,475,092행 × 49컬럼) | ignore (368MB) |
| `trackman_history.csv` | 2019~2024 Trackman 로그 (1,793,078행 × 30컬럼) | ignore (354MB) |
| `test.csv` | 평가 입력 — **형식 확인용 5행 샘플** | 커밋 |
| `sample_submission.csv` | 제출 양식 — 5행 샘플 | 커밋 |

`test.csv`와 `sample_submission.csv`는 실제 평가 데이터가 아니다. 서버에서 **245,789행**짜리 실제 데이터로 교체된다.

## 7. 실험할 때 절대 어기면 안 되는 것

대회 실격이나 리더보드 오류로 직결되는 항목이다. 자세한 건 [`AGENTS.md`](AGENTS.md) Part B.

- **미래를 보지 않는다.** Val2024는 2019~2023만 학습. TrackMan도 예측 시즌 이전 것만. 2025 TrackMan은 사용 금지.
- **test의 다른 행을 보지 않는다.** 평가 데이터 각 행은 독립적으로 예측해야 한다. test 내부 누적 통계·빈도·target encoding·rolling 전부 금지.
- **확률로 낸다.** 0/1 라벨이 아니라 0~1 실수. AUC 말고 **calibration**을 본다.
- **결과는 항상 같이 적는다.** Brier, 전체 BSS, R/F별 BSS, 월별 Brier, prediction mean.

## 8. 제출할 때 — 규정 점검은 건너뛸 수 없다

ZIP 루트에 딱 세 개만 넣는다.

```text
submit_NNN.zip
├── model/            # 학습된 가중치·lookup 전부
├── script.py         # 추론 전용 (학습 코드 금지)
└── requirements.txt
```

- 저장 위치: `submit/<날짜>/submit_NNN.zip` — **기존 파일을 덮어쓰지 말고 새 번호로.**
- 결과는 `output/submission.csv`에 `row_id`, `control_success` 순서로.
- 평가 서버는 **인터넷이 없고** 추론 10분 제한(245,789행)이다. 필요한 자산은 전부 `model/`에 넣는다.
- **1일 제출 5회 제한.** smoke test 없이 올려서 런타임 에러가 나면 횟수를 그냥 버린다.

**제출 패키지를 만들 때마다 [`AGENTS.md`](AGENTS.md) B1의 6단계 점검을 처음부터 끝까지 밟는다.** "지난번에 통과했으니 괜찮다"는 근거가 아니다.

1. [`cowork/RULES.md`](cowork/RULES.md) 전문 재독
2. 행 독립성 기계 검증 — `predict(단독 행) == predict(전체)[i]`
3. `script.py` 금지 패턴 전수 스캔 (groupby·rolling·통계량 보정·네트워크·학습 코드·절대경로)
4. 패키지 구조 + 오프라인 smoke test
5. 근거 문서화 — 상수값의 출처가 학습 데이터임을 설명
6. 체크리스트 두 개(RULES.md 부록 + AGENTS.md B11) 통과

결과는 `submit/<날짜>/SUBMISSION_LOG.md`에 남긴다. **09.11 코드 검증에서 사람이 코드를 읽는다.** 기록이 없으면 방어할 수 없다.

## 9. 마감이 4개다

| 날짜 | 마감 |
| --- | --- |
| 08.26 | 팀 병합 |
| **09.01** | 리더보드 제출 |
| **09.07** | 코드 및 PPT 제출 |
| **09.11** | 코드 검증 |

09.01과 09.07은 별개다. 점수가 높아도 09.07 제출과 09.11 검증을 통과하지 못하면 Phase 3에 못 간다. 09.07에는 **`model/`을 raw data에서 재현하는 학습 파이프라인 전체**가 필요하다.

## 10. 자주 하는 실수

| 실수 | 왜 문제인가 |
| --- | --- |
| `plan.md`를 수시로 고침 | 요약본이 로그가 되어버린다. 리뷰 때만 재작성 |
| `task.jsonl` 기존 줄 수정 | append-only 원칙 위반. 이력이 사라진다 |
| `task.jsonl`에 pretty-print JSON | 한 줄 = 한 항목 규칙이 깨지고 파싱이 실패한다 |
| 남의 폴더 직접 푸시 | PR 없이는 금지 |
| test 5행 샘플에 맞춘 하드코딩 | 실제 평가는 245,789행이다 |
| `script.py`에 절대경로 하드코딩 | 서버 경로가 다르다. `Path(__file__).resolve().parent` 사용 |
| 리더보드 점수 보고 상수 재조정 | "전체 평가 데이터 평균을 이용한 보정"이라 8/13 재공지 위반이다 |
| 안 쓰는 분기라고 위험 코드 방치 | 09.11 검증에서 사람이 읽는다. 죽은 코드도 지운다 |
| `data/` 원본 커밋 | 두 파일 합쳐 690MB다. `.gitignore`에 있으니 강제 add 하지 않는다 |
