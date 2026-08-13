# LA9 elephant miracle

LG Aimers 9기 Phase 2 — **투구 제구 성공 확률 예측**

각 투구가 제구에 성공할 확률을 0~1 실수로 예측한다. 지표는 Brier Skill Score.

## 문서

| 파일 | 무엇인가 | 언제 |
| --- | --- | --- |
| [`AGENTS.md`](AGENTS.md) | 협업 규칙 + 제출 작업 절차 | 세션 시작할 때 |
| [`cowork/plan.md`](cowork/plan.md) | 지금 상태 + 다음 할 일 | 작업 고를 때 |
| [`cowork/task.jsonl`](cowork/task.jsonl) | 누가 언제 뭘 했는지 (append-only) | 작업 끝날 때마다 |
| [`cowork/RULES.md`](cowork/RULES.md) | 대회 규정 원본 | 제출본 만들 때 |
| [`data_description.md`](data_description.md) | 데이터 컬럼 설명 | 피처 만들 때 |

`task.jsonl`은 계속 쌓는 일기장, `plan.md`는 그걸 읽고 리뷰 시점에 다시 쓰는 요약본이다.

## 폴더

```text
cowork/
├── RULES.md  plan.md  task.jsonl   ← 공용, PR로만 수정
└── sj/ ye/ cw/ yn/ hw/             ← 각자 폴더, 자유 푸시
```

**내 폴더는 자유, 나머지는 PR.** 이게 협업 규칙의 전부다.

## 작업 흐름

```bash
git checkout -b sj/pitcher-prior          # <initial>/<short-desc>
# cowork/sj/ 안에서 작업
git commit -m "[sj] feat: add pitcher prior"   # [<initial>] <type>: <설명>
git push origin sj/pitcher-prior
```

`type`은 `feat` · `fix` · `exp` · `doc` · `chore`.

작업이 끝나면 `cowork/task.jsonl`에 한 줄 추가한다. **JSON 배열이 아니라 JSONL이다** — 한 줄에 항목 하나, `>>`로 파일 끝에 붙인다. 그래야 동시에 기록해도 git이 합쳐준다. 기존 줄은 고치지 않는다.

```bash
echo '{"id":"2026-08-13T14:32:10Z-sj","author":"sj","ts":"2026-08-13T14:32:10+09:00","type":"exp","title":"투수 prior 피처","detail":"2023 이전만으로 shrinkage prior","paths":["cowork/sj/feat_prior.py"],"result":"Val2024 Brier 0.2431 / BSS 836.5","next":"타자손 교호작용"}' >> cowork/task.jsonl
```

`id`는 타임스탬프 기반으로. 스키마는 [`AGENTS.md`](AGENTS.md) A3.

## 실험할 때 어기면 안 되는 것

- **미래를 보지 않는다.** Val2024는 2019~2023만 학습. TrackMan도 예측 시즌 이전만. 2025 TrackMan 금지.
- **test의 다른 행을 보지 않는다.** `predict(단독 행) == predict(전체)[i]`가 성립해야 한다.
- **확률로 낸다.** AUC 말고 calibration을 본다.
- **결과는 항상 같이 적는다.** Brier, 전체 BSS, R/F별 BSS, 월별 Brier, prediction mean.

## 제출할 때

ZIP 루트에 `model/` · `script.py`(추론 전용) · `requirements.txt` 세 개만. 결과는 `output/submission.csv`에 `row_id`, `control_success` 순서로.

서버는 **인터넷 없음**, 추론 **10분 / 245,789행**, 제출 **1일 5회**.

**제출 패키지를 만들 때마다 [`AGENTS.md`](AGENTS.md) B1의 6단계 점검을 처음부터 밟는다.** "지난번에 통과했으니 괜찮다"는 근거가 아니다.

> RULES.md 재독 → 행 독립성 기계 검증 → 금지 패턴 스캔 → 오프라인 smoke test → 근거 문서화 → 체크리스트 2개 통과

결과는 `submit/<날짜>/SUBMISSION_LOG.md`에 남긴다. **09.11 코드 검증에서 사람이 코드를 읽는다.**

## 마감

| 08.26 | 09.01 | 09.07 | 09.11 |
| --- | --- | --- | --- |
| 팀 병합 | 리더보드 제출 | 코드 및 PPT | 코드 검증 |

09.01과 09.07은 별개다. 점수가 높아도 09.07·09.11을 통과하지 못하면 Phase 3에 못 간다. 09.07에는 `model/`을 raw data에서 재현하는 **학습 파이프라인 전체**가 필요하다.

## 자주 하는 실수

| 실수 | 왜 |
| --- | --- |
| `plan.md` 수시 수정 | 요약본이 로그가 된다. 리뷰 때만 재작성 |
| `task.jsonl` 기존 줄 수정 / pretty-print | append-only 위반, 파싱 실패 |
| 리더보드 보고 상수 재조정 | "전체 평가 데이터 평균 이용 보정"이라 위반 |
| 안 쓰는 분기라고 위험 코드 방치 | 09.11에서 사람이 읽는다. 죽은 코드도 지운다 |
| test 5행 샘플에 맞춘 하드코딩 | 실제 평가는 245,789행 |
| `script.py` 절대경로 하드코딩 | `Path(__file__).resolve().parent` 사용 |
| `data/` 원본 커밋 | 690MB다. `.gitignore`에 있으니 강제 add 금지 |
