# `final_submissions/` 협업 규약

`performance_tracking/` 은 **실험 등록부**고, `final_submissions/` 는 **대회에 실제로
올릴 후보**만 놓는 곳이다. 둘을 섞지 않는다.

    performance_tracking/    실험한 모든 것 (기각된 것 포함). 판정 근거가 산다
    final_submissions/       올릴 것만. 팀 합의된 후보

---

## 1. 현재 상태

```text
final_submissions/
├── README.md            (비어 있음)
└── cw/
    └── submit_v17_base.zip
```

**`<멤버>/<zip>` 구조**를 이미 cw 가 만들어뒀다. 이 규약을 따른다.

---

## 2. 폴더 규약

```text
final_submissions/
├── README.md                    <- 후보 목록과 현재 1순위. 올릴 때마다 갱신
├── cw/  <name>.zip
├── sj/  <name>.zip
├── hw/  <name>.zip
├── yn/  <name>.zip
└── team/                        <- 2인 이상 결합본
    └── <name>.zip
```

**규칙**

1. **이름은 `performance_tracking/models/<name>.zip` 과 글자 하나까지 같게** 쓴다.
   등록부에 없는 zip 은 올리지 않는다 — 근거 없는 제출본을 막기 위해서다.
2. zip 안은 `script.py` + `requirements.txt` + `model/` **셋뿐**이다 (`cowork/RULES.md` §5).
   올리기 전 `cowork/sj/sj_final/src/check_submit.py` 를 **반드시** 돌린다.
3. **남의 폴더를 건드리지 않는다.** 결합본에서 남의 모듈을 담을 때는
   **바이트 그대로** 옮기고 md5 로 확인한다 (`build_submit_zip.py` 가 한다).
4. 제출해서 점수가 나오면 **`results.csv` 의 `public` 열**에 즉시 적는다.
   점수 없는 후보와 있는 후보를 눈으로 구분할 수 있어야 한다.

---

## 3. 결합본을 만들 때

챔피언 `script.py` 는 이미 **모듈 일반형**이다. 멤버를 늘리는 데 코드 수정이 필요 없다.

```text
script.py                     결합. p = r + Σ w_m (p_m − r) − center_shift
model/blend_weights.json      members / bucket_edges / buckets.<label>.w
model/<member>/script.py      각자 것 그대로
model/<member>/model/         각자 가중치
```

**가중을 정할 때 지킬 것 — 실측으로 확인된 것만 적는다.**

1. **합을 1 로 묶는다.** 자유 적합은 정직 분할에서 **−24.8**.
   제출3 이 합 1.143 으로 나가 Public **−6.7** 이었다.
2. **합이 1 이면 `center_shift` 는 0 이다.** shift 는 평균만 맞추고 분산은 못 되돌린다.
3. **적합 fold 와 평가 fold 를 분리한다.** 같은 fold 로 적합·평가하면 반드시 부풀려진다
   (way 계열평균 자체적합 +9.6 → 정직 +1.2).
4. **멤버 편입은 비음수 제약으로 판정한다.** `w* = M⁻¹A` 자유해는 음수 가중을 내고
   그건 전이되지 않는다. 비음수·합=1 에서 0 이 붙으면 그 멤버는 정보가 중복이다.
5. **비교 대상은 실제 결합 상대여야 한다.** `yn` 은 sj 의 cw 스택과 ρ 0.8949 지만
   실제 상대인 `sj3way` 와는 **0.9572** 다. 상대를 틀리면 결론이 뒤집힌다.

**정직한 가중 판정 계기** — `cowork/sj/sj_final/src/` 에 있다. 그대로 쓰면 된다.

    team_blend.py      멤버 조합 전수 + ρ 행렬 + 월전방분할 정직 평가
    final_team_w.py    실제 모듈 예측으로 w 격자, 정·역방향 양쪽
    bucket_blend.py    구간별 가중 12축 + 수축 λ
    set_blend.py       완성 zip 의 blend_weights.json 만 교체 (재학습 불필요)
    check_submit.py    RULES §5 규격 9항 자동 검사

---

## 4. val 예측을 반드시 남길 것 (규칙 3)

결합 가중을 정직하게 적합하려면 **모든 멤버의 val 예측**이 필요하다.
이게 없어서 오랫동안 팀 결합을 재적합하지 못했다.

```csv
row_id,pred
TRAIN_1221586,0.514991283673795
```

`val/<name>_2024.csv` · `val/<name>_2022.csv` 두 개.
**해당 시즌을 빼고 학습한 정직한 OOF** 여야 한다. 시즌을 포함해 학습한 예측을 내면
BSS 가 1500 대로 나오고 그걸로 적합한 가중은 전이되지 않는다
(실제로 그렇게 적합해 `cw −0.479 / sj 1.658` 이 나온 적이 있다).

**현재 공백**

| 멤버 | val2024 | val2022 |
|---|---|---|
| cw · sj | ✅ | ✅ |
| **hw** | ✅ | ❌ **없음** |
| **yn** | ✅ | ❌ **없음** |
| **ye** | ✅ | ✅ | (sj 가 `models/sj_stdmlp/ye_repro.py` 로 노트북에서 재현) |

val2022 가 없으면 **규칙 1 의 비하락 관문을 그 모델에 적용할 수 없다.**
`<=2021` 학습으로 만든 val2022 예측을 hw·yn 에게 요청해야 한다.

---

## 5. 마감 전 점검 (`cowork/RULES.md` §7)

| 항목 | 상태 |
|---|---|
| zip 규격 (`check_submit.py`) | ✅ 자동화됨 |
| 행 독립성 (§2) | ⚠️ **정적 검사로 판정 불가.** 코드를 읽어 확인할 것 |
| 학습 코드 제출 (09.11 검증) | ❌ **`yn_fa10c` 미충족** — `models/yn_fa10c/MISSING.md` 참조 |

`yn` 은 `model/` 63파일을 raw data 에서 재생성하는 **학습 스크립트가 저장소에 없다.**
팀 최종 패키지에 들어간다면 이건 반드시 채워야 한다.

---

## 6. 큰 파일 주의

GitHub 은 **100MB 를 넘는 파일을 거부**한다. 팀 결합본은 남의 모듈까지 담아
쉽게 300MB 를 넘는다 (sj 결합본 375MB).

- 단일 멤버 zip (6~54MB) 은 그대로 커밋해도 된다
- **결합본은 커밋하지 않는다.** 재조립 스크립트와 각 멤버 zip 만 남기면
  `build_submit_zip.py` 로 언제든 다시 만들 수 있다
- 꼭 저장소에 둬야 하면 Git LFS 를 먼저 설정한다 (`.gitattributes` 필요)
