# cw / v17 — 168피처 · CatBoost(RMSE) + FT-Transformer + MLP

Public **983** (단독) · sj와 결합해 **1072** (팀 최고, 2026-08-25)

제출본: [`final_submissions/cw/submit_v17_base.zip`](../../../final_submissions/cw)

---

## 무엇이 v16(990)과 다른가

| | v16 | v17 |
| --- | --- | --- |
| CatBoost 손실 | Logloss | **RMSE (= Brier, 평가지표 그 자체)** |
| depth | 6 | **5** |
| 학습률 / 트리수 | 0.06 / 900 | **0.02 / 3000** |
| l2_leaf_reg | 6 | **10000** |
| MLP | batch 4096 · 15에폭 | **batch 1024 · 8에폭** |
| 시드 | 6 | 3 (시드 표준편차가 ±10.7 → ±2.9로 줄어 3으로 충분) |
| target_rate | 0.47353 (리더보드 역산) | **0.47469 (학습 데이터 외삽)** |
| 전역 배율 | 1.0721 (리더보드 역산) | **없음** |

마지막 두 줄은 점수를 2점 포기하고 넣은 것이다. 08-21 데이콘 재공지가 허용 출처를
네 가지로 못박았는데 리더보드 점수는 거기 없다. **v17에는 리더보드에서 나온 상수가
하나도 없다.**

CatBoost 재설계는 val에서 +6.5%였으나 리더보드는 990 → 983으로 내려갔다.
원인은 [§5 교훈](#5-교훈)에 적었다.

---

## 1. 피처 168개

```
원본 72      원본 컬럼 + 원핫 + 파생 + 플래툰 인코딩
시즌폼 8     asof_n x asof_rate - 전년말 통산 = 올 시즌 성적
TrackMan 55  구종별(직구/변화구/체인지업) 릴리스 일관성의 표준편차, 구종 간 차이
볼카운트 27  12종 개별 + 투수 볼성향과의 곱
역할 5       선발/불펜/스윙맨 (등판 이닝 분포)
결측표시 1   TrackMan 프로파일 유무
```

**시즌폼**이 가장 큰 이득이었다(+7.93% / +4.72%). `asof_pitcher_n`은 시즌을 넘어
통산으로 쌓이므로 통산 3,000구 투수가 올해 500구를 던져도 `asof_rate`가 거의 안
움직인다. 학습 데이터로 만든 "전년말 통산상태" 룩업을 빼서 올 시즌 성적만 복원한다.

**TrackMan**은 구종별로 따로 잰다(+3.3% / +2.0%). 릴리스 포인트는 구종마다 다른 게
정상이라 전체 표준편차를 재면 제구력이 아니라 구종 다양성을 재게 된다.

모든 룩업은 **직전 시즌까지**로만 만든다. s년 행은 s년 미만 데이터로 만든 룩업을 본다.

---

## 2. 실행 순서

```bash
# [1] 80피처 행렬 + 시즌폼 룩업          _work/X80.npy, model/season_lut.npz
python src/prep_v12.py

# [2] 도메인 3블록 추가 -> 168피처        _work/X168.npy
python src/build_v13.py --gpu

# [3] 추론용 룩업 생성                    model/domain_lut.npz
python src/build_luts_v13.py

# [4] 학습 (약 90분)
#     [A] val 2024·2022 에서 계열별 배율과 블렌드 가중치
#     [B] 2019~2024 전체 학습 + numpy 내보내기
#     [C] 2025 모사 분포에서 베이스율 상수 C0
python src/train_v13.py --gpu

# [5] 패키징 + 3관문
python src/make_v13.py --out submit_v17_base.zip
python src/timeit_v13.py
python src/check_rules.py submit_v17_base.zip
python src/verify_v13.py
```

`src/script_v13.py`가 제출 zip 안에 `script.py`로 들어간다. 평가 서버가 이걸 실행한다.

---

## 3. 세 모델

| | 무엇 | 왜 |
| --- | --- | --- |
| `cb` | CatBoost, RMSE, depth 5, 3000트리, l2 10000 | oblivious(대칭) 트리라 순수 numpy로 내보낼 수 있다. `catboost` 패키지 없이 추론된다 |
| `ft` | FT-Transformer, d_token 64, 3층 | 피처 하나를 토큰으로 보고 **한 행 안에서만** attention. 행 독립성 유지 |
| `mlp` | 잔차 MLP, width 384, 3층 | 빠르고 안정적 |

```
p = r + w_cb (p_cb - r) + w_ft (p_ft - r) + w_mlp (p_mlp - r)

r = 0.47469  (시즌별 성공률의 선형 외삽)
w = 0.5993 / 0.2902 / 0.1211   (val 2024·2022 에서 w* = M^-1 A, 두 해 평균)
```

`cb_export.py`가 CatBoost를 numpy 배열로 바꾼다. oblivious 트리는 깊이 d에
(피처, 임계값) d쌍과 리프값 2^d개뿐이라 이렇게 계산된다.

```
idx = sum_i 2^i * [ x[f_i] > border_i ]
```

RMSE로 학습하면 raw가 곧 확률이므로 시그모이드를 씌우면 안 된다. `cb_link` 플래그로
구분한다.

---

## 4. 하이퍼파라미터 출처

`tuning/tune_cb.py` — 5라운드, 40개 설정 × 2해 × **8시드**

```
                    val2024     val2022     min
기준                  800.1      2328.0       -
RMSE                 +2.68%      +1.17%   +1.17%
lr .02 x3000         +3.85%      +1.78%   +1.78%
+ l2 20              +5.49%      +2.73%   +2.73%
+ depth 5, l2 150    +6.38%      +4.91%   +4.91%
+ l2 10000           +7.92%      +6.52%   +6.52%   <- 채택 (t = 16.2 / 27.7)
  l2 30000           +5.80%      +6.41%   +5.80%   <- 꺾임
```

기각된 것도 남겨둔다. **시즌 가중(최근 시즌에 무게)은 다섯 강도 전부, 두 해 전부
마이너스**였다(-3% ~ -12%). 옛날 데이터가 방해하는 게 아니라 필요했다.
**depth를 키우면 무너진다**(6→8에서 -11.5%). **투수 주효과를 baseline 오프셋으로
빼는 것**도 -0.78%였다.

`tuning/tune_dl.py` — MLP `batch 4096→1024`, `epochs 15→8` (+5.97%).
셋 이상 조합하면 오히려 떨어졌다. 전부 같은 병(과적합)을 고치는 거라 겹친다.

**FT는 아직 튜닝하지 않았다.** MLP가 같은 방식으로 +6% 나왔으니 여지가 있다.

---

## 5. 교훈

### 개별 모델을 키우는 것과 앙상블을 키우는 것은 다른 일이다

CatBoost 단독을 +6.5% 올렸는데 리더보드는 990 → 983으로 내려갔다.

```
A (판별력)  +2.12%     실제로 좋아졌다
V (산포)    +4.81%     그보다 더 벌어졌다
A²/V        -0.50%     점수를 결정하는 양은 오히려 나빠졌다
```

상관행렬이 답을 갖고 있었다.

```
          cb-ft    cb-mlp
v16      0.9286   0.9200
v17      0.9565   0.9400      전부 상승
```

RMSE + 극단 정칙화로 만든 예측은 매끄럽고 눌려 있어 **신경망이 내놓는 것과 성격이
같아진다.** 구설정의 날카로운 예측이 앙상블에 넣어주던 다른 정보를 잃었다.

**모델을 바꿀 때는 단독 성능과 상관을 반드시 같이 본다.**

### 전이율이 축마다 다르다

```
피처 이득      val +5.3%  ->  실제 +3.26%   전이율 0.62
앙상블 이득    val +9.19% ->  실제 +9.05%   전이율 0.99
```

피처 이득은 "새 정보가 다음 해에도 재현되는가"라는 불확실성을 안고 있지만,
앙상블 이득은 **상관행렬의 기하학**에서 나온다. 상관은 라벨이 필요 없는 양이라
연도를 넘어도 크게 흔들리지 않는다.

### 3시드는 아무것도 판정하지 못한다

3시드에서 기준선이 2.2% 흔들려 승자를 통째로 뒤집은 적이 있다. 8시드로 다시 재니
"이겼다"던 세 블록이 전부 0이었다. 이후 모든 판정을 8시드로 고정했다.

### 신호가 있는 것과 가져올 수 있는 것은 다르다

잔차 스캔은 투수×카운트에 980점이 실재한다고 했다. 그런데 직전 시즌 룩업으로는
0을 얻었다. 투수는 해마다 변한다.

시도했다 실패한 룩업: 투수×카운트, 타자×카운트, 투수×주자, 투수×이닝,
팀(포수) 원시 성공률(**-12.5%**), 팀 잔차(-5.1%).

---

## 6. 규정

허용 출처는 넷뿐이다 (08-21 데이콘 재공지).

```
1. 그 행의 입력 변수
2. 그 행의 입력 변수만으로 만든 파생변수
3. 주최측 공식 학습 데이터
4. 공식 학습 데이터만으로 만든 통계·모델·파생변수
```

v17이 지키는 방식이다.

- 모든 룩업(`season_lut`, `domain_lut`, `encodings`)은 `train.csv`로만 만들고,
  추론 때는 그 행 자신의 `pitcher_id` / `batter_id`로 **조회만** 한다
- `target_rate`는 학습 데이터 시즌별 성공률의 선형 외삽이다
- 블렌드 가중치는 val 2024·2022 실제 라벨에서 구했다
- **리더보드에서 역산한 값은 하나도 없다**
- 신경망 추론은 fp32로 고정한다. fp16은 배치 크기에 따라 GEMM 타일링이 달라져
  같은 행이라도 값이 미세하게 바뀐다

`src/check_rules.py`가 공지 3번 기준을 실측한다 — test.csv에 1행만 있을 때와
전체가 있을 때의 예측값 비교. v17 실측 최대차 **6.19e-09** (부동소수점 오차).

---

## 7. 파일

```
src/
  common.py           원본 피처 · 플래툰 인코딩
  season_form.py      시즌폼 8피처 (학습·추론 공용 — 규칙이 어긋나면 즉시 망가진다)
  prep_v12.py         X80 행렬 + season_lut
  build_v13.py        도메인 3블록 -> X168
  build_luts_v13.py   추론용 domain_lut
  dl.py               FT-Transformer / MLP 정의
  cb_export.py        CatBoost -> 순수 numpy
  train_v13.py        3단계 학습
  script_v13.py       추론 (제출 zip 안의 script.py)
  make_v13.py         패키징
  check_rules.py      행 독립성
  verify_v13.py       베이스율
  timeit_v13.py       추론 시간
  lb_recalib.py       (v14~v16 에서 쓴 배율 산출. v17 에서는 미사용)
  apply_recalib.py    (--keep-target 으로 target_rate 는 유지)

tuning/
  tune_cb.py          CatBoost 5라운드
  tune_dl.py          MLP / FT 스윕

assets/
  pitcher_id_map2.csv TrackMan ID 매핑 (605명)
  params_v13.json     최종 상수

model/                제출 zip 안의 파일과 바이트 단위로 동일 (26 MB)
  cb.npz         1.34 MB   CatBoost 3시드, 순수 numpy (feature/border/leaf 배열)
  ft.pt          2.00 MB   FT-Transformer 3시드 state_dict
  mlp.pt        21.83 MB   MLP 3시드 state_dict
  prep.npz       0.06 MB   분위수 경계 (신경망 전처리)
  encodings.npz  0.04 MB   플래툰 스플릿 인코딩
  season_lut.npz 0.02 MB   2024년 말 통산상태 (시즌폼용)
  domain_lut.npz 0.26 MB   TrackMan 프로파일 + 역할 룩업
  params.json              r · 블렌드 가중치 · 계열별 캘리브레이션 상수
```

`.gitignore`가 `*.pt`(26줄)와 `*.npz`(34줄)를 막으므로 `git add -f`가 필요하다.

`model/` 만 있으면 **학습 없이 바로 추론된다.**

```bash
mkdir -p run/model run/data
cp src/script_v13.py run/script.py
cp model/* run/model/
cp <test.csv, sample_submission.csv> run/data/
cd run && python script.py        # -> output/submission.csv
```

`_work/` 중간 산출물(X80/X168 행렬, 1.5 GB)만 저장소에 없다.
다시 학습하려면 `§2 실행 순서`의 [1]~[2]로 재생성한다.
