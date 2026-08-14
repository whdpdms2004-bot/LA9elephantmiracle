# Control Success 협업용 피처 코드 규격

## 권장 구조

```text
features/
  common.py                       # 공통 계약·검사·조립 함수
  feat_jisu_count.py              # 팀원별 피처 모듈
  feat_minsu_pitcher_history.py
  feat_yuna_trackman.py
  registry.py                     # 채택된 모듈 목록
notebooks/
  feat_jisu_count_review.ipynb    # 검토·그래프용, 로직 원본 아님
feature_outputs/
  <feature_set>__manifest.json
  <feature_set>__feature_summary.csv
```

이 폴더의 `feature_example.py`에는 공통 함수와 예제 피처가 한 파일에 들어 있다. 팀 규격을 확정한 뒤에는 공통 함수만 `common.py`로 옮기고, 각 팀원 파일에는 `SPEC`, `PARAMS`, `build_features()`만 두면 된다.

## 한 모듈의 필수 계약

| 항목 | 규칙 |
|---|---|
| 입력 | 원본 데이터와 같은 행 단위 DataFrame |
| 출력 | `row_id + 해당 모듈의 신규 피처` |
| 행 | 수, `row_id` 값, 순서를 그대로 유지 |
| 이름 | 모든 피처가 고유 prefix로 시작 |
| dtype | 모델 투입 가능한 숫자형을 기본으로 사용 |
| target | 일반 피처 모듈에서 사용 금지 |
| test | 다른 test 행을 이용한 빈도·집계·rolling 금지 |
| 이력 | 제공된 `asof_*` 또는 명확히 과거로 제한된 정보만 사용 |
| 상태 | 학습이 필요한 모듈은 `stateful=True`, CV fold 안에서만 fit |

## 파일명과 prefix

```text
파일명: feat_<이름>_<주제>.py
prefix: <이름>_<주제>__

예시
feat_jisu_count.py          -> jisu_count__
feat_minsu_history.py       -> minsu_hist__
feat_yuna_trackman.py       -> yuna_tm__
```

prefix는 피처명 충돌을 방지하고, 모델 중요도를 피처 묶음 단위로 집계하는 데 사용한다.

## 팀원이 수정할 곳

`feature_example.py`를 복사한 뒤 아래 항목만 우선 수정한다.

1. `SPEC.name`, `version`, `owner`, `prefix`, `description`
2. `SPEC.required_columns`
3. `SPEC.feature_columns`
4. `PARAMS`
5. `build_features()`

`validate_feature_block`, `assert_row_independence`, `merge_feature_blocks`, manifest 생성 코드는 임의로 제거하지 않는다.

## 실행 방법

스모크 테스트:

```powershell
python feature_example.py --nrows 20000 --output-dir smoke_outputs
```

전체 생성:

```powershell
python feature_example.py --output-dir feature_outputs --format parquet
```

성공하면 다음 파일이 생성된다.

```text
<name>__train.parquet
<name>__test.parquet
<name>__feature_summary.csv
<name>__manifest.json
```

manifest에는 소유자, 버전, 입력·출력 컬럼, 파라미터, 행 검증 결과, 코드 해시, 누수 체크리스트가 들어간다.

## 모델 담당자의 조립 방법

```python
from feature_common import merge_feature_blocks
from feat_jisu_count import build_features as build_count
from feat_minsu_history import build_features as build_history

count_block = build_count(train)
history_block = build_history(train)

X_train = merge_feature_blocks(
    train[["row_id"]],
    [count_block, history_block],
)
```

모듈별 결과는 반드시 `row_id` one-to-one으로 합친다. 단순 `concat(axis=1)`은 어느 한 모듈이 행을 정렬하거나 필터링했을 때 조용히 잘못 붙을 수 있으므로 사용하지 않는다.

## Stateless와 Stateful 피처

### Stateless: 미리 생성해서 공유 가능

- 카운트 조합
- 점수 차·이닝·주자 상호작용
- 제공된 `asof_*`의 smoothing·결측 플래그
- 행 내부 정보만 사용하는 변환

`feature_example.py`가 이 유형이다.

### Stateful: 완성된 train 피처 파일로 공유하면 안 됨

- target encoding
- 선수 embedding
- 학습 데이터로 구한 평균·분산·클러스터
- Trackman 선수 crosswalk를 학습하는 과정
- 모델 기반 예측값 stacking

이 유형은 다음 인터페이스를 사용하고 모델의 각 CV fold 안에서 fit해야 한다.

```python
class StatefulFeatureBuilder:
    def fit(self, train_fold, y_train):
        ...
        return self

    def transform(self, frame):
        # row_id + 신규 피처 반환
        ...
```

전체 train을 fit한 뒤 validation을 transform하면 누수다. 최종 제출 모델을 전체 train으로 다시 학습할 때만 전체 train fit을 허용한다.

## 자동 검사가 잡는 문제

- 필수 입력 컬럼 누락
- 행 수 변경
- `row_id` 중복·결측·순서 변경
- prefix 위반과 피처명 충돌
- target이 출력에 섞임
- 비수치 피처 또는 무한대
- test를 셔플하거나 일부 행만 남겼을 때 값이 변하는 피처
- 모듈 병합 시 one-to-one 위반

## 리뷰 시 사람이 확인할 항목

자동 검사만으로 모든 누수를 찾을 수는 없다. 리뷰어는 다음을 추가로 확인한다.

- 사용한 원본 컬럼이 투구 전에 확인 가능한가
- 같은 시즌 Trackman 집계에 미래 날짜가 섞이지 않았는가
- `row_id` 숫자나 행 순서를 시간 피처처럼 쓰지 않았는가
- smoothing prior를 validation까지 포함해 계산하지 않았는가
- 새 선수·결측·희소 팀 fallback이 정의되어 있는가
- 2023·2024 순방향 검증에서 효과가 유지되는가

## 제출 전 최소 정보

각 팀원은 코드와 함께 다음 내용을 전달한다.

```text
피처 묶음 이름 / 버전 / 담당자
가설: 왜 control_success와 관련 있다고 보는가
필수 입력 컬럼
생성 피처 목록
시간·누수 안전 규칙
스모크 manifest
순방향 CV ablation 결과(가능하면 전체 및 game_type별)
```

노트북은 리뷰 편의를 위한 보조 자료다. 최종 모델은 반드시 `.py` 모듈을 호출해 동일 피처를 재생성해야 한다.
