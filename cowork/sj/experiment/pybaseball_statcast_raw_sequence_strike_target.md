# MLB Statcast 2017–2019 원본 시퀀스 수집 및 스트라이크 여부 예측 데이터셋 설계서

## 0. 목적과 범위

이 문서는 `pybaseball.statcast()`로 MLB의 **2017–2019년 정규시즌 투구 단위 데이터**를 수집하여 다음 네 축의 데이터셋을 만드는 방법을 정리한다.

1. **DATA 01 — 경기 상황 정보**
2. **DATA 02 — 경기 중요도 정보**
3. **DATA 03 — 과거 이력 정보**
4. **DATA 04 — TrackMan 과거 로그**

MLB Statcast는 2019시즌까지 투구 추적에 TrackMan 레이더를 사용했고, 2020년부터 Hawk-Eye 카메라로 전환했다. 또한 구속은 2017년부터 Statcast 기준으로 제공된다. 따라서 TrackMan 기반 MLB 데이터로 연습하려면 **2017–2019년**을 사용하는 것이 가장 깔끔하다.

> 원본 수집 범위: 2017–2019 시즌 날짜 범위, **열·행 필터 없이 저장**  
> 최종 분석 표본: 데이터셋화 단계에서 `game_type == "R"` 적용  
> 분석 단위: 공 하나당 한 행  
> 주 예측 과제: 투구 전 정보와 과거 로그를 이용한 현재 투구의 **스트라이크 결과 여부** 예측

---

## 1. 처리 유형 표기

| 표기 | 의미 |
|---|---|
| **직접** | pybaseball에서 그대로 가져오는 열 |
| **행 계산** | 같은 행에 있는 투구 전 정보로 즉시 계산 |
| **Lag** | 직전 투구 값을 이동하여 생성 |
| **누적** | 현재 시점 이전 데이터만 누적하여 계산 |
| **Rolling** | 최근 N구·N타석·N경기 구간을 계산 |
| **별도 산출** | 승리확률표, LI 표, 물리 모델 등이 필요 |
| **누수 주의** | 현재 투구 결과가 나온 후에만 알 수 있는 값 |

모든 과거 통계는 반드시 현재 행을 제외해야 한다.

\[
\text{PastFeature}_{i}
=
f(x_1,\ldots,x_{i-1})
\]

Pandas에서는 일반적으로 `shift(1)`을 적용한 뒤 `expanding()` 또는 `rolling()`을 사용한다.

---

# 2. 원본 시퀀스 우선 수집

## 2.1 수집 원칙

이 단계에서는 Statcast가 반환하는 자료를 **데이터셋으로 가공하지 않고 원본 아카이브로 저장**한다.

다음 작업은 하지 않는다.

- 열 선택 또는 열 삭제
- `game_type == "R"` 필터
- `pitch_type` 결측 제거
- 중복 제거
- 희귀 구종 통합
- 결측치 대체
- LI 계산
- Lag·누적·Rolling 변수 생성
- 학습·검증 데이터 분할

원본 수집 단계에서 허용하는 처리는 다음 두 가지뿐이다.

1. pybaseball 반환 당시의 행 순서를 `_source_row`로 보존
2. `game_date → game_pk → at_bat_number → pitch_number` 순으로 안정 정렬하여 `_sequence_in_chunk` 부여

따라서 실제 Statcast 원본 열은 하나도 삭제하거나 변환하지 않는다.

---

## 2.2 저장 구조

```text
data/
└── statcast_raw_sequence/
    ├── 2017/
    │   ├── statcast_2017-04-02_2017-04-30.parquet
    │   ├── statcast_2017-04-02_2017-04-30.json
    │   └── ...
    ├── 2018/
    ├── 2019/
    └── manifest.csv
```

각 파일의 역할:

| 파일 | 내용 |
|---|---|
| `*.parquet` | pybaseball이 반환한 모든 투구 행과 모든 열 |
| `*.json` | 해당 구간의 날짜, 행 수, 열 수, 열 목록, 정렬 정보 |
| `manifest.csv` | 전체 월별 파일의 시즌·순서·기간·행 수·경로 |

Parquet은 CSV보다 자료형 보존과 압축 효율이 좋으므로 원본 보관에 적합하다.

---

## 2.3 설치

```bash
pip install pybaseball pandas pyarrow
```

---

## 2.4 권장 전체 수집 코드

아래 코드는 2017–2019년을 월별로 나누어 수집하고, 중간에 실행이 중단되어도 이미 저장된 구간은 건너뛴다.

```python
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from pybaseball import cache, statcast


SEASONS = {
    2017: ("2017-04-02", "2017-10-01"),
    2018: ("2018-03-29", "2018-10-01"),
    2019: ("2019-03-20", "2019-09-29"),
}

OUTPUT_DIR = Path("data/statcast_raw_sequence")

SEQUENCE_COLUMNS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
]


def iter_month_ranges(start_date: str, end_date: str):
    current = pd.Timestamp(start_date)
    final = pd.Timestamp(end_date)

    while current <= final:
        month_end = current + pd.offsets.MonthEnd(0)
        chunk_end = min(month_end, final)

        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + pd.Timedelta(days=1)


def collect_chunk(
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    for attempt in range(1, max_retries + 1):
        try:
            return statcast(
                start_dt=start_date,
                end_dt=end_date,
                verbose=True,
                parallel=True,
            )
        except Exception as exc:
            if attempt == max_retries:
                raise

            wait_seconds = 10 * attempt
            print(
                f"[재시도 {attempt}/{max_retries}] "
                f"{start_date}~{end_date}: {exc}"
            )
            time.sleep(wait_seconds)

    raise RuntimeError("도달할 수 없는 코드입니다.")


def save_raw_sequence(
    data: pd.DataFrame,
    output_path: Path,
    start_date: str,
    end_date: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = data.copy()

    # pybaseball 반환 당시의 행 순서를 보존
    data.insert(0, "_source_row", range(len(data)))

    # 원본 열은 유지하고 순서만 경기-타석-투구 순으로 정렬
    available_sort_columns = [
        column for column in SEQUENCE_COLUMNS
        if column in data.columns
    ]

    if available_sort_columns:
        data = (
            data.sort_values(
                available_sort_columns,
                kind="stable",
                na_position="last",
            )
            .reset_index(drop=True)
        )

    # 월별 파일 안에서의 순차 번호
    data.insert(0, "_sequence_in_chunk", range(len(data)))

    data.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    metadata = {
        "start_date": start_date,
        "end_date": end_date,
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "columns": list(data.columns),
        "sort_columns_used": available_sort_columns,
        "data_processing": [
            "no column selection",
            "no game_type filtering",
            "no missing-value removal",
            "no pitch_type filtering",
            "no deduplication",
            "no feature engineering",
            "stable chronological sorting only",
        ],
    }

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.enable()

    manifest = []

    for season, (season_start, season_end) in SEASONS.items():
        season_dir = OUTPUT_DIR / str(season)
        season_dir.mkdir(parents=True, exist_ok=True)

        for chunk_order, (start_date, end_date) in enumerate(
            iter_month_ranges(season_start, season_end),
            start=1,
        ):
            file_name = f"statcast_{start_date}_{end_date}.parquet"
            output_path = season_dir / file_name
            metadata_path = output_path.with_suffix(".json")

            if output_path.exists() and metadata_path.exists():
                print(f"[건너뜀] {output_path}")
            else:
                print(f"\n[수집] {start_date} ~ {end_date}")

                raw_chunk = collect_chunk(start_date, end_date)

                if raw_chunk.empty:
                    print("[빈 구간] 저장하지 않습니다.")
                    continue

                save_raw_sequence(
                    data=raw_chunk,
                    output_path=output_path,
                    start_date=start_date,
                    end_date=end_date,
                )

                print(
                    f"[저장 완료] {output_path} "
                    f"({len(raw_chunk):,} pitches, "
                    f"{len(raw_chunk.columns):,} original columns)"
                )

            if metadata_path.exists():
                chunk_meta = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                manifest.append(
                    {
                        "season": season,
                        "chunk_order": chunk_order,
                        "start_date": chunk_meta["start_date"],
                        "end_date": chunk_meta["end_date"],
                        "row_count": chunk_meta["row_count"],
                        "column_count": chunk_meta["column_count"],
                        "file_path": str(output_path),
                    }
                )

    if manifest:
        manifest_df = pd.DataFrame(manifest).sort_values(
            ["season", "chunk_order"]
        )

        manifest_path = OUTPUT_DIR / "manifest.csv"
        manifest_df.to_csv(
            manifest_path,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n수집이 끝났습니다.")
        print(f"파일 위치: {OUTPUT_DIR.resolve()}")
        print(f"매니페스트: {manifest_path.resolve()}")
        print(f"총 저장 행 수: {manifest_df['row_count'].sum():,}")
    else:
        print("저장된 데이터가 없습니다.")


if __name__ == "__main__":
    main()
```

실행:

```bash
python collect_statcast_raw_sequence.py
```

### 코드의 핵심 동작

- `cache.enable()`로 반복 다운로드를 줄인다.
- 월별로 받아 즉시 Parquet으로 저장한다.
- Statcast가 반환한 원본 열을 전부 유지한다.
- 정규시즌 필터링을 하지 않는다.
- 결측 행과 미분류 구종 행도 삭제하지 않는다.
- 파일별 JSON 메타데이터를 함께 남긴다.
- 전체 파일 목록과 행 수를 `manifest.csv`에 기록한다.

---

## 2.5 가장 단순한 최소 예시

실패 복구나 월별 저장이 필요 없고 우선 빠르게 시험만 할 경우:

```python
from pathlib import Path

from pybaseball import cache, statcast

cache.enable()

out_dir = Path("data/statcast_raw_sequence_test")
out_dir.mkdir(parents=True, exist_ok=True)

raw_data = statcast(
    start_dt="2017-04-02",
    end_dt="2017-04-07",
    parallel=True,
    verbose=True,
)

# 열 선택, 필터, 결측 제거 없이 그대로 저장
raw_data.to_parquet(
    out_dir / "statcast_2017-04-02_2017-04-07.parquet",
    index=False,
    engine="pyarrow",
    compression="zstd",
)

print(raw_data.shape)
print(raw_data.columns.tolist())
```

이 코드는 수집 동작 확인용이며, 3개 시즌 전체 수집에는 앞의 월별 저장 코드를 권장한다.

---

## 2.6 나중에 원본 파일 불러오기

### 전체 기간 결합

```python
from pathlib import Path

import pandas as pd

raw_dir = Path("data/statcast_raw_sequence")

files = sorted(raw_dir.glob("*/*.parquet"))

if not files:
    raise FileNotFoundError(
        f"Parquet 파일을 찾지 못했습니다: {raw_dir.resolve()}"
    )

raw_data = pd.concat(
    [pd.read_parquet(path) for path in files],
    ignore_index=True,
)

raw_data = (
    raw_data.sort_values(
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ],
        kind="stable",
        na_position="last",
    )
    .reset_index(drop=True)
)

print(f"전체 크기: {raw_data.shape}")
print(f"경기 수: {raw_data['game_pk'].nunique():,}")
print(
    "기간:",
    raw_data["game_date"].min(),
    "~",
    raw_data["game_date"].max(),
)
```

### 특정 시즌만 불러오기

```python
from pathlib import Path

import pandas as pd

season = 2019
files = sorted(
    Path(f"data/statcast_raw_sequence/{season}").glob("*.parquet")
)

raw_2019 = pd.concat(
    [pd.read_parquet(path) for path in files],
    ignore_index=True,
)

raw_2019 = raw_2019.sort_values(
    [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
    ],
    kind="stable",
).reset_index(drop=True)
```

### 정규시즌 필터는 데이터셋화 단계에서 적용

```python
df_regular = raw_data[
    raw_data["game_type"].eq("R")
].copy()
```

원본 파일에는 포스트시즌 등 다른 경기 유형을 그대로 남기고, 분석용 DataFrame에서만 필터링한다.

---

## 2.7 원본 수집 결과 점검

```python
sequence_key = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
]

print("행 수:", len(raw_data))
print("열 수:", len(raw_data.columns))
print("시즌별 행 수:")
print(raw_data["game_year"].value_counts(dropna=False).sort_index())

print("\n경기 유형:")
print(raw_data["game_type"].value_counts(dropna=False))

print("\npitch_type 결측:")
print(raw_data["pitch_type"].isna().sum())

print("\n중복 시퀀스 키:")
print(raw_data.duplicated(sequence_key).sum())
```

중복 시퀀스 키가 발견되더라도 원본 단계에서는 삭제하지 않는다. 데이터셋화 단계에서 중복 원인을 확인한 뒤 제거 여부를 결정한다.

---

## 2.8 원본과 가공 데이터 분리

권장 디렉터리:

```text
data/
├── statcast_raw_sequence/   # 수정하지 않는 원본
├── interim/                 # 중간 처리 자료
└── processed/               # 최종 모델 입력 데이터셋
```

원본 파일은 덮어쓰지 않고, 이후 계산 결과는 반드시 별도 디렉터리에 저장한다.

예:

```python
from pathlib import Path

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

df_model.to_parquet(
    processed_dir / "strike_prediction_dataset.parquet",
    index=False,
)
```

---

# 3. 데이터셋화 단계에서 확인할 원본 열

## 3.1 식별·정렬·인물

| 데이터 | pybaseball 열 | 처리 |
|---|---|---|
| 경기 ID | `game_pk` | 직접 |
| 경기 날짜 | `game_date` | 직접 |
| 시즌 | `game_year` | 직접 |
| 경기 유형 | `game_type` | 직접, `R` 필터 |
| 홈팀 | `home_team` | 직접 |
| 원정팀 | `away_team` | 직접 |
| 타석 번호 | `at_bat_number` | 직접 |
| 타석 내 투구 번호 | `pitch_number` | 직접 |
| 투수 ID | `pitcher` | 직접 |
| 타자 ID | `batter` | 직접 |
| 포수 ID | `fielder_2` | 직접 |
| 타자 좌·우타 | `stand` | 직접 |
| 투수 좌·우완 | `p_throws` | 직접 |

## 3.2 예측 목표: 스트라이크 여부

### 기본 권장 정의 — 투구 결과 기준 스트라이크

Statcast의 `type` 열은 투구 결과를 다음과 같이 나타낸다.

| `type` | 의미 |
|---|---|
| `B` | Ball |
| `S` | Strike |
| `X` | In play |

따라서 기본 이진 타깃은 다음과 같이 만든다.

\[
y_i
=
\text{is\_strike}_i
=
\mathbb{1}(\text{type}_i=S)
\]

```python
df["is_strike"] = df["type"].eq("S").astype("int8")
```

이 정의에서는 다음이 양성 클래스다.

- 루킹 스트라이크
- 헛스윙 스트라이크
- 파울 스트라이크
- 파울팁 등 Statcast가 `type == "S"`로 분류한 투구

`type == "X"`인 인플레이 타구는 실제 공이 존 안에 들어왔을 수 있어도 이 타깃에서는 0이다. 따라서 이 타깃은 **공이 물리적으로 스트라이크존을 통과했는가**가 아니라 **해당 투구의 기록상 결과가 스트라이크였는가**를 의미한다.

결측 `type`은 원본에서는 그대로 보존하고, 데이터셋화 단계에서만 제거한다.

```python
model_df = raw_data[raw_data["type"].notna()].copy()
model_df["is_strike"] = model_df["type"].eq("S").astype("int8")
```

---

### 대안 1 — 스트라이크존 통과 여부

연구 질문이 “공이 스트라이크존 안으로 들어갔는가?”라면 `type`이 아니라 `zone`을 타깃으로 사용한다.

Statcast의 일반적인 존 번호에서 1–9는 존 내부로 취급한다.

\[
y^{zone}_i
=
\text{in\_zone}_i
=
\mathbb{1}(1\le zone_i\le9)
\]

```python
zone_valid = raw_data["zone"].notna()

zone_df = raw_data[zone_valid].copy()
zone_df["in_zone"] = zone_df["zone"].between(1, 9).astype("int8")
```

이 경우 `zone`, `plate_x`, `plate_z`는 타깃을 직접 또는 거의 직접 나타내므로 입력 변수로 사용하면 안 된다.

---

### 대안 2 — 루킹 스트라이크 여부

심판의 스트라이크 판정만 예측하려면 스윙하지 않은 투구로 범위를 제한한 뒤 `description == "called_strike"`를 양성으로 정의한다.

\[
y^{called}_i
=
\mathbb{1}
(\text{description}_i=\text{called\_strike})
\]

예시:

```python
called_strike_descriptions = {"called_strike"}

taken_ball_descriptions = {
    "ball",
    "blocked_ball",
    "pitchout",
}

called_df = raw_data[
    raw_data["description"].isin(
        called_strike_descriptions | taken_ball_descriptions
    )
].copy()

called_df["is_called_strike"] = (
    called_df["description"].eq("called_strike")
).astype("int8")
```

연구 목적이 별도로 명시되지 않았다면 본 문서에서는 **`type == "S"`인 투구 결과 기준 스트라이크 여부**를 기본 타깃으로 사용한다.

---

## 3.3 구종 정보의 역할

`pitch_type`과 `pitch_name`은 더 이상 예측 목표가 아니라 설명변수 또는 층화 변수다.

| 데이터 | pybaseball 열 | 역할 |
|---|---|---|
| 구종 코드 | `pitch_type` | 현재 구종 또는 과거 구종 패턴 |
| 구종 이름 | `pitch_name` | 설명·검증용 |

현재 투구의 `pitch_type`을 입력으로 사용할 수 있는지는 예측 시점에 따라 달라진다.

- **투구 종류가 이미 결정된 뒤 스트라이크 여부를 예측:** 현재 `pitch_type` 사용 가능
- **구종 선택 전부터 스트라이크 여부를 예측:** 현재 `pitch_type` 사용 불가
- 어느 경우든 이전 투구의 구종과 과거 구종 사용률은 사용 가능

일반적인 구종 코드 예시는 다음과 같다.

| 코드 | 의미 |
|---|---|
| `FF` | Four-Seam Fastball |
| `SI` | Sinker |
| `FC` | Cutter |
| `SL` | Slider |
| `CU` | Curveball |
| `CH` | Changeup |
| `FS` | Split-Finger |
| `KC` | Knuckle Curve |
| `ST` | Sweeper |

2017–2019 자료에서는 시즌별 구종 분류 체계 차이가 있을 수 있으므로 분석 단계에서 희귀 구종 처리 기준을 정한다.

---

# 4. DATA 01 — 경기 상황 정보

## 4.1 직접 가져오는 열

| 데이터 | pybaseball 열 | 처리 |
|---|---|---|
| 볼 | `balls` | 직접 |
| 스트라이크 | `strikes` | 직접 |
| 아웃 | `outs_when_up` | 직접 |
| 이닝 | `inning` | 직접 |
| 초·말 | `inning_topbot` | 직접 |
| 1루 주자 ID | `on_1b` | 직접 |
| 2루 주자 ID | `on_2b` | 직접 |
| 3루 주자 ID | `on_3b` | 직접 |
| 홈팀 점수 | `home_score` | 직접 |
| 원정팀 점수 | `away_score` | 직접 |
| 공격팀 점수 | `bat_score` | 직접 |
| 수비팀 점수 | `fld_score` | 직접 |
| 타자 방향 | `stand` | 직접 |
| 투수 방향 | `p_throws` | 직접 |
| 존 상단 | `sz_top` | 직접 |
| 존 하단 | `sz_bot` | 직접 |
| 내야 수비 배치 | `if_fielding_alignment` | 직접, 선택 |
| 외야 수비 배치 | `of_fielding_alignment` | 직접, 선택 |

---

## 4.2 점수 차

### 공격팀 기준 점수 차

\[
\text{ScoreDiff}_{bat}
=
\text{bat\_score}-\text{fld\_score}
\]

```python
df["score_diff_bat"] = df["bat_score"] - df["fld_score"]
```

### 홈팀 기준 점수 차

\[
\text{ScoreDiff}_{home}
=
\text{home\_score}-\text{away\_score}
\]

```python
df["score_diff_home"] = df["home_score"] - df["away_score"]
```

모델의 관점을 하나로 통일하려면 구종 예측에서는 일반적으로 공격팀 기준 점수 차를 사용한다.

---

## 4.3 주자 상태

각 베이스의 주자 존재 여부를 다음과 같이 정의한다.

\[
I_1=\mathbb{1}(\text{on\_1b 존재})
\]

\[
I_2=\mathbb{1}(\text{on\_2b 존재})
\]

\[
I_3=\mathbb{1}(\text{on\_3b 존재})
\]

### 8개 주자 상태 코드

\[
\text{BaseState}
=
I_1+2I_2+4I_3
\]

| BaseState | 주자 상황 |
|---:|---|
| 0 | 주자 없음 |
| 1 | 1루 |
| 2 | 2루 |
| 3 | 1·2루 |
| 4 | 3루 |
| 5 | 1·3루 |
| 6 | 2·3루 |
| 7 | 만루 |

```python
i1 = df["on_1b"].notna().astype("int8")
i2 = df["on_2b"].notna().astype("int8")
i3 = df["on_3b"].notna().astype("int8")

df["base_state"] = i1 + 2 * i2 + 4 * i3
```

### 주자 수

\[
\text{RunnerCount}=I_1+I_2+I_3
\]

```python
df["runner_count"] = i1 + i2 + i3
```

### 득점권 주자

\[
\text{RISP}
=
\mathbb{1}(I_2=1 \lor I_3=1)
\]

```python
df["risp"] = ((i2 == 1) | (i3 == 1)).astype("int8")
```

### 만루

\[
\text{BasesLoaded}
=
I_1I_2I_3
\]

```python
df["bases_loaded"] = ((i1 == 1) & (i2 == 1) & (i3 == 1)).astype("int8")
```

---

## 4.4 카운트 관련 변수

### 카운트 문자열

\[
\text{CountState}=\text{balls}\text{-}\text{strikes}
\]

```python
df["count_state"] = (
    df["balls"].astype("Int64").astype(str)
    + "-"
    + df["strikes"].astype("Int64").astype(str)
)
```

### 2스트라이크 여부

\[
\text{TwoStrike}
=
\mathbb{1}(\text{strikes}=2)
\]

```python
df["two_strike"] = df["strikes"].eq(2).astype("int8")
```

### 풀카운트 여부

\[
\text{FullCount}
=
\mathbb{1}(\text{balls}=3 \land \text{strikes}=2)
\]

```python
df["full_count"] = (
    df["balls"].eq(3) & df["strikes"].eq(2)
).astype("int8")
```

### 투수 유리 카운트

연구자가 기준을 사전에 정의해야 한다. 예:

\[
\text{PitcherAhead}
=
\mathbb{1}\left[
(b,s)\in\{(0,1),(0,2),(1,2)\}
\right]
\]

```python
pitcher_ahead = {(0, 1), (0, 2), (1, 2)}
df["pitcher_ahead"] = [
    int((b, s) in pitcher_ahead)
    for b, s in zip(df["balls"], df["strikes"])
]
```

### 타자 유리 카운트

예:

\[
\text{BatterAhead}
=
\mathbb{1}\left[
(b,s)\in\{(1,0),(2,0),(2,1),(3,0),(3,1)\}
\right]
\]

```python
batter_ahead = {(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)}
df["batter_ahead"] = [
    int((b, s) in batter_ahead)
    for b, s in zip(df["balls"], df["strikes"])
]
```

카운트 집단 정의는 절대적인 공식이 아니므로 연구 목적에 맞게 고정하고 보고서에 명시한다.

---

## 4.5 좌우 매치업

\[
\text{SameHanded}
=
\mathbb{1}(\text{p\_throws}=\text{stand})
\]

```python
df["same_handed_matchup"] = (
    df["p_throws"].eq(df["stand"])
).astype("int8")
```

범주형 변수로는 다음 네 가지를 사용할 수 있다.

\[
\text{Matchup}
=
\text{p\_throws}\_\text{stand}
\]

```python
df["matchup"] = df["p_throws"] + "_" + df["stand"]
```

예: `R_R`, `R_L`, `L_R`, `L_L`.

---

## 4.6 이닝·경기 단계

### 후반 이닝

\[
\text{LateInning}
=
\mathbb{1}(\text{inning}\ge 7)
\]

```python
df["late_inning"] = df["inning"].ge(7).astype("int8")
```

### 연장전

2017–2019 MLB에서는 정규 9이닝 이후를 연장으로 볼 수 있다.

\[
\text{ExtraInning}
=
\mathbb{1}(\text{inning}\ge 10)
\]

```python
df["extra_inning"] = df["inning"].ge(10).astype("int8")
```

### 접전 여부

예를 들어 2점 차 이내를 접전으로 정의하면:

\[
\text{CloseGame}
=
\mathbb{1}(|\text{ScoreDiff}_{bat}|\le 2)
\]

```python
df["close_game"] = df["score_diff_bat"].abs().le(2).astype("int8")
```

이 기준도 연구자가 사전에 정의해야 한다.

---

# 5. DATA 02 — 경기 중요도 정보

## 5.1 직접 확보하는 원자료

| 데이터 | pybaseball 열 | 사용 |
|---|---|---|
| 홈팀 승리확률 변화 | `delta_home_win_exp` | LI 표 산출용, 현재 행 입력 금지 |
| 득점기대값 변화 | `delta_run_exp` | 과거 성과 계산용, 현재 행 입력 금지 |
| 홈팀 승리확률 | `home_win_exp` | 실제 반환 여부·결측 확인 |
| 공격팀 승리확률 | `bat_win_exp` | 실제 반환 여부·결측 확인 |

`delta_home_win_exp`와 `delta_run_exp`는 현재 투구 또는 타석 결과가 반영된 값이므로, **현재 투구의 스트라이크 여부를 예측하는 입력으로 직접 사용하면 데이터 누수**가 발생한다.

---

## 5.2 레버리지 인덱스의 이론적 정의

현재 경기 상태를 \(s\), 발생 가능한 타석 결과를 \(o\), 결과 발생 후 상태를 \(s_o\)라고 하면:

\[
LI(s)
=
\frac{
\sum_o P(o\mid s)
\left|
WE(s_o)-WE(s)
\right|
}{
\mathbb{E}_{s,o}
\left[
\left|
WE(s_o)-WE(s)
\right|
\right]
}
\]

- \(WE(s)\): 현재 상태의 승리확률
- \(P(o\mid s)\): 현재 상태에서 결과 \(o\)가 발생할 확률
- 분자: 현재 상태가 만들어낼 수 있는 승리확률 변화의 기대 절댓값
- 분모: 리그 전체 평균 승리확률 변화의 절댓값
- 정규화 결과: 평균적인 상황의 LI가 약 1.0

중요한 점은 현재 타석에서 **실제로 발생한 단 하나의 WPA**를 그 타석의 LI로 쓰는 것이 아니라, 동일한 상황에서 발생 가능한 결과들의 승리확률 변화 폭을 평가한다는 것이다.

---

## 5.3 실증적 PA 단위 LI 추정

실제 연구에서는 과거의 동일 상황에서 관측된 절대 승리확률 변화량을 평균하여 근사할 수 있다.

상태 벡터:

\[
s_{\text{PA}}
=
(
\text{inning},
\text{inning\_topbot},
\text{score\_diff},
\text{outs},
\text{base\_state}
)
\]

동일 상태 \(s\)의 과거 타석 집합을 \(A_s\)라 하면:

\[
\widehat{Swing}(s)
=
\frac{1}{|A_s|}
\sum_{j\in A_s}
|\Delta WE_j|
\]

전체 리그 평균:

\[
\overline{Swing}
=
\frac{1}{N}
\sum_{j=1}^{N}
|\Delta WE_j|
\]

최종 추정 LI:

\[
\widehat{LI}(s)
=
\frac{\widehat{Swing}(s)}
{\overline{Swing}}
\]

### 타석 단위 데이터 만들기

```python
pa_keys = ["game_pk", "at_bat_number"]

df = df.sort_values(
    ["game_pk", "at_bat_number", "pitch_number"]
).copy()

pa_first = df.groupby(pa_keys, as_index=False).first()
pa_last = (
    df.groupby(pa_keys, as_index=False)
      .last()[pa_keys + ["delta_home_win_exp"]]
      .rename(columns={"delta_home_win_exp": "pa_delta_home_win_exp"})
)

pa = pa_first.merge(pa_last, on=pa_keys, how="left")

pa["base_state"] = (
    pa["on_1b"].notna().astype("int8")
    + 2 * pa["on_2b"].notna().astype("int8")
    + 4 * pa["on_3b"].notna().astype("int8")
)

pa["score_diff_home"] = pa["home_score"] - pa["away_score"]
pa["abs_wpa"] = pa["pa_delta_home_win_exp"].abs()
```

### 상태별 LI 표

```python
state_cols = [
    "inning",
    "inning_topbot",
    "score_diff_home",
    "outs_when_up",
    "base_state",
]

valid_pa = pa.dropna(subset=["abs_wpa"]).copy()

league_mean_swing = valid_pa["abs_wpa"].mean()

li_table = (
    valid_pa.groupby(state_cols, dropna=False)["abs_wpa"]
    .agg(mean_abs_wpa="mean", n_state="size")
    .reset_index()
)

li_table["li_pa"] = li_table["mean_abs_wpa"] / league_mean_swing
```

---

## 5.4 희소 상태 보정

정확히 같은 상태의 표본 수가 적으면 LI가 불안정해진다.

### 점수 차 클리핑

예:

\[
\text{ScoreDiffBin}
=
\min(5,\max(-5,\text{ScoreDiff}))
\]

```python
pa["score_diff_bin"] = pa["score_diff_home"].clip(-5, 5)
```

### 후기 이닝 묶기

예:

\[
\text{InningBin}
=
\begin{cases}
1,\ldots,8, & inning \le 8\\
9, & inning = 9\\
10, & inning \ge 10
\end{cases}
\]

```python
pa["inning_bin"] = pa["inning"].clip(upper=10)
```

### 전체 평균을 이용한 수축 추정

상태 \(s\)의 관측 수가 \(n_s\), 상태 평균이 \(\bar{x}_s\), 전체 평균이 \(\mu\), 수축 강도가 \(\lambda\)일 때:

\[
\widetilde{Swing}(s)
=
\frac{
n_s\bar{x}_s+\lambda\mu
}{
n_s+\lambda
}
\]

\[
\widetilde{LI}(s)
=
\frac{\widetilde{Swing}(s)}{\mu}
\]

```python
lambda_shrink = 100

li_table["shrunk_swing"] = (
    li_table["n_state"] * li_table["mean_abs_wpa"]
    + lambda_shrink * league_mean_swing
) / (
    li_table["n_state"] + lambda_shrink
)

li_table["li_pa_shrunk"] = (
    li_table["shrunk_swing"] / league_mean_swing
)
```

`lambda_shrink`는 검증 데이터에서 조정해야 한다.

---

## 5.5 학습·검증 누수 방지 LI

검증연도 또는 테스트연도의 결과를 이용해 LI 표를 만들면 누수가 생긴다.

예:

- 2017년으로 LI 표 생성 → 2018년 적용
- 2017–2018년으로 LI 표 생성 → 2019년 적용

\[
LI_{2019}(s)
=
f(\text{2017--2018 데이터만})
\]

```python
train_pa = pa[pa["game_year"].isin([2017, 2018])].copy()
test_pa = pa[pa["game_year"].eq(2019)].copy()
```

교차검증에서도 각 fold의 학습 데이터로만 LI 표를 만든 뒤 검증 fold에 조인해야 한다.

---

## 5.6 투구 단위 LI 확장

구종 선택은 볼·스트라이크 카운트의 영향을 크게 받으므로 다음 상태를 사용할 수 있다.

\[
s_{\text{pitch}}
=
(
\text{inning},
\text{inning\_topbot},
\text{score\_diff},
\text{outs},
\text{base\_state},
\text{balls},
\text{strikes}
)
\]

투구별 실제 \(|\Delta WE|\)를 동일 방식으로 평균하면:

\[
\widehat{LI}_{pitch}(s)
=
\frac{
\operatorname{mean}(|\Delta WE|\mid s)
}{
\operatorname{mean}(|\Delta WE|)
}
\]

다만 이는 전통적인 PA 단위 LI를 볼카운트까지 확장한 **연구용 투구 단위 지표**임을 명시해야 한다.

---

## 5.7 고레버리지 구간화

예시:

\[
\text{LeverageClass}
=
\begin{cases}
Low, & LI < 0.85\\
Medium, & 0.85 \le LI < 2.0\\
High, & LI \ge 2.0
\end{cases}
\]

```python
import numpy as np

df["leverage_class"] = np.select(
    [
        df["li_pa"].lt(0.85),
        df["li_pa"].lt(2.0),
    ],
    [
        "low",
        "medium",
    ],
    default="high",
)
```

절단점은 연구 설계에 따라 변경할 수 있으며, 연속형 LI를 그대로 쓰는 것이 정보 손실은 적다.

---

# 6. DATA 03 — 과거 이력 정보

## 6.1 공통 원칙

현재 투구 \(i\)를 예측할 때 사용할 수 있는 값:

\[
\{x_1,x_2,\ldots,x_{i-1}\}
\]

사용하면 안 되는 값:

\[
\{x_i,x_{i+1},\ldots\}
\]

모든 그룹별 누적·rolling 변수는 다음 순서를 지킨다.

```python
df = df.sort_values(
    ["game_date", "game_pk", "at_bat_number", "pitch_number"]
).copy()
```

---

## 6.2 투수 시즌 누적 투구 수

투수 \(p\)의 현재 투구 직전까지 투구 수:

\[
N^{past}_{p,i}
=
\sum_{j<i}\mathbb{1}(pitcher_j=p)
\]

```python
df["pitcher_pitch_count_before"] = (
    df.groupby(["game_year", "pitcher"]).cumcount()
)
```

`cumcount()`는 첫 행에 0을 반환하므로 현재 행을 포함하지 않는다.

---

## 6.3 투수 시즌 구종 사용률

투수 \(p\), 구종 \(k\)의 현재 시점 이전 사용률:

\[
Usage_{p,k,i}
=
\frac{
\sum_{j<i}
\mathbb{1}(pitcher_j=p,\ pitch\_type_j=k)
}{
\sum_{j<i}
\mathbb{1}(pitcher_j=p)
}
\]

특정 구종 `FF` 예:

```python
g = df.groupby(["game_year", "pitcher"], sort=False)

df["ff_before"] = (
    df["pitch_type"].eq("FF")
      .groupby([df["game_year"], df["pitcher"]])
      .cumsum()
      .groupby([df["game_year"], df["pitcher"]])
      .shift(1)
      .fillna(0)
)

df["ff_usage_season"] = (
    df["ff_before"]
    / df["pitcher_pitch_count_before"].replace(0, pd.NA)
)
```

모든 구종에 대해 계산하려면 one-hot encoding 후 그룹별 누적합을 사용한다.

---

## 6.4 최근 N구 구종 사용률

투수의 현재 투구 이전 최근 \(N\)구 중 구종 \(k\)의 비율:

\[
Usage^{(N)}_{p,k,i}
=
\frac{1}{N_i}
\sum_{j=i-N_i}^{i-1}
\mathbb{1}(pitch\_type_j=k)
\]

여기서 \(N_i=\min(N,\text{현재까지 과거 투구 수})\).

```python
def previous_rolling_mean(series, window, min_periods=1):
    return series.shift(1).rolling(window, min_periods=min_periods).mean()

df["is_ff"] = df["pitch_type"].eq("FF").astype("int8")

df["ff_usage_last_20"] = (
    df.groupby("pitcher", group_keys=False)["is_ff"]
      .apply(lambda s: previous_rolling_mean(s, 20))
)

df["ff_usage_last_50"] = (
    df.groupby("pitcher", group_keys=False)["is_ff"]
      .apply(lambda s: previous_rolling_mean(s, 50))
)
```

---

## 6.5 상황별 구종 사용률

### 타자 손별

\[
Usage_{p,k,h,i}
=
\frac{
\sum_{j<i}
\mathbb{1}(pitcher_j=p,\ stand_j=h,\ pitch_j=k)
}{
\sum_{j<i}
\mathbb{1}(pitcher_j=p,\ stand_j=h)
}
\]

그룹 키:

```python
["game_year", "pitcher", "stand"]
```

### 카운트별

\[
Usage_{p,k,b,s,i}
=
P(pitch\_type=k
\mid pitcher=p,\ balls=b,\ strikes=s,\ past)
\]

그룹 키:

```python
["game_year", "pitcher", "balls", "strikes"]
```

### 주자 상태별

그룹 키:

```python
["game_year", "pitcher", "base_state"]
```

### 2스트라이크 결정구 사용률

\[
PutawayUsage_{p,k,i}
=
P(pitch\_type=k
\mid pitcher=p,\ strikes=2,\ past)
\]

그룹 키:

```python
["game_year", "pitcher", "two_strike"]
```

표본이 적은 상황별 비율은 투수 전체 비율과 베이지안 수축을 적용할 수 있다.

\[
\widetilde{Usage}_{group}
=
\frac{
n_{group}Usage_{group}+\lambda Usage_{pitcher}
}{
n_{group}+\lambda
}
\]

---

## 6.6 최근 평균 구속·회전·무브먼트

현재 투구의 물리값은 구종이 이미 던져진 후 알 수 있으므로 입력으로 쓰면 안 된다. 대신 이전 투구의 값이나 최근 평균을 사용한다.

변수 \(x\)의 최근 \(N\)구 평균:

\[
\overline{x}^{(N)}_{p,i}
=
\frac{1}{N_i}
\sum_{j=i-N_i}^{i-1}x_j
\]

예:

```python
physical_cols = [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
]

for col in physical_cols:
    df[f"{col}_last20_mean"] = (
        df.groupby("pitcher", group_keys=False)[col]
          .apply(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )
```

구종별 물리 평균은 그룹을 다음과 같이 확장한다.

```python
["pitcher", "pitch_type"]
```

스트라이크 여부가 타깃인 경우 현재 `pitch_type`은 예측 시점에 구종이 이미 알려져 있다면 입력으로 사용할 수 있다. 구종 선택 전 예측이라면 현재 구종은 제외하고, 이전 투구의 구종과 투수의 과거 구종 사용률만 사용한다.

---

## 6.7 최근 변동성·안정성

최근 \(N\)구의 표준편차:

\[
SD^{(N)}_{p,i}(x)
=
\sqrt{
\frac{1}{N_i-1}
\sum_{j=i-N_i}^{i-1}
(x_j-\overline{x}^{(N)}_{p,i})^2
}
\]

```python
df["release_x_sd_last20"] = (
    df.groupby("pitcher", group_keys=False)["release_pos_x"]
      .apply(lambda s: s.shift(1).rolling(20, min_periods=5).std())
)

df["release_z_sd_last20"] = (
    df.groupby("pitcher", group_keys=False)["release_pos_z"]
      .apply(lambda s: s.shift(1).rolling(20, min_periods=5).std())
)
```

2차원 릴리스 포인트 안정성:

\[
ReleaseDispersion
=
\sqrt{
SD_x^2+SD_z^2
}
\]

```python
df["release_dispersion_last20"] = (
    df["release_x_sd_last20"].pow(2)
    + df["release_z_sd_last20"].pow(2)
).pow(0.5)
```

값이 작을수록 최근 릴리스 포인트가 일관적이다.

---

## 6.8 평균 대비 최근 구속·회전 변화

최근 \(N\)구 평균과 시즌 이전 전체 평균의 차이:

\[
VelocityDelta_{p,i}
=
\overline{V}^{(N)}_{p,i}
-
\overline{V}^{season,past}_{p,i}
\]

```python
g_pitcher = df.groupby(["game_year", "pitcher"], group_keys=False)

df["velo_season_mean_before"] = (
    g_pitcher["release_speed"]
    .apply(lambda s: s.shift(1).expanding(min_periods=10).mean())
)

df["velo_last20_mean"] = (
    g_pitcher["release_speed"]
    .apply(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
)

df["velo_delta_last20_vs_season"] = (
    df["velo_last20_mean"] - df["velo_season_mean_before"]
)
```

회전수도 같은 방식이다.

\[
SpinDelta_{p,i}
=
\overline{Spin}^{(N)}_{p,i}
-
\overline{Spin}^{season,past}_{p,i}
\]

---

## 6.9 직전 경기 투구 수

투수 \(p\)의 경기 \(g\) 내 총 투구 수:

\[
GamePitchCount_{p,g}
=
\sum_i
\mathbb{1}(pitcher_i=p,\ game_i=g)
\]

```python
game_pitch_count = (
    df.groupby(["pitcher", "game_pk"])
      .size()
      .rename("game_pitch_count")
      .reset_index()
)
```

투수별 경기 순서로 한 경기 이동:

```python
pitcher_games = (
    df[["pitcher", "game_pk", "game_date"]]
    .drop_duplicates()
    .merge(game_pitch_count, on=["pitcher", "game_pk"], how="left")
    .sort_values(["pitcher", "game_date", "game_pk"])
)

pitcher_games["prev_game_pitch_count"] = (
    pitcher_games.groupby("pitcher")["game_pitch_count"].shift(1)
)
```

그 후 원본 투구 데이터에 조인한다.

---

## 6.10 휴식일

현재 등판일과 직전 등판일의 차이:

\[
RestDays_{p,g}
=
Date_{p,g}
-
Date_{p,g-1}
\]

```python
pitcher_games["prev_game_date"] = (
    pitcher_games.groupby("pitcher")["game_date"].shift(1)
)

pitcher_games["rest_days"] = (
    pd.to_datetime(pitcher_games["game_date"])
    - pd.to_datetime(pitcher_games["prev_game_date"])
).dt.days
```

같은 날 더블헤더 등은 `game_pk` 순서와 경기 시작시각이 없으면 해석에 주의한다.

---

## 6.11 타순 몇 바퀴째

현재 경기에서 동일 타자를 몇 번째 상대하는지:

\[
TimesFaced_{p,b,g,i}
=
1+
\sum_{j<i}
\mathbb{1}
(
pitcher_j=p,\ batter_j=b,\ game_j=g,\ PA\ start
)
\]

타석 단위로 계산한다.

```python
pa_order = (
    df[["game_pk", "pitcher", "batter", "at_bat_number"]]
    .drop_duplicates()
    .sort_values(["game_pk", "at_bat_number"])
)

pa_order["times_faced_in_game"] = (
    pa_order.groupby(["game_pk", "pitcher", "batter"]).cumcount() + 1
)
```

원본 투구 데이터에 `game_pk`, `pitcher`, `batter`, `at_bat_number`로 조인한다.

---

## 6.12 타자 최근 타율

타석 결과를 다음처럼 정의한다.

\[
AB_j=
\mathbb{1}
(\text{공식 타수에 포함되는 타석})
\]

\[
H_j=
\mathbb{1}
(\text{single, double, triple, home\_run})
\]

최근 \(M\)타석 타율:

\[
BA^{(M)}_{b,i}
=
\frac{
\sum_{j=i-M}^{i-1}H_j
}{
\sum_{j=i-M}^{i-1}AB_j
}
\]

일반적인 타수 제외 예: 볼넷, 사구, 희생번트, 희생플라이, 포수방해.

```python
hit_events = {"single", "double", "triple", "home_run"}

non_ab_events = {
    "walk",
    "hit_by_pitch",
    "sac_bunt",
    "sac_fly",
    "catcher_interf",
}

pa["is_hit"] = pa["events"].isin(hit_events).astype("int8")
pa["is_ab"] = (
    pa["events"].notna()
    & ~pa["events"].isin(non_ab_events)
).astype("int8")
```

최근 20타석:

```python
pa = pa.sort_values(
    ["batter", "game_date", "game_pk", "at_bat_number"]
).copy()

pa["hits_last20_pa"] = (
    pa.groupby("batter", group_keys=False)["is_hit"]
      .apply(lambda s: s.shift(1).rolling(20, min_periods=5).sum())
)

pa["ab_last20_pa"] = (
    pa.groupby("batter", group_keys=False)["is_ab"]
      .apply(lambda s: s.shift(1).rolling(20, min_periods=5).sum())
)

pa["ba_last20_pa"] = (
    pa["hits_last20_pa"]
    / pa["ab_last20_pa"].replace(0, pd.NA)
)
```

---

## 6.13 타자 출루율

\[
OBP
=
\frac{H+BB+HBP}
{AB+BB+HBP+SF}
\]

희생번트는 분모에서 제외하고 희생플라이는 포함한다.

각 타석을 0/1 지표로 만든 뒤 과거 rolling 합을 사용한다.

\[
OBP^{(M)}_{b,i}
=
\frac{
H^{past}+BB^{past}+HBP^{past}
}{
AB^{past}+BB^{past}+HBP^{past}+SF^{past}
}
\]

---

## 6.14 타자 장타율

총루타:

\[
TB
=
1B+2(2B)+3(3B)+4(HR)
\]

장타율:

\[
SLG=\frac{TB}{AB}
\]

최근 \(M\)타석:

\[
SLG^{(M)}_{b,i}
=
\frac{
\sum TB_j
}{
\sum AB_j
}
\]

```python
tb_map = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,
}

pa["total_bases"] = pa["events"].map(tb_map).fillna(0)
```

---

## 6.15 삼진율·볼넷율

타석 기준:

\[
K\%=\frac{K}{PA}
\]

\[
BB\%=\frac{BB}{PA}
\]

```python
strikeout_events = {"strikeout", "strikeout_double_play"}
walk_events = {"walk", "intent_walk"}

pa["is_k"] = pa["events"].isin(strikeout_events).astype("int8")
pa["is_bb"] = pa["events"].isin(walk_events).astype("int8")
```

최근 \(M\)타석:

\[
K\%^{(M)}_{b,i}
=
\frac{1}{M_i}
\sum_{j=i-M_i}^{i-1}K_j
\]

\[
BB\%^{(M)}_{b,i}
=
\frac{1}{M_i}
\sum_{j=i-M_i}^{i-1}BB_j
\]

---

## 6.16 헛스윙률

투구 단위 헛스윙 지표:

\[
Whiff_i
=
\mathbb{1}
(
description_i
\in
\{
swinging\_strike,
swinging\_strike\_blocked,
missed\_bunt
\}
)
\]

스윙 여부:

\[
Swing_i
=
\mathbb{1}
(
description_i
\in
\text{스윙 결과 집합}
)
\]

헛스윙률:

\[
WhiffRate
=
\frac{\sum Whiff_i}{\sum Swing_i}
\]

```python
whiff_desc = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
}

swing_desc = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
}

df["is_whiff"] = df["description"].isin(whiff_desc).astype("int8")
df["is_swing"] = df["description"].isin(swing_desc).astype("int8")
```

투수·타자·구종별 과거 헛스윙률:

\[
WhiffRate_{group,i}
=
\frac{
\sum_{j<i}Whiff_j
}{
\sum_{j<i}Swing_j
}
\]

스트라이크 여부 예측에서는 현재 구종이 추론 시점에 알려진다는 설계라면 구종별 과거 헛스윙률을 사용할 수 있다. 다만 현재 행의 결과를 포함하지 않도록 반드시 `shift(1)` 또는 과거 구간 집계를 적용한다.

---

## 6.17 투수–타자 맞대결 이력

투수 \(p\), 타자 \(b\)의 과거 맞대결 투구 수:

\[
MatchupPitches_{p,b,i}
=
\sum_{j<i}
\mathbb{1}(pitcher_j=p,batter_j=b)
\]

```python
df["matchup_pitch_count_before"] = (
    df.groupby(["pitcher", "batter"]).cumcount()
)
```

맞대결 구종 사용률:

\[
MatchupUsage_{p,b,k,i}
=
\frac{
\sum_{j<i}
\mathbb{1}(p_j=p,b_j=b,pitch_j=k)
}{
MatchupPitches_{p,b,i}
}
\]

맞대결 타석 결과 기반 삼진율·볼넷율·안타율도 PA 단위로 같은 방식으로 계산한다.

표본이 매우 적으므로 최소 표본 기준 또는 수축 추정을 권장한다.

---

## 6.18 같은 경기에서 타자가 이미 본 구종

현재 경기·투수·타자 조합에서 이전에 본 총 투구 수:

\[
SeenPitches_{g,p,b,i}
=
\sum_{j<i}
\mathbb{1}(game_j=g,p_j=p,b_j=b)
\]

특정 구종 \(k\)를 본 횟수:

\[
SeenPitchType_{g,p,b,k,i}
=
\sum_{j<i}
\mathbb{1}(game_j=g,p_j=p,b_j=b,pitch_j=k)
\]

현재 투구의 스트라이크 여부를 예측할 때는 후보 구종별 이전 노출 횟수를 별도 wide feature로 만든다.

예:

- `seen_ff_before`
- `seen_sl_before`
- `seen_ch_before`
- `seen_cu_before`

---

# 7. 구종 시퀀스 변수

## 7.1 직전 구종

같은 경기에서 현재 투수의 직전 투구:

\[
PrevPitchType_i=pitch\_type_{i-1}
\]

```python
game_pitcher_group = df.groupby(["game_pk", "pitcher"], sort=False)

df["prev_pitch_type_1"] = game_pitcher_group["pitch_type"].shift(1)
df["prev_pitch_type_2"] = game_pitcher_group["pitch_type"].shift(2)
df["prev_pitch_type_3"] = game_pitcher_group["pitch_type"].shift(3)
```

타석 경계를 넘기지 않으려면 그룹에 `at_bat_number`를 추가한다.

```python
pa_group = df.groupby(
    ["game_pk", "at_bat_number", "pitcher"],
    sort=False,
)

df["prev_pitch_type_pa_1"] = pa_group["pitch_type"].shift(1)
```

---

## 7.2 직전 구종과 동일 여부

이 값은 현재 구종이 있어야 계산되므로 **학습 정답 분석용**이며 현재 투구의 스트라이크 여부 예측 입력으로는 사용할 수 없다.

\[
RepeatPitch_i
=
\mathbb{1}
(pitch\_type_i=pitch\_type_{i-1})
\]

예측 입력으로는 직전 구종 자체만 사용한다.

---

## 7.3 직전 투구 물리값

\[
PrevVelocity_i=release\_speed_{i-1}
\]

\[
PrevPlateX_i=plate\_x_{i-1}
\]

\[
PrevPlateZ_i=plate\_z_{i-1}
\]

```python
for col in [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "release_pos_x",
    "release_pos_z",
]:
    df[f"prev_{col}"] = game_pitcher_group[col].shift(1)
```

직전 공은 이미 관측되었으므로 다음 공 예측에 사용할 수 있다.

---

## 7.4 직전 투구 결과

```python
df["prev_description"] = game_pitcher_group["description"].shift(1)
df["prev_pitch_result_type"] = game_pitcher_group["type"].shift(1)
```

예:

- 직전 공 헛스윙 여부
- 직전 공 파울 여부
- 직전 공 볼 여부
- 직전 공 인플레이 여부

\[
PrevWhiff_i=Whiff_{i-1}
\]

\[
PrevBall_i=\mathbb{1}(type_{i-1}=B)
\]

---

## 7.5 연속 동일 구종 횟수

현재 투구 예측 시점에는 직전까지 동일 구종이 몇 번 연속됐는지를 계산한다.

예: 과거 시퀀스가 `FF, FF, FF`이면 다음 공 직전 연속 FF 횟수는 3.

개념식:

\[
RunLength_{i-1}
=
\max
\left\{
r:
pitch_{i-1}=pitch_{i-2}=\cdots=pitch_{i-r}
\right\}
\]

구현 예:

```python
def previous_run_length(series: pd.Series) -> pd.Series:
    values = series.tolist()
    output = []
    prev = None
    run = 0

    for value in values:
        output.append(run)

        if value == prev:
            run += 1
        else:
            prev = value
            run = 1

    return pd.Series(output, index=series.index)

df["prev_same_pitch_run"] = (
    df.groupby(["game_pk", "pitcher"], group_keys=False)["pitch_type"]
      .apply(previous_run_length)
)
```

---

# 8. DATA 04 — TrackMan 과거 로그

## 8.1 직접 확보하는 물리 변수

### 구속·회전

| 데이터 | pybaseball 열 | 단위 |
|---|---|---|
| 릴리스 구속 | `release_speed` | mph |
| 유효 구속 | `effective_speed` | mph |
| 회전수 | `release_spin_rate` | rpm |
| 회전축 | `spin_axis` | degree |

### 무브먼트·위치

| 데이터 | pybaseball 열 | 단위 |
|---|---|---|
| 수평 무브먼트 | `pfx_x` | ft |
| 수직 무브먼트 | `pfx_z` | ft |
| 플레이트 좌우 위치 | `plate_x` | ft |
| 플레이트 높이 | `plate_z` | ft |
| 존 번호 | `zone` | 범주 |
| 초기 x 속도 | `vx0` | ft/s |
| 초기 y 속도 | `vy0` | ft/s |
| 초기 z 속도 | `vz0` | ft/s |
| x 가속도 | `ax` | ft/s² |
| y 가속도 | `ay` | ft/s² |
| z 가속도 | `az` | ft/s² |

### 릴리스 정보

| 데이터 | pybaseball 열 | 단위 |
|---|---|---|
| 릴리스 좌우 위치 | `release_pos_x` | ft |
| 릴리스 전후 위치 | `release_pos_y` | ft |
| 릴리스 높이 | `release_pos_z` | ft |
| 익스텐션 | `release_extension` | ft |

---

## 8.2 무브먼트 인치 변환

Statcast의 `pfx_x`, `pfx_z`는 피트 단위이므로:

\[
pfx\_x_{inch}=12\cdot pfx\_x
\]

\[
pfx\_z_{inch}=12\cdot pfx\_z
\]

```python
df["pfx_x_in"] = 12 * df["pfx_x"]
df["pfx_z_in"] = 12 * df["pfx_z"]
```

KBO TrackMan 열과 비교할 때 좌우 부호 방향 및 무브먼트 정의를 반드시 확인한다.

---

## 8.3 스트라이크존 정규화 높이

타자마다 `sz_bot`, `sz_top`이 다르므로 플레이트 높이를 0–1로 정규화할 수 있다.

\[
PlateZNorm
=
\frac{
plate\_z-sz\_bot
}{
sz\_top-sz\_bot
}
\]

```python
zone_height = df["sz_top"] - df["sz_bot"]

df["plate_z_norm"] = (
    (df["plate_z"] - df["sz_bot"])
    / zone_height.replace(0, pd.NA)
)
```

해석:

- 0: 존 하단
- 0.5: 존 중앙
- 1: 존 상단

현재 구종 선택 예측에서는 현재 공의 `plate_z`가 아직 알려지지 않으므로 입력으로 쓰면 안 된다. 직전 투구 위치 또는 과거 위치 경향에 사용한다.

---

## 8.4 존 중앙으로부터 거리

개인별 존 중앙:

\[
ZoneCenterZ
=
\frac{sz\_top+sz\_bot}{2}
\]

정규화된 수직 거리:

\[
VerticalDistance
=
\frac{
plate\_z-ZoneCenterZ
}{
sz\_top-sz\_bot
}
\]

수평·수직을 결합한 근사 거리:

\[
ZoneCenterDistance
=
\sqrt{
plate\_x^2
+
VerticalDistance^2
}
\]

수평과 수직의 단위가 다르게 정규화될 수 있으므로 분석 목적에 맞게 스케일링한다.

---

## 8.5 궤적 복원

Statcast에서 제공되는 초기 속도와 가속도가 투구 구간 동안 일정하다고 근사하면:

\[
x(t)
=
x_0+v_{x0}t+\frac{1}{2}a_xt^2
\]

\[
y(t)
=
y_0+v_{y0}t+\frac{1}{2}a_yt^2
\]

\[
z(t)
=
z_0+v_{z0}t+\frac{1}{2}a_zt^2
\]

여기서:

- \(x_0,y_0,z_0\): 기준 시점 위치
- \(v_{x0},v_{y0},v_{z0}\): 기준 시점 속도
- \(a_x,a_y,a_z\): 가속도

이는 원시 레이더의 프레임별 좌표가 아니라, 공개된 적합 파라미터를 이용한 **근사 궤적**이다.

---

## 8.6 특정 y 평면 도달시간

목표 평면을 \(y^\*\)라 하면:

\[
y^\*
=
y_0+v_{y0}t+\frac{1}{2}a_yt^2
\]

정리하면:

\[
\frac{1}{2}a_yt^2+v_{y0}t+(y_0-y^\*)=0
\]

근의 공식:

\[
t
=
\frac{
-v_{y0}
\pm
\sqrt{
v_{y0}^2-2a_y(y_0-y^\*)
}
}{
a_y
}
\]

물리적으로 유효한 양의 시간 중 투구 비행 구간에 해당하는 근을 선택한다.

\(a_y\approx0\)이면:

\[
t
\approx
\frac{y^\*-y_0}{v_{y0}}
\]

---

## 8.7 목표 평면에서의 속도

도달시간 \(t^\*\)를 구한 뒤:

\[
v_x(t^\*)=v_{x0}+a_xt^\*
\]

\[
v_y(t^\*)=v_{y0}+a_yt^\*
\]

\[
v_z(t^\*)=v_{z0}+a_zt^\*
\]

---

## 8.8 수직 접근각 근사

목표 평면에서 공의 진행방향에 대한 수직 접근각:

\[
VAA
=
\tan^{-1}
\left(
\frac{v_z(t^\*)}
{|v_y(t^\*)|}
\right)
\cdot
\frac{180}{\pi}
\]

```python
import numpy as np

df["vaa_approx"] = np.degrees(
    np.arctan2(df["vz_at_plate"], np.abs(df["vy_at_plate"]))
)
```

좌표계 정의에 따라 하강하는 공의 VAA가 음수가 되도록 부호를 확인한다.

---

## 8.9 수평 접근각 근사

\[
HAA
=
\tan^{-1}
\left(
\frac{v_x(t^\*)}
{|v_y(t^\*)|}
\right)
\cdot
\frac{180}{\pi}
\]

```python
df["haa_approx"] = np.degrees(
    np.arctan2(df["vx_at_plate"], np.abs(df["vy_at_plate"]))
)
```

KBO TrackMan의 `VertApprAngle`, `HorzApprAngle`과 비교할 때는 기준 평면·부호·좌표축 정의를 맞춰야 한다.

---

## 8.10 릴리스 포인트 거리

투수의 최근 평균 릴리스 포인트를 \((\bar{x},\bar{z})\)라 할 때 특정 투구의 이탈 거리:

\[
ReleaseDeviation_i
=
\sqrt{
(release\_pos\_x_i-\bar{x})^2
+
(release\_pos\_z_i-\bar{z})^2
}
\]

현재 투구 예측에서는 현재 릴리스 포인트를 쓸 수 없으므로, 직전 투구의 이탈값 또는 최근 분산을 사용한다.

---

## 8.11 구종 간 물리적 분리도

투수의 과거 구종 \(k_1\), \(k_2\) 평균 특성 벡터를 다음처럼 정의한다.

\[
\boldsymbol{\mu}_{k}
=
(
\overline{Velocity}_k,
\overline{pfx_x}_k,
\overline{pfx_z}_k,
\overline{ReleaseX}_k,
\overline{ReleaseZ}_k
)
\]

표준화된 유클리드 거리:

\[
Distance(k_1,k_2)
=
\sqrt{
\sum_m
\left(
\frac{
\mu_{k_1,m}-\mu_{k_2,m}
}{
\sigma_m
}
\right)^2
}
\]

이 값은 구종 간 차별성을 나타내는 투수 프로필로 사용할 수 있다. 현재 투구의 스트라이크 여부 예측 시에는 학습 시점 이전 자료로만 프로필을 계산해야 한다.

---

## 8.12 터널링 근사

직전 구종 \(a\)와 현재 후보 구종 \(b\)가 타자 인지 구간의 특정 \(y^\*\) 평면에서 얼마나 가까운지:

\[
TunnelDistance_{a,b}(y^\*)
=
\sqrt{
[x_a(t_a^\*)-x_b(t_b^\*)]^2
+
[z_a(t_a^\*)-z_b(t_b^\*)]^2
}
\]

플레이트에서의 최종 분리도:

\[
PlateSeparation_{a,b}
=
\sqrt{
(plate\_x_a-plate\_x_b)^2
+
(plate\_z_a-plate\_z_b)^2
}
\]

터널링 효과를 단순화하면:

\[
TunnelScore
=
PlateSeparation-TunnelDistance
\]

값이 클수록 초반 궤적은 비슷하지만 후반 결과 위치가 크게 갈라지는 조합으로 해석할 수 있다.

단, 현재 실제 구종 \(b\)의 궤적은 예측 시점에 알 수 없으므로 후보 구종별 과거 평균 궤적을 사용해야 한다.

---

# 9. 결과·성과 원자료

다음 열은 현재 투구의 결과이므로 **현재 투구의 스트라이크 여부 예측 입력에는 넣지 않고**, 과거 통계 계산 및 정답 검증에 사용한다.

| 데이터 | pybaseball 열 | 용도 |
|---|---|---|
| 투구 결과 설명 | `description` | 과거 헛스윙·파울·볼 집계 |
| 볼·스트라이크 결과 코드 | `type` | 과거 결과 집계 |
| 타석 최종 결과 | `events` | 타율·출루율·삼진율 |
| 타구 유형 | `bb_type` | 과거 타구 성향 |
| 타구 속도 | `launch_speed` | 과거 타격 성과 |
| 발사각 | `launch_angle` | 과거 타격 성과 |
| 타구 거리 | `hit_distance_sc` | 과거 타격 성과 |
| 기대 타율 | `estimated_ba_using_speedangle` | 과거 집계 |
| 기대 wOBA | `estimated_woba_using_speedangle` | 과거 집계 |
| wOBA 값 | `woba_value` | 과거 집계 |
| BABIP 값 | `babip_value` | 과거 집계 |
| ISO 값 | `iso_value` | 과거 집계 |
| 승리확률 변화 | `delta_home_win_exp` | LI 표 구축 |
| 득점기대 변화 | `delta_run_exp` | 과거 성과 분석 |
| 투구 후 점수 | `post_home_score` 등 | 결과 검증 |

---

# 10. 스트라이크 여부 예측 시 데이터 누수

## 10.1 기본 예측 시점

본 문서의 기본 예측 시점은 **현재 투구가 던져지기 직전**이다.

따라서 현재 투구가 끝난 후 확정되는 결과·궤적·플레이트 통과 위치는 입력으로 사용하지 않는다.

## 10.2 반드시 제외할 현재 행 변수

```python
leakage_columns_current_pitch = [
    # 타깃 또는 타깃을 직접 표현
    "type",
    "description",

    # 스트라이크존 통과 위치를 직접 표현
    "zone",
    "plate_x",
    "plate_z",

    # 현재 투구가 진행된 후 확정되는 물리 로그
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",

    # 현재 투구·타석 결과
    "events",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "babip_value",
    "iso_value",

    # 결과가 반영된 점수·기대값
    "post_home_score",
    "post_away_score",
    "post_bat_score",
    "post_fld_score",
    "delta_home_win_exp",
    "delta_run_exp",
]
```

## 10.3 사용 가능한 형태

| 원자료 | 현재 행 | 과거·Lag 형태 |
|---|---:|---:|
| `type`, `description` | 타깃 생성 후 입력 제외 | 직전 결과·과거 스트라이크율 가능 |
| `zone`, `plate_x`, `plate_z` | 입력 금지 | 직전 위치·최근 분포 가능 |
| 현재 구속·회전·무브먼트 | 기본 설계에서는 금지 | 직전 값·최근 평균 가능 |
| 현재 WPA·득점기대 변화 | 금지 | 과거 자료로 만든 LI 가능 |
| 현재 `pitch_type` | 예측 시점에 이미 알면 가능 | 이전 구종·과거 사용률 가능 |
| 볼·스트라이크 카운트 | 가능 | 현재 투구 전 상태 |
| 이닝·점수·주자·아웃 | 가능 | 현재 투구 전 상태 |

## 10.4 실시간 TrackMan 예측은 별도 문제

공이 손을 떠난 직후 측정된 릴리스 구속·릴리스 위치를 사용해 플레이트 도달 전 스트라이크 여부를 예측하는 경우에는 일부 현재 물리값을 입력으로 사용할 수 있다.

하지만 이 경우:

- 어떤 값이 어느 시점에 실제로 उपलब्ध한지 명확히 정의해야 한다.
- `plate_x`, `plate_z`, `zone`, `type`, `description`은 여전히 사용할 수 없다.
- 사전 상황 예측 모델과 실시간 궤적 예측 모델을 분리해서 평가해야 한다.

---


# 11. 최종 권장 원본 열 목록

```python
required_columns = [
    # 식별 및 정렬
    "game_pk",
    "game_date",
    "game_year",
    "game_type",
    "home_team",
    "away_team",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "fielder_2",

    # 타깃 생성 및 구종 정보
    "type",
    "description",
    "pitch_type",
    "pitch_name",

    # 투구 전 경기 상황
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "on_1b",
    "on_2b",
    "on_3b",
    "home_score",
    "away_score",
    "bat_score",
    "fld_score",
    "stand",
    "p_throws",
    "sz_top",
    "sz_bot",
    "if_fielding_alignment",
    "of_fielding_alignment",

    # 물리 로그
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",

    # 결과 및 과거 통계 생성
    "description",
    "type",
    "events",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "babip_value",
    "iso_value",

    # LI 및 기대값
    "home_win_exp",
    "bat_win_exp",
    "delta_home_win_exp",
    "delta_run_exp",

    # 결과 검증
    "post_home_score",
    "post_away_score",
    "post_bat_score",
    "post_fld_score",
]
```

실제 반환 열은 pybaseball·Baseball Savant의 시점과 시즌에 따라 다를 수 있으므로 다음처럼 교집합만 선택한다.

```python
available_columns = [
    col for col in required_columns
    if col in df.columns
]

missing_columns = sorted(set(required_columns) - set(available_columns))

print("사용 가능:", len(available_columns))
print("누락:", missing_columns)

df_model_base = df[available_columns].copy()
```

---

# 12. 최종 파생변수 체크리스트

## DATA 01 — 경기 상황

| 변수 | 공식·방법 | 처리 |
|---|---|---|
| `score_diff_bat` | `bat_score - fld_score` | 행 계산 |
| `score_diff_home` | `home_score - away_score` | 행 계산 |
| `base_state` | \(I_1+2I_2+4I_3\) | 행 계산 |
| `runner_count` | \(I_1+I_2+I_3\) | 행 계산 |
| `risp` | \(\mathbb{1}(I_2\lor I_3)\) | 행 계산 |
| `bases_loaded` | \(I_1I_2I_3\) | 행 계산 |
| `count_state` | `balls-strikes` | 행 계산 |
| `two_strike` | \(\mathbb{1}(strikes=2)\) | 행 계산 |
| `full_count` | \(\mathbb{1}(balls=3,strikes=2)\) | 행 계산 |
| `pitcher_ahead` | 사전 정의 카운트 집합 | 행 계산 |
| `batter_ahead` | 사전 정의 카운트 집합 | 행 계산 |
| `matchup` | `p_throws + "_" + stand` | 행 계산 |
| `same_handed_matchup` | \(\mathbb{1}(p\_throws=stand)\) | 행 계산 |
| `late_inning` | \(\mathbb{1}(inning\ge7)\) | 행 계산 |
| `extra_inning` | \(\mathbb{1}(inning\ge10)\) | 행 계산 |
| `close_game` | \(\mathbb{1}(|score\_diff|\le c)\) | 행 계산 |

## DATA 02 — 경기 중요도

| 변수 | 공식·방법 | 처리 |
|---|---|---|
| `li_pa` | 상태별 평균 \(|\Delta WE|\) ÷ 전체 평균 | 별도 산출 |
| `li_pa_shrunk` | 상태 평균과 전체 평균 수축 | 별도 산출 |
| `li_pitch` | PA 상태 + 볼·스트라이크 | 별도 산출 |
| `leverage_class` | LI 구간화 | 행 계산 |
| `win_expectancy_state` | 동일 상태의 과거 승률 | 별도 산출 |

## DATA 03 — 과거 이력

| 변수 | 공식·방법 | 처리 |
|---|---|---|
| 시즌 이전 투구 수 | `groupby().cumcount()` | 누적 |
| 시즌 구종 사용률 | 과거 구종 수 ÷ 과거 총 투구 수 | 누적 |
| 최근 N구 사용률 | 이전 N구 one-hot 평균 | Rolling |
| 손잡이별 사용률 | 투수×타자 손 그룹 누적 | 누적 |
| 카운트별 사용률 | 투수×카운트 그룹 누적 | 누적 |
| 주자 상태별 사용률 | 투수×BaseState 그룹 누적 | 누적 |
| 최근 평균 구속 | 이전 N구 평균 | Rolling |
| 최근 평균 회전수 | 이전 N구 평균 | Rolling |
| 최근 평균 무브먼트 | 이전 N구 평균 | Rolling |
| 릴리스 안정성 | 이전 N구 릴리스 좌표 SD | Rolling |
| 최근-시즌 구속 차 | 최근 평균−과거 시즌 평균 | 누적+Rolling |
| 직전 경기 투구 수 | 경기별 집계 후 1경기 shift | Lag |
| 휴식일 | 현재 등판일−직전 등판일 | Lag |
| 타순 몇 바퀴째 | 경기 내 동일 타자 상대 PA 누적 | 누적 |
| 최근 타율 | 과거 H ÷ 과거 AB | Rolling |
| 최근 출루율 | \((H+BB+HBP)/(AB+BB+HBP+SF)\) | Rolling |
| 최근 장타율 | TB ÷ AB | Rolling |
| 최근 K% | K ÷ PA | Rolling |
| 최근 BB% | BB ÷ PA | Rolling |
| 헛스윙률 | Whiff ÷ Swing | 누적·Rolling |
| 맞대결 투구 수 | 투수×타자 `cumcount()` | 누적 |
| 맞대결 구종률 | 맞대결 과거 구종 수 ÷ 총 투구 수 | 누적 |
| 직전 1–3구 | 그룹별 `shift(1:3)` | Lag |
| 직전 물리값 | 그룹별 `shift(1)` | Lag |
| 직전 결과 | 그룹별 `shift(1)` | Lag |
| 연속 동일 구종 수 | 직전까지 run length | 누적 |

## DATA 04 — TrackMan 과거 로그

| 변수 | 공식·방법 | 처리 |
|---|---|---|
| 무브먼트 인치 | \(12\times pfx\) | 행 계산 |
| 정규화 높이 | \((plate_z-sz_bot)/(sz_top-sz_bot)\) | 행 계산 |
| 궤적 좌표 | \(r(t)=r_0+v_0t+\frac12at^2\) | 별도 산출 |
| 목표 평면 시간 | y축 2차방정식의 물리적 근 | 별도 산출 |
| VAA | \(\arctan(v_z/|v_y|)\) | 별도 산출 |
| HAA | \(\arctan(v_x/|v_y|)\) | 별도 산출 |
| 릴리스 분산 | \(\sqrt{SD_x^2+SD_z^2}\) | Rolling |
| 구종 물리 분리도 | 표준화 특성 간 거리 | 누적 |
| 터널 거리 | 인지 평면에서 궤적 간 거리 | 별도 산출 |

---

# 13. 최소 모델 입력안

타깃:

```python
target_column = "is_strike"
```

## 13.1 베이스라인 — 투구 전 상황만 사용

```python
baseline_features = [
    "pitcher",
    "batter",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "score_diff_bat",
    "base_state",
    "pitch_number",
    "times_faced_in_game",
]
```

## 13.2 구종이 알려진 설계

투구 직전 또는 구종 결정 이후에 예측하며 현재 구종을 알고 있다고 가정한다.

```python
pitch_known_features = baseline_features + [
    "pitch_type",
    "risp",
    "bases_loaded",
    "two_strike",
    "full_count",
    "same_handed_matchup",
    "late_inning",
    "li_pa",
]
```

## 13.3 과거 로그 포함

```python
history_features = pitch_known_features + [
    "prev_pitch_type_1",
    "prev_pitch_type_2",
    "prev_description",
    "prev_release_speed",
    "prev_plate_x",
    "prev_plate_z",
    "pitcher_pitch_count_before",
    "ff_usage_last_20",
    "sl_usage_last_20",
    "ch_usage_last_20",
    "cu_usage_last_20",
    "velo_last20_mean",
    "release_dispersion_last20",
    "prev_game_pitch_count",
    "rest_days",
]
```

## 13.4 구종 선택 전 예측

현재 구종을 모르는 시점이라면 `pitch_type`을 제외한다.

```python
pre_selection_features = [
    feature
    for feature in history_features
    if feature != "pitch_type"
]
```

## 13.5 과거 스트라이크율

투수 \(p\)의 현재 투구 이전 스트라이크율:

\[
StrikeRate_{p,i}
=
\frac{
\sum_{j<i}\mathbb{1}(type_j=S)
}{
\sum_{j<i}\mathbb{1}(type_j\in\{B,S,X\})
}
\]

```python
df["is_strike_observed"] = df["type"].eq("S").astype("int8")

df["pitcher_strike_rate_before"] = (
    df.groupby(["game_year", "pitcher"], group_keys=False)[
        "is_strike_observed"
    ]
    .apply(lambda s: s.shift(1).expanding(min_periods=10).mean())
)
```

최근 \(N\)구 스트라이크율:

\[
StrikeRate^{(N)}_{p,i}
=
\frac{1}{N_i}
\sum_{j=i-N_i}^{i-1}\mathbb{1}(type_j=S)
\]

```python
df["pitcher_strike_rate_last20"] = (
    df.groupby("pitcher", group_keys=False)["is_strike_observed"]
      .apply(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
)
```

카운트별·구종별·타자 손별 과거 스트라이크율도 동일한 방식으로 만들 수 있다.

---


# 14. 검증 설계

랜덤 분할보다 시간 분할이 안전하다.

## 권장 예시

- 학습: 2017–2018
- 테스트: 2019

또는:

- Fold 1: 2017 상반기 → 2017 하반기
- Fold 2: 2017 → 2018
- Fold 3: 2017–2018 → 2019

LI 표, 선수 누적 통계, 구종별 프로필, 인코딩 값은 각 학습 구간에서만 생성한다.

\[
\text{FeatureMap}_{test}
=
f(\text{train only})
\]

새 투수·타자 또는 표본 부족 상황은 리그 평균·투수 전체 평균·손잡이 평균으로 계층적 대체한다.

---

# 15. 핵심 결론

1. **경기 상황 정보**는 pybaseball 원본에서 대부분 직접 확보할 수 있다.
2. **경기 중요도**의 핵심인 LI는 이닝·초말·점수 차·아웃·주자 상태와 과거 승리확률 변화량을 이용해 별도로 만들어야 한다.
3. **과거 이력 정보**는 현재 행을 제외한 누적·rolling·lag 계산이 핵심이다.
4. **TrackMan 로그**의 구속·회전·무브먼트·릴리스·속도·가속도 계수는 직접 확보할 수 있다.
5. 현재 투구의 스트라이크 여부 예측에서는 현재 공의 구속·회전·궤적·위치·결과·WPA를 입력하면 안 된다.
6. 투구 전 예측에서는 현재 공의 물리값 대신 직전 투구 또는 과거 평균·분산을 사용한다.
7. 2017–2019 MLB 자료는 TrackMan 기반 연습 데이터로 적합하지만, KBO TrackMan과 좌표계·단위·지표 정의가 완전히 동일하다고 가정해서는 안 된다.

---

# 16. 참고자료

- [pybaseball GitHub 저장소](https://github.com/jldbc/pybaseball)
- [pybaseball Statcast 문서](https://github.com/jldbc/pybaseball/blob/master/docs/statcast.md)
- [Baseball Savant Statcast CSV 열 정의](https://baseballsavant.mlb.com/csv-docs)
- [MLB Pitch-tracking Era](https://www.mlb.com/glossary/miscellaneous/pitch-tracking-era)
- [FanGraphs Leverage Index 설명](https://library.fangraphs.com/misc/li/)

---

## 부록 A. 전처리 시작 코드

```python
import numpy as np
import pandas as pd

df = df[df["game_type"].eq("R")].copy()

df = df.sort_values(
    ["game_date", "game_pk", "at_bat_number", "pitch_number"]
).reset_index(drop=True)

# 기본 상황 변수
df["score_diff_bat"] = df["bat_score"] - df["fld_score"]
df["score_diff_home"] = df["home_score"] - df["away_score"]

i1 = df["on_1b"].notna().astype("int8")
i2 = df["on_2b"].notna().astype("int8")
i3 = df["on_3b"].notna().astype("int8")

df["base_state"] = i1 + 2 * i2 + 4 * i3
df["runner_count"] = i1 + i2 + i3
df["risp"] = ((i2 == 1) | (i3 == 1)).astype("int8")
df["bases_loaded"] = ((i1 == 1) & (i2 == 1) & (i3 == 1)).astype("int8")

df["two_strike"] = df["strikes"].eq(2).astype("int8")
df["full_count"] = (
    df["balls"].eq(3) & df["strikes"].eq(2)
).astype("int8")

df["matchup"] = df["p_throws"] + "_" + df["stand"]
df["same_handed_matchup"] = (
    df["p_throws"].eq(df["stand"])
).astype("int8")

df["late_inning"] = df["inning"].ge(7).astype("int8")
df["extra_inning"] = df["inning"].ge(10).astype("int8")
df["close_game"] = df["score_diff_bat"].abs().le(2).astype("int8")

# 단위 변환
df["pfx_x_in"] = 12 * df["pfx_x"]
df["pfx_z_in"] = 12 * df["pfx_z"]

# 투수의 이전 투구 수
df["pitcher_pitch_count_before"] = (
    df.groupby(["game_year", "pitcher"]).cumcount()
)

# 직전 구종 및 직전 물리값
game_pitcher = df.groupby(["game_pk", "pitcher"], sort=False)

df["prev_pitch_type_1"] = game_pitcher["pitch_type"].shift(1)
df["prev_pitch_type_2"] = game_pitcher["pitch_type"].shift(2)
df["prev_pitch_type_3"] = game_pitcher["pitch_type"].shift(3)

for col in [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "release_pos_x",
    "release_pos_z",
]:
    df[f"prev_{col}"] = game_pitcher[col].shift(1)

# 최근 투수 물리 평균
for col in [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
]:
    df[f"{col}_last20_mean"] = (
        df.groupby("pitcher", group_keys=False)[col]
          .apply(
              lambda s:
              s.shift(1).rolling(20, min_periods=5).mean()
          )
    )

# 타깃 생성용 type 결측은 데이터셋화 단계에서 제거
model_df = df[df["type"].notna()].copy()
model_df["is_strike"] = model_df["type"].eq("S").astype("int8")
```
