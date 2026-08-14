"""kirby_index.py 검증 테스트.

실데이터 없이 물리 시뮬레이션으로 검증한다:
1. 알려진 릴리스 각도로 궤적을 생성(정방향) → add_release_angles가 각도를 복원하는가
2. 각도+릴리스 좌표만으로 플레이트 위치가 거의 완전히 설명되는가 (원문 R^2 재현)
3. 반복성이 좋은 투수가 Kirby Index 상위에 오는가
4. add_command_features가 과거 투구만 사용하는가 (누수 검사)

실행: python test_kirby_index.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kirby_index import (
    add_command_features,
    add_release_angles,
    kirby_index_table,
    location_model_report,
)

RNG = np.random.default_rng(42)
Y_PLATE = 17.0 / 12.0
GRAVITY = -32.174


def simulate_pitches(
    n: int,
    vra_mean: float,
    hra_mean: float,
    vra_sd: float,
    hra_sd: float,
    release_sd: float = 0.05,
    pitcher_id: int = 1,
) -> pd.DataFrame:
    """릴리스 각도/좌표 분포에서 투구를 생성해 Statcast 형식 행을 만든다.

    Statcast 규약대로 vx0/vy0/vz0/ax/ay/az는 y=50ft 평면 기준으로 기록한다.
    """
    vra = np.radians(RNG.normal(vra_mean, vra_sd, n))
    hra = np.radians(RNG.normal(hra_mean, hra_sd, n))
    x_r = RNG.normal(-1.5, release_sd, n)
    z_r = RNG.normal(6.0, release_sd, n)
    y_r = RNG.normal(54.0, 0.02, n)
    speed = RNG.normal(139.0, 2.0, n)              # ~95mph (ft/s)

    # 각도 → 릴리스 속도 벡터 (vra = atan(vz/|vy|), hra = atan(vx/|vy|))
    vy_r = -speed / np.sqrt(1 + np.tan(vra) ** 2 + np.tan(hra) ** 2)
    vz_r = -vy_r * np.tan(vra)
    vx_r = -vy_r * np.tan(hra)

    ax = RNG.normal(-6.0, 1.0, n)                  # 무브먼트+항력
    ay = RNG.normal(28.0, 1.5, n)
    az = GRAVITY + RNG.normal(14.0, 1.5, n)        # 중력+백스핀 양력

    def t_to_plane(y_from, vy, a, y_to):
        # 0.5*a*t^2 + vy*t + (y_from - y_to) = 0, 물리적(양수) 근
        disc = vy * vy - 2 * a * (y_from - y_to)
        return (-vy - np.sqrt(disc)) / a

    # 릴리스 → y=50 평면 (Statcast 파라미터 기준점)
    t50 = t_to_plane(y_r, vy_r, ay, 50.0)
    vx0 = vx_r + ax * t50
    vy0 = vy_r + ay * t50
    vz0 = vz_r + az * t50

    # y=50 평면 → 홈플레이트 (검증용 실제 위치)
    tp = t_to_plane(50.0, vy0, ay, Y_PLATE)
    x50 = x_r + vx_r * t50 + 0.5 * ax * t50**2
    z50 = z_r + vz_r * t50 + 0.5 * az * t50**2
    plate_x = x50 + vx0 * tp + 0.5 * ax * tp**2
    plate_z = z50 + vz0 * tp + 0.5 * az * tp**2

    return pd.DataFrame({
        "pitcher": pitcher_id, "player_name": f"Pitcher {pitcher_id}",
        "game_year": 2017, "pitch_type": "FF",
        "vx0": vx0, "vy0": vy0, "vz0": vz0, "ax": ax, "ay": ay, "az": az,
        "release_pos_x": x_r, "release_pos_y": y_r, "release_pos_z": z_r,
        "release_extension": 60.5 - y_r,
        "plate_x": plate_x, "plate_z": plate_z,
        "_true_vra_deg": np.degrees(vra), "_true_hra_deg": np.degrees(hra),
    })


def test_angle_recovery() -> None:
    df = simulate_pitches(2000, vra_mean=-2.0, hra_mean=1.0, vra_sd=0.5, hra_sd=0.5)
    out = add_release_angles(df)
    err_v = np.abs(out["vra_deg"] - out["_true_vra_deg"]).max()
    err_h = np.abs(out["hra_deg"] - out["_true_hra_deg"]).max()
    assert err_v < 1e-3 and err_h < 1e-3, f"angle recovery failed: {err_v=}, {err_h=}"
    print(f"PASS 1. 각도 복원  (최대 오차 VRA {err_v:.2e}°, HRA {err_h:.2e}°)")


def test_location_r2() -> None:
    frames = [
        simulate_pitches(1500, RNG.normal(-2, 0.3), RNG.normal(1, 0.3),
                         vra_sd=0.45, hra_sd=0.5, pitcher_id=i)
        for i in range(1, 13)
    ]
    df = add_release_angles(pd.concat(frames, ignore_index=True))
    report = location_model_report(df, n_estimators=100)
    assert report["r2_plate_z"] > 0.85, report
    assert report["r2_plate_x"] > 0.80, report
    print(f"PASS 2. 위치 모델  (R² 수직 {report['r2_plate_z']:.3f}, "
          f"수평 {report['r2_plate_x']:.3f}) — 원문 0.92/0.85와 부합")


def test_index_ranking() -> None:
    tight = simulate_pitches(600, -2.0, 1.0, vra_sd=0.25, hra_sd=0.28, pitcher_id=1)   # Kirby형
    mid = simulate_pitches(600, -2.2, 0.8, vra_sd=0.45, hra_sd=0.50, pitcher_id=2)
    loose = simulate_pitches(600, -1.8, 1.2, vra_sd=0.75, hra_sd=0.85, pitcher_id=3)   # Sandoval형
    df = add_release_angles(pd.concat([tight, mid, loose], ignore_index=True))
    table = kirby_index_table(df, min_pitches=100)
    order = table.sort_values("kirby_index", ascending=False)["pitcher"].tolist()
    assert order == [1, 2, 3], f"ranking wrong: {order}"
    print(f"PASS 3. 랭킹      (tight→loose 순서 유지: {order})")


def test_no_leakage() -> None:
    df = simulate_pitches(300, -2.0, 1.0, vra_sd=0.4, hra_sd=0.4)
    df["game_date"] = pd.Timestamp("2017-05-01")
    df["game_pk"] = 1
    df["at_bat_number"] = np.repeat(np.arange(60), 5)
    df["pitch_number"] = np.tile(np.arange(1, 6), 60)
    out = add_command_features(df, window=40, min_periods=10)

    # i번째 행의 피처는 i번째 투구의 각도와 무관해야 한다:
    # 마지막 투구의 각도를 크게 바꿔도 해당 행의 피처가 변하지 않아야 함.
    tampered = df.copy()
    tampered.loc[tampered.index[-1], ["vz0", "vx0"]] *= 3.0
    out2 = add_command_features(tampered, window=40, min_periods=10)
    col = "cmd_vra_sd_last40"
    same = np.isclose(
        out[col].iloc[-1], out2[col].iloc[-1], equal_nan=True
    )
    assert same, "current-pitch leakage detected in rolling command feature"
    print("PASS 4. 누수 없음  (현재 투구 변조가 자신의 피처에 영향 없음)")


if __name__ == "__main__":
    test_angle_recovery()
    test_location_r2()
    test_index_ranking()
    test_no_leakage()
    print("\n모든 테스트 통과 — 실데이터 수집 후 python kirby_index.py 실행 가능")
