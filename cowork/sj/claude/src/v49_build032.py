"""V49: submit_032.zip — 구간 벡터를 규칙이 아니라 최적화로 푼 값.

경로 (V44~V48, "변수 선택 기준 다 버리고 조합으로 평가")
    V44  2024 캐시 31종으로 탐욕 앙상블   890.27  (+53.8)
    V45  해석적 NNLS + K 제약            K=1 이 이미 상한 근처
    V46  분해 — 이득은 라인이 아니라 가중치  라인 교체 +1.98, 가중치 +11.11
    V47  세 fold 풀링으로 다시 풀기        w 가 0.230 으로 회귀
    V48  구간 벡터 자유 최적화             아래 표

    2024 한 fold 만 보고 크게 나온 것들은 세 fold 를 보여주면 전부 사라졌다.
    라인 단순 평균도 세 fold 전부에서 이득 0 (두 라인 사이를 정확히 보간).
    남은 유일한 실질 자유도가 구간 벡터였다.

구간 벡터 후보 (성분 라인은 현행 R0 고정)
    벡터                                Δ2022    Δ2023    Δ2024     최악       합
    [.25 .25 .25 .25 .45]  submit_031  +58.21   +61.73   +39.38   +39.38  +159.33
    [.256 .249 .275 .321 .407] 자유해   +61.66   +40.98   +41.08   +40.98  +143.71
    [.25 .25 .25 .30 .40]  단순 C      +60.57   +47.87   +40.73   +40.73  +149.17   <- 채택
    [.25 .25 .25 .25 .25]  submit_030  +51.91   +23.61   +38.27   +23.61  +113.79

    단순 C 를 고른 이유
      - 2022, 2024, 최악 세 지표 전부 submit_031 보다 좋다
      - 2023 만 진다. 그런데 2023 base 는 시즌 offset 이 없는 enhanced 25종
        평균이라 −140 으로 불구다(V32). 프로덕션 base 였다면 +60 이었다.
        2024 가 실제 프로덕션 base 를 쓰는 유일한 fold 다.
      - 자유해를 그대로 쓰지 않고 두 구간만 움직이는 형태로 단순화했다.
        자유해 대비 최악 −0.25, 합 +5.46 이다.

기대값을 정직하게
    submit_031 대비 2024 +1.35 다. Public 이 가중치 조정의 전이율을 4% 로
    못박았으므로(029 w_eff 0.684 963 vs 030 w 0.25 964) 실제 차이는 0 에 가깝다.
    비용이 0 이라 싣지만, 이걸로 점수가 오를 것이라고 보지 않는다.

출력: submit/2026-08-15/submit_032.zip
"""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submit" / "2026-08-15" / "submit_031.zip"
OUT = ROOT / "submit" / "2026-08-15" / "submit_032.zip"
BW = [0.25, 0.25, 0.25, 0.30, 0.40]

if OUT.exists():
    raise FileExistsError(OUT)
assert len(OUT.name) < 30

with ZipFile(SRC) as z:
    script = z.read("script.py").decode("utf-8")
    metadata = json.loads(z.read("model/metadata.json"))
    payload = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}

for tag in ["def component_features", "def component_blend",
            "prediction = component_blend"]:
    assert script.count(tag) == 1, f"'{tag}' {script.count(tag)}회"
assert 'cfg["weight_by_volume"]' in script, "구간 가중치 코드가 없다"

cb = metadata["component_blend"]
assert cb["volume_cuts"] == [100, 500, 2000, 4000]
old = list(cb["weight_by_volume"])
cb["weight_by_volume"] = BW
cb["weight_source"] = (
    "V48: per-bucket vector fitted directly on all three forward-chained folds "
    "(minimax over folds), then simplified to move only two buckets. no rule-based "
    "gating was applied - the vector is the optimizer's answer. "
    "delta vs base: 2022 +60.57, 2023 +47.87, 2024 +40.73. worst fold +40.73, "
    "which beats submit_031 (+39.38) and submit_030 (+23.61). "
    "2023 is the only fold where submit_031 wins, and its base is the enhanced-25 "
    "average without the season offset (raw BSS -140), so leaning on the component "
    "line is over-rewarded there. 2024 is the only fold whose base is the real "
    "production stack")
cb["search_log"] = (
    "V44 greedy ensemble over 31 cached component lines reached 890.27 on 2024 "
    "(+53.8) but V47 showed the gain vanishes once all three folds are in the fit "
    "(w returns to 0.230). V46 decomposed it: line swap worth +1.98, weight fitting "
    "+11.11. a plain average of two component lines interpolates exactly between "
    "them on every fold - no ensemble gain. the bucket vector is the only real "
    "remaining degree of freedom")
cb["validation_val2024"]["bucket_weight_delta"] = {
    "2022": 60.57, "2023": 47.87, "2024": 40.73, "worst_fold": 40.73}
cb["validation_val2024"]["expected_public_effect"] = (
    "near zero. public evidence pins blend-weight transfer at about 4 percent "
    "(submit_029 w_eff 0.684 scored 963 vs submit_030 w 0.25 scored 964), so a "
    "+1.35 internal difference over submit_031 should not move the score. shipped "
    "because it costs nothing, not because it is expected to gain")
metadata["track"] = "reverse20_s0475_comp5_platoon4_cal_volw_opt"
metadata["version"] = 19

payload["model/metadata.json"] = json.dumps(metadata, indent=1,
                                            sort_keys=True).encode("utf-8")


def info(name):
    zi = ZipInfo(name)
    zi.date_time = (2026, 8, 15, 0, 0, 0)
    zi.compress_type = ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    return zi


with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
    for name in sorted(payload):
        assert chr(92) not in name
        z.writestr(info(name), payload[name])

with ZipFile(SRC) as a, ZipFile(OUT) as b:
    ca = {i.filename: i.CRC for i in a.infolist()}
    cb2 = {i.filename: i.CRC for i in b.infolist()}
    changed = sorted(k for k in ca if ca[k] != cb2[k])
print(f"생성 완료 {OUT}  {OUT.stat().st_size/2**20:.1f}MB  파일 {len(payload)}개")
print(f"  031 대비 변경: {changed}")
print(f"  나머지 {len(ca)-len(changed)}개 멤버 CRC 동일  (script.py 포함 무변경)")
print(f"  구간 가중치 {old} -> {BW}")
