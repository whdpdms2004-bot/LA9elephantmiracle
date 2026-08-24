"""V28: submit_030.zip — 깨끗한 재빌드.

발견한 결함
    submit_025 -> 027 -> 028 -> 029 를 이어서 빌드하면서 매번 component_features /
    component_blend 를 주입했는데, 훅 지점(검증 줄)이 계속 남아 있어 다음 빌드가
    또 앞에 끼워 넣었다. 결과:

        submit_025  정의 1 / 훅 1    정상
        submit_026  정의 1 / 훅 1    정상 (025 의 블록을 치환만 함)
        submit_027  정의 2 / 훅 2
        submit_028  정의 3 / 훅 3
        submit_029  정의 4 / 훅 4    1325,1327,1329,1331 행

    훅이 4번이면 혼합이 4번 적용되어 유효 가중치가
        w_eff = 1 - (1 - 0.25)^4 = 0.684
    가 된다. 의도한 0.25 가 아니다. Val2024 측정치는 Python 에서 1회 적용으로
    낸 값이라 유효하지만 ZIP 이 그 구성을 구현하지 않았다.
    6단계 검증은 라벨이 없어 BSS 를 못 재므로 이걸 잡지 못했다.

이번 빌드
    깨끗한 submit_026 을 base 로 삼아 주입을 딱 한 번만 한다.
    성분 모델 81개(c29_*, spec, 룩업 4종)는 submit_029 에서 그대로 가져온다.
    026 의 c25_* 는 제거한다.

    설정: 5성분 + 플래툰 4종(투수/타자/카운트/이닝) + 균일 w=0.25 + 절편 0.0077
    Val2024 예상 +40.73 (V26 K3_intercept)

출력: submit/2026-08-15/submit_030.zip
"""
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "submit" / "2026-08-15" / "submit_026.zip"
DONOR = ROOT / "submit" / "2026-08-15" / "submit_029.zip"
OUT = ROOT / "submit" / "2026-08-15" / "submit_030.zip"
C0, W = 0.0077, 0.25
NL = chr(10)

if OUT.exists():
    raise FileExistsError(OUT)
assert len(OUT.name) < 30

with ZipFile(BASE) as z:
    script = z.read("script.py").decode("utf-8")
    metadata = json.loads(z.read("model/metadata.json"))
    payload = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}

# base 가 깨끗한지 확인
assert script.count("def component_features") == 1, "base 에 정의가 1개가 아니다"
assert script.count("prediction = component_blend") == 1, "base 에 훅이 1개가 아니다"

# 029 에서 성분 산출물을 가져오고 025 세대는 버린다
with ZipFile(DONOR) as d:
    donor = {n: d.read(n) for n in d.namelist()
             if "/c29_" in n or n.endswith(("platoon_2025.csv", "c29_spec.json"))}
for stale in [n for n in payload if "/c25_" in n or "/comp_" in n
              or n.endswith("platoon_2025.csv")]:
    del payload[stale]
payload.update(donor)
print(f"성분 산출물 {len(donor)}개 이식 (submit_029 에서)")

# ---- 029 의 최신 component_features / component_blend 만 추출해 교체
with ZipFile(DONOR) as d:
    dscript = d.read("script.py").decode("utf-8")
starts = [m.start() for m in re.finditer(r"^COMP_RATES|^C25_RATES|^C27_RATES|^C29_RATES",
                                         dscript, re.M)]
assert starts, "주입 블록 시작을 찾지 못했다"
blk_start = starts[-1]
blk_end = dscript.index(NL + "def main():", blk_start)
newest = dscript[blk_start:blk_end]
assert newest.count("def component_features") == 1
assert "inning_platoon" in newest and "count_platoon" in newest
print(f"최신 주입 블록 {len(newest.splitlines())}줄 추출")

# base 의 옛 블록을 찾아 교체
bstarts = [m.start() for m in re.finditer(r"^COMP_RATES|^C25_RATES", script, re.M)]
assert len(bstarts) == 1, f"base 주입 블록이 {len(bstarts)}개"
bend = script.index(NL + "def main():", bstarts[0])
script = script[:bstarts[0]] + newest + script[bend:]

# 절편 적용
OLD_U = NL.join([
    '    union = np.clip(1.0 - (parts["m"] + parts["r"] - parts["mr"]',
    '                           + parts["ob"] + parts["oz"]), 1e-7, 1.0 - 1e-7)'])
NEW_U = NL.join([
    '    union = np.clip(1.0 - (parts["m"] + parts["r"] - parts["mr"]',
    '                           + parts["ob"] + parts["oz"])',
    '                    - float(cfg["composite_intercept"]), 1e-7, 1.0 - 1e-7)'])
assert script.count(OLD_U) == 1, f"합성식 {script.count(OLD_U)}개"
script = script.replace(OLD_U, NEW_U, 1)

assert script.count("def component_features") == 1
assert script.count("def component_blend") == 1
assert script.count("prediction = component_blend") == 1
print(f"script.py 정리 완료  {len(script.splitlines())}줄, 정의 1개, 훅 1개")

cb = json.loads(DONOR.read_bytes() and ZipFile(DONOR).read(
    "model/metadata.json"))["component_blend"]
cb["weight_by_game_type"] = {"R": W, "F": W}
cb["weight_default"] = W
cb["composite_intercept"] = C0
cb["formula"] = ("P(success) = 1 - (p_m + p_r - p_mr + p_ob + p_oz) "
                 "- composite_intercept")
cb["intercept_source"] = (
    "least squares on forward-chained OOF folds 2022 and 2023 only; validation "
    "season 2024 never used. one parameter. per-component coefficients were also "
    "fitted but overfit (c_mr -1.00 -> -2.54, Val2024 -11 BSS) and were rejected")
cb["validation_val2024"] = {
    "base": "submit_021", "base_bss": 836.503, "delta_bss": 40.73, "t_row": 7.66,
    "weight_policy": ("w uniform 0.25 = largest weight positive on all three folds "
                      "(2022 +52.23 / 2023 +3.38 / 2024 +39.25); 0.30 turns 2023 "
                      "negative and expected value goes to -1.08"),
    "ablation": {"pitcher_platoon": 13.75, "count_platoon": 8.44,
                 "f_row_weight": 4.07, "inning_platoon": 2.98,
                 "outside_split": 2.64, "batter_platoon": 2.44,
                 "catboost_family": 1.23, "per_component_params": 0.91,
                 "composite_intercept": 1.37},
    "rebuild_note": ("submit_027/028/029 stacked the injection block and duplicated "
                     "the hook, applying the blend 2/3/4 times (w_eff up to 0.684). "
                     "this package injects once from the clean submit_026 base"),
}
metadata["component_blend"] = cb
metadata["track"] = "reverse20_s0475_comp5_uniform_platoon4_cal"
metadata["version"] = 17

payload["script.py"] = script.encode("utf-8")
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
print(f"{NL}생성 완료 {OUT}  {OUT.stat().st_size/2**20:.1f}MB  파일 {len(payload)}개")
print(f"  ZIP 루트: {sorted({n.split('/')[0] for n in payload})}")
print(f"  w = {W} (균일), composite_intercept = {C0}")
