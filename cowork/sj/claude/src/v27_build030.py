"""V27: submit_030.zip — 합성식 절편 보정. [실패 — v28 로 대체됨]

    이 스크립트는 55행의 assert 에서 멈췄다. submit_029 안에 합성식이 4벌 있었기
    때문이다. 그 실패가 훅 중복 주입 결함을 드러냈다(V29 참조). submit_029 를 base
    로 삼는 것 자체가 틀렸으므로 v28_build030_clean.py 로 다시 만들었다.
    기록용으로 남긴다.

V26 (Val2024, w=0.25, 프로덕션 836.503 대비)
    K0_fixed      +39.36  합성단독 765.37  corr 0.8323  pred_mean 0.48795
    K1_free       +28.38  합성단독 699.50  corr 0.7775  <- 성분별 계수 과적합
    K3_intercept  +40.73  합성단독 782.75  corr 0.8321  pred_mean 0.48603

    K1 은 OOF(2022+2023) 최소제곱으로 성분별 계수를 적합한 것인데 -11점이다.
    c_mr 이 -1.00 -> -2.54 로 2.5배가 되는 과적합이 났다. 기각.

    K3 은 계수를 고정하고 절편만 적합한다. c0 = 0.0077.
        편향 +0.185%p -> -0.008%p
        벌점 401,000 * 0.00185^2 = 1.37 BSS,  실측 +1.366
    팀 벌점 공식과 소수점까지 맞는다.

식
    P(success) = 1 - (p_m + p_r - p_mr + p_ob + p_oz) - c0

    c0 는 2022/2023 순방향 OOF 에서만 적합했고 검증 시즌(2024)을 쓰지 않았다.
    파라미터 1개이고 메커니즘이 벌점 공식으로 확인되므로 과적합 위험이 낮다.

모델은 submit_029 와 완전히 동일하다. script.py 와 metadata.json 만 바뀐다.
출력: submit/2026-08-15/submit_030.zip
"""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submit" / "2026-08-15" / "submit_029.zip"
OUT = ROOT / "submit" / "2026-08-15" / "submit_030.zip"
C0 = 0.0077
NL = chr(10)

if OUT.exists():
    raise FileExistsError(OUT)
assert len(OUT.name) < 30

# 여러 줄 리터럴을 쓰지 않는다. 이 소스가 CRLF 로 저장되면 매칭이 깨진다.
OLD = NL.join([
    '    union = np.clip(1.0 - (parts["m"] + parts["r"] - parts["mr"]',
    '                           + parts["ob"] + parts["oz"]), 1e-7, 1.0 - 1e-7)',
])
NEW = NL.join([
    '    union = np.clip(1.0 - (parts["m"] + parts["r"] - parts["mr"]',
    '                           + parts["ob"] + parts["oz"])',
    '                    - float(cfg["composite_intercept"]), 1e-7, 1.0 - 1e-7)',
])

with ZipFile(SRC) as z:
    script = z.read("script.py").decode("utf-8")
    metadata = json.loads(z.read("model/metadata.json"))
    payload = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}

assert script.count(OLD) == 1, "합성식을 찾지 못했다"
script = script.replace(OLD, NEW, 1)

cb = metadata["component_blend"]
cb["composite_intercept"] = C0
cb["formula"] = ("P(success) = 1 - (p_m + p_r - p_mr + p_ob + p_oz) "
                 "- composite_intercept")
cb["intercept_source"] = (
    "least squares on forward-chained OOF folds 2022 and 2023 only; validation "
    "season 2024 never used. one parameter. per-component coefficients were also "
    "fitted but overfit (c_mr -1.00 -> -2.54, Val2024 -11 BSS) and were rejected")
cb["validation_val2024"]["delta_bss"] = 40.73
cb["validation_val2024"]["t_row"] = 7.66
cb["validation_val2024"]["prev_submit_029_delta"] = 39.36
cb["validation_val2024"]["ablation"]["composite_intercept"] = 1.37
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

with ZipFile(SRC) as a, ZipFile(OUT) as b:
    ca = {i.filename: i.CRC for i in a.infolist()}
    cb2 = {i.filename: i.CRC for i in b.infolist()}
    changed = sorted(k for k in ca if ca[k] != cb2[k])
print(f"생성 완료 {OUT}  {OUT.stat().st_size/2**20:.1f}MB  파일 {len(payload)}개")
print(f"  029 대비 변경: {changed}")
print(f"  나머지 {len(ca)-len(changed)}개 멤버 CRC 동일")
print(f"  composite_intercept = {C0}")
