"""V11: submit_026.zip — 성분 결합을 F 행에도 적용한다.

V10 실측 (Val2024, 20시드, 프로덕션 836.503 대비)
    w_R\\w_F    0.00    0.05    0.10    0.15    0.20
       0.20   +17.65  +18.90  +20.00  +20.94  +21.72
       0.30   +20.08  +21.34  +22.44  +23.37  +24.15

    균일 w=0.20 : +21.722  t_row 5.55   R 854.08  F 563.44
    R한정(025)  : +17.648  t_row 4.83   R 854.08  F 528.81
    차이 +4.075, F 개선 +34.62

왜 뒤집혔나
    P12 에서는 4성분(m, r, o, mr)으로 F 에 적용하면 -8.01 손해라 R 한정으로 갔다.
    V6 의 OUTSIDE 분할(5성분: m, r, mr, ob, oz) 이후 F 가 개선으로 바뀌었다.
    2023 F 붕괴는 실패 유형 구성 변화였는데 ball=1 / ball=0 을 쪼개니 모델이
    F 의 다른 실패 조합을 잡을 손잡이를 얻었다.

가중치 결정
    R 한정이었던 이유가 "F 가 손상되니까"였는데 이제 개선되므로 특별 취급할
    근거가 사라졌다. 따라서 w_F = w_R = 0.20 균일 적용.
    w_R 은 사전 등록값 0.20 유지 (그리드는 0.30 에서 +24.15 지만 그 점에 맞추지 않는다).
    w_F = 0.20 은 그리드 끝이므로 그 값 자체를 고른 게 아니라 균일성 원칙의 결과다.

모델은 submit_025 와 완전히 동일하다. script.py 와 metadata.json 만 바뀐다.
출력: submit/2026-08-15/submit_026.zip
"""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submit" / "2026-08-14" / "submit_025.zip"
OUT = ROOT / "submit" / "2026-08-15" / "submit_026.zip"
W_R, W_F = 0.20, 0.20

if OUT.exists():
    raise FileExistsError(OUT)
assert len(OUT.name) < 30

OLD_BLOCK = '''    weight = float(cfg["weight"])
    mask = (test["game_type"].astype(str).to_numpy() == cfg["apply_game_type"])
    blended = prediction.copy()
    blended[mask] = weight * union[mask] + (1.0 - weight) * prediction[mask]
    return np.clip(blended, 1e-6, 1.0 - 1e-6)'''

NEW_BLOCK = '''    weights = cfg["weight_by_game_type"]
    default_weight = float(cfg["weight_default"])
    game_type = test["game_type"].astype(str).to_numpy()
    weight = np.full(len(game_type), default_weight, dtype=float)
    for key, value in weights.items():
        weight[game_type == key] = float(value)
    blended = weight * union + (1.0 - weight) * prediction
    return np.clip(blended, 1e-6, 1.0 - 1e-6)'''

with ZipFile(SRC) as z:
    script = z.read("script.py").decode("utf-8")
    metadata = json.loads(z.read("model/metadata.json"))
    payload = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}

assert script.count(OLD_BLOCK) == 1, "혼합 블록을 찾지 못했다"
script = script.replace(OLD_BLOCK, NEW_BLOCK, 1)

cb = metadata["component_blend"]
cb.pop("weight", None)
cb.pop("apply_game_type", None)
cb["weight_by_game_type"] = {"R": W_R, "F": W_F}
cb["weight_default"] = W_R
cb["validation_val2024"] = {
    "base": "submit_021", "base_bss": 836.503,
    "delta_bss": 21.722, "t_row": 5.55,
    "r_bss": 854.083, "f_bss": 563.437,
    "prev_submit_025": {"delta_bss": 17.648, "t_row": 4.83, "f_bss": 528.814},
    "weight_policy": ("w_R fixed 0.20 pre-registered; w_F set equal to w_R because "
                      "the R-only restriction existed only to protect F, and the "
                      "5-component OUTSIDE split reversed that damage into a gain"),
    "seed_saturation": "8 vs 20 seeds differ by +0.05 BSS; 8 is sufficient",
}
metadata["track"] = "reverse20_s0475_component5_uniform"
metadata["version"] = 13

payload["script.py"] = script.encode("utf-8")
payload["model/metadata.json"] = json.dumps(metadata, indent=1,
                                            sort_keys=True).encode("utf-8")


def info(name):
    zi = ZipInfo(name)
    zi.date_time = (2026, 8, 15, 0, 0, 0)
    zi.compress_type = ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    return zi


OUT.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
    for name in sorted(payload):
        assert "\\" not in name
        z.writestr(info(name), payload[name])

with ZipFile(SRC) as a, ZipFile(OUT) as b:
    ca = {i.filename: i.CRC for i in a.infolist()}
    cb2 = {i.filename: i.CRC for i in b.infolist()}
    changed = sorted(k for k in ca if ca[k] != cb2[k])
print(f"생성 완료 {OUT}  {OUT.stat().st_size/2**20:.1f}MB  파일 {len(payload)}개")
print(f"  025 대비 변경: {changed}")
print(f"  나머지 {len(ca)-len(changed)}개 멤버 CRC 동일")
