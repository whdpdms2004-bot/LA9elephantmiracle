"""V41: submit_031.zip — 구간별 결합 가중치 (V30 W1).

V30 결과 (세 fold 전부 W0 이상, 규칙 통과)
    구간(asof_pitcher_n)   평균격차     w
    0-99                   +1819.7    0.25
    100-499                +1182.1    0.25
    500-1999                +596.8    0.25
    2000-3999               +386.7    0.25
    4000+                    -40.0    0.45   <- 성분 라인이 base 를 이기는 유일한 구간

    W1 − W0 = 2022 +6.02 / 2023 +38.19 / 2024 +0.94

    2024 이득이 +0.94 뿐이라 점수 상승용이 아니다. 2023 같은 레짐 변화 해에서
    +38 을 지키는 보험이다. 재학습이 없고 asof_pitcher_n 은 입력 피처이므로
    비용이 0 이다.

submit_030 대비 바뀌는 것
    script.py 의 component_blend 에서 가중치 계산만. 모델은 완전히 동일하다.
    game_type 별 가중치는 R/F 둘 다 0.25 였으므로 구간 가중치로 대체한다.
    (V10 에서 F행은 '학습 가중치' 0.20 으로 처리했고 결합 가중치는 아니었다.)

행 독립성
    asof_pitcher_n 은 각 test 행에 이미 들어 있는 열이다. 다른 test 행을
    참조하지 않는다. 6단계 검증 6번 항목이 이를 기계로 확인한다.

출력: submit/2026-08-15/submit_031.zip
"""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "submit" / "2026-08-15" / "submit_030.zip"
OUT = ROOT / "submit" / "2026-08-15" / "submit_031.zip"
CUTS = [100, 500, 2000, 4000]
BW = [0.25, 0.25, 0.25, 0.25, 0.45]
NL = chr(10)

if OUT.exists():
    raise FileExistsError(OUT)
assert len(OUT.name) < 30

OLD = NL.join([
    '    weights = cfg["weight_by_game_type"]',
    '    default_weight = float(cfg["weight_default"])',
    '    game_type = test["game_type"].astype(str).to_numpy()',
    '    weight = np.full(len(game_type), default_weight, dtype=float)',
    '    for key, value in weights.items():',
    '        weight[game_type == key] = float(value)',
])
NEW = NL.join([
    '    default_weight = float(cfg["weight_default"])',
    '    cuts = [float(v) for v in cfg["volume_cuts"]]',
    '    bw = [float(v) for v in cfg["weight_by_volume"]]',
    '    n = pd.to_numeric(test["asof_pitcher_n"], errors="coerce").to_numpy()',
    '    n = np.where(np.isfinite(n), n, 0.0)',
    '    bucket = np.digitize(n, cuts)',
    '    weight = np.full(len(n), default_weight, dtype=float)',
    '    for k, value in enumerate(bw):',
    '        weight[bucket == k] = value',
])

with ZipFile(SRC) as z:
    script = z.read("script.py").decode("utf-8")
    metadata = json.loads(z.read("model/metadata.json"))
    payload = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}

assert script.count(OLD) == 1, f"가중치 블록 {script.count(OLD)}개"
script = script.replace(OLD, NEW, 1)

# 구조 불변식 — submit_027~029 의 결함을 막는 검사
for tag in ["def component_features", "def component_blend",
            "prediction = component_blend"]:
    assert script.count(tag) == 1, f"'{tag}' {script.count(tag)}회"
assert "weight_by_game_type" not in script

cb = metadata["component_blend"]
cb.pop("weight_by_game_type", None)
cb["volume_cuts"] = CUTS
cb["weight_by_volume"] = BW
cb["weight_key"] = "asof_pitcher_n"
cb["weight_source"] = (
    "V30: per-bucket largest weight positive on all three forward-chained folds. "
    "only the 4000+ bucket moves (0.25 -> 0.45) because it is the only bucket where "
    "the component line beats the production base (mean gap -40.0 vs +386 to +1820 "
    "elsewhere). the 2024-optimal per-bucket vector was rejected: it scores -24.16 "
    "on 2023")
cb["validation_val2024"]["bucket_weight_delta"] = {
    "2022": 6.02, "2023": 38.19, "2024": 0.94}
cb["validation_val2024"]["note"] = (
    "bucket weighting is regime insurance, not a scoring change. public evidence "
    "(submit_029 w_eff 0.684 -> 963 vs submit_030 w 0.25 -> 964) shows blend-weight "
    "changes transfer at about 4 percent. feature changes transfer at about 137 "
    "percent (submit_024 -> submit_030: internal +13.86, public +19)")
metadata["track"] = "reverse20_s0475_comp5_platoon4_cal_volw"
metadata["version"] = 18

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

with ZipFile(SRC) as a, ZipFile(OUT) as b2:
    ca = {i.filename: i.CRC for i in a.infolist()}
    cb2 = {i.filename: i.CRC for i in b2.infolist()}
    changed = sorted(k for k in ca if ca[k] != cb2[k])
print(f"생성 완료 {OUT}  {OUT.stat().st_size/2**20:.1f}MB  파일 {len(payload)}개")
print(f"  030 대비 변경: {changed}")
print(f"  나머지 {len(ca)-len(changed)}개 멤버 CRC 동일")
print(f"  구간 {CUTS} -> 가중치 {BW}")
