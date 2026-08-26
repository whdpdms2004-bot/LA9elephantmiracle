# -*- coding: utf-8 -*-
"""v13 패키징 — 168피처 · CatBoost + FT-Transformer + MLP.

    p = r + w_cb(p_cb − r) + w_ft(p_ft − r) + w_mlp(p_mlp − r)

v12 와 달리 sklearn 트리 2계열(A·B, 4500트리)을 뺐다.
168피처 CatBoost 하나(828)가 v12 블렌드 전체(803)보다 낫기 때문이고,
덕분에 추론이 크게 가벼워졌다.

리더보드 유래 상수 없음
    target_rate   시즌별 성공률 선형 외삽 (방법은 롤링 백테스트로 선정)
    블렌드 가중치  val 2024·2022 에서 w*=M^-1A, 두 해 평균
    계열별 C0     2024행의 season→2025 분포에서 mean(p)==target 해
    logit_scale   val 2024 에서 최적화

실행:
    python make_v13.py
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
V7 = os.path.join(HERE, "submit_v7_blend.zip")
STAGE = os.path.join(HERE, "_v13")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submit_v13.zip")
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()

    need = ["v13_cb.npz", "v13_ft.pt", "v13_mlp.pt", "v13_prep.npz",
            "params_v13.json", "season_lut.npz", "domain_lut.npz", "encodings.npz"]
    for f in need:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            sys.exit("없음: model/%s" % f)
    pv = json.load(open(os.path.join(MODEL_DIR, "params_v13.json"), encoding="utf-8"))

    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(os.path.join(STAGE, "model"))
    with zipfile.ZipFile(V7) as z:      # feature_names / floor / ceil / requirements 재사용
        z.extractall(STAGE)
    old = json.load(open(os.path.join(STAGE, "model", "params.json"), encoding="utf-8"))
    for f in ("trees_a.npz", "trees_b.npz"):
        p = os.path.join(STAGE, "model", f)
        if os.path.exists(p):
            os.remove(p)

    params = {
        "feature_names": old["feature_names"],
        "n_features": pv["n_features"],
        "target_rate": pv["target_rate"],
        # 출처 문자열은 params_v13.json 이 갖고 있는 것을 따른다.
        # apply_recalib.py 로 리더보드 실측값을 넣으면 그에 맞게 바뀌어야 한다.
        "target_rate_source": pv.get(
            "target_rate_source",
            "train-only: 시즌별 성공률의 선형 외삽 (롤링 백테스트로 방법 선정)"),
        "logit_target_C1": pv["logit_target_C1"],
        "model_cb": pv["model_cb"], "model_ft": pv["model_ft"], "model_mlp": pv["model_mlp"],
        "blend_w_cb": pv["blend_w_cb"], "blend_w_ft": pv["blend_w_ft"],
        "blend_w_mlp": pv["blend_w_mlp"],
        "blend_source": pv["blend_source"],
        "floor": old["floor"], "ceil": old["ceil"],
        "seeds": pv["seeds"],
        "note": ("168피처 = 원본 72 + 시즌폼 8 + TrackMan 구종별 릴리스 일관성 55 "
                 "+ 볼카운트 12종 27 + 역할 5 + 결측표시 1. "
                 "모든 룩업은 학습 데이터로만 만들고 추론 때는 그 행의 pitcher_id 로 조회만 한다. "
                 "모델은 CatBoost(oblivious, numpy 내보내기) + FT-Transformer + MLP, 각 3시드."),
    }
    for src, dst in (("v13_cb.npz", "cb.npz"), ("v13_ft.pt", "ft.pt"),
                     ("v13_mlp.pt", "mlp.pt"), ("v13_prep.npz", "prep.npz"),
                     ("season_lut.npz", "season_lut.npz"),
                     ("domain_lut.npz", "domain_lut.npz")):
        shutil.copy(os.path.join(MODEL_DIR, src), os.path.join(STAGE, "model", dst))
    json.dump(params, open(os.path.join(STAGE, "model", "params.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    shutil.copy(os.path.join(HERE, "script_v13.py"), os.path.join(STAGE, "script.py"))
    # torch 는 평가서버 기본 설치. requirements 에 명시하면 오히려 설치 오류 위험.
    open(os.path.join(STAGE, "requirements.txt"), "w", encoding="utf-8").write(
        "# 평가 서버 기본 설치 패키지만 사용합니다: numpy 1.26.4 / pandas 2.0.3 / torch 2.7.1\n"
        "# 버전을 덮어쓰면 설치 오류 위험이 있어 아무것도 명시하지 않습니다.\n")

    if not a.skip_test:
        src = os.path.join(HERE, "..", "open", "data")
        os.makedirs(os.path.join(STAGE, "data"), exist_ok=True)
        for f in ("test.csv", "sample_submission.csv"):
            shutil.copy(os.path.join(src, f), os.path.join(STAGE, "data", f))
        print("── 스모크 테스트 ──", flush=True)
        t = time.time()
        if subprocess.run([sys.executable, "script.py"], cwd=STAGE).returncode != 0:
            sys.exit("스모크 테스트 실패 — 제출하지 마세요.")
        with open(os.path.join(STAGE, "output", "submission.csv"), encoding="utf-8") as fh:
            vals = [float(x["control_success"]) for x in csv.DictReader(fh)]
        print("\n출력 %d행: %s  (%.0f초)"
              % (len(vals), ["%.5f" % v for v in vals], time.time() - t))
        assert all(0.0 <= x <= 1.0 for x in vals), "확률 범위 벗어남"
        assert len(set("%.6f" % x for x in vals)) > 1, "모든 예측이 동일"
        print("통과: 범위 정상, 행별로 값이 다름")
        shutil.rmtree(os.path.join(STAGE, "data"))
        shutil.rmtree(os.path.join(STAGE, "output"), ignore_errors=True)

    out = os.path.join(HERE, a.out)
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in ("model/cb.npz", "model/ft.pt", "model/mlp.pt", "model/prep.npz",
                  "model/encodings.npz", "model/season_lut.npz", "model/domain_lut.npz",
                  "model/params.json", "script.py", "requirements.txt"):
            z.write(os.path.join(STAGE, n), n)

    print("\n" + "=" * 66)
    print("생성: %s (%.1f MB)" % (a.out, os.path.getsize(out) / 1e6))
    with zipfile.ZipFile(out) as z:
        for i in z.infolist():
            print("  %9d  %s" % (i.file_size, i.filename))
    print("\n결합: p = %.5f + %.3f(cb−r) + %.3f(ft−r) + %.3f(mlp−r)"
          % (params["target_rate"], params["blend_w_cb"], params["blend_w_ft"],
             params["blend_w_mlp"]))
    print("\n다음:  python timeit_v13.py   →   python check_rules.py %s   →   python verify_v13.py"
          % a.out)
    print("=" * 66)


if __name__ == "__main__":
    main()
