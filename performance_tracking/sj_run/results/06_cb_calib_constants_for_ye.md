# CB 배포 보정 상수 (ye 요청) + 확인하면서 생긴 질문 하나

작성 hw · 2026-08-29

---

## 1. ye 님 질문에 대한 답 -- zip 없이도 됩니다

> "CB 의 실제 배포용 calibration 상수(logit_scale/center 등, depth6·id_freq
> 176피처 버전 기준)가 필요한데, 이 상수는 sj_stdmlp.zip 안에만 있어서
> 못 구했습니다."

**`model_cb` 는 재조립 과정에서 바뀌지 않아서, 이미 저장소에 있는
`cw_v17_base.zip` 값을 그대로 쓰시면 됩니다.**

```python
model_cb = {
    "logit_scale":     1.0499999999999998,
    "logit_center_C0": -0.048777143615610016,
    "cap":             0.2,
    "target_rate":     0.47469465355297163,
    "logit_target_C1": -0.10130794309825776,
}
```

`model_ft` 도 같습니다. `model_mlp` 만 sj 님이 새로 뽑으셨습니다.

```python
model_ft = {"logit_scale": 0.8999999999999999,
            "logit_center_C0": -0.056385597214102745,
            "cap": 0.2, "target_rate": 0.47469465355297163,
            "logit_target_C1": -0.10130794309825776}

# model_mlp 는 sj_stdmlp 에서 갱신됨 (sj_stdmlp.md:168-172)
#   logit_scale 0.80 · logit_center_C0 -0.061225 · target_rate 0.474695
```

### 보정 식 (배포 script.py 와 동일)

```python
def apply_calibration(p, prm):
    eps = 1e-6
    p  = np.clip(p, eps, 1 - eps)
    lg = np.log(p / (1 - p))
    z  = prm["logit_scale"] * (lg - prm["logit_center_C0"]) + prm["logit_target_C1"]
    q  = 1.0 / (1.0 + np.exp(-z))
    lo = prm["target_rate"] - prm["cap"]
    hi = prm["target_rate"] + prm["cap"]
    return np.clip(q, max(eps, lo), min(1 - eps, hi))
```

### 그 위 두 층의 가중 (같이 필요하실 것 같아 적습니다)

```
cw 모듈 내부  p_cw = r + 0.7432*(p_cb - r) + 0.1598*(p_ft - r) + 0.1646*(p_mlp - r)
팀 결합       p    = r + 0.45461*(p_cw - r) + 0.64333*(p_sj3way - r) - 0.003223
```

내부 가중은 `cw_v17_base.zip` 값(0.5993/0.2902/0.1211)이
아니라 위 값이 배포본입니다 (DECK 부록 B). 팀 가중은
`build_submit_zip.py:10` · DECK D4 기준입니다 -- 제가 어제 0.6/0.4 로 잘못
읽은 적이 있어서 이 값으로 봐주세요.

---

## 2. 왜 안 바뀌는지 -- 확인한 근거

어제 제가 챔피언 가중을 잘못 읽은 적이 있어서, 이번엔 재조립 경로를 단계마다
따라가 봤습니다.

**(1) 챔피언 zip 의 cw params 가 `cw_v17_base.zip` 과 같습니다.** 로컬에 푼
`submit_cw_sj_final` 의 `model/cw/model/params.json` 과 키별로 비교했는데
`target_rate` · `logit_target_C1` · `model_cb` · `model_ft` · `model_mlp` ·
`blend_w_*` · `n_features` 가 전부 동일합니다.

**(2) `build_submit_zip.py` 가 params.json 을 교체 대상에 넣지 않습니다.**
`REPLACE`(42~48행)가 `cb.npz` · `idfreq_lut.npz` · `mlp.pt` · `stdprep.npz` ·
`script.py` 다섯이고, 나머지는 89행에서 원본 그대로 복사됩니다.

**(3) `set_cw_params.py`(52~72행)는 `blend_w_*` 와 `model_mlp` 만 씁니다.**

`build_final_cb.py` 도 보정 상수를 내보내지 않습니다(`idfreq_meta.json` 은
피처 이름만 담습니다).

> `n_features` 가 168 로 남아 있는 것도 확인했습니다. 패치된 스크립트가 168열
> `X` 를 만든 뒤 cb 에만 id_freq 8열을 붙여 `Xcb`(176)를 만들기 때문에
> (`patch_cw_script.py:89`) assert 가 그대로 통과합니다. 설계대로입니다.

---

## 3. 확인하다 생긴 질문 -- sj 님께

확실하지 않아서 질문으로 남깁니다. **틀렸으면 알려주세요.**

`logit_center_C0` 는 그 모델의 예측 분포를 보고 뽑는 값으로 이해했습니다.
그런데 위 (1)~(3) 대로면 `model_cb` 의 C0 는 168피처·depth 5 시점 값이고,
그 뒤 cb 가 176피처 · depth 6 · 시드분산 상보가중으로 바뀌는 동안 그대로
유지됩니다.

`sj_stdmlp.md:168` 에 mlp 는 "원본 `train_v13.py` 와 같은 절차로 다시
뽑았다" 고 적혀 있는데, cb 도 같이 다시 뽑으셨는지가 문서에서 안 보여서
여쭙습니다. **의도적으로 유지하신 것이면 이 절은 무시해 주세요** -- 제가
파이프라인을 잘못 읽었을 수 있습니다.

만약 아직이라면, DECK 부록 A 의 최종 평균 어긋남(`0.476804 vs 목표
0.474695`, +0.0021, 최대 손실 ~1.8점) 중 일부가 여기서 올 가능성이
있을까 해서요. mlp 때 쓰신 절차(2024행의 season 을 2025 로 바꿔 예측 ->
보정 후 평균이 target_rate 가 되도록 이분법)를 cb 에 그대로 돌리면
나올 것 같은데, `sj_stdmlp.zip`(375MB)이 제 로컬에 없어서 확인을 못
했습니다.

**기대값은 작습니다** -- 어긋남 전체가 ~1.8점이고 그중 일부입니다. 다만
재학습 없이 상수 하나만 다시 푸는 것이라 비용이 거의 없어 보여서 적어둡니다.
지금 GPU 큐가 밀려 있으니 우선순위는 낮게 봐주세요.
