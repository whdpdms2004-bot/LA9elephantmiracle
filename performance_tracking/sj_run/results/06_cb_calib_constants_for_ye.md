# CB 배포 보정 상수 -- ye 요청 (hw, 2026-08-29)

**질문**: "CB 의 실제 배포용 calibration 상수(logit_scale/center 등,
depth6·id_freq 176피처 버전 기준)가 sj_stdmlp.zip 안에만 있어서 못 구했다."

**답**: **`model_cb` 는 처음부터 한 번도 안 바뀌었다.** `cw_v17_base.zip` 의 값이
그대로 배포본까지 간다. 그 zip 은 이미 저장소에 있다
(`performance_tracking/models/cw_v17_base.zip`, 24MB).

## 근거 -- 코드 경로

`models/sj_stdmlp/set_cw_params.py:52-72` 가 덮어쓰는 것은 **딱 둘**이다.

```
--w cb=..,ft=..,mlp=..   ->  blend_w_cb / blend_w_ft / blend_w_mlp
--mlp-cal <meta.json>    ->  model_mlp
```

`model_cb` 와 `model_ft` 는 손대지 않는다. DECK 부록 B 의 재조립 절차도
`build_final_cb.py -> ... -> set_cw_params.py` 로 끝나고 중간에 cb 보정을
다시 뽑는 단계가 없다.

## 값 -- 그대로 쓰면 된다

```python
model_cb = {
    "logit_scale":     1.0499999999999998,
    "logit_center_C0": -0.048777143615610016,
    "cap":             0.2,
    "target_rate":     0.47469465355297163,
    "logit_target_C1": -0.10130794309825776,
}
```

나머지도 같이 적는다 (`model_ft` 도 안 바뀜, `model_mlp` 는 **바뀜**).

```python
model_ft = {"logit_scale": 0.8999999999999999,
            "logit_center_C0": -0.056385597214102745,
            "cap": 0.2, "target_rate": 0.47469465355297163,
            "logit_target_C1": -0.10130794309825776}

# model_mlp 는 sj_stdmlp 에서 갈아끼웠다 (sj_stdmlp.md:168-172)
#   logit_scale 0.80 · logit_center_C0 -0.061225 · target_rate 0.474695
# cw_v17_base 의 옛 값은 참고용:
model_mlp_old = {"logit_scale": 0.8499999999999999,
                 "logit_center_C0": -0.0460211094468832}
```

## 보정 식 (배포 script.py 와 동일)

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

## cw 모듈 내부 가중 (배포본 = sj_stdmlp 기준)

```
p_cw = r + 0.7432*(p_cb - r) + 0.1598*(p_ft - r) + 0.1646*(p_mlp - r)
```

cw_v17_base 의 값(0.5993 / 0.2902 / 0.1211)이
아니라 **위 값**을 써야 배포본과 같다 (DECK 부록 B).

## 팀 가중 (그 위 층)

```
p = r + 0.45461*(p_cw - r) + 0.64333*(p_sj3way - r) - 0.003223
```

`build_submit_zip.py:10` · DECK D4 "팀 가중은 한 번도 안 바꿨다".
(내가 어제 0.6/0.4 로 잘못 쓴 적이 있으니 이 값으로 봐주세요.)
