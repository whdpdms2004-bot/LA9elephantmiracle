"""전처리 원자 변환 레지스트리.

새 전처리를 추가하려면 이 폴더에 파일 하나를 만들면 된다. 자동으로 발견된다.
규약은 example_template.py 를 보라. 기존 코드는 건드릴 필요가 없다.

내장 변환 15개는 cowork/sj/feature_campaign_1000/v85_preprocess_screen.py 의
함수들을 그대로 감싼 것이다. 복사하지 않고 import 하므로 결과가 계속 비교 가능하다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
SJ = LAB.parent
CAMPAIGN = SJ / "feature_campaign_1000"
MODEL_OPT = SJ / "experiment" / "model_optimization"
for p in (MODEL_OPT, CAMPAIGN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# name -> {"apply": fn, "targets": [...], "note": str, "conflicts": [...], "source": str}
REGISTRY: dict[str, dict] = {}


def register(name, apply, targets=(), note="", conflicts=(), source="lab"):
    if name in REGISTRY:
        raise ValueError(f"변환 이름 중복: {name}")
    REGISTRY[name] = {"apply": apply, "targets": list(targets), "note": note,
                      "conflicts": list(conflicts), "source": source}


def _load_builtin():
    """v85_preprocess_screen 의 원자 변환 15개를 감싼다."""
    try:
        import v85_preprocess_screen as M
    except Exception as exc:                                   # noqa: BLE001
        print(f"[transforms] 내장 변환을 불러오지 못했다: {exc}")
        return

    def extras_only(fn, needs_mask=False, needs_fold=False):
        def _apply(frame, features, categorical, train_mask, fold):
            if needs_mask:
                out = fn(frame, train_mask)
            elif needs_fold:
                out = fn(frame, fold)
            else:
                out = fn(frame)
            if isinstance(out, tuple):          # (values, categorical)
                return out[0], features, categorical + list(out[1])
            return out, features, categorical
        return _apply

    def drop_prefix(prefixes):
        def _apply(frame, features, categorical, train_mask, fold):
            keep = [c for c in features if not c.startswith(prefixes)]
            return {}, keep, [c for c in categorical if c in keep]
        return _apply

    B = [
        ("rate_multiscale", extras_only(M.rate_multiscale, needs_mask=True),
         ["asof_*_rate"], "성공률을 여러 강도로 EB 평활 + reliability"),
        ("rate_geometry", extras_only(M.rate_geometry),
         ["asof_*_rate", "pitchmix"], "logit 변환 + 투타 격차 + 구질 엔트로피/로그비"),
        ("count_multiscale", extras_only(M.count_multiscale),
         ["asof_*_n"], "sqrt + 버킷 범주화 + reliability"),
        ("recent_shape", extras_only(M.recent_shape),
         ["asof_pitcher_prev*"], "최근 등판의 가중합/기울기/곡률/shock"),
        ("temporal_cyclic", extras_only(M.temporal_cyclic, needs_fold=True),
         ["game_month", "game_dayofweek", "season"],
         "월·요일 sin/cos + 예측 시즌까지 남은 연·월"),
        ("context_robust", extras_only(M.context_robust),
         ["li", "score_diff", "base_state", "outs"],
         "signed-log, li 캡, 주자압력, 남은아웃"),
        ("trackman_quality", extras_only(M.trackman_quality, needs_mask=True),
         ["tm500_*"], "결측수/비율, 스타일 L2, 분산 요약"),
        ("component_shape", extras_only(M.component_shape),
         ["sx_cf_*"], "성분 절대값/부호/상대비"),
        ("id_frequency", None, ["pitcher_id", "batter_id", "*_team_id"],
         "원본 ID 제거 + 빈도 인코딩(log1p, unseen 플래그)"),
        ("drop_ids", None, ["pitcher_id", "batter_id", "*_team_id"], "ID 통째 제거"),
        ("ordinal_numeric", None, ["범주형 전체"], "명목형만 남기고 나머지는 수치 취급"),
        ("trackman_compact", None, ["tm500_*"], "TrackMan 열 축소"),
        ("component_compact", None, ["sx_cf_*"], "성분 열 축소"),
        ("no_trackman", drop_prefix(("tm500_", "cw_")), ["tm500_*", "cw_*"],
         "TrackMan 통째 제거"),
        ("no_component", drop_prefix(("sx_cf_",)), ["sx_cf_*"], "성분 통째 제거"),
    ]

    def id_freq(frame, features, categorical, train_mask, fold):
        keep = [c for c in features if c not in M.ID_COLUMNS]
        return (M.id_frequency(frame, train_mask), keep,
                [c for c in categorical if c in keep])

    def drop_ids(frame, features, categorical, train_mask, fold):
        keep = [c for c in features if c not in M.ID_COLUMNS]
        return {}, keep, [c for c in categorical if c in keep]

    def ordinal(frame, features, categorical, train_mask, fold):
        return {}, features, [c for c in categorical if c in M.NOMINAL_CATEGORICAL]

    def tm_compact(frame, features, categorical, train_mask, fold):
        keep = M.compact_trackman(features)
        return {}, keep, [c for c in categorical if c in keep]

    def cf_compact(frame, features, categorical, train_mask, fold):
        keep = M.compact_component(features)
        return {}, keep, [c for c in categorical if c in keep]

    special = {"id_frequency": id_freq, "drop_ids": drop_ids,
               "ordinal_numeric": ordinal, "trackman_compact": tm_compact,
               "component_compact": cf_compact}
    conflicts = {"drop_ids": ["id_frequency"], "id_frequency": ["drop_ids"],
                 "no_trackman": ["trackman_quality", "trackman_compact"],
                 "trackman_quality": ["no_trackman"], "trackman_compact": ["no_trackman"],
                 "no_component": ["component_shape", "component_compact"],
                 "component_shape": ["no_component"], "component_compact": ["no_component"]}
    for name, fn, targets, note in B:
        register(name, special.get(name, fn), targets, note,
                 conflicts.get(name, ()), source="builtin(v85)")


def _load_contributed():
    """이 폴더의 사용자 기여 변환을 자동 발견한다."""
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(f"_tf_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:                               # noqa: BLE001
            print(f"[transforms] {path.name} 로드 실패: {exc}")
            continue
        if not hasattr(mod, "NAME") or not hasattr(mod, "apply"):
            continue
        if getattr(mod, "DISABLED", False):
            continue
        register(mod.NAME, mod.apply, getattr(mod, "TARGETS", ()),
                 getattr(mod, "NOTE", ""), getattr(mod, "CONFLICTS", ()),
                 source=path.name)


def load_all():
    if not REGISTRY:
        _load_builtin()
        _load_contributed()
    return REGISTRY


def compatible(names) -> bool:
    reg = load_all()
    s = set(names)
    for n in s:
        if n in reg and set(reg[n]["conflicts"]) & s:
            return False
    return True


def build(frame, features, categorical, names, train_mask, fold):
    """변환들을 순서대로 적용한다. 반환: (frame, features, categorical)."""
    reg = load_all()
    import v85_preprocess_screen as M
    extras: dict = {}
    for name in names:
        if name not in reg:
            raise ValueError(f"모르는 변환: {name}. 등록된 것: {sorted(reg)}")
        new_extras, features, categorical = reg[name]["apply"](
            frame, features, categorical, train_mask, fold)
        extras.update(new_extras or {})
    frame = M.add_columns(frame, extras)
    features = list(dict.fromkeys(list(features) + list(extras)))
    categorical = list(dict.fromkeys([c for c in categorical if c in features]))
    return frame, features, categorical
