# plan.md

reviewed against task.jsonl up to (없음 — 최초 작성)
최종 갱신: 2026-08-13

> 이 문서는 **리뷰 시점에만** `task.jsonl`을 훑어서 다시 쓴다. 수시 수정 금지. 규칙은 [`AGENTS.md`](../AGENTS.md) A4 참고.

---

## 지금 상태

- 저장소 스캐폴딩 완료. `cowork/` 아래 개인 폴더 5개(sj, ye, cw, yn, hw)와 공용 파일(`plan.md`, `task.jsonl`) 생성.
- 협업 규칙과 제출 규칙을 [`AGENTS.md`](../AGENTS.md)에 정리.
- 아직 실험 항목 없음. `task.jsonl`은 빈 파일.

### 물려받은 기술적 현황 (출처: 루트 [`README.md`](../README.md))

- Public 최고: `submit_013` — **895.404000081**
- 내부 Val2024 최고: `submit_021` — **836.502924** (Public 미검증, 개선 신뢰구간이 0을 포함)
- 전체 AUC가 약 0.55 수준이라 분류 경계보다 **calibration / shrinkage / residual ensemble**이 더 중요했음
- 가장 강한 신호: 시즌 drift + 투수 과거 성공률 + 투수×타자손×count 반응

## 다음에 할 일

1. **TrackMan 완전 미사용 clean validation 재구축** — Val2024 기준선이 최신 시스템과 분리되지 않은 상태. 최우선.
2. **Nested selection** 적용 — 현재 피처/모델 선택이 검증 세트를 재사용하고 있을 여지.
3. **R residual 강화**
4. **Outside component**를 작은 가중치로 연결
5. 각자 폴더에서 작업 시작 → `task.jsonl`에 append → 다음 리뷰 때 이 문서 재작성

## 담당

| 이니셜 | 담당 영역 |
| --- | --- |
| sj | 미정 |
| ye | 미정 |
| cw | 미정 |
| yn | 미정 |
| hw | 미정 |
