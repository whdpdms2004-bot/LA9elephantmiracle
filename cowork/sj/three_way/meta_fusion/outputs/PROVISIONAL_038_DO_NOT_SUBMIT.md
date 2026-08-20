# submit_038.zip — 제출 금지 상태

`cowork/sj/submit/2026-08-19/submit_038.zip`은 패키지 구조·행 독립성·오프라인 실행은
통과했지만, 사건 보정기를 학습한 OOF가 채점 fold 라벨 기반 조기 종료
(`use_best_model=True`) 예측임을 사후 확인했다.

- 규정 위반 코드는 아님
- 그러나 최종 고정 900회 teacher와 확률 분포가 다를 수 있음
- Val2024 +13.83 BSS는 낙관 편향 가능성이 있어 제출 근거로 사용하지 않음

`meta_fusion/src/build_honest_oof.py`로 2023/2024 고정 900회 strict-forward OOF를
재생성하고 개선을 재확인한 뒤 새 번호·새 날짜 패키지를 만든다.
