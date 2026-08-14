const fs = require("fs");
const d = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
        AlignmentType, HeadingLevel, ImageRun, ShadingType, BorderStyle, PageBreak } = d;

const KO = "Malgun Gothic";
const W = 9500;                               // A4 본문 폭(DXA)

const t = (text, o = {}) => new TextRun({ text, font: KO, size: o.size || 19,
  bold: o.bold, color: o.color, italics: o.italics });
const p = (runs, o = {}) => new Paragraph({ children: Array.isArray(runs) ? runs : [runs],
  spacing: { after: o.after === undefined ? 62 : o.after, before: o.before || 0, line: 232 },
  alignment: o.align, heading: o.heading, border: o.border });

const H1 = (s) => new Paragraph({ children: [t(s, { size: 24, bold: true, color: "1F3864" })],
  spacing: { before: 105, after: 55 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3864" } } });
const body = (s, o = {}) => p(t(s, o), o);

// 표 만들기 (헤더 음영 + 강조행)
function table(rows, widths, opt = {}) {
  const bold = opt.boldRows || [];
  return new Table({
    columnWidths: widths,
    width: { size: W, type: WidthType.DXA },
    rows: rows.map((cells, ri) =>
      new TableRow({
        children: cells.map((c, ci) =>
          new TableCell({
            width: { size: widths[ci], type: WidthType.DXA },
            shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "DEE6F1" }
                   : bold.includes(ri) ? { type: ShadingType.CLEAR, fill: "F2F7EC" } : undefined,
            margins: { top: 34, bottom: 34, left: 80, right: 80 },
            children: [new Paragraph({
              children: [t(String(c), { size: 17, bold: ri === 0 || bold.includes(ri) })],
              spacing: { after: 0, line: 218 },
              alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })],
          })),
      })),
  });
}

const img = fs.readFileSync("out/v3/FINAL_SUMMARY.png");

const doc = new Document({
  styles: { default: { document: { run: { font: KO, size: 19 } } } },
  sections: [{
    properties: { page: { margin: { top: 600, right: 660, bottom: 520, left: 660 } } },
    children: [
      // ───────────── 표지/제목 ─────────────
      new Paragraph({ children: [t("MLB 투구 전 CSW 예측 모델 — 결과 요약", { size: 30, bold: true, color: "1F3864" })],
        spacing: { after: 60 }, alignment: AlignmentType.CENTER }),
      new Paragraph({ children: [t("Statcast 2017–2019 · 투구 단위 · train 2017–18 / test 2019 · 2026-07", { size: 17, color: "666666" })],
        spacing: { after: 180 }, alignment: AlignmentType.CENTER }),

      // 요약 박스
      new Table({ columnWidths: [W], width: { size: W, type: WidthType.DXA },
        rows: [new TableRow({ children: [new TableCell({
          width: { size: W, type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: "F2F7EC" },
          margins: { top: 110, bottom: 110, left: 140, right: 140 },
          children: [
            p([t("핵심 결론  ", { bold: true, size: 20 }),
               t("최종 모델은 LogLoss 0.5721 / ROC-AUC 0.6378로 기준선(count-only 0.5790)을 명확히 이기며 확률 보정이 거의 완벽하다. "),
               t("그러나 개별 투구를 ‘CSW다/아니다’로 판정하는 용도로는 무의미하다", { bold: true }),
               t("(정확도 0.714 vs ‘항상 아니다’ 0.717). ")], { after: 60 }),
            p([t("원인 진단  ", { bold: true, size: 20 }),
               t("CSW 신호의 대부분은 "),
               t("‘그 공이 어디로 갔는가’", { bold: true }),
               t("에 있고 이는 정의상 투구 전에 알 수 없다. 현재 투구 위치를 넣으면 AUC가 0.613→0.739로 뛰지만, "),
               t("투구 전 피처 311개 전부로는 0.613→0.638에 그친다. 즉 남은 격차는 모델이 아니라 "),
               t("정보의 부재", { bold: true }), t("다.")], { after: 0 }),
          ] })] })] }),
      body("", { after: 70 }),

      // ───────────── 1. 과제와 데이터 ─────────────
      H1("1. 과제와 데이터"),
      body("CSW(Called Strike + Whiff)는 소표본에서 ERA·WHIP보다 안정적인 투수 지배력 지표다. 본 과제는 투구 전에 알 수 있는 정보만으로 그 투구의 CSW 확률을 예측한다."),
      table([
        ["항목", "내용"],
        ["원천 데이터", "MLB Statcast 2017–2019 정규시즌 (pybaseball), 2,201,095투구 × 122열"],
        ["라벨", "is_csw = called_strike + swinging_strike(+blocked), 기저 비율 28.3%(2019)"],
        ["분할", "train 2017–2018 (238,054) / test 2019 (94,150) — 미래 시즌 외삽"],
        ["분석 표본", "상위 40투수 (계산 자원 제약)"],
      ], [1900, 7600]),
      body("", { after: 50 }),

      // ───────────── 2. 방법 ─────────────
      H1("2. 방법"),
      table([
        ["구분", "설계"],
        ["예측 시점", "엄격한 투구 전 — 예측 대상 투구의 물리값·위치·릴리스각·구종·결과를 전부 입력에서 제외"],
        ["누수 차단·평가", "모든 이력·인코딩을 시간 정렬 후 shift(1)로 계산(금지 열은 assert 검증). prequential — 2019 진행 중 이력 피처만 갱신하고 파라미터는 2017–18에 고정"],
        ["피처 311개", "투수 이력 6지표×7창 · 구종별 아스널 · 릴리스 반복성 · 타자 73개 · 워크로드 12개 · 시퀀싱·직전타석·타순"],
        ["모델", "로지스틱 / RF / ExtraTrees / HistGB / LightGBM / XGBoost — 계열별 Optuna 독립 튜닝"],
        ["구조 개선", "CS/W 분해:  P(CSW) = P(swing)·P(whiff|swing) + P(take)·P(called|take)"],
      ], [1500, 8000]),
      body("", { after: 50 }),

      // ───────────── 3. 결과 ─────────────
      H1("3. 결과 (TEST 2019, 94,150투구)"),
      table([
        ["모델", "LogLoss ↓", "ROC-AUC ↑", "F1", "정확도"],
        ["기준: 리그평균 상수", "0.5957", "0.500", "0.441", "0.717"],
        ["기준: count-only (카운트+좌우)", "0.5790", "0.614", "0.458", "0.717"],
        ["E  +PitchPredict 피처 +Optuna 40회", "0.5734", "0.633", "0.468", "0.708"],
        ["I  계열 최고 (LightGBM, 311피처)", "0.5730", "0.635", "0.469", "0.708"],
        ["최종  LightGBM + CS/W 분해 앙상블", "0.5721", "0.6378", "0.470", "0.704"],
        ["(대조) 현재 투구 위치 포함 — 투구 후", "0.5355", "0.7205", "—", "0.736"],
      ], [3700, 1450, 1450, 1450, 1450], { boldRows: [5] }),
      body("주: ‘항상 아니다’ 전략의 정확도가 0.7171이므로 정확도 기준으로는 어떤 모델도 기준선을 넘지 못한다. 마지막 행은 예측 시점 규칙을 의도적으로 위반한 대조군(달성 가능한 상한).",
        { size: 15, color: "666666", after: 40 }),

      // ───────────── 4. 그림 ─────────────
      H1("4. 종합 결과"),
      new Paragraph({ children: [new ImageRun({ data: img, type: "png",
        transformation: { width: 468, height: 268 } })], alignment: AlignmentType.CENTER,
        spacing: { after: 60 } }),
      body("그림. ①라운드별 성능 ②천장 진단 ③피처군 기여도 ④확률 보정 ⑤이진 판정 한계 ⑥비율 예측 비교",
        { size: 16, color: "666666", align: AlignmentType.CENTER }),
      body("", { after: 50 }),

      // ───────────── 5. 핵심 발견 ─────────────
      H1("5. 핵심 발견"),
      p([t("① 과제의 천장이 낮다 — 그리고 그 이유를 정량화했다. ", { bold: true }),
            t("현재 투구의 위치 4개 컬럼만 추가하면 AUC가 0.613→0.739(+0.126)로 오르는데, 투구 전 피처 311개 전부로는 +0.025에 불과하다. CSW 신호의 약 9할이 ‘실제 투구 실행’에 있고, 이는 예측 시점에 원리적으로 접근할 수 없다.")]),
      p([t("② 이진 판정기로는 무가치, 확률 추정기로는 유효하다. ", { bold: true }),
            t("정확도 0.714는 ‘항상 아니다’(0.717)와 동급, F1 0.470은 ‘전부 CSW’(0.441)를 겨우 넘는다. 반면 예측 확률의 십분위별 실제 CSW율은 15%→15%, 46%→45%로 일치하며 최저·최고를 3배로 구분한다. 활용처는 판정이 아니라 기대 CSW율 산출이다.")]),
      p([t("③ 성능 개선은 피처가 아니라 구조에서 나왔다. ", { bold: true }),
            t("피처 137개를 추가해도 LogLoss는 정체됐으나(AUC만 상승), CSW를 콜드스트라이크와 헛스윙으로 분해하니 일관되게 개선됐다(단일 0.5734 → 분해 0.5727 → 앙상블 0.5721). 계열 교체 효과는 미미(최고↔최하 0.0076)했고 동일 계열 앙상블은 단일 최고보다 나빴다.")]),
      body("", { after: 40 }),

      // ───────────── 6. 부가 발견 ─────────────
      H1("6. 부가 발견"),
      p([t("피처 격리 측정  ", { bold: true }),
         t("타자 정보 73개의 순기여는 −0.0022로 모델 중요도의 31%를 차지한다. 누적 ablation에서는 다른 피처와 상쇄돼 보이지 않아 격리 측정이 필수였다. 워크로드 12개는 −0.0008로 작지만, 평소 대비 1.25배 초과 투구 시 CSW가 0.211, 1.5배 초과 시 0.148까지 떨어지는 뚜렷한 피로 신호를 확인했다(평균 0.283, 해당 구간 표본 0.2%).")], { after: 60 }),
      p([t("설계 교훈  ", { bold: true }),
         t("투수별 개별 모델은 LightGBM으로는 전체 모델보다 열등(0.605 vs 0.577)했으나 TabPFN은 0.576으로 양호해, 소표본에는 부분 풀링이나 사전학습 모델이 적합함을 확인했다. 또한 경기그룹 번갈아 K-fold는 시간을 뒤섞어 낙관 편향을 만들어(XGBoost 검증 2위→테스트 4위) 모델 선택을 왜곡하므로, 시간순 분할이 안전하다.")], { after: 40 }),

      // ───────────── 7. 결론 ─────────────
      H1("7. 결론 및 다음 단계"),
      body("투구 전 정보만 쓰는 개별 투구 CSW 예측은 사실상 한계에 도달했다. 기준선 대비 개선과 확률 보정은 실재하나 절대 개선폭(−0.007)은 실용 가치가 제한적이다. 모델링 실패가 아니라 과제 정의의 구조적 한계이므로 아래 전환을 권고한다.", { after: 60 }),
      table([
        ["방향", "내용", "근거"],
        ["① xCSW — 투구 품질 평가", "현재 투구의 위치·구위를 의도적으로 입력해 ‘기대 CSW’ 산출. 예측이 아니라 운·수비·심판 편차를 제거한 투수 평가 지표(Stuff+ 계열)", "동일 설정 AUC 0.756"],
        ["② 경기 단위 CSW% 직접 예측", "투구 확률 평균이 아니라 투수-경기 비율을 직접 모델링(상대 타선·구장·휴식 포함)", "현재 R² 0.075 vs 이동평균 0.244"],
      ], [2300, 5200, 2000]),
      body("단기 후보: 분해 3모델 개별 튜닝(현재 파라미터 공유), 확률 보정(isotonic), 전체 투수로 표본 확대, 부분 풀링.", { after: 50 }),
      body("산출물: 노트북 A~I (라운드별 실험·그래프), AGENT_HANDOFF.md (인수인계), out/ (지표 JSON·CSV·그림)",
        { size: 16, color: "666666" }),
    ],
  }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync("CSW_결과요약_2page.docx", b); console.log("wrote CSW_결과요약_2page.docx", b.length, "bytes"); });
