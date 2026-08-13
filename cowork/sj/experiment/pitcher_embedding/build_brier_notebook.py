from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "pitcher_embedding_prototype.ipynb"
DEST = HERE / "pitcher_embedding_brier_submission.ipynb"

source_nb = nbf.read(SOURCE, as_version=4)
cells = source_nb.cells[:10]
for cell in cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

cells[0].source = r"""# 투수 임베딩 v1 — Brier 최적화와 제출 artifact

이 노트북은 v0의 세부 실패 라벨, Main↔Trackman crosswalk, 이전 완료 시즌 Trackman 집계를 그대로 사용하면서 대회 평가·추론 규칙에 맞게 학습과 제출 단계를 보강합니다.

핵심 검증 순서는 다음과 같습니다.

1. `2019~2022 학습 → 2023 모델 선택·확률 보정 → 2024 최종 홀드아웃`
2. `control_success` 직접 예측 모델과 `reverse/middle/far residual` 조건부 모델 비교
3. 최종 성공확률의 Brier Score를 주 손실 및 모델 선택 기준으로 사용
4. 고정된 학습 통계·Trackman lookup·ID 사전만 제출 artifact에 포함
5. 평가 데이터의 다른 행이나 전체 분포를 전혀 사용하지 않는 행 독립 추론 코드와 결합

> **보조 라벨 주의**  
> reverse/middle 복원은 공식 `train.csv`만 이용하고 테스트 데이터는 사용하지 않습니다. 공개 규칙에 명시적 금지는 없지만, 누적률을 이용한 공개되지 않은 보조 정답 복원이 허용되는지는 운영진 확인 전까지 실험 기능으로 간주합니다. 그래서 직접 success head를 항상 함께 비교합니다.
"""

cells.append(nbf.v4.new_markdown_cell(r"""## 5. Brier 중심 시간 검증 설계

2024를 모델 선택에 사용하지 않기 위해 아래처럼 완전히 시간 순방향으로 분리합니다.

- 학습: 2019~2022
- 선택·보정: 2023
- 최종 홀드아웃: 2024

빠른 모드는 각 구간을 표본화합니다. `QUICK_RUN=False`로 바꾸면 전체 행을 사용합니다. 전처리 통계와 투수 ID 사전은 학습 구간에서만 적합합니다."""))

cells.append(nbf.v4.new_code_cell(r"""HISTORY_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before", "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li", "pitcher_hand", "batter_hand",
] + ASOF_COLS

TRACKMAN_COUNT_RAW = ["tm_prior_n", "tm_prev_season_n", "tm_prior_max_season_n"]
TRACKMAN_BASE_FEATURES = [
    c for c in model_df.columns
    if c.startswith("tm_") and c != "tm_available"
] + ["tm_available"]

for source in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
    new_col = f"log1p_{source}"
    model_df[new_col] = np.log1p(model_df[source].clip(lower=0))
    HISTORY_FEATURES.remove(source)
    HISTORY_FEATURES.append(new_col)

TRACKMAN_FEATURES = TRACKMAN_BASE_FEATURES.copy()
for source in TRACKMAN_COUNT_RAW:
    new_col = f"log1p_{source}"
    model_df[new_col] = np.log1p(model_df[source].clip(lower=0))
    TRACKMAN_FEATURES.remove(source)
    TRACKMAN_FEATURES.append(new_col)

def sample_indices(pool, limit, seed):
    pool = np.asarray(pool)
    if limit is None or len(pool) <= limit:
        return np.sort(pool)
    return np.sort(np.random.default_rng(seed).choice(pool, limit, replace=False))

eligible = model_df.component_label_valid
fit_pool = model_df.index[eligible & model_df.season.le(2022)]
select_pool = model_df.index[eligible & model_df.season.eq(2023)]
holdout_pool = model_df.index[eligible & model_df.season.eq(2024)]

fit_idx = sample_indices(fit_pool, 350_000 if QUICK_RUN else None, SEED)
select_idx = sample_indices(select_pool, 100_000 if QUICK_RUN else None, SEED + 1)
holdout_idx = sample_indices(holdout_pool, 100_000 if QUICK_RUN else None, SEED + 2)

def fit_preprocessor(frame, columns):
    x = frame[columns].to_numpy(dtype=np.float64)
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x = np.where(np.isfinite(x), x, median)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return {"median": median.astype("float32"), "mean": mean.astype("float32"), "scale": scale.astype("float32")}

def transform_numeric(frame, columns, prep):
    x = frame[columns].to_numpy(dtype=np.float32)
    x = np.where(np.isfinite(x), x, prep["median"])
    return ((x - prep["mean"]) / prep["scale"]).astype("float32")

hist_prep = fit_preprocessor(model_df.loc[fit_idx], HISTORY_FEATURES)
tm_prep = fit_preprocessor(model_df.loc[fit_idx], TRACKMAN_FEATURES)

known_pitchers = np.sort(model_df.loc[model_df.season.le(2022), "pitcher_id"].unique())
pitcher_to_index = {int(pid): i + 1 for i, pid in enumerate(known_pitchers)}
COHORTS = ["UNSEEN", "ROOKIE_1_25", "ROOKIE_26_100", "RETURNING", "VETERAN"]
cohort_to_index = {name: i for i, name in enumerate(COHORTS)}

TARGET_COLS = ["y_reverse", "y_middle", "y_far_residual", "control_success"]

def encode_metadata(frame, pitcher_map):
    pitcher_idx = frame.pitcher_id.map(pitcher_map).fillna(0).astype("int64").to_numpy()
    cohort_idx = frame.experience_cohort.map(cohort_to_index).astype("int64").to_numpy()
    asof_n = frame.asof_pitcher_n.astype("float32").to_numpy()
    return pitcher_idx, cohort_idx, asof_n

def make_arrays(index, hist_preprocessor, tm_preprocessor, pitcher_map):
    frame = model_df.loc[index]
    h = transform_numeric(frame, HISTORY_FEATURES, hist_preprocessor)
    t = transform_numeric(frame, TRACKMAN_FEATURES, tm_preprocessor)
    p, c, n = encode_metadata(frame, pitcher_map)
    y = frame[TARGET_COLS].to_numpy("float32")
    return h, t, p, c, n, y

fit_arrays = make_arrays(fit_idx, hist_prep, tm_prep, pitcher_to_index)
select_arrays = make_arrays(select_idx, hist_prep, tm_prep, pitcher_to_index)
holdout_arrays = make_arrays(holdout_idx, hist_prep, tm_prep, pitcher_to_index)

def make_loader(arrays, shuffle, batch_size=BATCH_SIZE):
    tensors = [torch.from_numpy(x) for x in arrays]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())

fit_loader = make_loader(fit_arrays, True)
select_loader = make_loader(select_arrays, False)
holdout_loader = make_loader(holdout_arrays, False)

print(f"fit={len(fit_idx):,}, select={len(select_idx):,}, holdout={len(holdout_idx):,}")
print(f"history={len(HISTORY_FEATURES)}, trackman={len(TRACKMAN_FEATURES)}, pitcher tokens={len(pitcher_to_index)}")
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 6. 비교 모델

두 모델은 같은 backbone과 32차원 임베딩을 사용합니다.

- `direct`: `control_success`를 직접 예측하고 Brier Loss로 학습
- `component`: reverse/middle/far 조건부 확률을 학습하고 곱으로 성공확률 계산

두 모델 모두 최종 성공확률 Brier를 주 손실로 사용합니다. component 모델의 조건부 BCE는 표현 학습을 돕는 보조 손실입니다. LayerNorm을 사용해 추론 배치의 다른 행에 영향을 받지 않도록 했습니다."""))

cells.append(nbf.v4.new_code_cell(r"""class PitcherBrierNet(nn.Module):
    def __init__(self, hist_dim, tm_dim, n_pitchers, n_cohorts, mode, embedding_dim=32):
        super().__init__()
        if mode not in {"direct", "component"}:
            raise ValueError(mode)
        self.mode = mode
        self.pitcher_embedding = nn.Embedding(n_pitchers + 1, 16, padding_idx=0)
        self.cohort_embedding = nn.Embedding(n_cohorts, 8)
        self.history_tower = nn.Sequential(
            nn.Linear(hist_dim, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(96, 48), nn.SiLU(),
        )
        self.trackman_tower = nn.Sequential(
            nn.Linear(tm_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(64, 24), nn.SiLU(),
        )
        self.individual_projection = nn.Linear(16, 16)
        self.cohort_projection = nn.Linear(8, 16)
        self.fusion = nn.Sequential(
            nn.Linear(48 + 24 + 16 + 8, 64), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(64, embedding_dim), nn.LayerNorm(embedding_dim), nn.SiLU(),
        )
        self.direct_head = nn.Linear(embedding_dim, 1)
        self.reverse_head = nn.Linear(embedding_dim, 1)
        self.middle_head = nn.Linear(embedding_dim, 1)
        self.far_head = nn.Linear(embedding_dim, 1)

    def forward(self, history_x, trackman_x, pitcher_idx, cohort_idx, asof_n):
        history_h = self.history_tower(history_x)
        trackman_h = self.trackman_tower(trackman_x)
        individual = self.individual_projection(self.pitcher_embedding(pitcher_idx))
        cohort_raw = self.cohort_embedding(cohort_idx)
        cohort = self.cohort_projection(cohort_raw)
        alpha = (asof_n / (asof_n + 100.0)).clamp(0, 1).unsqueeze(1)
        alpha = alpha * pitcher_idx.ne(0).float().unsqueeze(1)
        pitcher_h = alpha * individual + (1 - alpha) * cohort
        embedding = self.fusion(torch.cat([history_h, trackman_h, pitcher_h, cohort_raw], dim=1))
        direct_logit = self.direct_head(embedding).squeeze(1)
        component_logits = torch.cat([
            self.reverse_head(embedding), self.middle_head(embedding), self.far_head(embedding)
        ], dim=1)
        return direct_logit, component_logits, embedding

    def success_probability(self, direct_logit, component_logits):
        if self.mode == "direct":
            return torch.sigmoid(direct_logit)
        component_p = torch.sigmoid(component_logits)
        return torch.prod(1 - component_p, dim=1)


binary_bce = nn.BCEWithLogitsLoss(reduction="none")

def model_loss(model, direct_logit, component_logits, y):
    success_p = model.success_probability(direct_logit, component_logits)
    brier = torch.mean((success_p - y[:, 3]) ** 2)
    if model.mode == "direct":
        success_bce = binary_bce(direct_logit, y[:, 3]).mean()
        return brier + 0.05 * success_bce, {"brier": brier.detach(), "aux": success_bce.detach()}

    reverse_loss = binary_bce(component_logits[:, 0], y[:, 0]).mean()
    no_reverse = y[:, 0].eq(0)
    middle_loss = binary_bce(component_logits[no_reverse, 1], y[no_reverse, 1]).mean()
    neither = no_reverse & y[:, 1].eq(0)
    far_loss = binary_bce(component_logits[neither, 2], y[neither, 2]).mean()
    auxiliary = reverse_loss + middle_loss + far_loss
    return brier + 0.10 * auxiliary, {"brier": brier.detach(), "aux": auxiliary.detach()}

@torch.no_grad()
def predict_model(model, loader, return_embeddings=False):
    model.eval()
    probabilities, targets, embeddings = [], [], []
    for history_x, trackman_x, pitcher_idx, cohort_idx, asof_n, y in loader:
        direct_logit, component_logits, embedding = model(
            history_x.to(DEVICE), trackman_x.to(DEVICE), pitcher_idx.to(DEVICE),
            cohort_idx.to(DEVICE), asof_n.to(DEVICE),
        )
        probabilities.append(model.success_probability(direct_logit, component_logits).cpu().numpy())
        targets.append(y[:, 3].numpy())
        if return_embeddings:
            embeddings.append(embedding.cpu().numpy())
    result = (np.concatenate(probabilities), np.concatenate(targets))
    if return_embeddings:
        result += (np.concatenate(embeddings),)
    return result

def brier_report(y, p):
    p = np.clip(np.asarray(p), 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    brier = float(np.mean((p - y) ** 2))
    reference = float(y.mean() * (1 - y.mean()))
    skill = max(0.0, 100000.0 * (1 - brier / reference)) if reference > 0 else 0.0
    return {"brier": brier, "bss": skill, "logloss": log_loss(y, p, labels=[0, 1]),
            "auc": roc_auc_score(y, p), "pred_mean": float(p.mean()), "target_mean": float(y.mean())}

def train_candidate(mode, epochs=EPOCHS):
    torch.manual_seed(SEED + (0 if mode == "direct" else 100))
    model = PitcherBrierNet(len(HISTORY_FEATURES), len(TRACKMAN_FEATURES), len(pitcher_to_index),
                            len(COHORTS), mode=mode).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_brier = None, float("inf")
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_brier = seen = 0.0
        for history_x, trackman_x, pitcher_idx, cohort_idx, asof_n, y in fit_loader:
            history_x, trackman_x = history_x.to(DEVICE), trackman_x.to(DEVICE)
            pitcher_idx, cohort_idx, asof_n, y = pitcher_idx.to(DEVICE), cohort_idx.to(DEVICE), asof_n.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            direct_logit, component_logits, _ = model(history_x, trackman_x, pitcher_idx, cohort_idx, asof_n)
            loss, parts = model_loss(model, direct_logit, component_logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = len(y)
            total_loss += loss.item() * n
            total_brier += parts["brier"].item() * n
            seen += n
        select_p, select_y = predict_model(model, select_loader)
        report = brier_report(select_y, select_p)
        rows.append({"mode": mode, "epoch": epoch, "train_loss": total_loss / seen,
                     "train_brier": total_brier / seen, "select_brier": report["brier"], "select_bss": report["bss"]})
        if report["brier"] < best_brier:
            best_brier = report["brier"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{mode:9s} epoch {epoch:02d} | train_brier={total_brier/seen:.6f} | "
              f"2023_brier={report['brier']:.6f} | BSS={report['bss']:.2f}")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


candidates = {}
training_curves = []
for candidate_mode in ["direct", "component"]:
    candidate_model, curve = train_candidate(candidate_mode)
    candidates[candidate_mode] = candidate_model
    training_curves.append(curve)

display(pd.concat(training_curves, ignore_index=True))
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 7. 2023 고정 확률 보정과 2024 홀드아웃

각 모델의 2023 예측에 2개 파라미터 affine-logit 보정을 적합합니다. 보정은 학습 데이터에서 고정된 함수이므로 테스트 전체 분포를 사용하지 않으며, 테스트의 각 행에 독립적으로 적용됩니다. 모델 선택은 2023 보정 BSS로만 하고 2024는 최종 보고에만 사용합니다."""))

cells.append(nbf.v4.new_code_cell(r"""def fit_brier_calibrator(probability, target):
    probability = np.clip(np.asarray(probability), 1e-5, 1 - 1e-5)
    logit = np.log(probability / (1 - probability))
    x = torch.tensor(logit, dtype=torch.float64)
    y = torch.tensor(target, dtype=torch.float64)
    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], lr=0.2, max_iter=80, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        p = torch.sigmoid(a * x + b)
        loss = torch.mean((p - y) ** 2)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(a.detach()), float(b.detach())

def apply_calibrator(probability, calibrator):
    p = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    a, b = calibrator
    return 1 / (1 + np.exp(-(a * logit + b)))


comparison_rows = []
calibrators = {}
holdout_predictions = {}
for mode, candidate_model in candidates.items():
    select_p, select_y = predict_model(candidate_model, select_loader)
    calibrator = fit_brier_calibrator(select_p, select_y)
    calibrators[mode] = calibrator
    select_cal = apply_calibrator(select_p, calibrator)

    holdout_p, holdout_y = predict_model(candidate_model, holdout_loader)
    holdout_cal = apply_calibrator(holdout_p, calibrator)
    holdout_predictions[mode] = holdout_cal

    for split_name, y, raw, calibrated in [
        ("2023_select", select_y, select_p, select_cal),
        ("2024_holdout", holdout_y, holdout_p, holdout_cal),
    ]:
        for calibration_name, p in [("raw", raw), ("calibrated", calibrated)]:
            comparison_rows.append({"mode": mode, "split": split_name, "probability": calibration_name,
                                    **brier_report(y, p), "cal_a": calibrator[0], "cal_b": calibrator[1]})

comparison = pd.DataFrame(comparison_rows)
selection_table = comparison.query("split == '2023_select' and probability == 'calibrated'")
selected_mode = selection_table.sort_values(["brier", "mode"]).iloc[0]["mode"]

comparison.to_csv(OUTPUT_DIR / "brier_model_comparison.csv", index=False)
display(comparison.sort_values(["split", "brier"]))
print("selected mode from 2023 only:", selected_mode, "calibrator:", calibrators[selected_mode])

plt.figure(figsize=(9, 4.5))
sns.barplot(data=comparison.query("split == '2024_holdout'"), x="mode", y="bss", hue="probability")
plt.axhline(549.51, color="red", linestyle="--", label="completion threshold 549.51")
plt.title("2024 holdout Brier Skill Score")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "brier_model_comparison.png", dpi=160, bbox_inches="tight")
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 8. 선택 모델 전체 재학습과 제출용 artifact

모델 구조·손실·보정 방식 선택이 끝났으므로 2019~2024 전체 학습 데이터로 선택 모델을 다시 학습합니다. 빠른 모드에서는 최대 60만 행을 사용합니다.

제출 artifact에는 다음만 저장합니다.

- PyTorch 가중치
- NumPy 전처리 통계
- 고정 투수 ID 사전 및 cohort 사전
- 공식 train+Trackman으로 미리 만든 2025 cutoff Trackman lookup
- 2023에서 고정한 2-파라미터 확률 보정값

평가 시 test 전체를 집계하거나 crosswalk를 다시 만들지 않습니다."""))

cells.append(nbf.v4.new_code_cell(r"""final_pool = model_df.index[eligible]
final_idx = sample_indices(final_pool, 600_000 if QUICK_RUN else None, SEED + 10)
final_hist_prep = fit_preprocessor(model_df.loc[final_idx], HISTORY_FEATURES)
final_tm_prep = fit_preprocessor(model_df.loc[final_idx], TRACKMAN_FEATURES)
final_pitchers = np.sort(model_df.pitcher_id.unique())
final_pitcher_to_index = {int(pid): i + 1 for i, pid in enumerate(final_pitchers)}
final_arrays = make_arrays(final_idx, final_hist_prep, final_tm_prep, final_pitcher_to_index)
final_loader = make_loader(final_arrays, True)

torch.manual_seed(SEED + 999)
final_model = PitcherBrierNet(len(HISTORY_FEATURES), len(TRACKMAN_FEATURES), len(final_pitcher_to_index),
                              len(COHORTS), mode=selected_mode).to(DEVICE)
optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
for epoch in range(1, EPOCHS + 1):
    final_model.train()
    totals = np.zeros(2)
    seen = 0
    for history_x, trackman_x, pitcher_idx, cohort_idx, asof_n, y in final_loader:
        history_x, trackman_x = history_x.to(DEVICE), trackman_x.to(DEVICE)
        pitcher_idx, cohort_idx, asof_n, y = pitcher_idx.to(DEVICE), cohort_idx.to(DEVICE), asof_n.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        direct_logit, component_logits, _ = final_model(history_x, trackman_x, pitcher_idx, cohort_idx, asof_n)
        loss, parts = model_loss(final_model, direct_logit, component_logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(final_model.parameters(), 5.0)
        optimizer.step()
        n = len(y)
        totals += np.array([loss.item(), parts["brier"].item()]) * n
        seen += n
    print(f"final epoch {epoch:02d} | loss={totals[0]/seen:.6f} | brier={totals[1]/seen:.6f}")

# cutoff=2025: 2019~2024만 포함한 고정 Trackman 요약
tm_2025 = build_lagged_trackman(tm, cutoffs=[2025])
track_lookup_table = crosswalk[["pitcher_id", "pitcher_trackman_id"]].merge(
    tm_2025, on="pitcher_trackman_id", how="left"
)
for col in TRACKMAN_COUNT_RAW:
    track_lookup_table[col] = track_lookup_table[col].fillna(0)
track_lookup_table["tm_available"] = track_lookup_table.tm_prior_n.gt(0).astype("float32")

lookup_columns = [c for c in TRACKMAN_BASE_FEATURES if c in track_lookup_table.columns]
track_lookup = {}
for row in track_lookup_table[["pitcher_id"] + lookup_columns].itertuples(index=False):
    values = {}
    for name, value in zip(lookup_columns, row[1:]):
        values[name] = None if pd.isna(value) else float(value)
    track_lookup[int(row.pitcher_id)] = values

SUBMIT_DIR = PROJECT_ROOT / "experiment" / "pitcher_embedding" / "submit_v1"
MODEL_DIR = SUBMIT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

checkpoint = {
    "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
    "mode": selected_mode,
    "embedding_dim": 32,
    "history_features": HISTORY_FEATURES,
    "trackman_features": TRACKMAN_FEATURES,
    "trackman_base_features": TRACKMAN_BASE_FEATURES,
    "trackman_count_raw": TRACKMAN_COUNT_RAW,
    "history_preprocessor": final_hist_prep,
    "trackman_preprocessor": final_tm_prep,
    "pitcher_to_index": final_pitcher_to_index,
    "cohort_to_index": cohort_to_index,
    "track_lookup": track_lookup,
    "calibrator": calibrators[selected_mode],
    "metadata": {
        "quick_run": QUICK_RUN, "epochs": EPOCHS, "seed": SEED,
        "train_rows": int(len(final_idx)), "crosswalk_pitchers": int(len(crosswalk)),
        "test_independent": True, "auxiliary_labels": selected_mode == "component",
    },
}
checkpoint_path = MODEL_DIR / "model.pt"
torch.save(checkpoint, checkpoint_path)

summary = {
    "selected_mode": selected_mode,
    "calibrator": list(calibrators[selected_mode]),
    "quick_run": QUICK_RUN,
    "final_train_rows": int(len(final_idx)),
    "crosswalk_pitchers": int(len(track_lookup)),
    "checkpoint_mb": checkpoint_path.stat().st_size / 1024**2,
    "holdout": comparison.query("mode == @selected_mode and split == '2024_holdout' and probability == 'calibrated'").iloc[0].to_dict(),
}
with open(OUTPUT_DIR / "submission_artifact_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
print("checkpoint:", checkpoint_path)
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 9. 운영진 확인용 질문 문안

아래 문안은 자동 게시하지 않습니다.

> **[DACON 답변 요청] train.csv 누적 asof 피처를 이용한 보조 라벨 생성 가능 여부**  
> 안녕하세요. 동일 투수의 다음 학습 행에 제공된 `asof_pitcher_reverse_rate` 및 `asof_pitcher_middle_rate`의 누적 변화량을 이용하여 현재 학습 행의 reverse/middle 보조 라벨을 복원하고, 이를 학습용 정답으로만 사용하는 것이 허용되는지 문의드립니다. 모델 입력에는 현재 행에서 투구 직전까지 확인 가능한 정보만 사용하며, 테스트 데이터의 다른 행·순서·분포·누적 통계는 일절 사용하지 않습니다. 해당 방식이 허용되지 않는다면 `control_success`만 직접 학습하는 모델을 사용하겠습니다.

최종 제출 전에는 `QUICK_RUN=False` 전체 학습, 서버 기본 버전과 같은 환경의 샘플 실행, zip 내부 최상위 구조 확인이 필요합니다."""))

nb = nbf.v4.new_notebook(cells=cells, metadata=source_nb.metadata)
nbf.write(nb, DEST)
print(DEST)
