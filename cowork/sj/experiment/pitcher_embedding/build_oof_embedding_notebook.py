from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "pitcher_embedding_brier_submission.ipynb"
DEST = HERE / "pitcher_embedding_oof_features.ipynb"

source_nb = nbf.read(SOURCE, as_version=4)
cells = source_nb.cells[:10]
for cell in cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

cells[0].source = r"""# 투수 임베딩 v1 — 시즌 순방향 OOF 피처

팀의 다음 단계 모델이 학습 데이터에서 투수 임베딩을 안전하게 사용할 수 있도록 시즌 순방향 OOF(out-of-fold) lookup을 만듭니다.

- 시즌 `s`의 임베딩 모델은 `season < s`인 정답만 학습합니다.
- 시즌 `s`의 모든 행에는 시즌 첫 투구 이전에 확정된 동일한 투수-시즌 임베딩을 붙입니다.
- 출력은 `(pitcher_id, season)`당 한 행입니다.
- 48차원 계약: 투수 ID 16 + 과거 Trackman tower 24 + 신인/경험 cohort 8
- 2019~2020은 충분한 이전 Trackman-supervised 모델을 만들 수 없어 0 벡터와 `oof_available=False`를 제공합니다.

> component 모델은 reverse/middle 보조 라벨을 사용하므로 운영진 답변 전까지 실험 피처입니다. 테스트 행이나 테스트 분포는 전혀 사용하지 않습니다.
"""

cells.append(nbf.v4.new_markdown_cell(r"""## 5. OOF 학습 설정

2021, 2022, 2023, 2024 시즌을 각각 독립 fold로 만듭니다. 빠른 협업용 생성에서는 fold별 최대 30만 행과 2 epoch를 사용합니다. 모델 성능 제출이 아니라 누수 없는 표현 생성이 목적입니다."""))

cells.append(nbf.v4.new_code_cell(r"""OOF_MAX_ROWS = 300_000
OOF_EPOCHS = 2

HISTORY_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before", "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li", "pitcher_hand", "batter_hand",
] + ASOF_COLS
TRACKMAN_COUNT_RAW = ["tm_prior_n", "tm_prev_season_n", "tm_prior_max_season_n"]
TRACKMAN_BASE_FEATURES = [
    c for c in model_df.columns if c.startswith("tm_") and c != "tm_available"
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

COHORTS = ["UNSEEN", "ROOKIE_1_25", "ROOKIE_26_100", "RETURNING", "VETERAN"]
cohort_to_index = {name: i for i, name in enumerate(COHORTS)}
TARGET_COLS = ["y_reverse", "y_middle", "y_far_residual", "control_success"]

def sample_indices(pool, limit, seed):
    pool = np.asarray(pool)
    if len(pool) <= limit:
        return np.sort(pool)
    return np.sort(np.random.default_rng(seed).choice(pool, limit, replace=False))

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
"""))

cells.append(nbf.v4.new_code_cell(r"""class OOFEmbeddingNet(nn.Module):
    def __init__(self, hist_dim, tm_dim, n_pitchers, n_cohorts):
        super().__init__()
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
            nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU(),
        )
        self.reverse_head = nn.Linear(32, 1)
        self.middle_head = nn.Linear(32, 1)
        self.far_head = nn.Linear(32, 1)

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
        logits = torch.cat([self.reverse_head(embedding), self.middle_head(embedding), self.far_head(embedding)], dim=1)
        return logits

    def static_embedding(self, trackman_x, pitcher_idx, cohort_idx):
        individual = self.individual_projection(self.pitcher_embedding(pitcher_idx))
        trackman = self.trackman_tower(trackman_x)
        cohort = self.cohort_embedding(cohort_idx)
        return torch.cat([individual, trackman, cohort], dim=1)


bce = nn.BCEWithLogitsLoss(reduction="none")

def component_loss(logits, y):
    component_p = torch.sigmoid(logits)
    success_p = torch.prod(1 - component_p, dim=1)
    brier = torch.mean((success_p - y[:, 3]) ** 2)
    reverse = bce(logits[:, 0], y[:, 0]).mean()
    no_reverse = y[:, 0].eq(0)
    middle = bce(logits[no_reverse, 1], y[no_reverse, 1]).mean()
    neither = no_reverse & y[:, 1].eq(0)
    far = bce(logits[neither, 2], y[neither, 2]).mean()
    return brier + 0.10 * (reverse + middle + far), brier
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 6. 시즌 순방향 학습과 48차원 추출

임베딩 입력에서 경기 상황 tower는 제거하고, 투수 ID·과거 Trackman·신인 cohort 표현만 연결합니다. 따라서 같은 시즌의 모든 투구에 동일하게 붙일 수 있습니다."""))

cells.append(nbf.v4.new_code_cell(r"""season_start = (
    model_df.sort_values(["season", "pitcher_id", "asof_pitcher_n", "row_id"])
    .groupby(["season", "pitcher_id"], sort=False, as_index=False)
    .head(1).copy()
)

embedding_columns = [
    *[f"pitcher_embedding_{i:02d}" for i in range(16)],
    *[f"trackman_embedding_{i:02d}" for i in range(24)],
    *[f"cohort_embedding_{i:02d}" for i in range(8)],
]
oof_frames = []

# 2019~2020은 prior supervised Trackman tower가 불충분하므로 명시적 0 fallback.
fallback = season_start[season_start.season.le(2020)][[
    "pitcher_id", "season", "experience_cohort", "asof_pitcher_n",
    "tm_prior_n", "tm_prev_season_n", "tm_available",
]].copy()
fallback["oof_available"] = False
fallback["trained_through_season"] = fallback.season - 1
fallback["pitcher_known_before_season"] = False
for column in embedding_columns:
    fallback[column] = 0.0
oof_frames.append(fallback)

for target_season in range(2021, 2025):
    pool = model_df.index[model_df.component_label_valid & model_df.season.lt(target_season)]
    train_idx = sample_indices(pool, OOF_MAX_ROWS, SEED + target_season)
    train_frame = model_df.loc[train_idx]
    hist_prep = fit_preprocessor(train_frame, HISTORY_FEATURES)
    tm_prep = fit_preprocessor(train_frame, TRACKMAN_FEATURES)
    known_pitchers = np.sort(model_df.loc[model_df.season.lt(target_season), "pitcher_id"].unique())
    pitcher_map = {int(pid): i + 1 for i, pid in enumerate(known_pitchers)}

    history_x = transform_numeric(train_frame, HISTORY_FEATURES, hist_prep)
    trackman_x = transform_numeric(train_frame, TRACKMAN_FEATURES, tm_prep)
    pitcher_idx = train_frame.pitcher_id.map(pitcher_map).fillna(0).astype("int64").to_numpy()
    cohort_idx = train_frame.experience_cohort.map(cohort_to_index).astype("int64").to_numpy()
    asof_n = train_frame.asof_pitcher_n.astype("float32").to_numpy()
    y = train_frame[TARGET_COLS].to_numpy("float32")
    loader = DataLoader(TensorDataset(
        torch.from_numpy(history_x), torch.from_numpy(trackman_x), torch.from_numpy(pitcher_idx),
        torch.from_numpy(cohort_idx), torch.from_numpy(asof_n), torch.from_numpy(y),
    ), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    torch.manual_seed(SEED + target_season)
    fold_model = OOFEmbeddingNet(len(HISTORY_FEATURES), len(TRACKMAN_FEATURES), len(pitcher_map), len(COHORTS)).to(DEVICE)
    optimizer = torch.optim.AdamW(fold_model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, OOF_EPOCHS + 1):
        fold_model.train()
        total_brier = seen = 0.0
        for hx, tx, pi, ci, nn_, yy in loader:
            hx, tx, pi, ci, nn_, yy = hx.to(DEVICE), tx.to(DEVICE), pi.to(DEVICE), ci.to(DEVICE), nn_.to(DEVICE), yy.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = fold_model(hx, tx, pi, ci, nn_)
            loss, brier_part = component_loss(logits, yy)
            loss.backward()
            nn.utils.clip_grad_norm_(fold_model.parameters(), 5.0)
            optimizer.step()
            total_brier += brier_part.item() * len(yy)
            seen += len(yy)
        print(f"target={target_season} epoch={epoch} train_rows={len(train_idx):,} brier={total_brier/seen:.6f}")

    target = season_start[season_start.season.eq(target_season)].copy()
    target_tm = transform_numeric(target, TRACKMAN_FEATURES, tm_prep)
    target_p = target.pitcher_id.map(pitcher_map).fillna(0).astype("int64").to_numpy()
    target_c = target.experience_cohort.map(cohort_to_index).astype("int64").to_numpy()
    fold_model.eval()
    with torch.inference_mode():
        vector = fold_model.static_embedding(
            torch.from_numpy(target_tm).to(DEVICE), torch.from_numpy(target_p).to(DEVICE),
            torch.from_numpy(target_c).to(DEVICE),
        ).cpu().numpy()

    out = target[[
        "pitcher_id", "season", "experience_cohort", "asof_pitcher_n",
        "tm_prior_n", "tm_prev_season_n", "tm_available",
    ]].reset_index(drop=True)
    out["oof_available"] = True
    out["trained_through_season"] = target_season - 1
    out = pd.concat([out, pd.DataFrame(vector, columns=embedding_columns)], axis=1)
    out["pitcher_known_before_season"] = target_p != 0
    oof_frames.append(out)

oof = pd.concat(oof_frames, ignore_index=True).sort_values(["season", "pitcher_id"]).reset_index(drop=True)
if oof[["season", "pitcher_id"]].duplicated().any():
    raise ValueError("Duplicate pitcher-season key")
if not np.isfinite(oof[embedding_columns].to_numpy()).all():
    raise ValueError("Non-finite OOF embedding")
if (oof.loc[oof.oof_available, "trained_through_season"] >= oof.loc[oof.oof_available, "season"]).any():
    raise ValueError("Temporal leakage detected")

oof_path = OUTPUT_DIR / "pitcher_season_embedding_oof.parquet"
oof.to_parquet(oof_path, index=False)
oof.to_csv(OUTPUT_DIR / "pitcher_season_embedding_oof.csv", index=False)
print("saved:", oof_path)
print("shape:", oof.shape, "available:", oof.oof_available.mean())
display(oof.groupby("season").agg(pitchers=("pitcher_id", "size"), available=("oof_available", "mean"),
                                   known=("pitcher_known_before_season", "mean")))
display(oof.head())
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 7. 팀 사용 계약

학습 데이터에는 `pitcher_season_embedding_oof.parquet`를 `(pitcher_id, season)`으로 결합합니다. 2025 평가 행에는 `pitcher_embedding_lookup_2025.parquet`를 `pitcher_id`로 결합합니다.

- OOF 파일: 2019~2024 학습용
- 2025 lookup: 평가/추론용
- 2019~2020 OOF: 0 벡터, `oof_available=False`
- 새 투수: `pitcher_known_before_season=False`, cohort 표현과 Trackman fallback 사용
- reverse/middle 보조 라벨이 불허되면 이 OOF 파일은 폐기하고 direct-success 또는 비지도 임베딩으로 다시 생성해야 합니다.
"""))

nb = nbf.v4.new_notebook(cells=cells, metadata=source_nb.metadata)
nbf.write(nb, DEST)
print(DEST)
