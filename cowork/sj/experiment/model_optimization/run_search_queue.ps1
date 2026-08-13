$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\isj67\Desktop\LGAIMERS"
$runner = Join-Path $projectRoot "experiment\model_optimization\run_optuna_family.py"
$python = "C:\Users\isj67\anaconda3\python.exe"
Set-Location -LiteralPath $projectRoot

Write-Output "[$(Get-Date -Format o)] XGBoost search start"
& $python $runner `
    --family xgboost `
    --trials 140 `
    --folds 2023,2024 `
    --study-name xgboost_v1_full_2023_2024
if ($LASTEXITCODE -ne 0) {
    throw "XGBoost search failed with exit code $LASTEXITCODE"
}

Write-Output "[$(Get-Date -Format o)] CatBoost search start"
& $python $runner `
    --family catboost `
    --trials 140 `
    --folds 2023,2024 `
    --study-name catboost_v1_full_2023_2024
if ($LASTEXITCODE -ne 0) {
    throw "CatBoost search failed with exit code $LASTEXITCODE"
}

Write-Output "[$(Get-Date -Format o)] Search queue complete"
