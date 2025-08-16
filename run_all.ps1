# =========================
# Playwright ML Workflow
# =========================
Write-Host "▶ Starting Playwright ML Test Maintenance Workflow..." -ForegroundColor Green

# 1. Scrape training data
Write-Host "`n--- Step 1: Scraping Training Data ---" -ForegroundColor Yellow
python -m src.cli scrape --urls-file urls.txt --output training_data.json --max-pages 50

# 2. Train the model
Write-Host "`n--- Step 2: Training Model ---" -ForegroundColor Yellow
python -m src.cli train --data-file training_data.json --model-output locator_model.joblib --test-size 0.2

# 3. Run Playwright tests and capture the error log (UTF-8, no ANSI codes)
Write-Host "`n--- Step 3: Running Tests & Capturing Failures ---" -ForegroundColor Yellow
npx playwright test tests/ 2>&1 | ForEach-Object { $_ -replace '\x1b\[[0-9;]*m', '' } | Out-File -FilePath test_errors.log -Encoding utf8

# 4. Batch fix failed locators
Write-Host "`n--- Step 4: Applying Batch Fix ---" -ForegroundColor Yellow
python -m src.cli batch-fix --test-dir tests/ --error-log test_errors.log --model-file locator_model.joblib --apply-fixes

# 5. Batch predictions
Write-Host "`n--- Step 5: Running Batch Predictions ---" -ForegroundColor Yellow
python -m src.cli batch-predict --model-file locator_model.joblib --urls-file urls.txt --output batch_predictions.json

# 6. Analyze predictions
Write-Host "`n--- Step 6: Analyzing Predictions ---" -ForegroundColor Yellow
python -m src.cli analyze --data-file batch_predictions.json --output-dir prediction_analysis

Write-Host "`n✅ Workflow Finished!" -ForegroundColor Green
