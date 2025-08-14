# Playwright ML Test Maintenance Tool

A powerful CLI tool that uses machine learning to predict and fix web locators for Playwright tests. The tool combines web scraping, feature extraction, and ML models to help maintain robust test automation.

## Features

- **🔧 Test Locator Fixing**: Automatically fix failed Playwright test locators using ML
- **📝 Test Parsing**: Parse Playwright test files to extract locator information and context
- **🤖 Machine Learning**: Train models to predict locator quality and suggest improvements
- **🌐 Web Scraping**: Extract elements and features from web pages using Playwright and BeautifulSoup
- **🔄 Batch Operations**: Fix multiple failed locators from test runs automatically
- **📊 Robust Data Handling**: Clean and process data with pandas, handling missing/None/empty values
- **💻 CLI Interface**: Command-line tools for training, prediction, fixing, and analysis
- **📋 Debug Logging**: Detailed logging for troubleshooting and monitoring

## Installation

### 1. Set up Python Virtual Environment

```bash
# Create virtual environment
python -m venv playwright_ml_env

# Activate virtual environment
# On Windows:
playwright_ml_env\Scripts\activate
# On macOS/Linux:
source playwright_ml_env/bin/activate
```

### 2. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Playwright browsers
playwright install
```

### 3. Verify Installation

```bash
# Test the CLI
python cli.py --help
```

## Usage

### 1. Fix a Failed Locator

When a Playwright test fails due to a locator not finding an element:

```bash
# Fix a single failed locator
python cli.py fix-locator \
  --test-file tests/test_login.py \
  --failed-selector "#old-login-btn" \
  --model-file locator_model.joblib \
  --page-url "https://example.com/login" \
  --apply-fix
```

### 2. Batch Fix Multiple Failed Locators

Fix all failed locators from a test run:

```bash
# Run your tests and capture errors
pytest tests/ > test_errors.log 2>&1

# Fix all failed locators
python cli.py batch-fix \
  --test-dir tests/ \
  --error-log test_errors.log \
  --model-file locator_model.joblib \
  --apply-fixes
```

### 3. Parse Test Files

Analyze test files to understand locator usage:

```bash
python cli.py parse-test --test-file tests/test_login.py --output parsed_test.json
```

### 4. Scrape Training Data

Create a file `urls.txt` with URLs to scrape (one per line):

```
https://example.com/login
https://example.com/signup
https://example.com/dashboard
```

Scrape the pages:

```bash
python cli.py scrape --urls-file urls.txt --output training_data.json --max-pages 50
```

### 5. Train the Model

Train ML models on the scraped data:

```bash
python cli.py train --data-file training_data.json --model-output locator_model.joblib --test-size 0.2
```

### 6. Predict Locators

Get locator suggestions for a specific URL:

```bash
python cli.py predict --model-file locator_model.joblib --url "https://example.com/login" --selector "#old-login-btn"
```

### 7. Batch Predictions

Run predictions on multiple URLs:

```bash
python cli.py batch-predict --model-file locator_model.joblib --urls-file urls.txt --output batch_predictions.json
```

### 8. Analyze Data

Generate analysis reports:

```bash
# Analyze training data
python cli.py analyze --data-file training_data.json --output-dir analysis_reports

# Analyze prediction results
python cli.py analyze --data-file batch_predictions.json --output-dir prediction_analysis
```

## CLI Options

### Global Options

- `--verbose, -v`: Enable verbose logging
- `--config, -c`: Specify configuration file path

### Command-Specific Options

#### fix-locator
- `--test-file, -t`: Test file containing failed locator (required)
- `--failed-selector, -s`: The selector that failed (required)
- `--model-file, -m`: Trained model file (required)
- `--page-url, -u`: URL of the page being tested (optional)
- `--error-log, -e`: Error log file from test execution (optional)
- `--apply-fix, -a`: Automatically apply the best fix (flag)

#### batch-fix
- `--test-dir, -t`: Directory containing test files (required)
- `--error-log, -e`: Error log file from test execution (required)
- `--model-file, -m`: Trained model file (required)
- `--output, -o`: Output file for fix results (default: batch_fix_results.json)
- `--apply-fixes, -a`: Automatically apply the best fixes (flag)

#### parse-test
- `--test-file, -t`: Test file to parse (required)
- `--output, -o`: Output file for parsed test info (default: parsed_test.json)

#### scrape
- `--urls-file, -u`: File containing URLs to scrape (required)
- `--output, -o`: Output file for training data (default: training_data.json)
- `--max-pages, -m`: Maximum pages to scrape (default: 50)

#### train
- `--data-file, -d`: Training data file (required)
- `--model-output, -m`: Output model file (default: locator_model.joblib)
- `--test-size, -t`: Test set size 0.0-1.0 (default: 0.2)

#### predict
- `--model-file, -m`: Trained model file (required)
- `--url, -u`: URL to analyze (required)
- `--selector, -s`: Current selector to improve (optional)

#### batch-predict
- `--model-file, -m`: Trained model file (required)
- `--urls-file, -u`: File containing URLs (required)
- `--output, -o`: Output file (default: batch_predictions.json)
- `--max-predictions, -n`: Max predictions per URL (default: 3)

## Configuration

Create a `config.json` file to customize settings:

```json
{
  "scraping": {
    "timeout": 30000,
    "headless": true,
    "max_concurrent": 3,
    "wait_time": 2000
  },
  "ml": {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 10
  },
  "features": {
    "text_max_length": 200,
    "max_selectors": 5,
    "quality_threshold": 2.0
  }
}
```

## Architecture

### Core Modules

1. **cli.py**: Main CLI interface with Click commands for all operations
2. **locator_fixer.py**: Core locator fixing logic using ML and context analysis
3. **test_parser.py**: Parse Playwright test files and extract locator information
4. **scraper.py**: Web scraping using Playwright and BeautifulSoup
5. **features.py**: Feature extraction and data cleaning
6. **models.py**: ML models using scikit-learn for locator prediction
7. **utils.py**: Utility functions and configuration management

### Data Flow

#### For Training:
1. **Scraping**: Extract HTML content and element information from URLs
2. **Feature Extraction**: Convert elements to numerical features
3. **Data Cleaning**: Handle missing values and normalize data
4. **Training**: Train ML models on cleaned features

#### For Fixing Failed Locators:
1. **Test Parsing**: Extract failed locator and context from test file
2. **Page Scraping**: Get current page elements for ML analysis
3. **Context Analysis**: Understand test intent and expected interactions
4. **ML Prediction**: Generate locator suggestions with confidence scores
5. **Fix Application**: Replace failed locators in test files with backups

### Feature Engineering

The tool extracts these key features from web elements:

- **Basic Features**: tag, text content, element type
- **Attribute Features**: ID, classes, name, placeholder, aria-label, etc.
- **Parent Features**: parent tag, parent attributes
- **Selector Features**: available selectors and their quality scores
- **Specificity Features**: calculated reliability scores
- **Context Features**: form element, interactive element flags

### Test Context Analysis

The tool analyzes test context to improve fix suggestions:

- **Intent Detection**: Login forms, submit buttons, search elements, navigation
- **Interaction Analysis**: Click, fill, type, assertion patterns
- **Element Type Hints**: Button, input, link classification
- **Surrounding Actions**: Context from nearby test code

## Example Workflow

### Training the Model

```bash
# 1. Prepare URLs file
echo "https://github.com/login" > test_urls.txt
echo "https://github.com/signup" >> test_urls.txt

# 2. Scrape training data
python cli.py scrape -u test_urls.txt -o github_data.json -v

# 3. Train model
python cli.py train -d github_data.json -m github_model.joblib -v

# 4. Get predictions
python cli.py predict -m github_model.joblib -u "https://github.com/login" -v

# 5. Analyze results
python cli.py analyze -d github_data.json -o github_analysis -v
```

### Fixing Failed Tests

```bash
# 1. Run tests and capture failures
pytest tests/test_login.py > test_errors.log 2>&1

# 2. Fix a specific failed locator
python cli.py fix-locator \
  -t tests/test_login.py \
  -s "#login-button" \
  -m github_model.joblib \
  -u "https://github.com/login" \
  -e test_errors.log \
  --apply-fix

# 3. Or fix all failed locators at once
python cli.py batch-fix \
  -t tests/ \
  -e test_errors.log \
  -m github_model.joblib \
  --apply-fixes

# 4. Re-run tests to verify fixes
pytest tests/test_login.py
```

## Troubleshooting

### Common Issues

1. **Browser Installation**: Make sure Playwright browsers are installed
   ```bash
   playwright install
   ```

2. **Memory Issues**: Reduce batch size or max pages for large datasets
   ```bash
   python cli.py scrape -u urls.txt --max-pages 20
   ```

3. **Test Parsing Issues**: Ensure test files follow standard Playwright patterns
   ```python
   # Good patterns that the parser recognizes:
   page.locator("#login-button").click()
   page.get_by_test_id("submit-btn").click()
   page.get_by_text("Login").click()
   ```

4. **Network Timeouts**: Increase timeout in config.json
   ```json
   {
     "scraping": {
       "timeout": 60000
     }
   }
   ```

5. **Model Not Found**: Make sure to train a model first
   ```bash
   python cli.py train -d training_data.json -m locator_model.joblib
   ```

### Debug Logging

Use verbose mode to see detailed logs:

```bash
python cli.py --verbose scrape -u urls.txt
```

Logs are saved to `logs/playwright_ml_tool.log`.

### Test File Backup

The tool automatically creates backups when applying fixes:
- Original: `tests/test_login.py`
- Backup: `tests/test_login.py.backup`

You can restore from backup if needed:
```bash
cp tests/test_login.py.backup tests/test_login.py
```

## Dependencies

- **click**: CLI interface framework
- **scikit-learn**: Machine learning models
- **tensorflow**: Advanced ML capabilities (optional)
- **pandas**: Data manipulation and cleaning
- **playwright**: Web automation and scraping
- **beautifulsoup4**: HTML parsing
- **textdistance**: Text similarity calculations
- **numpy**: Numerical computations
- **lxml**: XML/HTML processing

## Real-World Usage Examples

### Example 1: Login Test Fix

**Failed Test:**
```python
def test_login(page):
    page.goto("https://example.com/login")
    page.locator("#login-btn").click()  # This fails
    # ... rest of test
```

**Fix Command:**
```bash
python cli.py fix-locator -t tests/test_login.py -s "#login-btn" -m model.joblib -u "https://example.com/login" --apply-fix
```

**Result:**
```python
def test_login(page):
    page.goto("https://example.com/login")
    page.get_by_test_id("login-button").click()  # Fixed!
    # ... rest of test
```

### Example 2: Batch Fix After UI Changes

When a website redesign breaks multiple tests:

```bash
# Run all tests and capture failures
pytest tests/ --tb=short > test_failures.log 2>&1

# Fix all failed locators automatically
python cli.py batch-fix -t tests/ -e test_failures.log -m model.joblib --apply-fixes

# Verify fixes worked
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License.