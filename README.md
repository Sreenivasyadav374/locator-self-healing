# Playwright ML Test Maintenance Tool

A powerful CLI tool that uses machine learning to predict and fix web locators for Playwright tests. The tool combines web scraping, feature extraction, and ML models to help maintain robust test automation.

## Features

- **Web Scraping**: Extract elements and features from web pages using Playwright and BeautifulSoup
- **Machine Learning**: Train models to predict locator quality and suggest improvements
- **Robust Data Handling**: Clean and process data with pandas, handling missing/None/empty values
- **CLI Interface**: Command-line tools for training, prediction, batch processing, and analysis
- **Batch Operations**: Process multiple URLs and generate comprehensive reports
- **Debug Logging**: Detailed logging for troubleshooting and monitoring

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

### 1. Scrape Training Data

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

### 2. Train the Model

Train ML models on the scraped data:

```bash
python cli.py train --data-file training_data.json --model-output locator_model.joblib --test-size 0.2
```

### 3. Predict Locators

Get locator suggestions for a specific URL:

```bash
python cli.py predict --model-file locator_model.joblib --url "https://example.com/login" --selector "#old-login-btn"
```

### 4. Batch Predictions

Run predictions on multiple URLs:

```bash
python cli.py batch-predict --model-file locator_model.joblib --urls-file urls.txt --output batch_predictions.json
```

### 5. Analyze Data

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

1. **cli.py**: Main CLI interface with Click commands
2. **scraper.py**: Web scraping using Playwright and BeautifulSoup
3. **features.py**: Feature extraction and data cleaning
4. **models.py**: ML models using scikit-learn and TensorFlow
5. **utils.py**: Utility functions and configuration management

### Data Flow

1. **Scraping**: Extract HTML content and element information
2. **Feature Extraction**: Convert elements to numerical features
3. **Data Cleaning**: Handle missing values and normalize data
4. **Training**: Train ML models on cleaned features
5. **Prediction**: Generate locator suggestions with confidence scores

### Feature Engineering

The tool extracts these key features from web elements:

- **Basic Features**: tag, text content, element type
- **Attribute Features**: ID, classes, name, placeholder, aria-label, etc.
- **Parent Features**: parent tag, parent attributes
- **Selector Features**: available selectors and their quality scores
- **Specificity Features**: calculated reliability scores
- **Context Features**: form element, interactive element flags

## Example Workflow

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

3. **Network Timeouts**: Increase timeout in config.json
   ```json
   {
     "scraping": {
       "timeout": 60000
     }
   }
   ```

### Debug Logging

Use verbose mode to see detailed logs:

```bash
python cli.py --verbose scrape -u urls.txt
```

Logs are saved to `logs/playwright_ml_tool.log`.

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License.