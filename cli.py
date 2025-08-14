#!/usr/bin/env python3
"""
Main CLI entry point for Playwright ML Test Maintenance Tool.
Provides commands for training, prediction, batch operations, and analysis.
"""

import click
import logging
import sys
from pathlib import Path

from scraper import WebScraper
from features import FeatureExtractor
from models import MLPredictor
from utils import setup_logging, load_config


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', default='config.json', help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """Playwright ML Test Maintenance Tool - Predict and fix web locators using ML."""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_level)
    
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    logging.info("Playwright ML Test Maintenance Tool initialized")


@cli.command()
@click.option('--urls-file', '-u', required=True, help='File containing URLs to scrape')
@click.option('--output', '-o', default='training_data.json', help='Output file for training data')
@click.option('--max-pages', '-m', default=50, help='Maximum pages to scrape')
@click.pass_context
def scrape(ctx, urls_file, output, max_pages):
    """Scrape web pages and extract features for training data."""
    try:
        scraper = WebScraper()
        extractor = FeatureExtractor()
        
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        logging.info(f"Starting to scrape {min(len(urls), max_pages)} URLs")
        
        training_data = []
        for i, url in enumerate(urls[:max_pages]):
            try:
                logging.info(f"Scraping {i+1}/{min(len(urls), max_pages)}: {url}")
                
                # Scrape the page
                html_content = scraper.scrape_page(url)
                if not html_content:
                    logging.warning(f"Failed to scrape {url}")
                    continue
                
                # Extract features
                features = extractor.extract_page_features(html_content, url)
                if features:
                    training_data.extend(features)
                    
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
                continue
        
        # Save training data
        import json
        with open(output, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        logging.info(f"Scraped {len(training_data)} features and saved to {output}")
        click.echo(f"✓ Scraping complete. {len(training_data)} features saved to {output}")
        
    except Exception as e:
        logging.error(f"Scraping failed: {e}")
        click.echo(f"✗ Scraping failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data-file', '-d', required=True, help='Training data file (JSON)')
@click.option('--model-output', '-m', default='locator_model.joblib', help='Output model file')
@click.option('--test-size', '-t', default=0.2, help='Test set size (0.0-1.0)')
@click.pass_context
def train(ctx, data_file, model_output, test_size):
    """Train machine learning model on scraped data."""
    try:
        predictor = MLPredictor()
        extractor = FeatureExtractor()
        
        logging.info(f"Loading training data from {data_file}")
        
        # Load and prepare data
        import json
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
        
        # Convert to DataFrame and clean
        df = extractor.prepare_training_dataframe(raw_data)
        logging.info(f"Loaded {len(df)} training samples")
        
        # Train model
        model_info = predictor.train_model(df, test_size=test_size)
        
        # Save model
        predictor.save_model(model_output)
        
        click.echo(f"✓ Model trained successfully!")
        click.echo(f"  - Accuracy: {model_info['accuracy']:.3f}")
        click.echo(f"  - Training samples: {model_info['train_samples']}")
        click.echo(f"  - Test samples: {model_info['test_samples']}")
        click.echo(f"  - Model saved to: {model_output}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        click.echo(f"✗ Training failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-file', '-m', required=True, help='Trained model file')
@click.option('--url', '-u', required=True, help='URL to analyze')
@click.option('--selector', '-s', help='Current selector to fix (optional)')
@click.pass_context
def predict(ctx, model_file, url, selector):
    """Predict best locators for a given URL and optional current selector."""
    try:
        predictor = MLPredictor()
        scraper = WebScraper()
        extractor = FeatureExtractor()
        
        # Load model
        predictor.load_model(model_file)
        logging.info(f"Model loaded from {model_file}")
        
        # Scrape target page
        logging.info(f"Scraping target page: {url}")
        html_content = scraper.scrape_page(url)
        
        if not html_content:
            raise Exception(f"Failed to scrape {url}")
        
        # Extract features and predict
        features = extractor.extract_page_features(html_content, url)
        
        if not features:
            raise Exception("No features extracted from page")
        
        predictions = predictor.predict_locators(features, current_selector=selector)
        
        # Display results
        click.echo(f"✓ Locator predictions for {url}:")
        click.echo("="*60)
        
        for i, pred in enumerate(predictions[:5], 1):
            confidence = pred.get('confidence', 0)
            locator = pred.get('locator', 'N/A')
            element_type = pred.get('element_type', 'unknown')
            
            click.echo(f"{i}. {locator}")
            click.echo(f"   Type: {element_type} | Confidence: {confidence:.3f}")
            click.echo()
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        click.echo(f"✗ Prediction failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-file', '-m', required=True, help='Trained model file')
@click.option('--urls-file', '-u', required=True, help='File containing URLs to analyze')
@click.option('--output', '-o', default='batch_predictions.json', help='Output file for predictions')
@click.option('--max-predictions', '-n', default=3, help='Max predictions per URL')
@click.pass_context
def batch_predict(ctx, model_file, urls_file, output, max_predictions):
    """Run batch predictions on multiple URLs."""
    try:
        predictor = MLPredictor()
        scraper = WebScraper()
        extractor = FeatureExtractor()
        
        # Load model
        predictor.load_model(model_file)
        
        # Load URLs
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        logging.info(f"Starting batch prediction for {len(urls)} URLs")
        
        all_predictions = {}
        
        for i, url in enumerate(urls):
            try:
                logging.info(f"Processing {i+1}/{len(urls)}: {url}")
                
                # Scrape and extract features
                html_content = scraper.scrape_page(url)
                if not html_content:
                    continue
                
                features = extractor.extract_page_features(html_content, url)
                if not features:
                    continue
                
                # Predict locators
                predictions = predictor.predict_locators(features)
                all_predictions[url] = predictions[:max_predictions]
                
                click.echo(f"  ✓ {len(predictions)} predictions generated")
                
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
                all_predictions[url] = []
                continue
        
        # Save results
        import json
        with open(output, 'w') as f:
            json.dump(all_predictions, f, indent=2, default=str)
        
        successful = len([p for p in all_predictions.values() if p])
        click.echo(f"✓ Batch prediction complete!")
        click.echo(f"  - URLs processed: {len(urls)}")
        click.echo(f"  - Successful predictions: {successful}")
        click.echo(f"  - Results saved to: {output}")
        
    except Exception as e:
        logging.error(f"Batch prediction failed: {e}")
        click.echo(f"✗ Batch prediction failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--data-file', '-d', required=True, help='Training/prediction data file')
@click.option('--output-dir', '-o', default='analysis_output', help='Output directory for analysis')
@click.pass_context
def analyze(ctx, data_file, output_dir):
    """Analyze training data or prediction results and generate reports."""
    try:
        from pathlib import Path
        import json
        import pandas as pd
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        extractor = FeatureExtractor()
        
        # Determine data type and analyze
        if isinstance(data, list) and data and 'features' in str(data[0]):
            # Training data analysis
            df = extractor.prepare_training_dataframe(data)
            analysis = extractor.analyze_features(df)
            
            # Save analysis
            analysis_file = Path(output_dir) / 'feature_analysis.json'
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            click.echo(f"✓ Feature analysis complete!")
            click.echo(f"  - Features analyzed: {len(df.columns)}")
            click.echo(f"  - Samples: {len(df)}")
            
        elif isinstance(data, dict):
            # Prediction results analysis
            total_predictions = sum(len(preds) for preds in data.values())
            avg_predictions = total_predictions / len(data) if data else 0
            
            summary = {
                'total_urls': len(data),
                'total_predictions': total_predictions,
                'average_predictions_per_url': avg_predictions,
                'urls_with_predictions': len([url for url, preds in data.items() if preds])
            }
            
            summary_file = Path(output_dir) / 'prediction_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            click.echo(f"✓ Prediction analysis complete!")
            click.echo(f"  - URLs analyzed: {summary['total_urls']}")
            click.echo(f"  - Total predictions: {summary['total_predictions']}")
        
        click.echo(f"  - Analysis saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        click.echo(f"✗ Analysis failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()