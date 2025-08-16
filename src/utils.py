"""
Utility functions for configuration, logging, and data processing.
Provides robust helper functions for the ML test maintenance tool.
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler('logs/playwright_ml_tool.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('playwright').setLevel(logging.WARNING)
    

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logging.info(f"Configuration loaded from {config_path}")
        return config
        
    except FileNotFoundError:
        logging.warning(f"Configuration file {config_path} not found, using defaults")
        return get_default_config()
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'scraping': {
            'timeout': 30000,
            'headless': True,
            'max_concurrent': 3,
            'wait_time': 2000
        },
        'ml': {
            'test_size': 0.2,
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 10
        },
        'features': {
            'text_max_length': 200,
            'max_selectors': 5,
            'quality_threshold': 2.0
        }
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Configuration saved to {config_path}")
        
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")
        raise


def clean_dataframe(df: pd.DataFrame, numeric_default: float = 0.0, string_default: str = '') -> pd.DataFrame:
    """
    Clean pandas DataFrame by handling missing values robustly.
    
    Args:
        df: Input DataFrame
        numeric_default: Default value for numeric columns
        string_default: Default value for string columns
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        logging.warning("DataFrame is empty, returning as is")
        return df
    
    df_cleaned = df.copy()
    
    try:
        # Handle numeric columns
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Replace NaN, inf, -inf with default
            df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)
            df_cleaned[col] = df_cleaned[col].fillna(numeric_default)
            
            # Ensure all values are finite
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(numeric_default)
        
        # Handle string/object columns
        object_columns = df_cleaned.select_dtypes(include=['object']).columns
        
        for col in object_columns:
            # Skip raw data columns
            if 'raw_' in col or '_info' in col:
                continue
                
            # Replace NaN, None, empty strings
            df_cleaned[col] = df_cleaned[col].fillna(string_default)
            df_cleaned[col] = df_cleaned[col].astype(str)
            df_cleaned[col] = df_cleaned[col].replace(['nan', 'None', 'null', 'NaN'], string_default)
            
            # Clean empty strings if needed
            if string_default != '':
                df_cleaned[col] = df_cleaned[col].replace('', string_default)
        
        logging.debug(f"DataFrame cleaned: {len(df_cleaned)} rows, {len(df_cleaned.columns)} columns")
        
        return df_cleaned
        
    except Exception as e:
        logging.error(f"Error cleaning DataFrame: {e}")
        return df  # Return original if cleaning fails


def validate_urls(urls: list) -> list:
    """
    Validate and clean a list of URLs.
    
    Args:
        urls: List of URL strings
        
    Returns:
        List of valid URLs
    """
    import re
    
    valid_urls = []
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    for url in urls:
        url = url.strip()
        if url and url_pattern.match(url):
            valid_urls.append(url)
        else:
            logging.warning(f"Invalid URL skipped: {url}")
    
    logging.info(f"Validated {len(valid_urls)} URLs out of {len(urls)}")
    return valid_urls


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        from textdistance import cosine
        
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1 = str(text1).lower().strip()
        text2 = str(text2).lower().strip()
        
        if text1 == text2:
            return 1.0
        
        # Use cosine similarity
        similarity = cosine.normalized_similarity(text1, text2)
        return float(similarity)
        
    except Exception as e:
        logging.warning(f"Error calculating text similarity: {e}")
        return 0.0


def extract_numbers_from_text(text: str) -> list:
    """
    Extract all numbers from text string.
    
    Args:
        text: Input text
        
    Returns:
        List of numbers found in text
    """
    import re
    
    if not text:
        return []
    
    try:
        # Find all numbers (integers and floats)
        numbers = re.findall(r'-?\d+\.?\d*', str(text))
        return [float(num) for num in numbers if num]
    except Exception as e:
        logging.warning(f"Error extracting numbers from text: {e}")
        return []


def safe_dict_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary with robust error handling.
    
    Args:
        dictionary: Source dictionary
        key: Key to retrieve
        default: Default value if key not found or error occurs
        
    Returns:
        Value from dictionary or default
    """
    try:
        if not isinstance(dictionary, dict):
            return default
        
        value = dictionary.get(key, default)
        
        # Handle None values
        if value is None:
            return default
        
        # Handle NaN values for numeric types
        if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
            return default if default is not None else 0.0
        
        # Handle empty strings
        if isinstance(value, str) and not value.strip():
            return default if default is not None else ''
        
        return value
        
    except Exception as e:
        logging.warning(f"Error getting value for key '{key}': {e}")
        return default


def ensure_directory_exists(directory_path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path object for the directory
    """
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logging.error(f"Error creating directory {directory_path}: {e}")
        raise


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024.0
        unit_index += 1
    
    return f"{size_bytes:.1f} {units[unit_index]}"


def batch_process_with_progress(items: list, process_func, batch_size: int = 10, description: str = "Processing") -> list:
    """
    Process items in batches with progress indication.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        batch_size: Number of items to process in each batch
        description: Description for progress indication
        
    Returns:
        List of processed results
    """
    results = []
    total_items = len(items)
    
    for i in range(0, total_items, batch_size):
        batch = items[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_items + batch_size - 1) // batch_size
        
        logging.info(f"{description} - Batch {batch_num}/{total_batches} ({len(batch)} items)")
        
        for item in batch:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                logging.warning(f"Error processing item {item}: {e}")
                results.append(None)
    
    successful_results = [r for r in results if r is not None]
    logging.info(f"{description} complete: {len(successful_results)}/{total_items} successful")
    
    return results