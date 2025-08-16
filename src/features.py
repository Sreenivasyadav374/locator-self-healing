"""
Feature extraction module for web elements.
Handles robust data cleaning, feature engineering, and DataFrame preparation.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from textdistance import levenshtein, cosine
from bs4 import BeautifulSoup
from src.scraper import WebScraper


class FeatureExtractor:
    """Extract and clean features from web elements for ML training/prediction."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.logger = logging.getLogger(__name__)
        self.scraper = WebScraper()
    
    def extract_page_features(self, html_content: str, url: str) -> List[Dict[str, Any]]:
        """
        Extract features from a single page's HTML content.
        
        Args:
            html_content: Raw HTML content
            url: Page URL
            
        Returns:
            List of feature dictionaries for each element
        """
        features = []
        
        try:
            # Parse HTML
            soup = self.scraper.parse_html(html_content)
            if not soup:
                self.logger.error(f"Failed to parse HTML for {url}")
                return features
            
            # Extract element information
            elements_info = self.scraper.extract_elements_info(soup)
            
            # Convert to features
            for element_info in elements_info:
                try:
                    element_features = self._extract_element_features(element_info, url)
                    if element_features:
                        features.append(element_features)
                except Exception as e:
                    self.logger.warning(f"Error extracting features for element: {e}")
                    continue
            
            self.logger.debug(f"Extracted {len(features)} feature sets from {url}")
            
        except Exception as e:
            self.logger.error(f"Error extracting page features: {e}")
        
        return features
    
    def _extract_element_features(self, element_info: Dict[str, Any], url: str) -> Optional[Dict[str, Any]]:
        """Extract ML features from element information."""
        try:
            features = {
                # Basic features
                'url': url,
                'tag': element_info.get('tag', ''),
                'element_type': self._classify_element_type(element_info),

                # Text features
                'has_text': 1.0 if element_info.get('text', '').strip() else 0.0,
                'text_length': float(len(element_info.get('text', ''))),
                'text_word_count': float(len(element_info.get('text', '').split())),

                # Attribute features...
                'has_id': 1.0 if element_info.get('id', '').strip() else 0.0,
                'has_classes': 1.0 if element_info.get('classes', '').strip() else 0.0,
                # (rest unchanged) ...
            }

            # ✅ NEW: intent-word matching if provided by parser/context
            intent_word = element_info.get('test_intent', '').lower()
            text_val = element_info.get('text', '').lower()
            if intent_word and intent_word in text_val:
                features['intent_word_in_text'] = 1.0
            else:
                features['intent_word_in_text'] = 0.0

            # Parent features...
            # Selector features...
            # Specificity features...
            # Context features...
            features['raw_element_info'] = element_info

            features = self._clean_features(features)
            return features

        except Exception as e:
            self.logger.warning(f"Error extracting element features: {e}")
            return None
    
    def _classify_element_type(self, element_info: Dict[str, Any]) -> str:
        """Classify element type for better predictions."""
        tag = element_info.get('tag', '').lower()
        input_type = element_info.get('type', '').lower()
        
        if tag == 'input':
            if input_type in ['submit', 'button']:
                return 'button'
            elif input_type in ['text', 'email', 'password', 'number']:
                return 'text_input'
            elif input_type in ['checkbox', 'radio']:
                return 'choice_input'
            else:
                return 'input'
        elif tag == 'button':
            return 'button'
        elif tag == 'a':
            return 'link'
        elif tag in ['select']:
            return 'dropdown'
        elif tag in ['textarea']:
            return 'text_area'
        else:
            return 'other'
    
    def _get_best_selector(self, selectors: List[str]) -> str:
        """Get the most reliable selector from a list."""
        if not selectors:
            return ''
        
        # Prioritize selectors: ID > data-testid > name > class > tag
        priority_order = ['#', '[data-testid', '[name', '.', 'tag']
        
        for priority in priority_order:
            for selector in selectors:
                if selector.startswith(priority):
                    return selector
        
        return selectors[0] if selectors else ''
    
    def _calculate_id_specificity(self, element_info: Dict[str, Any]) -> float:
        """Calculate specificity score for ID-based selection."""
        element_id = element_info.get('id', '').strip()
        if not element_id:
            return 0.0
        
        # Higher score for meaningful IDs
        meaningful_patterns = ['btn', 'button', 'input', 'form', 'submit', 'search']
        score = 1.0
        
        for pattern in meaningful_patterns:
            if pattern in element_id.lower():
                score += 0.5
        
        return min(score, 3.0)  # Cap at 3.0
    
    def _calculate_class_specificity(self, element_info: Dict[str, Any]) -> float:
        """Calculate specificity score for class-based selection."""
        classes = element_info.get('classes', '').strip()
        if not classes:
            return 0.0
        
        class_list = classes.split()
        base_score = len(class_list) * 0.3
        
        # Bonus for semantic classes
        semantic_patterns = ['btn', 'button', 'input', 'form', 'submit', 'nav', 'menu']
        for pattern in semantic_patterns:
            for class_name in class_list:
                if pattern in class_name.lower():
                    base_score += 0.3
        
        return min(base_score, 2.0)  # Cap at 2.0
    
    def _calculate_attribute_specificity(self, element_info: Dict[str, Any]) -> float:
        """Calculate specificity score for attribute-based selection."""
        score = 0.0
        
        # High-value attributes
        high_value_attrs = ['data_testid', 'name', 'aria_label']
        for attr in high_value_attrs:
            if element_info.get(attr, '').strip():
                score += 1.0
        
        # Medium-value attributes
        medium_value_attrs = ['type', 'role', 'title']
        for attr in medium_value_attrs:
            if element_info.get(attr, '').strip():
                score += 0.5
        
        return min(score, 3.0)  # Cap at 3.0
    
    def _clean_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize feature values."""
        cleaned = {}
        
        for key, value in features.items():
            # Skip raw data from numeric cleaning
            if key == 'raw_element_info':
                cleaned[key] = value
                continue
            
            # Handle different data types
            if isinstance(value, str):
                # Keep strings as is, but handle empty strings
                cleaned[key] = value if value.strip() else ''
            elif isinstance(value, (int, float)):
                # Convert to float and handle NaN/inf
                float_val = float(value)
                if np.isnan(float_val) or np.isinf(float_val):
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = float_val
            elif value is None:
                # Convert None to appropriate default
                if 'has_' in key or key.endswith('_specificity') or key in ['is_form_element', 'is_interactive', 'is_clickable']:
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = ''
            else:
                cleaned[key] = value
        
        return cleaned
    
    def prepare_training_dataframe(self, feature_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert feature data to a clean pandas DataFrame for training.
        
        Args:
            feature_data: List of feature dictionaries
            
        Returns:
            Cleaned pandas DataFrame ready for ML
        """
        try:
            # Create DataFrame
            df = pd.DataFrame(feature_data)
            
            if df.empty:
                self.logger.warning("No data to process")
                return df
            
            self.logger.debug(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
            
            # Separate numeric and categorical columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Remove raw data column from processing
            if 'raw_element_info' in categorical_columns:
                categorical_columns.remove('raw_element_info')
            
            self.logger.debug(f"Numeric columns: {len(numeric_columns)}, Categorical columns: {len(categorical_columns)}")
            
            # Clean numeric columns
            for col in numeric_columns:
                # Replace NaN, inf with 0.0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                df[col] = df[col].replace([np.inf, -np.inf], 0.0)
            
            # Clean categorical columns
            for col in categorical_columns:
                # Replace NaN, None, empty strings with meaningful defaults
                df[col] = df[col].fillna('')
                df[col] = df[col].astype(str)
                df[col] = df[col].replace(['nan', 'None', 'null'], '')
            
            self.logger.debug("Data cleaning completed")
            
            # Generate target variable for training (selector quality score)
            if 'best_selector' in df.columns:
                df['target_quality'] = self._calculate_selector_quality(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing training dataframe: {e}")
            return pd.DataFrame()
    
    def _calculate_selector_quality(self, df: pd.DataFrame) -> pd.Series:
        """Modified quality scoring to account for intent-word matches."""
        try:
            scores = []
            for _, row in df.iterrows():
                score = 0.0
                if row.get('has_id', 0) == 1:
                    score += 3.0
                if row.get('has_data_testid', 0) == 1:
                    score += 2.5
                if row.get('has_name', 0) == 1:
                    score += 2.0
                if row.get('has_aria_label', 0) == 1:
                    score += 1.5
                if row.get('has_classes', 0) == 1:
                    score += 1.0
                if row.get('has_text', 0) == 1:
                    score += 0.5

                score += row.get('id_specificity', 0) * 0.3
                score += row.get('class_specificity', 0) * 0.2
                score += row.get('attribute_specificity', 0) * 0.3

                # ✅ NEW: Big boost if element's visible text matches test intent
                if row.get('intent_word_in_text', 0) == 1:
                    score += 3.0

                scores.append(score)
            return pd.Series(scores)
        except Exception as e:
            self.logger.error(f"Error calculating selector quality: {e}")
            return pd.Series([0.0] * len(df))
    
    def analyze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze feature distribution and quality for debugging.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Analysis results dictionary
        """
        try:
            analysis = {
                'total_samples': len(df),
                'total_features': len(df.columns),
                'missing_values': {},
                'feature_stats': {},
                'element_type_distribution': {},
                'quality_score_stats': {}
            }
            
            # Missing values analysis
            for col in df.columns:
                if col != 'raw_element_info':
                    missing_count = df[col].isnull().sum()
                    if isinstance(df[col].iloc[0], str):
                        empty_count = (df[col] == '').sum()
                        analysis['missing_values'][col] = {
                            'null_count': int(missing_count),
                            'empty_count': int(empty_count),
                            'total_missing': int(missing_count + empty_count)
                        }
                    else:
                        analysis['missing_values'][col] = {'null_count': int(missing_count)}
            
            # Feature statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df.columns:
                    analysis['feature_stats'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'median': float(df[col].median())
                    }
            
            # Element type distribution
            if 'element_type' in df.columns:
                type_dist = df['element_type'].value_counts().to_dict()
                analysis['element_type_distribution'] = {k: int(v) for k, v in type_dist.items()}
            
            # Quality score statistics
            if 'target_quality' in df.columns:
                quality_col = df['target_quality']
                analysis['quality_score_stats'] = {
                    'mean': float(quality_col.mean()),
                    'std': float(quality_col.std()),
                    'min': float(quality_col.min()),
                    'max': float(quality_col.max()),
                    'distribution': quality_col.describe().to_dict()
                }
            
            self.logger.debug("Feature analysis completed")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing features: {e}")
            return {}