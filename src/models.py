"""
Machine learning models for locator prediction and fixing.
Uses scikit-learn and TensorFlow for training and prediction.
"""

import logging
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from textdistance import levenshtein


class MLPredictor:
    """Machine learning predictor for web locator quality and suggestions."""
    
    def __init__(self):
        """Initialize the ML predictor."""
        self.logger = logging.getLogger(__name__)
        self.quality_model = None
        self.type_classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_trained = False
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the ML models on the provided dataset.
        
        Args:
            df: Training DataFrame with features and targets
            test_size: Fraction of data to use for testing
            
        Returns:
            Training results and metrics
        """
        try:
            self.logger.info("Starting model training...")
            
            if df.empty:
                raise ValueError("Training DataFrame is empty")
            
            # Prepare features and targets
            X, y_quality, y_type = self._prepare_training_data(df)
            
            if X.empty:
                raise ValueError("No valid features for training")
            
            # Split data
            X_train, X_test, y_quality_train, y_quality_test, y_type_train, y_type_test = train_test_split(
                X, y_quality, y_type, test_size=test_size, random_state=42, stratify=y_type
            )
            
            self.logger.debug(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train quality regression model
            self.quality_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            self.quality_model.fit(X_train_scaled, y_quality_train)
            
            # Train type classification model
            self.type_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            self.type_classifier.fit(X_train_scaled, y_type_train)
            
            # Evaluate models
            quality_pred = self.quality_model.predict(X_test_scaled)
            type_pred = self.type_classifier.predict(X_test_scaled)
            
            quality_mse = mean_squared_error(y_quality_test, quality_pred)
            type_accuracy = accuracy_score(y_type_test, type_pred)
            
            self.feature_columns = X.columns.tolist()
            self.is_trained = True
            
            results = {
                'quality_mse': float(quality_mse),
                'type_accuracy': float(type_accuracy),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(self.feature_columns),
                'accuracy': type_accuracy  # For backward compatibility
            }
            
            self.logger.info(f"Training completed - Quality MSE: {quality_mse:.3f}, Type Accuracy: {type_accuracy:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data from raw DataFrame."""
        try:
            # Select numeric features for training
            numeric_features = [
                'has_text', 'text_length', 'text_word_count',
                'has_id', 'has_classes', 'has_name', 'has_placeholder',
                'has_value', 'has_href', 'has_role', 'has_aria_label',
                'has_title', 'has_data_testid', 'has_parent',
                'parent_has_id', 'parent_has_classes', 'selector_count',
                'id_specificity', 'class_specificity', 'attribute_specificity',
                'is_form_element', 'is_interactive', 'is_clickable'
            ]
            
            # Filter available features
            available_features = [col for col in numeric_features if col in df.columns]
            
            if not available_features:
                raise ValueError("No numeric features available for training")
            
            X = df[available_features].copy()
            
            # Ensure all features are numeric
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            
            # Prepare targets
            if 'target_quality' in df.columns:
                y_quality = df['target_quality']
            else:
                # Generate quality scores if not available
                y_quality = self._generate_quality_scores(df)
            
            if 'element_type' in df.columns:
                y_type = df['element_type'].fillna('other')
                # Encode labels
                y_type = pd.Series(self.label_encoder.fit_transform(y_type))
            else:
                # Default type classification
                y_type = pd.Series(['other'] * len(df))
                y_type = pd.Series(self.label_encoder.fit_transform(y_type))
            
            self.logger.debug(f"Prepared training data: {len(X)} samples, {len(available_features)} features")
            
            return X, y_quality, y_type
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
    
    def _generate_quality_scores(self, df: pd.DataFrame) -> pd.Series:
        """Generate quality scores when not available in data."""
        scores = []
        
        for _, row in df.iterrows():
            score = 0.0
            
            # Simple heuristic scoring
            if row.get('has_id', 0) == 1:
                score += 3.0
            if row.get('has_data_testid', 0) == 1:
                score += 2.5
            if row.get('has_name', 0) == 1:
                score += 2.0
            if row.get('has_classes', 0) == 1:
                score += 1.0
            if row.get('has_text', 0) == 1:
                score += 0.5
            
            scores.append(score)
        
        return pd.Series(scores)
    
    def predict_locators(self, features: List[Dict[str, Any]], current_selector: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict best locators for given features.
        
        Args:
            features: List of feature dictionaries
            current_selector: Current selector to improve (optional)
            
        Returns:
            List of locator predictions with confidence scores
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained. Call train_model() first.")
            
            predictions = []
            
            for feature_dict in features:
                try:
                    pred = self._predict_single_locator(feature_dict, current_selector)
                    if pred:
                        predictions.append(pred)
                except Exception as e:
                    self.logger.warning(f"Error predicting for single feature set: {e}")
                    continue
            
            # Sort by confidence and return top predictions
            predictions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            self.logger.debug(f"Generated {len(predictions)} locator predictions")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return []
    
    def _predict_single_locator(self, feature_dict: Dict[str, Any], current_selector: Optional[str]) -> Optional[Dict[str, Any]]:
        """Predict locator for a single element."""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(feature_dict)
            
            if feature_vector is None:
                return None
            
            # Scale features
            feature_scaled = self.scaler.transform([feature_vector])
            
            # Predict quality and type
            quality_score = self.quality_model.predict(feature_scaled)[0]
            type_probs = self.type_classifier.predict_proba(feature_scaled)[0]
            predicted_type = self.label_encoder.inverse_transform([self.type_classifier.predict(feature_scaled)[0]])[0]
            
            # Get raw element info
            element_info = feature_dict.get('raw_element_info', {})
            
            # Generate selector suggestions
            selector_suggestions = self._generate_selector_suggestions(element_info, quality_score)
            
            if not selector_suggestions:
                return None
            
            # Calculate confidence based on quality score and model certainty
            max_type_prob = max(type_probs)
            confidence = (quality_score / 10.0 + max_type_prob) / 2.0  # Normalize to 0-1
            confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
            
            # If current selector provided, compare improvement
            improvement_score = 0.0
            if current_selector:
                improvement_score = self._calculate_improvement_score(
                    current_selector, selector_suggestions[0], element_info
                )
            
            return {
                'locator': selector_suggestions[0],
                'alternative_locators': selector_suggestions[1:3],
                'confidence': confidence,
                'quality_score': quality_score,
                'element_type': predicted_type,
                'improvement_score': improvement_score,
                'reasoning': self._generate_reasoning(element_info, quality_score)
            }
            
        except Exception as e:
            self.logger.warning(f"Error predicting single locator: {e}")
            return None
    
    def _prepare_feature_vector(self, feature_dict: Dict[str, Any]) -> Optional[List[float]]:
        """Prepare feature vector for prediction."""
        try:
            if not self.feature_columns:
                raise ValueError("Feature columns not defined. Train model first.")
            
            vector = []
            
            for col in self.feature_columns:
                value = feature_dict.get(col, 0.0)
                
                # Convert to float and handle edge cases
                if isinstance(value, str):
                    value = 1.0 if value.strip() else 0.0
                elif value is None:
                    value = 0.0
                else:
                    value = float(value)
                
                # Handle NaN and inf
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                
                vector.append(value)
            
            return vector
            
        except Exception as e:
            self.logger.warning(f"Error preparing feature vector: {e}")
            return None
    
    def _generate_selector_suggestions(self, element_info: Dict[str, Any], quality_score: float) -> List[str]:
        """Generate selector suggestions based on element info and quality score."""
        suggestions = []
        
        try:
            # Priority-based selector generation
            
            # 1. ID selector (highest priority)
            element_id = element_info.get('id', '').strip()
            if element_id:
                suggestions.append(f"#{element_id}")
            
            # 2. data-testid (very high priority)
            data_testid = element_info.get('data_testid', '').strip()
            if data_testid:
                suggestions.append(f"[data-testid='{data_testid}']")
            
            # 3. Name attribute
            name = element_info.get('name', '').strip()
            if name:
                suggestions.append(f"[name='{name}']")
            
            # 4. Aria-label
            aria_label = element_info.get('aria_label', '').strip()
            if aria_label:
                suggestions.append(f"[aria-label='{aria_label}']")
            
            # 5. Class selectors
            classes = element_info.get('classes', '').strip()
            if classes:
                class_list = classes.split()
                if len(class_list) == 1:
                    suggestions.append(f".{class_list[0]}")
                elif len(class_list) > 1:
                    # Use most specific class combination
                    suggestions.append('.' + '.'.join(class_list[:2]))
            
            # 6. Type-based selectors for inputs
            tag = element_info.get('tag', '').strip()
            input_type = element_info.get('type', '').strip()
            if tag == 'input' and input_type:
                suggestions.append(f"input[type='{input_type}']")
            
            # 7. Text-based selectors (lower priority)
            text = element_info.get('text', '').strip()
            if text and len(text) <= 50:
                if tag == 'button':
                    suggestions.append(f"button:contains('{text}')")
                elif tag == 'a':
                    suggestions.append(f"a:contains('{text}')")
            
            # 8. Tag selector as fallback
            if tag:
                suggestions.append(tag)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_suggestions = []
            for suggestion in suggestions:
                if suggestion not in seen:
                    seen.add(suggestion)
                    unique_suggestions.append(suggestion)
            
            return unique_suggestions[:5]  # Return top 5
            
        except Exception as e:
            self.logger.warning(f"Error generating selector suggestions: {e}")
            return []
    
    def _calculate_improvement_score(self, current_selector: str, suggested_selector: str, element_info: Dict[str, Any]) -> float:
        """Calculate how much the suggested selector improves over current."""
        try:
            # Simple heuristic for improvement scoring
            current_score = self._score_selector(current_selector, element_info)
            suggested_score = self._score_selector(suggested_selector, element_info)
            
            improvement = suggested_score - current_score
            return max(0.0, improvement / 10.0)  # Normalize
            
        except Exception as e:
            self.logger.warning(f"Error calculating improvement score: {e}")
            return 0.0
    
    def _score_selector(self, selector: str, element_info: Dict[str, Any]) -> float:
        """Score a selector based on reliability and maintainability."""
        if not selector:
            return 0.0
        
        score = 0.0
        
        # ID selectors are most reliable
        if selector.startswith('#'):
            score += 8.0
        
        # data-testid selectors are very good
        elif 'data-testid' in selector:
            score += 7.0
        
        # Name attribute is good
        elif 'name=' in selector:
            score += 6.0
        
        # Aria-label is good for accessibility
        elif 'aria-label' in selector:
            score += 5.5
        
        # Class selectors are medium
        elif selector.startswith('.'):
            score += 4.0
        
        # Type selectors are medium
        elif 'type=' in selector:
            score += 3.5
        
        # Text-based selectors are less reliable
        elif 'contains' in selector:
            score += 2.0
        
        # Tag selectors are least specific
        else:
            score += 1.0
        
        return score
    
    def _generate_reasoning(self, element_info: Dict[str, Any], quality_score: float) -> str:
        """Generate human-readable reasoning for the prediction."""
        reasons = []
        
        if element_info.get('id'):
            reasons.append("Element has ID - most reliable locator")
        elif element_info.get('data_testid'):
            reasons.append("Element has data-testid - very reliable for testing")
        elif element_info.get('name'):
            reasons.append("Element has name attribute - good for form elements")
        elif element_info.get('classes'):
            reasons.append("Element has classes - moderately reliable")
        else:
            reasons.append("Limited reliable attributes - using best available option")
        
        if quality_score > 5.0:
            reasons.append("High quality locator available")
        elif quality_score > 2.0:
            reasons.append("Medium quality locator")
        else:
            reasons.append("Limited locator options - consider adding test IDs")
        
        return ". ".join(reasons)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        try:
            model_data = {
                'quality_model': self.quality_model,
                'type_classifier': self.type_classifier,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.quality_model = model_data['quality_model']
            self.type_classifier = model_data['type_classifier']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise