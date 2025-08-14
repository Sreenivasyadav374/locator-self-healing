"""
Locator fixing module that combines ML predictions with test context to fix failed locators.
Provides intelligent suggestions and automatic test file updates.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from test_parser import PlaywrightTestParser
from models import MLPredictor
from scraper import WebScraper
from features import FeatureExtractor


class LocatorFixer:
    """Fix failed Playwright test locators using ML predictions and test context."""
    
    def __init__(self, model_path: str):
        """
        Initialize the locator fixer.
        
        Args:
            model_path: Path to the trained ML model
        """
        self.logger = logging.getLogger(__name__)
        self.test_parser = PlaywrightTestParser()
        self.predictor = MLPredictor()
        self.scraper = WebScraper()
        self.feature_extractor = FeatureExtractor()
        
        # Load the trained model
        try:
            self.predictor.load_model(model_path)
            self.logger.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def fix_failed_locator(self, test_file_path: str, failed_selector: str, 
                          page_url: Optional[str] = None, error_log: Optional[str] = None) -> Dict[str, Any]:
        """
        Fix a single failed locator using ML predictions and test context.
        
        Args:
            test_file_path: Path to the test file containing the failed locator
            failed_selector: The selector that failed
            page_url: URL of the page being tested (optional)
            error_log: Error log from test execution (optional)
            
        Returns:
            Dictionary with fix suggestions and confidence scores
        """
        try:
            self.logger.info(f"Fixing failed locator: {failed_selector} in {test_file_path}")
            
            # Parse the test file to get context
            test_info = self.test_parser.parse_test_file(test_file_path)
            
            # Find the specific failed locator in the test
            failed_locator_info = None
            for locator in test_info.get('locators', []):
                if locator['selector'] == failed_selector:
                    failed_locator_info = locator
                    break
            
            if not failed_locator_info:
                self.logger.warning(f"Failed locator {failed_selector} not found in test file")
                return {'success': False, 'error': 'Locator not found in test file'}
            
            # Determine the page URL if not provided
            if not page_url:
                page_urls = test_info.get('page_urls', [])
                if page_urls:
                    page_url = page_urls[0]  # Use the first URL found
                else:
                    self.logger.warning("No page URL found, cannot scrape for ML predictions")
                    return {'success': False, 'error': 'No page URL available'}
            
            # Scrape the page to get current elements
            html_content = self.scraper.scrape_page(page_url)
            if not html_content:
                self.logger.error(f"Failed to scrape page: {page_url}")
                return {'success': False, 'error': 'Failed to scrape page'}
            
            # Extract features from the page
            features = self.feature_extractor.extract_page_features(html_content, page_url)
            if not features:
                self.logger.error("No features extracted from page")
                return {'success': False, 'error': 'No features extracted'}
            
            # Get ML predictions
            predictions = self.predictor.predict_locators(features, current_selector=failed_selector)
            
            # Analyze test context to improve suggestions
            context_analysis = self._analyze_test_context(failed_locator_info, test_info)
            
            # Generate fix suggestions
            fix_suggestions = self._generate_fix_suggestions(
                failed_selector, predictions, context_analysis, failed_locator_info
            )
            
            result = {
                'success': True,
                'failed_selector': failed_selector,
                'page_url': page_url,
                'suggestions': fix_suggestions,
                'context_analysis': context_analysis,
                'test_file': test_file_path,
                'locator_method': failed_locator_info.get('method', 'locator')
            }
            
            self.logger.info(f"Generated {len(fix_suggestions)} fix suggestions for {failed_selector}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fixing locator: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_test_context(self, failed_locator_info: Dict[str, Any], 
                             test_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the test context to understand the intent of the failed locator."""
        context = failed_locator_info.get('context', {})
        
        analysis = {
            'intent': self._determine_locator_intent(context),
            'surrounding_actions': self._extract_surrounding_actions(context),
            'element_type_hint': self._guess_element_type(failed_locator_info),
            'interaction_type': self._determine_interaction_type(context)
        }
        
        return analysis
    
    def _determine_locator_intent(self, context: Dict[str, Any]) -> str:
        """Determine what the locator is trying to find based on context."""
        current_line = context.get('current', '').lower()
        before_lines = ' '.join(context.get('before', [])).lower()
        after_lines = ' '.join(context.get('after', [])).lower()
        
        all_context = f"{before_lines} {current_line} {after_lines}"
        
        # Common patterns
        if any(word in all_context for word in ['login', 'sign in', 'signin']):
            return 'login_form'
        elif any(word in all_context for word in ['submit', 'send', 'save']):
            return 'submit_button'
        elif any(word in all_context for word in ['search', 'find']):
            return 'search_element'
        elif any(word in all_context for word in ['menu', 'nav', 'navigation']):
            return 'navigation'
        elif any(word in all_context for word in ['input', 'field', 'text']):
            return 'input_field'
        elif any(word in all_context for word in ['button', 'click']):
            return 'button'
        else:
            return 'generic'
    
    def _extract_surrounding_actions(self, context: Dict[str, Any]) -> List[str]:
        """Extract actions happening around the failed locator."""
        actions = []
        
        all_lines = (context.get('before', []) + 
                    [context.get('current', '')] + 
                    context.get('after', []))
        
        for line in all_lines:
            line = line.strip().lower()
            if '.click()' in line:
                actions.append('click')
            elif '.fill(' in line:
                actions.append('fill')
            elif '.type(' in line:
                actions.append('type')
            elif '.select_option(' in line:
                actions.append('select')
            elif '.check()' in line:
                actions.append('check')
            elif '.hover()' in line:
                actions.append('hover')
            elif 'expect(' in line:
                actions.append('assertion')
        
        return actions
    
    def _guess_element_type(self, locator_info: Dict[str, Any]) -> str:
        """Guess the element type based on locator method and selector."""
        method = locator_info.get('method', '')
        selector = locator_info.get('selector', '').lower()
        
        if method == 'get_by_role':
            return selector  # Role is usually the element type
        elif method == 'get_by_test_id':
            if 'btn' in selector or 'button' in selector:
                return 'button'
            elif 'input' in selector or 'field' in selector:
                return 'input'
            else:
                return 'unknown'
        elif method in ['get_by_text', 'get_by_label']:
            return 'text_element'
        elif method == 'get_by_placeholder':
            return 'input'
        else:
            # Analyze CSS selector
            if selector.startswith('#'):
                return 'id_element'
            elif 'button' in selector:
                return 'button'
            elif 'input' in selector:
                return 'input'
            else:
                return 'unknown'
    
    def _determine_interaction_type(self, context: Dict[str, Any]) -> str:
        """Determine what type of interaction is expected."""
        current_line = context.get('current', '').lower()
        after_lines = ' '.join(context.get('after', [])).lower()
        
        if '.click()' in current_line or '.click()' in after_lines:
            return 'click'
        elif '.fill(' in current_line or '.fill(' in after_lines:
            return 'fill'
        elif '.type(' in current_line or '.type(' in after_lines:
            return 'type'
        elif 'expect(' in after_lines:
            return 'assertion'
        else:
            return 'unknown'
    
    def _generate_fix_suggestions(self, failed_selector: str, ml_predictions: List[Dict[str, Any]], 
                                 context_analysis: Dict[str, Any], 
                                 locator_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized fix suggestions combining ML and context analysis."""
        suggestions = []
        
        # Get the original locator method
        original_method = locator_info.get('method', 'locator')
        
        # Process ML predictions
        for i, prediction in enumerate(ml_predictions[:5]):  # Top 5 predictions
            suggested_selector = prediction.get('locator', '')
            confidence = prediction.get('confidence', 0.0)
            
            # Determine the best Playwright method for this selector
            suggested_method = self._determine_best_playwright_method(
                suggested_selector, context_analysis
            )
            
            # Generate the full Playwright locator code
            playwright_code = self._generate_playwright_code(suggested_method, suggested_selector)
            
            # Calculate priority based on ML confidence and context match
            context_bonus = self._calculate_context_bonus(
                suggested_selector, context_analysis, suggested_method
            )
            
            priority_score = confidence + context_bonus
            
            suggestion = {
                'rank': i + 1,
                'selector': suggested_selector,
                'method': suggested_method,
                'playwright_code': playwright_code,
                'confidence': confidence,
                'context_bonus': context_bonus,
                'priority_score': priority_score,
                'reasoning': prediction.get('reasoning', ''),
                'improvement_score': prediction.get('improvement_score', 0.0)
            }
            
            suggestions.append(suggestion)
        
        # Add context-based suggestions if ML didn't cover them
        context_suggestions = self._generate_context_based_suggestions(
            failed_selector, context_analysis, original_method
        )
        
        for context_suggestion in context_suggestions:
            # Check if we already have a similar suggestion
            if not any(s['selector'] == context_suggestion['selector'] for s in suggestions):
                suggestions.append(context_suggestion)
        
        # Sort by priority score
        suggestions.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        # Add rank after sorting
        for i, suggestion in enumerate(suggestions):
            suggestion['final_rank'] = i + 1
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def _determine_best_playwright_method(self, selector: str, context_analysis: Dict[str, Any]) -> str:
        """Determine the best Playwright method for a given selector."""
        intent = context_analysis.get('intent', '')
        element_type = context_analysis.get('element_type_hint', '')
        
        # ID selectors
        if selector.startswith('#'):
            return 'locator'
        
        # data-testid selectors
        elif selector.startswith('[data-testid'):
            test_id = re.search(r'data-testid=[\'"]([^\'"]+)[\'"]', selector)
            if test_id:
                return 'get_by_test_id'
            return 'locator'
        
        # Text-based selectors
        elif 'contains' in selector.lower() or selector.startswith('text='):
            return 'get_by_text'
        
        # Role-based suggestions
        elif element_type in ['button', 'link', 'textbox', 'checkbox']:
            return 'get_by_role'
        
        # Default to locator for CSS selectors
        else:
            return 'locator'
    
    def _generate_playwright_code(self, method: str, selector: str) -> str:
        """Generate the actual Playwright code for a method and selector."""
        if method == 'get_by_test_id':
            # Extract test ID from selector
            test_id_match = re.search(r'data-testid=[\'"]([^\'"]+)[\'"]', selector)
            if test_id_match:
                test_id = test_id_match.group(1)
                return f'page.get_by_test_id("{test_id}")'
            else:
                return f'page.locator("{selector}")'
        
        elif method == 'get_by_text':
            # Extract text from selector
            if 'contains' in selector:
                text_match = re.search(r'contains\([\'"]([^\'"]+)[\'"]\)', selector)
                if text_match:
                    text = text_match.group(1)
                    return f'page.get_by_text("{text}")'
            elif selector.startswith('text='):
                text = selector[5:]  # Remove 'text=' prefix
                return f'page.get_by_text("{text}")'
            
            return f'page.locator("{selector}")'
        
        elif method == 'get_by_role':
            # This would need more context to determine the role
            return f'page.get_by_role("button")  # Adjust role as needed'
        
        elif method == 'get_by_label':
            return f'page.get_by_label("{selector}")'
        
        elif method == 'get_by_placeholder':
            return f'page.get_by_placeholder("{selector}")'
        
        else:  # Default to locator
            return f'page.locator("{selector}")'
    
    def _calculate_context_bonus(self, selector: str, context_analysis: Dict[str, Any], 
                                method: str) -> float:
        """Calculate bonus score based on how well the suggestion matches context."""
        bonus = 0.0
        
        intent = context_analysis.get('intent', '')
        element_type = context_analysis.get('element_type_hint', '')
        interaction_type = context_analysis.get('interaction_type', '')
        
        # Bonus for matching intent
        if intent == 'login_form' and ('login' in selector.lower() or 'signin' in selector.lower()):
            bonus += 0.3
        elif intent == 'submit_button' and ('submit' in selector.lower() or 'send' in selector.lower()):
            bonus += 0.3
        elif intent == 'search_element' and 'search' in selector.lower():
            bonus += 0.3
        
        # Bonus for matching element type
        if element_type == 'button' and method == 'get_by_role':
            bonus += 0.2
        elif element_type == 'input' and ('input' in selector.lower() or method == 'get_by_placeholder'):
            bonus += 0.2
        
        # Bonus for high-quality selectors
        if selector.startswith('#'):  # ID selector
            bonus += 0.4
        elif 'data-testid' in selector:  # Test ID
            bonus += 0.35
        elif method == 'get_by_test_id':
            bonus += 0.35
        
        return bonus
    
    def _generate_context_based_suggestions(self, failed_selector: str, 
                                          context_analysis: Dict[str, Any], 
                                          original_method: str) -> List[Dict[str, Any]]:
        """Generate additional suggestions based purely on test context."""
        suggestions = []
        
        intent = context_analysis.get('intent', '')
        element_type = context_analysis.get('element_type_hint', '')
        
        # Common fallback patterns based on intent
        if intent == 'login_form':
            fallback_selectors = [
                '[data-testid="login-button"]',
                'button:has-text("Login")',
                'input[type="submit"]',
                '.login-btn',
                '#login'
            ]
        elif intent == 'submit_button':
            fallback_selectors = [
                '[data-testid="submit-button"]',
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Submit")',
                '.submit-btn'
            ]
        elif intent == 'search_element':
            fallback_selectors = [
                '[data-testid="search-input"]',
                'input[placeholder*="search" i]',
                '.search-input',
                '#search',
                'input[type="search"]'
            ]
        else:
            fallback_selectors = []
        
        # Generate suggestions for fallback selectors
        for i, selector in enumerate(fallback_selectors):
            method = self._determine_best_playwright_method(selector, context_analysis)
            playwright_code = self._generate_playwright_code(method, selector)
            
            suggestion = {
                'rank': i + 100,  # Lower priority than ML suggestions
                'selector': selector,
                'method': method,
                'playwright_code': playwright_code,
                'confidence': 0.5,  # Medium confidence for context-based
                'context_bonus': 0.3,
                'priority_score': 0.8,
                'reasoning': f'Context-based suggestion for {intent}',
                'improvement_score': 0.0,
                'source': 'context_analysis'
            }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def apply_fix_to_test(self, test_file_path: str, failed_selector: str, 
                         new_selector: str, new_method: str = None, 
                         backup: bool = True) -> Dict[str, Any]:
        """
        Apply a fix to the test file by replacing the failed locator.
        
        Args:
            test_file_path: Path to the test file
            failed_selector: The selector to replace
            new_selector: The new selector to use
            new_method: The new Playwright method to use (optional)
            backup: Whether to create a backup of the original file
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Read the original file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Create backup if requested
            if backup:
                backup_path = f"{test_file_path}.backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                self.logger.info(f"Created backup at {backup_path}")
            
            # Generate the new Playwright code
            if new_method:
                new_code = self._generate_playwright_code(new_method, new_selector)
            else:
                new_code = f'page.locator("{new_selector}")'
            
            # Find and replace the failed locator
            updated_content = self._replace_locator_in_content(
                original_content, failed_selector, new_code
            )
            
            if updated_content == original_content:
                return {
                    'success': False,
                    'error': 'No changes made - locator not found or already correct'
                }
            
            # Write the updated content
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            self.logger.info(f"Applied fix to {test_file_path}: {failed_selector} -> {new_selector}")
            
            return {
                'success': True,
                'original_selector': failed_selector,
                'new_selector': new_selector,
                'new_method': new_method,
                'backup_created': backup,
                'backup_path': f"{test_file_path}.backup" if backup else None
            }
            
        except Exception as e:
            self.logger.error(f"Error applying fix to test file: {e}")
            return {'success': False, 'error': str(e)}
    
    def _replace_locator_in_content(self, content: str, old_selector: str, new_code: str) -> str:
        """Replace locator in file content."""
        # Escape special regex characters in the old selector
        escaped_selector = re.escape(old_selector)
        
        # Pattern to match various Playwright locator methods
        patterns = [
            rf'page\.locator\([\'\"]{escaped_selector}[\'\"]\)',
            rf'page\.query_selector\([\'\"]{escaped_selector}[\'\"]\)',
            rf'page\.wait_for_selector\([\'\"]{escaped_selector}[\'\"]\)',
        ]
        
        updated_content = content
        
        for pattern in patterns:
            if re.search(pattern, updated_content):
                updated_content = re.sub(pattern, new_code, updated_content)
                break
        
        return updated_content
    
    def batch_fix_failed_locators(self, test_directory: str, error_log_file: str) -> Dict[str, Any]:
        """
        Fix multiple failed locators from a test run.
        
        Args:
            test_directory: Directory containing test files
            error_log_file: File containing error logs from test execution
            
        Returns:
            Dictionary with batch fix results
        """
        try:
            # Read error log
            with open(error_log_file, 'r', encoding='utf-8') as f:
                error_log = f.read()
            
            # Parse all test files
            all_tests = self.test_parser.batch_parse_tests(test_directory)
            
            # Find all failed locators
            all_failed_locators = []
            for test_file, test_info in all_tests.items():
                failed_locators = self.test_parser.find_failed_locators(test_file, error_log)
                all_failed_locators.extend(failed_locators)
            
            self.logger.info(f"Found {len(all_failed_locators)} failed locators to fix")
            
            # Fix each failed locator
            fix_results = []
            for failed_locator in all_failed_locators:
                try:
                    fix_result = self.fix_failed_locator(
                        failed_locator['test_file'],
                        failed_locator['selector'],
                        failed_locator.get('page_urls', [None])[0] if failed_locator.get('page_urls') else None,
                        error_log
                    )
                    
                    fix_result['original_error'] = failed_locator.get('error_message', '')
                    fix_results.append(fix_result)
                    
                except Exception as e:
                    self.logger.error(f"Error fixing locator {failed_locator['selector']}: {e}")
                    fix_results.append({
                        'success': False,
                        'failed_selector': failed_locator['selector'],
                        'error': str(e)
                    })
            
            # Summary
            successful_fixes = [r for r in fix_results if r.get('success', False)]
            
            summary = {
                'total_failed_locators': len(all_failed_locators),
                'successful_fixes': len(successful_fixes),
                'failed_fixes': len(fix_results) - len(successful_fixes),
                'fix_results': fix_results
            }
            
            self.logger.info(f"Batch fix complete: {len(successful_fixes)}/{len(all_failed_locators)} successful")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in batch fix: {e}")
            return {'success': False, 'error': str(e)}