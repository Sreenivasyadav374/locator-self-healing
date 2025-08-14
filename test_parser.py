"""
Test parsing module for analyzing Playwright test files and extracting locator information.
Handles parsing of test code to identify failed locators and their context.
"""

import logging
import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class PlaywrightTestParser:
    """Parse Playwright test files to extract locator information and context."""
    
    def __init__(self):
        """Initialize the test parser."""
        self.logger = logging.getLogger(__name__)
        
        # Common Playwright locator patterns
        self.locator_patterns = [
            r'page\.locator\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get_by_role\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get_by_text\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get_by_label\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get_by_placeholder\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get_by_alt_text\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get_by_title\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get_by_test_id\([\'"]([^\'"]+)[\'"]\)',
            r'page\.query_selector\([\'"]([^\'"]+)[\'"]\)',
            r'page\.wait_for_selector\([\'"]([^\'"]+)[\'"]\)',
            r'\.click\(\)\s*#.*locator:\s*([^\n]+)',
            r'\.fill\([\'"][^\'"]*[\'"]\)\s*#.*locator:\s*([^\n]+)',
        ]
    
    def parse_test_file(self, test_file_path: str) -> Dict[str, Any]:
        """
        Parse a Playwright test file and extract locator information.
        
        Args:
            test_file_path: Path to the test file
            
        Returns:
            Dictionary containing parsed test information
        """
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            test_info = {
                'file_path': test_file_path,
                'locators': self._extract_locators(content),
                'test_functions': self._extract_test_functions(content),
                'imports': self._extract_imports(content),
                'page_urls': self._extract_page_urls(content),
                'assertions': self._extract_assertions(content)
            }
            
            self.logger.debug(f"Parsed test file {test_file_path}: found {len(test_info['locators'])} locators")
            
            return test_info
            
        except Exception as e:
            self.logger.error(f"Error parsing test file {test_file_path}: {e}")
            return {}
    
    def _extract_locators(self, content: str) -> List[Dict[str, Any]]:
        """Extract all locators from test content."""
        locators = []
        
        for pattern in self.locator_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                locator_info = {
                    'selector': match.group(1),
                    'method': self._determine_locator_method(match.group(0)),
                    'line_number': content[:match.start()].count('\n') + 1,
                    'context': self._get_line_context(content, match.start(), match.end()),
                    'full_match': match.group(0)
                }
                locators.append(locator_info)
        
        return locators
    
    def _determine_locator_method(self, match_text: str) -> str:
        """Determine the Playwright locator method used."""
        if 'get_by_role' in match_text:
            return 'get_by_role'
        elif 'get_by_text' in match_text:
            return 'get_by_text'
        elif 'get_by_label' in match_text:
            return 'get_by_label'
        elif 'get_by_placeholder' in match_text:
            return 'get_by_placeholder'
        elif 'get_by_test_id' in match_text:
            return 'get_by_test_id'
        elif 'query_selector' in match_text:
            return 'query_selector'
        elif 'wait_for_selector' in match_text:
            return 'wait_for_selector'
        else:
            return 'locator'
    
    def _get_line_context(self, content: str, start: int, end: int, context_lines: int = 3) -> Dict[str, Any]:
        """Get context around a locator match."""
        lines = content.split('\n')
        match_line = content[:start].count('\n')
        
        context_start = max(0, match_line - context_lines)
        context_end = min(len(lines), match_line + context_lines + 1)
        
        return {
            'before': lines[context_start:match_line],
            'current': lines[match_line] if match_line < len(lines) else '',
            'after': lines[match_line + 1:context_end],
            'line_number': match_line + 1
        }
    
    def _extract_test_functions(self, content: str) -> List[Dict[str, Any]]:
        """Extract test function information."""
        test_functions = []
        
        # Pattern for test functions
        test_pattern = r'def\s+(test_\w+)\s*\([^)]*\):'
        matches = re.finditer(test_pattern, content)
        
        for match in matches:
            func_info = {
                'name': match.group(1),
                'line_number': content[:match.start()].count('\n') + 1,
                'full_match': match.group(0)
            }
            test_functions.append(func_info)
        
        return test_functions
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements."""
        import_pattern = r'^(?:from\s+\S+\s+)?import\s+.+$'
        imports = re.findall(import_pattern, content, re.MULTILINE)
        return imports
    
    def _extract_page_urls(self, content: str) -> List[str]:
        """Extract page URLs from goto statements."""
        url_pattern = r'page\.goto\([\'"]([^\'"]+)[\'"]\)'
        urls = re.findall(url_pattern, content)
        return urls
    
    def _extract_assertions(self, content: str) -> List[Dict[str, Any]]:
        """Extract assertion statements."""
        assertions = []
        
        assertion_patterns = [
            r'expect\([^)]+\)\.to_be_visible\(\)',
            r'expect\([^)]+\)\.to_have_text\([^)]+\)',
            r'expect\([^)]+\)\.to_be_enabled\(\)',
            r'expect\([^)]+\)\.to_be_checked\(\)',
            r'assert\s+[^#\n]+',
        ]
        
        for pattern in assertion_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                assertion_info = {
                    'text': match.group(0),
                    'line_number': content[:match.start()].count('\n') + 1,
                    'type': self._determine_assertion_type(match.group(0))
                }
                assertions.append(assertion_info)
        
        return assertions
    
    def _determine_assertion_type(self, assertion_text: str) -> str:
        """Determine the type of assertion."""
        if 'to_be_visible' in assertion_text:
            return 'visibility'
        elif 'to_have_text' in assertion_text:
            return 'text_content'
        elif 'to_be_enabled' in assertion_text:
            return 'enabled_state'
        elif 'to_be_checked' in assertion_text:
            return 'checked_state'
        else:
            return 'generic'
    
    def find_failed_locators(self, test_file_path: str, error_log: str) -> List[Dict[str, Any]]:
        """
        Identify failed locators from test error logs.
        
        Args:
            test_file_path: Path to the test file
            error_log: Error log content from test execution
            
        Returns:
            List of failed locator information
        """
        try:
            test_info = self.parse_test_file(test_file_path)
            failed_locators = []
            
            # Common error patterns for failed locators
            error_patterns = [
                r'Locator\([\'"]([^\'"]+)[\'"]\).*not found',
                r'waiting for selector\s+[\'"]([^\'"]+)[\'"].*timeout',
                r'No element found for selector\s+[\'"]([^\'"]+)[\'"]',
                r'Element not found:\s+[\'"]([^\'"]+)[\'"]',
                r'TimeoutError.*[\'"]([^\'"]+)[\'"]',
            ]
            
            for pattern in error_patterns:
                matches = re.finditer(pattern, error_log, re.IGNORECASE)
                
                for match in matches:
                    failed_selector = match.group(1)
                    
                    # Find matching locator in parsed test
                    matching_locator = None
                    for locator in test_info.get('locators', []):
                        if locator['selector'] == failed_selector:
                            matching_locator = locator
                            break
                    
                    if matching_locator:
                        failed_info = {
                            'selector': failed_selector,
                            'error_message': match.group(0),
                            'locator_info': matching_locator,
                            'test_file': test_file_path,
                            'page_urls': test_info.get('page_urls', [])
                        }
                        failed_locators.append(failed_info)
            
            self.logger.info(f"Found {len(failed_locators)} failed locators in {test_file_path}")
            
            return failed_locators
            
        except Exception as e:
            self.logger.error(f"Error finding failed locators: {e}")
            return []
    
    def batch_parse_tests(self, test_directory: str) -> Dict[str, Any]:
        """
        Parse all test files in a directory.
        
        Args:
            test_directory: Directory containing test files
            
        Returns:
            Dictionary with all parsed test information
        """
        try:
            test_dir = Path(test_directory)
            all_tests = {}
            
            # Find all Python test files
            test_files = list(test_dir.rglob('test_*.py')) + list(test_dir.rglob('*_test.py'))
            
            for test_file in test_files:
                try:
                    test_info = self.parse_test_file(str(test_file))
                    if test_info:
                        all_tests[str(test_file)] = test_info
                except Exception as e:
                    self.logger.warning(f"Error parsing {test_file}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(all_tests)} test files from {test_directory}")
            
            return all_tests
            
        except Exception as e:
            self.logger.error(f"Error batch parsing tests: {e}")
            return {}