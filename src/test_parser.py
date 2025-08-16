import logging
import re
from typing import List, Dict, Any
from pathlib import Path

class PlaywrightTestParser:
    """Parse Playwright test files (Python or JS/TS) to extract locator information and context."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common Playwright locator patterns for JS/TS + PY
        self.locator_patterns = [
            r'page\.locator\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get[_]?by[_]?role\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get[_]?by[_]?text\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get[_]?by[_]?label\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get[_]?by[_]?placeholder\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get[_]?by[_]?alt[_]?text\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get[_]?by[_]?title\([\'"]([^\'"]+)[\'"]\)',
            r'page\.get[_]?by[_]?test[_]?id\([\'"]([^\'"]+)[\'"]\)',
            r'page\.query[_]?selector\([\'"]([^\'"]+)[\'"]\)',
            r'page\.wait[_]?for[_]?selector\([\'"]([^\'"]+)[\'"]\)'
        ]

    def parse_test_file(self, test_file_path: str) -> Dict[str, Any]:
        """Parse a Playwright test file and extract locator information."""
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
        locators = []
        for pattern in self.locator_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                selector = match.group(1)
                locators.append({
                    'selector': selector,
                    'method': self._determine_locator_method(match.group(0)),
                    'line_number': content[:match.start()].count('\n') + 1,
                    'context': self._get_line_context(content, match.start(), match.end()),
                    'full_match': match.group(0)
                })
        return locators

    def _determine_locator_method(self, match_text: str) -> str:
        lowered = match_text.lower()
        if 'get_by_role' in lowered or 'getbyrole' in lowered:
            return 'get_by_role'
        if 'get_by_text' in lowered or 'getbytext' in lowered:
            return 'get_by_text'
        if 'get_by_label' in lowered or 'getbylabel' in lowered:
            return 'get_by_label'
        if 'get_by_placeholder' in lowered or 'getbyplaceholder' in lowered:
            return 'get_by_placeholder'
        if 'get_by_test_id' in lowered or 'getbytestid' in lowered:
            return 'get_by_test_id'
        if 'query_selector' in lowered or 'queryselector' in lowered:
            return 'query_selector'
        if 'wait_for_selector' in lowered or 'waitforselector' in lowered:
            return 'wait_for_selector'
        return 'locator'

    def _get_line_context(self, content: str, start: int, end: int, context_lines: int = 3) -> Dict[str, Any]:
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
        """Support both Python and Playwright JS `test()` syntax"""
        test_functions = []
        # Match Python def tests
        for match in re.finditer(r'def\s+(test_\w+)\s*\([^)]*\):', content):
            test_functions.append({'name': match.group(1), 'line_number': content[:match.start()].count('\n')+1})
        # Match JS/TS Playwright test('...', async ()=>{})
        for match in re.finditer(r'test\s*\(\s*[\'"](.+)[\'"]\s*,', content):
            test_functions.append({'name': match.group(1), 'line_number': content[:match.start()].count('\n')+1})
        return test_functions

    def _extract_imports(self, content: str) -> List[str]:
        imports = re.findall(r'^(?:from\s+\S+\s+)?import\s+.+$', content, re.MULTILINE)
        imports += re.findall(r'^\s*const\s+\{\s*test\s*,\s*expect\s*\}\s*=\s*require\([\'"]@playwright/test[\'"]\)', content, re.MULTILINE)
        return imports

    def _extract_page_urls(self, content: str) -> List[str]:
        urls = re.findall(r'page\.goto\([\'"]([^\'"]+)[\'"]\)', content)
        return urls

    def _extract_assertions(self, content: str) -> List[Dict[str, Any]]:
        assertions = []
        patterns = [
            r'expect\([^)]+\)\.toBeVisible\(\)',
            r'expect\([^)]+\)\.toHaveText\([^)]+\)',
            r'expect\([^)]+\)\.toBeEnabled\(\)',
            r'expect\([^)]+\)\.toBeChecked\(\)',
            r'assert\s+[^#\n]+'
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                assertions.append({
                    'text': match.group(0),
                    'line_number': content[:match.start()].count('\n') + 1,
                    'type': self._determine_assertion_type(match.group(0))
                })
        return assertions

    def _determine_assertion_type(self, assertion_text: str) -> str:
        l = assertion_text.lower()
        if 'tobevisible' in l:
            return 'visibility'
        if 'tohavetext' in l:
            return 'text_content'
        if 'tobeenabled' in l:
            return 'enabled_state'
        if 'tobechecked' in l:
            return 'checked_state'
        return 'generic'

    def find_failed_locators(self, test_file_path: str, error_log: str) -> List[Dict[str, Any]]:
        """
        Identify failed locators from test error logs.
        """
        try:
            import re, os

            # Strip ANSI escape sequences (color codes)
            ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
            clean_error_log = ansi_escape.sub('', error_log)

            test_info = self.parse_test_file(test_file_path)
            failed_locators = []

            # Comprehensive Playwright failure patterns
            error_patterns = [
                # Main Playwright JS/TS patterns
                r"Locator:\s*locator\(['\"]([^'\"]+)['\"]\)",
                r"waiting for locator\(['\"]([^'\"]+)['\"]\)",
                r"Timed out \d+ms waiting for.*?locator\(['\"]([^'\"]+)['\"]\)",

                # Common assertion timeouts
                r"Timed out \d+ms waiting for expect\(locator\)\.toBeVisible\(\).*?Locator:\s*locator\(['\"]([^'\"]+)['\"]\)",
                r"expect\(.*?\)\.toBeVisible\(\).*?locator\(['\"]([^'\"]+)['\"]\)",
                r"expect\(.*?\)\.toBeEnabled\(\).*?locator\(['\"]([^'\"]+)['\"]\)",
                r"expect\(.*?\)\.toBeChecked\(\).*?locator\(['\"]([^'\"]+)['\"]\)",
                r"expect\(.*?\)\.toHaveText\(.*?\).*?locator\(['\"]([^'\"]+)['\"]\)",

                # Element not found messages
                r"<element\(s\) not found>.*?locator\(['\"]([^'\"]+)['\"]\)",
                r"Element not found.*?locator\(['\"]([^'\"]+)['\"]\)",
                r"No element found for selector\s+['\"]([^'\"]+)['\"]",
                r"Element not found:\s+['\"]([^'\"]+)['\"]",

                # Direct locator references
                r"Locator\(['\"]([^'\"]+)['\"]\).*?not found",

                # Selector timeouts
                r"waiting for selector\s+['\"]([^'\"]+)['\"].*?timeout",
                r"Timed out \d+ms waiting for selector\s+['\"]([^'\"]+)['\"]",

                # Interaction failures
                r"click\(\) failed.*?locator\(['\"]([^'\"]+)['\"]\)",
                r"fill\(\) failed.*?locator\(['\"]([^'\"]+)['\"]\)",
                r"hover\(\) failed.*?locator\(['\"]([^'\"]+)['\"]\)",

                # Test and generic timeouts
                r"Test timeout of \d+ms exceeded.*?locator\(['\"]([^'\"]+)['\"]\)",
                r"TimeoutError.*?['\"]([^'\"]+)['\"]",
                r"Error:.*?locator\(['\"]([^'\"]+)['\"]\)",

                # Page navigation
                r"page\.goto\(\) timeout.*?['\"]([^'\"]+)['\"]",
                r"Navigation timeout.*?['\"]([^'\"]+)['\"]",

                # Frame/context errors
                r"frame detached.*?locator\(['\"]([^'\"]+)['\"]\)",
                r"execution context destroyed.*?locator\(['\"]([^'\"]+)['\"]\)",

                # Python Playwright compatibility
                r"playwright\._impl\._api_types\.TimeoutError.*?['\"]([^'\"]+)['\"]",
                r"playwright\._impl\._api_types\.Error.*?['\"]([^'\"]+)['\"]"
            ]

            for pattern in error_patterns:
                for match in re.finditer(pattern, clean_error_log, re.IGNORECASE | re.DOTALL):
                    failed_selector = match.group(1)

                    # Match against parsed locators in the test file
                    matching_locator = next(
                        (loc for loc in test_info.get('locators', []) if loc['selector'] == failed_selector),
                        None
                    )

                    if matching_locator:
                        failed_info = {
                            'selector': failed_selector,
                            'error_message': match.group(0),
                            'locator_info': matching_locator,
                            'test_file': test_file_path,
                            'page_urls': test_info.get('page_urls', [])
                        }

                        # Fallback page URL from env or default local
                        if not failed_info['page_urls']:
                            default_url = os.getenv('BASE_URL')
                            failed_info['page_urls'] = [default_url] if default_url else [
                                "http://127.0.0.1:5500/html/sample_train.html"
                            ]

                        # Avoid duplicates
                        if not any(fl['selector'] == failed_selector and fl['test_file'] == test_file_path
                                for fl in failed_locators):
                            failed_locators.append(failed_info)

            self.logger.info(f"Found {len(failed_locators)} failed locators in {test_file_path}")
            return failed_locators

        except Exception as e:
            self.logger.error(f"Error finding failed locators: {e}")
            return []




    def batch_parse_tests(self, test_directory: str) -> Dict[str, Any]:
        """Now also parse .spec.js and .spec.ts files"""
        try:
            test_dir = Path(test_directory)
            all_tests = {}
            test_files = []
            # Python tests
            test_files += list(test_dir.rglob('test_*.py')) + list(test_dir.rglob('*_test.py'))
            # Playwright JS/TS tests
            test_files += list(test_dir.rglob('*.spec.js')) + list(test_dir.rglob('*.spec.ts'))
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
