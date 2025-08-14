"""
Web scraping module using Playwright and BeautifulSoup.
Handles page scraping, content extraction, and robust error handling.
"""

import logging
import asyncio
from typing import Optional, Dict, List, Any
from bs4 import BeautifulSoup


class WebScraper:
    """Web scraper using Playwright for dynamic content and BeautifulSoup for parsing."""
    
    def __init__(self, headless: bool = True, timeout: int = 30000):
        """Initialize the web scraper with configuration."""
        self.headless = headless
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._browser = None
        self._context = None
    
    async def _init_browser(self):
        """Initialize Playwright browser if not already initialized."""
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(headless=self.headless)
                self._context = await self._browser.new_context()
                self.logger.debug("Browser initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize browser: {e}")
                raise
    
    async def _cleanup_browser(self):
        """Clean up browser resources."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if hasattr(self, '_playwright'):
                await self._playwright.stop()
            self.logger.debug("Browser cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during browser cleanup: {e}")
    
    def scrape_page(self, url: str) -> Optional[str]:
        """
        Scrape a web page and return HTML content.
        
        Args:
            url: URL to scrape
            
        Returns:
            HTML content as string, or None if scraping failed
        """
        return asyncio.run(self._scrape_page_async(url))
    
    async def _scrape_page_async(self, url: str) -> Optional[str]:
        """Async implementation of page scraping."""
        try:
            await self._init_browser()
            
            page = await self._context.new_page()
            
            self.logger.debug(f"Navigating to: {url}")
            
            # Navigate to the page with timeout
            await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            
            # Wait for any dynamic content to load
            await page.wait_for_timeout(2000)
            
            # Get the page content
            html_content = await page.content()
            
            await page.close()
            
            self.logger.debug(f"Successfully scraped {len(html_content)} characters from {url}")
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None
        finally:
            # Clean up for single page scraping
            if self._browser:
                await self._cleanup_browser()
                self._browser = None
                self._context = None
    
    def parse_html(self, html_content: str) -> Optional[BeautifulSoup]:
        """
        Parse HTML content using BeautifulSoup.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            BeautifulSoup object or None if parsing failed
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            self.logger.debug(f"HTML parsed successfully, found {len(soup.find_all())} elements")
            return soup
        except Exception as e:
            self.logger.error(f"Error parsing HTML: {e}")
            return None
    
    def extract_elements_info(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract detailed information about HTML elements for feature extraction.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of element information dictionaries
        """
        elements_info = []
        
        try:
            # Find all interactive elements
            interactive_tags = ['button', 'input', 'select', 'textarea', 'a', 'form']
            elements = soup.find_all(interactive_tags)
            
            for element in elements:
                try:
                    info = self._extract_element_info(element)
                    if info:
                        elements_info.append(info)
                except Exception as e:
                    self.logger.warning(f"Error extracting info for element: {e}")
                    continue
            
            self.logger.debug(f"Extracted info for {len(elements_info)} elements")
            
        except Exception as e:
            self.logger.error(f"Error extracting elements info: {e}")
        
        return elements_info
    
    def _extract_element_info(self, element) -> Optional[Dict[str, Any]]:
        """Extract information from a single HTML element."""
        try:
            # Basic element information
            info = {
                'tag': element.name or '',
                'id': element.get('id', ''),
                'classes': ' '.join(element.get('class', [])),
                'text': (element.get_text(strip=True) or '')[:200],  # Limit text length
                'type': element.get('type', ''),
                'name': element.get('name', ''),
                'placeholder': element.get('placeholder', ''),
                'value': element.get('value', ''),
                'href': element.get('href', ''),
                'role': element.get('role', ''),
                'aria_label': element.get('aria-label', ''),
                'title': element.get('title', ''),
                'data_testid': element.get('data-testid', ''),
            }
            
            # Parent information
            parent = element.parent
            if parent:
                info['parent_tag'] = parent.name or ''
                info['parent_classes'] = ' '.join(parent.get('class', []))
                info['parent_id'] = parent.get('id', '')
            else:
                info['parent_tag'] = ''
                info['parent_classes'] = ''
                info['parent_id'] = ''
            
            # Generate potential selectors
            info['selectors'] = self._generate_selectors(element)
            
            return info
            
        except Exception as e:
            self.logger.warning(f"Error extracting element info: {e}")
            return None
    
    def _generate_selectors(self, element) -> List[str]:
        """Generate various CSS selectors for an element."""
        selectors = []
        
        try:
            # ID selector
            if element.get('id'):
                selectors.append(f"#{element['id']}")
            
            # Class selector
            classes = element.get('class', [])
            if classes:
                class_selector = '.' + '.'.join(classes)
                selectors.append(class_selector)
            
            # Attribute selectors
            for attr in ['name', 'type', 'data-testid']:
                if element.get(attr):
                    selectors.append(f"[{attr}='{element[attr]}']")
            
            # Tag selector
            if element.name:
                selectors.append(element.name)
            
            # Text-based selector (for links and buttons)
            text = element.get_text(strip=True)
            if text and len(text) <= 50:
                if element.name == 'a':
                    selectors.append(f"a:contains('{text}')")
                elif element.name == 'button':
                    selectors.append(f"button:contains('{text}')")
            
        except Exception as e:
            self.logger.warning(f"Error generating selectors: {e}")
        
        return selectors
    
    def batch_scrape(self, urls: List[str], max_concurrent: int = 3) -> Dict[str, Optional[str]]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping URLs to their HTML content (or None if failed)
        """
        return asyncio.run(self._batch_scrape_async(urls, max_concurrent))
    
    async def _batch_scrape_async(self, urls: List[str], max_concurrent: int) -> Dict[str, Optional[str]]:
        """Async implementation of batch scraping."""
        await self._init_browser()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_single(url: str) -> tuple[str, Optional[str]]:
            async with semaphore:
                try:
                    page = await self._context.new_page()
                    await page.goto(url, timeout=self.timeout, wait_until='networkidle')
                    await page.wait_for_timeout(1000)
                    content = await page.content()
                    await page.close()
                    return url, content
                except Exception as e:
                    self.logger.error(f"Error scraping {url}: {e}")
                    return url, None
        
        try:
            # Execute all scraping tasks
            tasks = [scrape_single(url) for url in urls]
            results = await asyncio.gather(*tasks)
            
            # Convert to dictionary
            result_dict = dict(results)
            
            successful = sum(1 for content in result_dict.values() if content)
            self.logger.info(f"Batch scraping complete: {successful}/{len(urls)} successful")
            
            return result_dict
            
        finally:
            await self._cleanup_browser()