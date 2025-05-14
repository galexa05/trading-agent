 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Yahoo Finance News Scraper for AAPL Stock

This script uses Selenium to scrape news articles related to AAPL stock from Yahoo Finance.
It scrolls through the page to load all available articles and extracts relevant information.
"""

import time
import json
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class YahooFinanceNewsScraper:
    """Class to scrape news articles from Yahoo Finance for a specific stock."""
    
    def __init__(self, ticker="AAPL", headless=False, debug_mode=True, page_load_timeout=30):
        """
        Initialize the scraper.
        
        Args:
            ticker (str): Stock ticker symbol (default: "AAPL")
            headless (bool): Whether to run browser in headless mode (default: False)
            debug_mode (bool): Enable additional debugging output and screenshots (default: True)
            page_load_timeout (int): Maximum time to wait for page to load in seconds (default: 30)
        """
        self.ticker = ticker
        self.url = f"https://finance.yahoo.com/quote/{ticker}/news/"
        self.articles = []
        self.debug_mode = debug_mode
        self.consent_handled = False  # Flag to track if we've already handled the consent dialog
        
        # Create data directory if it doesn't exist
        import os
        os.makedirs("../data", exist_ok=True)
        
        # Set up Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")  # Using newer headless mode
        
        # Performance and stability settings
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-dev-shm-usage")  # Helps with stability
        chrome_options.add_argument("--disable-web-security")  # Helps with CORS issues
        chrome_options.add_argument("--disable-features=IsolateOrigins,site-per-process")  # Helps with iframes
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent detection
        
        # Add user agent to appear more like a regular browser
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Initialize Chrome experimental options - important for bypassing detection
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Add Chrome preferences - helps with cookie handling
        prefs = {
            "profile.default_content_setting_values.cookies": 1,  # Allow cookies
            "profile.default_content_settings.popups": 0,        # Block popups
            "profile.default_content_setting_values.notifications": 2  # Block notifications
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Initialize the Chrome driver with modified options
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Set page load timeout - helps with infinitely loading pages
        self.driver.set_page_load_timeout(page_load_timeout)
        
        # Set script timeout - helps with slow JavaScript execution
        self.driver.set_script_timeout(page_load_timeout)
        
    def _handle_yahoo_consent_dialog(self):
        """
        Handle the Yahoo cookie consent dialog based on the provided screenshot.
        This method attempts multiple approaches to accept cookies and proceed to the content.
        
        Returns:
            bool: True if successful or dialog not present, False if failed to handle dialog
        """
        try:
            print("Looking for Yahoo consent dialog...")
            
            # Take initial screenshot to see what's actually on the page
            if self.debug_mode:
                self.driver.save_screenshot("../data/initial_page.png")
                print("Saved initial page screenshot")
                
            # Wait longer for full page load - including iframes and overlays
            time.sleep(5)
            
            # The Yahoo consent dialog doesn't always use the standard role='dialog' attribute
            # We'll check for multiple possible indicators of the consent dialog
            consent_indicators = [
                "div[role='dialog']",
                ".consent-wizard", 
                "#consent-page",
                ".consent-form",
                ".cookie-consent",
                ".privacy-consent",
                "div.consent",
                # From the screenshot - looking for specific visible elements
                "div:contains('cookies')",
                "div:contains('Accept all')",
                "div:contains('Reject all')",
                "div.yahoo-consent",  # Class might be used
                # Any dialog div containing the Yahoo logo
                "div.yahoo-logo",
                "div > img[alt='Yahoo']",
                "div > img[src*='yahoo']"
            ]
            
            dialog_present = False
            
            # Check all indicators
            for indicator in consent_indicators:
                try:
                    # Special handling for contains selectors which are jQuery but not CSS
                    if ":contains(" in indicator:
                        # Replace with more complex XPath that achieves similar functionality
                        text = indicator.split(":")[1].replace("contains('", "").replace("')", "")
                        xpath = f"//div[contains(text(), '{text}')]"
                        elem = self.driver.find_element(By.XPATH, xpath)
                    else:
                        elem = self.driver.find_element(By.CSS_SELECTOR, indicator)
                        
                    if elem and elem.is_displayed():
                        print(f"Found consent dialog indicator: {indicator}")
                        dialog_present = True
                        break
                except:
                    # Element not found with this selector
                    pass
                    
            # Final fallback - check for the text "cookies" on the page
            if not dialog_present:
                try:
                    body_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
                    if "cookie" in body_text and ("accept" in body_text or "agree" in body_text):
                        print("Found cookie-related text on page - dialog likely present")
                        dialog_present = True
                except:
                    pass
                    
            if not dialog_present:
                print("No consent dialog detected - proceeding")
                return True
            
            print("Yahoo consent dialog is likely present")
            
            # Take screenshot if in debug mode
            if self.debug_mode:
                self.driver.save_screenshot("../data/consent_dialog.png")
                print("Saved screenshot of consent dialog for debugging")
            
            # 1. First try: Click the "Accept all" button directly based on the screenshot
            yahoo_accept_selectors = [
                # Exact match from screenshot
                "//button[text()='Accept all']",  # Exact button text from screenshot
                "//a[text()='Accept all']",      # In case it's an anchor
                # Class name approach
                ".accept-all",
                "button.accept-all",
                "a.accept-all",
                # Partial text match approach
                "//button[contains(text(), 'Accept')]",
                "//a[contains(text(), 'Accept')]",
                # First button in dialog
                "//div[@role='dialog']//button[1]",
                # Trying to target the specific button from screenshot
                "//div[contains(@class, 'consent')]//button[1]",
                "//div[contains(@class, 'privacy')]//button[1]",
                # Additional selectors that might work
                "[data-action='accept']",
                "[data-click='accept']",
                "[data-test='accept-all']",
                ".consent-wizard .primary",
                # From your screenshot - the specific first button in the dialog
                "//form[contains(@class, 'consent')]//button[1]"
            ]
            
            # Add more specific selectors from the screenshot
            # The button text is "Accept all" in a white button
            dialog_specific_selectors = [
                "//div[@id='consent-page']//button[1]",
                "//div[contains(@class, 'consent-wizard')]//button[1]",
                # The Yahoo dialog has buttons at the bottom
                "//div[contains(., 'Yahoo')]//button[1]",
                "//div[contains(., 'privacy policy')]//button[1]"
            ]
            
            yahoo_accept_selectors.extend(dialog_specific_selectors)
            
            # Try each selector
            for selector in yahoo_accept_selectors:
                try:
                    print(f"Trying selector: {selector}")
                    if selector.startswith("//"):
                        button = WebDriverWait(self.driver, 3).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        button = WebDriverWait(self.driver, 3).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    
                    # Print button text for debugging
                    button_text = button.text.strip()
                    print(f"Found button with text: '{button_text}'")
                    
                    # Click the button
                    button.click()
                    print(f"Clicked button using selector: {selector}")
                    time.sleep(3)  # Wait for action to take effect
                    return True
                except Exception as e:
                    print(f"Selector failed: {selector}, Error: {str(e)}")
            
            # 2. Check if dialog is in an iframe
            print("Checking for dialog in iframes...")
            frames = self.driver.find_elements(By.TAG_NAME, "iframe")
            main_window = self.driver.current_window_handle
            frame_found = False
            
            if frames:
                print(f"Found {len(frames)} iframes on page, checking each...")
                
                # Save current window handle
                original_window = self.driver.current_window_handle
                
                for i, frame in enumerate(frames):
                    try:
                        print(f"Switching to iframe {i+1} of {len(frames)}")
                        self.driver.switch_to.frame(frame)
                        
                        # Take screenshot of iframe content
                        if self.debug_mode:
                            self.driver.save_screenshot(f"../data/iframe_{i+1}.png")
                        
                        # Look for accept button in the iframe
                        for selector in yahoo_accept_selectors:
                            try:
                                if selector.startswith("//"):
                                    button = self.driver.find_element(By.XPATH, selector)
                                else:
                                    button = self.driver.find_element(By.CSS_SELECTOR, selector)
                                
                                if button and button.is_displayed():
                                    print(f"Found button in iframe {i+1} using selector: {selector}")
                                    button.click()
                                    print(f"Clicked consent button in iframe {i+1}")
                                    time.sleep(3)
                                    frame_found = True
                                    break
                            except:
                                pass
                        
                        # Return to main document
                        self.driver.switch_to.default_content()
                    except Exception as e:
                        print(f"Error handling iframe {i+1}: {str(e)}")
                        self.driver.switch_to.default_content()
            
            if frame_found:
                print("Successfully handled dialog in iframe")
                return True
            
            # 3. Enhanced JavaScript approach for finding and clicking the Accept button
            print("Trying enhanced JavaScript approach...")
            js_result = self.driver.execute_script("""
            // More aggressive approach to find and click 'Accept all' button
            function findAndClickAcceptButton() {
                // Try exact text match first
                var exactMatches = Array.from(document.querySelectorAll('button, a, input, span')).filter(el => 
                    el.innerText && el.innerText.trim() === 'Accept all' && window.getComputedStyle(el).display !== 'none');
                
                if (exactMatches.length > 0) {
                    console.log('Found exact match for Accept all button');  
                    exactMatches[0].click();
                    return true;
                }
                
                // Try contains match
                var containsMatches = Array.from(document.querySelectorAll('button, a, input, span')).filter(el => 
                    el.innerText && el.innerText.toLowerCase().includes('accept') && window.getComputedStyle(el).display !== 'none');
                
                if (containsMatches.length > 0) {
                    console.log('Found partial match for Accept button');
                    containsMatches[0].click();
                    return true;
                }
                
                // Check for first button in any visible dialog
                var dialogs = Array.from(document.querySelectorAll('div[role="dialog"], .consent-wizard, #consent-page, .consent'));
                
                for (var i = 0; i < dialogs.length; i++) {
                    var dialog = dialogs[i];
                    if (window.getComputedStyle(dialog).display !== 'none') {
                        var buttons = dialog.querySelectorAll('button, a.button, input[type="button"]');
                        if (buttons.length > 0) {
                            console.log('Clicking first button in consent dialog');
                            buttons[0].click();
                            return true;
                        }
                    }
                }
                
                // Last resort: look for any visible button in the document that might be relevant
                var allButtons = document.querySelectorAll('button, a.button');
                for (var i = 0; i < allButtons.length; i++) {
                    var btn = allButtons[i];
                    var text = btn.innerText.toLowerCase();
                    if ((text.includes('accept') || text.includes('agree') || text.includes('continue')) && 
                        window.getComputedStyle(btn).display !== 'none') {
                        console.log('Found potential consent button: ' + btn.innerText);
                        btn.click();
                        return true;
                    }
                }
                
                return false;
            }
            
            return findAndClickAcceptButton();
            """)
            
            if js_result:
                print("Successfully clicked button using enhanced JavaScript")
                time.sleep(3)  # Wait for action to take effect
                return True
                
            # 4. Try to click directly at the location where the Accept button usually appears
            # Based on the screenshot, the Accept button is often at the bottom left of dialogs
            try:
                print("Attempting to click at the likely location of the Accept button...")
                # Get window size
                window_size = self.driver.get_window_size()
                # Calculate approximate positions (based on typical cookie banner layouts)
                # Bottom left button position
                action = webdriver.ActionChains(self.driver)
                # Click bottom left position (first white button in screenshot)
                action.move_to_element_with_offset(self.driver.find_element(By.TAG_NAME, 'body'), 250, window_size['height'] - 100)
                action.click()
                action.perform()
                print("Clicked at likely Accept button position")
                time.sleep(3)
            except Exception as e:
                print(f"Error clicking at position: {str(e)}")
            
            # 5. Final attempt: Try to bypass the dialog by manipulating cookies directly
            print("Attempting to bypass consent by setting cookies...")
            try:
                # Try multiple cookie combinations that might work
                common_consent_cookies = [
                    {"name": "cookieConsent", "value": "true"},
                    {"name": "EuConsent", "value": "true"},
                    {"name": "yahoo_consent", "value": "true"},
                    {"name": "consent", "value": "accept"},
                    {"name": "gdpr", "value": "accepted"},
                    {"name": "privacy_policy_accepted", "value": "true"}
                ]
                
                for cookie in common_consent_cookies:
                    try:
                        self.driver.add_cookie(cookie)
                        print(f"Added cookie: {cookie['name']}")
                    except:
                        pass
                        
                # Refresh the page
                self.driver.refresh()
                time.sleep(3)
            except Exception as e:
                print(f"Error setting cookies: {str(e)}")
            
            # Take another screenshot in debug mode
            if self.debug_mode:
                self.driver.save_screenshot("../data/after_all_attempts.png")
                print("Saved final screenshot after all attempts")
                
            # Last resort - try one more page refresh
            try:
                self.driver.refresh()
                time.sleep(5)  # Wait longer
                print("Performed final page refresh")
            except:
                pass
                
            # At this point we've tried everything but continue anyway
            print("Proceeding despite potential dialog issues...")
            return False
        except Exception as e:
            print(f"Error in consent dialog handling: {str(e)}")
            return False
    
    def _force_stop_page_loading(self):
        """
        Force stop the page loading which can help when the browser shows infinite loading.
        """
        try:
            print("Attempting to stop any ongoing page loading...")
            self.driver.execute_script("window.stop();")  # JavaScript command to stop page loading
            print("Page loading stopped")
        except Exception as e:
            print(f"Error stopping page load: {str(e)}")
    
    def _find_news_articles_directly(self):
        """
        Directly search the page source for news articles when standard methods fail.
        
        Returns:
            bool: True if articles were found, False otherwise
        """
        print("Attempting to find news articles directly in page source...")
        try:
            # Try multiple potential CSS selectors for article elements
            article_selectors = [
                "ul[data-testid='news-list'] li",         # Standard selector
                ".js-stream-content li",                 # Possible alternate class
                ".NewsStream li",                       # Another possible class
                "div[data-test='news-stream'] li",       # Yet another possibility
                "[data-test='news-item']",              # Individual news items
                "article",                              # Generic article tag
                "div.news-item",                        # Generic class name
                ".Pos\\(r\\)",                         # Yahoo sometimes uses this pattern
                ".Pos\\(r\\) > div"                     # Children of the above
            ]
            
            # Try each selector
            articles_found = False
            for selector in article_selectors:
                try:
                    articles = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if articles and len(articles) > 0:
                        print(f"Found {len(articles)} articles with selector: {selector}")
                        articles_found = True
                        break
                except Exception as e:
                    continue
            
            # If we still found nothing, try looking for HTML patterns in the source
            if not articles_found:
                page_source = self.driver.page_source.lower()
                if "news-list" in page_source or ">news<" in page_source or "news stream" in page_source:
                    print("Found news-related content in page source")
                    articles_found = True
                    
            return articles_found
        except Exception as e:
            print(f"Error searching for articles directly: {str(e)}")
            return False
    
    def scroll_and_scrape(self, max_articles=100, scroll_pause_time=1.0, force_scrape=True):
        """
        Scroll through the page to load articles and scrape them.
        
        Args:
            max_articles (int): Maximum number of articles to scrape (default: 100)
            scroll_pause_time (float): Time to pause between scrolls in seconds (default: 1.0)
            force_scrape (bool): Whether to force scraping even if article elements aren't found (default: True)
        
        Returns:
            list: A list of dictionaries containing article data
        """
        try:
            # Load the page with error handling for timeout
            print(f"Loading Yahoo Finance news for {self.ticker}...")
            try:
                self.driver.get(self.url)
                print("Page loaded successfully")
            except Exception as e:
                print(f"Error during initial page load: {str(e)}")
                print("Stopping any ongoing loading...")
                self._force_stop_page_loading()
            
            # Take initial screenshot to see what loaded
            if self.debug_mode:
                self.driver.save_screenshot("../data/initial_load.png")
            
            # Only handle the cookie consent dialog once at the beginning
            if not self.consent_handled:
                print("Handling cookie consent dialog (only once at the beginning)...")
                self._handle_yahoo_consent_dialog()
                self.consent_handled = True
            else:
                print("Consent dialog already handled, skipping...")
            
            # More aggressive handling for potential persistent consent dialogs
            max_attempts = 4  # Increased attempts
            attempt = 0
            content_found = False
            
            while attempt < max_attempts and not content_found:
                attempt += 1
                print(f"Content loading attempt {attempt}/{max_attempts}")
                
                # Stop any ongoing loading that might be preventing content access
                if attempt > 1:
                    self._force_stop_page_loading()
                
                # Try to find news content
                try:
                    # First try standard approach with explicit wait
                    print("Waiting for news content to load...")
                    WebDriverWait(self.driver, 8).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "ul[data-testid='news-list'] li"))
                    )
                    print("News content successfully loaded with standard selector")
                    content_found = True
                except TimeoutException:
                    # If standard approach fails, try direct search in page source
                    print("Standard selector not found. Trying alternative approaches...")
                    content_found = self._find_news_articles_directly()
                    
                    if content_found:
                        print("Found news content using alternative methods")
                    else:
                        print(f"WARNING: News content not found on attempt {attempt}")
                        
                        if self.debug_mode:
                            self.driver.save_screenshot(f"../data/attempt_{attempt}_issue.png")
                            print(f"Saved screenshot of attempt {attempt}")
                        
                        # If dialog might still be blocking, try additional approaches
                        if attempt < max_attempts:
                            print("Dialog might still be present, trying alternate approaches...")
                            
                            # Try the most direct JavaScript approach to click Accept
                            self.driver.execute_script("""
                            // Direct approach to find and click any Accept button
                            Array.from(document.querySelectorAll('button, a')).forEach(el => {
                                if (el.innerText && 
                                    (el.innerText.toLowerCase().includes('accept') || 
                                     el.innerText.toLowerCase().includes('agree')) && 
                                    window.getComputedStyle(el).display !== 'none') {
                                    console.log("Clicking:", el.innerText);
                                    el.click();
                                }
                            });
                            """)
                            
                            # Try clicking at common positions for Accept buttons
                            try:
                                action = webdriver.ActionChains(self.driver)
                                # Try bottom left (common location)
                                action.move_by_offset(250, 500).click().perform()
                                time.sleep(0.5)
                                # Try bottom right
                                action.move_by_offset(500, 500).click().perform()
                                time.sleep(0.5)
                                # Try center
                                action.move_by_offset(400, 300).click().perform()
                            except:
                                pass
                                
                            # Reload the page with updated settings
                            print("Attempting new page load...")
                            try:
                                # First clear cookies and cache
                                self.driver.delete_all_cookies()
                                # Try a new URL approach - sometimes helps bypass consent
                                modified_url = self.url
                                if "?" not in modified_url:
                                    modified_url += "?bypass=true"
                                self.driver.get(modified_url)
                                time.sleep(3)
                            except:
                                print("Error during reload, stopping page load")
                                self._force_stop_page_loading()
            
            # Even if we couldn't find the content using selectors, we can still try to scrape
            # the page as it might have partial content or we might be able to extract something
            if not content_found and force_scrape:
                print("WARNING: Standard content detection failed")
                print("Proceeding with scraping attempt regardless")
                
                # Take final screenshot
                if self.debug_mode:
                    self.driver.save_screenshot("../data/final_page_state.png")
                    # Save final page state for debugging
                    with open("../data/final_page_source.html", "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    print("Saved final page state for debugging")
            
            # Define a function to extract articles using multiple selectors
            def try_extract_articles():
                selectors = [
                    "ul[data-testid='news-list'] li",      # Standard selector
                    ".js-stream-content li",              # Alternate class
                    ".NewsStream li",                    # Another class
                    "[data-test='news-item']",           # Individual items
                    "article",                           # Generic article
                    ".news-item",                        # Common class
                    "div.Pos\\(r\\)",                    # Yahoo pattern
                ]
                
                for selector in selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements and len(elements) > 0:
                            print(f"Found {len(elements)} elements with selector: {selector}")
                            return elements
                    except Exception as e:
                        continue
                        
                # If all selectors fail, extract potential article elements by HTML structure
                try:
                    # Last resort - look for divs with links that might be news items
                    print("Attempting fallback extraction by HTML structure")
                    # Look for divs containing links and text paragraphs (common news article structure)
                    news_divs = self.driver.find_elements(By.XPATH, 
                        "//div[.//a and (.//p or .//span)][not(ancestor::footer)][not(ancestor::header)]")
                    if news_divs and len(news_divs) > 0:
                        print(f"Found {len(news_divs)} potential news elements with fallback method")
                        return news_divs
                except:
                    pass
                    
                return []
            
            # Keep track of article count to avoid infinite scrolling
            last_count = 0
            scroll_attempts = 0
            max_scroll_attempts = 25  # Limit to avoid truly infinite scrolling
            consecutive_no_new = 0  # Counter for consecutive attempts with no new articles
            
            while len(self.articles) < max_articles and scroll_attempts < max_scroll_attempts:
                scroll_attempts += 1
                print(f"Scroll attempt {scroll_attempts}/{max_scroll_attempts}")
                
                # Scroll down by one window height to load more content
                try:
                    current_position = self.driver.execute_script("return window.pageYOffset;")
                    window_height = self.driver.execute_script("return window.innerHeight;")
                    self.driver.execute_script(f"window.scrollTo(0, {current_position + window_height});")
                except Exception as e:
                    print(f"Error during scrolling: {str(e)}")
                
                # Use a longer wait every few scrolls to ensure content loads
                if scroll_attempts % 5 == 0:
                    time.sleep(scroll_pause_time * 2)  # Longer pause every 5 scrolls
                else:
                    time.sleep(scroll_pause_time)
                
                # Force stop loading if we're stuck
                if scroll_attempts > 10 and scroll_attempts % 5 == 0:
                    self._force_stop_page_loading()
                
                # Extract articles using multiple methods
                article_elements = try_extract_articles()
                
                # Safety check for article_elements
                if article_elements is None:
                    article_elements = []
                
                # Check if we've found new articles
                if len(article_elements) <= last_count and last_count > 0:
                    print(f"No new articles found after scrolling (still at {len(article_elements)})")
                    consecutive_no_new += 1
                    if consecutive_no_new >= 3:  # If no new articles after 3 attempts, we're probably at the end
                        print("Reached end of article list")
                        break
                else:
                    consecutive_no_new = 0
                    
                if len(article_elements) > 0:
                    print(f"Found {len(article_elements)} article elements")
                    
                last_count = len(article_elements)
                
                # Process each article
                for article_element in article_elements:
                    # Skip if we already have enough articles
                    if len(self.articles) >= max_articles:
                        break
                        
                    # Check if we've already processed this article
                    if self._is_article_already_scraped(article_element):
                        continue
                        
                    # Extract article data
                    article_data = self._extract_article_data(article_element)
                    if article_data:
                        self.articles.append(article_data)
                
                print(f"Scraped {len(self.articles)} articles so far...")
                
            return self.articles
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return self.articles
        finally:
            self.close()
    
    def _is_article_already_scraped(self, article_element):
        """
        Check if an article has already been scraped.
        
        Args:
            article_element: Selenium element representing an article
            
        Returns:
            bool: True if the article has already been scraped, False otherwise
        """
        try:
            # Extract a unique identifier for the article (e.g., title or URL)
            link_element = article_element.find_element(By.CSS_SELECTOR, "a[data-test='content-link']")
            article_url = link_element.get_attribute("href")
            
            # Check if this URL is already in our scraped articles
            return any(article["url"] == article_url for article in self.articles)
        except NoSuchElementException:
            # If we can't extract a URL, assume we haven't scraped it yet
            return False
    
    def _extract_article_data(self, article_element):
        """
        Extract data from an article element.
        
        Args:
            article_element: Selenium element representing an article
            
        Returns:
            dict: Dictionary containing article data, or None if extraction failed
        """
        title, url, source, timestamp, summary = "Unknown", "Unknown", "Unknown", "Unknown", ""

        # Helper function to find element with multiple selectors
        def find_element_with_selectors(element, selectors, data_name):
            for i, selector in enumerate(selectors):
                try:
                    found_el = element.find_element(By.CSS_SELECTOR, selector)
                    print(f"Successfully found {data_name} using selector #{i+1}: '{selector}'")
                    return found_el
                except NoSuchElementException:
                    # print(f"Selector #{i+1} for {data_name} failed: '{selector}'")
                    continue
            print(f"All selectors failed for {data_name} within this article element.")
            return None

        try:
            # Extract title and URL
            link_selectors = [
                "a[data-test='content-link']",  # Primary
                "h3 a", 
                "a.mega-item-header-link",
                "a[href*='/news/']", # Links that likely point to news articles
                "div[class*='title'] a",
                "a[aria-label]" # Links with aria-labels often contain titles
            ]
            link_element = find_element_with_selectors(article_element, link_selectors, "title/link")
            
            if link_element:
                title_text = link_element.text.strip()
                # If title is empty, try to get it from aria-label or title attribute
                if not title_text:
                    title_text = link_element.get_attribute('aria-label') or link_element.get_attribute('title')
                title = title_text.strip() if title_text else "Unknown"
                url = link_element.get_attribute("href")
                if not url or not url.startswith('http'): # Ensure it's a full URL
                    # Try to find any link if specific one failed for URL
                    any_link = article_element.find_elements(By.CSS_SELECTOR, "a[href]")
                    if any_link:
                        url = any_link[0].get_attribute('href')
            else:
                print("Could not extract title/URL for an article.")
                # If no link_element found, try to get a URL from any 'a' tag as a last resort
                try:
                    any_link_tag = article_element.find_element(By.CSS_SELECTOR, "a[href]")
                    url = any_link_tag.get_attribute('href')
                    # Attempt to get a title from a heading tag if no link found
                    heading_tags = article_element.find_elements(By.CSS_SELECTOR, "h1, h2, h3, h4, h5, h6")
                    if heading_tags:
                        title = heading_tags[0].text.strip()
                except NoSuchElementException:
                    pass # No link or heading found

            # Extract source and timestamp
            source_time_selectors = [
                "div[data-test='source-and-date']", # Primary
                "div[class*='source'], div[class*='meta']",
                "span[class*='source'], span[class*='meta']",
                "div.Fz\(xs\)",
                "div:has(> span + span)" # Divs with two sibling spans (often source and time)
            ]
            source_time_element = find_element_with_selectors(article_element, source_time_selectors, "source/timestamp")
            
            if source_time_element:
                source_time_text = source_time_element.text.strip()
                parts = source_time_text.split("·") # Yahoo often uses '·' or similar separators
                if len(parts) == 1:
                    parts = source_time_text.split("-") # Try hyphen
                if len(parts) == 1:
                    parts = source_time_text.split("|") # Try pipe
                    
                source = parts[0].strip() if len(parts) > 0 else "Unknown"
                timestamp = parts[1].strip() if len(parts) > 1 else "Unknown"
                
                # If timestamp still unknown, try to find a time tag
                if timestamp == "Unknown":
                    try:
                        time_tag = source_time_element.find_element(By.CSS_SELECTOR, "time")
                        if time_tag:
                            timestamp = time_tag.text.strip() or time_tag.get_attribute('datetime')
                    except NoSuchElementException:
                        pass
            else:
                print("Could not extract source/timestamp for an article.")

            # Extract summary
            summary_selectors = [
                "p[data-test='content-snippet']",   # Primary
                "p[class*='summary']",
                "p[class*='snippet']",
                "p[class*='dek']",
                "p.js-content-viewer", 
                "div.js-content-viewer",
                "p.content-viewer", 
                "div.content-viewer",
                "p[itemprop='description']", 
                "div[itemprop='description']",
                "p.Lh\(1.38em\)"
            ]
            summary_element = find_element_with_selectors(article_element, summary_selectors, "summary")
            
            if summary_element:
                summary = summary_element.text.strip()
            else:
                print("Could not extract summary for an article.")

            # Ensure we have at least a URL or a title to consider it a valid article
            if url == "Unknown" and title == "Unknown":
                print("Failed to extract both URL and Title, discarding entry.")
                return None
                
            # Generate unique ID if URL is available
            article_id = "N/A"
            if url != "Unknown" and url is not None:
                try:
                    import hashlib
                    article_id = hashlib.md5(url.encode('utf-8')).hexdigest()
                except Exception as e:
                    print(f"Error generating ID for URL {url}: {e}")

            return {
                "id": article_id,
                "title": title,
                "source": source,
                "timestamp": timestamp,
                "summary": summary,
                "url": url,
                "ticker": self.ticker,
                "scraped_at": datetime.now().isoformat()
            }
            
        except Exception as e: # Changed from NoSuchElementException to catch broader errors during extraction
            print(f"General error extracting article data: {str(e)}. Article element HTML: {article_element.get_attribute('outerHTML')[:500]}...")
            return None
    
    def save_to_csv(self, filename=None):
        """
        Save the scraped articles to a CSV file.
        
        Args:
            filename (str): Path to save the CSV file (default: None, which uses ticker and date)
        
        Returns:
            str: Path to the saved file
        """
        if not filename:
            today = datetime.now().strftime("%Y-%m-%d")
            filename = f"../data/{self.ticker}_news_{today}.csv"
        
        df = pd.DataFrame(self.articles)
        df.to_csv(filename, index=False)
        print(f"Saved {len(self.articles)} articles to {filename}")
        return filename
    
    def save_to_json(self, filename=None):
        """
        Save the scraped articles to a JSON file.
        
        Args:
            filename (str): Path to save the JSON file (default: None, which uses ticker and date)
        
        Returns:
            str: Path to the saved file
        """
        if not filename:
            today = datetime.now().strftime("%Y-%m-%d")
            filename = f"../data/{self.ticker}_news_{today}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, indent=4)
        
        print(f"Saved {len(self.articles)} articles to {filename}")
        return filename
    
    def close(self):
        """Close the browser."""
        if hasattr(self, 'driver'):
            self.driver.quit()


def main():
    """Main function to run the scraper."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Scrape news articles from Yahoo Finance.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--max-articles', type=int, default=100, help='Maximum number of articles to scrape')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'both'], default='both',
                        help='Output format (csv, json, or both)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    args = parser.parse_args()
    
    # Initialize the scraper
    scraper = YahooFinanceNewsScraper(ticker=args.ticker, headless=args.headless)
    
    # Scroll and scrape
    print(f"Scraping Yahoo Finance news for {args.ticker}...")
    articles = scraper.scroll_and_scrape(max_articles=args.max_articles)
    
    # Save the results
    if args.format in ['csv', 'both']:
        scraper.save_to_csv()
    
    if args.format in ['json', 'both']:
        scraper.save_to_json()
    
    print(f"Scraped a total of {len(articles)} articles.")


if __name__ == "__main__":
    main()
