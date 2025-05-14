#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the enhanced Yahoo Finance News Scraper

This script provides a command line interface to run the Yahoo Finance scraper
with various options and improved handling of consent dialogs and loading issues.
"""

import os
import sys
import argparse
from yahoo_finance_scraper import YahooFinanceNewsScraper

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Scrape news articles from Yahoo Finance with enhanced features.')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--max-articles', type=int, default=50, help='Maximum number of articles to scrape')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no browser UI)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with extra logging and screenshots')
    parser.add_argument('--scroll-pause', type=float, default=1.0, help='Time to pause between scrolls (seconds)')
    parser.add_argument('--timeout', type=int, default=30, help='Page load timeout in seconds')
    parser.add_argument('--output-format', type=str, choices=['csv', 'json', 'both'], default='both',
                        help='Output format (csv, json, or both)')
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join('..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n===== Yahoo Finance News Scraper for {args.ticker} =====")
    print(f"Max Articles: {args.max_articles}")
    print(f"Headless Mode: {'Enabled' if args.headless else 'Disabled'}")
    print(f"Debug Mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"Output Format: {args.output_format}")
    print("=" * 50 + "\n")
    
    try:
        # Initialize the scraper with our improved settings
        print(f"Initializing scraper for {args.ticker}...")
        scraper = YahooFinanceNewsScraper(
            ticker=args.ticker,
            headless=args.headless,
            debug_mode=args.debug,
            page_load_timeout=args.timeout
        )
        
        # Run the scraping process
        print(f"Starting scraping process...")
        articles = scraper.scroll_and_scrape(
            max_articles=args.max_articles,
            scroll_pause_time=args.scroll_pause,
            force_scrape=True
        )
        
        # Save the results based on chosen format
        if args.output_format in ['csv', 'both']:
            csv_file = scraper.save_to_csv()
        
        if args.output_format in ['json', 'both']:
            json_file = scraper.save_to_json()
        
        # Print success message
        print("\n" + "=" * 50)
        print(f"Completed scraping {len(articles)} news articles for {args.ticker} from Yahoo Finance")
        if args.output_format in ['csv', 'both']:
            print(f"CSV data saved to: {csv_file}")
        if args.output_format in ['json', 'both']:
            print(f"JSON data saved to: {json_file}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during scraping: {str(e)}")
        sys.exit(1)
    finally:
        # Ensure we close the browser even if there's an error
        if 'scraper' in locals():
            scraper.close()

if __name__ == "__main__":
    main()
