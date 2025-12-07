#!/usr/bin/env python3
"""
Main entry point for running the course scraper.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    TARGET_URL,
    ALLOWED_DOMAINS,
    REQUEST_DELAY,
    USER_AGENT,
    HTML_CONTENT_SELECTORS,
    REMOVE_TAGS,
    SKIP_EXTENSIONS,
    SKIP_PATTERNS,
    PDF_MIN_CHARS,
    OUTPUT_FILE,
    LOGS_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    DATA_DIR
)
from src.scraper.engine import CourseScraper


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"scraper_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def main():
    """Main function to run the scraper."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("COURSE SCRAPER - Data Acquisition")
    logger.info("=" * 60)
    logger.info(f"Target URL: {TARGET_URL}")
    logger.info(f"Allowed domains: {ALLOWED_DOMAINS}")
    logger.info(f"Request delay: {REQUEST_DELAY}s")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info("=" * 60)

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    # Initialize scraper
    scraper = CourseScraper(
        start_url=TARGET_URL,
        allowed_domains=ALLOWED_DOMAINS,
        request_delay=REQUEST_DELAY,
        user_agent=USER_AGENT,
        html_selectors=HTML_CONTENT_SELECTORS,
        remove_tags=REMOVE_TAGS,
        skip_extensions=SKIP_EXTENSIONS,
        skip_patterns=SKIP_PATTERNS,
        pdf_min_chars=PDF_MIN_CHARS
    )

    try:
        # Run the crawl
        logger.info("Starting crawl...")
        results = scraper.crawl()

        # Save results
        logger.info(f"Saving {len(results)} items to {OUTPUT_FILE}...")
        scraper.save_results(str(OUTPUT_FILE))

        # Print summary
        stats = scraper.get_summary_stats()
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Total content items: {stats['total_items']}")
        print(f"   - HTML pages: {stats['pages_processed']}")
        print(f"   - PDF documents: {stats['pdfs_extracted']}")
        print(f"   - Video transcripts: {stats['transcripts_extracted']}")
        print(f"URLs visited: {stats['urls_visited']}")
        print(f"Errors: {stats['errors']}")
        print(f"Duration: {stats['duration_seconds']:.2f} seconds")
        print(f"Output: {OUTPUT_FILE}")
        print("=" * 60)

        # Log all URLs
        print("\n" + "=" * 60)
        print("ALL INGESTED URLs")
        print("=" * 60)
        for item in results:
            print(f"[{item['source_type']}] {item['url']}")
        print("=" * 60)

        logger.info("Scraping completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
