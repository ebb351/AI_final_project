"""
Core Scraper Engine with BFS Crawler

This module implements a breadth-first search (BFS) crawler that recursively
explores the course website, extracting HTML pages and PDF documents.

Features:
- BFS traversal starting from course homepage
- Rate limiting (0.75s delay between requests) to respect server load
- Multi-modal extraction: HTML pages, PDF documents, YouTube links
- Duplicate detection via URL normalization
- Structured JSON output with metadata tracking
"""
import json
import logging
import time
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, Set, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .utils import (
    is_valid_url,
    should_skip_url,
    normalize_url,
    extract_text_from_html,
    html_to_markdown,
    extract_title_from_html,
    generate_content_id,
    infer_section_from_url,
    calculate_url_depth
)
from .extractors import YouTubeExtractor, PDFExtractor

logger = logging.getLogger(__name__)


class CourseScraper:
    """
    Recursive web scraper with BFS traversal for course content extraction.
    """

    def __init__(
        self,
        start_url: str,
        allowed_domains: List[str],
        request_delay: float = 0.75,
        user_agent: str = None,
        html_selectors: List[str] = None,
        remove_tags: List[str] = None,
        skip_extensions: List[str] = None,
        skip_patterns: List[str] = None,
        pdf_min_chars: int = 50
    ):
        """
        Initialize the scraper.

        Args:
            start_url: Starting URL for crawling
            allowed_domains: List of allowed domain names
            request_delay: Delay between requests in seconds
            user_agent: User agent string for requests
            html_selectors: CSS selectors for content extraction
            remove_tags: HTML tags to remove
            skip_extensions: File extensions to skip
            skip_patterns: URL patterns to skip
            pdf_min_chars: Minimum chars for readable PDF
        """
        self.start_url = start_url
        self.allowed_domains = allowed_domains
        self.request_delay = request_delay
        self.user_agent = user_agent or "Mozilla/5.0 (compatible; CourseScraperBot/1.0)"
        self.html_selectors = html_selectors or ["article", "main", "body"]
        self.remove_tags = remove_tags or ["nav", "footer", "script", "style"]
        self.skip_extensions = skip_extensions or []
        self.skip_patterns = skip_patterns or []
        self.pdf_min_chars = pdf_min_chars

        # Tracking
        self.visited: Set[str] = set()
        self.queue: deque = deque()
        self.results: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            'pages_processed': 0,
            'pdfs_extracted': 0,
            'transcripts_extracted': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def fetch_url(self, url: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Fetch content from URL.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (content, status_code) or (None, None) if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text, response.status_code

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            self.stats['errors'] += 1
            return None, None

    def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """
        Extract all links from HTML content.

        Args:
            html_content: HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of normalized URLs
        """
        links = []

        try:
            soup = BeautifulSoup(html_content, 'lxml')

            for anchor in soup.find_all('a', href=True):
                href = anchor['href']

                # Normalize URL
                normalized = normalize_url(href, base_url)

                # Validate
                if not is_valid_url(normalized, self.allowed_domains):
                    continue

                # Check skip patterns
                if should_skip_url(normalized, self.skip_extensions, self.skip_patterns):
                    continue

                links.append(normalized)

        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {e}")

        return links

    def process_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Process a single page and extract content.

        Args:
            url: URL to process

        Returns:
            Extracted content data or None if processing fails
        """
        logger.info(f"Processing: {url}")

        # Fetch HTML
        html_content, status_code = self.fetch_url(url)
        if not html_content:
            return None

        # Extract main content
        content_html = extract_text_from_html(
            html_content,
            self.html_selectors,
            self.remove_tags
        )

        if not content_html:
            logger.warning(f"No content extracted from {url}")
            return None

        # Convert to markdown
        markdown_content = html_to_markdown(content_html)

        # Extract title
        title = extract_title_from_html(html_content, url)

        # Extract YouTube videos
        video_ids = YouTubeExtractor.extract_video_ids(html_content)
        for video_id in video_ids:
            transcript_data = YouTubeExtractor.get_transcript(video_id)
            if transcript_data:
                # Append transcript to content
                markdown_content += YouTubeExtractor.format_transcript_markdown(transcript_data)

                # Also create separate entry for transcript
                self.results.append({
                    'id': generate_content_id(transcript_data['url'], transcript_data['text']),
                    'url': transcript_data['url'],
                    'title': f"{title} - Lecture Video",
                    'content': transcript_data['text'],
                    'source_type': 'video_transcript',
                    'metadata': {
                        'parent_page': url,
                        'video_id': video_id,
                        'is_auto_generated': transcript_data['is_auto_generated']
                    }
                })

                self.stats['transcripts_extracted'] += 1
                logger.info(f"Extracted transcript for video {video_id}")

        # Extract PDFs
        pdf_links = PDFExtractor.extract_pdf_links(html_content, url)
        for pdf_url in pdf_links:
            pdf_data = PDFExtractor.extract_text_from_pdf(pdf_url, self.pdf_min_chars)
            if pdf_data:
                # Create separate entry for PDF
                self.results.append({
                    'id': generate_content_id(pdf_url, pdf_data['text']),
                    'url': pdf_url,
                    'title': f"PDF: {pdf_url.split('/')[-1]}",
                    'content': pdf_data['text'],
                    'source_type': 'pdf_document',
                    'metadata': {
                        'parent_page': url,
                        'page_count': pdf_data['page_count'],
                        'is_readable': pdf_data['is_readable'],
                        'char_count': pdf_data['char_count']
                    }
                })

                self.stats['pdfs_extracted'] += 1
                logger.info(f"Extracted PDF: {pdf_url}")

        # Create page entry
        page_data = {
            'id': generate_content_id(url, markdown_content),
            'url': url,
            'title': title,
            'content': markdown_content,
            'source_type': 'html_page',
            'metadata': {
                'section': infer_section_from_url(url),
                'depth': calculate_url_depth(url, self.start_url),
                'video_count': len(video_ids),
                'pdf_count': len(pdf_links)
            }
        }

        self.stats['pages_processed'] += 1

        return page_data

    def crawl(self) -> List[Dict[str, Any]]:
        """
        Perform BFS crawl starting from start_url.

        Returns:
            List of extracted content data
        """
        logger.info(f"Starting crawl from {self.start_url}")
        self.stats['start_time'] = datetime.now()

        # Initialize queue
        self.queue.append(self.start_url)
        self.visited.add(self.start_url)

        # Progress bar
        pbar = tqdm(desc="Crawling", unit="page")

        try:
            while self.queue:
                # Get next URL
                current_url = self.queue.popleft()

                # Process page
                page_data = self.process_page(current_url)
                if page_data:
                    self.results.append(page_data)

                # Fetch HTML again to extract links (we need fresh content)
                html_content, _ = self.fetch_url(current_url)
                if html_content:
                    # Extract and queue new links
                    links = self.extract_links(html_content, current_url)

                    for link in links:
                        if link not in self.visited:
                            self.visited.add(link)
                            self.queue.append(link)
                            logger.debug(f"Queued: {link}")

                # Rate limiting
                time.sleep(self.request_delay)

                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'pages': self.stats['pages_processed'],
                    'queue': len(self.queue),
                    'pdfs': self.stats['pdfs_extracted'],
                    'videos': self.stats['transcripts_extracted']
                })

        except KeyboardInterrupt:
            logger.warning("Crawl interrupted by user")

        finally:
            pbar.close()

        self.stats['end_time'] = datetime.now()
        self._log_summary()

        return self.results

    def _log_summary(self):
        """Log crawl summary statistics."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        logger.info("=" * 60)
        logger.info("CRAWL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Pages processed: {self.stats['pages_processed']}")
        logger.info(f"PDFs extracted: {self.stats['pdfs_extracted']}")
        logger.info(f"Transcripts extracted: {self.stats['transcripts_extracted']}")
        logger.info(f"Total content items: {len(self.results)}")
        logger.info(f"URLs visited: {len(self.visited)}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info("=" * 60)

    def save_results(self, output_file: str):
        """
        Save crawl results to JSON file.

        Args:
            output_file: Path to output JSON file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            'total_items': len(self.results),
            'urls_visited': len(self.visited),
            'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds() if self.stats['end_time'] else 0
        }
