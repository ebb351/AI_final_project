"""
Utility functions for the scraper.
"""
import hashlib
import logging
import re
from typing import Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md

logger = logging.getLogger(__name__)


def is_valid_url(url: str, allowed_domains: list[str]) -> bool:
    """
    Check if URL is valid and belongs to allowed domains.

    Args:
        url: The URL to validate
        allowed_domains: List of allowed domain names

    Returns:
        True if URL is valid and allowed, False otherwise
    """
    try:
        parsed = urlparse(url)

        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False

        # Must be http or https
        if parsed.scheme not in ['http', 'https']:
            return False

        # Must be in allowed domains
        domain_match = any(domain in parsed.netloc for domain in allowed_domains)

        return domain_match
    except Exception as e:
        logger.debug(f"Error validating URL {url}: {e}")
        return False


def should_skip_url(url: str, skip_extensions: list[str], skip_patterns: list[str]) -> bool:
    """
    Check if URL should be skipped based on patterns and extensions.

    Args:
        url: The URL to check
        skip_extensions: List of file extensions to skip
        skip_patterns: List of URL patterns to skip

    Returns:
        True if URL should be skipped, False otherwise
    """
    url_lower = url.lower()

    # Check extensions
    if any(url_lower.endswith(ext) for ext in skip_extensions):
        return True

    # Check patterns
    if any(pattern in url_lower for pattern in skip_patterns):
        return True

    return False


def normalize_url(url: str, base_url: str) -> str:
    """
    Normalize URL by resolving relative paths and removing fragments.

    Args:
        url: The URL to normalize
        base_url: The base URL for resolving relative paths

    Returns:
        Normalized absolute URL
    """
    # Resolve relative URLs
    absolute_url = urljoin(base_url, url)

    # Remove fragment
    parsed = urlparse(absolute_url)
    normalized = parsed._replace(fragment="").geturl()

    return normalized


def extract_text_from_html(html_content: str, selectors: list[str], remove_tags: list[str]) -> Optional[str]:
    """
    Extract main text content from HTML using a hierarchy of selectors.

    Args:
        html_content: Raw HTML content
        selectors: List of CSS selectors to try (in order)
        remove_tags: List of tag names to remove

    Returns:
        Extracted text content or None if extraction fails
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove unwanted tags
        for tag_name in remove_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Try selectors in order
        content_element = None
        for selector in selectors:
            if '.' in selector:
                # Class selector
                tag, class_name = selector.split('.', 1)
                content_element = soup.find(tag, class_=class_name)
            elif '#' in selector:
                # ID selector
                tag, id_name = selector.split('#', 1)
                content_element = soup.find(tag, id=id_name)
            else:
                # Simple tag selector
                content_element = soup.find(selector)

            if content_element:
                logger.debug(f"Content found using selector: {selector}")
                break

        if not content_element:
            logger.warning("No content element found, using entire body")
            content_element = soup

        return str(content_element)

    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}")
        return None


def html_to_markdown(html_content: str) -> str:
    """
    Convert HTML content to Markdown format.

    Args:
        html_content: HTML content to convert

    Returns:
        Markdown formatted text
    """
    try:
        # Convert to markdown while preserving structure
        markdown_text = md(
            html_content,
            heading_style="ATX",  # Use # style headers
            bullets="-",  # Use - for bullet points
            strip=['script', 'style']  # Remove these tags
        )

        # Clean up excessive whitespace
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
        markdown_text = markdown_text.strip()

        return markdown_text

    except Exception as e:
        logger.error(f"Error converting HTML to Markdown: {e}")
        return html_content  # Return original if conversion fails


def extract_title_from_html(html_content: str, url: str) -> str:
    """
    Extract title from HTML content.

    Args:
        html_content: Raw HTML content
        url: The URL of the page (fallback for title)

    Returns:
        Extracted title or URL-based fallback
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')

        # Try to find title tag
        title_tag = soup.find('title')
        if title_tag and title_tag.text.strip():
            return title_tag.text.strip()

        # Try to find h1
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.text.strip():
            return h1_tag.text.strip()

        # Fallback to URL path
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        if path_parts:
            return path_parts[-1].replace('-', ' ').replace('_', ' ').title()

        return "Untitled"

    except Exception as e:
        logger.error(f"Error extracting title: {e}")
        return "Untitled"


def generate_content_id(url: str, content: str) -> str:
    """
    Generate a unique ID for content based on URL and content hash.

    Args:
        url: The source URL
        content: The content text

    Returns:
        Unique identifier string
    """
    # Create hash from URL and first 1000 chars of content
    content_sample = content[:1000] if content else ""
    hash_input = f"{url}:{content_sample}"
    content_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    return content_hash[:16]  # Use first 16 chars


def infer_section_from_url(url: str) -> Optional[str]:
    """
    Infer the course section from URL structure.

    Args:
        url: The page URL

    Returns:
        Section name or None
    """
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p]

    # Common patterns: /book/section/, /courses/section/, etc.
    if len(path_parts) >= 2:
        # Second part is usually the section
        section = path_parts[1].replace('-', ' ').replace('_', ' ').title()
        return section

    return None


def calculate_url_depth(url: str, base_url: str) -> int:
    """
    Calculate the depth of a URL relative to base URL.

    Args:
        url: The URL to measure
        base_url: The base URL

    Returns:
        Depth level (0 = base, 1 = one level deep, etc.)
    """
    base_parsed = urlparse(base_url)
    url_parsed = urlparse(url)

    base_parts = [p for p in base_parsed.path.split('/') if p]
    url_parts = [p for p in url_parsed.path.split('/') if p]

    return len(url_parts) - len(base_parts)
