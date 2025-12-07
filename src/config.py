"""
Configuration settings for the Course Scraper.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Scraper settings
TARGET_URL = "https://pantelis.github.io/"
ALLOWED_DOMAINS = ["pantelis.github.io"]

# Rate limiting (seconds between requests)
REQUEST_DELAY = 0.75  # 0.75 seconds between requests to be polite

# Content extraction settings
HTML_CONTENT_SELECTORS = [
    "article.markdown-body",
    "article",
    "div#main-content",
    "div.post-content",
    "main",
    "body"
]

# Tags to remove from HTML (navigation, footers, scripts)
REMOVE_TAGS = ["nav", "footer", "script", "style", "header", "aside"]

# File patterns to skip
SKIP_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".woff", ".woff2"]
SKIP_PATTERNS = ["/tag/", "/category/", "/search/", "/feed/"]

# PDF extraction settings
PDF_MIN_CHARS = 50  # Flag PDFs with less than this as scanned/unreadable

# Logging settings
LOG_LEVEL = "DEBUG"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Output settings
OUTPUT_FILE = DATA_DIR / "course_data.json"

# User agent for requests
USER_AGENT = "Mozilla/5.0 (compatible; CourseScraperBot/1.0; +Educational Research)"
