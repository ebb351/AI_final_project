"""
Extractors for multimedia content (YouTube transcripts, PDFs).
"""
import io
import logging
import re
from typing import Optional, List, Dict, Any
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class YouTubeExtractor:
    """Extract transcripts from YouTube videos."""

    @staticmethod
    def extract_video_ids(html_content: str) -> List[str]:
        """
        Extract YouTube video IDs from HTML content.

        Args:
            html_content: HTML content to search

        Returns:
            List of video IDs found
        """
        video_ids = set()

        # Pattern 1: iframe embeds
        iframe_pattern = r'<iframe[^>]+src=["\']https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)'
        video_ids.update(re.findall(iframe_pattern, html_content))

        # Pattern 2: Regular YouTube links
        link_pattern = r'https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)'
        video_ids.update(re.findall(link_pattern, html_content))

        # Pattern 3: Shortened youtu.be links
        short_pattern = r'https?://youtu\.be/([a-zA-Z0-9_-]+)'
        video_ids.update(re.findall(short_pattern, html_content))

        return list(video_ids)

    @staticmethod
    def get_transcript(video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get transcript for a YouTube video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dict with transcript data or None if extraction fails
        """
        try:
            # Try to get transcript (prefer manual captions)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try manual captions first
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
                logger.info(f"Found manual transcript for video {video_id}")
            except:
                # Fallback to auto-generated
                transcript = transcript_list.find_generated_transcript(['en'])
                logger.info(f"Using auto-generated transcript for video {video_id}")

            # Fetch the actual transcript
            transcript_data = transcript.fetch()

            # Combine all text segments
            full_text = " ".join([entry['text'] for entry in transcript_data])

            # Clean up the text
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            return {
                'video_id': video_id,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'text': full_text,
                'is_auto_generated': transcript.is_generated
            }

        except TranscriptsDisabled:
            logger.warning(f"Transcripts disabled for video {video_id}")
            return None
        except NoTranscriptFound:
            logger.warning(f"No transcript found for video {video_id}")
            return None
        except Exception as e:
            logger.error(f"Error extracting transcript for video {video_id}: {e}")
            return None

    @staticmethod
    def format_transcript_markdown(transcript_data: Dict[str, Any]) -> str:
        """
        Format transcript data as markdown.

        Args:
            transcript_data: Transcript data dictionary

        Returns:
            Markdown formatted transcript
        """
        markdown = f"\n\n## Lecture Transcript\n\n"
        markdown += f"**Video URL**: {transcript_data['url']}\n\n"

        if transcript_data['is_auto_generated']:
            markdown += "*Note: This transcript was auto-generated.*\n\n"

        markdown += transcript_data['text']

        return markdown


class PDFExtractor:
    """Extract text content from PDF documents."""

    @staticmethod
    def extract_pdf_links(html_content: str, base_url: str) -> List[str]:
        """
        Extract PDF links from HTML content.

        Args:
            html_content: HTML content to search
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute PDF URLs
        """
        pdf_links = set()

        # Find all links ending in .pdf
        pattern = r'<a[^>]+href=["\']([^"\']+\.pdf)["\']'
        matches = re.findall(pattern, html_content, re.IGNORECASE)

        for link in matches:
            # Make absolute URL
            if link.startswith('http'):
                pdf_links.add(link)
            else:
                # Resolve relative URL
                from urllib.parse import urljoin
                absolute_url = urljoin(base_url, link)
                pdf_links.add(absolute_url)

        return list(pdf_links)

    @staticmethod
    def extract_text_from_pdf(pdf_url: str, min_chars: int = 50) -> Optional[Dict[str, Any]]:
        """
        Download and extract text from a PDF file.

        Args:
            pdf_url: URL of the PDF file
            min_chars: Minimum characters to consider PDF readable

        Returns:
            Dict with PDF data or None if extraction fails
        """
        try:
            logger.info(f"Downloading PDF: {pdf_url}")

            # Download PDF
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            # Read PDF from bytes
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)

            # Extract text from all pages
            text_content = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num} of {pdf_url}: {e}")

            # Combine all text
            full_text = "\n\n".join(text_content)
            full_text = full_text.strip()

            # Check if PDF is readable
            is_readable = len(full_text) >= min_chars

            if not is_readable:
                logger.warning(f"PDF appears to be scanned/unreadable (< {min_chars} chars): {pdf_url}")

            return {
                'url': pdf_url,
                'text': full_text,
                'page_count': len(reader.pages),
                'is_readable': is_readable,
                'char_count': len(full_text)
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF {pdf_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_url}: {e}")
            return None

    @staticmethod
    def format_pdf_markdown(pdf_data: Dict[str, Any]) -> str:
        """
        Format PDF data as markdown.

        Args:
            pdf_data: PDF data dictionary

        Returns:
            Markdown formatted PDF content
        """
        if not pdf_data['is_readable']:
            return f"\n\n## PDF Document (Scanned/Unreadable)\n\n**URL**: {pdf_data['url']}\n\n*This PDF contains {pdf_data['page_count']} pages but appears to be scanned or image-based ({pdf_data['char_count']} characters extracted).*\n"

        markdown = f"\n\n## PDF Document\n\n"
        markdown += f"**URL**: {pdf_data['url']}\n"
        markdown += f"**Pages**: {pdf_data['page_count']}\n\n"
        markdown += pdf_data['text']

        return markdown
