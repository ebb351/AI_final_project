#!/usr/bin/env python3
"""
Text Chunking Module for GraphRAG Construction

This module processes the raw course data and splits it into token-based chunks
for entity extraction. Chunking preserves semantic coherence while staying
within LLM context limits.

Process:
1. Load course_data.json (302 documents from scraping)
2. Split each document into chunks of 1200 tokens with 100-token overlap
3. Generate unique IDs for each chunk
4. Save to chunks.json

The overlap ensures entities mentioned near chunk boundaries aren't lost.
Uses tiktoken for accurate token counting with cl100k_base encoding (GPT-4/Qwen compatible).
"""
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import tiktoken

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 1200  # tokens per chunk
CHUNK_OVERLAP = 100  # overlap between chunks
ENCODING_NAME = "cl100k_base"  # GPT-4/Qwen tokenizer


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    source_id: str
    source_url: str
    source_title: str
    source_type: str
    chunk_index: int
    total_chunks: int
    token_count: int
    metadata: Dict[str, Any]


class TextChunker:
    """Chunks text into token-based segments with overlap."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize text chunker.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(ENCODING_NAME)

        logger.info(f"Initialized TextChunker: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks based on token count.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            # Text fits in single chunk
            return [text]

        chunks = []
        start_idx = 0

        while start_idx < total_tokens:
            # Calculate end index
            end_idx = min(start_idx + self.chunk_size, total_tokens)

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start index forward, accounting for overlap
            start_idx += (self.chunk_size - self.chunk_overlap)

            # Break if we've covered all tokens
            if end_idx >= total_tokens:
                break

        return chunks

    def generate_chunk_id(self, source_id: str, chunk_index: int, text: str) -> str:
        """
        Generate unique chunk ID.

        Args:
            source_id: ID of source document
            chunk_index: Index of chunk within document
            text: Chunk text (for hash)

        Returns:
            Unique chunk ID
        """
        # Create hash from source ID, index, and text sample
        hash_input = f"{source_id}:{chunk_index}:{text[:100]}"
        text_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"{source_id}_chunk_{chunk_index}_{text_hash}"

    def chunk_document(self, document: Dict[str, Any]) -> List[TextChunk]:
        """
        Chunk a single document into multiple chunks.

        Args:
            document: Document from course_data.json

        Returns:
            List of TextChunk objects
        """
        source_id = document["id"]
        content = document["content"]
        source_url = document["url"]
        source_title = document["title"]
        source_type = document["source_type"]
        metadata = document["metadata"]

        # Split content into chunks
        text_chunks = self.chunk_text(content)

        # Create TextChunk objects
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk = TextChunk(
                id=self.generate_chunk_id(source_id, idx, chunk_text),
                text=chunk_text,
                source_id=source_id,
                source_url=source_url,
                source_title=source_title,
                source_type=source_type,
                chunk_index=idx,
                total_chunks=len(text_chunks),
                token_count=self.count_tokens(chunk_text),
                metadata={
                    **metadata,  # Include original metadata
                    "is_first_chunk": idx == 0,
                    "is_last_chunk": idx == len(text_chunks) - 1
                }
            )
            chunks.append(chunk)

        return chunks

    def chunk_all_documents(self, documents: List[Dict[str, Any]]) -> List[TextChunk]:
        """
        Chunk all documents in the corpus.

        Args:
            documents: List of documents from course_data.json

        Returns:
            List of all chunks
        """
        all_chunks = []

        logger.info(f"Chunking {len(documents)} documents...")

        for doc_idx, document in enumerate(documents, 1):
            if doc_idx % 10 == 0:
                logger.info(f"Processing document {doc_idx}/{len(documents)}...")

            try:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking document {document.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        return all_chunks


def main():
    """Main function to chunk course data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / "course_data.json"
    output_file = project_root / "data" / "chunks.json"

    logger.info("=" * 80)
    logger.info("TEXT CHUNKING")
    logger.info("=" * 80)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Chunk size: {CHUNK_SIZE} tokens")
    logger.info(f"Chunk overlap: {CHUNK_OVERLAP} tokens")
    logger.info("=" * 80)

    # Load course data
    logger.info("Loading course data...")
    with open(input_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info(f"Loaded {len(documents)} documents")

    # Calculate total tokens
    chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    total_tokens = sum(chunker.count_tokens(doc["content"]) for doc in documents)
    logger.info(f"Total tokens in corpus: {total_tokens:,}")

    # Estimate chunks
    avg_chunk_size = CHUNK_SIZE - (CHUNK_OVERLAP / 2)
    estimated_chunks = int(total_tokens / avg_chunk_size)
    logger.info(f"Estimated chunks: ~{estimated_chunks:,}")

    # Chunk all documents
    logger.info("\nChunking documents...")
    all_chunks = chunker.chunk_all_documents(documents)

    # Convert to dict for JSON serialization
    chunks_dict = [asdict(chunk) for chunk in all_chunks]

    # Save to JSON
    logger.info(f"\nSaving {len(chunks_dict)} chunks to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, indent=2, ensure_ascii=False)

    # Statistics
    logger.info("\n" + "=" * 80)
    logger.info("CHUNKING STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total documents: {len(documents)}")
    logger.info(f"Total chunks: {len(chunks_dict)}")
    logger.info(f"Average chunks per document: {len(chunks_dict) / len(documents):.2f}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Average tokens per chunk: {sum(c['token_count'] for c in chunks_dict) / len(chunks_dict):.2f}")

    # Breakdown by source type
    from collections import Counter
    source_types = Counter(chunk["source_type"] for chunk in chunks_dict)
    logger.info("\nChunks by source type:")
    for source_type, count in source_types.items():
        logger.info(f"  {source_type}: {count}")

    # File size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"\nOutput file size: {file_size_mb:.2f} MB")
    logger.info("=" * 80)
    logger.info("Chunking complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
