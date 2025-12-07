#!/usr/bin/env python3
"""
Local Embedding Generation Script

Generates embeddings for entities and text chunks using nomic-ai/nomic-embed-text-v1.5.
This script runs locally on CPU - no Modal required.

Input:
  - data/entities_normalized.json (16,728 unique entities)
  - data/chunks_filtered.json (7,636 text chunks)

Output:
  - data/embeddings.json (entity and chunk embeddings)

Estimated time: 30-60 minutes on modern CPU (M1/M2 Mac)
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_embedding_model():
    """Load the nomic embedding model."""
    logger.info("Loading embedding model: nomic-ai/nomic-embed-text-v1.5")
    logger.info("This may take a minute on first run (downloading ~500MB)...")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_entity_embeddings(
    model,
    entities: List[Dict[str, Any]],
    batch_size: int = 64
) -> Dict[str, List[float]]:
    """
    Generate embeddings for entities.

    Format: "search_document: {entity_name}: {entity_description}"
    """
    logger.info(f"Generating embeddings for {len(entities)} entities...")

    entity_embeddings = {}

    # Prepare texts
    texts = []
    entity_names = []
    for entity in entities:
        name = entity.get("name", "")
        description = entity.get("description", "")
        entity_type = entity.get("type", "")

        # Format: "Entity Name (TYPE): Description"
        if description:
            text = f"{name} ({entity_type}): {description}"
        else:
            text = f"{name} ({entity_type})"

        texts.append(text)
        entity_names.append(name)

    # Generate embeddings in batches
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Entity embeddings"):
        batch_texts = texts[i:i + batch_size]

        # Add nomic prefix for document embedding
        prefixed = ["search_document: " + t for t in batch_texts]

        # Generate embeddings
        batch_embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        all_embeddings.extend(batch_embeddings.tolist())

    # Map to entity names
    for name, embedding in zip(entity_names, all_embeddings):
        entity_embeddings[name] = embedding

    logger.info(f"Generated {len(entity_embeddings)} entity embeddings")
    return entity_embeddings


def generate_chunk_embeddings(
    model,
    chunks: List[Dict[str, Any]],
    batch_size: int = 32,
    max_length: int = 8192
) -> Dict[str, List[float]]:
    """
    Generate embeddings for text chunks.

    Format: "search_document: {chunk_text}"
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")

    chunk_embeddings = {}

    # Prepare texts (truncate to max length)
    texts = []
    chunk_ids = []
    for chunk in chunks:
        chunk_id = chunk.get("id", "")
        text = chunk.get("text", "")[:max_length]  # Truncate very long chunks

        texts.append(text)
        chunk_ids.append(chunk_id)

    # Generate embeddings in batches
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Chunk embeddings"):
        batch_texts = texts[i:i + batch_size]

        # Add nomic prefix for document embedding
        prefixed = ["search_document: " + t for t in batch_texts]

        # Generate embeddings
        batch_embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        all_embeddings.extend(batch_embeddings.tolist())

    # Map to chunk IDs
    for chunk_id, embedding in zip(chunk_ids, all_embeddings):
        chunk_embeddings[chunk_id] = embedding

    logger.info(f"Generated {len(chunk_embeddings)} chunk embeddings")
    return chunk_embeddings


def main():
    """Main embedding generation pipeline."""
    project_root = Path(__file__).parent.parent

    # Input files
    entities_file = project_root / "data" / "entities_normalized.json"
    chunks_file = project_root / "data" / "chunks_filtered.json"

    # Output file
    output_file = project_root / "data" / "embeddings.json"

    logger.info("=" * 80)
    logger.info("EMBEDDING GENERATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Entities file: {entities_file}")
    logger.info(f"Chunks file: {chunks_file}")
    logger.info(f"Output file: {output_file}")

    # Load normalized entities
    logger.info("\nLoading normalized entities...")
    with open(entities_file, "r", encoding="utf-8") as f:
        normalized_data = json.load(f)

    entities = normalized_data.get("entities", [])
    logger.info(f"Loaded {len(entities)} unique entities")

    # Load chunks
    logger.info("\nLoading chunks...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks")

    # Load model
    logger.info("\n" + "=" * 40)
    start_time = time.time()
    model = load_embedding_model()
    load_time = time.time() - start_time
    logger.info(f"Model load time: {load_time:.1f}s")

    # Generate entity embeddings
    logger.info("\n" + "=" * 40)
    start_time = time.time()
    entity_embeddings = generate_entity_embeddings(
        model,
        entities,
        batch_size=64
    )
    entity_time = time.time() - start_time
    logger.info(f"Entity embedding time: {entity_time:.1f}s ({len(entities)/entity_time:.1f} entities/s)")

    # Generate chunk embeddings
    logger.info("\n" + "=" * 40)
    start_time = time.time()
    chunk_embeddings = generate_chunk_embeddings(
        model,
        chunks,
        batch_size=32
    )
    chunk_time = time.time() - start_time
    logger.info(f"Chunk embedding time: {chunk_time:.1f}s ({len(chunks)/chunk_time:.1f} chunks/s)")

    # Calculate statistics
    sample_embedding = next(iter(entity_embeddings.values()))
    embedding_dim = len(sample_embedding)

    # Prepare output
    output_data = {
        "metadata": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "embedding_dimension": embedding_dim,
            "entity_count": len(entity_embeddings),
            "chunk_count": len(chunk_embeddings),
            "total_embeddings": len(entity_embeddings) + len(chunk_embeddings),
            "generation_time_seconds": {
                "model_load": round(load_time, 1),
                "entity_embeddings": round(entity_time, 1),
                "chunk_embeddings": round(chunk_time, 1),
                "total": round(load_time + entity_time + chunk_time, 1)
            }
        },
        "entity_embeddings": entity_embeddings,
        "chunk_embeddings": chunk_embeddings
    }

    # Save output
    logger.info(f"\nSaving embeddings to {output_file}...")
    logger.info("(This may take a moment for large files...)")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f)

    # Calculate file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    # Summary
    total_time = load_time + entity_time + chunk_time
    logger.info("\n" + "=" * 80)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Entity embeddings: {len(entity_embeddings)}")
    logger.info(f"Chunk embeddings: {len(chunk_embeddings)}")
    logger.info(f"Total embeddings: {len(entity_embeddings) + len(chunk_embeddings)}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Output file size: {file_size_mb:.1f} MB")
    logger.info(f"Output saved to: {output_file}")
    logger.info("=" * 80)

    return output_data


if __name__ == "__main__":
    main()
