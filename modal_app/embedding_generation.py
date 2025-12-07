#!/usr/bin/env python3
"""
Modal-based Embedding Generation with Parallel CPUs

Generates embeddings for entities and text chunks using up to 10 parallel Modal CPUs.
Uses nomic-ai/nomic-embed-text-v1.5 (768 dimensions).

Run with: modal run modal_app/embedding_generation.py

Cost estimate: ~$0.05-0.10 for full dataset (CPU-only, parallel)
"""
import modal
import json
import time
from typing import List, Dict, Any, Tuple

# Modal App definition
app = modal.App("graphrag-embeddings")

# CPU image with sentence-transformers - pre-download the model
embedding_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.35.0",
        "einops",
        "tqdm",
    )
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)\""
    )
)

# Volume for caching model (backup)
model_cache = modal.Volume.from_name("embedding-model-cache", create_if_missing=True)

# Maximum parallel instances
MAX_PARALLEL = 10
# Smaller batch size to ensure completion within timeout
BATCH_SIZE = 500


@app.function(
    image=embedding_image,
    volumes={"/cache": model_cache},
    cpu=2,  # 2 CPUs per instance (sufficient for inference)
    memory=4096,  # 4GB RAM (model is ~500MB)
    timeout=1800,  # 30 min timeout per batch (increased from 15)
    retries=2,
)
def generate_embeddings_batch(
    texts: List[str],
    ids: List[str],
    batch_id: int,
    prefix: str = "search_document: "
) -> Tuple[int, Dict[str, List[float]]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: List of texts to embed
        ids: List of IDs corresponding to texts
        batch_id: Batch identifier for ordering
        prefix: Nomic prefix for embedding type

    Returns:
        Tuple of (batch_id, embedding dictionary)
    """
    import os
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache"

    from sentence_transformers import SentenceTransformer

    print(f"Batch {batch_id}: Loading model (pre-cached in image)...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )

    print(f"Batch {batch_id}: Generating embeddings for {len(texts)} texts...")

    # Add prefix
    prefixed_texts = [prefix + t for t in texts]

    # Generate embeddings in smaller sub-batches for progress tracking
    all_embeddings = []
    sub_batch_size = 32  # Smaller sub-batches
    for i in range(0, len(prefixed_texts), sub_batch_size):
        sub_batch = prefixed_texts[i:i + sub_batch_size]
        embeddings = model.encode(
            sub_batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings)
        if (i // sub_batch_size) % 5 == 0:
            print(f"Batch {batch_id}: Progress {i + len(sub_batch)}/{len(texts)}")

    # Build result dictionary
    result = {}
    for id_, emb in zip(ids, all_embeddings):
        result[id_] = emb.tolist()

    print(f"Batch {batch_id}: Done! Generated {len(result)} embeddings")
    return batch_id, result


@app.local_entrypoint()
def main():
    """
    Main entrypoint - loads data locally and runs parallel embedding generation on Modal.
    """
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    entities_file = project_root / "data" / "entities_normalized.json"
    chunks_file = project_root / "data" / "chunks_filtered.json"
    output_file = project_root / "data" / "embeddings.json"

    print("=" * 80)
    print("MODAL PARALLEL EMBEDDING GENERATION")
    print(f"Max parallel instances: {MAX_PARALLEL}")
    print("=" * 80)
    print(f"Entities file: {entities_file}")
    print(f"Chunks file: {chunks_file}")
    print(f"Output file: {output_file}")

    # Load data locally
    print("\nLoading data...")
    with open(entities_file, "r") as f:
        entities_data = json.load(f)

    with open(chunks_file, "r") as f:
        chunks_data = json.load(f)

    entities = entities_data.get("entities", [])
    print(f"Loaded {len(entities)} entities")
    print(f"Loaded {len(chunks_data)} chunks")

    # Prepare entity data
    entity_texts = []
    entity_names = []
    for entity in entities:
        name = entity.get("name", "")
        description = entity.get("description", "")
        entity_type = entity.get("type", "")

        if description:
            text = f"{name} ({entity_type}): {description}"
        else:
            text = f"{name} ({entity_type})"

        entity_texts.append(text)
        entity_names.append(name)

    # Prepare chunk data
    chunk_texts = []
    chunk_ids = []
    for chunk in chunks_data:
        chunk_id = chunk.get("id", "")
        text = chunk.get("text", "")[:8192]  # Truncate to model max
        chunk_texts.append(text)
        chunk_ids.append(chunk_id)

    # Calculate batch sizes - use smaller batches for reliability
    total_items = len(entity_texts) + len(chunk_texts)
    items_per_batch = BATCH_SIZE  # Fixed smaller batch size for reliability

    print(f"\nTotal items: {total_items}")
    print(f"Items per batch: {items_per_batch}")
    print(f"Expected batches: {(total_items + items_per_batch - 1) // items_per_batch}")

    # Create batches - mix entities and chunks for load balancing
    all_texts = entity_texts + chunk_texts
    all_ids = ["entity:" + n for n in entity_names] + ["chunk:" + c for c in chunk_ids]

    batches = []
    for i in range(0, len(all_texts), items_per_batch):
        batch_texts = all_texts[i:i + items_per_batch]
        batch_ids = all_ids[i:i + items_per_batch]
        batches.append((batch_texts, batch_ids))

    print(f"Created {len(batches)} batches")

    # Run on Modal in parallel
    print("\n" + "=" * 80)
    print(f"Running {len(batches)} batches on Modal (up to {MAX_PARALLEL} parallel)...")
    print("=" * 80)

    start_time = time.time()

    # Launch all batches in parallel
    batch_args = [
        (batch_texts, batch_ids, batch_id)
        for batch_id, (batch_texts, batch_ids) in enumerate(batches)
    ]

    # Use starmap for parallel execution
    results = []
    for batch_id, result in generate_embeddings_batch.starmap(batch_args):
        results.append((batch_id, result))
        print(f"Received results from batch {batch_id}")

    elapsed = time.time() - start_time

    # Merge results
    print("\nMerging results...")
    entity_embeddings = {}
    chunk_embeddings = {}

    for batch_id, batch_result in sorted(results, key=lambda x: x[0]):
        for id_, embedding in batch_result.items():
            if id_.startswith("entity:"):
                entity_name = id_[7:]  # Remove "entity:" prefix
                entity_embeddings[entity_name] = embedding
            elif id_.startswith("chunk:"):
                chunk_id = id_[6:]  # Remove "chunk:" prefix
                chunk_embeddings[chunk_id] = embedding

    # Get embedding dimension from first result
    sample_emb = next(iter(entity_embeddings.values()))
    embedding_dim = len(sample_emb)

    # Build output
    output_data = {
        "metadata": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "embedding_dimension": embedding_dim,
            "entity_count": len(entity_embeddings),
            "chunk_count": len(chunk_embeddings),
            "total_embeddings": len(entity_embeddings) + len(chunk_embeddings),
            "generation_time_seconds": round(elapsed, 1),
            "parallel_batches": len(batches)
        },
        "entity_embeddings": entity_embeddings,
        "chunk_embeddings": chunk_embeddings
    }

    # Save results
    print(f"\nSaving embeddings to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(output_data, f)

    # Summary
    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 80)
    print(f"Entity embeddings: {len(entity_embeddings)}")
    print(f"Chunk embeddings: {len(chunk_embeddings)}")
    print(f"Total embeddings: {len(entity_embeddings) + len(chunk_embeddings)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Parallel batches: {len(batches)}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Output file size: {file_size_mb:.1f} MB")
    print(f"Output saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
