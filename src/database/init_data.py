"""
Data initialization module for GraphRAG.
Auto-imports MongoDB data and pulls Ollama models if needed on startup.

This module is self-contained (does not depend on scripts/) so it works
inside the Docker container where only src/ is copied.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import httpx
from pymongo import MongoClient, ASCENDING
from pymongo.database import Database
from pymongo.errors import BulkWriteError
from bson import json_util

logger = logging.getLogger(__name__)

REQUIRED_COLLECTIONS = ["text_chunks", "graph_nodes", "graph_edges", "embeddings"]
ALL_COLLECTIONS = ["text_chunks", "graph_nodes", "graph_edges", "embeddings", "community_reports"]


def is_mongodb_populated(db: Database) -> bool:
    """
    Check if MongoDB has data in ALL required collections.
    Returns True only if all collections have at least one document.
    """
    for collection_name in REQUIRED_COLLECTIONS:
        count = db[collection_name].count_documents({}, limit=1)
        if count == 0:
            logger.info(f"Collection '{collection_name}' is empty")
            return False
    return True


def _get_data_directory() -> Path:
    """Determine the data directory path."""
    # Docker path
    docker_path = Path("/app/data/mongodb_export")
    if docker_path.exists():
        return docker_path

    # Local development path
    local_path = Path(__file__).parent.parent.parent / "data" / "mongodb_export"
    return local_path


def _import_collection(db: Database, collection_name: str, filepath: Path, batch_size: int = 500) -> Dict[str, Any]:
    """Import a single collection from JSON file."""
    logger.info(f"Importing {collection_name}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    total = len(documents)
    logger.info(f"  Loaded {total:,} documents from {filepath.name}")

    collection = db[collection_name]

    # Clear existing data
    existing = collection.count_documents({})
    if existing > 0:
        logger.info(f"  Clearing {existing:,} existing documents...")
        collection.delete_many({})

    # Import in batches
    imported = 0
    errors = 0

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        batch_docs = [json_util.loads(json.dumps(doc)) for doc in batch]

        try:
            result = collection.insert_many(batch_docs, ordered=False)
            imported += len(result.inserted_ids)
        except BulkWriteError as e:
            imported += e.details.get("nInserted", 0)
            errors += len(e.details.get("writeErrors", []))

        if (i + batch_size) % 2000 == 0 or i + batch_size >= total:
            logger.info(f"  Progress: {min(i + batch_size, total):,} / {total:,}")

    logger.info(f"  Imported {imported:,} documents ({errors} errors)")
    return {"collection": collection_name, "total": total, "imported": imported, "errors": errors}


def _create_indexes(db: Database):
    """Create indexes for query performance."""
    logger.info("Creating indexes...")

    indexes = {
        "text_chunks": [[("source_id", ASCENDING)], [("source_type", ASCENDING)]],
        "graph_nodes": [[("type", ASCENDING)], [("community_id", ASCENDING)]],
        "graph_edges": [
            [("source", ASCENDING), ("target", ASCENDING)],
            [("source", ASCENDING)],
            [("target", ASCENDING)],
            [("type", ASCENDING)]
        ],
        "embeddings": [
            [("entity_id", ASCENDING)],
            [("chunk_id", ASCENDING)],
            [("embedding_type", ASCENDING)]
        ],
        "community_reports": [[("level", ASCENDING)], [("community_id", ASCENDING)]]
    }

    for collection_name, index_list in indexes.items():
        collection = db[collection_name]
        for index_keys in index_list:
            try:
                collection.create_index(index_keys)
            except Exception as e:
                logger.warning(f"  Index warning for {collection_name}: {e}")

    logger.info("  Indexes created")


def auto_import_data(mongodb_uri: str, database_name: str) -> bool:
    """
    Import data from JSON files if MongoDB is empty.
    Returns True if import succeeded, False on error.
    """
    data_dir = _get_data_directory()

    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        logger.warning("Database will remain empty. Download data files first.")
        return False

    # Check required files exist
    files = {}
    for collection_name in ALL_COLLECTIONS:
        filepath = data_dir / f"{collection_name}.json"
        if filepath.exists():
            files[collection_name] = filepath
        else:
            logger.warning(f"Missing data file: {filepath}")

    if len(files) < len(REQUIRED_COLLECTIONS):
        logger.error("Missing required data files. Cannot import.")
        return False

    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[database_name]

        logger.info(f"Starting auto-import from {data_dir}")

        # Import each collection
        for collection_name, filepath in files.items():
            _import_collection(db, collection_name, filepath)

        # Create indexes
        _create_indexes(db)

        # Verify
        total = sum(db[name].count_documents({}) for name in ALL_COLLECTIONS)
        logger.info(f"Auto-import complete: {total:,} total documents")

        client.close()
        return True

    except Exception as e:
        logger.error(f"Auto-import failed: {e}")
        return False


# =============================================================================
# Ollama Model Management
# =============================================================================

def is_ollama_model_available(ollama_host: str, model: str) -> bool:
    """
    Check if a specific Ollama model is already pulled.

    Args:
        ollama_host: Ollama API host URL
        model: Model name to check

    Returns:
        True if model is available, False otherwise
    """
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{ollama_host}/api/tags")
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check if model name matches (with or without :latest tag)
            return any(model in name or name.startswith(f"{model}:") for name in model_names)

    except Exception as e:
        logger.warning(f"Could not check Ollama models: {e}")
        return False


def pull_ollama_model(ollama_host: str, model: str) -> bool:
    """
    Pull an Ollama model if not already available.

    Args:
        ollama_host: Ollama API host URL
        model: Model name to pull

    Returns:
        True if model is ready (already existed or pulled successfully)
    """
    # Check if model already exists
    if is_ollama_model_available(ollama_host, model):
        logger.info(f"Ollama model '{model}' is already available")
        return True

    logger.info(f"Pulling Ollama model '{model}'... (this may take several minutes)")

    try:
        # Use streaming pull endpoint for progress updates
        with httpx.Client(timeout=600) as client:  # 10 minute timeout for large models
            response = client.post(
                f"{ollama_host}/api/pull",
                json={"name": model, "stream": True},
                timeout=600
            )
            response.raise_for_status()

            # Process streaming response for progress
            last_status = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")

                        # Log progress updates (but avoid spam)
                        if status != last_status:
                            if "pulling" in status:
                                logger.info(f"  {status}")
                            elif status == "success":
                                logger.info(f"  Model '{model}' pulled successfully")
                            last_status = status

                    except json.JSONDecodeError:
                        pass

            # Verify model is now available
            if is_ollama_model_available(ollama_host, model):
                logger.info(f"Ollama model '{model}' is ready")
                return True
            else:
                logger.error(f"Model '{model}' not available after pull")
                return False

    except httpx.TimeoutException:
        logger.error(f"Timeout pulling model '{model}' - try pulling manually")
        return False
    except Exception as e:
        logger.error(f"Failed to pull Ollama model '{model}': {e}")
        return False


def ensure_ollama_model(ollama_host: Optional[str] = None, model: str = "qwen2.5") -> bool:
    """
    Ensure Ollama model is available, pulling if necessary.

    Args:
        ollama_host: Ollama API host URL (defaults to env var or localhost)
        model: Model name to ensure

    Returns:
        True if model is ready, False otherwise
    """
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # First check if Ollama is reachable
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{host}/api/tags")
            response.raise_for_status()
    except Exception as e:
        logger.warning(f"Ollama not reachable at {host}: {e}")
        return False

    return pull_ollama_model(host, model)
