#!/usr/bin/env python3
"""
Docker MongoDB Import Script

Imports exported GraphRAG data into a Docker MongoDB instance.
Designed to work with docker-compose setup where MongoDB runs as 'mongodb' service.

Usage:
    # With docker-compose running:
    python scripts/docker_import.py

    # Specify custom MongoDB URI:
    python scripts/docker_import.py --uri mongodb://localhost:27017/

    # Specify custom data directory:
    python scripts/docker_import.py --data-dir /path/to/mongodb_export
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError, ServerSelectionTimeoutError
from bson import json_util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DockerMongoImporter:
    """Import data into Docker MongoDB instance."""

    def __init__(
        self,
        mongodb_uri: str = None,
        database_name: str = None,
        data_dir: str = None
    ):
        """Initialize importer."""
        # Default to Docker MongoDB service
        self.mongodb_uri = mongodb_uri or os.getenv(
            "MONGODB_URI",
            "mongodb://localhost:27017/"
        )
        self.database_name = database_name or os.getenv(
            "MONGODB_DATABASE",
            "graphrag_course_db"
        )

        # Default data directory
        project_root = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else project_root / "data" / "mongodb_export"

        logger.info(f"MongoDB URI: {self.mongodb_uri}")
        logger.info(f"Database: {self.database_name}")
        logger.info(f"Data directory: {self.data_dir}")

        # Connect to MongoDB
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB")
        except ServerSelectionTimeoutError:
            logger.error("Cannot connect to MongoDB. Is the container running?")
            logger.error("   Try: docker-compose up -d mongodb")
            sys.exit(1)

        self.db = self.client[self.database_name]

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")

    def check_data_files(self) -> Dict[str, Path]:
        """Check that all required data files exist."""
        required_files = [
            "text_chunks.json",
            "graph_nodes.json",
            "graph_edges.json",
            "embeddings.json",
            "community_reports.json"
        ]

        files = {}
        missing = []

        for filename in required_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                files[filename.replace(".json", "")] = filepath
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"  Found: {filename} ({size_mb:.1f} MB)")
            else:
                missing.append(filename)
                logger.warning(f"  Missing: {filename}")

        if missing:
            logger.error(f"\nMissing files: {missing}")
            logger.error("Run 'python scripts/export_mongodb.py' first to export data")
            sys.exit(1)

        return files

    def import_collection(
        self,
        collection_name: str,
        filepath: Path,
        batch_size: int = 500
    ) -> Dict[str, Any]:
        """Import a single collection from JSON file."""
        logger.info(f"\nImporting {collection_name}...")

        # Load JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        total = len(documents)
        logger.info(f"  Loaded {total:,} documents from {filepath.name}")

        # Get collection
        collection = self.db[collection_name]

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

            # Convert back from JSON to BSON-compatible format
            batch_docs = []
            for doc in batch:
                # Parse any BSON-specific types
                parsed_doc = json_util.loads(json.dumps(doc))
                batch_docs.append(parsed_doc)

            try:
                result = collection.insert_many(batch_docs, ordered=False)
                imported += len(result.inserted_ids)
            except BulkWriteError as e:
                imported += e.details.get("nInserted", 0)
                errors += len(e.details.get("writeErrors", []))

            if (i + batch_size) % 2000 == 0 or i + batch_size >= total:
                logger.info(f"  Progress: {min(i + batch_size, total):,} / {total:,}")

        logger.info(f"  Imported {imported:,} documents ({errors} errors)")

        return {
            "collection": collection_name,
            "total": total,
            "imported": imported,
            "errors": errors
        }

    def create_indexes(self):
        """Create indexes for query performance."""
        logger.info("\nCreating indexes...")

        indexes = {
            "text_chunks": [
                [("source_id", ASCENDING)],
                [("source_type", ASCENDING)]
            ],
            "graph_nodes": [
                [("type", ASCENDING)],
                [("community_id", ASCENDING)]
            ],
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
            "community_reports": [
                [("level", ASCENDING)],
                [("community_id", ASCENDING)]
            ]
        }

        for collection_name, index_list in indexes.items():
            collection = self.db[collection_name]
            for index_keys in index_list:
                try:
                    collection.create_index(index_keys)
                except Exception as e:
                    logger.warning(f"  Index warning for {collection_name}: {e}")

        logger.info("  Indexes created")

    def verify_import(self) -> Dict[str, int]:
        """Verify imported data counts."""
        logger.info("\nVerifying import...")

        counts = {}
        collections = [
            "text_chunks",
            "graph_nodes",
            "graph_edges",
            "embeddings",
            "community_reports"
        ]

        for name in collections:
            count = self.db[name].count_documents({})
            counts[name] = count
            logger.info(f"  {name}: {count:,} documents")

        return counts

    def import_all(self) -> Dict[str, Any]:
        """Import all collections."""
        logger.info("\n" + "=" * 60)
        logger.info("Checking data files...")
        logger.info("=" * 60)

        files = self.check_data_files()

        logger.info("\n" + "=" * 60)
        logger.info("Importing collections...")
        logger.info("=" * 60)

        results = []
        for collection_name, filepath in files.items():
            result = self.import_collection(collection_name, filepath)
            results.append(result)

        # Create indexes
        self.create_indexes()

        # Verify
        counts = self.verify_import()

        return {
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "database": self.database_name,
            "results": results,
            "final_counts": counts
        }


def main():
    """Run import."""
    parser = argparse.ArgumentParser(
        description="Import GraphRAG data into Docker MongoDB"
    )
    parser.add_argument(
        "--uri",
        help="MongoDB URI (default: mongodb://localhost:27017/)"
    )
    parser.add_argument(
        "--database",
        help="Database name (default: graphrag_course_db)"
    )
    parser.add_argument(
        "--data-dir",
        help="Directory containing exported JSON files"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DOCKER MONGODB IMPORT")
    logger.info("=" * 60)

    importer = DockerMongoImporter(
        mongodb_uri=args.uri,
        database_name=args.database,
        data_dir=args.data_dir
    )

    try:
        results = importer.import_all()

        logger.info("\n" + "=" * 60)
        logger.info("IMPORT COMPLETE")
        logger.info("=" * 60)

        total_docs = sum(results["final_counts"].values())
        logger.info(f"Total documents imported: {total_docs:,}")
        logger.info("\nYou can now test the API:")
        logger.info("  curl http://localhost:8000/health")

    finally:
        importer.close()


if __name__ == "__main__":
    main()
