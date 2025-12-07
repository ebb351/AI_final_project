#!/usr/bin/env python3
"""
MongoDB Export Script

Exports all GraphRAG collections to JSON files for portable deployment.
Export files can be used with docker_import.py to initialize a Docker MongoDB instance.

Collections exported:
- text_chunks
- graph_nodes
- graph_edges
- embeddings
- community_reports
"""
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
from pymongo import MongoClient
from bson import json_util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MongoDBExporter:
    """Export MongoDB collections to JSON files."""

    def __init__(
        self,
        mongodb_uri: str = None,
        database_name: str = None,
        output_dir: str = None
    ):
        """Initialize exporter."""
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.database_name = database_name or os.getenv("MONGODB_DATABASE", "graphrag_course_db")

        # Default output directory
        project_root = Path(__file__).parent.parent
        self.output_dir = Path(output_dir) if output_dir else project_root / "data" / "mongodb_export"

        logger.info(f"Connecting to MongoDB: {self.mongodb_uri}")
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[self.database_name]

        logger.info(f"Database: {self.database_name}")
        logger.info(f"Output directory: {self.output_dir}")

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")

    def export_collection(
        self,
        collection_name: str,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Export a single collection to JSON file."""
        collection = self.db[collection_name]
        count = collection.count_documents({})

        logger.info(f"Exporting {collection_name}: {count:,} documents")

        output_file = self.output_dir / f"{collection_name}.json"

        # Export in batches for memory efficiency
        documents = []
        cursor = collection.find({})

        for doc in cursor:
            # Convert BSON types to JSON-serializable format
            doc_json = json.loads(json_util.dumps(doc))
            documents.append(doc_json)

            if len(documents) % batch_size == 0:
                logger.info(f"  Processed {len(documents):,} / {count:,}")

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=None)  # No indent for smaller file size

        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"  Exported to {output_file.name} ({file_size:.1f} MB)")

        return {
            "collection": collection_name,
            "documents": count,
            "file": str(output_file),
            "size_mb": round(file_size, 1)
        }

    def export_all(self) -> Dict[str, Any]:
        """Export all collections."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        collections = [
            "text_chunks",
            "graph_nodes",
            "graph_edges",
            "embeddings",
            "community_reports"
        ]

        results = []
        total_size = 0

        for collection_name in collections:
            try:
                result = self.export_collection(collection_name)
                results.append(result)
                total_size += result["size_mb"]
            except Exception as e:
                logger.error(f"Failed to export {collection_name}: {e}")
                results.append({
                    "collection": collection_name,
                    "error": str(e)
                })

        # Write metadata file
        metadata = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "database": self.database_name,
            "collections": results,
            "total_size_mb": round(total_size, 1)
        }

        metadata_file = self.output_dir / "export_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\nMetadata written to {metadata_file}")

        return metadata


def main():
    """Run export."""
    logger.info("=" * 60)
    logger.info("MONGODB EXPORT")
    logger.info("=" * 60)

    exporter = MongoDBExporter()

    try:
        metadata = exporter.export_all()

        logger.info("\n" + "=" * 60)
        logger.info("EXPORT COMPLETE")
        logger.info("=" * 60)

        for result in metadata["collections"]:
            if "error" in result:
                logger.error(f"  Failed {result['collection']}: {result['error']}")
            else:
                logger.info(f"  {result['collection']}: {result['documents']:,} docs ({result['size_mb']} MB)")

        logger.info(f"\nTotal size: {metadata['total_size_mb']} MB")
        logger.info(f"Output directory: {exporter.output_dir}")

    finally:
        exporter.close()


if __name__ == "__main__":
    main()
