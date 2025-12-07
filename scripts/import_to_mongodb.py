#!/usr/bin/env python3
"""
MongoDB Import Script

Imports all graph data into MongoDB:
1. Text chunks (from chunks_filtered.json)
2. Graph nodes/entities (from entities_normalized.json with communities)
3. Graph edges/relationships (from entities_normalized.json)
4. Embeddings (from embeddings.json)
5. Community reports (from community_reports.json)

This script should be run after:
- normalize_entities.py
- generate_embeddings.py
- community_detection.py
"""
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.errors import BulkWriteError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MongoDBImporter:
    """Import GraphRAG data into MongoDB."""

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        database_name: Optional[str] = None
    ):
        """Initialize MongoDB connection."""
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.database_name = database_name or os.getenv("MONGODB_DATABASE", "graphrag_course_db")

        logger.info(f"Connecting to MongoDB: {self.mongodb_uri}")
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[self.database_name]

        # Collection references
        self.chunks = self.db["text_chunks"]
        self.nodes = self.db["graph_nodes"]
        self.edges = self.db["graph_edges"]
        self.embeddings_col = self.db["embeddings"]
        self.community_reports = self.db["community_reports"]

        logger.info(f"Connected to database: {self.database_name}")

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")

    def clear_collections(self, confirm: bool = False):
        """Clear all collections (use with caution!)."""
        if not confirm:
            logger.warning("clear_collections called without confirmation, skipping")
            return

        logger.warning("Clearing all collections...")
        self.chunks.delete_many({})
        self.nodes.delete_many({})
        self.edges.delete_many({})
        self.embeddings_col.delete_many({})
        self.community_reports.delete_many({})
        logger.info("All collections cleared")

    def import_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Import text chunks into text_chunks collection."""
        logger.info(f"Importing {len(chunks)} chunks...")

        operations = []
        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            if not chunk_id:
                continue

            doc = {
                "_id": chunk_id,
                "text": chunk.get("text", ""),
                "source_id": chunk.get("source_id", ""),
                "source_url": chunk.get("source_url", ""),
                "source_title": chunk.get("source_title", ""),
                "source_type": chunk.get("source_type", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "token_count": chunk.get("token_count", 0),
                "processed": True,
                "imported_at": datetime.now(timezone.utc)
            }

            operations.append(
                UpdateOne(
                    {"_id": chunk_id},
                    {"$set": doc},
                    upsert=True
                )
            )

        # Bulk write
        inserted = 0
        errors = 0
        if operations:
            try:
                result = self.chunks.bulk_write(operations, ordered=False)
                inserted = result.upserted_count + result.modified_count
                logger.info(f"Chunks: {inserted} upserted/modified")
            except BulkWriteError as e:
                inserted = e.details.get("nInserted", 0) + e.details.get("nModified", 0)
                errors = len(e.details.get("writeErrors", []))
                logger.error(f"Chunk import errors: {errors}")

        return {"imported": inserted, "errors": errors}

    def import_entities(
        self,
        entities: List[Dict[str, Any]],
        entity_sources: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, int]:
        """Import entities into graph_nodes collection."""
        logger.info(f"Importing {len(entities)} entities...")

        operations = []
        for entity in entities:
            name = entity.get("name", "")
            if not name:
                continue

            # Get source chunks
            source_chunks = []
            if entity_sources and name in entity_sources:
                source_chunks = entity_sources[name]

            doc = {
                "_id": name,
                "type": entity.get("type", "CONCEPT"),
                "description": entity.get("description", ""),
                "community_id": entity.get("community_id"),
                "source_chunks": source_chunks,
                "imported_at": datetime.now(timezone.utc)
            }

            operations.append(
                UpdateOne(
                    {"_id": name},
                    {"$set": doc},
                    upsert=True
                )
            )

        # Bulk write
        inserted = 0
        errors = 0
        if operations:
            try:
                result = self.nodes.bulk_write(operations, ordered=False)
                inserted = result.upserted_count + result.modified_count
                logger.info(f"Entities: {inserted} upserted/modified")
            except BulkWriteError as e:
                inserted = e.details.get("nInserted", 0) + e.details.get("nModified", 0)
                errors = len(e.details.get("writeErrors", []))
                logger.error(f"Entity import errors: {errors}")

        return {"imported": inserted, "errors": errors}

    def import_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Import relationships into graph_edges collection."""
        logger.info(f"Importing {len(relationships)} relationships...")

        # First, get set of valid entity names
        valid_entities = set(doc["_id"] for doc in self.nodes.find({}, {"_id": 1}))
        logger.info(f"Found {len(valid_entities)} valid entities for validation")

        operations = []
        skipped = 0

        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "NEAR_TRANSFER")

            if not source or not target:
                skipped += 1
                continue

            # Validate entities exist
            if source not in valid_entities or target not in valid_entities:
                skipped += 1
                continue

            doc = {
                "source": source,
                "target": target,
                "type": rel_type,
                "description": rel.get("description", ""),
                "weight": self._get_relationship_weight(rel_type),
                "source_chunk": rel.get("source_chunk", ""),
                "imported_at": datetime.now(timezone.utc)
            }

            # Use source+target+type as unique key
            operations.append(
                UpdateOne(
                    {"source": source, "target": target, "type": rel_type},
                    {"$set": doc},
                    upsert=True
                )
            )

        logger.info(f"Skipped {skipped} relationships (invalid entities)")

        # Bulk write
        inserted = 0
        errors = 0
        if operations:
            try:
                result = self.edges.bulk_write(operations, ordered=False)
                inserted = result.upserted_count + result.modified_count
                logger.info(f"Relationships: {inserted} upserted/modified")
            except BulkWriteError as e:
                inserted = e.details.get("nInserted", 0) + e.details.get("nModified", 0)
                errors = len(e.details.get("writeErrors", []))
                logger.error(f"Relationship import errors: {errors}")

        return {"imported": inserted, "errors": errors, "skipped": skipped}

    def _get_relationship_weight(self, rel_type: str) -> float:
        """Get edge weight for relationship type."""
        weights = {
            "PREREQUISITE_FOR": 3.0,
            "COMPONENT_OF": 2.5,
            "IS_A": 2.0,
            "PART_OF": 2.0,
            "EXPLAINS": 2.0,
            "SOLVES": 1.5,
            "EXEMPLIFIES": 1.5,
            "APPLIES_TO": 1.0,
            "NEAR_TRANSFER": 0.8,
            "CONTRASTS_WITH": 0.5,
        }
        return weights.get(rel_type, 1.0)

    def import_embeddings(
        self,
        entity_embeddings: Dict[str, List[float]],
        chunk_embeddings: Dict[str, List[float]],
        model_name: str = "nomic-embed-text-v1.5"
    ) -> Dict[str, int]:
        """Import embeddings into embeddings collection."""
        logger.info(f"Importing {len(entity_embeddings)} entity embeddings and {len(chunk_embeddings)} chunk embeddings...")

        operations = []

        # Entity embeddings
        for entity_name, embedding in entity_embeddings.items():
            doc = {
                "_id": f"entity:{entity_name}",
                "entity_id": entity_name,
                "chunk_id": None,
                "embedding_type": "entity",
                "embedding": embedding,
                "model": model_name,
                "dimension": len(embedding),
                "imported_at": datetime.now(timezone.utc)
            }
            operations.append(
                UpdateOne(
                    {"_id": f"entity:{entity_name}"},
                    {"$set": doc},
                    upsert=True
                )
            )

        # Chunk embeddings
        for chunk_id, embedding in chunk_embeddings.items():
            doc = {
                "_id": f"chunk:{chunk_id}",
                "entity_id": None,
                "chunk_id": chunk_id,
                "embedding_type": "chunk",
                "embedding": embedding,
                "model": model_name,
                "dimension": len(embedding),
                "imported_at": datetime.now(timezone.utc)
            }
            operations.append(
                UpdateOne(
                    {"_id": f"chunk:{chunk_id}"},
                    {"$set": doc},
                    upsert=True
                )
            )

        # Bulk write in batches (embeddings are large)
        inserted = 0
        errors = 0
        batch_size = 1000

        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            try:
                result = self.embeddings_col.bulk_write(batch, ordered=False)
                inserted += result.upserted_count + result.modified_count
            except BulkWriteError as e:
                inserted += e.details.get("nInserted", 0) + e.details.get("nModified", 0)
                errors += len(e.details.get("writeErrors", []))

            logger.info(f"Embeddings batch {i // batch_size + 1}: {inserted} total")

        logger.info(f"Embeddings: {inserted} upserted/modified, {errors} errors")
        return {"imported": inserted, "errors": errors}

    def import_community_reports(
        self,
        communities: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Import community reports into community_reports collection."""
        logger.info(f"Importing {len(communities)} community reports...")

        operations = []
        for community in communities:
            community_id = community.get("community_id")
            if community_id is None:
                continue

            doc = {
                "_id": f"community:{community_id}",
                "community_id": str(community_id),  # Convert to string for schema validation
                "level": community.get("level", 0),
                "title": community.get("title", ""),
                "summary": community.get("summary", ""),
                "entity_count": community.get("entity_count", 0),
                "relationship_count": community.get("relationship_count", 0),
                "entities": community.get("entities", []),
                "key_entities": community.get("key_entities", []),
                "generated_with_llm": community.get("generated_with_llm", False),
                "imported_at": datetime.now(timezone.utc)
            }

            operations.append(
                UpdateOne(
                    {"_id": f"community:{community_id}"},
                    {"$set": doc},
                    upsert=True
                )
            )

        # Bulk write
        inserted = 0
        errors = 0
        if operations:
            try:
                result = self.community_reports.bulk_write(operations, ordered=False)
                inserted = result.upserted_count + result.modified_count
                logger.info(f"Community reports: {inserted} upserted/modified")
            except BulkWriteError as e:
                inserted = e.details.get("nInserted", 0) + e.details.get("nModified", 0)
                errors = len(e.details.get("writeErrors", []))
                logger.error(f"Community report import errors: {errors}")

        return {"imported": inserted, "errors": errors}

    def create_indexes(self):
        """Create indexes for query performance."""
        logger.info("Creating indexes...")

        def safe_create_index(collection, keys, **kwargs):
            """Create index, ignoring if already exists."""
            try:
                collection.create_index(keys, **kwargs)
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"Index already exists: {keys}")
                else:
                    logger.warning(f"Index creation warning: {e}")

        # text_chunks indexes
        safe_create_index(self.chunks, [("source_id", ASCENDING)])
        safe_create_index(self.chunks, [("source_type", ASCENDING)])

        # graph_nodes indexes
        safe_create_index(self.nodes, [("type", ASCENDING)])
        safe_create_index(self.nodes, [("community_id", ASCENDING)])

        # graph_edges indexes
        safe_create_index(self.edges, [("source", ASCENDING), ("target", ASCENDING)])
        safe_create_index(self.edges, [("source", ASCENDING)])
        safe_create_index(self.edges, [("target", ASCENDING)])
        safe_create_index(self.edges, [("type", ASCENDING)])

        # embeddings indexes
        safe_create_index(self.embeddings_col, [("entity_id", ASCENDING)])
        safe_create_index(self.embeddings_col, [("chunk_id", ASCENDING)])
        safe_create_index(self.embeddings_col, [("embedding_type", ASCENDING)])

        # community_reports indexes
        safe_create_index(self.community_reports, [("level", ASCENDING)])
        safe_create_index(self.community_reports, [("community_id", ASCENDING)])

        logger.info("Indexes created")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "text_chunks": self.chunks.count_documents({}),
            "graph_nodes": self.nodes.count_documents({}),
            "graph_edges": self.edges.count_documents({}),
            "embeddings": self.embeddings_col.count_documents({}),
            "community_reports": self.community_reports.count_documents({})
        }

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        logger.info("=" * 50)
        logger.info("MONGODB COLLECTION STATISTICS")
        logger.info("=" * 50)
        for collection, count in stats.items():
            logger.info(f"  {collection}: {count:,}")
        logger.info("=" * 50)


def main():
    """Main import pipeline."""
    project_root = Path(__file__).parent.parent

    # Input files
    chunks_file = project_root / "data" / "chunks_filtered.json"
    entities_file = project_root / "data" / "entities_normalized.json"
    embeddings_file = project_root / "data" / "embeddings.json"
    communities_file = project_root / "data" / "community_reports.json"
    graph_file = project_root / "data" / "graph_with_communities.json"

    logger.info("=" * 80)
    logger.info("MONGODB IMPORT PIPELINE")
    logger.info("=" * 80)

    # Check which files exist
    files_status = {
        "chunks": chunks_file.exists(),
        "entities": entities_file.exists(),
        "embeddings": embeddings_file.exists(),
        "communities": communities_file.exists(),
        "graph_with_communities": graph_file.exists()
    }

    for name, exists in files_status.items():
        status = "found" if exists else "missing"
        logger.info(f"  {name}: {status}")

    if not all([files_status["chunks"], files_status["entities"]]):
        logger.error("Required files missing! Run normalization first.")
        return

    # Initialize importer
    importer = MongoDBImporter()

    try:
        # Clear existing data (optional - uncomment if needed)
        # importer.clear_collections(confirm=True)

        # Load data files
        logger.info("\nLoading data files...")

        # Load chunks
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks")

        # Load entities (prefer graph with communities if available)
        if graph_file.exists():
            logger.info("Using graph_with_communities.json for entity data")
            with open(graph_file, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            entities = graph_data.get("entities", [])
            relationships = graph_data.get("relationships", [])
        else:
            with open(entities_file, "r", encoding="utf-8") as f:
                normalized_data = json.load(f)
            entities = normalized_data.get("entities", [])
            relationships = normalized_data.get("relationships", [])

        logger.info(f"Loaded {len(entities)} entities and {len(relationships)} relationships")

        # Load entity sources (for source_chunks field)
        with open(entities_file, "r", encoding="utf-8") as f:
            normalized_data = json.load(f)
        entity_sources = normalized_data.get("entity_sources", {})

        # Import chunks
        logger.info("\n" + "-" * 40)
        chunk_stats = importer.import_chunks(chunks)

        # Import entities
        logger.info("\n" + "-" * 40)
        entity_stats = importer.import_entities(entities, entity_sources)

        # Import relationships
        logger.info("\n" + "-" * 40)
        rel_stats = importer.import_relationships(relationships)

        # Import embeddings (if available)
        embedding_stats = {"imported": 0, "errors": 0}
        if embeddings_file.exists():
            logger.info("\n" + "-" * 40)
            with open(embeddings_file, "r", encoding="utf-8") as f:
                embeddings_data = json.load(f)

            entity_embeddings = embeddings_data.get("entity_embeddings", {})
            chunk_embeddings = embeddings_data.get("chunk_embeddings", {})

            embedding_stats = importer.import_embeddings(
                entity_embeddings,
                chunk_embeddings
            )
        else:
            logger.warning("Embeddings file not found, skipping embedding import")

        # Import community reports (if available)
        community_stats = {"imported": 0, "errors": 0}
        if communities_file.exists():
            logger.info("\n" + "-" * 40)
            with open(communities_file, "r", encoding="utf-8") as f:
                communities_data = json.load(f)

            communities = communities_data.get("communities", [])
            community_stats = importer.import_community_reports(communities)
        else:
            logger.warning("Community reports file not found, skipping community import")

        # Create indexes
        logger.info("\n" + "-" * 40)
        importer.create_indexes()

        # Print final stats
        logger.info("\n")
        importer.print_stats()

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("IMPORT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Chunks: {chunk_stats['imported']} imported, {chunk_stats['errors']} errors")
        logger.info(f"Entities: {entity_stats['imported']} imported, {entity_stats['errors']} errors")
        logger.info(f"Relationships: {rel_stats['imported']} imported, {rel_stats.get('skipped', 0)} skipped, {rel_stats['errors']} errors")
        logger.info(f"Embeddings: {embedding_stats['imported']} imported, {embedding_stats['errors']} errors")
        logger.info(f"Communities: {community_stats['imported']} imported, {community_stats['errors']} errors")
        logger.info("=" * 80)

    finally:
        importer.close()


if __name__ == "__main__":
    main()
