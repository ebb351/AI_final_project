"""
MongoDB Storage Adapters

This module provides functions to store extracted entities, relationships,
text chunks, and embeddings in MongoDB with proper deduplication and validation.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient, UpdateOne, InsertOne
from pymongo.errors import BulkWriteError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphRAGStorage:
    """Storage adapter for GraphRAG MongoDB operations."""

    def __init__(self, mongodb_uri: Optional[str] = None, database_name: Optional[str] = None):
        """
        Initialize MongoDB connection.

        Args:
            mongodb_uri: MongoDB connection URI (defaults to env var or localhost)
            database_name: Database name (defaults to env var or graphrag_course_db)
        """
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.database_name = database_name or os.getenv("MONGODB_DATABASE", "graphrag_course_db")

        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[self.database_name]

        # Collection references
        self.nodes = self.db["graph_nodes"]
        self.edges = self.db["graph_edges"]
        self.chunks = self.db["text_chunks"]
        self.embeddings_col = self.db["embeddings"]
        self.community_reports = self.db["community_reports"]

        logger.info(f"Connected to MongoDB: {self.database_name}")

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")

    def store_entities(
        self,
        entities: List[Dict[str, Any]],
        source_chunk_id: Optional[str] = None,
        merge_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Store entities in graph_nodes collection with deduplication.

        Entities with the same name are merged, combining their source_chunks
        and descriptions.

        Args:
            entities: List of entity dicts with 'name', 'type', 'description'
            source_chunk_id: Optional chunk ID to add to source_chunks
            merge_duplicates: If True, merge entities with same name

        Returns:
            Dict with 'inserted', 'updated', 'errors' counts
        """
        if not entities:
            return {"inserted": 0, "updated": 0, "errors": 0}

        logger.info(f"Storing {len(entities)} entities...")

        inserted = 0
        updated = 0
        errors = 0

        for entity in entities:
            try:
                # Validate required fields
                if not entity.get("name") or not entity.get("type"):
                    logger.warning(f"Skipping entity without name or type: {entity}")
                    errors += 1
                    continue

                name = entity["name"]
                entity_type = entity["type"]
                description = entity.get("description", "")

                # Build source_chunks list
                source_chunks = []
                if source_chunk_id:
                    source_chunks.append(source_chunk_id)

                if merge_duplicates:
                    # Try to find existing entity by _id (which is the name)
                    existing = self.nodes.find_one({"_id": name})

                    if existing:
                        # Update existing entity
                        update_ops = {
                            "$addToSet": {},
                            "$set": {"updated_at": datetime.utcnow()}
                        }

                        # Add source chunk if provided
                        if source_chunk_id:
                            update_ops["$addToSet"]["source_chunks"] = source_chunk_id

                        # Append description if different
                        if description and description != existing.get("description", ""):
                            update_ops["$addToSet"]["descriptions"] = description

                        self.nodes.update_one(
                            {"_id": name},
                            update_ops
                        )
                        updated += 1
                    else:
                        # Insert new entity with name as _id
                        node_doc = {
                            "_id": name,  # Use entity name as _id
                            "type": entity_type,
                            "description": description,
                            "descriptions": [description] if description else [],
                            "aliases": entity.get("aliases", []),
                            "source_chunks": source_chunks,
                            "attributes": entity.get("attributes", {}),
                            "difficulty_level": entity.get("difficulty_level"),
                            "created_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                        self.nodes.insert_one(node_doc)
                        inserted += 1
                else:
                    # Insert without checking for duplicates (use name as _id)
                    node_doc = {
                        "_id": name,  # Use entity name as _id
                        "type": entity_type,
                        "description": description,
                        "descriptions": [description] if description else [],
                        "aliases": entity.get("aliases", []),
                        "source_chunks": source_chunks,
                        "attributes": entity.get("attributes", {}),
                        "difficulty_level": entity.get("difficulty_level"),
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    self.nodes.insert_one(node_doc)
                    inserted += 1

            except Exception as e:
                logger.error(f"Error storing entity {entity.get('name')}: {e}")
                errors += 1

        logger.info(f"Entities: inserted={inserted}, updated={updated}, errors={errors}")
        return {"inserted": inserted, "updated": updated, "errors": errors}

    def store_relationships(
        self,
        relationships: List[Dict[str, Any]],
        source_chunk_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store relationships in graph_edges collection.

        Args:
            relationships: List of relationship dicts with 'source', 'target', 'type'
            source_chunk_id: Optional chunk ID to add to source_chunks

        Returns:
            Dict with 'inserted', 'errors' counts
        """
        if not relationships:
            return {"inserted": 0, "errors": 0}

        logger.info(f"Storing {len(relationships)} relationships...")

        inserted = 0
        errors = 0

        for rel in relationships:
            try:
                # Validate required fields
                if not rel.get("source") or not rel.get("target") or not rel.get("type"):
                    logger.warning(f"Skipping relationship without source/target/type: {rel}")
                    errors += 1
                    continue

                # Check that source and target entities exist (using _id which is the name)
                source_exists = self.nodes.find_one({"_id": rel["source"]})
                target_exists = self.nodes.find_one({"_id": rel["target"]})

                if not source_exists:
                    logger.warning(f"Source entity not found: {rel['source']}")
                    errors += 1
                    continue

                if not target_exists:
                    logger.warning(f"Target entity not found: {rel['target']}")
                    errors += 1
                    continue

                # Build edge document
                edge_doc = {
                    "source": rel["source"],
                    "target": rel["target"],
                    "type": rel["type"],
                    "description": rel.get("description", ""),
                    "weight": rel.get("weight", 1.0),
                    "source_chunks": [source_chunk_id] if source_chunk_id else [],
                    "attributes": rel.get("attributes", {}),
                    "created_at": datetime.utcnow()
                }

                self.edges.insert_one(edge_doc)
                inserted += 1

            except Exception as e:
                logger.error(f"Error storing relationship {rel.get('source')}->{rel.get('target')}: {e}")
                errors += 1

        logger.info(f"Relationships: inserted={inserted}, errors={errors}")
        return {"inserted": inserted, "errors": errors}

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store text chunks in text_chunks collection.

        Args:
            chunks: List of chunk dicts with 'id', 'text', 'source_id', etc.

        Returns:
            Dict with 'inserted', 'errors' counts
        """
        if not chunks:
            return {"inserted": 0, "errors": 0}

        logger.info(f"Storing {len(chunks)} chunks...")

        operations = []
        for chunk in chunks:
            try:
                # Validate required fields
                if not chunk.get("id"):
                    logger.warning(f"Skipping chunk without id: {chunk}")
                    continue

                chunk_doc = {
                    "chunk_id": chunk["id"],
                    "text": chunk.get("text", ""),
                    "source_id": chunk.get("source_id", ""),
                    "source_url": chunk.get("source_url", ""),
                    "source_title": chunk.get("source_title", ""),
                    "source_type": chunk.get("source_type", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 1),
                    "token_count": chunk.get("token_count", 0),
                    "metadata": chunk.get("metadata", {}),
                    "processed": False,
                    "created_at": datetime.utcnow()
                }

                # Upsert by chunk_id
                operations.append(
                    UpdateOne(
                        {"chunk_id": chunk["id"]},
                        {"$setOnInsert": chunk_doc},
                        upsert=True
                    )
                )

            except Exception as e:
                logger.error(f"Error preparing chunk {chunk.get('id')}: {e}")

        # Bulk write
        inserted = 0
        errors = 0
        if operations:
            try:
                result = self.chunks.bulk_write(operations, ordered=False)
                inserted = result.upserted_count
                logger.info(f"Chunks: inserted={inserted}, errors={errors}")
            except BulkWriteError as e:
                inserted = e.details.get("nInserted", 0)
                errors = len(e.details.get("writeErrors", []))
                logger.error(f"Bulk write errors: {errors}")

        return {"inserted": inserted, "errors": errors}

    def store_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        model_name: str = "nomic-embed-text-v1.5"
    ) -> Dict[str, Any]:
        """
        Store embeddings in embeddings collection.

        Args:
            embeddings: List of embedding dicts with 'chunk_id', 'embedding' (vector)
            model_name: Name of the embedding model used

        Returns:
            Dict with 'inserted', 'errors' counts
        """
        if not embeddings:
            return {"inserted": 0, "errors": 0}

        logger.info(f"Storing {len(embeddings)} embeddings...")

        operations = []
        for emb in embeddings:
            try:
                # Validate required fields
                if not emb.get("chunk_id") or not emb.get("embedding"):
                    logger.warning(f"Skipping embedding without chunk_id or embedding: {emb}")
                    continue

                emb_doc = {
                    "chunk_id": emb["chunk_id"],
                    "embedding": emb["embedding"],
                    "model": model_name,
                    "dimension": len(emb["embedding"]),
                    "created_at": datetime.utcnow()
                }

                # Upsert by chunk_id and model
                operations.append(
                    UpdateOne(
                        {"chunk_id": emb["chunk_id"], "model": model_name},
                        {"$setOnInsert": emb_doc},
                        upsert=True
                    )
                )

            except Exception as e:
                logger.error(f"Error preparing embedding for {emb.get('chunk_id')}: {e}")

        # Bulk write
        inserted = 0
        errors = 0
        if operations:
            try:
                result = self.embeddings_col.bulk_write(operations, ordered=False)
                inserted = result.upserted_count
                logger.info(f"Embeddings: inserted={inserted}, errors={errors}")
            except BulkWriteError as e:
                inserted = e.details.get("nInserted", 0)
                errors = len(e.details.get("writeErrors", []))
                logger.error(f"Bulk write errors: {errors}")

        return {"inserted": inserted, "errors": errors}

    def store_extraction_results(
        self,
        extraction_results: List[Dict[str, Any]],
        store_chunks_data: bool = False
    ) -> Dict[str, Any]:
        """
        Store complete extraction results (entities + relationships) from Modal output.

        Args:
            extraction_results: List of extraction result dicts from Modal
            store_chunks_data: If True, also store the chunk text data

        Returns:
            Dict with summary statistics
        """
        logger.info(f"Storing extraction results from {len(extraction_results)} chunks...")

        total_stats = {
            "chunks_processed": 0,
            "entities_inserted": 0,
            "entities_updated": 0,
            "relationships_inserted": 0,
            "errors": 0
        }

        for result in extraction_results:
            try:
                chunk_id = result.get("chunk_id")

                # Store entities
                entities = result.get("entities", [])
                if entities:
                    entity_stats = self.store_entities(
                        entities,
                        source_chunk_id=chunk_id,
                        merge_duplicates=True
                    )
                    total_stats["entities_inserted"] += entity_stats["inserted"]
                    total_stats["entities_updated"] += entity_stats["updated"]
                    total_stats["errors"] += entity_stats["errors"]

                # Store relationships
                relationships = result.get("relationships", [])
                if relationships:
                    rel_stats = self.store_relationships(
                        relationships,
                        source_chunk_id=chunk_id
                    )
                    total_stats["relationships_inserted"] += rel_stats["inserted"]
                    total_stats["errors"] += rel_stats["errors"]

                total_stats["chunks_processed"] += 1

                # Mark chunk as processed
                if chunk_id:
                    self.chunks.update_one(
                        {"chunk_id": chunk_id},
                        {"$set": {"processed": True, "processed_at": datetime.utcnow()}}
                    )

            except Exception as e:
                logger.error(f"Error storing extraction result: {e}")
                total_stats["errors"] += 1

        logger.info(f"Storage complete: {total_stats}")
        return total_stats

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dict with counts and basic statistics
        """
        stats = {
            "nodes": self.nodes.count_documents({}),
            "edges": self.edges.count_documents({}),
            "chunks": self.chunks.count_documents({}),
            "embeddings": self.embeddings_col.count_documents({}),
            "community_reports": self.community_reports.count_documents({}),
            "nodes_by_type": {},
            "edges_by_type": {},
            "processed_chunks": self.chunks.count_documents({"processed": True})
        }

        # Node type breakdown
        node_types = self.nodes.aggregate([
            {"$group": {"_id": "$type", "count": {"$sum": 1}}}
        ])
        stats["nodes_by_type"] = {item["_id"]: item["count"] for item in node_types}

        # Edge type breakdown
        edge_types = self.edges.aggregate([
            {"$group": {"_id": "$type", "count": {"$sum": 1}}}
        ])
        stats["edges_by_type"] = {item["_id"]: item["count"] for item in edge_types}

        return stats

    def print_graph_stats(self):
        """Print formatted graph statistics."""
        stats = self.get_graph_stats()

        print("=" * 80)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("=" * 80)
        print(f"Total nodes (entities): {stats['nodes']}")
        print(f"Total edges (relationships): {stats['edges']}")
        print(f"Total text chunks: {stats['chunks']}")
        print(f"Processed chunks: {stats['processed_chunks']}")
        print(f"Total embeddings: {stats['embeddings']}")
        print(f"Community reports: {stats['community_reports']}")

        if stats['nodes_by_type']:
            print("\nNodes by type:")
            for node_type, count in sorted(stats['nodes_by_type'].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {node_type}: {count}")

        if stats['edges_by_type']:
            print("\nEdges by type:")
            for edge_type, count in sorted(stats['edges_by_type'].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {edge_type}: {count}")

        print("=" * 80)


def main():
    """Test storage functions with sample data."""
    storage = GraphRAGStorage()

    try:
        # Sample test data
        test_entities = [
            {
                "name": "Neural Network",
                "type": "CONCEPT",
                "description": "A computational model inspired by biological neural networks"
            },
            {
                "name": "Backpropagation",
                "type": "ALGORITHM",
                "description": "Algorithm for training neural networks using gradient descent"
            }
        ]

        test_relationships = [
            {
                "source": "Backpropagation",
                "target": "Neural Network",
                "type": "APPLIES_TO",
                "description": "Backpropagation is used to train neural networks"
            }
        ]

        # Test storage
        print("\nTesting entity storage...")
        entity_stats = storage.store_entities(test_entities, source_chunk_id="test_chunk_1")
        print(f"Result: {entity_stats}")

        print("\nTesting relationship storage...")
        rel_stats = storage.store_relationships(test_relationships, source_chunk_id="test_chunk_1")
        print(f"Result: {rel_stats}")

        print("\nGraph statistics:")
        storage.print_graph_stats()

    finally:
        storage.close()


if __name__ == "__main__":
    main()
