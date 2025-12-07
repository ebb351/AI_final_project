#!/usr/bin/env python3
"""
Query Processor for GraphRAG

Processes user queries by:
1. Generating query embeddings
2. Finding similar entities via cosine similarity
3. Expanding subgraph with graph traversal
"""
import logging
import os
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pymongo import MongoClient

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Process queries and retrieve relevant entities."""

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        """Initialize query processor."""
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.database_name = database_name or os.getenv("MONGODB_DATABASE", "graphrag_course_db")
        self.model_name = model_name
        self._model = None
        self._client = None
        self._db = None
        self._embeddings_cache = None

    def _get_db(self):
        """Get MongoDB connection."""
        if self._client is None:
            self._client = MongoClient(self.mongodb_uri)
            self._db = self._client[self.database_name]
        return self._db

    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True
            )
        return self._model

    def _load_entity_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Load all entity embeddings from MongoDB."""
        if self._embeddings_cache is not None:
            return self._embeddings_cache

        logger.info("Loading entity embeddings from MongoDB...")
        db = self._get_db()

        # Fetch entity embeddings
        cursor = db.embeddings.find(
            {"embedding_type": "entity"},
            {"entity_id": 1, "embedding": 1}
        )

        entity_ids = []
        embeddings = []

        for doc in cursor:
            entity_ids.append(doc["entity_id"])
            embeddings.append(doc["embedding"])

        embeddings_array = np.array(embeddings, dtype=np.float32)
        logger.info(f"Loaded {len(entity_ids)} entity embeddings")

        self._embeddings_cache = (entity_ids, embeddings_array)
        return self._embeddings_cache

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        model = self._get_model()
        # Use search_query prefix for query embedding
        prefixed_query = "search_query: " + query
        embedding = model.encode(
            prefixed_query,
            normalize_embeddings=True
        )
        return np.array(embedding, dtype=np.float32)

    def find_similar_entities(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Find entities most similar to query embedding."""
        entity_ids, embeddings = self._load_entity_embeddings()

        if len(entity_ids) == 0:
            return []

        # Compute cosine similarities (embeddings are normalized)
        similarities = np.dot(embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            results.append({
                "entity_id": entity_ids[idx],
                "similarity": float(similarities[idx])
            })

        return results

    def get_entity_details(
        self,
        entity_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Get full entity details from MongoDB."""
        db = self._get_db()

        entities = {}
        cursor = db.graph_nodes.find({"_id": {"$in": entity_ids}})

        for doc in cursor:
            entities[doc["_id"]] = {
                "name": doc["_id"],
                "type": doc.get("type", "CONCEPT"),
                "description": doc.get("description", ""),
                "community_id": doc.get("community_id"),
                "source_chunks": doc.get("source_chunks", [])
            }

        return entities

    def _get_entity_metadata(
        self,
        entity_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Get source_chunks and community_id for entities in one query."""
        db = self._get_db()
        result = {}
        cursor = db.graph_nodes.find(
            {"_id": {"$in": entity_ids}},
            {"source_chunks": 1, "community_id": 1}
        )
        for doc in cursor:
            result[doc["_id"]] = {
                "source_chunks": doc.get("source_chunks", []),
                "community_id": doc.get("community_id")
            }
        return result

    def expand_subgraph(
        self,
        seed_entities: List[str],
        max_hops: int = 2,
        max_edges: int = 100
    ) -> Dict[str, Any]:
        """Expand subgraph from seed entities."""
        db = self._get_db()

        visited_entities = set(seed_entities)
        all_relationships = []

        current_entities = set(seed_entities)

        for hop in range(max_hops):
            if len(all_relationships) >= max_edges:
                break

            # Find relationships for current entities
            cursor = db.graph_edges.find({
                "$or": [
                    {"source": {"$in": list(current_entities)}},
                    {"target": {"$in": list(current_entities)}}
                ]
            }).limit(max_edges - len(all_relationships))

            next_entities = set()
            for rel in cursor:
                all_relationships.append({
                    "source": rel["source"],
                    "target": rel["target"],
                    "type": rel.get("type", "RELATED_TO"),
                    "description": rel.get("description", ""),
                    "weight": rel.get("weight", 1.0)
                })

                # Add connected entities for next hop
                if rel["source"] not in visited_entities:
                    next_entities.add(rel["source"])
                    visited_entities.add(rel["source"])
                if rel["target"] not in visited_entities:
                    next_entities.add(rel["target"])
                    visited_entities.add(rel["target"])

            current_entities = next_entities

            if not current_entities:
                break

        return {
            "entities": list(visited_entities),
            "relationships": all_relationships,
            "hops_expanded": hop + 1 if all_relationships else 0
        }

    def get_relevant_chunks(
        self,
        entity_ids: List[str],
        max_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        """Get relevant text chunks for entities."""
        db = self._get_db()

        # Get source chunks from entities
        entities = db.graph_nodes.find(
            {"_id": {"$in": entity_ids}},
            {"source_chunks": 1}
        )

        chunk_ids = set()
        for entity in entities:
            for chunk_id in entity.get("source_chunks", [])[:3]:
                chunk_ids.add(chunk_id)
                if len(chunk_ids) >= max_chunks:
                    break

        if not chunk_ids:
            return []

        # Fetch chunks
        chunks = []
        cursor = db.text_chunks.find({"_id": {"$in": list(chunk_ids)}})

        for doc in cursor:
            chunks.append({
                "id": doc["_id"],
                "text": doc.get("text", ""),
                "source_url": doc.get("source_url", ""),
                "source_title": doc.get("source_title", "")
            })

        return chunks[:max_chunks]

    def get_community_info(
        self,
        entity_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get community reports for entities."""
        db = self._get_db()

        # Get community IDs for entities
        entities = db.graph_nodes.find(
            {"_id": {"$in": entity_ids}},
            {"community_id": 1}
        )

        community_ids = set()
        for entity in entities:
            if entity.get("community_id") is not None:
                community_ids.add(str(entity["community_id"]))

        if not community_ids:
            return []

        # Fetch community reports
        reports = []
        cursor = db.community_reports.find({
            "community_id": {"$in": list(community_ids)}
        })

        for doc in cursor:
            reports.append({
                "community_id": doc.get("community_id"),
                "title": doc.get("title", ""),
                "summary": doc.get("summary", ""),
                "entity_count": doc.get("entity_count", 0)
            })

        return reports

    def _get_chunks_from_metadata(
        self,
        entity_metadata: Dict[str, Dict[str, Any]],
        max_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        """Get relevant text chunks using pre-fetched entity metadata."""
        db = self._get_db()

        chunk_ids = set()
        for entity_id, meta in entity_metadata.items():
            for chunk_id in meta.get("source_chunks", [])[:3]:
                chunk_ids.add(chunk_id)
                if len(chunk_ids) >= max_chunks:
                    break

        if not chunk_ids:
            return []

        chunks = []
        cursor = db.text_chunks.find({"_id": {"$in": list(chunk_ids)}})

        for doc in cursor:
            chunks.append({
                "id": doc["_id"],
                "text": doc.get("text", ""),
                "source_url": doc.get("source_url", ""),
                "source_title": doc.get("source_title", "")
            })

        return chunks[:max_chunks]

    def _get_communities_from_metadata(
        self,
        entity_metadata: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get community reports using pre-fetched entity metadata."""
        db = self._get_db()

        community_ids = set()
        for entity_id, meta in entity_metadata.items():
            if meta.get("community_id") is not None:
                community_ids.add(str(meta["community_id"]))

        if not community_ids:
            return []

        reports = []
        cursor = db.community_reports.find({
            "community_id": {"$in": list(community_ids)}
        })

        for doc in cursor:
            reports.append({
                "community_id": doc.get("community_id"),
                "title": doc.get("title", ""),
                "summary": doc.get("summary", ""),
                "entity_count": doc.get("entity_count", 0)
            })

        return reports

    def process_query(
        self,
        query: str,
        top_k_entities: int = 20,
        max_hops: int = 2,
        max_chunks: int = 10
    ) -> Dict[str, Any]:
        """
        Process a query and return relevant information.

        Returns:
            Dict containing:
            - query: Original query
            - query_embedding: Query vector
            - similar_entities: Top-k similar entities with scores
            - entity_details: Full entity information
            - subgraph: Expanded graph around entities
            - chunks: Relevant text chunks
            - communities: Related community summaries
        """
        logger.info(f"Processing query: {query[:100]}...")

        # Step 1: Embed query
        query_embedding = self.embed_query(query)

        # Step 2: Find similar entities
        similar_entities = self.find_similar_entities(query_embedding, top_k_entities)
        entity_ids = [e["entity_id"] for e in similar_entities]

        # Step 3-6: Parallelize independent database operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks in parallel
            details_future = executor.submit(self.get_entity_details, entity_ids)
            subgraph_future = executor.submit(self.expand_subgraph, entity_ids, max_hops)
            metadata_future = executor.submit(self._get_entity_metadata, entity_ids)

            # Wait for results
            entity_details = details_future.result()
            subgraph = subgraph_future.result()
            entity_metadata = metadata_future.result()

        # Use cached metadata for chunks and communities (sequential, but fast)
        chunks = self._get_chunks_from_metadata(entity_metadata, max_chunks)
        communities = self._get_communities_from_metadata(entity_metadata)

        return {
            "query": query,
            "similar_entities": similar_entities,
            "entity_details": entity_details,
            "subgraph": subgraph,
            "chunks": chunks,
            "communities": communities
        }

    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
