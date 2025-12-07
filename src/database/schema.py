"""
MongoDB Schema Definitions for GraphRAG Knowledge Graph

This module defines the MongoDB schema for storing the knowledge graph constructed
from AI/ML course materials. The schema supports entity extraction, relationship
mapping, embedding storage, and community detection.

Collections:
- graph_nodes: Entities extracted from course materials (concepts, algorithms, tools, people)
- graph_edges: Relationships between entities (prerequisite, component, etc.)
- text_chunks: Chunked text segments from source materials
- embeddings: Vector embeddings for chunks (768-dim from nomic-embed-text)
- community_reports: Hierarchical community summaries from Leiden algorithm

The text_chunks collection is populated during chunking. The remaining collections
are populated during graph construction, embedding generation, and community detection.
"""
from typing import Dict, Any

# Entity types based on project requirements
ENTITY_TYPES = [
    "CONCEPT",              # Abstract ideas (e.g., "Reinforcement Learning")
    "ALGORITHM",            # Specific algorithms (e.g., "Q-Learning")
    "TOOL",                 # Software/frameworks (e.g., "PyTorch", "ROS2")
    "PERSON",               # Historical figures (e.g., "Bellman", "Kalman")
    "MATHEMATICAL_CONCEPT", # Math concepts (e.g., "Jensen's Inequality")
    "RESOURCE",             # Course resources (PDF, video, webpage)
    "EXAMPLE",              # Worked examples
]

# Relationship types based on project requirements
RELATIONSHIP_TYPES = [
    "PREREQUISITE_FOR",     # A is prerequisite for B (e.g., Calculus → Backprop)
    "COMPONENT_OF",         # A is part of B (e.g., Q-Learning is part of RL)
    "SOLVES",               # Algorithm solves problem (e.g., Value Iteration solves MDP)
    "APPLIES_TO",           # Concept applies to domain
    "NEAR_TRANSFER",        # Sibling/related concepts (e.g., LSTM ↔ GRU)
    "CONTRASTS_WITH",       # Opposing or contrasting concepts
    "IS_A",                 # Type hierarchy (e.g., LSTM is_a RNN)
    "PART_OF",              # Compositional relationship
    "EXPLAINS",             # Resource explains concept
    "EXEMPLIFIES",          # Example demonstrates concept
]

# MongoDB Collection Schemas

GRAPH_NODES_SCHEMA = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "type"],
            "properties": {
                "_id": {
                    "bsonType": "string",
                    "description": "Entity name (e.g., 'Markov Decision Process')"
                },
                "type": {
                    "enum": ENTITY_TYPES,
                    "description": "Type of entity"
                },
                "description": {
                    "bsonType": "string",
                    "description": "Entity description from LLM"
                },
                "aliases": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"},
                    "description": "Alternative names (e.g., ['MDP', 'Markov Decision Process'])"
                },
                "source_chunks": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"},
                    "description": "Chunk IDs where this entity appears"
                },
                "difficulty": {
                    "bsonType": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "description": "Difficulty level (if applicable)"
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "When entity was created"
                },
                "metadata": {
                    "bsonType": "object",
                    "description": "Additional metadata"
                }
            }
        }
    }
}

GRAPH_EDGES_SCHEMA = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["source", "target", "type"],
            "properties": {
                "source": {
                    "bsonType": "string",
                    "description": "Source entity ID"
                },
                "target": {
                    "bsonType": "string",
                    "description": "Target entity ID"
                },
                "type": {
                    "enum": RELATIONSHIP_TYPES,
                    "description": "Relationship type"
                },
                "description": {
                    "bsonType": "string",
                    "description": "Description of the relationship"
                },
                "weight": {
                    "bsonType": "double",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Relationship strength/confidence (0-1)"
                },
                "source_chunks": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"},
                    "description": "Chunk IDs where this relationship was found"
                },
                "keywords": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"},
                    "description": "Keywords associated with relationship"
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "When relationship was created"
                }
            }
        }
    }
}

TEXT_CHUNKS_SCHEMA = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "text", "source_id"],
            "properties": {
                "_id": {
                    "bsonType": "string",
                    "description": "Unique chunk ID"
                },
                "text": {
                    "bsonType": "string",
                    "description": "Chunk text content"
                },
                "source_id": {
                    "bsonType": "string",
                    "description": "ID of source document"
                },
                "source_url": {
                    "bsonType": "string",
                    "description": "URL of source document"
                },
                "source_title": {
                    "bsonType": "string",
                    "description": "Title of source document"
                },
                "source_type": {
                    "enum": ["html_page", "pdf_document", "video_transcript"],
                    "description": "Type of source"
                },
                "chunk_index": {
                    "bsonType": "int",
                    "description": "Index of chunk within source document"
                },
                "token_count": {
                    "bsonType": "int",
                    "description": "Number of tokens in chunk"
                },
                "processed": {
                    "bsonType": "bool",
                    "description": "Whether chunk has been processed for entity extraction"
                },
                "metadata": {
                    "bsonType": "object",
                    "description": "Additional metadata from source"
                }
            }
        }
    }
}

EMBEDDINGS_SCHEMA = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "chunk_id", "embedding"],
            "properties": {
                "_id": {
                    "bsonType": "string",
                    "description": "Embedding ID (same as chunk_id)"
                },
                "chunk_id": {
                    "bsonType": "string",
                    "description": "Reference to text chunk"
                },
                "embedding": {
                    "bsonType": "array",
                    "description": "Vector embedding (768-dim for nomic-embed-text)"
                },
                "model": {
                    "bsonType": "string",
                    "description": "Embedding model used (e.g., 'nomic-embed-text-v1.5')"
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "When embedding was created"
                }
            }
        }
    }
}

COMMUNITY_REPORTS_SCHEMA = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "level", "community_id"],
            "properties": {
                "_id": {
                    "bsonType": "string",
                    "description": "Unique report ID"
                },
                "level": {
                    "bsonType": "int",
                    "description": "Hierarchy level (0=micro, 1=macro, 2=themes)"
                },
                "community_id": {
                    "bsonType": "string",
                    "description": "Community identifier"
                },
                "title": {
                    "bsonType": "string",
                    "description": "Community title/summary"
                },
                "summary": {
                    "bsonType": "string",
                    "description": "Detailed community summary from LLM"
                },
                "entities": {
                    "bsonType": "array",
                    "items": {"bsonType": "string"},
                    "description": "Entity IDs in this community"
                },
                "entity_count": {
                    "bsonType": "int",
                    "description": "Number of entities in community"
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "When report was created"
                }
            }
        }
    }
}

# Index definitions for performance
INDEXES = {
    "graph_nodes": [
        {"keys": [("type", 1)], "name": "idx_type"},
        {"keys": [("source_chunks", 1)], "name": "idx_source_chunks"},
        {"keys": [("difficulty", 1)], "name": "idx_difficulty"},
    ],
    "graph_edges": [
        {"keys": [("source", 1), ("target", 1)], "name": "idx_source_target", "unique": False},
        {"keys": [("type", 1)], "name": "idx_type"},
        {"keys": [("source", 1)], "name": "idx_source"},
        {"keys": [("target", 1)], "name": "idx_target"},
    ],
    "text_chunks": [
        {"keys": [("source_id", 1)], "name": "idx_source_id"},
        {"keys": [("source_type", 1)], "name": "idx_source_type"},
        {"keys": [("processed", 1)], "name": "idx_processed"},
    ],
    "embeddings": [
        {"keys": [("chunk_id", 1)], "name": "idx_chunk_id", "sparse": True},  # sparse allows nulls
        {"keys": [("model", 1)], "name": "idx_model"},
        {"keys": [("embedding_type", 1)], "name": "idx_embedding_type"},  # Critical for entity lookup
        {"keys": [("entity_id", 1)], "name": "idx_entity_id", "sparse": True},  # sparse allows nulls
    ],
    "community_reports": [
        {"keys": [("level", 1)], "name": "idx_level"},
        {"keys": [("community_id", 1)], "name": "idx_community_id"},
    ],
}


def get_collection_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Get all collection schemas.

    Returns:
        Dictionary mapping collection names to schemas
    """
    return {
        "graph_nodes": GRAPH_NODES_SCHEMA,
        "graph_edges": GRAPH_EDGES_SCHEMA,
        "text_chunks": TEXT_CHUNKS_SCHEMA,
        "embeddings": EMBEDDINGS_SCHEMA,
        "community_reports": COMMUNITY_REPORTS_SCHEMA,
    }


def get_collection_indexes() -> Dict[str, list]:
    """
    Get all collection indexes.

    Returns:
        Dictionary mapping collection names to index definitions
    """
    return INDEXES
