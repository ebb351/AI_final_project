#!/usr/bin/env python3
"""
Entity and Relationship Normalization Script

This script normalizes the extracted entities and relationships:
1. Maps 180+ detected entity types to 7 standard types
2. Fixes relationship type typos and maps to 10 standard types
3. Deduplicates entities by name
4. Handles duplicate chunk IDs by merging

Input: data/extracted_entities.json
Output: data/entities_normalized.json (deduplicated and normalized)
"""
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Standard entity types from schema.py
STANDARD_ENTITY_TYPES = {
    "CONCEPT",
    "ALGORITHM",
    "TOOL",
    "PERSON",
    "MATHEMATICAL_CONCEPT",
    "RESOURCE",
    "EXAMPLE"
}

# Entity type normalization mapping
ENTITY_TYPE_MAPPING = {
    # Direct matches (no change needed)
    "CONCEPT": "CONCEPT",
    "ALGORITHM": "ALGORITHM",
    "TOOL": "TOOL",
    "RESOURCE": "RESOURCE",
    "PERSON": "PERSON",
    "MATHEMATICAL_CONCEPT": "MATHEMATICAL_CONCEPT",
    "EXAMPLE": "EXAMPLE",

    # Algorithmic / Method types -> ALGORITHM
    "METHOD": "ALGORITHM",
    "MODEL": "ALGORITHM",
    "PROCESS": "ALGORITHM",
    "OPERATION": "ALGORITHM",
    "TASK": "ALGORITHM",
    "TECHNIQUE": "ALGORITHM",
    "PROCEDURE": "ALGORITHM",
    "ARCHITECTURE": "ALGORITHM",
    "NETWORK": "ALGORITHM",
    "LAYER": "ALGORITHM",
    "OPTIMIZER": "ALGORITHM",
    "LOSS_FUNCTION": "ALGORITHM",
    "ACTIVATION_FUNCTION": "ALGORITHM",
    "REGULARIZATION": "ALGORITHM",

    # Conceptual types -> CONCEPT
    "FUNCTION": "CONCEPT",
    "PARAMETER": "CONCEPT",
    "ENTITY": "CONCEPT",
    "DOMAIN": "CONCEPT",
    "VECTOR": "CONCEPT",
    "SCALAR": "CONCEPT",
    "CONSTANT": "CONCEPT",
    "VARIABLE": "CONCEPT",
    "METRIC": "CONCEPT",
    "PROBLEM": "CONCEPT",
    "EVENT": "CONCEPT",
    "ACTION": "CONCEPT",
    "OBJECT": "CONCEPT",
    "FEATURE": "CONCEPT",
    "REPRESENTATION": "CONCEPT",
    "EMBEDDING": "CONCEPT",
    "TENSOR": "CONCEPT",
    "MATRIX": "CONCEPT",
    "HYPERPARAMETER": "CONCEPT",
    "TERM": "CONCEPT",
    "PRINCIPLE": "CONCEPT",
    "THEORY": "CONCEPT",
    "PARADIGM": "CONCEPT",
    "APPROACH": "CONCEPT",
    "STRATEGY": "CONCEPT",
    "FRAMEWORK_CONCEPT": "CONCEPT",
    "DATA_STRUCTURE": "CONCEPT",
    "DATATYPE": "CONCEPT",
    "TYPE": "CONCEPT",

    # Mathematical -> MATHEMATICAL_CONCEPT
    "THEOREM": "MATHEMATICAL_CONCEPT",
    "FORMULA": "MATHEMATICAL_CONCEPT",
    "EQUATION": "MATHEMATICAL_CONCEPT",
    "DISTRIBUTION": "MATHEMATICAL_CONCEPT",
    "FUNCTION_MATHEMATICAL": "MATHEMATICAL_CONCEPT",
    "PROBABILITY": "MATHEMATICAL_CONCEPT",
    "STATISTIC": "MATHEMATICAL_CONCEPT",
    "MATH_CONCEPT": "MATHEMATICAL_CONCEPT",

    # Tool types -> TOOL
    "CLASS": "TOOL",
    "LIBRARY": "TOOL",
    "FRAMEWORK": "TOOL",
    "APPLICATION": "TOOL",
    "SOFTWARE": "TOOL",
    "PLATFORM": "TOOL",
    "API": "TOOL",
    "PACKAGE": "TOOL",
    "MODULE": "TOOL",
    "INTERFACE": "TOOL",
    "FUNCTION_CALL": "TOOL",

    # Resource types -> RESOURCE
    "DATASET": "RESOURCE",
    "ORGANIZATION": "RESOURCE",
    "PAPER": "RESOURCE",
    "BOOK": "RESOURCE",
    "COURSE": "RESOURCE",
    "LECTURE": "RESOURCE",
    "BENCHMARK": "RESOURCE",
    "DOCUMENTATION": "RESOURCE",
    "COMPANY": "RESOURCE",
    "INSTITUTION": "RESOURCE",
    "UNIVERSITY": "RESOURCE",
    "RESEARCH_GROUP": "RESOURCE",

    # Example types -> EXAMPLE
    "CASE_STUDY": "EXAMPLE",
    "USE_CASE": "EXAMPLE",
    "APPLICATION_EXAMPLE": "EXAMPLE",
    "DEMONSTRATION": "EXAMPLE",
    "EXPERIMENT": "EXAMPLE",
}

# Standard relationship types from schema.py
STANDARD_RELATIONSHIP_TYPES = {
    "PREREQUISITE_FOR",
    "COMPONENT_OF",
    "SOLVES",
    "APPLIES_TO",
    "NEAR_TRANSFER",
    "CONTRASTS_WITH",
    "IS_A",
    "PART_OF",
    "EXPLAINS",
    "EXEMPLIFIES"
}

# Relationship type normalization mapping
RELATIONSHIP_TYPE_MAPPING = {
    # Direct matches
    "PREREQUISITE_FOR": "PREREQUISITE_FOR",
    "COMPONENT_OF": "COMPONENT_OF",
    "SOLVES": "SOLVES",
    "APPLIES_TO": "APPLIES_TO",
    "NEAR_TRANSFER": "NEAR_TRANSFER",
    "CONTRASTS_WITH": "CONTRASTS_WITH",
    "IS_A": "IS_A",
    "PART_OF": "PART_OF",
    "EXPLAINS": "EXPLAINS",
    "EXEMPLIFIES": "EXEMPLIFIES",

    # Typo corrections for EXPLAINS
    "EXPLANIES": "EXPLAINS",
    "EXPLANES": "EXPLAINS",
    "EXPLIES": "EXPLAINS",
    "EXPALINS": "EXPLAINS",
    "EXPLAIN": "EXPLAINS",

    # Typo corrections for PREREQUISITE_FOR
    "PREPREREQUISITE_FOR": "PREREQUISITE_FOR",
    "PRE.REQUISITE_FOR": "PREREQUISITE_FOR",
    "PREREQUSITE_FOR": "PREREQUISITE_FOR",
    "PREREQUISTE_FOR": "PREREQUISITE_FOR",
    "PRECONDITION_FOR": "PREREQUISITE_FOR",
    "REQUIRES": "PREREQUISITE_FOR",
    "DEPENDS_ON": "PREREQUISITE_FOR",

    # Similarity variants -> NEAR_TRANSFER
    "SIMILAR_TO": "NEAR_TRANSFER",
    "SIMILARITY": "NEAR_TRANSFER",
    "SIMILAR": "NEAR_TRANSFER",
    "RELATED_TO": "NEAR_TRANSFER",
    "COMPLEMENTARY_OF": "NEAR_TRANSFER",
    "RELATED": "NEAR_TRANSFER",
    "ASSOCIATED_WITH": "NEAR_TRANSFER",
    "ANALOGOUS_TO": "NEAR_TRANSFER",

    # PART_OF variants -> COMPONENT_OF or PART_OF
    "CONTENTS_OF": "COMPONENT_OF",
    "CONTAINS": "COMPONENT_OF",
    "INCLUDES": "COMPONENT_OF",
    "HAS_COMPONENT": "COMPONENT_OF",
    "COMPOSED_OF": "COMPONENT_OF",

    # Type hierarchies -> IS_A
    "TYPE_OF": "IS_A",
    "SUBTYPE_OF": "IS_A",
    "INSTANCE_OF": "IS_A",
    "KIND_OF": "IS_A",
    "VARIANT_OF": "IS_A",
    "SPECIALIZATION_OF": "IS_A",

    # Author/resource relationships -> EXPLAINS
    "AUTHORED_BY": "EXPLAINS",
    "PUBLISHED_BY": "EXPLAINS",
    "AUTHOR": "EXPLAINS",
    "DESCRIBES": "EXPLAINS",
    "DEFINES": "EXPLAINS",
    "INTRODUCES": "EXPLAINS",
    "PRESENTS": "EXPLAINS",

    # Usage/application -> APPLIES_TO
    "USED_IN": "APPLIES_TO",
    "USED_FOR": "APPLIES_TO",
    "APPLIED_IN": "APPLIES_TO",
    "USES": "APPLIES_TO",
    "UTILIZES": "APPLIES_TO",
    "IMPLEMENTS": "APPLIES_TO",
    "EMPLOYS": "APPLIES_TO",

    # Problem solving -> SOLVES
    "ADDRESSES": "SOLVES",
    "HANDLES": "SOLVES",
    "RESOLVES": "SOLVES",
    "TACKLES": "SOLVES",

    # Contrast -> CONTRASTS_WITH
    "DIFFERS_FROM": "CONTRASTS_WITH",
    "OPPOSITE_OF": "CONTRASTS_WITH",
    "ALTERNATIVE_TO": "CONTRASTS_WITH",
    "COMPETES_WITH": "CONTRASTS_WITH",

    # Example -> EXEMPLIFIES
    "DEMONSTRATES": "EXEMPLIFIES",
    "ILLUSTRATES": "EXEMPLIFIES",
    "SHOWS": "EXEMPLIFIES",
    "EXAMPLE_OF": "EXEMPLIFIES",
}


def normalize_entity_type(entity_type: str) -> str:
    """Normalize an entity type to one of the standard types."""
    # Try direct lookup first
    entity_type_upper = entity_type.upper().strip()

    if entity_type_upper in STANDARD_ENTITY_TYPES:
        return entity_type_upper

    if entity_type_upper in ENTITY_TYPE_MAPPING:
        return ENTITY_TYPE_MAPPING[entity_type_upper]

    # Default to CONCEPT for unknown types
    return "CONCEPT"


def normalize_relationship_type(rel_type: str) -> str:
    """Normalize a relationship type to one of the standard types."""
    rel_type_upper = rel_type.upper().strip()

    if rel_type_upper in STANDARD_RELATIONSHIP_TYPES:
        return rel_type_upper

    if rel_type_upper in RELATIONSHIP_TYPE_MAPPING:
        return RELATIONSHIP_TYPE_MAPPING[rel_type_upper]

    # Default to NEAR_TRANSFER for unknown types
    return "NEAR_TRANSFER"


def deduplicate_entities(
    extraction_results: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]], Dict[str, Set[str]]]:
    """
    Deduplicate entities across all chunks.

    Returns:
        - unique_entities: dict mapping entity name -> merged entity info
        - relationships_by_chunk: dict mapping chunk_id -> list of relationships
        - entity_sources: dict mapping entity name -> set of chunk IDs
    """
    unique_entities: Dict[str, Dict] = {}
    entity_sources: Dict[str, Set[str]] = defaultdict(set)
    relationships_by_chunk: Dict[str, List[Dict]] = {}

    type_stats = defaultdict(int)
    original_type_stats = defaultdict(int)

    for result in extraction_results:
        chunk_id = result.get("chunk_id", "")
        source_url = result.get("source_url", "")

        # Process entities
        for entity in result.get("entities", []):
            name = entity.get("name", "").strip()
            if not name:
                continue

            original_type = entity.get("type", "CONCEPT")
            original_type_stats[original_type] += 1

            normalized_type = normalize_entity_type(original_type)
            type_stats[normalized_type] += 1

            description = entity.get("description", "")

            if name in unique_entities:
                # Merge with existing entity
                existing = unique_entities[name]

                # Keep longer description
                if len(description) > len(existing.get("description", "")):
                    existing["description"] = description

                # Collect all descriptions for later summarization
                if description and description not in existing.get("all_descriptions", []):
                    existing.setdefault("all_descriptions", []).append(description)

                # Use most common type (keep existing if same)
                existing.setdefault("type_votes", defaultdict(int))
                existing["type_votes"][normalized_type] += 1

            else:
                # New entity
                unique_entities[name] = {
                    "name": name,
                    "type": normalized_type,
                    "description": description,
                    "all_descriptions": [description] if description else [],
                    "type_votes": defaultdict(int, {normalized_type: 1}),
                    "original_type": original_type
                }

            entity_sources[name].add(chunk_id)

        # Process relationships for this chunk
        chunk_relationships = []
        for rel in result.get("relationships", []):
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            rel_type = rel.get("type", "NEAR_TRANSFER")
            description = rel.get("description", "")

            if not source or not target:
                continue

            normalized_rel_type = normalize_relationship_type(rel_type)

            chunk_relationships.append({
                "source": source,
                "target": target,
                "type": normalized_rel_type,
                "description": description,
                "original_type": rel_type
            })

        relationships_by_chunk[chunk_id] = chunk_relationships

    # Finalize entity types based on votes
    for name, entity in unique_entities.items():
        if entity.get("type_votes"):
            # Choose most voted type
            best_type = max(entity["type_votes"].items(), key=lambda x: x[1])[0]
            entity["type"] = best_type
            del entity["type_votes"]

        # Clean up all_descriptions if only one
        if len(entity.get("all_descriptions", [])) <= 1:
            del entity["all_descriptions"]

    logger.info(f"Original entity type distribution (top 20):")
    for t, count in sorted(original_type_stats.items(), key=lambda x: -x[1])[:20]:
        logger.info(f"  {t}: {count}")

    logger.info(f"\nNormalized entity type distribution:")
    for t, count in sorted(type_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {t}: {count}")

    return unique_entities, relationships_by_chunk, entity_sources


def filter_valid_relationships(
    relationships_by_chunk: Dict[str, List[Dict]],
    unique_entities: Dict[str, Dict]
) -> List[Dict]:
    """
    Filter relationships to only include those where both source and target exist.
    Also deduplicate relationships.
    """
    valid_entity_names = set(unique_entities.keys())

    # Track unique relationships (source, target, type)
    seen_relationships: Set[Tuple[str, str, str]] = set()
    valid_relationships: List[Dict] = []

    rel_type_stats = defaultdict(int)
    original_rel_type_stats = defaultdict(int)

    invalid_count = 0
    duplicate_count = 0

    for chunk_id, relationships in relationships_by_chunk.items():
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            rel_type = rel["type"]
            original_type = rel.get("original_type", rel_type)

            original_rel_type_stats[original_type] += 1

            # Check if both entities exist
            if source not in valid_entity_names or target not in valid_entity_names:
                invalid_count += 1
                continue

            # Check for duplicates
            rel_key = (source, target, rel_type)
            if rel_key in seen_relationships:
                duplicate_count += 1
                continue

            seen_relationships.add(rel_key)
            rel_type_stats[rel_type] += 1

            valid_relationships.append({
                "source": source,
                "target": target,
                "type": rel_type,
                "description": rel.get("description", ""),
                "source_chunk": chunk_id
            })

    logger.info(f"\nOriginal relationship type distribution (top 20):")
    for t, count in sorted(original_rel_type_stats.items(), key=lambda x: -x[1])[:20]:
        logger.info(f"  {t}: {count}")

    logger.info(f"\nNormalized relationship type distribution:")
    for t, count in sorted(rel_type_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {t}: {count}")

    logger.info(f"\nRelationship filtering:")
    logger.info(f"  Invalid (missing entities): {invalid_count}")
    logger.info(f"  Duplicates removed: {duplicate_count}")
    logger.info(f"  Valid unique relationships: {len(valid_relationships)}")

    return valid_relationships


def main():
    """Main normalization pipeline."""
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "extracted_entities.json"
    output_file = project_root / "data" / "entities_normalized.json"

    logger.info("=" * 80)
    logger.info("ENTITY NORMALIZATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    # Load extraction results
    logger.info("\nLoading extraction results...")
    with open(input_file, "r", encoding="utf-8") as f:
        extraction_results = json.load(f)

    logger.info(f"Loaded {len(extraction_results)} chunk results")

    # Count original totals
    total_entities = sum(len(r.get("entities", [])) for r in extraction_results)
    total_relationships = sum(len(r.get("relationships", [])) for r in extraction_results)
    logger.info(f"Total entities (with duplicates): {total_entities}")
    logger.info(f"Total relationships (with duplicates): {total_relationships}")

    # Deduplicate and normalize entities
    logger.info("\nDeduplicating and normalizing entities...")
    unique_entities, relationships_by_chunk, entity_sources = deduplicate_entities(extraction_results)

    logger.info(f"\nUnique entities: {len(unique_entities)}")

    # Filter and normalize relationships
    logger.info("\nFiltering and normalizing relationships...")
    valid_relationships = filter_valid_relationships(relationships_by_chunk, unique_entities)

    # Prepare output
    output_data = {
        "metadata": {
            "source_file": str(input_file),
            "total_chunks_processed": len(extraction_results),
            "original_entity_count": total_entities,
            "original_relationship_count": total_relationships,
            "unique_entity_count": len(unique_entities),
            "unique_relationship_count": len(valid_relationships),
            "entity_deduplication_ratio": f"{len(unique_entities)/total_entities*100:.1f}%" if total_entities > 0 else "N/A",
            "standard_entity_types": list(STANDARD_ENTITY_TYPES),
            "standard_relationship_types": list(STANDARD_RELATIONSHIP_TYPES)
        },
        "entities": list(unique_entities.values()),
        "relationships": valid_relationships,
        "entity_sources": {k: list(v) for k, v in entity_sources.items()}
    }

    # Save output
    logger.info(f"\nSaving normalized data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("NORMALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Original entities: {total_entities}")
    logger.info(f"Unique entities: {len(unique_entities)} ({len(unique_entities)/total_entities*100:.1f}% of original)")
    logger.info(f"Original relationships: {total_relationships}")
    logger.info(f"Valid unique relationships: {len(valid_relationships)} ({len(valid_relationships)/total_relationships*100:.1f}% of original)")
    logger.info(f"Output saved to: {output_file}")
    logger.info("=" * 80)

    return output_data


if __name__ == "__main__":
    main()
