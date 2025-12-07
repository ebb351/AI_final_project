#!/usr/bin/env python3
"""
Community Detection using Leiden Algorithm

This script builds a knowledge graph from normalized entities and relationships,
then detects communities using the Leiden algorithm and generates summaries
for each community using local Ollama.

Input:
  - data/entities_normalized.json (16,728 unique entities, 19,794 relationships)

Output:
  - data/community_reports.json (community summaries)
  - data/graph_with_communities.json (entities with community assignments)

Dependencies:
  - networkx: Graph construction
  - igraph: Leiden algorithm implementation
  - leidenalg: Leiden community detection
  - ollama: Local LLM for community summarization
"""
import json
import logging
import subprocess
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Edge weights by relationship type (higher = stronger connection)
EDGE_WEIGHTS = {
    "PREREQUISITE_FOR": 3.0,   # Strongest pedagogical signal
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


def build_networkx_graph(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> nx.Graph:
    """
    Build a NetworkX graph from entities and relationships.

    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries

    Returns:
        NetworkX graph with entities as nodes and relationships as edges
    """
    logger.info("Building NetworkX graph...")

    G = nx.Graph()

    # Add nodes (entities)
    for entity in entities:
        name = entity.get("name", "")
        if not name:
            continue

        G.add_node(
            name,
            type=entity.get("type", "CONCEPT"),
            description=entity.get("description", "")[:500]  # Truncate long descriptions
        )

    logger.info(f"Added {G.number_of_nodes()} nodes")

    # Add edges (relationships)
    edge_count = 0
    for rel in relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")
        rel_type = rel.get("type", "NEAR_TRANSFER")

        if not source or not target:
            continue

        if source not in G or target not in G:
            continue

        # Get edge weight
        weight = EDGE_WEIGHTS.get(rel_type, 1.0)

        # Add or update edge
        if G.has_edge(source, target):
            # Increase weight for multiple relationships
            G[source][target]["weight"] += weight
        else:
            G.add_edge(
                source,
                target,
                type=rel_type,
                weight=weight,
                description=rel.get("description", "")[:200]
            )
            edge_count += 1

    logger.info(f"Added {edge_count} edges")

    return G


def run_leiden_community_detection(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: int = 42
) -> Dict[str, int]:
    """
    Run Leiden algorithm for community detection.

    Args:
        G: NetworkX graph
        resolution: Resolution parameter (higher = more communities)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node name to community ID
    """
    import igraph as ig
    import leidenalg

    logger.info(f"Running Leiden algorithm (resolution={resolution})...")

    # Convert NetworkX to igraph
    # Create a mapping from node names to indices
    node_list = list(G.nodes())
    node_to_idx = {name: idx for idx, name in enumerate(node_list)}

    # Create igraph graph
    edges = []
    weights = []
    for u, v, data in G.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get("weight", 1.0))

    G_ig = ig.Graph(n=len(node_list), edges=edges, directed=False)
    G_ig.vs["name"] = node_list
    G_ig.es["weight"] = weights

    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        G_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights=G_ig.es["weight"],
        resolution_parameter=resolution,
        n_iterations=10,
        seed=seed
    )

    # Map nodes to communities
    node_to_community = {}
    for idx, community_id in enumerate(partition.membership):
        node_name = node_list[idx]
        node_to_community[node_name] = community_id

    num_communities = len(set(partition.membership))
    logger.info(f"Detected {num_communities} communities")

    # Log community size distribution
    community_sizes = defaultdict(int)
    for community_id in partition.membership:
        community_sizes[community_id] += 1

    sizes = sorted(community_sizes.values(), reverse=True)
    logger.info(f"Largest communities: {sizes[:10]}")
    logger.info(f"Smallest communities: {sizes[-10:]}")

    return node_to_community


def get_community_entities(
    node_to_community: Dict[str, int],
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    """
    Group entities and relationships by community.

    Returns:
        Dictionary mapping community_id to {entities, relationships}
    """
    communities = defaultdict(lambda: {"entities": [], "relationships": []})

    # Create entity lookup
    entity_lookup = {e["name"]: e for e in entities}

    # Group entities by community
    for entity_name, community_id in node_to_community.items():
        if entity_name in entity_lookup:
            communities[community_id]["entities"].append(entity_lookup[entity_name])

    # Group relationships by community (both endpoints in same community)
    for rel in relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")

        source_community = node_to_community.get(source)
        target_community = node_to_community.get(target)

        if source_community is not None and source_community == target_community:
            communities[source_community]["relationships"].append(rel)

    return dict(communities)


def generate_community_summary(
    community_id: int,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    model: str = "qwen2.5",
    timeout: int = 120
) -> Dict[str, Any]:
    """
    Generate a summary for a community using local Ollama.

    Args:
        community_id: The community identifier
        entities: List of entities in the community
        relationships: List of relationships in the community
        model: Ollama model to use
        timeout: Timeout in seconds

    Returns:
        Dictionary with title and summary
    """
    # Build context from entities (top 20 by type distribution)
    type_counts = defaultdict(int)
    for e in entities:
        type_counts[e.get("type", "CONCEPT")] += 1

    # Sort entities by type importance
    entity_texts = []
    for entity in sorted(entities, key=lambda x: -type_counts.get(x.get("type", ""), 0))[:20]:
        name = entity.get("name", "")
        entity_type = entity.get("type", "")
        description = entity.get("description", "")[:100]
        entity_texts.append(f"- {name} ({entity_type}): {description}")

    entities_str = "\n".join(entity_texts)

    # Build relationship context (top 15)
    rel_texts = []
    for rel in relationships[:15]:
        source = rel.get("source", "")
        target = rel.get("target", "")
        rel_type = rel.get("type", "")
        rel_texts.append(f"- {source} --[{rel_type}]--> {target}")

    relationships_str = "\n".join(rel_texts) if rel_texts else "(No internal relationships)"

    # Build prompt
    prompt = f"""Analyze this cluster of AI/ML concepts and provide:
1. A short title (5-10 words) that captures the main theme
2. A summary (2-3 sentences) explaining what this cluster represents and why these concepts are related

ENTITIES ({len(entities)} total):
{entities_str}

RELATIONSHIPS ({len(relationships)} total):
{relationships_str}

Respond in JSON format only:
{{"title": "...", "summary": "..."}}"""

    try:
        # Call Ollama
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        response = result.stdout.strip()

        # Try to parse JSON from response
        # Find JSON in response (may have extra text)
        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            return {
                "title": parsed.get("title", f"Community {community_id}"),
                "summary": parsed.get("summary", ""),
                "generated": True
            }
        else:
            # Fallback: use response as summary
            return {
                "title": f"Community {community_id}",
                "summary": response[:500] if response else "No summary generated",
                "generated": False
            }

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout generating summary for community {community_id}")
        return {
            "title": f"Community {community_id}",
            "summary": "Summary generation timed out",
            "generated": False
        }
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for community {community_id}: {e}")
        return {
            "title": f"Community {community_id}",
            "summary": response[:500] if response else "Failed to parse summary",
            "generated": False
        }
    except Exception as e:
        logger.error(f"Error generating summary for community {community_id}: {e}")
        return {
            "title": f"Community {community_id}",
            "summary": f"Error: {str(e)}",
            "generated": False
        }


def generate_fallback_summary(
    community_id: int,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a simple fallback summary without LLM.
    """
    # Get type distribution
    type_counts = defaultdict(int)
    for e in entities:
        type_counts[e.get("type", "CONCEPT")] += 1

    dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "CONCEPT"

    # Get top entity names
    top_entities = [e["name"] for e in entities[:5]]

    title = f"{dominant_type.title()} Cluster ({len(entities)} concepts)"
    summary = f"A cluster of {len(entities)} entities primarily of type {dominant_type}. Key concepts include: {', '.join(top_entities)}. Connected by {len(relationships)} internal relationships."

    return {
        "title": title,
        "summary": summary,
        "generated": False
    }


def main():
    """Main community detection pipeline."""
    project_root = Path(__file__).parent.parent.parent

    # Input file
    entities_file = project_root / "data" / "entities_normalized.json"

    # Output files
    communities_file = project_root / "data" / "community_reports.json"
    graph_file = project_root / "data" / "graph_with_communities.json"

    logger.info("=" * 80)
    logger.info("COMMUNITY DETECTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input: {entities_file}")
    logger.info(f"Output communities: {communities_file}")
    logger.info(f"Output graph: {graph_file}")

    # Load normalized data
    logger.info("\nLoading normalized entities...")
    with open(entities_file, "r", encoding="utf-8") as f:
        normalized_data = json.load(f)

    entities = normalized_data.get("entities", [])
    relationships = normalized_data.get("relationships", [])

    logger.info(f"Loaded {len(entities)} entities and {len(relationships)} relationships")

    # Build graph
    G = build_networkx_graph(entities, relationships)

    # Run Leiden
    node_to_community = run_leiden_community_detection(G, resolution=1.0)

    # Group by community
    communities = get_community_entities(node_to_community, entities, relationships)

    logger.info(f"\nFound {len(communities)} communities")

    # Generate summaries for each community
    logger.info("\nGenerating community summaries with Ollama...")

    community_reports = []
    total_communities = len(communities)

    # Sort communities by size (largest first)
    sorted_communities = sorted(
        communities.items(),
        key=lambda x: len(x[1]["entities"]),
        reverse=True
    )

    for idx, (community_id, data) in enumerate(sorted_communities):
        community_entities = data["entities"]
        community_relationships = data["relationships"]

        logger.info(f"Processing community {idx + 1}/{total_communities} "
                   f"(id={community_id}, entities={len(community_entities)}, "
                   f"relationships={len(community_relationships)})")

        # Generate summary
        if len(community_entities) >= 3:
            # Use LLM for communities with 3+ entities
            summary_data = generate_community_summary(
                community_id,
                community_entities,
                community_relationships
            )
        else:
            # Use fallback for tiny communities
            summary_data = generate_fallback_summary(
                community_id,
                community_entities,
                community_relationships
            )

        report = {
            "community_id": community_id,
            "level": 0,  # Base level
            "title": summary_data["title"],
            "summary": summary_data["summary"],
            "entity_count": len(community_entities),
            "relationship_count": len(community_relationships),
            "entities": [e["name"] for e in community_entities],
            "key_entities": [e["name"] for e in community_entities[:10]],
            "generated_with_llm": summary_data.get("generated", False)
        }

        community_reports.append(report)

        # Small delay between LLM calls
        if summary_data.get("generated", False):
            time.sleep(0.5)

    # Save community reports
    logger.info(f"\nSaving community reports to {communities_file}...")
    output_reports = {
        "metadata": {
            "total_communities": len(community_reports),
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "resolution": 1.0,
            "algorithm": "leiden"
        },
        "communities": community_reports
    }

    with open(communities_file, "w", encoding="utf-8") as f:
        json.dump(output_reports, f, indent=2, ensure_ascii=False)

    # Save graph with community assignments
    logger.info(f"Saving graph with communities to {graph_file}...")

    # Add community assignments to entities
    entities_with_communities = []
    for entity in entities:
        entity_copy = entity.copy()
        entity_copy["community_id"] = node_to_community.get(entity["name"])
        entities_with_communities.append(entity_copy)

    graph_output = {
        "metadata": {
            "total_entities": len(entities_with_communities),
            "total_relationships": len(relationships),
            "total_communities": len(community_reports)
        },
        "entities": entities_with_communities,
        "relationships": relationships
    }

    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(graph_output, f, indent=2, ensure_ascii=False)

    # Summary statistics
    llm_generated = sum(1 for r in community_reports if r.get("generated_with_llm", False))

    logger.info("\n" + "=" * 80)
    logger.info("COMMUNITY DETECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total communities: {len(community_reports)}")
    logger.info(f"LLM-generated summaries: {llm_generated}")
    logger.info(f"Fallback summaries: {len(community_reports) - llm_generated}")
    logger.info(f"Largest community: {max(r['entity_count'] for r in community_reports)} entities")
    logger.info(f"Smallest community: {min(r['entity_count'] for r in community_reports)} entities")
    logger.info(f"Average community size: {sum(r['entity_count'] for r in community_reports) / len(community_reports):.1f} entities")
    logger.info(f"Output saved to:")
    logger.info(f"  - {communities_file}")
    logger.info(f"  - {graph_file}")
    logger.info("=" * 80)

    return community_reports


if __name__ == "__main__":
    main()
