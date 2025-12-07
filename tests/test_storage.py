"""
Test script for MongoDB storage adapters.

This script loads the test extraction results and stores them in MongoDB.
"""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.storage import GraphRAGStorage


def main():
    """Test storage with actual extraction results."""
    # Load test extraction results (path relative to project root)
    test_results_file = project_root / "data" / "test_entities.json"

    if not test_results_file.exists():
        print(f"Test results file not found: {test_results_file}")
        print("Run: modal run modal_app/entity_extraction_test.py first")
        return 1

    print("=" * 80)
    print("TESTING MONGODB STORAGE ADAPTERS")
    print("=" * 80)

    # Load test results
    print(f"\nLoading test results from {test_results_file}...")
    with open(test_results_file, "r", encoding="utf-8") as f:
        test_results = json.load(f)

    print(f"Loaded {len(test_results)} extraction results")

    # Calculate totals
    total_entities = sum(len(r["entities"]) for r in test_results)
    total_relationships = sum(len(r["relationships"]) for r in test_results)

    print(f"Total entities to store: {total_entities}")
    print(f"Total relationships to store: {total_relationships}")

    # Initialize storage
    print("\nConnecting to MongoDB...")
    storage = GraphRAGStorage()

    try:
        # Get initial stats
        print("\nInitial graph state:")
        initial_stats = storage.get_graph_stats()
        print(f"  Nodes: {initial_stats['nodes']}")
        print(f"  Edges: {initial_stats['edges']}")

        # Store extraction results
        print("\nStoring extraction results...")
        result_stats = storage.store_extraction_results(test_results)

        print("\nStorage Results:")
        print(f"  Chunks processed: {result_stats['chunks_processed']}")
        print(f"  Entities inserted: {result_stats['entities_inserted']}")
        print(f"  Entities updated: {result_stats['entities_updated']}")
        print(f"  Relationships inserted: {result_stats['relationships_inserted']}")
        print(f"  Errors: {result_stats['errors']}")

        # Get final stats
        print("\nFinal graph state:")
        storage.print_graph_stats()

        # Validate some data
        print("\nValidation checks:")

        # Check for specific entities
        test_entity_names = [
            "Natural Language Processing",
            "Language Models",
            "Character-Level Language Modeling"
        ]

        for entity_name in test_entity_names:
            entity = storage.nodes.find_one({"name": entity_name})
            if entity:
                print(f"  [OK] Found entity: {entity_name} (type: {entity['type']})")
            else:
                print(f"  [FAIL] Entity not found: {entity_name}")

        # Check for relationships
        rel = storage.edges.find_one({"source": "Natural Language Processing"})
        if rel:
            print(f"  [OK] Found relationship: {rel['source']} -> {rel['target']} ({rel['type']})")
        else:
            print(f"  [FAIL] No relationships found from Natural Language Processing")

        print("\n" + "=" * 80)
        print("STORAGE TEST COMPLETE")
        print("=" * 80)

    finally:
        storage.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
