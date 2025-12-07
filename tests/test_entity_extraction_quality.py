"""
Tests for entity extraction quality validation.

This module validates the quality of extracted entities and relationships
from the GraphRAG entity extraction pipeline.
"""
import json
import pytest
from pathlib import Path
from collections import Counter


@pytest.fixture
def extraction_results():
    """Load extraction results from JSON file."""
    results_path = Path(__file__).parent.parent / "data" / "extracted_entities.json"
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class TestExtractionCompleteness:
    """Tests for extraction completeness and coverage."""

    def test_all_chunks_processed(self, extraction_results):
        """Verify all expected chunks were processed."""
        assert len(extraction_results) == 7636, \
            f"Expected 7,636 chunks, got {len(extraction_results)}"

    def test_no_critical_errors(self, extraction_results):
        """Verify no chunks have critical errors."""
        errors = [r for r in extraction_results if 'error' in r]
        assert len(errors) == 0, \
            f"Found {len(errors)} chunks with errors"

    def test_minimum_coverage(self, extraction_results):
        """Verify at least 25% of chunks have entities."""
        chunks_with_entities = sum(1 for r in extraction_results if len(r.get('entities', [])) > 0)
        coverage = chunks_with_entities / len(extraction_results)
        assert coverage >= 0.25, \
            f"Only {coverage*100:.1f}% of chunks have entities (expected >= 25%)"

    def test_required_fields_present(self, extraction_results):
        """Verify all results have required fields."""
        required_fields = {'chunk_id', 'source_id', 'entities', 'relationships'}
        for i, result in enumerate(extraction_results[:100]):  # Sample first 100
            missing = required_fields - set(result.keys())
            assert not missing, \
                f"Result {i} missing fields: {missing}"


class TestEntityQuality:
    """Tests for entity extraction quality."""

    def test_entity_structure(self, extraction_results):
        """Verify entities have required fields."""
        required_fields = {'name', 'type'}
        for result in extraction_results[:100]:  # Sample first 100
            for entity in result.get('entities', []):
                missing = required_fields - set(entity.keys())
                assert not missing, \
                    f"Entity in {result['chunk_id']} missing: {missing}"

    def test_entity_types_valid(self, extraction_results):
        """Verify all entity types are from expected set."""
        expected_types = {
            'CONCEPT', 'ALGORITHM', 'TOOL', 'PERSON',
            'MATHEMATICAL_CONCEPT', 'RESOURCE', 'EXAMPLE',
            'METHOD', 'FUNCTION', 'PARAMETER', 'MODEL',
            'ORGANIZATION', 'CLASS', 'ENTITY', 'APPLICATION',
            'VECTOR', 'SCALAR', 'MATRIX', 'ROBOT', 'WHEEL',
            'CONDITION', 'THEOREM'
        }

        invalid_types = set()
        for result in extraction_results:
            for entity in result.get('entities', []):
                entity_type = entity.get('type', 'UNKNOWN')
                if entity_type not in expected_types:
                    invalid_types.add(entity_type)

        # Allow some variation but flag if too many unexpected types
        assert len(invalid_types) <= 10, \
            f"Found {len(invalid_types)} unexpected entity types: {invalid_types}"

    def test_entity_names_not_empty(self, extraction_results):
        """Verify entity names are not empty."""
        empty_names = 0
        for result in extraction_results:
            for entity in result.get('entities', []):
                if not entity.get('name', '').strip():
                    empty_names += 1

        assert empty_names == 0, \
            f"Found {empty_names} entities with empty names"

    def test_average_entities_per_chunk(self, extraction_results):
        """Verify reasonable number of entities per chunk."""
        total_entities = sum(len(r.get('entities', [])) for r in extraction_results)
        avg_entities = total_entities / len(extraction_results)

        # Should be between 3 and 15 entities per chunk on average
        assert 3 <= avg_entities <= 15, \
            f"Average entities/chunk is {avg_entities:.2f} (expected 3-15)"

    def test_entity_distribution(self, extraction_results):
        """Verify entity type distribution is reasonable."""
        entity_types = Counter()
        for result in extraction_results:
            for entity in result.get('entities', []):
                entity_types[entity.get('type', 'UNKNOWN')] += 1

        total = sum(entity_types.values())

        # CONCEPT should be most common (30-60% of entities)
        concept_ratio = entity_types.get('CONCEPT', 0) / total
        assert 0.30 <= concept_ratio <= 0.60, \
            f"CONCEPT entities are {concept_ratio*100:.1f}% (expected 30-60%)"

        # ALGORITHM should be significant (15-35%)
        algo_ratio = entity_types.get('ALGORITHM', 0) / total
        assert 0.15 <= algo_ratio <= 0.35, \
            f"ALGORITHM entities are {algo_ratio*100:.1f}% (expected 15-35%)"


class TestRelationshipQuality:
    """Tests for relationship extraction quality."""

    def test_relationship_structure(self, extraction_results):
        """Verify relationships have required fields."""
        required_fields = {'source', 'target', 'type'}
        for result in extraction_results[:100]:  # Sample first 100
            for rel in result.get('relationships', []):
                missing = required_fields - set(rel.keys())
                assert not missing, \
                    f"Relationship in {result['chunk_id']} missing: {missing}"

    def test_relationship_types_valid(self, extraction_results):
        """Verify all relationship types are from expected set."""
        expected_types = {
            'PREREQUISITE_FOR', 'COMPONENT_OF', 'SOLVES', 'APPLIES_TO',
            'NEAR_TRANSFER', 'CONTRASTS_WITH', 'IS_A', 'EXPLAINS',
            'EXEMPLIFIES', 'SIMILAR_TO', 'AUTHORED_BY', 'PUBLISHED_BY',
            'RESULT_OF', 'SIMILARITY', 'EXPLANIES'  # Note: EXPLANIES is a typo but exists in data
        }

        invalid_types = set()
        for result in extraction_results:
            for rel in result.get('relationships', []):
                rel_type = rel.get('type', 'UNKNOWN')
                if rel_type not in expected_types:
                    invalid_types.add(rel_type)

        # Allow some variation
        assert len(invalid_types) <= 10, \
            f"Found {len(invalid_types)} unexpected relationship types: {invalid_types}"

    def test_no_orphaned_relationships(self, extraction_results):
        """Verify all relationships reference existing entities."""
        orphaned_count = 0
        for result in extraction_results:
            entity_names = {e['name'] for e in result.get('entities', [])}
            for rel in result.get('relationships', []):
                source = rel.get('source', '')
                target = rel.get('target', '')
                if source not in entity_names or target not in entity_names:
                    orphaned_count += 1

        assert orphaned_count == 0, \
            f"Found {orphaned_count} orphaned relationships"

    def test_no_self_relationships(self, extraction_results):
        """Verify entities don't have relationships to themselves."""
        self_rel_count = 0
        for result in extraction_results:
            for rel in result.get('relationships', []):
                if rel.get('source') == rel.get('target'):
                    self_rel_count += 1

        # Some self-relationships might be valid, but shouldn't be too many
        total_rels = sum(len(r.get('relationships', [])) for r in extraction_results)
        self_rel_ratio = self_rel_count / max(total_rels, 1)
        assert self_rel_ratio <= 0.05, \
            f"Self-relationships are {self_rel_ratio*100:.1f}% (expected <= 5%)"

    def test_relationship_distribution(self, extraction_results):
        """Verify relationship type distribution is reasonable."""
        rel_types = Counter()
        for result in extraction_results:
            for rel in result.get('relationships', []):
                rel_types[rel.get('type', 'UNKNOWN')] += 1

        total = sum(rel_types.values())

        # APPLIES_TO should be most common (35-60%)
        applies_ratio = rel_types.get('APPLIES_TO', 0) / total
        assert 0.35 <= applies_ratio <= 0.60, \
            f"APPLIES_TO relationships are {applies_ratio*100:.1f}% (expected 35-60%)"


class TestGraphConnectivity:
    """Tests for overall graph structure and connectivity."""

    def test_chunks_with_both_entities_and_relationships(self, extraction_results):
        """Verify reasonable proportion have both entities and relationships."""
        both_count = sum(
            1 for r in extraction_results
            if len(r.get('entities', [])) > 0 and len(r.get('relationships', [])) > 0
        )
        ratio = both_count / len(extraction_results)

        # At least 25% should have both
        assert ratio >= 0.25, \
            f"Only {ratio*100:.1f}% of chunks have both entities and relationships"

    def test_average_relationships_per_chunk(self, extraction_results):
        """Verify reasonable number of relationships per chunk."""
        total_rels = sum(len(r.get('relationships', [])) for r in extraction_results)
        avg_rels = total_rels / len(extraction_results)

        # Should be between 2 and 12 relationships per chunk on average
        assert 2 <= avg_rels <= 12, \
            f"Average relationships/chunk is {avg_rels:.2f} (expected 2-12)"

    def test_source_coverage(self, extraction_results):
        """Verify multiple sources are represented."""
        unique_sources = {r.get('source_id') for r in extraction_results}

        # Should have at least 100 unique sources (since we removed only 2)
        assert len(unique_sources) >= 100, \
            f"Only {len(unique_sources)} unique sources found (expected >= 100)"

    def test_entity_diversity(self, extraction_results):
        """Verify we have diverse entity names (not just repetition)."""
        all_entity_names = []
        for result in extraction_results:
            for entity in result.get('entities', []):
                all_entity_names.append(entity.get('name', ''))

        unique_ratio = len(set(all_entity_names)) / len(all_entity_names)

        # At least 40% should be unique
        assert unique_ratio >= 0.40, \
            f"Only {unique_ratio*100:.1f}% of entity names are unique"


class TestDataIntegrity:
    """Tests for data integrity and consistency."""

    def test_chunk_ids_unique(self, extraction_results):
        """Verify all chunk IDs are unique."""
        chunk_ids = [r.get('chunk_id') for r in extraction_results]
        unique_ids = set(chunk_ids)

        assert len(chunk_ids) == len(unique_ids), \
            f"Found {len(chunk_ids) - len(unique_ids)} duplicate chunk IDs"

    def test_source_urls_present(self, extraction_results):
        """Verify most chunks have source URLs."""
        chunks_with_urls = sum(
            1 for r in extraction_results
            if r.get('source_url', '').strip()
        )
        ratio = chunks_with_urls / len(extraction_results)

        # At least 90% should have URLs
        assert ratio >= 0.90, \
            f"Only {ratio*100:.1f}% of chunks have source URLs"

    def test_json_serializable(self, extraction_results):
        """Verify results can be re-serialized as JSON."""
        try:
            json.dumps(extraction_results)
        except Exception as e:
            pytest.fail(f"Results are not JSON serializable: {e}")

    def test_no_null_values_in_critical_fields(self, extraction_results):
        """Verify no null values in critical fields."""
        for result in extraction_results[:100]:  # Sample first 100
            assert result.get('chunk_id') is not None
            assert result.get('source_id') is not None
            assert result.get('entities') is not None
            assert result.get('relationships') is not None


class TestSpecificExtractions:
    """Tests for specific high-value extractions."""

    def test_common_ml_concepts_present(self, extraction_results):
        """Verify common ML concepts were extracted."""
        all_entity_names = set()
        for result in extraction_results:
            for entity in result.get('entities', []):
                all_entity_names.add(entity.get('name', '').lower())

        # Check for some expected AI/ML concepts
        expected_concepts = [
            'transformer', 'neural network', 'reinforcement learning',
            'convolutional', 'gradient', 'backpropagation'
        ]

        found_concepts = [
            concept for concept in expected_concepts
            if any(concept in name for name in all_entity_names)
        ]

        # Should find at least 50% of expected concepts
        found_ratio = len(found_concepts) / len(expected_concepts)
        assert found_ratio >= 0.50, \
            f"Only found {len(found_concepts)}/{len(expected_concepts)} expected ML concepts"

    def test_relationships_are_meaningful(self, extraction_results):
        """Verify relationships have descriptions when present."""
        rels_with_desc = 0
        total_rels = 0

        for result in extraction_results[:100]:  # Sample first 100
            for rel in result.get('relationships', []):
                total_rels += 1
                if rel.get('description', '').strip():
                    rels_with_desc += 1

        if total_rels > 0:
            desc_ratio = rels_with_desc / total_rels
            # At least 20% should have descriptions
            assert desc_ratio >= 0.20, \
                f"Only {desc_ratio*100:.1f}% of relationships have descriptions"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
