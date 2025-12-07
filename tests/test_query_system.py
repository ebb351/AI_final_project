#!/usr/bin/env python3
"""
Test Query System for GraphRAG AI Tutor

Tests the three required questions:
1. "What is attention in transformers and can you provide a python example?"
2. "What is CLIP and how it is used in computer vision applications?"
3. "Can you explain the variational lower bound and how it relates to Jensen's inequality?"
"""
import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query.processor import QueryProcessor
from src.query.context_builder import ContextBuilder
from src.query.generator import AnswerGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_query(
    query: str,
    processor: QueryProcessor,
    context_builder: ContextBuilder,
    generator: AnswerGenerator,
    verbose: bool = True
) -> dict:
    """Test a single query."""
    start_time = time.time()

    # Process query
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"QUERY: {query}")
        logger.info("="*80)

    # Step 1: Process query
    query_result = processor.process_query(query, top_k_entities=15, max_chunks=8)

    if verbose:
        logger.info(f"\nFound {len(query_result['similar_entities'])} similar entities:")
        for i, entity in enumerate(query_result['similar_entities'][:5]):
            logger.info(f"  {i+1}. {entity['entity_id']} (similarity: {entity['similarity']:.3f})")

        logger.info(f"\nSubgraph: {len(query_result['subgraph']['entities'])} entities, "
                   f"{len(query_result['subgraph']['relationships'])} relationships")

        logger.info(f"\nCommunities: {len(query_result['communities'])}")
        for community in query_result['communities'][:2]:
            logger.info(f"  - {community['title']}")

    # Step 2: Build context
    context = context_builder.build_context(query_result)

    if verbose:
        logger.info(f"\nContext length: {len(context['context_text'])} chars")
        logger.info(f"Entities used: {len(context['entities_used'])}")

    # Step 3: Generate answer
    response = generator.generate_answer(query, context, use_api=False)

    elapsed = time.time() - start_time

    if verbose:
        logger.info(f"\n{'='*40}")
        logger.info("ANSWER:")
        logger.info("="*40)
        logger.info(response.answer[:2000] + ("..." if len(response.answer) > 2000 else ""))
        logger.info(f"\n{'='*40}")
        logger.info(f"Model: {response.model}")
        logger.info(f"Success: {response.success}")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Entities: {response.entities_used[:5]}")

    return {
        "query": query,
        "answer": response.answer,
        "success": response.success,
        "elapsed": elapsed,
        "entities": response.entities_used,
        "sources": response.sources
    }


def main():
    """Run all test queries."""
    # Test questions from project requirements
    test_questions = [
        "What is attention in transformers and can you provide a python example?",
        "What is CLIP and how it is used in computer vision applications?",
        "Can you explain the variational lower bound and how it relates to Jensen's inequality?"
    ]

    logger.info("="*80)
    logger.info("GRAPHRAG QUERY SYSTEM TEST")
    logger.info("="*80)

    # Initialize components
    logger.info("\nInitializing query system...")
    processor = QueryProcessor()
    context_builder = ContextBuilder()
    generator = AnswerGenerator()

    # Check Ollama status
    ollama_status = generator.check_ollama_status()
    logger.info(f"Ollama status: {ollama_status}")

    if not ollama_status.get("available"):
        logger.error("Ollama is not available. Please start Ollama first.")
        return

    # Run tests
    results = []
    for question in test_questions:
        result = test_query(question, processor, context_builder, generator)
        results.append(result)
        time.sleep(1)  # Brief pause between queries

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    for i, result in enumerate(results):
        status = "PASS" if result["success"] else "FAIL"
        logger.info(f"{i+1}. [{status}] {result['query'][:60]}... ({result['elapsed']:.1f}s)")

    total_success = sum(1 for r in results if r["success"])
    logger.info(f"\nTotal: {total_success}/{len(results)} passed")

    # Cleanup
    processor.close()

    return results


if __name__ == "__main__":
    main()
