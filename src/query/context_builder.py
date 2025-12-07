#!/usr/bin/env python3
"""
Context Builder for GraphRAG

Builds structured context from query results for answer generation.
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build context for answer generation."""

    def __init__(
        self,
        max_entities: int = 15,
        max_relationships: int = 30,
        max_chunks: int = 5,
        max_context_chars: int = 12000
    ):
        """Initialize context builder."""
        self.max_entities = max_entities
        self.max_relationships = max_relationships
        self.max_chunks = max_chunks
        self.max_context_chars = max_context_chars

    def build_context(
        self,
        query_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build structured context from query results.

        Returns:
            Dict containing:
            - context_text: Formatted context string
            - entities_used: List of entity names
            - sources: List of source URLs
            - metadata: Additional info
        """
        sections = []
        entities_used = []
        sources = []

        # Section 1: Community Context (high-level overview)
        community_section = self._build_community_section(
            query_result.get("communities", [])
        )
        if community_section:
            sections.append(community_section)

        # Section 2: Entity Information
        entity_section, used_entities = self._build_entity_section(
            query_result.get("similar_entities", []),
            query_result.get("entity_details", {})
        )
        if entity_section:
            sections.append(entity_section)
            entities_used.extend(used_entities)

        # Section 3: Relationships
        relationship_section = self._build_relationship_section(
            query_result.get("subgraph", {}).get("relationships", [])
        )
        if relationship_section:
            sections.append(relationship_section)

        # Section 4: Source Text Chunks
        chunk_section, chunk_sources = self._build_chunk_section(
            query_result.get("chunks", [])
        )
        if chunk_section:
            sections.append(chunk_section)
            sources.extend(chunk_sources)

        # Combine sections
        context_text = "\n\n".join(sections)

        # Truncate if too long
        if len(context_text) > self.max_context_chars:
            context_text = context_text[:self.max_context_chars] + "\n...[truncated]"

        return {
            "context_text": context_text,
            "entities_used": entities_used,
            "sources": sources,
            "metadata": {
                "total_entities": len(entities_used),
                "total_sources": len(sources),
                "context_length": len(context_text)
            }
        }

    def _build_community_section(
        self,
        communities: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Build community context section."""
        if not communities:
            return None

        lines = ["## Topic Overview"]

        for community in communities[:3]:  # Top 3 communities
            title = community.get("title", "Unknown")
            summary = community.get("summary", "")[:300]
            lines.append(f"### {title}")
            lines.append(summary)

        return "\n".join(lines)

    def _build_entity_section(
        self,
        similar_entities: List[Dict[str, Any]],
        entity_details: Dict[str, Dict[str, Any]]
    ) -> tuple:
        """Build entity information section."""
        if not similar_entities:
            return None, []

        lines = ["## Key Concepts"]
        used_entities = []

        for i, entity in enumerate(similar_entities[:self.max_entities]):
            entity_id = entity.get("entity_id", "")
            details = entity_details.get(entity_id, {})

            name = details.get("name", entity_id)
            entity_type = details.get("type", "CONCEPT")
            description = details.get("description", "")[:200]
            similarity = entity.get("similarity", 0)

            # Format entity
            lines.append(f"**{name}** ({entity_type}) [relevance: {similarity:.2f}]")
            if description:
                lines.append(f"  {description}")

            used_entities.append(name)

        return "\n".join(lines), used_entities

    def _build_relationship_section(
        self,
        relationships: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Build relationships section."""
        if not relationships:
            return None

        lines = ["## Concept Relationships"]

        # Group by relationship type
        by_type: Dict[str, List] = {}
        for rel in relationships[:self.max_relationships]:
            rel_type = rel.get("type", "RELATED_TO")
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)

        # Format relationships
        for rel_type, rels in sorted(by_type.items()):
            lines.append(f"### {rel_type.replace('_', ' ').title()}")
            for rel in rels[:10]:  # Max 10 per type
                source = rel.get("source", "?")
                target = rel.get("target", "?")
                desc = rel.get("description", "")
                if desc:
                    lines.append(f"- {source} -> {target}: {desc[:100]}")
                else:
                    lines.append(f"- {source} -> {target}")

        return "\n".join(lines)

    def _build_chunk_section(
        self,
        chunks: List[Dict[str, Any]]
    ) -> tuple:
        """Build source text section."""
        if not chunks:
            return None, []

        lines = ["## Source Material"]
        sources = []

        for i, chunk in enumerate(chunks[:self.max_chunks]):
            source_title = chunk.get("source_title", "Unknown Source")
            source_url = chunk.get("source_url", "")
            text = chunk.get("text", "")[:500]

            lines.append(f"### Source {i+1}: {source_title}")
            lines.append(text)

            if source_url:
                sources.append({
                    "title": source_title,
                    "url": source_url
                })

        return "\n".join(lines), sources

    def format_for_llm(
        self,
        query: str,
        context: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Format context for LLM input.

        Returns:
            Dict with 'system' and 'user' prompts
        """
        if system_prompt is None:
            system_prompt = """You are Erica, an expert AI tutor specializing in artificial intelligence and machine learning.

Your role is to:
1. Provide clear, accurate explanations of AI/ML concepts
2. Use the provided context to give grounded answers
3. Include relevant code examples when helpful
4. Cite sources when referencing specific information
5. Acknowledge if information is not in the context

When providing code examples, use Python and common ML libraries (PyTorch, TensorFlow, scikit-learn).
Keep explanations accessible but technically accurate."""

        user_prompt = f"""## Question
{query}

## Available Context
{context.get('context_text', '')}

## Instructions
Please answer the question using the context provided. If the context doesn't contain enough information, say so. Include relevant code examples if appropriate for the question.

If you reference specific information, mention the source."""

        return {
            "system": system_prompt,
            "user": user_prompt
        }
