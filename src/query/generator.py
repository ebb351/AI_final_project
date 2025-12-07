#!/usr/bin/env python3
"""
Answer Generator for GraphRAG

Generates answers using Ollama with structured context.
"""
import logging
import os
import re
import subprocess
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnswerResponse:
    """Response from answer generation."""
    answer: str
    entities_used: List[str]
    sources: List[Dict[str, str]]
    model: str
    success: bool
    error: Optional[str] = None


class AnswerGenerator:
    """Generate answers using Ollama."""

    def __init__(
        self,
        model: str = "qwen2.5",
        ollama_host: Optional[str] = None,
        timeout: int = 300
    ):
        """Initialize answer generator."""
        self.model = model
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

    def _call_ollama_subprocess(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Call Ollama using subprocess."""
        try:
            # Build full prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            else:
                full_prompt = prompt

            result = subprocess.run(
                ["ollama", "run", self.model, full_prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                logger.error(f"Ollama error: {result.stderr}")
                return ""

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            logger.error(f"Ollama timeout after {self.timeout}s")
            return ""
        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama.")
            return ""
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""

    def _call_ollama_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Call Ollama using HTTP API."""
        try:
            import httpx

            url = f"{self.ollama_host}/api/generate"

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 2000
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()

                result = response.json()
                return result.get("response", "")

        except ImportError:
            logger.warning("httpx not available, falling back to subprocess")
            return self._call_ollama_subprocess(prompt, system_prompt)
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return self._call_ollama_subprocess(prompt, system_prompt)

    def generate_answer(
        self,
        query: str,
        context: Dict[str, Any],
        use_api: bool = True
    ) -> AnswerResponse:
        """
        Generate answer for query using context.

        Args:
            query: User question
            context: Context from ContextBuilder
            use_api: Use HTTP API (True) or subprocess (False)

        Returns:
            AnswerResponse with answer and metadata
        """
        # Get prompts from context
        context_text = context.get("context_text", "")
        entities_used = context.get("entities_used", [])
        sources = context.get("sources", [])

        # Build system prompt
        system_prompt = """You are Erica, an expert AI tutor specializing in artificial intelligence and machine learning.

Your role is to:
1. Provide clear, accurate explanations of AI/ML concepts
2. Use the provided context to give grounded answers
3. Include relevant code examples when helpful (use Python)
4. Cite sources when referencing specific information
5. Acknowledge if information is not in the context

Format code examples with proper markdown code blocks.
Keep explanations accessible but technically accurate."""

        # Build user prompt
        user_prompt = f"""## Question
{query}

## Available Context
{context_text}

## Instructions
Please answer the question using the context provided. If the context doesn't contain enough information, acknowledge that and provide what you can.

Include relevant Python code examples if appropriate for the question.
If you reference specific information from the sources, mention it."""

        # Generate answer
        logger.info(f"Generating answer with {self.model}...")

        if use_api:
            answer = self._call_ollama_api(user_prompt, system_prompt)
        else:
            answer = self._call_ollama_subprocess(user_prompt, system_prompt)

        if not answer:
            return AnswerResponse(
                answer="I apologize, but I encountered an error generating a response. Please try again.",
                entities_used=entities_used,
                sources=sources,
                model=self.model,
                success=False,
                error="Failed to generate response"
            )

        # Post-process answer
        answer = self._postprocess_answer(answer)

        return AnswerResponse(
            answer=answer,
            entities_used=entities_used,
            sources=sources,
            model=self.model,
            success=True
        )

    def _postprocess_answer(self, answer: str) -> str:
        """Post-process generated answer."""
        # Remove any "System:" or "User:" artifacts
        answer = re.sub(r'^(System:|User:)\s*', '', answer, flags=re.MULTILINE)

        # Clean up excessive newlines
        answer = re.sub(r'\n{3,}', '\n\n', answer)

        return answer.strip()

    def extract_citations(
        self,
        answer: str,
        available_sources: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Extract citations from answer text."""
        citations = []

        # Check if any source titles are mentioned
        for source in available_sources:
            title = source.get("title", "")
            if title and title.lower() in answer.lower():
                citations.append(source)

        return citations

    def check_ollama_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and model is available."""
        # Try HTTP API first (works in Docker environments)
        try:
            import httpx

            url = f"{self.ollama_host}/api/tags"
            with httpx.Client(timeout=10) as client:
                response = client.get(url)
                response.raise_for_status()

                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]

                return {
                    "available": True,
                    "models": model_names,
                    "target_model_available": any(self.model in m for m in model_names)
                }

        except ImportError:
            pass  # Fall through to subprocess
        except Exception as e:
            # API failed, try subprocess fallback
            logger.debug(f"Ollama API check failed: {e}, trying subprocess")

        # Fallback to subprocess (works for local Ollama CLI)
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {"available": False, "error": "Ollama not responding"}

            models = result.stdout.strip().split("\n")
            model_names = [m.split()[0] for m in models[1:] if m.strip()]

            return {
                "available": True,
                "models": model_names,
                "target_model_available": any(self.model in m for m in model_names)
            }

        except subprocess.TimeoutExpired:
            return {"available": False, "error": "Ollama timeout"}
        except FileNotFoundError:
            return {"available": False, "error": "Ollama not installed"}
        except Exception as e:
            return {"available": False, "error": str(e)}
