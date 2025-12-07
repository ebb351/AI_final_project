"""
Configuration for Modal.com GPU Processing

This module contains all configuration settings for running entity extraction
on Modal's serverless GPU platform.

Key Settings:
- GPU: L4 ($0.80/hr) - chosen for cost optimization over A100 ($2.10/hr)
- Model: Qwen2.5-3B-Instruct - chosen for speed/cost over 7B model
- Batch Size: 25 chunks/batch - optimized for L4 GPU memory constraints
- Parallelization: 15 concurrent GPUs - maximizes throughput within Modal limits
"""
import os

# Optional: load .env file if available (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available in Modal container

# Modal settings
APP_NAME = "graphrag-entity-extraction"
MODAL_ENVIRONMENT = os.getenv("MODAL_ENVIRONMENT", "dev")

# GPU configuration
GPU_TYPE = "L4"  # Cost-optimized GPU ($0.80/hr vs A100 $2.10/hr)
GPU_COUNT = 1  # GPUs per function call

# Batch processing
BATCH_SIZE_PER_GPU = 25  # Reduced to 25 for L4 GPU timeout constraints
MAX_PARALLEL_CALLS = 15  # Maximum parallel GPU instances (high parallelization for speed)

# Timeouts (in seconds)
ENTITY_EXTRACTION_TIMEOUT = 1800  # 30 minutes per batch
EMBEDDING_TIMEOUT = 600  # 10 minutes per batch

# Retry configuration
MAX_RETRIES = 2  # Retry failed chunks
RETRY_DELAY = 5  # Seconds before retry

# Model configuration
ENTITY_EXTRACTION_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # HuggingFace model ID (3B for speed)
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # Embedding model

# Volume names
DATA_VOLUME_NAME = "graphrag-data"
MODEL_CACHE_VOLUME_NAME = "huggingface-cache"

# Model quantization
USE_8BIT_QUANTIZATION = False  # Set to True for A100 40GB to save memory
USE_FLOAT16 = True  # Use half-precision (faster, saves memory)

# Prompt templates
# These prompts guide the LLM to extract entities and relationships from course chunks.
# The prompts enforce JSON output format and specify entity/relationship types.
ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph builder specializing in AI, machine learning, and computer science education.

Your task is to extract entities and relationships from course material text.

**Entity Types:**
- CONCEPT: Abstract ideas, theories, methodologies
- ALGORITHM: Specific algorithms and techniques
- TOOL: Software, frameworks, libraries
- PERSON: Historical figures, researchers
- MATHEMATICAL_CONCEPT: Mathematical formulas, theorems
- RESOURCE: Course materials, papers, books
- EXAMPLE: Worked examples, case studies

**Relationship Types:**
- PREREQUISITE_FOR: A is required knowledge for B
- COMPONENT_OF: A is part of B
- SOLVES: Algorithm/method solves problem
- APPLIES_TO: Concept applies to domain
- NEAR_TRANSFER: Related/sibling concepts
- CONTRASTS_WITH: Opposing concepts
- IS_A: Type hierarchy
- EXPLAINS: Resource explains concept
- EXEMPLIFIES: Example demonstrates concept

**Output Format:**
Return ONLY valid JSON with this exact structure:
{
  "entities": [
    {"name": "Entity Name", "type": "CONCEPT|ALGORITHM|...", "description": "Clear, concise description"}
  ],
  "relationships": [
    {"source": "Entity A", "target": "Entity B", "type": "PREREQUISITE_FOR|...", "description": "Why/how they're related"}
  ]
}

**Guidelines:**
1. Extract 5-15 entities per chunk
2. Focus on key concepts, not trivial details
3. Use full, proper names (e.g., "Convolutional Neural Network" not "CNN")
4. Create 2-3x as many relationships as entities
5. Be specific in descriptions
6. Ensure source and target entities exist in the entities list
"""

ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE = """Extract entities and relationships from this AI/ML course text:

{text}

Return only the JSON output:"""

# Embedding configuration
EMBEDDING_BATCH_SIZE = 32  # Embeddings per batch (can be larger)
EMBEDDING_MAX_LENGTH = 8192  # nomic-embed-text supports 8192 tokens

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
ENTITIES_OUTPUT_FILE = os.path.join(DATA_DIR, "extracted_entities.json")
EMBEDDINGS_OUTPUT_FILE = os.path.join(DATA_DIR, "extracted_embeddings.json")
