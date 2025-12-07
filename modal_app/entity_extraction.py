"""
Modal App for Entity Extraction using Qwen2.5-3B on L4 GPUs

This module runs entity and relationship extraction on Modal's serverless platform.
Uses HuggingFace Transformers with Qwen2.5-3B-Instruct for LLM inference.
"""
import json
import re
from typing import List, Dict, Any
import modal

# Configuration - ALL VALUES INLINED (no imports from modal_app.config)
APP_NAME = "graphrag-entity-extraction"
GPU_TYPE = "L4"
ENTITY_EXTRACTION_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ENTITY_EXTRACTION_TIMEOUT = 1800  # 30 minutes
MAX_RETRIES = 2
BATCH_SIZE_PER_GPU = 25
MAX_PARALLEL_CALLS = 10  # User's Modal limit
DATA_VOLUME_NAME = "graphrag-data"
MODEL_CACHE_VOLUME_NAME = "huggingface-cache"

# Prompts
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

# Create Modal app
app = modal.App(APP_NAME)

# Create volumes for data and model cache
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
model_cache = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.40.0",
        "torch>=2.2.0",
        "accelerate>=0.28.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "bitsandbytes>=0.42.0",  # For 8-bit quantization
    )
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=ENTITY_EXTRACTION_TIMEOUT,
    volumes={
        "/data": data_volume,
        "/cache": model_cache,
    },
    retries=MAX_RETRIES,
    scaledown_window=120,
)
def extract_entities_batch(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract entities and relationships from a batch of text chunks.

    This function runs on Modal with L4 GPU. It processes multiple chunks
    in a single call to amortize model loading time.

    Args:
        chunks: List of chunk dictionaries with 'id', 'text', 'source_id', etc.

    Returns:
        List of extraction results with entities and relationships
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Processing batch of {len(chunks)} chunks on {GPU_TYPE}...")
    print(f"Model: {ENTITY_EXTRACTION_MODEL}")

    # Load model and tokenizer (cached in container)
    print(f"Loading model: {ENTITY_EXTRACTION_MODEL}")
    print("Using float16 precision")

    tokenizer = AutoTokenizer.from_pretrained(
        ENTITY_EXTRACTION_MODEL,
        cache_dir="/cache",
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        ENTITY_EXTRACTION_MODEL,
        cache_dir="/cache",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Model loaded. Device: {model.device}")

    # Process each chunk
    results = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {idx}/{len(chunks)}: {chunk['id']}")

        try:
            # Extract entities and relationships
            result = extract_from_chunk(model, tokenizer, chunk)
            results.append(result)
            print(f"  Extracted {len(result['entities'])} entities, {len(result['relationships'])} relationships")

        except Exception as e:
            print(f"  Error processing chunk {chunk['id']}: {e}")
            # Return empty result with error
            results.append({
                "chunk_id": chunk["id"],
                "source_id": chunk["source_id"],
                "entities": [],
                "relationships": [],
                "error": str(e)
            })

    print(f"Batch processing complete: {len(results)} results")
    return results


def extract_from_chunk(
    model,
    tokenizer,
    chunk: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract entities and relationships from a single chunk.

    Args:
        model: Loaded LLM model
        tokenizer: Model tokenizer
        chunk: Chunk dictionary

    Returns:
        Extraction result
    """
    import torch

    # Construct prompt
    user_prompt = ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE.format(text=chunk["text"])

    # Format for Qwen chat template
    messages = [
        {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # Apply chat template and tokenize
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Parse JSON from response
    try:
        # Try to find JSON in the response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
        else:
            # No JSON found
            print(f"  Warning: No JSON found in response: {response[:200]}...")
            result = {"entities": [], "relationships": []}

    except json.JSONDecodeError as e:
        print(f"  Warning: JSON decode error: {e}")
        print(f"  Response: {response[:500]}...")
        result = {"entities": [], "relationships": []}

    # Validate and clean result
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])

    # Ensure entities have required fields
    valid_entities = []
    for entity in entities:
        if isinstance(entity, dict) and "name" in entity and "type" in entity:
            valid_entities.append(entity)

    # Ensure relationships have required fields
    valid_relationships = []
    entity_names = {e["name"] for e in valid_entities}
    for rel in relationships:
        if (isinstance(rel, dict) and
            "source" in rel and "target" in rel and "type" in rel and
            rel["source"] in entity_names and rel["target"] in entity_names):
            valid_relationships.append(rel)

    return {
        "chunk_id": chunk["id"],
        "source_id": chunk["source_id"],
        "source_url": chunk.get("source_url", ""),
        "entities": valid_entities,
        "relationships": valid_relationships,
    }


@app.local_entrypoint()
def process_all_chunks():
    """
    Local entrypoint: Load chunks, process in batches on Modal, save results.

    This function runs locally and orchestrates the Modal processing.
    Uses filtered chunks (with giant notebooks removed).
    """
    import json
    from pathlib import Path

    # Paths - USE FILTERED CHUNKS (giant notebooks removed)
    project_root = Path(__file__).parent.parent
    chunks_file = project_root / "data" / "chunks_filtered.json"
    output_file = project_root / "data" / "extracted_entities.json"

    print("=" * 80)
    print("ENTITY EXTRACTION - Modal Processing")
    print("=" * 80)
    print(f"Chunks file: {chunks_file}")
    print(f"Output file: {output_file}")
    print(f"GPU type: {GPU_TYPE}")
    print(f"Batch size: {BATCH_SIZE_PER_GPU} chunks per GPU")
    print("=" * 80)

    # Load chunks
    print("\nLoading chunks...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # Split into batches
    batches = []
    for i in range(0, len(chunks), BATCH_SIZE_PER_GPU):
        batch = chunks[i:i + BATCH_SIZE_PER_GPU]
        batches.append(batch)

    print(f"Created {len(batches)} batches ({BATCH_SIZE_PER_GPU} chunks/batch)")
    print(f"\nProcessing on Modal with up to {MAX_PARALLEL_CALLS} parallel {GPU_TYPE} GPUs...")

    # Calculate estimated time
    time_per_batch_min = (BATCH_SIZE_PER_GPU * 25) / 60  # 25 seconds per chunk (3B model)
    batches_per_track = (len(batches) + MAX_PARALLEL_CALLS - 1) // MAX_PARALLEL_CALLS  # ceiling division
    wall_clock_time_min = batches_per_track * time_per_batch_min
    total_gpu_hours = len(batches) * (time_per_batch_min / 60)
    estimated_cost = total_gpu_hours * 0.80  # L4 GPU is $0.80/hour
    print(f"Estimated wall-clock time: {wall_clock_time_min:.1f} minutes ({wall_clock_time_min/60:.1f} hours)")
    print(f"Estimated total GPU-hours: {total_gpu_hours:.1f}")
    print(f"Estimated cost: ${estimated_cost:.2f}")

    # Process batches in parallel using .map()
    all_results = []
    for i, batch_results in enumerate(extract_entities_batch.map(batches, return_exceptions=True)):
        if isinstance(batch_results, Exception):
            print(f"Batch {i+1}/{len(batches)} failed: {batch_results}")
        else:
            all_results.extend(batch_results)
            print(f"Batch {i+1}/{len(batches)} complete ({len(batch_results)} results)")

    # Save results
    print(f"\nSaving {len(all_results)} results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Statistics
    total_entities = sum(len(r["entities"]) for r in all_results)
    total_relationships = sum(len(r["relationships"]) for r in all_results)
    errors = sum(1 for r in all_results if "error" in r)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Chunks processed: {len(all_results)}")
    print(f"Total entities: {total_entities}")
    print(f"Total relationships: {total_relationships}")
    print(f"Errors: {errors}")
    if len(all_results) > 0:
        print(f"Average entities/chunk: {total_entities / len(all_results):.2f}")
        print(f"Average relationships/chunk: {total_relationships / len(all_results):.2f}")
    print("=" * 80)

    return all_results
