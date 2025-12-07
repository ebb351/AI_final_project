# AI Course Tutor - "Erica"

GraphRAG-based AI tutor for the "Introduction to AI" course using Knowledge Graphs, MongoDB, and local LLMs.

---

## Project Overview

**Erica** is an AI course tutor capable of answering questions, providing explanations, and assisting with learning using GraphRAG (Graph-based Retrieval-Augmented Generation). The system constructs a Knowledge Graph from course materials and uses local LLM inference (qwen2.5) to generate contextually-aware, pedagogically-sound responses.

### Key Features

- Knowledge Graph with 16,728 entities and 26,557 relationships
- Multi-source ingestion (HTML pages, PDFs)
- MongoDB persistence for graph storage
- Local LLM inference via Ollama (qwen2.5)
- Semantic entity retrieval with embeddings (24,219 vectors)
- Community detection with 6,934 summaries
- Docker deployment with automatic data initialization
- Citation support linking to source materials
- Web-based chat interface with LaTeX rendering

---

## Quick Start

### Prerequisites
- Docker and Docker Compose
- 8GB+ RAM available
- ~10GB disk space (for data + Ollama model)

### Step 1: Download Data

Download the pre-built knowledge graph data:

**[Download mongodb_export.zip](https://drive.google.com/file/d/1lBV8RMEooa2ohEWfiSF7cNJZayy4etXQ/view?usp=sharing)**

Extract to `data/mongodb_export/` directory.

### Step 2: Start Services

```bash
docker-compose up -d
```

On first startup, the system **automatically**:
- Imports 75,000+ documents into MongoDB
- Downloads the qwen2.5 LLM model (~4.7GB)
- Loads entity embeddings

Monitor progress: `docker-compose logs -f api`

### Step 3: Open Chat Interface

Open your browser to:

```
http://localhost:8000/chat
```

The chat interface provides:
- Example questions to get started
- Loading indicator during query processing (3-5 min on CPU)
- Formatted responses with markdown and LaTeX math rendering
- Source citations and entity links

### Step 4: Verify (Optional)

```bash
curl http://localhost:8000/health
```

Expected: `{"status": "healthy", "mongodb": "connected", "ollama": "ready", ...}`

### API Query (Alternative)

You can also query via the REST API:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is attention in transformers?"}'
```

---

## Architecture

The system uses a Docker Compose stack with three services:

**FastAPI Server (port 8000)**: Handles queries, generates embeddings, builds context, and orchestrates the pipeline. Includes a Gradio-powered chat interface at `/chat` for interactive use. On startup, automatically initializes MongoDB data and pulls the Ollama model if needed.

**MongoDB (port 27017)**: Stores the knowledge graph including text chunks, entities (graph nodes), relationships (graph edges), vector embeddings, and community reports.

**Ollama (port 11434)**: Provides local LLM inference using qwen2.5 (7B parameters) for answer generation.

### Query Pipeline

1. **Query Embedding**: Convert question to vector using nomic-embed-text
2. **Entity Retrieval**: Find top-k similar entities via cosine similarity
3. **Subgraph Expansion**: 2-hop graph traversal from seed entities
4. **Context Building**: Aggregate entities, relationships, chunks, communities
5. **LLM Generation**: Generate answer with citations using qwen2.5

---

## Milestone Summary

### M1: Environment and Tooling
- Docker Compose configuration for development environment
- Three-service stack: API, MongoDB, Ollama
- Automatic data initialization on startup

### M2: Data Ingestion
- 257 HTML pages scraped from course website
- 45 PDF documents extracted (3,289 pages)
- 302 content items totaling 33 MB

**Source URLs:**
- Course website: `https://pantelis.github.io/artificial-intelligence/`
- All subpages and linked resources automatically discovered
- PDF documents from course slides and referenced papers

The complete list of ingested URLs is stored in `data/course_data.json` (gitignored. Reproduce by running full pipeline. Or download [here](https://drive.google.com/file/d/1yybtcIXxP-p-iugDTJT7EatdUTPUEnv5/view?usp=sharing)). 

**Note**: YouTube video transcripts were not included because YouTube's rate limiting blocked programmatic transcript extraction. Manual extraction was possible but we prioritized reproducibility through automated pipelines. The HTML and PDF sources provided sufficient coverage of course material.

### M3: Knowledge Graph Construction
- 7,636 text chunks created (1200 tokens, 100 overlap)
- Entity extraction using Qwen2.5-3B (43,844 raw entities)
- Entity normalization and deduplication (16,728 unique entities)
- Relationship extraction (26,557 edges)
- Embedding generation (24,219 vectors) using nomic-embed-text
- Community detection using Leiden algorithm (6,934 communities)

**Knowledge Graph Schema:**

Nodes:
- `concept`: AI/ML concepts with title, description, and aliases
- `resource`: Source materials (PDF, web page) with URLs and spans
- `chunk`: Text segments linking concepts to source content

Edges:
- `prereq_of`: Prerequisite relationships between concepts (DAG structure)
- `explains`: Links resources to concepts they explain
- `related_to`: Near-transfer relationships (siblings, contrasts, is_a, part_of)

**GPU Processing**: Entity extraction and embedding generation were performed on [Modal](https://modal.com/), a serverless GPU compute platform. This was necessary because local CPU processing would take prohibitively long (estimated 20+ hours vs ~3.6 hours on GPU). Modal provides on-demand GPU access with simple Python decorators - see `modal_app/` for implementation. For quick start deployment, GPU processing is not needed since embeddings and entities are pre-computed.

### M4: Query and Generation
- Query processor with semantic entity retrieval
- 2-hop subgraph expansion for context
- Context builder aggregating entities, relationships, and source chunks
- Answer generator with Ollama integration
- Citation support with source URLs

**System Prompt:**

```
You are Erica, an expert AI tutor specializing in artificial intelligence and machine learning.

Your role is to:
1. Provide clear, accurate explanations of AI/ML concepts
2. Use the provided context to give grounded answers
3. Include relevant code examples when helpful (use Python)
4. Cite sources when referencing specific information
5. Acknowledge if information is not in the context

Format code examples with proper markdown code blocks.
Keep explanations accessible but technically accurate.
```

**Test Questions:**

The following test questions from the project specification are supported:

1. **"What is attention in transformers and can you provide a Python example?"**
   - Expected: Explains dot-product self-attention, Q/K/V vectors, code snippet
   - Entities used: Multi-head Attention, Self-Attention, Query/Key/Value, Transformer

2. **"What is CLIP and how is it used in computer vision applications?"**
   - Expected: Text/image encoders, contrastive loss, zero-shot classification
   - Entities used: CLIP, Contrastive Learning, Vision Encoder, Text Encoder

3. **"Can you explain the variational lower bound and how it relates to Jensen's inequality?"**
   - Expected: Jensen's inequality, variational inference, ELBO derivation, VAE losses
   - Entities used: Variational Lower Bound, Jensen's Inequality, ELBO, VAE

Each response includes the knowledge graph nodes used and citations to source materials.

### M5: Deployment Infrastructure
- Docker Compose with three-service stack (API, MongoDB, Ollama)
- Automatic MongoDB data initialization on first startup
- Automatic Ollama model pull if not present
- Health check endpoint for service monitoring

### M6: Frontend Interface
- Gradio-powered chat interface mounted at `/chat`
- Loading indicator during query processing
- Markdown rendering for formatted responses
- LaTeX math rendering for equations
- Source citations and entity concept links
- Example questions for quick start

---

## Data Statistics

### Knowledge Graph

| Collection | Count | Description |
|------------|-------|-------------|
| text_chunks | 7,491 | Source text segments |
| graph_nodes | 16,728 | Unique entities |
| graph_edges | 26,557 | Entity relationships |
| embeddings | 24,219 | Vector representations |
| community_reports | 6,934 | Cluster summaries |
| **Total** | **75,195** | **Documents in MongoDB** |

### Topics Covered

- Foundations (Linear Regression, SGD, MLE, Classification)
- Deep Neural Networks (Backprop, Batch Norm, Regularization)
- CNNs & Computer Vision (Object Detection, Segmentation)
- NLP & Transformers (Attention, Word2Vec, Language Models)
- Reinforcement Learning (MDPs, Value/Policy Iteration, Q-Learning)
- Planning (Task Planning, PDDL, Motion Planning)
- State Estimation (Kalman Filters, SLAM, Localization)
- Vision-Language Models (CLIP, BLIP-2, LLaVA, ViT)
- Robotics & Kinematics (Configuration Space, Wheeled Robots)
- Generative Models (VAE, Diffusion, Stable Diffusion)

---

## Project Structure

```
AI_final_project/
├── data/                              # All data files
│   ├── course_data.json               # Raw scraped data (33 MB)
│   ├── chunks_filtered.json           # Processed chunks (29 MB)
│   ├── extracted_entities.json        # Raw entities (13 MB)
│   └── mongodb_export/                # Docker deployment data (~451 MB)
│       ├── text_chunks.json
│       ├── graph_nodes.json
│       ├── graph_edges.json
│       ├── embeddings.json
│       └── community_reports.json
│
├── modal_app/                         # Modal.com GPU processing
│   ├── entity_extraction.py           # Entity extraction (Qwen2.5-3B)
│   ├── upload_chunks.py               # One-time script for data upload
│   └── embedding_generation.py        # Embedding generation
│
├── scripts/                           # Utility scripts
│   ├── chunk_documents.py             # Text chunking
│   ├── normalize_entities.py          # Entity deduplication
│   ├── import_to_mongodb.py           # Local MongoDB import
│   ├── export_mongodb.py              # Export to JSON
│   └── docker_import.py               # Docker MongoDB import
│
├── src/                               # Core application
│   ├── api/
│   │   └── server.py                  # FastAPI server
│   ├── frontend/
│   │   └── chat.py                    # Gradio chat interface
│   ├── query/
│   │   ├── processor.py               # Query processing
│   │   ├── context_builder.py         # Context assembly
│   │   └── generator.py               # Answer generation
│   ├── graph/
│   │   └── community_detection.py     # Leiden clustering
│   ├── database/
│   │   ├── schema.py                  # MongoDB schema
│   │   ├── storage.py                 # Graph storage
│   │   └── init_data.py               # Auto-initialization
│   ├── scraper/
│   │   ├── engine.py                  # Web crawler
│   │   └── extractors.py              # Content extractors
│   └── chunking/
│       └── text_chunker.py            # Token-based chunking
│
├── tests/
│   ├── test_query_system.py           # Query system tests
│   ├── test_entity_extraction_quality.py  # Entity extraction tests
│   └── test_storage.py                # MongoDB storage adapter tests
│
├── docker-compose.yml                 # Docker services
├── Dockerfile                         # API container
├── DEPLOYMENT.md                      # Deployment guide
└── requirements.txt                   # Python dependencies
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Redirects to chat interface |
| GET | `/chat` | Interactive chat interface |
| GET | `/health` | Service health check |
| GET | `/api` | API info and version |
| POST | `/query` | Answer questions (JSON API) |

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "mongodb": "connected",
  "ollama": "ready",
  "embeddings_loaded": 24219,
  "model": "qwen2.5"
}
```

### Query Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is attention in transformers?",
    "max_entities": 20,
    "max_chunks": 10
  }'
```

### Query Response

```json
{
  "answer": "Attention in transformers is a mechanism that allows...",
  "entities_used": ["Multi-head Attention", "Cross Attention", "..."],
  "citations": [
    {"title": "The Annotated Transformer", "url": "https://..."}
  ],
  "model": "qwen2.5",
  "latency_ms": 265275
}
```

---

## Deployment

### Docker Deployment

See `DEPLOYMENT.md` for full instructions.

```bash
# Download data to data/mongodb_export/ first, then:
docker-compose up -d

# System auto-initializes on first start
# Monitor with: docker-compose logs -f api
```

### Full Pipeline Rebuild (Advanced)

For rebuilding the knowledge graph from source (requires Modal account for GPU):

1. `python -m src.scraper.engine` - Scrape course materials
2. `python scripts/chunk_documents.py` - Create text chunks
3. `modal run modal_app/entity_extraction.py` - Extract entities (GPU)
4. `python scripts/normalize_entities.py` - Deduplicate entities
5. `modal run modal_app/embedding_generation.py` - Generate embeddings (GPU)
6. `python -m src.graph.community_detection` - Run Leiden clustering
7. `python scripts/import_to_mongodb.py` - Import to MongoDB
8. `python scripts/export_mongodb.py` - Export for Docker

---

## Performance

### Query Latency (Docker on CPU)

| Stage | Time |
|-------|------|
| Query embedding | ~0.5s |
| Entity retrieval | ~0.5s |
| Subgraph expansion | ~1s |
| Context building | ~0.5s |
| LLM generation | 200-300s |
| **Total** | **~3-5 minutes** |

### Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 10 GB | 15 GB |
| CPU | 4 cores | 8 cores |
| GPU | None | NVIDIA (for speed) |

---

## Technical Stack

### Data Processing
- `requests`, `beautifulsoup4` - Web scraping
- `pypdf` - PDF extraction
- `tiktoken` - Token-based chunking
- `modal` - Serverless GPU compute

### Machine Learning
- `transformers` - Qwen2.5-3B for entity extraction
- `sentence-transformers` - nomic-embed-text for embeddings
- `networkx`, `graspologic` - Leiden community detection
- `ollama` - Local LLM inference (qwen2.5 7B)

### Infrastructure
- `mongodb` / `pymongo` - Graph database
- `fastapi`, `uvicorn` - API server
- `gradio` - Chat interface with LaTeX support
- `docker`, `docker-compose` - Containerization
- `httpx` - HTTP client

---

## Known Limitations

1. **YouTube Transcripts**: Not included due to YouTube rate limiting on programmatic access
2. **CPU Inference**: LLM generation is slow (~4-5 min) without GPU
3. **Memory**: Requires 8GB+ RAM for embedding model + LLM

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Port 27017 in use | Stop local MongoDB: `brew services stop mongodb-community` |
| Slow first startup | Normal - auto-imports data and downloads model |
| Model pull timeout | Manual pull: `docker exec graphrag-ollama ollama pull qwen2.5` |
| Empty query results | Check logs: `docker-compose logs api` |
| High memory usage | Reduce `max_entities` in queries |

### Useful Commands

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# Restart services
docker-compose restart

# Full cleanup (removes all data)
docker-compose down -v
```

---

## References

- **Course Website**: https://pantelis.github.io/
- **GraphRAG Paper**: Microsoft Research
- **nano-graphrag**: https://github.com/gusye1234/nano-graphrag
- **Ollama**: https://ollama.ai/
- **Modal**: https://modal.com/

---

## License

Educational project for AI course final.
