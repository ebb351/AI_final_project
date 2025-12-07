# Deployment Guide - GraphRAG AI Course Tutor (Erica)

This guide covers deployment of the Erica AI tutor system.

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available for containers
- ~2GB disk space for MongoDB data
- ~5GB disk space for Ollama model

---

## Quick Deploy (3 Steps)

### Step 1: Download Data Files

Download the pre-built knowledge graph data from Google Drive:

**[Download mongodb_export.zip](https://drive.google.com/file/d/1lBV8RMEooa2ohEWfiSF7cNJZayy4etXQ/view?usp=sharing)**

Extract and place the files in `data/mongodb_export/`:
```
data/mongodb_export/
├── text_chunks.json
├── graph_nodes.json
├── graph_edges.json
├── embeddings.json (~400MB)
└── community_reports.json
```

### Step 2: Start Docker Services

```bash
docker-compose up -d
```

On first startup, the system automatically:
1. Detects empty MongoDB and imports all data (~75,000 documents)
2. Pulls the qwen2.5 LLM model if not present (~4.7GB download)
3. Loads entity embeddings into memory

Monitor startup progress:
```bash
docker-compose logs -f api
```

Wait for: `Application startup complete`

### Step 3: Verify Deployment

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "mongodb": "connected",
  "ollama": "ready",
  "embeddings_loaded": 24219,
  "model": "qwen2.5"
}
```

---

## Query the System

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is attention in transformers?"}'
```

First query takes 3-5 minutes on CPU. Subsequent queries are faster once models are loaded.

---

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI GraphRAG server |
| mongodb | 27017 | MongoDB database |
| ollama | 11434 | Ollama LLM server |

---

## Startup Behavior

The API container performs automatic initialization on startup:

1. **MongoDB Check**: If all required collections are empty, automatically imports data from `data/mongodb_export/`
2. **Ollama Model Check**: If qwen2.5 model is not present, automatically pulls it
3. **Embedding Load**: Pre-loads 16,728 entity embeddings into memory

This means:
- First startup takes longer (data import + model download)
- Subsequent startups are fast (data persists in Docker volumes)
- No manual import commands needed

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Redirects to chat interface |
| GET | `/chat` | Interactive chat interface |
| GET | `/health` | Health check |
| GET | `/api` | API info |
| POST | `/query` | Answer questions (JSON API) |

### Query Request

```json
{
  "question": "What is attention in transformers?",
  "max_entities": 20,
  "max_chunks": 10
}
```

### Query Response

```json
{
  "answer": "Attention in transformers...",
  "entities_used": ["Multi-Head Attention", "..."],
  "citations": [{"title": "...", "url": "..."}],
  "model": "qwen2.5",
  "latency_ms": 250000
}
```

---

## Troubleshooting

### Startup Issues

**Check service status:**
```bash
docker-compose ps
```

**View API logs:**
```bash
docker-compose logs -f api
```

### MongoDB Issues

If data import fails:
```bash
# Manual import (if auto-import didn't work)
python scripts/docker_import.py
```

### Ollama Model Issues

If model pull times out or fails:
```bash
# Manual model pull
docker exec graphrag-ollama ollama pull qwen2.5

# Verify model exists
docker exec graphrag-ollama ollama list
```

### Port Conflicts

If port 27017 is in use:
```bash
# Stop local MongoDB (macOS)
brew services stop mongodb-community

# Or change port in docker-compose.yml
```

### High Memory Usage

The embedding model requires ~2GB RAM. Reduce `max_entities` in queries if needed:
```json
{"question": "...", "max_entities": 10}
```

---

## Stopping Services

```bash
# Stop all services (data preserved)
docker-compose down

# Stop and remove volumes (deletes all data!)
docker-compose down -v
```

---

## Data Persistence

- **MongoDB data**: Docker volume `mongodb_data`
- **Ollama models**: Docker volume `ollama_models`
- **Source data**: `./data/mongodb_export/` (mounted read-only)

After `docker-compose down`, data persists in volumes. After `docker-compose down -v`, all data is deleted and will be re-imported on next startup.

---

## Full Pipeline Rebuild (Advanced)

For rebuilding the knowledge graph from source materials, see the "Full Pipeline Rebuild" section in README.md. This requires:
- Modal account for GPU processing
- Python 3.11+ environment
- Docker for MongoDB
