"""
Upload chunks to Modal volume (one-time operation).
"""
import modal
from pathlib import Path

from .config import DATA_VOLUME_NAME

app = modal.App("upload-chunks")
volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)


@app.local_entrypoint()
def upload():
    """Upload chunks.json to Modal volume."""
    project_root = Path(__file__).parent.parent
    chunks_file = project_root / "data" / "chunks.json"

    if not chunks_file.exists():
        print(f"Chunks file not found: {chunks_file}")
        print("Run: python src/chunking/text_chunker.py first")
        return 1

    print(f"Uploading {chunks_file} to Modal volume '{DATA_VOLUME_NAME}'...")

    # Upload file
    with volume.batch_upload() as batch:
        batch.put_file(str(chunks_file), "/data/chunks.json")

    print("Upload complete!")
    return 0
