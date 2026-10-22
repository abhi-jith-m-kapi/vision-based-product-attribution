"""Shared constants and path resolution for the Streamlit app."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# YOLO object detection model. Override with YOLO_MODEL_PATH when needed.
YOLO_MODEL_PATH = Path(
    os.environ.get("YOLO_MODEL_PATH", BASE_DIR / "models" / "machine_parts_best.pt")
).expanduser()

# Amazon Nova ChromaDB configuration
CHROMA_DIR = BASE_DIR / "data" / "chroma_nova_db"
COLLECTION_NAME = "mechanical_parts_nova"

# Amazon Nova multimodal embedding configuration
EMBEDDING_MODEL_NAME = "amazon.nova-2-multimodal-embeddings-v1:0"
EMBEDDING_DIM = 1024

# Mechanical part classes
CLASS_NAMES = {
    0: "Bearing",
    1: "Bolt",
    2: "Gear",
    3: "Nut",
}

# Supported image formats
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
