import os
import json
import base64
import io
import boto3
import numpy as np

from PIL import Image
from dotenv import load_dotenv

# Load configuration from .env
load_dotenv()

MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
REGION = os.getenv("AWS_REGION", "us-east-1")
EMBEDDING_DIMENSION = 1024


def load_embedding(model_name: str = MODEL_ID):
    """Initialize the Amazon Nova embedding client."""

    api_key = os.getenv("BEDROCK_API_KEY")

    if not api_key:
        raise ValueError(
            "BEDROCK_API_KEY is missing from the .env file."
        )

    # Configure Bedrock bearer-token authentication.
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
    )

    return client, model_name, EMBEDDING_DIMENSION


def embed_image(
    client,
    model_name: str,
    dimension: int,
    image: Image.Image,
) -> np.ndarray:
    """
    Generate an Amazon Nova image embedding.

    Returns:
        L2-normalized float32 vector with the configured dimension.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.Image object.")

    image = image.convert("RGB")

    # Encode the image as PNG bytes.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    request_body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": dimension,
            "image": {
                "format": "png",
                "source": {
                    "bytes": base64.b64encode(
                        image_bytes
                    ).decode("utf-8")
                },
            },
        },
    }

    response = client.invoke_model(
        modelId=model_name,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())

    embedding = np.asarray(
        result["embeddings"][0]["embedding"],
        dtype=np.float32,
    )

    if embedding.shape != (dimension,):
        raise ValueError(
            f"Expected embedding shape ({dimension},), "
            f"received {embedding.shape}."
        )

    # L2 normalization for consistent cosine similarity handling.
    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Nova returned a zero-vector embedding.")

    embedding = embedding / norm

    return embedding.astype(np.float32)