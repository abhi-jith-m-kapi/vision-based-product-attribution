# src/attribute_extraction.py

import os
import json
import re
from io import BytesIO
from pathlib import Path
from functools import lru_cache

import boto3
from dotenv import load_dotenv
from PIL import Image


# Load the existing project configuration.
PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env", override=False)

DEFAULT_MODEL_ID = "global.anthropic.claude-haiku-5-5"


def _get_bedrock_client():
    """Initialize Bedrock using the project's bearer-token authentication."""

    api_key = os.getenv("BEDROCK_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "BEDROCK_API_KEY is missing. Check the project's .env file."
        )

    # Use the same authentication method as the successful notebook test.
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

    region = os.getenv("AWS_REGION", "us-east-1")

    return boto3.client(
        "bedrock-runtime",
        region_name=region,
    )


@lru_cache(maxsize=1)
def _get_client():
    """Reuse the Bedrock client across Streamlit reruns."""
    return _get_bedrock_client()


def _parse_json_response(text: str) -> dict:
    """Parse JSON even if the model wraps it in Markdown code fences."""

    text = text.strip()

    # Remove optional Markdown code fences.
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # Parse the response directly first.
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: locate the JSON object in surrounding text.
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Claude did not return a JSON object. "
                f"Response begins with: {text[:200]!r}"
            )

        result = json.loads(text[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from Claude.")

    return result


def extract_attributes(
    image: Image.Image,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 1800,
) -> dict:
    """
    Extract structured attributes from a mechanical-part image using Claude Haiku.

    Args:
        image: PIL image containing one mechanical part.
        model_id: Amazon Bedrock Claude model or inference-profile ID.
        max_tokens: Maximum response tokens.

    Returns:
        Dictionary containing the extracted attributes.

    Raises:
        TypeError: If image is not a PIL Image.
        ValueError: If the image or model response is invalid.
        EnvironmentError: If the Bedrock API key is missing.
        Exception: Bedrock API errors are propagated to the caller.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")

    if image.width == 0 or image.height == 0:
        raise ValueError("The supplied image is empty.")

    # Convert to RGB and encode as JPEG for Bedrock.
    rgb_image = image.convert("RGB")

    buffer = BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()

    prompt = """
You are a mechanical engineering visual inspection assistant.

Analyze the supplied image of a mechanical part and extract its attributes.

Return exactly one valid JSON object with this structure:

{
  "part_type": {
    "value": null,
    "confidence": "low",
    "evidence": ""
  },
  "geometry": {
    "value": null,
    "confidence": "low",
    "evidence": ""
  },
  "material": {
    "value": null,
    "confidence": "low",
    "evidence": ""
  },
  "dimensions": {
    "value": null,
    "confidence": "low",
    "evidence": ""
  },
  "manufacturer": {
    "value": null,
    "confidence": "low",
    "evidence": ""
  },
  "manufacturer_part_number": {
    "value": null,
    "confidence": "low",
    "evidence": ""
  },
  "visible_markings": [],
  "physical_features": [],
  "additional_observations": [],
  "unresolved_attributes": []
}

Extraction rules:

1. Return only valid JSON. Do not include Markdown fences or explanations
   outside the JSON object.

2. Confidence must be "high", "medium", or "low". These are qualitative
   assessments, not calibrated probabilities.

3. Describe the visible part type, geometry, shape, holes, threads, grooves,
   teeth, edges, and other relevant physical features.

4. Report dimensions only if supported by reliable visual evidence or an
   explicitly supplied measurement reference. Do not estimate exact dimensions
   from pixels or appearance alone.

5. Do not guess the manufacturer, manufacturer part number, material grade,
   or engineering specifications.

6. Distinguish direct visual observations from interpretations in the evidence
   field. For uncertain interpretations, explicitly state the uncertainty.

7. Transcribe markings only when they are actually legible.

8. Use null for an attribute that cannot be determined. Add unresolved
   attributes to the unresolved_attributes list.

9. Do not infer an identity from the image filename.

10. Treat the supplied image as the only source of evidence.
"""

    client = _get_client()

    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": image_bytes},
                        }
                    },
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
        },
    )

    content = response["output"]["message"]["content"]

    response_text = "".join(
        block["text"]
        for block in content
        if "text" in block
    ).strip()

    if not response_text:
        raise ValueError(
            "Claude returned no text. "
            f"Stop reason: {response.get('stopReason', 'unknown')}"
        )

    attributes = _parse_json_response(response_text)

    # Basic schema validation.
    required_fields = [
        "part_type",
        "geometry",
        "material",
        "dimensions",
        "manufacturer",
        "manufacturer_part_number",
        "visible_markings",
        "physical_features",
        "additional_observations",
        "unresolved_attributes",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in attributes
    ]

    if missing_fields:
        raise ValueError(
            f"Claude response is missing fields: {missing_fields}"
        )

    return attributes