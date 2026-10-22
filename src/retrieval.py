
"""ChromaDB retrieval helpers for Amazon Nova embeddings."""

import chromadb
import numpy as np


def connect_collection(
    chroma_dir: str,
    collection_name: str = "mechanical_parts_nova",
):
    """Connect to the persistent Nova ChromaDB collection."""

    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Connect to the existing collection.
    # Avoid creating an empty collection if the name is incorrect.
    collection = client.get_collection(name=collection_name)

    # Validate the collection configuration.
    if collection.metadata:
        dimension = collection.metadata.get("embedding_dimension")
        if dimension is not None and dimension != 1024:
            raise ValueError(
                f"Expected 1024-dimensional Nova embeddings, "
                f"but collection metadata specifies {dimension}."
            )

    return client, collection


def query_similar(
    collection,
    query_embedding,
    top_k: int = 5,
    class_filter: list[str] | None = None,
):
    """Query Nova embeddings using cosine distance."""

    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    if query_embedding.shape != (1024,):
        raise ValueError(
            f"Expected a 1024-dimensional query embedding, "
            f"received shape {query_embedding.shape}."
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Optional class filtering
    where = None

    if class_filter:
        where = {"class_name": {"$in": class_filter}}

    # Avoid requesting more results than matching records.
    count = collection.count()

    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, count),
        where=where,
        include=["metadatas", "distances"],
    )

    matches = []

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for i, (record_id, metadata, distance) in enumerate(
        zip(ids, metadatas, distances)
    ):
        metadata = metadata or {}
        distance = float(distance)

        # ChromaDB cosine distance = 1 - cosine similarity.
        similarity = 1.0 - distance

        matches.append(
            {
                "rank": i + 1,
                "id": record_id,
                "similarity": similarity,
                "similarity_pct": similarity * 100.0,
                "cosine_distance": distance,
                "crop_id": metadata.get("crop_id", record_id),
                "class_name": metadata.get("class_name", "Unknown"),
                "sku": metadata.get("sku", "N/A"),
                "manufacturer": metadata.get("manufacturer", "N/A"),
                "manufacturer_part_number": metadata.get(
                    "manufacturer_part_number", "N/A"
                ),
                "data_origin": metadata.get("data_origin", "unknown"),
                "verification_status": metadata.get(
                    "verification_status", "unknown"
                ),
                "crop_path": metadata.get("crop_path", ""),
            }
        )

    return matches