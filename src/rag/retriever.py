import json
from pathlib import Path

import numpy as np

from .document_loader import load_and_chunk_documents
from .embeddings import (
    load_embedding_model,
    create_embeddings
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAG_DIR = PROJECT_ROOT / "data" / "rag"

CHUNKS_PATH = RAG_DIR / "chunks.json"
EMBEDDINGS_PATH = RAG_DIR / "embeddings.npy"


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    numerator = np.dot(
        vector_a,
        vector_b
    )

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def build_knowledge_index(model):
    """
    Load knowledge chunks and create embeddings.
    """

    chunks = load_and_chunk_documents()

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        model,
        texts
    )

    return chunks, embeddings


def save_knowledge_index(
    chunks,
    embeddings
):
    """
    Save knowledge chunks and embeddings to disk.
    """

    RAG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    print()
    print("Knowledge index saved.")
    print(
        f"Chunks: {CHUNKS_PATH}"
    )
    print(
        f"Embeddings: {EMBEDDINGS_PATH}"
    )


def load_knowledge_index():
    """
    Load previously generated chunks and embeddings.
    """

    if not CHUNKS_PATH.exists():
        return None, None

    if not EMBEDDINGS_PATH.exists():
        return None, None

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    return chunks, embeddings


def search_knowledge(
    query,
    chunks,
    embeddings,
    model,
    top_k=3
):
    """
    Find the most semantically similar knowledge chunks.
    """

    query_embedding = create_embeddings(
        model,
        [query]
    )[0]

    similarities = []

    for index, embedding in enumerate(
        embeddings
    ):

        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        similarities.append(
            (index, similarity)
        )

    similarities.sort(
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for index, similarity in similarities[:top_k]:

        results.append({
            "source": chunks[index]["source"],
            "chunk_id": chunks[index]["chunk_id"],
            "content": chunks[index]["content"],
            "similarity": float(similarity)
        })

    return results

def build_quality_query(quality_report):
    """
    Create a knowledge-search query from the structured
    quality report.
    """

    process_quality = (
        quality_report["process"]["predicted_quality"]
    )

    process_confidence = (
        quality_report["process"]["confidence"]
    )

    visual_prediction = (
        quality_report["visual"]["prediction"]
    )

    visual_confidence = (
        quality_report["visual"]["confidence"]
    )

    status = (
        quality_report["decision"]["status"]
    )

    query = (
        f"Process model predicts Quality {process_quality} "
        f"with {process_confidence:.2%} confidence. "
        f"Visual model predicts {visual_prediction} "
        f"with {visual_confidence:.2%} confidence. "
        f"The overall status is {status}. "
        f"What should be investigated and what relevant "
        f"foundry knowledge should be considered?"
    )

    return query

if __name__ == "__main__":

    print("=== FOUNDRY KNOWLEDGE RETRIEVER ===")
    print()

    print("Loading embedding model...")

    model = load_embedding_model()

    print()

    # Try loading an existing index
    chunks, embeddings = load_knowledge_index()

    if chunks is not None:

        print("Existing knowledge index found.")
        print("Loading index from disk...")

    else:

        print("No existing knowledge index found.")
        print("Building knowledge index...")

        chunks, embeddings = build_knowledge_index(
            model
        )

        save_knowledge_index(
            chunks,
            embeddings
        )

    print()

    print(
        f"Knowledge chunks: {len(chunks)}"
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    # Test query
    query = (
        "The process and visual models disagree. "
        "What should I investigate?"
    )

    print()
    print("QUERY")
    print("-----")
    print(query)

    results = search_knowledge(
        query,
        chunks,
        embeddings,
        model,
        top_k=3
    )

    print()
    print("TOP RETRIEVED KNOWLEDGE")
    print("-----------------------")

    for index, result in enumerate(
        results,
        start=1
    ):

        print()

        print(
            f"{index}. "
            f"{result['source']} "
            f"(chunk {result['chunk_id']})"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            result["content"]
        )

        print("-" * 60)