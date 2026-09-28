from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the sentence-transformer embedding model.
    """

    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model, texts):
    """
    Convert texts into vector embeddings using
    the already-loaded embedding model.
    """

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":

    model = load_embedding_model()

    sample_texts = [
        "Casting defects can include porosity and shrinkage.",
        "Cycle time is a manufacturing process parameter."
    ]

    embeddings = create_embeddings(
        model,
        sample_texts
    )

    print()
    print("=== EMBEDDING TEST ===")
    print(
        "Number of embeddings:",
        len(embeddings)
    )

    print(
        "Embedding dimensions:",
        embeddings.shape[1]
    )