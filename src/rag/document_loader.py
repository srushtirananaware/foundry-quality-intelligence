from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def load_documents():
    """
    Load all Markdown knowledge documents from the knowledge directory.
    """

    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.md"):

        content = file_path.read_text(
            encoding="utf-8"
        )

        documents.append({
            "source": file_path.name,
            "content": content,
        })

    return documents


def chunk_text(text):
    """
    Split Markdown text into meaningful sections.

    Each heading starts a new section, and the section
    keeps all text until the next heading.
    """

    sections = []
    current_section = []

    for line in text.splitlines():

        if line.startswith("#") and current_section:

            section_text = "\n".join(
                current_section
            ).strip()

            # Only keep sections that contain actual content
            lines = section_text.splitlines()

            if len(lines) > 1:
                sections.append(section_text)

            current_section = []

        current_section.append(line)

    # Add final section
    if current_section:

        section_text = "\n".join(
            current_section
        ).strip()

        lines = section_text.splitlines()

        if len(lines) > 1:
            sections.append(section_text)

    return sections


def load_and_chunk_documents():
    """
    Load all knowledge documents and split them into chunks.
    """

    documents = load_documents()

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["content"]
        )

        for index, chunk in enumerate(
            document_chunks
        ):

            chunks.append({
                "source": document["source"],
                "chunk_id": index,
                "content": chunk,
            })

    return chunks


if __name__ == "__main__":

    chunks = load_and_chunk_documents()

    print("=== KNOWLEDGE BASE ===")
    print()

    print(
        f"Total chunks: {len(chunks)}"
    )

    print()

    for chunk in chunks:

        print(
            f"[{chunk['source']} | "
            f"chunk {chunk['chunk_id']}]"
        )

        print(
            chunk["content"][:150]
            .replace("\n", " ")
        )

        print("-" * 60)