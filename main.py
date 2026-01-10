import os
import argparse
from typing import List, Sequence, Optional, Tuple
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found. Please add it to your .env file")


def load_documents(file_path: str) -> List:
    """Load documents from a PDF file and return a list of Document objects.

    Args:
        file_path: Path to the PDF file to load.

    Returns:
        A list of documents as returned by `PyPDFLoader`.
    """
    if not file_path:
        raise ValueError("file_path must be provided")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path=file_path)
    return loader.load()


def document_chunking(docs: Sequence, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """Split loaded documents into chunks.

    Args:
        docs: Sequence of documents to split.
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        A list of split Document objects.
    """
    if not docs:
        raise ValueError("docs is empty")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return text_splitter.split_documents(docs)


def get_embeddings(splits: Sequence, model: str = "models/gemini-embedding-001") -> List[List[float]]:
    """Return embeddings for a sequence of splits.

    Uses a batch method if available on the embedding client to improve throughput.
    """
    if not splits:
        raise ValueError("No splits provided")

    embeddings = GoogleGenerativeAIEmbeddings(model=model)

    # Prefer batch method if available
    if hasattr(embeddings, "embed_documents"):
        texts = [getattr(s, "page_content", str(s)) for s in splits]
        return embeddings.embed_documents(texts)

    # Fall back to calling embed_query per item
    vectors = []
    for s in splits:
        text = getattr(s, "page_content", str(s))
        vectors.append(embeddings.embed_query(text))
    return vectors


def upsert_splits(
    pc: Pinecone,
    index_name: str,
    splits: Sequence,
    id_prefix: Optional[str] = None,
    batch_size: int = 100,
) -> int:
    """Create embeddings and upsert them into Pinecone in batches.

    Args:
        pc: Pinecone client instance.
        index_name: Name of the Pinecone index.
        splits: Sequence of split Document objects to upsert.
        id_prefix: Optional prefix for vector IDs.
        batch_size: Number of vectors to upsert per batch.

    Returns:
        Number of vectors upserted.
    """
    if not splits:
        raise ValueError("No document splits provided to upsert.")

    embeddings_client = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    total_upserted = 0

    def _do_upsert(batch_items: List[dict]):
        # Try different Pinecone client styles
        if hasattr(pc, "upsert"):
            try:
                pc.upsert(index_name, batch_items)
                return
            except TypeError:
                # Fallback to Index object
                pass

        if hasattr(pc, "Index"):
            idx_client = pc.Index(index_name)
            idx_client.upsert(batch_items)
            return

        raise RuntimeError("Unable to find a supported upsert method on the Pinecone client.")

    batch: List[dict] = []
    for i, doc in enumerate(splits):
        text = getattr(doc, "page_content", str(doc))
        # Try batch embedding if available
        try:
            vec = embeddings_client.embed_query(text)
        except Exception:
            # as a fallback, try the generic get_embeddings for single item
            vec = get_embeddings([doc])[0]

        vector_id = f"{id_prefix + '-'}{index_name}-{i}" if id_prefix else f"{index_name}-{i}"
        batch.append({"id": vector_id, "values": vec, "metadata": {"chunk_text": text}})

        if len(batch) >= batch_size:
            _do_upsert(batch)
            total_upserted += len(batch)
            batch = []

    if batch:
        _do_upsert(batch)
        total_upserted += len(batch)

    print(f"Upserted {total_upserted} vectors to index '{index_name}'")
    return total_upserted


def vector_db(
    api_key: Optional[str] = None,
    index_name: str = "developer-quickstart-py",
    cloud: str = "aws",
    region: str = "us-east-1",
    embed_model: str = "models/gemini-embedding-001",
) -> Tuple[Pinecone, str]:
    """Create or return a Pinecone client and ensure the index exists.

    Args:
        api_key: Pinecone API key (falls back to PINECONE_API_KEY env var).
        index_name: Name of the index to ensure.
        cloud: Cloud provider for index creation.
        region: Region for index creation.
        embed_model: The embedding model to declare in the index metadata.

    Returns:
        A tuple of (pinecone_client, index_name)
    """
    api_key = api_key or os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not found. Please add it to your .env file")

    pc = Pinecone(api_key=api_key)

    # Create the index if it doesn't already exist
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud=cloud,
            region=region,
            embed={"model": embed_model, "field_map": {"text": "chunk_text"}},
        )
        print(f"Created index '{index_name}'")
    else:
        print(f"Index '{index_name}' already exists")

    return pc, index_name


def _parse_args():
    parser = argparse.ArgumentParser(description="Index PDF content into Pinecone using Gemini embeddings")
    parser.add_argument("--file", "-f", help="Path to the PDF file to index", required=True)
    parser.add_argument("--index", "-i", help="Pinecone index name", default="developer-quickstart-py")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    parser.add_argument("--upsert", action="store_true", help="Perform upsert into Pinecone")
    parser.add_argument("--api-key", help="Pinecone API key (overrides env var)")
    return parser.parse_args()


def main():
    args = _parse_args()

    docs = load_documents(args.file)
    splits = document_chunking(docs)
    print(f"Prepared {len(splits)} document chunks")

    pc, index_name = vector_db(api_key=args.api_key, index_name=args.index)

    if args.dry_run:
        print("Dry run: no upsert performed")
        return

    if args.upsert:
        upsert_splits(pc, index_name, splits)


if __name__ == "__main__":
    main()
