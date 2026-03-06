import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_embeddings(provider: str):
    if provider == "openai":
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)

    if provider == "gemini":
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    raise ValueError(
        "EMBEDDING_PROVIDER invalido. Use 'openai' ou 'gemini'."
    )


def run_ingestion(pdf_path: str, recreate_collection: bool) -> None:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    connection = os.getenv(
        "PGVECTOR_CONNECTION",
        "postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
    )
    collection_name = os.getenv("PGVECTOR_COLLECTION", "pdf_chunks")
    use_jsonb = _as_bool(os.getenv("PGVECTOR_USE_JSONB", "true"), default=True)

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PDF nao encontrado em '{pdf_path}'. Ajuste PDF_PATH ou passe --pdf."
        )

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["source_file"] = path.name

    embeddings = get_embeddings(provider)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=use_jsonb,
        pre_delete_collection=recreate_collection,
    )

    ids = [f"{path.stem}-chunk-{i}" for i in range(len(chunks))]
    vector_store.add_documents(chunks, ids=ids)

    print("Ingestao concluida com sucesso.")
    print(f"PDF: {path}")
    print(f"Paginas carregadas: {len(pages)}")
    print(f"Chunks gerados: {len(chunks)}")
    print(f"Collection: {collection_name}")
    print(f"Embedding provider: {provider}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ingestao de PDF para PostgreSQL + pgVector usando LangChain"
    )
    parser.add_argument(
        "--pdf",
        default=os.getenv("PDF_PATH", "document.pdf"),
        help="Caminho do arquivo PDF para ingestao",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Recria (apaga e cria novamente) a collection antes de inserir",
    )

    args = parser.parse_args()
    run_ingestion(pdf_path=args.pdf, recreate_collection=args.recreate_collection)


if __name__ == "__main__":
    main()