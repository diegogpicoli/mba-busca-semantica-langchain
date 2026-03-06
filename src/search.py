import argparse
import os

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector


FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."

PROMPT_TEMPLATE = """CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
    "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUARIO:
{question}

RESPONDA A "PERGUNTA DO USUARIO"""  # noqa: E501


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

    raise ValueError("EMBEDDING_PROVIDER invalido. Use 'openai' ou 'gemini'.")


def get_llm(provider: str):
    if provider == "openai":
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano")
        return ChatOpenAI(model=model, temperature=0)

    if provider == "gemini":
        model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    raise ValueError("LLM_PROVIDER invalido. Use 'openai' ou 'gemini'.")


def get_vector_store(embeddings) -> PGVector:
    connection = os.getenv(
        "PGVECTOR_CONNECTION",
        "postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
    )
    collection_name = os.getenv("PGVECTOR_COLLECTION", "pdf_chunks")
    use_jsonb = _as_bool(os.getenv("PGVECTOR_USE_JSONB", "true"), default=True)

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=use_jsonb,
    )


def _build_context(results) -> str:
    if not results:
        return ""

    lines = []
    for i, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page", "?")
        chunk = doc.metadata.get("chunk_index", "?")
        lines.append(
            f"[Trecho {i} | page={page} | chunk={chunk} | score={score:.6f}]\n{doc.page_content}"
        )

    return "\n\n".join(lines)


def answer_question(question: str, k: int = 10) -> tuple[str, list]:
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    embeddings = get_embeddings(embedding_provider)
    vector_store = get_vector_store(embeddings)
    results = vector_store.similarity_search_with_score(query=question, k=k)

    context = _build_context(results)
    if not context.strip():
        return FALLBACK_ANSWER, results

    llm = get_llm(llm_provider)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)

    answer = (response.content or "").strip()
    if not answer:
        return FALLBACK_ANSWER, results

    return answer, results


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Busca semantica no PGVector e resposta com LLM"
    )
    parser.add_argument("question", help="Pergunta a ser respondida")
    parser.add_argument(
        "--k",
        type=int,
        default=int(os.getenv("TOP_K", "10")),
        help="Quantidade de documentos mais relevantes",
    )

    args = parser.parse_args()
    answer, _ = answer_question(question=args.question, k=args.k)
    print(answer)


if __name__ == "__main__":
    main()