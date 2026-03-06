# Ingestao e Busca Semantica com LangChain e Postgres

Projeto para ingestao de PDF em PostgreSQL com `pgvector` e busca semantica via CLI, usando LangChain.

## Estrutura

```text
.
|- docker-compose.yml
|- requirements.txt
|- .env.example
|- src/
|  |- ingest.py
|  |- search.py
|  \- chat.py
\- document.pdf
```

## Requisitos

- Python 3.11+
- Docker e Docker Compose
- API Key de um provedor:
- OpenAI (`OPENAI_API_KEY`)
- Google Gemini (`GOOGLE_API_KEY`)

## 1) Configurar ambiente

```bash
cp .env.example .env
```

Edite o arquivo `.env` e configure pelo menos:

- `EMBEDDING_PROVIDER=openai` ou `gemini`
- `LLM_PROVIDER=openai` ou `gemini`
- API key correspondente ao provider escolhido

Modelos padrao configurados no `.env.example`:

- OpenAI embeddings: `text-embedding-3-small`
- OpenAI LLM: `gpt-5-nano`
- Gemini embeddings: `models/embedding-001`
- Gemini LLM: `gemini-2.5-flash-lite`

## 2) Subir banco com pgvector

```bash
docker compose up -d
```

Banco padrao:

- Host: `localhost`
- Porta: `6024`
- Usuario: `langchain`
- Senha: `langchain`
- DB: `langchain`

## 3) Instalar dependencias

```bash
pip install -r requirements.txt
```

## 4) Ingerir o PDF

Coloque o arquivo `document.pdf` na raiz do projeto (ou configure `PDF_PATH` no `.env`).

```bash
python3 src/ingest.py --recreate-collection
```

Regras aplicadas na ingestao:

- Chunk size: `1000`
- Overlap: `150`
- Embeddings de cada chunk
- Armazenamento em PostgreSQL + pgvector

## 5) Chat via CLI

```bash
python3 src/chat.py
```

Exemplo:

```text
Faca sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhoes de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Nao tenho informacoes necessarias para responder sua pergunta.
```

## Observacoes

- O prompt de resposta em `src/search.py` foi implementado com as regras de restricao de contexto.
- Para trocar de provedor (OpenAI/Gemini), altere as variaveis no `.env`.
