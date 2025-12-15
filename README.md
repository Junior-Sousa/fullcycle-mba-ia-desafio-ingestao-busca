# 🚀 Desafio MBA Engenharia de Software com IA: Ingestão e Busca Semântica com LangChain e PostgreSQL

Este desafio visa construir um sistema de Q&A (Perguntas e Respostas) baseado em um documento PDF, utilizando o framework **LangChain** e o banco de dados vetorial **pgVector** no PostgreSQL.

## 🎯 Objetivo

Desenvolver um software com duas funcionalidades principais:

1.  **Ingestão:** Ler um arquivo PDF e persistir seus conteúdos como *embeddings* (vetores) em um banco de dados PostgreSQL com pgVector.
2.  **Busca (CLI):** Criar uma interface de linha de comando (CLI) que permita ao usuário fazer perguntas e receber respostas fundamentadas **exclusivamente** no conteúdo do PDF ingerido.

### Exemplo de Interação na CLI

| Tipo | Pergunta | Resposta |
| :--- | :--- | :--- |
| **No Contexto** | `Qual o faturamento da Empresa SuperTechIABrazil?` | `O faturamento foi de 10 milhões de reais.` |
| **Fora do Contexto** | `Quantos clientes temos em 2024?` | `Não tenho informações necessárias para responder sua pergunta.` |

---

## ✅ Requisitos Detalhados

### 1. Ingestão do PDF (`src/ingest.py`)

* O PDF (`document.pdf`) deve ser dividido usando as seguintes configurações:
    * **Chunks:** `1000` caracteres.
    * **Overlap:** `150` caracteres.
* Cada *chunk* deve ser vetorizado e armazenado no PostgreSQL com `pgVector` usando o componente `PGVector`. 

### 2. Consulta via CLI (`src/chat.py` & `src/search.py`)

* Deve ser implementado um script Python que simula um *chat* no terminal.
* **Fluxo RAG (Retrieval-Augmented Generation):** A pergunta do usuário deve ser vetorizada e usada para buscar os **10 resultados mais relevantes (k=10)** no banco.
* A LLM deve ser chamada com um prompt que inclui o contexto recuperado e segue as **REGRAS** estritas abaixo.

#### 📝 Prompt a ser Utilizado

Este é o template do prompt que será enviado ao LLM, contendo o contexto recuperado e as regras de resposta:

```markdown
CONTEXTO: {resultados concatenados do banco de dados}

REGRAS:

Responda somente com base no CONTEXTO.

Se a informação não estiver explicitamente no CONTEXTO, responda: "Não tenho informações necessárias para responder sua pergunta."

Nunca invente ou use conhecimento externo.

Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO: Pergunta: "Qual é a capital da França?" Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?" Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO: {pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
```

---

## ⚙️ Tecnologias Obrigatórias

| Categoria | Tecnologia |
| :--- | :--- |
| **Linguagem** | Python |
| **Framework** | LangChain |
| **Banco de Dados** | PostgreSQL + pgVector |
| **Execução DB** | Docker & Docker Compose |

## 📦 Pacotes Recomendados (LangChain)

| Módulo | Componente |
| :--- | :--- |
| **Split** | `RecursiveCharacterTextSplitter` |
| **Embeddings (Gemini)** | `GoogleGenerativeAIEmbeddings` |
| **PDF Loader** | `PyPDFLoader` |
| **Vector Store** | `PGVector` |
| **Busca** | `similarity_search_with_score(query, k=10)` |

## 🔑 Configuração API Key

### Gemini (Google)

* **Modelo de Embeddings:** `models/embedding-001`
* **Modelo de LLM para Resposta:** `gemini-2.5-flash-lite`

As chaves de API devem ser configuradas em um arquivo `.env` (baseado no `.env.example`).

---

## 📂 Estrutura Obrigatória do Projeto

```
├── docker-compose.yml
├── requirements.txt      # Dependências
├── .env.example          # Template da variável GOOGLE_API_KEY
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca
│   └── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução
```

---

## 🛠️ Pré-requisitos

Para executar este projeto, você precisará ter o seguinte software instalado em seu ambiente:

* **Python 3.8+:** Necessário para rodar os scripts (`python` ou `python3`).
* **Docker e Docker Compose:** Essencial para subir o banco de dados PostgreSQL com a extensão `pgVector`.

---

## 🚀 Ordem de Execução

### 1. Configurar Ambiente Virtual e Dependências

Esta etapa garante que o projeto utilize as versões corretas de Python e das bibliotecas necessárias.

| Item | Ação | Comandos (Linux/macOS) | Comandos (Windows Powershell) |
| :--- | :--- | :--- | :--- |
| **Criação do Venv** | Criar um ambiente virtual isolado. | `python3 -m venv venv` | `python -m venv venv` |
| **Ativação** | Ativar o ambiente virtual. | `source venv/bin/activate` | `.\venv\Scripts\activate` |
| **Instalação** | Instalar todas as dependências do `requirements.txt`. | `pip install -r requirements.txt` | `pip install -r requirements.txt` |

#### O procedimento abaixo pode demorar alguns minutos ...

```bash
fullcycle-mba-ia-desafio-ingestao-busca$ 
fullcycle-mba-ia-desafio-ingestao-busca$ python -m venv venv
fullcycle-mba-ia-desafio-ingestao-busca$ source venv/bin/activate
(venv) fullcycle-mba-ia-desafio-ingestao-busca$ pip install -r requirements.txt 
Collecting aiohappyeyeballs==2.6.1 (from -r requirements.txt (line 1))
  Using cached aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)

Installing collected packages: filetype, zstandard, urllib3, typing_extensions, tqdm, tenacity, sniffio, regex, PyYAML, python-dotenv, pypdf, pyasn1, psycopg2-binary, psycopg-binary, protobuf, propcache, packaging, orjson, numpy, mypy_extensions, multidict, jsonpointer, jiter, idna, httpx-sse, h11, grpcio, greenlet, frozenlist, distro, charset-normalizer, certifi, cachetools, attrs, asyncpg, annotated-types, aiohappyeyeballs, yarl, typing-inspection, typing-inspect, SQLAlchemy, rsa, requests, pydantic_core, pyasn1_modules, psycopg-pool, psycopg, proto-plus, pgvector, marshmallow, jsonpatch, httpcore, googleapis-common-protos, anyio, aiosignal, tiktoken, requests-toolbelt, pydantic, httpx, grpcio-status, google-auth, dataclasses-json, aiohttp, pydantic-settings, openai, langsmith, google-api-core, langchain-core, langchain-text-splitters, langchain-postgres, langchain-openai, google-ai-generativelanguage, langchain-google-genai, langchain, langchain-community

```


### 2. Subir o Banco de Dados

Utilize o Docker Compose fornecido para inicializar o PostgreSQL com `pgVector`:

```bash
docker compose up -d
```

#### Resultado Esperado
```bash
fullcycle-mba-ia-desafio-ingestao-busca$ docker compose up -d
[+] Running 4/4
 ✔ Network fullcycle-mba-ia-desafio-ingestao-busca_default                   Created                                                                                                                                                    0.1s
 ✔ Volume fullcycle-mba-ia-desafio-ingestao-busca_postgres_data              Created                                                                                                                                                    0.0s
 ✔ Container postgres_rag                                                    Healthy                                                                                                                                                   11.6s
 ✔ Container fullcycle-mba-ia-desafio-ingestao-busca-bootstrap_vector_ext-1  Started
```

#### Banco de Dados

```bash
(venv) fullcycle-mba-ia-desafio-ingestao-busca$ docker ps
CONTAINER ID   IMAGE                    COMMAND                  CREATED        STATUS                  PORTS                                         NAMES
a9994da3a871   pgvector/pgvector:pg17   "docker-entrypoint.s…"   18 hours ago   Up 18 hours (healthy)   0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp   postgres_rag

(venv) fullcycle-mba-ia-desafio-ingestao-busca$ docker exec -it postgres_rag psql -U postgres
psql (17.6 (Debian 17.6-1.pgdg12+1))
Type "help" for help.

postgres=# \l
                                                    List of databases
   Name    |  Owner   | Encoding | Locale Provider |  Collate   |   Ctype    | Locale | ICU Rules |   Access privileges   
-----------+----------+----------+-----------------+------------+------------+--------+-----------+-----------------------
 postgres  | postgres | UTF8     | libc            | en_US.utf8 | en_US.utf8 |        |           | 
 rag       | postgres | UTF8     | libc            | en_US.utf8 | en_US.utf8 |        |           | 
 template0 | postgres | UTF8     | libc            | en_US.utf8 | en_US.utf8 |        |           | =c/postgres          +
           |          |          |                 |            |            |        |           | postgres=CTc/postgres
 template1 | postgres | UTF8     | libc            | en_US.utf8 | en_US.utf8 |        |           | =c/postgres          +
           |          |          |                 |            |            |        |           | postgres=CTc/postgres
(4 rows)

postgres=#

```

### 3. Executar Ingestão do PDF:

```bash
python src/ingest.py
```

### 4. Rodar o Chat

```bash
python src/chat.py
```

---

## 🛠️ Troubleshootings

### 1. The container name "/postgres_rag" is already in use by container

```bash
 ✘ Container postgres_rag                                        Error response from daemon: Conflict. The container name "/postgres_rag" is already in use by container "d960959a7a24e84a98828cd0c1d8d54b4...                          0.0s
Error response from daemon: Conflict. The container name "/postgres_rag" is already in use by container "d960959a7a24e84a98828cd0c1d8d54b443482a899cb296a4d25b7713e826c12". You have to remove (or rename) that container to be able to reuse that name.
```

Este erro ocorre porque o Docker não consegue criar um novo container com o nome postgres_rag, pois já existe um container (parado ou em execução) utilizando esse nome.

Para resolver, você deve remover o container antigo antes de tentar inicializar um novo.

**Remova o container existente com o nome em conflito:**

```bash
docker rm postgres_rag
```

*Dica:* Se o container estiver em execução, use a flag -f para forçar a remoção: docker rm -f postgres_rag

**Tente executar sua aplicação Docker novamente (por exemplo, usando docker-compose):**

```bash
docker-compose up -d
```