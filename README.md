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

## 🚀 Ordem de Execução

### 1. Configurar Ambiente Virtual e Dependências

Esta etapa garante que o projeto utilize as versões corretas de Python e das bibliotecas necessárias.

| Item | Ação | Comandos (Linux/macOS) | Comandos (Windows Powershell) |
| :--- | :--- | :--- | :--- |
| **Criação do Venv** | Criar um ambiente virtual isolado. | `python3 -m venv venv` | `python -m venv venv` |
| **Ativação** | Ativar o ambiente virtual. | `source venv/bin/activate` | `.\venv\Scripts\activate` |
| **Instalação** | Instalar todas as dependências do `requirements.txt`. | `pip install -r requirements.txt` | `pip install -r requirements.txt` |


2.  **Subir o Banco de Dados:** `docker compose up -d`

Utilize o Docker Compose fornecido para inicializar o PostgreSQL com `pgVector`:

```bash
docker compose up -d

3.  **Executar Ingestão do PDF:** `python src/ingest.py`
4.  **Rodar o Chat:** `python src/chat.py`