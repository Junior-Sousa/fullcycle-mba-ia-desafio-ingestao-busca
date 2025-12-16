import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

# -----------------------------------------------------
# Prompt Template for Retrieval-Augmented Generation (RAG)
# -----------------------------------------------------
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

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

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

# -----------------------------------------------------
# Environment Configuration
# -----------------------------------------------------
# Load environment variables
load_dotenv()

REQUIRED_ENV_VARS_SEARCH = (
    "GOOGLE_API_KEY",
    "GOOGLE_EMBEDDING_MODEL",
    "PGVECTOR_URL",
    "PGVECTOR_COLLECTION",
)

def _validate_env_search() -> None:
    """Validates if the necessary environment variables are present."""
    missing = [var for var in REQUIRED_ENV_VARS_SEARCH if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Missing environment variables required for search: {', '.join(missing)}"
        )

def _create_vector_store() -> PGVector:
    """Initializes and returns the PGVector instance."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL")
    )

    # Note: We don't pass create_table=True as we assume the table was
    # created during ingestion.
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )

def _format_docs(docs: list[Document]) -> str:
    """Formats the retrieved documents into a single context string."""
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def search_prompt() -> Optional[Runnable]:
    """
    Creates and returns the LangChain Runnable Chain for search and response generation.
    Returns None in case of configuration error.
    """
    try:
        _validate_env_search()
        
        # 1. Initialize the Vector Store (Retriever)
        vector_store = _create_vector_store()

        # The retriever uses similarity search with score (similarity_search_with_score)
        # internally, configured to return k=10 documents.
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # 2. Initialize the LLM (Google Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # Model optimized for chat/RAG tasks
            temperature=0.0, # Low temperature for factual answers
        )

        # 3. Create the Prompt Template
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        # 4. Create the RAG Chain
        # The chain flow:
        #   - Input (question) -> 
        #   - Retriever (searches docs and formats) -> 
        #   - LLM (generates response with prompt) ->
        #   - Output Parser (returns string)
        rag_chain = (
            {
                "contexto": retriever | _format_docs, # Search and format the context
                "pergunta": RunnablePassthrough(), # Passes the user's question
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("Search module successfully initialized.")
        return rag_chain

    except RuntimeError as e:
        print(f"Configuration Error in search.py: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during search initialization: {e}")
        return None

# Optional block for quick testing of the search function (not needed for the chat)
if __name__ == "__main__":
    chain = search_prompt()
    if chain:
        # Example usage
        question = "Qual é a descrição do documento?"
        print(f"\n[TEST] Question: {question}")
        response = chain.invoke(question)
        print(f"[TEST] Answer: {response}")