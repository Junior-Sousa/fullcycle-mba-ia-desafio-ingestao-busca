import os
import hashlib
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# -----------------------------------------------------
# Service responsible for ingestion of PDFs into a vector store (PGVector).
# -----------------------------------------------------

class PDFIngestionService:
    """
    Service responsible for ingesting PDFs into a PGVector store.
    """

    # Class variable
    REQUIRED_ENV_VARS = (
        "GOOGLE_API_KEY",
        "GOOGLE_EMBEDDING_MODEL",
        "PGVECTOR_URL",
        "PGVECTOR_COLLECTION",
        "PDF_PATH",
    )

    # Instance initializer (Constructor)
    def __init__(self):
        load_dotenv()
        self._validate_env() 

        self.pdf_path = os.getenv("PDF_PATH")
        self.vector_store = self._create_vector_store()

    # -----------------------------------------------------
    # Validate environment variables (Instance method)
    # -----------------------------------------------------
    def _validate_env(self) -> None:
        # Accessing class variable via self.
        missing = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)] 
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    # -----------------------------------------------------
    # Initialize vector store (Instance method)
    # -----------------------------------------------------
    def _create_vector_store(self) -> PGVector:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL")
        )

        return PGVector(
            embeddings=embeddings,
            collection_name=os.getenv("PGVECTOR_COLLECTION"),
            connection=os.getenv("PGVECTOR_URL"),
            use_jsonb=True,
        )

    # -----------------------------------------------------
    # Generate deterministic document ID (Static method)
    # -----------------------------------------------------
    @staticmethod
    def _generate_document_id(document: Document) -> str:
        source = document.metadata.get("source", "")
        page = document.metadata.get("page", "")
        raw = f"{source}|{page}|{document.page_content}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"doc-{digest}"

    # -----------------------------------------------------
    # Load PDF documents (Instance method)
    # -----------------------------------------------------
    def _load_pdf(self) -> List[Document]:
        # Uses instance attribute self.pdf_path
        documents = PyPDFLoader(self.pdf_path).load() 
        return documents

    # -----------------------------------------------------
    # Split documents into chunks (Instance method)
    # -----------------------------------------------------
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)
        return chunks

    # -----------------------------------------------------
    # Enrich documents with IDs (Instance method)
    # -----------------------------------------------------
    def _enrich_documents(self, documents: List[Document]) -> List[Document]:
        enriched: List[Document] = []
        for doc in documents:
            # Calls the static method
            doc_id = self._generate_document_id(doc) 
            enriched.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        # Filtering empty/None metadata values
                        **{k: v for k, v in doc.metadata.items() if v not in ("", None)},
                        "document_id": doc_id,
                    },
                )
            )
        return enriched

    # -----------------------------------------------------
    # Run ingestion process (Public instance method)
    # -----------------------------------------------------
    def ingest_pdf(self) -> None:
        try:
            print("Starting PDF ingestion process...")
            documents = self._load_pdf()
            chunks = self._split_documents(documents)

            if not chunks:
                print("No chunks generated. Exiting.")
                return

            enriched_docs = self._enrich_documents(chunks)
            ids = [doc.metadata["document_id"] for doc in enriched_docs]

            # Uses instance attribute self.vector_store
            self.vector_store.add_documents(
                documents=enriched_docs,
                ids=ids,
            )
            print(f"Successfully ingested {len(ids)} documents.")

        except Exception as exc:
            print(f"Error during ingestion: {exc}")
            raise

# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
if __name__ == "__main__":
    try:
        # 1. Create an instance of the class (runs __init__ and validation)
        ingestion_service = PDFIngestionService()
        
        # 2. Call the ingestion method on the instance
        ingestion_service.ingest_pdf()
        
    except RuntimeError as e:
        # Handles missing environment variables gracefully
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")