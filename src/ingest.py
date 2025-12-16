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
# PDF Ingestion Service
# -----------------------------------------------------
class PdfIngestionService:
    """
    Service responsible for idempotent ingestion of PDFs into a vector store (PGVector).
    """

    REQUIRED_ENV_VARS = (
        "GOOGLE_API_KEY",
        "GOOGLE_EMBEDDING_MODEL",
        "PGVECTOR_URL",
        "PGVECTOR_COLLECTION",
        "PDF_PATH",
    )

    def __init__(self):
        load_dotenv()
        self._validate_env()

        self.pdf_path = os.getenv("PDF_PATH")
        self.vector_store = self._create_vector_store()

    # -----------------------------------------------------
    # Validate environment variables
    # -----------------------------------------------------
    def _validate_env(self) -> None:
        missing = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    # -----------------------------------------------------
    # Initialize vector store
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
    # Generate deterministic document ID for idempotency
    # -----------------------------------------------------
    @staticmethod
    def _generate_document_id(document: Document) -> str:
        source = document.metadata.get("source", "")
        page = document.metadata.get("page", "")
        raw = f"{source}|{page}|{document.page_content}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"doc-{digest}"

    # -----------------------------------------------------
    # Load PDF documents
    # -----------------------------------------------------
    def _load_pdf(self) -> List[Document]:
        documents = PyPDFLoader(self.pdf_path).load()
        return documents

    # -----------------------------------------------------
    # Split documents into chunks
    # -----------------------------------------------------
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)
        return chunks

    # -----------------------------------------------------
    # Enrich documents with IDs
    # -----------------------------------------------------
    def _enrich_documents(self, documents: List[Document]) -> List[Document]:
        enriched: List[Document] = []
        for doc in documents:
            doc_id = self._generate_document_id(doc)
            enriched.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **{k: v for k, v in doc.metadata.items() if v not in ("", None)},
                        "document_id": doc_id,
                    },
                )
            )
        return enriched

    # -----------------------------------------------------
    # Run ingestion process
    # -----------------------------------------------------
    def run(self) -> None:
        try:
            documents = self._load_pdf()
            chunks = self._split_documents(documents)

            if not chunks:
                return

            enriched_docs = self._enrich_documents(chunks)
            ids = [doc.metadata["document_id"] for doc in enriched_docs]

            self.vector_store.add_documents(
                documents=enriched_docs,
                ids=ids,
            )

        except Exception as exc:
            raise

# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
if __name__ == "__main__":
    PdfIngestionService().run()