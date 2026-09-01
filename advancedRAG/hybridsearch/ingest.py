from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

from config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
)


BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "document.pdf"


def load_pdf(pdf_path: str):
    """
    Load the PDF and return its pages.
    """

    loader = PyPDFLoader(pdf_path)

    return loader.load()


def split_documents(
    documents: list[Document],
):
    """
    Split documents into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    return splitter.split_documents(
        documents
    )


def create_vectorstore(
    documents: list[Document],
):
    """
    Create and persist the Chroma vector store.
    """

    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    return vectorstore


def main():

    print("Loading PDF...")

    documents = load_pdf(
        PDF_PATH
    )

    print(
        f"Loaded {len(documents)} pages."
    )

    print("Splitting documents...")

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Creating Chroma vector store...")

    create_vectorstore(
        chunks
    )

    print("Vector store created successfully.")


if __name__ == "__main__":
    main()