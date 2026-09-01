from langchain_chroma import Chroma
from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)
from langchain_community.retrievers import (
    BM25Retriever,
)
from langchain_core.documents import Document

from config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RERANKER_MODEL,
    VECTOR_TOP_K,
    BM25_TOP_K,
    HYBRID_CANDIDATE_K,
    FINAL_TOP_K,
)

from pipeline import HybridRAGPipeline
from reranker import DocumentReranker


def load_vectorstore():
    """
    Load the existing Chroma vector store.
    """

    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    return vectorstore


def create_vector_retriever(vectorstore):
    """
    Create the dense vector retriever.
    """

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": VECTOR_TOP_K,
        },
    )


def create_bm25_retriever(vectorstore):
    """
    Create the BM25 sparse retriever.

    BM25 operates on document text rather than
    vector embeddings.
    """

    data = vectorstore.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    documents = []

    for content, metadata in zip(
        data["documents"],
        data["metadatas"],
    ):
        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata=metadata or {},
            )
        )

    if not documents:
        raise ValueError(
            "No documents found in the vector store."
        )

    bm25_retriever = BM25Retriever.from_documents(
        documents
    )

    bm25_retriever.k = BM25_TOP_K

    return bm25_retriever


def create_llm():
    """
    Create the LLM used for answer generation.
    """

    return ChatMistralAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def create_reranker():
    """
    Create the Cross-Encoder reranker.
    """

    return DocumentReranker(
        model_name=RERANKER_MODEL,
        top_k=FINAL_TOP_K,
    )


def create_pipeline():
    """
    Initialize all RAG components and construct
    the complete Hybrid RAG pipeline.
    """

    # -----------------------------------------
    # Vector store
    # -----------------------------------------

    vectorstore = load_vectorstore()

    # -----------------------------------------
    # Retrievers
    # -----------------------------------------

    vector_retriever = create_vector_retriever(
        vectorstore
    )

    bm25_retriever = create_bm25_retriever(
        vectorstore
    )

    # -----------------------------------------
    # LLM
    # -----------------------------------------

    llm = create_llm()

    # -----------------------------------------
    # Reranker
    # -----------------------------------------

    reranker = create_reranker()

    # -----------------------------------------
    # Complete pipeline
    # -----------------------------------------

    return HybridRAGPipeline(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        llm=llm,
        candidate_k=HYBRID_CANDIDATE_K,
    )


def main():

    print("Initializing Hybrid RAG pipeline...")

    pipeline = create_pipeline()

    print("Pipeline ready.")

    while True:

        query = input(
            "\nAsk a question (type 'exit' to quit): "
        ).strip()

        if query.lower() == "exit":
            print("Exiting...")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:

            # =========================================
            # DEBUG RETRIEVAL
            # =========================================

            print("\nRunning retrieval pipeline...")

            debug = pipeline.debug_retrieval(
                query
            )

            # =========================================
            # VECTOR SEARCH
            # =========================================

            print(
                "\n========== VECTOR SEARCH =========="
            )

            for i, doc in enumerate(
                debug["vector"],
                start=1,
            ):
                print(
                    f"\n{i}. "
                    f"{doc.page_content[:300]}..."
                )

            # =========================================
            # BM25
            # =========================================

            print(
                "\n========== BM25 SEARCH =========="
            )

            for i, doc in enumerate(
                debug["bm25"],
                start=1,
            ):
                print(
                    f"\n{i}. "
                    f"{doc.page_content[:300]}..."
                )

            # =========================================
            # RRF
            # =========================================

            print(
                "\n========== AFTER RRF =========="
            )

            for i, doc in enumerate(
                debug["rrf"],
                start=1,
            ):
                print(
                    f"\n{i}. "
                    f"{doc.page_content[:300]}..."
                )

            # =========================================
            # RERANKING
            # =========================================

            print(
                "\n========== AFTER RERANKING =========="
            )

            for i, (doc, score) in enumerate(
                debug["reranked"],
                start=1,
            ):
                print(
                    f"\n{i}. Cross-Encoder Score: "
                    f"{score:.4f}"
                )

                print(
                    doc.page_content[:300]
                    + "..."
                )

            # =========================================
            # FINAL ANSWER
            # =========================================

            result = pipeline.invoke(query)

            print(
                "\n========== FINAL ANSWER =========="
            )

            print(result["answer"])

            # =========================================
            # SOURCES
            # =========================================

            sources = result["sources"]

            print("\n========== SOURCES ==========")

            if sources:

                seen_sources = set()

                for document in sources:

                    source = document.metadata.get(
                        "source",
                        "Unknown source",
                    )

                    if source in seen_sources:
                        continue

                    seen_sources.add(source)

                    print(f"- {source}")

            else:

                print("No sources found.")

        except Exception as error:

            print(
                f"\nError while processing query: "
                f"{error}"
            )


if __name__ == "__main__":
    main()
