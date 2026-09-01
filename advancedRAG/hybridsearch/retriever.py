from collections import defaultdict
from typing import List

from langchain_core.documents import Document


def get_document_id(document: Document) -> str:
    """
    Generate a stable unique ID for a document.
    Prefer chunk_id if available; otherwise fall back
    to source, page, and content.
    """

    chunk_id = document.metadata.get("chunk_id")

    if chunk_id:
        return str(chunk_id)

    source = document.metadata.get("source", "")
    page = document.metadata.get("page", "")

    return f"{source}_{page}_{document.page_content}"


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Combine multiple ranked result lists using
    Reciprocal Rank Fusion (RRF).

    RRF Score:
        score = 1 / (k + rank)
    """

    scores = defaultdict(float)
    documents = {}

    for results in result_lists:

        for rank, document in enumerate(results, start=1):

            document_id = get_document_id(document)

            # Add this document's contribution
            # from the current retrieval method.
            scores[document_id] += 1 / (k + rank)

            # Keep the actual Document object.
            documents[document_id] = document

    # Rank documents by their combined RRF score.
    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        documents[document_id]
        for document_id in ranked_ids
    ]


def deduplicate_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Remove duplicate documents while preserving
    their existing ranking order.
    """

    unique_documents = []
    seen_ids = set()

    for document in documents:

        document_id = get_document_id(document)

        if document_id in seen_ids:
            continue

        seen_ids.add(document_id)
        unique_documents.append(document)

    return unique_documents


def hybrid_retrieve(
    query: str,
    vector_retriever,
    bm25_retriever,
    max_candidates: int = 30,
) -> List[Document]:
    """
    Perform hybrid retrieval using:

    1. Dense vector search
    2. BM25 keyword search
    3. Reciprocal Rank Fusion
    4. Deduplication
    5. Candidate limiting

    Returns:
        Ranked candidate documents for reranking.
    """

    # -----------------------------------------
    # 1. Dense / semantic retrieval
    # -----------------------------------------

    vector_docs = vector_retriever.invoke(query)

    # -----------------------------------------
    # 2. Sparse / keyword retrieval
    # -----------------------------------------

    bm25_docs = bm25_retriever.invoke(query)

    # -----------------------------------------
    # 3. Combine rankings using RRF
    # -----------------------------------------

    candidates = reciprocal_rank_fusion(
        [
            vector_docs,
            bm25_docs,
        ]
    )

    # -----------------------------------------
    # 4. Remove duplicate chunks
    # -----------------------------------------

    candidates = deduplicate_documents(candidates)

    # -----------------------------------------
    # 5. Keep only candidates needed
    #    by the reranker
    # -----------------------------------------

    return candidates[:max_candidates]