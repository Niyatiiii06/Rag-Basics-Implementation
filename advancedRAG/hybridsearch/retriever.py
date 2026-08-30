from collections import defaultdict


def get_document_id(document):

    # Prefer a unique chunk ID
    if "chunk_id" in document.metadata:
        return document.metadata["chunk_id"]

    # Otherwise use source + page + content
    source = document.metadata.get(
        "source",
        ""
    )

    page = document.metadata.get(
        "page",
        ""
    )

    return (
        f"{source}_{page}_"
        f"{document.page_content}"
    )


def reciprocal_rank_fusion(
    result_lists,
    k=60,
):
    scores = defaultdict(float)
    documents = {}

    for results in result_lists:

        for rank, document in enumerate(
            results,
            start=1,
        ):

            document_id = get_document_id(
                document
            )

            # RRF formula
            scores[document_id] += (
                1 / (k + rank)
            )

            documents[document_id] = document

    # Sort using RRF score
    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids
    ]


def deduplicate_documents(documents):

    unique_documents = []
    seen_ids = set()

    for document in documents:

        document_id = get_document_id(
            document
        )

        if document_id in seen_ids:
            continue

        seen_ids.add(document_id)

        unique_documents.append(
            document
        )

    return unique_documents


def hybrid_retrieve(
    query,
    vector_retriever,
    bm25_retriever,
    max_candidates=30,
):

    # -----------------------------
    # 1. Vector search
    # -----------------------------

    vector_docs = vector_retriever.invoke(
        query
    )


    # -----------------------------
    # 2. BM25 search
    # -----------------------------

    bm25_docs = bm25_retriever.invoke(
        query
    )


    # -----------------------------
    # 3. RRF
    # -----------------------------

    candidates = reciprocal_rank_fusion(
        [
            vector_docs,
            bm25_docs,
        ]
    )


    # -----------------------------
    # 4. Remove duplicates
    # -----------------------------

    candidates = deduplicate_documents(
        candidates
    )


    # -----------------------------
    # 5. Candidate limit
    # -----------------------------

    candidates = candidates[
        :max_candidates
    ]


    return candidates