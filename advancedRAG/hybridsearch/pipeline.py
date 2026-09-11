from typing import Any, Dict, List

from langchain_core.documents import Document

from retriever import (
    hybrid_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
)

from reranker import DocumentReranker
from query_decomposer import QueryDecomposer


class HybridRAGPipeline:

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        reranker: DocumentReranker,
        llm,
        query_decomposer: QueryDecomposer,
        candidate_k: int = 30,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.llm = llm
        self.query_decomposer = query_decomposer
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:

        # -----------------------------------------
        # 1. Query Decomposition
        # -----------------------------------------

        sub_queries = self.query_decomposer.decompose(
            query
        )

        # -----------------------------------------
        # 2. Hybrid Search for each sub-query
        # -----------------------------------------

        all_candidates = []

        for sub_query in sub_queries:

            candidates = hybrid_retrieve(
                query=sub_query,
                vector_retriever=self.vector_retriever,
                bm25_retriever=self.bm25_retriever,
                max_candidates=self.candidate_k,
            )

            all_candidates.extend(candidates)

        # -----------------------------------------
        # 3. Merge + Deduplicate
        # -----------------------------------------

        candidates = deduplicate_documents(
            all_candidates
        )

        if not candidates:
            return []

        # -----------------------------------------
        # 4. Cross-Encoder Reranking
        # -----------------------------------------

        return self.reranker.rerank(
            query=query,
            documents=candidates,
        )

    def debug_retrieval(
        self,
        query: str,
    ) -> Dict[str, Any]:

        """
        Show intermediate retrieval stages.
        Useful for debugging and understanding
        how each retrieval method behaves.
        """

        # -----------------------------------------
        # 1. Vector Search
        # -----------------------------------------

        vector_docs = self.vector_retriever.invoke(
            query
        )

        # -----------------------------------------
        # 2. BM25 Search
        # -----------------------------------------

        bm25_docs = self.bm25_retriever.invoke(
            query
        )

        # -----------------------------------------
        # 3. Reciprocal Rank Fusion
        # -----------------------------------------

        rrf_docs = reciprocal_rank_fusion(
            [
                vector_docs,
                bm25_docs,
            ]
        )

        rrf_docs = rrf_docs[
            :self.candidate_k
        ]

        # -----------------------------------------
        # 4. Cross-Encoder Reranking
        # -----------------------------------------

        reranked_docs = (
            self.reranker.rerank_with_scores(
                query,
                rrf_docs,
            )
        )

        return {
            "vector": vector_docs,
            "bm25": bm25_docs,
            "rrf": rrf_docs,
            "reranked": reranked_docs,
        }

    def generate(
        self,
        query: str,
        documents: List[Document],
    ) -> str:

        # -----------------------------------------
        # 5. Check Retrieved Context
        # -----------------------------------------

        if not documents:
            return (
                "I could not find enough relevant "
                "information in the provided documents "
                "to answer this question."
            )

        # -----------------------------------------
        # 6. Context + Grounded Generation + Citations
        # -----------------------------------------

        context_parts = []

        for i, document in enumerate(documents, start=1):
            source = document.metadata.get(
                "source",
                "Unknown source",
            )
            page = document.metadata.get(
                "page",
                "Unknown page",
            )
            chunk_id = document.metadata.get(
                "chunk_id",
                f"chunk_{i}",
            )

            context_parts.append(
                f"[{chunk_id}] "
                f"Source: {source} | Page: {page}\n"
                f"{document.page_content}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a grounded RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- Every factual claim must be supported by the context.
- Add a citation after each factual claim.
- Use this format: [Source: chunk_id, Page: page]
- If the context does not contain enough information, say:
  "I don't have enough information in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()

    def invoke(
        self,
        query: str,
    ) -> Dict[str, Any]:

        # -----------------------------------------
        # Complete RAG Pipeline
        # -----------------------------------------

        documents = self.retrieve(query)

        answer = self.generate(
            query=query,
            documents=documents,
        )

        return {
            "answer": answer,
            "sources": documents,
        }