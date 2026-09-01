from typing import Any, Dict, List

from langchain_core.documents import Document

from retriever import (
    hybrid_retrieve,
    reciprocal_rank_fusion,
)
from reranker import DocumentReranker


class HybridRAGPipeline:

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        reranker: DocumentReranker,
        llm,
        candidate_k: int = 30,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.llm = llm
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:

        candidates = hybrid_retrieve(
            query=query,
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            max_candidates=self.candidate_k,
        )

        if not candidates:
            return []

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
        """

        # -----------------------------------------
        # Vector Search
        # -----------------------------------------

        vector_docs = self.vector_retriever.invoke(
            query
        )

        # -----------------------------------------
        # BM25 Search
        # -----------------------------------------

        bm25_docs = self.bm25_retriever.invoke(
            query
        )

        # -----------------------------------------
        # RRF
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
        # Reranking
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

        if not documents:
            return (
                "I could not find enough relevant "
                "information in the provided documents "
                "to answer this question."
            )

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using only the
provided context.

If the answer is not available in the context,
say that you do not have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content

    def invoke(
        self,
        query: str,
    ) -> Dict[str, Any]:

        documents = self.retrieve(query)

        answer = self.generate(
            query=query,
            documents=documents,
        )

        return {
            "answer": answer,
            "sources": documents,
        }