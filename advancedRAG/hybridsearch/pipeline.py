from typing import Any, Dict, List
from langchain_core.documents import Document

from retriever import hybrid_retrieve, reciprocal_rank_fusion
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

    # -----------------------------------------
    # Retrieval: Hybrid Search + RRF + Reranking
    # -----------------------------------------

    def retrieve(self, query: str) -> List[Document]:
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

    # -----------------------------------------
    # Debug: Show retrieval stages
    # -----------------------------------------

    def debug_retrieval(self, query: str) -> Dict[str, Any]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        rrf_docs = reciprocal_rank_fusion(
            [vector_docs, bm25_docs]
        )[:self.candidate_k]

        reranked_docs = self.reranker.rerank_with_scores(
            query,
            rrf_docs,
        )

        return {
            "vector": vector_docs,
            "bm25": bm25_docs,
            "rrf": rrf_docs,
            "reranked": reranked_docs,
        }

    # -----------------------------------------
    # Grounded Generation + Citations
    # -----------------------------------------

    def generate(
        self,
        query: str,
        documents: List[Document],
    ) -> str:

        if not documents:
            return (
                "I don't have enough information in the "
                "provided documents to answer this question."
            )

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

    # -----------------------------------------
    # Complete RAG Pipeline
    # -----------------------------------------

    def invoke(self, query: str) -> Dict[str, Any]:

        documents = self.retrieve(query)

        answer = self.generate(
            query=query,
            documents=documents,
        )

        return {
            "answer": answer,
            "sources": documents,
        }