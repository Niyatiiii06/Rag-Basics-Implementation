from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class DocumentReranker:
    """
    Reranks retrieved documents using a Cross-Encoder.
    """

    def __init__(
        self,
        model_name: str,
        top_k: int = 5,
    ):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return their relevance scores.
        """

        if not documents:
            return []

        pairs = [
            (
                query,
                document.page_content,
            )
            for document in documents
        ]

        scores = self.model.predict(pairs)

        scored_documents = list(
            zip(documents, scores)
        )

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_documents[:self.top_k]

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        Return only the reranked documents.
        """

        scored_documents = self.rerank_with_scores(
            query,
            documents,
        )

        return [
            document
            for document, _ in scored_documents
        ]