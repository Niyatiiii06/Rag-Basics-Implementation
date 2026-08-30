from sentence_transformers import CrossEncoder


class DocumentReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, documents):

        if not documents:
            return []

        # Create (query, document) pairs
        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Attach score to each document
        results = []

        for document, score in zip(documents, scores):
            results.append(
                {
                    "document": document,
                    "score": float(score),
                }
            )

        # Highest relevance first
        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        # Keep only top K
        return results[:self.top_k]